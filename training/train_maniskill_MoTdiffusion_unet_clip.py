import numpy as np
import torch
import torch.nn as nn
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
import argparse
import wandb
import os
import yaml
import pickle

import sys
sys.path.append('/home/lasserre/changhe/MoTDiffusion/diffusion_policy')
from model.unet_model import *

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from data.planflowdp_maniskill_multitasks_dataset import MultiTaskManiSkillVLMDataset


# ─────────────────────────────────────────────
# Language encoding helpers (same as transformer script)
# ─────────────────────────────────────────────


def create_injected_noise(num_train_timesteps:int, beta_schedule='squaredcos_cap_v2'):
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        # the choise of beta schedule has big impact on performance
        # we found squared cosine works the best
        beta_schedule=beta_schedule,
        # clip output to [-1,1] to improve stability
        clip_sample=True,
        # our network predicts noise (instead of denoised action)
        prediction_type='epsilon'
    )
    return noise_scheduler

def split_batch_by_id(batch, unique_ids):
    split_batches = []

    for unique_id in unique_ids:
        indices = torch.where(batch['id'] == unique_id)[0]
        mini_batch = {
            'image': batch['image'][indices],
            'agent_pos': batch['agent_pos'][indices],
            'action': batch['action'][indices],
            'id': batch['id'][indices]
        }
        split_batches.append(mini_batch)

    return split_batches

def save(ema, nets, models_save_dir):
    if not os.path.exists(models_save_dir):
        os.makedirs(models_save_dir)
    torch.save(ema.state_dict(), os.path.join(models_save_dir, "ema_nets.pth"))
    for model_name, model in nets.items():
        model_path = os.path.join(models_save_dir, f"{model_name}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"{model_name}.pth saved")

    print("All models have been saved successfully.")

def encode_language_with_clip(language_batch, tokenizer, text_encoder, device):
    """Encode language strings with a frozen CLIP text encoder.

    Handles the two collation formats that DataLoader produces:
      - [T, B]: default collate on a per-sample list[str]  →  tuple of B-length tuples
      - [B]:    single sentence per sample

    Returns
    -------
    text_features : Tensor  [B, T, D]
        One CLIP pooler embedding per (sample, timestep).
        For the UNet global-cond path we flatten this to [B, T*D] later.
    """
    if isinstance(language_batch, (list, tuple)) and len(language_batch) > 0:
        first_elem = language_batch[0]

        if isinstance(first_elem, (list, tuple)):
            # [T, B] → transpose to [B, T]
            time_major = [list(x) for x in language_batch]
            obs_horizon = len(time_major)
            batch_size  = len(time_major[0])
            batch_sentences = [
                [str(time_major[t][b]) for t in range(obs_horizon)]
                for b in range(batch_size)
            ]
        else:
            # [B] – single sentence per sample
            batch_sentences = [[str(x)] for x in language_batch]
            batch_size  = len(batch_sentences)
            obs_horizon = 1

    elif isinstance(language_batch, np.ndarray):
        if language_batch.ndim == 2:
            batch_sentences = [[str(x) for x in row] for row in language_batch]
            batch_size, obs_horizon = language_batch.shape
        elif language_batch.ndim == 1:
            batch_sentences = [[str(x)] for x in language_batch]
            batch_size  = len(batch_sentences)
            obs_horizon = 1
        else:
            raise ValueError(f"Unsupported language_batch ndarray shape: {language_batch.shape}")
    else:
        raise TypeError(f"Unsupported language_batch type: {type(language_batch)}")

    flat_sentences = [sentence for sample in batch_sentences for sentence in sample]

    tokenized = tokenizer(
        flat_sentences,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )
    tokenized = {k: v.to(device) for k, v in tokenized.items()}

    with torch.no_grad():
        text_features = text_encoder(**tokenized).pooler_output   # [B*T, D]

    text_features = text_features.view(batch_size, obs_horizon, -1)  # [B, T, D]
    return text_features


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Training script for UNet Diffusion Policy (ManiSkill multi-task, CLIP).')
    parser.add_argument('--config', type=str, default='./config/train_config.yml',
                        help='Path to the configuration YAML file.')
    args = parser.parse_args()

    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)

    # ── Hyper-parameters ────────────────────────────────────────────────
    num_epochs          = config['num_epochs']
    num_diffusion_iters = config['num_diffusion_iters']
    num_demos_per_task  = config['num_demos_per_task']   # replaces num_train_demos
    pred_horizon        = config['pred_horizon']
    obs_horizon         = config['obs_horizon']
    action_horizon      = config['action_horizon']
    eval_epoch          = config['eval_epoch']
    lr                  = config['lr']
    weight_decay        = config['weight_decay']
    batch_size          = config['batch_size']

    # Dataset directories  (replaces the old list-of-files approach)
    action_data_dir    = config['action_data_dir']      # directory of *.pkl action files
    language_data_dir  = config['language_data_dir']    # directory of *.jsonl sentence files

    output_dir         = config['output_dir']
    models_save_dir    = config['models_save_dir']
    verbose            = config['verbose']
    display_name       = config['display_name']
    resize_scale       = config['resize_scale']

    # CLIP model (new)
    clip_model_name    = config.get('clip_model_name', 'openai/clip-vit-base-patch32')

    if display_name == "default":
        display_name = None

    if config["wandb"]:
        wandb.init(
            project="maniskill_multitask_unet",
            config=config,
            name=display_name
        )
    else:
        print("warning: wandb flag set to False")

    print("Training parameters:")
    for k in ['num_epochs', 'num_diffusion_iters', 'num_demos_per_task',
              'pred_horizon', 'obs_horizon', 'action_horizon', 'eval_epoch']:
        print(f"  {k}: {config[k]}")
    print("\nUNet Diffusion Policy  ·  frozen CLIP language encoder")

    # ── Output directories ───────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(models_save_dir, exist_ok=True)

    # ── Dataset ─────────────────────────────────────────────────────────
    # MultiTaskManiSkillVLMDataset discovers all tasks from action_data_dir
    # and pairs them with .jsonl language files in language_data_dir.
    dataset = MultiTaskManiSkillVLMDataset(
        action_data_dir=action_data_dir,
        language_data_dir=language_data_dir,
        pred_horizon=pred_horizon,
        obs_horizon=obs_horizon,
        action_horizon=action_horizon,
        num_demos_per_task=num_demos_per_task,
        resize_scale=resize_scale,
        pretrained=config.get('use_pretrained', False),
        normalize_across_tasks=True,
    )

    # Save normalization statistics for inference
    stats = dataset.stats
    stats_path = os.path.join(models_save_dir, 'normalization_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)
    print(f"Saved normalization statistics to {stats_path}")
    print(stats)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True
    )

    if verbose:
        batch = next(iter(dataloader))
        print("batch['images_camera0'].shape:", batch['images_camera0'].shape)
        print("batch['agent_pos'].shape:     ", batch['agent_pos'].shape)
        print("batch['action'].shape:        ", batch['action'].shape)
        print("batch['language'] sample:     ", batch['language'][0])

    # ── Model dimensions ────────────────────────────────────────────────
    # CLIP pooler_output dim: 512 for clip-vit-base-patch32
    # (loaded below from clip_text_encoder.config.projection_dim)
    vision_feature_dim = 512   # ResNet18 output
    sample0 = dataset[0]
    lowdim_obs_dim = sample0['agent_pos'].shape[-1]
    action_dim = sample0['action'].shape[-1]

    # ── CLIP text encoder (frozen, NOT part of nets / EMA / optimizer) ──
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    clip_tokenizer    = CLIPTokenizer.from_pretrained(clip_model_name)
    clip_text_encoder = CLIPTextModel.from_pretrained(clip_model_name)
    clip_text_encoder.eval()
    for param in clip_text_encoder.parameters():
        param.requires_grad = False
    clip_text_encoder = clip_text_encoder.to(device)

    # CLIP pooler_output dimension
    lang_feature_dim = clip_text_encoder.config.projection_dim  # 512 for base patch32
    print(f"CLIP language feature dim: {lang_feature_dim}")

    # Total per-step observation dim used as global conditioning for UNet
    obs_dim = vision_feature_dim + lowdim_obs_dim + lang_feature_dim

    # ── Trainable nets ───────────────────────────────────────────────────
    nets = nn.ModuleDict({})
    noise_schedulers = {}

    if config.get('use_pretrained', False):
        vision_encoder = get_resnet(weights='IMAGENET1K_V1')
    else:
        vision_encoder = get_resnet()
    vision_encoder = replace_bn_with_gn(vision_encoder)
    nets['vision_encoder'] = vision_encoder

    # NOTE: LanguageProjector is removed; CLIP embeddings are used directly.

    nets['invariant'] = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim * obs_horizon   # flattened over obs_horizon steps
    )
    noise_schedulers['single'] = create_injected_noise(num_diffusion_iters)

    nets = nets.to(device)

    # ── EMA / Optimizer / LR scheduler ──────────────────────────────────
    ema = EMAModel(parameters=nets.parameters(), power=0.75)

    optimizer = torch.optim.AdamW(
        params=nets.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    lr_scheduler = get_scheduler(
        name='cosine',
        optimizer=optimizer,
        num_warmup_steps=config.get('num_warmup_steps', 500),
        num_training_steps=len(dataloader) * num_epochs
    )

    # ── Initial checkpoint ───────────────────────────────────────────────
    checkpoint_dir = f'{models_save_dir}/checkpoint_epoch_0'
    os.makedirs(checkpoint_dir, exist_ok=True)
    save(ema, nets, checkpoint_dir)

    # ── Training loop ────────────────────────────────────────────────────
    with tqdm(range(1, num_epochs + 1), desc='Epoch') as tglobal:
        for epoch_idx in tglobal:
            if config['wandb']:
                wandb.log({'epoch': epoch_idx})

            epoch_loss = []

            with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                for nbatch in tepoch:
                    if config['wandb']:
                        wandb.log({'learning_rate': lr_scheduler.get_last_lr()[0]})

                    # ── Transfer to device ───────────────────────────────
                    # Dataset now uses 'images_camera0' (not 'image0')
                    nimage0    = nbatch['images_camera0'][:, :obs_horizon].to(device, dtype=torch.float32)
                    nagent_pos = nbatch['agent_pos'][:, :obs_horizon].to(device, dtype=torch.float32)
                    naction    = nbatch['action'].to(device, dtype=torch.float32)
                    nlanguage  = nbatch['language']   # list of strings – stays on CPU for tokenizer
                    B = nagent_pos.shape[0]

                    # ── Vision encoding ──────────────────────────────────
                    image_features = nets['vision_encoder'](nimage0.flatten(end_dim=1))
                    image_features = image_features.reshape(*nimage0.shape[:2], -1)  # [B, T, 512]

                    # ── Language encoding (frozen CLIP) ──────────────────
                    # Returns [B, obs_horizon, lang_feature_dim]
                    lang_features = encode_language_with_clip(
                        nlanguage, clip_tokenizer, clip_text_encoder, device
                    )

                    # ── Observation conditioning ─────────────────────────
                    # Concatenate along feature dim: [B, T, obs_dim]
                    obs_features = torch.cat([image_features, nagent_pos, lang_features], dim=-1)
                    # Flatten timestep dimension for UNet global_cond: [B, T * obs_dim]
                    obs_cond = obs_features.flatten(start_dim=1)

                    # ── Diffusion forward pass ───────────────────────────
                    noise = torch.randn(naction.shape, device=device)

                    timesteps = torch.randint(
                        0, noise_schedulers['single'].config.num_train_timesteps,
                        (B,), device=device
                    ).long()

                    noisy_actions = noise_schedulers['single'].add_noise(naction, noise, timesteps)

                    noise_pred = nets['invariant'](noisy_actions, timesteps, global_cond=obs_cond)

                    # ── Loss and optimisation ────────────────────────────
                    loss = nn.functional.mse_loss(noise_pred, noise)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step()
                    ema.step(nets.parameters())

                    loss_cpu = loss.item()
                    if config['wandb']:
                        wandb.log({'loss': loss_cpu, 'epoch': epoch_idx})
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)

            avg_loss = np.mean(epoch_loss)
            tglobal.set_postfix(loss=avg_loss)

            # ── Periodic checkpointing ────────────────────────────────────
            save_epochs = {1, 100, 150, 200, 300, 400, num_epochs}
            if (epoch_idx in save_epochs) or (epoch_idx % 100 == 0):
                ckpt_dir = f'{models_save_dir}/checkpoint_epoch_{epoch_idx}'
                os.makedirs(ckpt_dir, exist_ok=True)
                save(ema, nets, ckpt_dir)


if __name__ == "__main__":
    main()