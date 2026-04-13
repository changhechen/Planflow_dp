from typing import Tuple, Sequence, Dict, Union, Optional, Callable
import numpy as np
import math
import torch
import torch.nn as nn
import torchvision
import pickle
import yaml
import argparse
import os
from datetime import datetime
import copy

import wandb

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler

import sys
#sys.path.append('/home/lasserre/changhe/MoTVLA')
from transformers import CLIPTextModel, CLIPTokenizer
from model.transformer_for_MoTdiffusion_new import TransformerForMoTDiffusion
from model.mask_generator import LowdimMaskGenerator
from tqdm.auto import tqdm
from einops import reduce

from data.planflowdp_maniskill_multitasks_dataset import MultiTaskManiSkillVLMDataset


def get_resnet(name:str, weights=None, **kwargs) -> nn.Module:
    """
    name: resnet18, resnet34, resnet50
    weights: "IMAGENET1K_V1", None
    """
    # Use standard ResNet implementation from torchvision
    func = getattr(torchvision.models, name)
    resnet = func(weights=weights, **kwargs)

    # remove the final fully connected layer
    # for resnet18, the output dim should be 512
    resnet.fc = torch.nn.Identity()
    return resnet

def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module

def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group,
            num_channels=x.num_features)
    )
    return root_module

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def encode_language_with_clip(language_batch, tokenizer, text_encoder, device):
    """Encode language strings with a frozen CLIP text encoder.
    Supports both:
      1) [B, T] style: list of samples, each sample is a list of T strings
      2) [T, B] style: default PyTorch collate on per-sample list[str]
    Returns:
      text_features: [B, T, D]
    """
    # Case 1: default DataLoader collate on list[str] from dataset
    # language_batch becomes a list of length T, each element containing B strings
    if isinstance(language_batch, (list, tuple)) and len(language_batch) > 0:
        first_elem = language_batch[0]

        # [T, B] case
        if isinstance(first_elem, (list, tuple)):
            # convert tuples -> lists
            time_major = [list(x) for x in language_batch]   # length T, each length B
            obs_horizon = len(time_major)
            batch_size = len(time_major[0])

            # transpose to [B, T]
            batch_sentences = [
                [str(time_major[t][b]) for t in range(obs_horizon)]
                for b in range(batch_size)
            ]

        else:
            # [B] case, single sentence per sample
            batch_sentences = [[str(x)] for x in language_batch]
            batch_size = len(batch_sentences)
            obs_horizon = 1

    elif isinstance(language_batch, np.ndarray):
        # optional fallback
        if language_batch.ndim == 2:
            batch_sentences = [[str(x) for x in row] for row in language_batch]
            batch_size, obs_horizon = language_batch.shape
        elif language_batch.ndim == 1:
            batch_sentences = [[str(x)] for x in language_batch]
            batch_size = len(batch_sentences)
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


def save_checkpoint(model, optimizer, lr_scheduler, epoch, step, ema, config, checkpoint_dir):
    """Save model checkpoint."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch}_step_{step}.pt')
    
    checkpoint = {
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict(),
        'ema_state_dict': ema.state_dict(),
        'config': config
    }
    
    torch.save(checkpoint, checkpoint_path)
    
    # Log to wandb
    wandb.save(checkpoint_path)
    
    return checkpoint_path

def save_ema_weights(ema_nets, config, save_path):
    """Save only the EMA network weights for inference."""
    # Save just the EMA network state dict
    torch.save(ema_nets.state_dict(), save_path)
    
    # Also save a minimal config for inference
    inference_config = {
        'model': config['model'],
        'dataset': {
            'resize_scale': config['dataset']['resize_scale'],
            'pretrained': config['dataset']['pretrained'],
        },
        'clip_model_name': config['model'].get('clip_model_name', 'openai/clip-vit-base-patch32')
    }
    config_path = save_path.replace('.ckpt', '_config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(inference_config, f)
    
    # Log to wandb
    wandb.save(save_path)
    wandb.save(config_path)
    
    print(f"Saved EMA weights for inference: {save_path}")

def load_datasets(dataset_config, model_config):
    """Load and combine multiple datasets."""
    dataset_names = {}
    
    dataset_path_dir = dataset_config['dataset_path_dir']
    
    # Check if we're using a single dataset file or a directory
    # Single dataset file
    print(f"Loading single dataset from {dataset_path_dir}")
    dataset = MultiTaskManiSkillVLMDataset(
        action_data_dir=dataset_path_dir,
        language_data_dir=dataset_config.get('language_dataset_path', 'None'),
        pred_horizon=model_config['pred_horizon'],
        obs_horizon=model_config['obs_horizon'],
        action_horizon=model_config['action_horizon'],
        num_demos_per_task=dataset_config['num_demos'],
        resize_scale=dataset_config['resize_scale'],
        pretrained=dataset_config.get('pretrained', False),
        normalize_across_tasks=True
    )
    dataset_names[0] = os.path.basename(dataset_path_dir).split('.')[0]
    
    return dataset, dataset.stats, dataset_names

def main(config_path):
    seed = 1
    # np.random.seed(1)
    # torch.manual_seed(1)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Load configuration
    config = load_config(config_path)
    
    # Initialize wandb
    wandb.init(
        project=config['wandb']['project'],
        name=config['wandb']['run_name'] if config['wandb']['run_name'] else None,
        config=config,
        tags=config['wandb'].get('tags', []),
        notes=config['wandb'].get('notes', '')
    )
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Extract config parameters
    dataset_config = config['dataset']
    model_config = config['model']
    training_config = config['training']
    checkpoint_config = config['checkpoint']
    
    # Create dataset
    dataset, stats, dataset_names = load_datasets(dataset_config, model_config)
    
    # Create dataloader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=training_config['batch_size'],
        num_workers=training_config['num_workers'],
        shuffle=True,
        pin_memory=True,
        persistent_workers=True
    )
    
    # Get data shapes and create shape_meta
    sample = dataset[0]
    shape_meta = {
        'agent_pos': {'shape': sample['agent_pos'].shape[1:]},
        'images_camera0': {'shape': sample['images_camera0'].shape[1:]},
        'action': {'shape': sample['action'].shape[1:]},
    }
    
    # Save normalization statistics for inference
    if hasattr(dataset, 'stats'):
        stats = dataset.stats
        stats_path = os.path.join(checkpoint_config['save_dir'], 'normalization_stats.pkl')
        os.makedirs(checkpoint_config['save_dir'], exist_ok=True)
        with open(stats_path, 'wb') as f:
            pickle.dump(stats, f)
        print(f"Saved normalization statistics to {stats_path}")
    
    # Log dataset info to wandb
    wandb.log({
        "dataset_size": len(dataset),
        "num_batches": len(dataloader),
        "agent_pos_shape": shape_meta['agent_pos']['shape'],
        "image_shape": shape_meta['images_camera0']['shape'],
        "action_shape": shape_meta['action']['shape'],
    })
    
    # construct ResNet encoder
    vision_encoder = get_resnet(model_config['vision_encoder']['name'])
    
    # IMPORTANT! replace all BatchNorm with GroupNorm to work with EMA
    vision_encoder = replace_bn_with_gn(vision_encoder)

    clip_model_name = model_config.get('clip_model_name', 'openai/clip-vit-base-patch32')
    clip_tokenizer = CLIPTokenizer.from_pretrained(clip_model_name)
    clip_text_encoder = CLIPTextModel.from_pretrained(clip_model_name)
    clip_text_encoder.eval()
    for param in clip_text_encoder.parameters():
        param.requires_grad = False
    
    # Model dimensions
    vision_feature_dim = model_config['vision_feature_dim']
    lowdim_obs_dim = model_config['lowdim_obs_dim']  # Should be 7 for first 7 DOF
    lang_features_dim = clip_text_encoder.config.hidden_size
    obs_dim = vision_feature_dim + lowdim_obs_dim + lang_features_dim  # Only one camera now
    #obs_dim = lowdim_obs_dim + lang_features_dim
    action_dim = model_config['action_dim']
    
    # Create transformer for diffusion
    noise_pred_net = TransformerForMoTDiffusion(
        input_dim=action_dim,
        output_dim=action_dim,
        horizon=model_config['pred_horizon'],
        n_obs_steps=model_config['obs_horizon'],
        cond_dim=obs_dim,
        n_layer=model_config['transformer']['n_layer'],
        n_head=model_config['transformer']['n_head'],
        n_emb=model_config['transformer']['n_emb'],
        p_drop_emb=model_config['transformer']['p_drop_emb'],
        p_drop_attn=model_config['transformer']['p_drop_attn'],
        causal_attn=model_config['transformer']['causal_attn'],
        time_as_cond=model_config['transformer']['time_as_cond'],
        obs_as_cond=model_config['transformer']['obs_as_cond'],
        n_cond_layers=model_config['transformer']['n_cond_layers']
    )
    
    # the final arch has 2 parts
    nets = nn.ModuleDict({
        'vision_encoder': vision_encoder,
        'noise_pred_net': noise_pred_net,
    })
    
    # Log model architecture to wandb
    wandb.watch(nets, log='all', log_freq=100)
    
    # Noise scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=model_config['diffusion']['num_diffusion_iters'],
        beta_schedule=model_config['diffusion']['beta_schedule'],
        clip_sample=model_config['diffusion']['clip_sample'],
        prediction_type=model_config['diffusion']['prediction_type']
    )
    
    # device transfer
    _ = nets.to(device)
    clip_text_encoder = clip_text_encoder.to(device)
    
    # Exponential Moving Average
    ema = EMAModel(
        parameters=nets.parameters(),
        power=training_config['ema']['power']
    )
    
    # Standard ADAM optimizer
    optimizer = torch.optim.AdamW(
        params=nets.parameters(),
        lr=training_config['optimizer']['lr'],
        weight_decay=training_config['optimizer']['weight_decay']
    )
    
    # Cosine LR schedule with linear warmup
    lr_scheduler = get_scheduler(
        name=training_config['lr_scheduler']['name'],
        optimizer=optimizer,
        num_warmup_steps=training_config['lr_scheduler']['num_warmup_steps'],
        num_training_steps=len(dataloader) * training_config['num_epochs'],
    )
    
    mask_generator = LowdimMaskGenerator(
        action_dim=action_dim,
        obs_dim=0 if model_config['transformer']['obs_as_cond'] else obs_dim,
        max_n_obs_steps=model_config['obs_horizon'],
        fix_obs_steps=True,
        action_visible=False
    )
    
    # Training loop
    global_step = 0
    best_loss = float('inf')
    
    with tqdm(range(training_config['num_epochs']), desc='Epoch') as tglobal:
        for epoch_idx in tglobal:
            epoch_loss = []
            
            with tqdm(dataloader, desc='Batch', leave=False) as tepoch:
                for batch_idx, nbatch in enumerate(tepoch):
                    # data normalized in dataset
                    # device transfer
                    nimage0 = nbatch['images_camera0'][:,:model_config['obs_horizon']].to(device)
                    nagent_pos = nbatch['agent_pos'][:,:model_config['obs_horizon']].to(device)
                    naction = nbatch['action'].to(device)
                    nlanguage = nbatch['language']
                    B = nagent_pos.shape[0]
                    
                    # encoder vision features
                    image0_features = nets['vision_encoder'](
                        nimage0.flatten(end_dim=1))
                    image0_features = image0_features.reshape(
                        *nimage0.shape[:2],-1)

                    # encode language features with frozen CLIP text encoder
                    lang_features = encode_language_with_clip(nlanguage, clip_tokenizer, clip_text_encoder, device)

                    # concatenate vision feature and low-dim obs
                    obs_features = torch.cat([image0_features, nagent_pos, lang_features], dim=-1) # (B, T, obs_dim)
                    #obs_features = torch.cat([nagent_pos, lang_features], dim=-1) # (B, T, obs_dim)
                    
                    # sample noise to add to actions
                    condition_mask = mask_generator(naction.shape).to(device)
                    noise = torch.randn(naction.shape, device=device)
                    
                    # sample a diffusion iteration for each data point
                    timesteps = torch.randint(
                        0, noise_scheduler.config.num_train_timesteps,
                        (B,), device=device
                    ).long()
                    
                    # add noise to the clean images according to the noise magnitude at each diffusion iteration
                    noisy_actions = noise_scheduler.add_noise(
                        naction, noise, timesteps)
                    
                    # compute loss mask
                    loss_mask = ~condition_mask

                    # apply conditioning
                    noisy_actions[condition_mask] = naction[condition_mask]
                    
                    # predict the noise residual
                    noise_pred = noise_pred_net(
                        noisy_actions, timesteps, cond=obs_features)
                    
                    # L2 loss
                    loss = nn.functional.mse_loss(noise_pred, noise)
                    loss = loss * loss_mask.type(loss.dtype)
                    loss = reduce(loss, 'b ... -> b (...)', 'mean')
                    loss = loss.mean()
                    
                    # optimize
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    lr_scheduler.step()
                    
                    # update Exponential Moving Average of the model weights
                    ema.step(nets.parameters())
                    
                    # logging
                    loss_cpu = loss.item()
                    epoch_loss.append(loss_cpu)
                    tepoch.set_postfix(loss=loss_cpu)
                    
                    # Log to wandb
                    if global_step % training_config['log_freq'] == 0:
                        wandb.log({
                            'train/loss': loss_cpu,
                            'train/learning_rate': lr_scheduler.get_last_lr()[0],
                            'train/epoch': epoch_idx,
                            'train/global_step': global_step,
                        }, step=global_step)
                    
                    global_step += 1
                    
                    # Save checkpoint periodically
                    if checkpoint_config['save_freq'] > 0 and global_step % checkpoint_config['save_freq'] == 0:
                        # Also save EMA weights only for inference
                        ema_nets = nn.ModuleDict({
                            'vision_encoder': nets['vision_encoder'],
                            'noise_pred_net': nets['noise_pred_net'],
                        })
                        ema.copy_to(ema_nets.parameters())
                        ema_path = os.path.join(checkpoint_config['save_dir'], f'ema_weights_step_{global_step}.ckpt')
                        save_ema_weights(ema_nets, config, ema_path)
                        wandb.log({'train/checkpoint_step': global_step}, step=global_step)
            
            # Log epoch metrics
            avg_epoch_loss = np.mean(epoch_loss)
            tglobal.set_postfix(loss=avg_epoch_loss)
            
            wandb.log({
                'train/epoch_loss': avg_epoch_loss,
                'train/epoch': epoch_idx,
            }, step=global_step)
            
            # Save best model
            if avg_epoch_loss < best_loss:
                best_loss = avg_epoch_loss
                if checkpoint_config['save_best']:
                    # Also save EMA weights only for inference
                    ema_nets = nn.ModuleDict({
                        'vision_encoder': nets['vision_encoder'],
                        'noise_pred_net': nets['noise_pred_net'],
                    })
                    ema.copy_to(ema_nets.parameters())
                    ema_path = os.path.join(checkpoint_config['save_dir'], 'best_ema_weights.ckpt')
                    save_ema_weights(ema_nets, config, ema_path)
                    
                    wandb.log({'train/best_loss': best_loss}, step=global_step)
    
    # Copy EMA weights for inference
    ema_nets = nn.ModuleDict({
        'vision_encoder': nets['vision_encoder'],
        'noise_pred_net': nets['noise_pred_net']
    })
    ema.copy_to(ema_nets.parameters())
    
    # Save final EMA weights
    final_ema_path = os.path.join(checkpoint_config['save_dir'], 'final_ema_weights.ckpt')
    save_ema_weights(ema_nets, config, final_ema_path)
    
    # Finish wandb run
    wandb.finish()
    
    return ema_nets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Transformer Diffusion Policy with WandB logging')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    main(args.config)