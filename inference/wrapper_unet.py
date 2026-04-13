import sys
import torch
import torch.nn as nn
import numpy as np
import yaml
import pickle
import os
import torchvision
from typing import Dict, List, Optional, Union, Tuple, Callable

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import CLIPTextModel, CLIPTokenizer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.unet_model import ConditionalUnet1D, get_resnet, replace_bn_with_gn


def load_unet_weights_for_inference(ckpt_dir: str, config: Dict, device='cuda'):
    """
    Load UNet EMA weights from a checkpoint directory.

    The checkpoint directory is expected to contain:
        - ema_nets.pth          : EMAModel state dict (shadow_params)
        - vision_encoder.pth    : vision encoder weights (fallback if EMA load fails)
        - invariant.pth         : UNet weights (fallback)
        - normalization_stats.pkl

    Args:
        ckpt_dir: Path to checkpoint directory produced by train_maniskill_MoTdiffusion_unet_clip.py
        config:   Flat config dict loaded from the training YAML
        device:   Device string

    Returns:
        nets:   nn.ModuleDict with 'vision_encoder' and 'invariant', loaded with EMA weights
        config: The same config dict (passed through for convenience)
    """
    use_pretrained = config.get('use_pretrained', False)
    lowdim_obs_dim = config['lowdim_obs_dim']
    action_dim     = config['action_dim']
    obs_horizon    = config['obs_horizon']

    # CLIP pooler_output dim is fixed at 512 for clip-vit-base-patch32.
    # For other CLIP variants, load the encoder to read config.projection_dim.
    clip_model_name = config.get('clip_model_name', 'openai/clip-vit-base-patch32')
    _clip_cfg = CLIPTextModel.from_pretrained(clip_model_name).config
    lang_feature_dim = _clip_cfg.projection_dim   # 512 for base/patch32
    del _clip_cfg

    vision_feature_dim = 512  # ResNet18 output dim
    obs_dim = vision_feature_dim + lowdim_obs_dim + lang_feature_dim

    # ── Build nets ────────────────────────────────────────────────────────
    vision_encoder = get_resnet(weights='IMAGENET1K_V1' if use_pretrained else None)
    vision_encoder = replace_bn_with_gn(vision_encoder)

    unet = ConditionalUnet1D(
        input_dim=action_dim,
        global_cond_dim=obs_dim * obs_horizon
    )

    nets = nn.ModuleDict({
        'vision_encoder': vision_encoder,
        'invariant': unet,
    })

    # ── Load EMA weights ──────────────────────────────────────────────────
    ema_path = os.path.join(ckpt_dir, 'ema_nets.pth')
    if os.path.exists(ema_path):
        ema = EMAModel(parameters=nets.parameters(), power=0.75)
        ema_state = torch.load(ema_path, map_location=device)
        ema.load_state_dict(ema_state)
        ema.copy_to(nets.parameters())
        print(f"Loaded EMA weights from: {ema_path}")
    else:
        # Fallback: load individual model weights
        print(f"Warning: ema_nets.pth not found in {ckpt_dir}, loading raw weights.")
        for key in ['vision_encoder', 'invariant']:
            pth = os.path.join(ckpt_dir, f'{key}.pth')
            if os.path.exists(pth):
                nets[key].load_state_dict(torch.load(pth, map_location=device))
                print(f"  Loaded {key} from {pth}")
            else:
                raise FileNotFoundError(f"Checkpoint not found: {pth}")

    nets.to(device)
    nets.eval()
    return nets


class DiffusionUNetInference:
    """Inference wrapper for the UNet-based diffusion policy trained with frozen CLIP language."""

    def __init__(self,
                 ckpt_dir: str,
                 config_path: str,
                 device: str = 'cuda'):
        """
        Args:
            ckpt_dir:    Path to a checkpoint directory (e.g. trained_model/checkpoint_epoch_200/)
            config_path: Path to the flat training YAML (maniskill_unet_lang_config.yaml)
            device:      'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # ── Load config ───────────────────────────────────────────────────
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # ── Core hyperparams ──────────────────────────────────────────────
        self.obs_horizon    = self.config['obs_horizon']
        self.pred_horizon   = self.config['pred_horizon']
        self.action_horizon = self.config['action_horizon']
        self.action_dim     = self.config['action_dim']
        self.resize_scale   = self.config.get('resize_scale', 96)
        self.use_pretrained = self.config.get('use_pretrained', False)
        self.num_diffusion_iters = self.config['num_diffusion_iters']
        self.clip_model_name = self.config.get('clip_model_name', 'openai/clip-vit-base-patch32')

        # ── Normalization stats (loaded before building model to get true dims) ──
        # Stats are saved at models_save_dir (top level). When ckpt_dir points
        # to a per-epoch subdir (e.g. checkpoint_epoch_200), fall back to parent.
        stats_path = os.path.join(ckpt_dir, 'normalization_stats.pkl')
        if not os.path.exists(stats_path):
            stats_path = os.path.join(os.path.dirname(ckpt_dir), 'normalization_stats.pkl')
        if not os.path.exists(stats_path):
            raise FileNotFoundError(
                f"normalization_stats.pkl not found in {ckpt_dir} or its parent directory."
            )
        with open(stats_path, 'rb') as f:
            self.stats = pickle.load(f)

        # Derive lowdim_obs_dim from the saved stats so it always matches the
        # model that was actually trained, regardless of what the config says.
        self.lowdim_obs_dim = int(self.stats['agent_pos']['min'].shape[-1])
        self.config['lowdim_obs_dim'] = self.lowdim_obs_dim

        # ── Load nets (vision encoder + UNet) ────────────────────────────
        self.nets = load_unet_weights_for_inference(ckpt_dir, self.config, str(self.device))

        # ── Frozen CLIP text encoder ──────────────────────────────────────
        self.clip_tokenizer = CLIPTokenizer.from_pretrained(self.clip_model_name)
        self.clip_text_encoder = CLIPTextModel.from_pretrained(self.clip_model_name)
        self.clip_text_encoder.eval()
        for p in self.clip_text_encoder.parameters():
            p.requires_grad = False
        self.clip_text_encoder = self.clip_text_encoder.to(self.device)
        self.lang_feature_dim = self.clip_text_encoder.config.projection_dim

        # ── Image transforms (match training) ────────────────────────────
        if self.use_pretrained:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(self.resize_scale, InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(self.resize_scale, InterpolationMode.BILINEAR),
                transforms.ToTensor(),
            ])

        # ── Noise scheduler ───────────────────────────────────────────────
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.num_diffusion_iters,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )

        print(f"UNet model loaded successfully!")
        print(f"  Observation horizon : {self.obs_horizon}")
        print(f"  Prediction horizon  : {self.pred_horizon}")
        print(f"  Action horizon      : {self.action_horizon}")
        print(f"  Action dimension    : {self.action_dim}")
        print(f"  CLIP model          : {self.clip_model_name}  (lang_feature_dim={self.lang_feature_dim})")
        print(f"  Image resize scale  : {self.resize_scale}")

    # ── Internal helpers ──────────────────────────────────────────────────

    def normalize_data(self, data: np.ndarray, stats: Dict) -> np.ndarray:
        ndata = (data - stats['min']) / (stats['max'] - stats['min'])
        ndata = ndata * 2 - 1
        return ndata

    def unnormalize_data(self, ndata: np.ndarray, stats: Dict) -> np.ndarray:
        ndata = (ndata + 1) / 2
        data = ndata * (stats['max'] - stats['min']) + stats['min']
        return data

    def preprocess_images(self, images: np.ndarray) -> torch.Tensor:
        """
        Args:
            images: (T, H, W, C) uint8 [0,255] or float [0,1]
        Returns:
            Tensor (T, C, H, W)
        """
        out = []
        for img in images:
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            elif img.dtype != np.uint8:
                img = img.astype(np.uint8)
            out.append(self.transform(img))
        return torch.stack(out)

    def _encode_language(self, language: Union[str, List[str]]) -> torch.Tensor:
        """
        Encode language with frozen CLIP.

        Args:
            language: a single string (repeated for every obs step) or a list
                      of obs_horizon strings (one per timestep, as in training).

        Returns:
            Tensor (1, obs_horizon, lang_feature_dim)
        """
        if isinstance(language, str):
            sentences = [language] * self.obs_horizon
        else:
            assert len(language) == self.obs_horizon, (
                f"Expected {self.obs_horizon} language strings, got {len(language)}"
            )
            sentences = list(language)

        tokenized = self.clip_tokenizer(
            sentences, padding=True, truncation=True, return_tensors='pt'
        )
        tokenized = {k: v.to(self.device) for k, v in tokenized.items()}

        with torch.no_grad():
            features = self.clip_text_encoder(**tokenized).pooler_output  # (obs_horizon, D)

        return features.unsqueeze(0)  # (1, obs_horizon, D)

    # ── Main inference ────────────────────────────────────────────────────

    def predict_action(self,
                       agent_pos: np.ndarray,
                       language: Union[str, List[str]],
                       images_camera0: np.ndarray) -> np.ndarray:
        """
        Run a single diffusion inference step and return the action chunk.

        Args:
            agent_pos:      (obs_horizon, lowdim_obs_dim)  proprioceptive state
            language:       A single task-description string, or a list of
                            obs_horizon strings (one per timestep).
            images_camera0: (obs_horizon, H, W, C) RGB images, uint8 or float

        Returns:
            actions: (action_horizon, action_dim) unnormalized robot actions
        """
        assert agent_pos.shape[0] == self.obs_horizon, (
            f"Expected obs_horizon={self.obs_horizon} agent_pos steps, got {agent_pos.shape[0]}"
        )
        assert images_camera0.shape[0] == self.obs_horizon, (
            f"Expected obs_horizon={self.obs_horizon} image steps, got {images_camera0.shape[0]}"
        )

        # Normalize agent positions
        nagent_pos = self.normalize_data(agent_pos, self.stats['agent_pos'])

        with torch.no_grad():
            # ── Vision ───────────────────────────────────────────────────
            nimages = self.preprocess_images(images_camera0).to(self.device)  # (T, C, H, W)
            nimages = nimages.unsqueeze(0)  # (1, T, C, H, W)
            B, T = nimages.shape[:2]

            image_features = self.nets['vision_encoder'](
                nimages.flatten(end_dim=1)          # (B*T, C, H, W)
            )
            image_features = image_features.reshape(B, T, -1)  # (1, T, 512)

            # ── Language (frozen CLIP) ────────────────────────────────────
            lang_features = self._encode_language(language)  # (1, T, lang_feature_dim)

            # ── Proprioception ────────────────────────────────────────────
            nagent_pos_t = torch.from_numpy(nagent_pos).to(
                self.device, dtype=torch.float32
            ).unsqueeze(0)  # (1, T, lowdim_obs_dim)

            # ── Build global conditioning for UNet ────────────────────────
            # Concatenate along feature dim then flatten timesteps
            obs_features = torch.cat(
                [image_features, nagent_pos_t, lang_features], dim=-1
            )  # (1, T, obs_dim)
            obs_cond = obs_features.flatten(start_dim=1)  # (1, T * obs_dim)

            # ── Diffusion denoising loop ──────────────────────────────────
            naction = torch.randn(
                (B, self.pred_horizon, self.action_dim), device=self.device
            )

            self.noise_scheduler.set_timesteps(self.num_diffusion_iters)

            for k in self.noise_scheduler.timesteps:
                noise_pred = self.nets['invariant'](
                    sample=naction,
                    timestep=k,
                    global_cond=obs_cond
                )
                naction = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample

            naction = naction[0].detach().cpu().numpy()  # (pred_horizon, action_dim)

        # Unnormalize and extract action horizon window
        action_pred = self.unnormalize_data(naction, self.stats['action'])
        start = self.obs_horizon - 1
        end   = start + self.action_horizon
        return action_pred[start:end, :]  # (action_horizon, action_dim)

    def get_model_info(self) -> Dict:
        return {
            'obs_horizon':       self.obs_horizon,
            'pred_horizon':      self.pred_horizon,
            'action_horizon':    self.action_horizon,
            'action_dim':        self.action_dim,
            'lowdim_obs_dim':    self.lowdim_obs_dim,
            'lang_feature_dim':  self.lang_feature_dim,
            'clip_model':        self.clip_model_name,
            'resize_scale':      self.resize_scale,
            'use_pretrained':    self.use_pretrained,
            'num_diffusion_iters': self.num_diffusion_iters,
            'device':            str(self.device),
        }


# ── Simple usage example ──────────────────────────────────────────────────────
if __name__ == "__main__":
    policy = DiffusionUNetInference(
        ckpt_dir='/media/aiden/Extreme SSD/Planflow_dp/trained_model_tool_usage_3/checkpoint_epoch_200',
        config_path='/media/aiden/Extreme SSD/Planflow_dp/training/maniskill_unet_lang_config.yaml',
        device='cuda',
    )

    print("\nModel Information:")
    for key, value in policy.get_model_info().items():
        print(f"  {key}: {value}")

    obs_horizon = policy.obs_horizon

    dummy_agent_pos    = np.random.randn(obs_horizon, policy.lowdim_obs_dim).astype(np.float32)
    dummy_images       = np.random.randint(0, 256, (obs_horizon, 128, 128, 3), dtype=np.uint8)
    dummy_language     = "pick up the red block and place it in the bin"

    actions = policy.predict_action(
        agent_pos=dummy_agent_pos,
        language=dummy_language,
        images_camera0=dummy_images,
    )

    print(f"\nPredicted actions shape: {actions.shape}")  # (action_horizon, action_dim)
