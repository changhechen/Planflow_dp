import sys
import torch
import torch.nn as nn
import numpy as np
import yaml
import pickle
import os
import torchvision
from typing import Dict, Optional, Union, Tuple, Callable
    
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torchvision import transforms
from torchvision.transforms import InterpolationMode

# Add your path here
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from modeling.action_expert.transformer_for_MoTdiffusion_new import TransformerForMoTDiffusion
from modeling.action_expert.language_projector import LanguageProjector

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

def load_transformer_weights_for_inference(ckpt_path, device='cuda'):
    """
    Load transformer EMA weights saved in the simplified format for inference.
    
    Args:
        ckpt_path: Path to the .ckpt file containing EMA weights
        device: Device to load the model on
        
    Returns:
        ema_nets: The loaded model ready for inference
        config: The configuration used for training
    """
    # Load config
    config_path = ckpt_path.replace('.ckpt', '_config.yaml')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    model_config = config['model']
    
    # Check if we need vision encoder
    use_vision = model_config['transformer'].get('use_image_cond', True)
    
    ema_nets = nn.ModuleDict({})
    
    if use_vision:
        # Construct vision encoder
        vision_encoder = get_resnet(model_config['vision_encoder']['name'])
        # Replace all BatchNorm with GroupNorm to work with EMA
        vision_encoder = replace_bn_with_gn(vision_encoder)
        ema_nets['vision_encoder'] = vision_encoder
    
    # construct language projector
    lang_projector = LanguageProjector(
        in_dim=model_config['lang_hidden_dim'],
        proj_dim=model_config['proj_dim'],
        num_tokens=model_config['n_tokens']
    )
    
    # Model dimensions
    vision_feature_dim = model_config.get('vision_feature_dim', 0) if use_vision else 0
    lowdim_obs_dim = model_config['lowdim_obs_dim']
    lang_features_dim = model_config['n_tokens'] * model_config['proj_dim']
    obs_dim = vision_feature_dim + lowdim_obs_dim + lang_features_dim  
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
    
   # Combined model
    ema_nets = nn.ModuleDict({
        'vision_encoder': vision_encoder,
        'lang_projector': lang_projector,
        'noise_pred_net': noise_pred_net,
    })
    
    # Load weights
    state_dict = torch.load(ckpt_path, map_location=device)
    ema_nets.load_state_dict(state_dict)
    ema_nets.to(device)
    ema_nets.eval()
    
    print(f"Loaded Transformer EMA weights from: {ckpt_path}")
    
    return ema_nets, config


class DiffusionTransformerInference:
    """Pure inference wrapper for transformer-based diffusion policy."""
    
    def __init__(self, 
                 ckpt_path: str,
                 device: str = 'cuda'):
        """
        Initialize the diffusion transformer inference wrapper.
        
        Args:
            ckpt_path: Path to the checkpoint file
            device: Device to run inference on
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Load model and config
        self.ema_nets, self.config = self._load_model(ckpt_path)
        
        # Load normalization statistics
        # stats_path = ckpt_path.replace('best_ema_weights.ckpt', 'normalization_stats.pkl')
        # if not os.path.exists(stats_path):
        stats_path = os.path.join(os.path.dirname(ckpt_path), 'normalization_stats.pkl')
        
        with open(stats_path, 'rb') as f:
            self.stats = pickle.load(f)
        
        # Extract model configuration
        self.model_config = self.config['model']
        self.dataset_config = self.config.get('dataset', {})
        
        self.obs_horizon = self.model_config['obs_horizon']
        self.pred_horizon = self.model_config['pred_horizon']
        self.action_horizon = self.model_config['action_horizon']
        self.action_dim = self.model_config['action_dim']
        
        # Check if we are using vision as input
        self.use_img_cond = self.model_config['transformer'].get('use_image_cond', True)
        
        # Model dimensions
        self.use_vision = 'vision_encoder' in self.ema_nets and self.use_img_cond
        self.vision_feature_dim = self.model_config.get('vision_feature_dim', 0) if self.use_vision else 0
        self.lowdim_obs_dim = self.model_config['lowdim_obs_dim']
        self.lang_hidden_dim = self.model_config['lang_hidden_dim']
        self.cond_dim = self.model_config['cond_dim']
        
        # Image preprocessing parameters
        self.resize_scale = self.dataset_config.get('resize_scale', 96)
        self.pretrained = self.dataset_config.get('pretrained', False)
        
        # Number of diffusion steps
        self.num_diffusion_iters = self.model_config['diffusion']['num_diffusion_iters']
        
        # Setup image transforms to match training
        if self.use_vision:
            if self.pretrained:
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
        
        # Initialize noise scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=self.model_config['diffusion']['num_diffusion_iters'],
            beta_schedule=self.model_config['diffusion']['beta_schedule'],
            clip_sample=self.model_config['diffusion']['clip_sample'],
            prediction_type=self.model_config['diffusion']['prediction_type']
        )
        
        print(f"Transformer model loaded successfully!")
        print(f"Observation horizon: {self.obs_horizon}")
        print(f"Prediction horizon: {self.pred_horizon}")
        print(f"Action horizon: {self.action_horizon}")
        print(f"Action dimension: {self.action_dim}")
        print(f"Using vision: {self.use_vision}")
        if self.use_vision:
            print(f"Image resize scale: {self.resize_scale}")
            print(f"Using pretrained normalization: {self.pretrained}")
    
    def _load_model(self, ckpt_path: str) -> Tuple[nn.ModuleDict, Dict]:
        """Load model from checkpoint."""
        return load_transformer_weights_for_inference(ckpt_path, self.device)
    
    def normalize_data(self, data: np.ndarray, stats: Dict) -> np.ndarray:
        """Normalize data to [-1, 1] using min-max statistics."""
        # Normalize to [0,1]
        ndata = (data - stats['min']) / (stats['max'] - stats['min'])
        # Normalize to [-1, 1]
        ndata = ndata * 2 - 1
        return ndata
    
    def unnormalize_data(self, ndata: np.ndarray, stats: Dict) -> np.ndarray:
        """Unnormalize data from [-1, 1] to original range."""
        ndata = (ndata + 1) / 2
        data = ndata * (stats['max'] - stats['min']) + stats['min']
        return data
    
    def preprocess_images(self, images: np.ndarray) -> torch.Tensor:
        """
        Preprocess images to match training format.
        
        Args:
            images: numpy array of shape (T, H, W, C) with values in [0, 255] or [0, 1]
            
        Returns:
            torch tensor of shape (T, C, H, W) after transforms
        """
        # Apply transforms to each image
        transformed_images = []
        for img in images:
            # Ensure image is in uint8 format for PIL
            if img.max() <= 1.0:
                img = (img * 255).astype(np.uint8)
            elif img.dtype != np.uint8:
                img = img.astype(np.uint8)
            
            # Apply the same transform used during training
            transformed_img = self.transform(img)
            transformed_images.append(transformed_img)
        
        # Stack into tensor of shape (T, C, H, W)
        return torch.stack(transformed_images)
    
    def predict_action(self,
                      agent_pos: np.ndarray,
                      lang_hidden: np.ndarray,
                      lang_mask: Optional[np.ndarray],
                      images_camera0: Optional[np.ndarray]) -> np.ndarray:
        """
        Main inference function that takes observations and returns actions.
        
        Args:
            agent_pos: Agent positions/joint angles of shape (obs_horizon, pos_dim)
            lang_hidden: Language hidden states of shape (n_tokens, hidden_dim)
            lang_mask: Language attention mask of shape (n_tokens,), optional
                      True = valid token, False = padding
            images_camera0: Base camera images of shape (obs_horizon, H, W, C), optional
                           Can be in [0, 255] uint8 or [0, 1] float format
            images_camera_wrist: Wrist camera images of shape (obs_horizon, H, W, C), optional
                                Can be in [0, 255] uint8 or [0, 1] float format
            
        Returns:
            actions: Predicted actions of shape (action_horizon, action_dim)
        """
        # Validate inputs
        assert images_camera0.shape[0] == self.obs_horizon, \
            f"Expected {self.obs_horizon} timesteps, got {images_camera0.shape[0]}"
        assert agent_pos.shape[0] == self.obs_horizon, \
            f"Expected {self.obs_horizon} timesteps, got {agent_pos.shape[0]}"
        
        # # Only use first 7 DOF of agent position
        # agent_pos = agent_pos[:, :7]
        
        # Normalize agent positions
        nagent_pos = self.normalize_data(agent_pos, self.stats['agent_pos'])
        
        with torch.no_grad():
            # Preprocess images using the same transforms as training
            nimages0 = self.preprocess_images(images_camera0).to(self.device)
            
            # Process other inputs
            nagent_pos = torch.from_numpy(nagent_pos).to(self.device, dtype=torch.float32)
            nhidden_state = torch.from_numpy(lang_hidden).to(self.device, dtype=torch.float32)
            nhidden_mask = torch.from_numpy(lang_mask).to(self.device, dtype=torch.bool)
            
            # Add batch dimension
            nimages0 = nimages0.unsqueeze(0)  # (1, T, C, H, W)
            nagent_pos = nagent_pos.unsqueeze(0)  # (1, T, pos_dim)
            nhidden_state = nhidden_state.unsqueeze(0)  # (1, T, n_tokens, hidden_dim)
            nhidden_mask = nhidden_mask.unsqueeze(0)  # (1, T, n_tokens)
            
            # Encode vision features
            B, T = nimages0.shape[:2]
            
            # Process camera 0
            image0_features = self.ema_nets['vision_encoder'](
                nimages0.flatten(end_dim=1))  # (B*T, feature_dim)
            image0_features = image0_features.reshape(B, T, -1)  # (B, T, feature_dim)

            # Process language features
            lang_features, masks, valid_step = self.ema_nets['lang_projector'](nhidden_state, nhidden_mask)  # (B, T, n_tokens *
            
            # Concatenate all features
            obs_features = torch.cat([
                image0_features, 
                nagent_pos,
                lang_features
            ], dim=-1)  # (B, T, obs_dim)
            
            # Initialize action from Gaussian noise
            noisy_action = torch.randn(
                (B, self.pred_horizon, self.action_dim), device=self.device)
            naction = noisy_action
            
            # Diffusion denoising loop
            self.noise_scheduler.set_timesteps(self.num_diffusion_iters)
            
            for k in self.noise_scheduler.timesteps:
                # Predict noise using transformer
                noise_pred = self.ema_nets['noise_pred_net'](
                    sample=naction,
                    timestep=k,
                    cond=obs_features  # Transformer uses 'cond' not 'global_cond'
                )
                
                # Inverse diffusion step (remove noise)
                naction = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=k,
                    sample=naction
                ).prev_sample
            
            # Move to CPU and remove batch dimension
            naction = naction[0].detach().cpu().numpy()  # (pred_horizon, action_dim)
            #print(naction[:,-1])
        
        # Unnormalize actions
        action_pred = self.unnormalize_data(naction, self.stats['action'])
        #print(action_pred[:,-1])
        # Extract action horizon
        start = self.obs_horizon - 1
        end = start + self.action_horizon
        actions = action_pred[start:end, :]  # (action_horizon, action_dim)
        
        return actions
    
    def get_model_info(self) -> Dict:
        """Get information about the loaded model."""
        return {
            'obs_horizon': self.obs_horizon,
            'pred_horizon': self.pred_horizon,
            'action_horizon': self.action_horizon,
            'action_dim': self.action_dim,
            'use_vision': self.use_vision,
            'vision_feature_dim': self.vision_feature_dim,
            'lowdim_obs_dim': self.lowdim_obs_dim,
            'lang_hidden_dim': self.lang_hidden_dim,
            'cond_dim': self.cond_dim,
            'resize_scale': self.resize_scale if self.use_vision else None,
            'pretrained': self.pretrained if self.use_vision else None,
            'device': str(self.device),
            'num_diffusion_iters': self.num_diffusion_iters
        }


# Simple usage example
if __name__ == "__main__":
    # Initialize the model
    policy = DiffusionTransformerInference(
        ckpt_path='/home/lasserre/changhe/MoTDiffusion/diffusion_policy/diffusion_policy/checkpoints/7b_maniskill_316kgeneral_sort_garbage_first/best_ema_weights.ckpt',
        device='cuda',
    )
    
    # Print model information
    print("\nModel Information:")
    for key, value in policy.get_model_info().items():
        print(f"  {key}: {value}")
    
    # Example: Single inference with vision and language
    obs_horizon = policy.obs_horizon
    
    # Prepare observations (matching the dataset format)
    # Agent positions
    dummy_agent_pos = np.random.randn(obs_horizon, 9).astype(np.float32)  # 9-DOF proprioceptive
    
    # Language embeddings (matching dataset: single timestep, multiple tokens)
    n_tokens = 16  # Example token count
    dummy_lang_hidden = np.random.randn(obs_horizon, n_tokens, 3584).astype(np.float32)  # Language embeddings
    dummy_lang_mask = np.random.randn(obs_horizon, n_tokens)  # All tokens valid
    
    # Images (optional)
    dummy_image_camera0 = np.random.randint(0, 256, (obs_horizon, 128, 128, 3), dtype=np.uint8)
    dummy_image_wrist = np.random.randint(0, 256, (obs_horizon, 128, 128, 3), dtype=np.uint8)
    
    # Predict actions with all modalities
    actions = policy.predict_action(
        agent_pos=dummy_agent_pos,
        lang_hidden=dummy_lang_hidden,
        lang_mask=dummy_lang_mask,
        images_camera0=dummy_image_camera0,
    )
    
    print(f"\nPredicted actions shape: {actions.shape}")
    print(f"First action: {actions.shape}")
    
    # # Example: Language-only inference (no images)
    # actions_lang_only = policy.predict_action(
    #     agent_pos=dummy_agent_pos,
    #     lang_hidden=dummy_lang_hidden,
    #     lang_mask=dummy_lang_mask,
    #     images_camera0=None,  # No images
    #     images_camera_wrist=None
    # )
    
    # print(f"\nLanguage-only predicted actions shape: {actions_lang_only.shape}")
    
    # # Example: Batch inference
    # batch_size = 4
    # batch_agent_pos = np.random.randn(batch_size, obs_horizon, 9).astype(np.float32)
    # batch_lang_hidden = np.random.randn(batch_size, n_tokens, 3584).astype(np.float32)
    # batch_lang_mask = np.ones((batch_size, n_tokens), dtype=bool)
    # batch_images_camera0 = np.random.randint(0, 256, (batch_size, obs_horizon, 128, 128, 3), dtype=np.uint8)
    # batch_images_wrist = np.random.randint(0, 256, (batch_size, obs_horizon, 128, 128, 3), dtype=np.uint8)
    
    # batch_actions = policy.predict_action_batch(
    #     agent_pos=batch_agent_pos,
    #     lang_hidden=batch_lang_hidden,
    #     lang_mask=batch_lang_mask,
    #     images_camera0=batch_images_camera0,
    #     images_camera_wrist=batch_images_wrist
    # )
    
    # print(f"\nBatch predicted actions shape: {batch_actions.shape}")