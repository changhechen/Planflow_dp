#@markdown ### **Imports**
from typing import Tuple, Sequence, Dict, Union, Optional, Callable, List
import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import collections
import json
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
from PIL import Image
import pickle
import os
from pathlib import Path


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_injected_noise(num_train_timesteps: int, beta_schedule='squaredcos_cap_v2'):
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        beta_schedule=beta_schedule,
        clip_sample=True,
        prediction_type='epsilon'
    )
    return noise_scheduler


def save(ema, nets, models_save_dir):
    if not os.path.exists(models_save_dir):
        os.makedirs(models_save_dir)
    torch.save(ema.state_dict(), os.path.join(models_save_dir, "ema_nets.pth"))
    for model_name, model in nets.items():
        model_path = os.path.join(models_save_dir, f"{model_name}.pth")
        torch.save(model.state_dict(), model_path)
        print(f"{model_name}.pth saved")
    print("All models have been saved successfully.")


def create_sample_indices(
        episode_ends: np.ndarray, sequence_length: int,
        pad_before: int = 0, pad_after: int = 0):
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i - 1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append([
                buffer_start_idx, buffer_end_idx,
                sample_start_idx, sample_end_idx])
    indices = np.array(indices)
    return indices


def sample_sequence(train_data, sequence_length,
                    buffer_start_idx, buffer_end_idx,
                    sample_start_idx, sample_end_idx):
    result = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(
                shape=(sequence_length,) + input_arr.shape[1:],
                dtype=input_arr.dtype)
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result


def get_data_stats(data):
    data = data.reshape(-1, data.shape[-1])
    stats = {
        'min': np.min(data, axis=0),
        'max': np.max(data, axis=0)
    }
    return stats


def normalize_data(data, stats):
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    ndata = ndata * 2 - 1
    return ndata


def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data


class MultiTaskDataset(torch.utils.data.Dataset):
    def __init__(self,
                 action_data_dir: str,           # Directory containing action .pkl files
                 language_data_dir: str,          # Directory containing language .jsonl files
                 pred_horizon: int = 16,
                 obs_horizon: int = 2,
                 action_horizon: int = 8,
                 num_demos_per_task: int = -1,    # -1 means use all demos
                 resize_scale: int = 96,
                 pretrained: bool = False,
                 tasks: Optional[List[str]] = None,   # Specific task names to load; None = auto-discover
                 normalize_across_tasks: bool = True):

        self.action_data_dir = Path(action_data_dir)
        self.language_data_dir = Path(language_data_dir)

        # Auto-discover tasks from .pkl filenames if not specified
        if tasks is None:
            action_files = list(self.action_data_dir.glob("*.pkl"))
            # Use the full stem as the task name (e.g. "maniskill_stack_dp_100" → that is the key
            # used to match language files); strip no prefix — just use the stem directly
            tasks = [f.stem for f in action_files]

        print(f"Loading data for tasks: {tasks}")

        # Store parameters
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon
        self.resize_scale = resize_scale
        self.pretrained = pretrained
        self.normalize_across_tasks = normalize_across_tasks
        self.num_demos_per_task = num_demos_per_task

        # Per-task storage
        self.task_names = []
        self.task_data = []          # list of normalized train_data dicts
        self.task_indices = []       # list of sample index arrays
        self.task_stats = []         # used only when normalize_across_tasks=False
        self.task_episode_ends = []
        self.all_indices_with_task = []   # [(task_id, (buf_start, buf_end, samp_start, samp_end))]

        # Accumulate raw arrays for global statistics
        all_actions = []
        all_agent_poses = []

        for task_idx, task in enumerate(tasks):
            # --- Locate files ---
            action_files = list(self.action_data_dir.glob(f"*{task}*.pkl"))
            if not action_files:
                print(f"Warning: No action file found for task '{task}', skipping")
                continue

            language_files = list(self.language_data_dir.glob(f"*{task}*.jsonl"))
            if not language_files:
                print(f"Warning: No language file found for task '{task}', skipping")
                continue

            action_file = action_files[0]
            language_file = language_files[0]
            print(f"Task '{task}':")
            print(f"  Action file:   {action_file.name}")
            print(f"  Language file: {language_file.name}")

            # --- Load action data ---
            with open(str(action_file), 'rb') as f:
                data_dict = pickle.load(f)

            # --- Limit demos ---
            num_max_demos = data_dict['end_frames'].shape[0]
            if self.num_demos_per_task > 0:
                num_demos = min(num_max_demos, self.num_demos_per_task)
            else:
                num_demos = num_max_demos
            num_max_frames = data_dict['end_frames'][num_demos - 1]
            episode_ends = data_dict['end_frames'][:num_demos]

            print(f"  Episodes: {len(episode_ends)}, Frames: {num_max_frames}")

            # --- Load language sentences ---
            sentences = []
            with open(str(language_file), 'r') as f:
                for line in f:
                    entry = json.loads(line.strip())
                    gpt_value = next(
                        conv['value'] for conv in entry['conversations']
                        if conv['from'] == 'gpt'
                    )
                    sentences.append(gpt_value)
            language_sentences = np.array(sentences, dtype=object)

            # --- Build raw train_data dict ---
            train_data = {
                'images_camera0': data_dict['all_images_camera0'][:num_max_frames],
                'agent_pos':      data_dict['all_agent_pose'][:num_max_frames],
                'action':         data_dict['all_actions'][:num_max_frames],
                'language':       language_sentences[:num_max_frames],
            }

            # Accumulate for global stats
            if self.normalize_across_tasks:
                all_actions.append(data_dict['all_actions'][:num_max_frames])
                all_agent_poses.append(data_dict['all_agent_pose'][:num_max_frames])

            # --- Compute sample indices ---
            indices = create_sample_indices(
                episode_ends=episode_ends,
                sequence_length=pred_horizon,
                pad_before=obs_horizon - 1,
                pad_after=action_horizon - 1
            )

            self.task_names.append(task)
            self.task_data.append(train_data)
            self.task_indices.append(indices)
            self.task_episode_ends.append(episode_ends)

            for idx_tuple in indices:
                self.all_indices_with_task.append((len(self.task_names) - 1, idx_tuple))

            print(f"  Loaded {len(indices)} samples from {num_demos} demos")

        if len(self.task_names) == 0:
            raise ValueError("No valid tasks found in the specified directories")

        self.num_tasks = len(self.task_names)

        # --- Normalization ---
        if self.normalize_across_tasks and len(all_actions) > 0:
            print("\nComputing global normalization statistics...")
            all_actions_cat = np.concatenate(all_actions, axis=0)
            all_agent_poses_cat = np.concatenate(all_agent_poses, axis=0)

            self.stats = {
                'action':    get_data_stats(all_actions_cat),
                'agent_pos': get_data_stats(all_agent_poses_cat)
            }

            for task_id, train_data in enumerate(self.task_data):
                normalized_data = {}
                for key, data in train_data.items():
                    if key in ('action', 'agent_pos'):
                        normalized_data[key] = normalize_data(data, self.stats[key])
                    else:
                        normalized_data[key] = data
                self.task_data[task_id] = normalized_data
        else:
            print("\nComputing per-task normalization statistics...")
            for task_id, train_data in enumerate(self.task_data):
                task_stats = {}
                normalized_data = {}
                for key, data in train_data.items():
                    if key in ('action', 'agent_pos'):
                        task_stats[key] = get_data_stats(data)
                        normalized_data[key] = normalize_data(data, task_stats[key])
                    else:
                        normalized_data[key] = data
                self.task_data[task_id] = normalized_data
                self.task_stats.append(task_stats)

        # --- Image transform (built once, reused in __getitem__) ---
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

        # --- Summary ---
        print(f"\n{'='*50}")
        print(f"Dataset initialized with {self.num_tasks} tasks")
        print(f"Total samples: {len(self)}")
        print(f"{'='*50}")
        for task_id, task_name in enumerate(self.task_names):
            task_samples = sum(1 for tid, _ in self.all_indices_with_task if tid == task_id)
            print(f"  {task_name:30s}: {task_samples:6d} samples")
        print(f"{'='*50}")

    def __len__(self):
        return len(self.all_indices_with_task)

    def __getitem__(self, idx):
        task_id, sample_indices = self.all_indices_with_task[idx]
        buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx = sample_indices

        normalized_train_data = self.task_data[task_id]

        nsample = sample_sequence(
            train_data=normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx
        )

        # Apply image transforms
        images0 = nsample['images_camera0'][:self.obs_horizon]
        images0 = [np.expand_dims(self.transform(image).numpy(), axis=0) for image in images0]
        images0 = np.concatenate(images0, axis=0)

        nsample['images_camera0'] = images0[:self.obs_horizon]
        nsample['agent_pos']      = nsample['agent_pos'][:self.obs_horizon]
        nsample['language']       = [str(x) for x in nsample['language'][:self.obs_horizon]]
        nsample['task_id']        = task_id
        nsample['task_name']      = self.task_names[task_id]

        return nsample

    def get_normalizer(self):
        """Return normalization stats (global or per-task) for use during inference."""
        if self.normalize_across_tasks:
            return self.stats
        return self.task_stats

    def get_task_info(self):
        """Return a summary dict describing the loaded tasks."""
        return {
            'task_names':         self.task_names,
            'num_tasks':          self.num_tasks,
            'task_episode_counts': [len(ep) for ep in self.task_episode_ends],
            'task_frame_counts':  [int(ep[-1]) for ep in self.task_episode_ends],
            'total_samples':      len(self.all_indices_with_task),
        }


if __name__ == "__main__":
    dataset = MultiTaskDataset(
        action_data_dir='/home/aiden/MoT_maniskill_sort_100/action_data',
        language_data_dir='/home/aiden/MoT_maniskill_sort_100/language_data',
        pred_horizon=16,
        obs_horizon=2,
        action_horizon=8,
        num_demos_per_task=5,
        resize_scale=96,
        pretrained=False,
        normalize_across_tasks=True,
        # tasks=['tool_usage', 'pick_place']  # optional: specify tasks explicitly
    )

    task_info = dataset.get_task_info()
    print(f"\nTask Information:")
    for key, value in task_info.items():
        print(f"  {key}: {value}")

    print(f"\nDataset length: {len(dataset)}")

    for i in range(min(3, len(dataset))):
        sample = dataset[i * len(dataset) // 3]
        print(f"\nSample {i}:")
        print(f"  Task:               {sample['task_name']} (ID: {sample['task_id']})")
        print(f"  Agent pos shape:    {sample['agent_pos'].shape}")
        print(f"  Action shape:       {sample['action'].shape}")
        print(f"  Images shape:       {sample['images_camera0'].shape}")
        print(f"  Language:           {sample['language']}")
