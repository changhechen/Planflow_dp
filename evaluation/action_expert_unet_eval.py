"""
Evaluation script for the UNet diffusion policy (Planflow-DP).

Language instructions are determined at runtime by querying GPT-4o:
  - A text+image prompt describes the meta-task and lists the stage decomposition
    extracted from the training language data.
  - GPT-4o returns a single stage number; the corresponding stage instruction
    string is then fed as the language conditioning to the UNet policy.
"""

import gymnasium as gym
import os
import sys
import json
import base64
import re
import argparse
import importlib
import textwrap
import collections

import numpy as np
import torch
from tqdm.auto import tqdm

from skvideo.io import vwrite
from PIL import Image, ImageDraw, ImageFont

from openai import OpenAI

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# ── Task registry ─────────────────────────────────────────────────────────────
# Each entry maps a task name to:
#   env_id       : ManiSkill gym ID
#   env_module   : Python module to import so the env is registered
#   language_file: path (relative to project root) to the .jsonl stage-label file
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

TASK_CONFIGS = {
    'fruit_first': {
        'env_id': 'TableSortOneactionExp1Type2',
        'env_module': 'maniskill_envs.sort_table.mani_skill.envs.tasks.tabletop.table_sort_my_oneaction_exp1_type2',
        'language_file': os.path.join(PROJECT_ROOT, 'language_data', '10_22_sort_fruit_first_295demos.jsonl'),
    },
    'garbage_first': {
        'env_id': 'TableSortOneactionExp1Type1',
        'env_module': 'maniskill_envs.sort_table.mani_skill.envs.tasks.tabletop.table_sort_my_oneaction_exp1_type1',
        'language_file': os.path.join(PROJECT_ROOT, 'language_data', '10_22_sort_garbage_first_295demos.jsonl'),
    },
    'pull_tool': {
        'env_id': 'PullCubeTool-v1',
        'env_module': 'maniskill_envs.sort_table.mani_skill.envs.tasks.tabletop.pull_cube_tool',
        'language_file': os.path.join(PROJECT_ROOT, 'language_data', '10_22_two_step_pulltool_295demos.jsonl'),
    },
}


# ── Language data helpers ─────────────────────────────────────────────────────

def load_stages_from_jsonl(jsonl_path: str):
    """
    Parse the jsonl language file to extract:
      - meta_task : the meta-task description string
      - stages    : ordered list of stage instruction strings (in execution order)

    Stage order is inferred from the first demo by tracking label transitions
    as the `id` field increases.
    """
    meta_task = None
    ordered_stages = []
    seen = set()
    prev_stage = None

    with open(jsonl_path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            human_turn = entry['conversations'][0]['value']
            gpt_turn   = entry['conversations'][1]['value']

            # Extract meta task from the first entry
            if meta_task is None:
                # Remove <image> token and "Your meta task is: " prefix
                meta_task = re.sub(r'<image>\s*', '', human_turn)
                meta_task = re.sub(r'Your meta task is:\s*', '', meta_task).strip()

            stage = gpt_turn.strip()

            # Track stage transitions within first demo (id resets at demo boundaries)
            if entry['id'] == 0 and prev_stage is not None:
                # We've hit the second demo — stop
                break

            if stage != prev_stage and stage not in seen:
                ordered_stages.append(stage)
                seen.add(stage)
            prev_stage = stage

    return meta_task, ordered_stages


def build_gpt_prompt(meta_task: str, stages: list) -> str:
    """Build the stage-classification prompt sent to GPT-4o."""
    stage_lines = '\n'.join(
        f'Stage {i+1}: {s}' for i, s in enumerate(stages)
    )
    prompt = (
        f"Your meta task is: {meta_task}\n"
        f"There are {len(stages)} stages in our decomposition:\n"
        f"{stage_lines}\n"
        f"What stage is the current task in? Answer only with the number of the stage."
    )
    return prompt


# ── GPT-4o interface ──────────────────────────────────────────────────────────

def image_to_base64(image_array: np.ndarray) -> str:
    """Convert an (H, W, 3) uint8 numpy array to a base64-encoded JPEG string."""
    pil_img = Image.fromarray(image_array.astype(np.uint8))
    import io
    buffer = io.BytesIO()
    pil_img.save(buffer, format='JPEG', quality=85)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def query_gpt_stage(
    client: OpenAI,
    prompt: str,
    current_image: np.ndarray,
    stages: list,
    model: str = 'gpt-4o',
) -> tuple:
    """
    Send the current camera image + stage-classification prompt to GPT-4o.

    Returns
    -------
    stage_idx  : int  (0-based index into stages list)
    stage_text : str  (the corresponding stage instruction)
    raw_response: str (raw GPT output, for logging)
    """
    b64 = image_to_base64(current_image)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                'role': 'user',
                'content': [
                    {
                        'type': 'image_url',
                        'image_url': {'url': f'data:image/jpeg;base64,{b64}'},
                    },
                    {
                        'type': 'text',
                        'text': prompt,
                    },
                ],
            }
        ],
        max_tokens=16,
        temperature=0.0,
    )

    raw = response.choices[0].message.content.strip()

    # Parse the first integer from the response
    numbers = re.findall(r'\d+', raw)
    if numbers:
        stage_num = int(numbers[0])
        # Clamp to valid range
        stage_num = max(1, min(stage_num, len(stages)))
    else:
        # Fallback to stage 1 if parsing fails
        print(f"  [GPT] Could not parse stage number from: '{raw}' — defaulting to stage 1")
        stage_num = 1

    stage_idx  = stage_num - 1
    stage_text = stages[stage_idx]
    return stage_idx, stage_text, raw


# ── Visualization helper ──────────────────────────────────────────────────────

def add_text_to_frame(frame: np.ndarray, text: str, max_width: int = 40) -> np.ndarray:
    """Overlay text on a rendered frame."""
    img  = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)
    except Exception:
        font = ImageFont.load_default()

    wrapped = textwrap.fill(text, width=max_width)
    y = 10
    for line in wrapped.split('\n'):
        bbox = draw.textbbox((10, y), line, font=font)
        draw.rectangle([bbox[0]-5, bbox[1]-5, bbox[2]+5, bbox[3]+5], fill=(0, 0, 0, 180))
        draw.text((10, y), line, fill=(255, 255, 0), font=font)
        y += bbox[3] - bbox[1] + 5
    return np.array(img)


# ── Main evaluation function ──────────────────────────────────────────────────

def evaluate(
    task_name: str,
    ckpt_dir: str,
    config_path: str,
    num_tests: int = 50,
    device: str = 'cuda:0',
    output_dir: str = './',
    start_index: int = 1000,
    render_videos: bool = True,
    replan_freq: int = 0,         # 0 → replan every action_horizon steps
    gpt_model: str = 'gpt-4o',
):
    """
    Run evaluation episodes for one task.

    Args:
        task_name   : one of 'fruit_first', 'garbage_first', 'pull_tool'
        ckpt_dir    : path to UNet checkpoint directory
        config_path : path to the flat training YAML
        num_tests   : number of evaluation episodes
        device      : torch device string
        output_dir  : directory for videos and result text
        start_index : first random seed
        render_videos: save mp4 files per episode
        replan_freq : re-query GPT every N environment steps
                      (0 = use action_horizon)
        gpt_model   : OpenAI model name to use for stage classification
    """
    assert task_name in TASK_CONFIGS, \
        f"Unknown task '{task_name}'. Choose from {list(TASK_CONFIGS.keys())}"

    cfg = TASK_CONFIGS[task_name]

    # ── Register the ManiSkill environment ───────────────────────────────────
    importlib.import_module(cfg['env_module'])
    from maniskill_envs.sort_table.mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv

    # ── Load stage decomposition from language data ──────────────────────────
    meta_task, stages = load_stages_from_jsonl(cfg['language_file'])
    gpt_prompt        = build_gpt_prompt(meta_task, stages)

    print(f"\nTask: {task_name}")
    print(f"Meta-task: {meta_task}")
    print(f"Stage decomposition ({len(stages)} stages):")
    for i, s in enumerate(stages):
        print(f"  Stage {i+1}: {s}")
    print(f"\nGPT prompt:\n{gpt_prompt}\n")

    # ── Initialize OpenAI client ─────────────────────────────────────────────
    # Reads OPENAI_API_KEY from the environment variable
    openai_client = OpenAI()

    # ── Load the UNet diffusion policy ───────────────────────────────────────
    from inference.wrapper_unet import DiffusionUNetInference

    print("Initializing UNet Diffusion Policy...")
    policy = DiffusionUNetInference(
        ckpt_dir=ckpt_dir,
        config_path=config_path,
        device=device,
    )

    obs_horizon    = policy.obs_horizon
    action_horizon = policy.action_horizon
    effective_replan_freq = replan_freq if replan_freq > 0 else action_horizon

    print(f"  obs_horizon    : {obs_horizon}")
    print(f"  action_horizon : {action_horizon}")
    print(f"  replan_freq    : {effective_replan_freq} steps")

    # ── Create ManiSkill environment ─────────────────────────────────────────
    env_kwargs = dict(
        obs_mode='pointcloud',
        control_mode='pd_joint_pos',
        render_mode='rgb_array',
        sensor_configs=dict(shader_pack='default'),
        human_render_camera_configs=dict(shader_pack='default'),
        viewer_camera_configs=dict(shader_pack='default'),
    )
    env = gym.make(cfg['env_id'], num_envs=1, **env_kwargs)
    env = ManiSkillVectorEnv(env, ignore_terminations=False, record_metrics=True, auto_reset=False)

    os.makedirs(output_dir, exist_ok=True)

    all_rewards   = []
    success_count = 0
    max_steps     = 500

    # ── Episode loop ─────────────────────────────────────────────────────────
    for test_idx in range(num_tests):
        seed = start_index + test_idx
        print(f'\n[Episode {test_idx+1}/{num_tests}]  seed={seed}')

        obs, _ = env.reset(seed=seed)
        obs_deque = collections.deque([obs] * obs_horizon, maxlen=obs_horizon)

        rewards = []
        done    = False
        step_idx = 0

        # Initial GPT query using the first observation
        current_image = obs['sensor_data']['base_camera']['rgb'][0].cpu().numpy()
        stage_idx, current_language, raw_gpt = query_gpt_stage(
            openai_client, gpt_prompt, current_image, stages, model=gpt_model
        )
        print(f"  [GPT] stage={stage_idx+1} → '{current_language}'  (raw: '{raw_gpt}')")

        if render_videos:
            first_frame = env.render().cpu().numpy()[0]
            imgs = [add_text_to_frame(first_frame, f"[{stage_idx+1}] {current_language}")]

        # ── Step loop ────────────────────────────────────────────────────────
        with tqdm(total=max_steps, desc=f'Seed {seed}') as pbar:
            while not done and step_idx < max_steps:

                # ── Build observation tensors ─────────────────────────────
                camera0_images  = []
                joint_positions = []

                for i in range(obs_horizon):
                    obs_t = obs_deque[-(obs_horizon - i)]
                    camera0_images.append(
                        obs_t['sensor_data']['base_camera']['rgb'][0].cpu().numpy()
                    )
                    joint_positions.append(obs_t['agent']['qpos'][0].cpu().numpy())

                camera0_stack = np.stack(camera0_images)   # (obs_horizon, H, W, 3)
                joint_stack   = np.stack(joint_positions)  # (obs_horizon, lowdim_obs_dim)

                # ── Predict action chunk ──────────────────────────────────
                actions = policy.predict_action(
                    agent_pos=joint_stack,
                    language=current_language,
                    images_camera0=camera0_stack,
                )

                # ── Execute action chunk ──────────────────────────────────
                for i in range(min(len(actions), action_horizon)):
                    action = actions[i]
                    # Ensure action is 8-D (7 joints + 1 gripper)
                    if len(action) < 8:
                        action = np.pad(action, (0, 8 - len(action)))
                    elif len(action) > 8:
                        action = action[:8]

                    obs, reward, terminated, truncated, info = env.step(action)
                    obs_deque.append(obs)

                    rewards.append(reward.item())
                    step_idx += 1
                    pbar.update(1)
                    pbar.set_postfix(
                        {'reward': f'{reward.item():.3f}', 'max': f'{max(rewards):.3f}'},
                        refresh=False,
                    )

                    if render_videos:
                        frame = env.render().cpu().numpy()[0]
                        imgs.append(add_text_to_frame(frame, f"[{stage_idx+1}] {current_language}"))

                    # if terminated or truncated:
                    #     done = True
                    #     break

                # ── Re-query GPT for current stage ────────────────────────
                if not done and (step_idx % effective_replan_freq == 0):
                    current_image = obs['sensor_data']['base_camera']['rgb'][0].cpu().numpy()
                    stage_idx, current_language, raw_gpt = query_gpt_stage(
                        openai_client, gpt_prompt, current_image, stages, model=gpt_model
                    )
                    print(f"  [GPT @ step {step_idx}] stage={stage_idx+1} → '{current_language}'")

        # ── Episode results ───────────────────────────────────────────────
        max_reward = max(rewards) if rewards else 0.0
        success    = max_reward > 0.99
        if success:
            success_count += 1
        all_rewards.append(max_reward)
        print(f"  → max_reward={max_reward:.3f}  success={success}")

        if render_videos and imgs:
            video_path = os.path.join(output_dir, f'episode_{seed}.mp4')
            vwrite(video_path, imgs)
            print(f"  → saved video: {video_path}")

    # ── Aggregate results ─────────────────────────────────────────────────────
    success_rate = success_count / num_tests
    avg_reward   = float(np.mean(all_rewards))

    env.close()

    print(f"\n{'='*60}")
    print(f"Task          : {task_name}")
    print(f"Success rate  : {success_rate:.2%} ({success_count}/{num_tests})")
    print(f"Avg max reward: {avg_reward:.3f}")
    print(f"Reward std    : {np.std(all_rewards):.3f}")

    results_path = os.path.join(
        output_dir, f'results_seed{start_index}_sr{success_rate:.2f}.txt'
    )
    with open(results_path, 'w') as f:
        f.write(f"Task: {task_name}\n")
        f.write(f"GPT model: {gpt_model}\n")
        f.write(f"Checkpoint: {ckpt_dir}\n")
        f.write(f"Seeds: {start_index} – {start_index + num_tests - 1}\n")
        f.write(f"Success rate: {success_rate:.2%}\n")
        f.write(f"Avg max reward: {avg_reward:.3f}\n")
        f.write(f"Reward std: {np.std(all_rewards):.3f}\n")
        f.write(f"Rewards: {all_rewards}\n")
        f.write(f"\nStage decomposition:\n")
        for i, s in enumerate(stages):
            f.write(f"  Stage {i+1}: {s}\n")
    print(f"Saved results : {results_path}")

    return success_rate, all_rewards


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Planflow-DP UNet evaluation with GPT-4o stage planner'
    )
    parser.add_argument('--task', type=str, default='fruit_first',
                        choices=list(TASK_CONFIGS.keys()),
                        help='Task to evaluate')
    parser.add_argument('--ckpt_dir', type=str,
                        default=os.path.join(PROJECT_ROOT,
                                             'trained_model_tool_usage_3',
                                             'checkpoint_epoch_200'),
                        help='Path to UNet checkpoint directory')
    parser.add_argument('--config_path', type=str,
                        default=os.path.join(PROJECT_ROOT,
                                             'training',
                                             'maniskill_unet_lang_config.yaml'),
                        help='Path to the flat training YAML config')
    parser.add_argument('--num_tests', type=int, default=50,
                        help='Number of evaluation episodes')
    parser.add_argument('--seed', type=int, default=1000,
                        help='Starting random seed')
    parser.add_argument('--output_dir', type=str,
                        default=os.path.join(PROJECT_ROOT, 'eval_results', 'unet'),
                        help='Directory for videos and result files')
    parser.add_argument('--no_video', action='store_true',
                        help='Disable video saving')
    parser.add_argument('--replan_freq', type=int, default=20,
                        help='Re-query GPT every N steps (0 = every action_horizon)')
    parser.add_argument('--gpt_model', type=str, default='gpt-4o',
                        help='OpenAI model to use for stage classification')
    parser.add_argument('--device', type=str, default='cuda:0')

    args = parser.parse_args()

    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.makedirs(args.output_dir, exist_ok=True)

    evaluate(
        task_name=args.task,
        ckpt_dir=args.ckpt_dir,
        config_path=args.config_path,
        num_tests=args.num_tests,
        device=args.device,
        output_dir=args.output_dir,
        start_index=args.seed,
        render_videos=not args.no_video,
        replan_freq=args.replan_freq,
        gpt_model=args.gpt_model,
    )


if __name__ == '__main__':
    main()
