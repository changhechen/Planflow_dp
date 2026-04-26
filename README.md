# Planflow-DP

Planflow-DP is a multi-task robot manipulation framework that combines a **1D UNet diffusion policy** with a **GPT-4o stage planner** (we will upload the Diffusion Transformer version very soon). During evaluation, a frozen CLIP text encoder provides language conditioning and GPT-4o dynamically selects task stages from live camera images to guide the policy through long-horizon manipulation tasks on [ManiSkill](https://github.com/haosulab/ManiSkill) environments.

## Repository Structure

```
Planflow_dp/
├── data/               # Dataset loaders (multi-task ManiSkill)
├── evaluation/         # Evaluation script and shell launcher
├── inference/          # Inference wrapper for the UNet policy
├── maniskill_envs/     # Custom ManiSkill task environments
├── model/              # UNet and language projector architectures
├── training/           # Training script, config YAML, and shell launcher
├── checkpoints/        # Pre-trained model checkpoints
└── conda_environment_new.yaml  # Full conda environment
```

## Environment Setup

### 1. Create the Conda Environment

```bash
conda env create -f conda_environment_new.yaml
conda activate robodiff
```

### 2. Install ManiSkill Environments

The custom environments in `maniskill_envs/` (table sorting, tool usage) must be registered before training or evaluation:

```bash
pip install -e maniskill_envs/sort_table
pip install -e maniskill_envs/tool_usage
```

---

## Data Setup

Training requires two parallel directory trees:

| Config key | Contents | Example path |
|---|---|---|
| `action_data_dir` | Per-task subdirectories of `.pkl` action trajectory files | `/home/aiden/MoT_maniskill_sort_100/action_data/` |
| `language_data_dir` | Per-task `.jsonl` language instruction files | `/home/aiden/MoT_maniskill_sort_100/language_data/` |

Set these paths in the training config file before running training:

```
training/maniskill_unet_lang_config.yaml
```

```yaml
action_data_dir:   /path/to/your/action_data/
language_data_dir: /path/to/your/language_data/
```

The dataset loader automatically discovers all task subdirectories inside `action_data_dir` and pairs them with the corresponding `.jsonl` files in `language_data_dir`.

---

## OpenAI API Key (Evaluation Only)

The evaluation script calls **GPT-4o** to classify task stages from live camera images. You must provide an OpenAI API key before running evaluation.

**Option A — export in your shell (recommended):**

```bash
export OPENAI_API_KEY="sk-..."
```

**Option B — edit the evaluation shell script directly:**

Open `evaluation/action_expert_unet_eval.sh` and replace the placeholder:

```bash
export OPENAI_API_KEY="Your Key"   # <-- replace with your actual key
```

---

## Training

All training runs from the repo root.

### Command

```bash
python -m training.train_maniskill_MoTdiffusion_unet_clip \
    --config training/maniskill_unet_lang_config.yaml
```

Or use the provided shell script:

```bash
bash training/training_unet_script.sh
```

### Key Config Options (`training/maniskill_unet_lang_config.yaml`)

| Key | Description | Default |
|---|---|---|
| `action_data_dir` | Path to action trajectory data | `/home/aiden/MoT_maniskill_sort_100/action_data/` |
| `language_data_dir` | Path to language instruction data | `/home/aiden/MoT_maniskill_sort_100/language_data/` |
| `models_save_dir` | Directory to save checkpoints | `trained_model_tool_usage_4` |
| `num_epochs` | Total training epochs | `200` |
| `num_demos_per_task` | Demonstrations per task | `100` |
| `batch_size` | Training batch size | `64` |
| `lr` | Learning rate | `0.0001` |
| `pred_horizon` | Action prediction horizon | `16` |
| `obs_horizon` | Observation horizon | `2` |
| `action_horizon` | Actions executed per inference step | `8` |
| `wandb` | Enable Weights & Biases logging | `false` |

Checkpoints are saved every `eval_epoch` epochs (default `20`) as `checkpoint_epoch_<N>/` inside `models_save_dir`.

---

## Evaluation

Evaluation runs the trained UNet policy inside ManiSkill and uses GPT-4o to dynamically select the language instruction for each stage of the task.

### Command

```bash
# Set API key first
export OPENAI_API_KEY="sk-..."

python evaluation/action_expert_unet_eval.py \
    --task fruit_first \
    --ckpt_dir trained_model_tool_usage_3/checkpoint_epoch_200 \
    --config_path training/maniskill_unet_lang_config.yaml \
    --num_tests 50 \
    --output_dir eval_results/unet
```

Or use the provided shell script (after setting the API key inside it):

```bash
bash evaluation/action_expert_unet_eval.sh
```

### Arguments

| Argument | Description | Default |
|---|---|---|
| `--task` | Task to evaluate (`fruit_first`, `garbage_first`, `pull_tool`) | `fruit_first` |
| `--ckpt_dir` | Path to the checkpoint directory | `trained_model_tool_usage_3/checkpoint_epoch_200` |
| `--config_path` | Path to the training YAML config | `training/maniskill_unet_lang_config.yaml` |
| `--num_tests` | Number of evaluation episodes | `50` |
| `--seed` | Starting random seed | `1000` |
| `--output_dir` | Directory for result files and videos | `eval_results/unet` |
| `--no_video` | Disable MP4 video saving | off |
| `--replan_freq` | Re-query GPT every N steps (0 = every `action_horizon`) | `20` |
| `--gpt_model` | OpenAI model for stage classification | `gpt-4o` |
| `--device` | Torch device | `cuda:0` |

### Outputs

Results are saved to `--output_dir`:

- `results_<task>_<timestamp>.json` — per-episode success flags and success rate
- `<task>_ep<N>.mp4` — rendered rollout videos with stage labels overlaid (unless `--no_video`)

An episode is counted as successful when the environment reward exceeds **0.99**.
