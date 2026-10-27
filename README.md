# Perceptual interfaces and memory for a VLM embodied agent

Code for a master's project that tests how a small vision-language model (Qwen2.5-VL-3B-Instruct, 4-bit) finds objects in a ThreeDWorld (TDW) scene. It compares three perceptual interfaces (text-mediated, direct visual, hybrid) and four memory settings (none, episodic, semantic, combined), plus a CoELA-style baseline in the three-task run.

## Setup

1. Install Python packages: `pip install -r requirements.txt`. Install the CUDA build of PyTorch that matches your GPU.
2. Download the TDW build for your system.
3. Copy `.env.example` to `.env` and fill in `BUILD_PATH` (the TDW build) and `HF_TOKEN`. Never commit `.env`.

## Run

| Command | What it runs | Results folder |
| --- | --- | --- |
| `python main.py` | Task 1, then Task 2 | see below |
| `python main.py --config experiment_config.json` | Task 1: 3 interfaces × 4 memory settings | `experiment_results/` |
| `python main.py --config experiment_config_hybrid.json` | Task 1, hybrid interface | `experiment_results_hybrid/` |
| `python main.py --config three_task_config.json` | Task 2: three tasks in sequence | `three_task_box_memory_results/` |
| `python main.py --config three_task_config_fixed_return.json` | Task 2 with a fixed start for the return task | `three_task_fixed_return_results/` |
| `python main.py --config experiment_config_smoke.json` | Short smoke test | `smoke_test_results/` |

`--config` also accepts a suite file with a `run_configs` list, which runs the listed configs in order.

## Other scripts

- `pack_results.py`: zips the newest run of each kind without images.
- `visualize_results.py`: draws charts for a finished run.
- `replay_visibility.py`: re-asks the model about logged frames.

## Tests

`python -m unittest`
