# Perceptual interfaces and memory for a VLM embodied agent

Code for a master's project that tests how a small vision-language model (Qwen2.5-VL-3B-Instruct, 4-bit) finds objects in a ThreeDWorld (TDW) scene. It compares three perceptual interfaces (text-mediated, direct visual, hybrid) and four memory settings (none, episodic, semantic, combined), plus a CoELA-style baseline in the three-task run.

## Versions

This branch holds the revised agent (shorter prompts, a target-first controller, an initial scan) and the final Qwen run, and it is the version described in the final paper. The earlier version, with the longer prompts and simpler controller, is kept for comparison:

- To get the earlier code and its results, download or check out the `earlier-version` tag (`git checkout earlier-version`, or choose the tag on GitHub under Releases / the branch menu).
- The paper-version configs (`experiment_config.json`, `three_task_config.json`, `three_task_config_fixed_return.json`) still run the earlier behavior on this branch, because every new option is off unless a config turns it on.

## Setup

1. Install Python packages: `pip install -r requirements.txt`. Install the CUDA build of PyTorch that matches your GPU.
2. Download the TDW build for your system.
3. Copy `.env.example` to `.env` and fill in `BUILD_PATH` (the TDW build) and `HF_TOKEN`. Never commit `.env`.

## Run

| Command | What it runs | Results folder |
| --- | --- | --- |
| `python main.py` | Final Qwen run: Task 1, then Task 2 (same as `run_qwen.json`) | `experiment_results_final/`, `three_task_final_results/` |
| `python main.py --config experiment_config.json` | Task 1, paper version: 2 interfaces × 4 memory settings | `experiment_results/` |
| `python main.py --config experiment_config_hybrid.json` | Task 1, hybrid interface | `experiment_results_hybrid/` |
| `python main.py --config three_task_config.json` | Task 2, paper version: three tasks in sequence | `three_task_box_memory_results/` |
| `python main.py --config three_task_config_fixed_return.json` | Task 2 with a fixed start for the return task | `three_task_fixed_return_results/` |
| `python main.py --config experiment_config_smoke.json` | Short smoke test | `smoke_test_results/` |
| `python main.py --config experiment_config_final_check.json` | Quick check: text-mediated, no memory, 3 starts, final settings | `final_check_results/` |
| `python main.py --config run_other_models.json` | Small comparison: SmolVLM2-2.2B and Gemma-3-4B, text-mediated with no memory and combined memory, 3 starts each, final settings | `experiment_results_smolvlm/`, `experiment_results_gemma/` |
| `python main.py --config run_qwen.json` | Final Qwen suite: Task 1 (3 interfaces × 4 memory settings × 3 starts) and Task 2 fixed-start return, with the final settings | `experiment_results_final/`, `three_task_final_results/` |
| `python main.py --config run_return_poses.json` | Task 2 again with two other fixed return poses where the pink box was out of view at every logged heading: A at (-7.75, 1.5) and B at (-15.0, 1.0), same conditions as the final run | `three_task_return_a_results/`, `three_task_return_b_results/` |

`--config` also accepts a suite file with a `run_configs` list, which runs the listed configs in order.

## Optional task_spec settings

All are off by default, so the original configs behave as before.

- `"simplified_rules": true` uses short prompts. The model describes what it sees before it picks an action, and the prompts have five decision rules, no `target_correction` field, and no `impossible` status.
- `"initial_scan": true` makes the agent turn in place to look around its starting position (up to six turns) before walking, until it first sees the target. Memory-guided moves are not replaced.
- `"controller_priority": "target_first"` sets the controller order: collision safety, then approaching a confirmed target, then the model's action. While the target is visible anywhere in the image, the agent moves forward toward it, because one 45-degree turn shifts the view by most of the image. If the target leaves the view at one side after a forward move, the agent turns once to that side. When moving forward is blocked, it goes around the obstacle (turn toward the target, step forward, turn back) instead of backing away. The extra re-check after a turn is turned off in this mode. An optional close-range re-check (`"close_range_recheck": true`, off by default) asks perception again when the target disappears right after the agent walks toward it; in testing it produced false sightings. A visible target is never dropped because it switched sides.

## Other scripts

- `pack_results.py`: zips results without images. With no arguments it takes the newest run from each final folder (`experiment_results_final`, `three_task_final_results`, `experiment_results_smolvlm`, `experiment_results_gemma`). Add `--all` for every results folder, or pass run folders by name.
- `visualize_results.py`: draws charts for a finished run.
- `replay_visibility.py`: re-asks the model about logged frames.

## Tests

`python -m unittest`
