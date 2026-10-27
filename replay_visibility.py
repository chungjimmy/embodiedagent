import argparse
import copy
import csv
import json
import re
import sys
from pathlib import Path

from planner import Planner
from task_contract import TaskContract


MEMORY_VARIANTS = (
    "logged",
    "none",
    "episodic_only",
    "semantic_only",
    "episodic_without_target_fields",
)
WORDINGS = {
    "without_partial_rule": Planner.TARGET_VISIBILITY_RULES_BASE,
    "with_partial_rule": Planner.TARGET_VISIBILITY_RULES,
}
EPISODIC_TARGET_FIELDS = (
    "target_visible",
    "target_location",
    "target_distance",
    "reason",
)


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description="Controlled offline replay of logged direct_visual frames.")
    parser.add_argument(
        "--episode",
        action="append",
        default=[],
        help="Logged episode report JSON (repeatable).",
    )
    parser.add_argument(
        "--run",
        action="append",
        default=[],
        help="Run folder: replay the missed frames of every direct_visual "
        "episode in it (repeatable).",
    )
    parser.add_argument(
        "--steps",
        action="append",
        nargs="+",
        type=int,
        help="Steps to replay for the preceding --episode (default: missed frames).",
    )
    parser.add_argument("--config", help="resolved_config.json (default: found next to the run).")
    parser.add_argument("--build-path", help="TDW build, to re-render frames without a saved image.")
    parser.add_argument("--out", default="replay_results.csv")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    steps = args.steps or []
    if len(steps) > len(args.episode):
        parser.error("--steps given more often than --episode")
    steps = steps + [None] * (len(args.episode) - len(steps))
    for run in args.run:
        found = reports_in_run(run)
        if not found:
            parser.error(f"no direct_visual episode reports under {run}")
        args.episode.extend(found)
        steps.extend([None] * len(found))
    if not args.episode:
        parser.error("give --run or --episode")
    args.steps = steps
    return args


def reports_in_run(run):
    skip = ("partial_history", "watchdog", "sequence_", "all_results",
            "resolved_config", "manual_review")
    reports = []
    for folder in sorted(Path(run).glob("*direct_visual*")):
        for path in sorted(folder.glob("*.json")):
            if not path.name.startswith(skip):
                reports.append(str(path))
    return reports


def _criterion(step, kind):
    def walk(node):
        if not isinstance(node, dict):
            return None
        if node.get("criterion_type") == kind:
            return node
        for child in node.get("criteria", []) or []:
            found = walk(child)
            if found is not None:
                return found
        return None

    return walk(step.get("environment_success_check")) or {}


def false_sighting_frames(report):
    return [
        step["step"]
        for step in report.get("history", [])
        if (step.get("observation") or {}).get("target_visible")
        and _criterion(step, "object_visibility")
        and not _criterion(step, "object_visibility").get("success")
    ]


def frame_type(report, number):
    step = next((s for s in report.get("history", []) if s.get("step") == number), {})
    visible = bool(_criterion(step, "object_visibility").get("success"))
    confirmed = bool((step.get("observation") or {}).get("target_visible"))
    if visible and not confirmed:
        return "missed"
    if confirmed and not visible:
        return "false_sighting"
    return "visible_confirmed" if visible else "not_visible"


def missed_frames(report):
    return [
        step["step"]
        for step in report.get("history", [])
        if _criterion(step, "object_visibility").get("success")
        and not (step.get("observation") or {}).get("target_visible")
    ]


def find_config(report_path, explicit):
    if explicit:
        return json.loads(Path(explicit).read_text(encoding="utf-8"))
    for parent in Path(report_path).resolve().parents:
        candidate = parent / "resolved_config.json"
        if candidate.exists():
            return json.loads(candidate.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"No resolved_config.json above {report_path}; pass --config.")


def task_spec_for(config, goal):
    for task in config.get("tasks", []):
        if task.get("goal") == goal:
            return task.get("task_spec", {})
    raise KeyError(f"No task with goal {goal!r} in resolved_config.json")


def scene_episode_id(report_path):
    return Path(report_path).resolve().parent.name.split("__")[-1]


def memory_variant(logged, name):
    if name == "logged":
        return copy.deepcopy(logged)
    if name == "none" or not isinstance(logged, dict):
        return None
    if name == "episodic_only":
        if "episodic_memory" not in logged:
            return "not_applicable"
        return {"memory_mode": logged.get("memory_mode"), "episodic_memory": copy.deepcopy(logged["episodic_memory"])}
    if name == "semantic_only":
        if "semantic_memory" not in logged:
            return "not_applicable"
        return {"memory_mode": logged.get("memory_mode"), "semantic_memory": copy.deepcopy(logged["semantic_memory"])}
    if name == "episodic_without_target_fields":
        if "episodic_memory" not in logged:
            return "not_applicable"
        stripped = copy.deepcopy(logged)
        stripped["episodic_memory"] = [
            {k: v for k, v in event.items() if k not in EPISODIC_TARGET_FIELDS}
            for event in stripped["episodic_memory"]
        ]
        return stripped
    raise ValueError(name)


class PromptRecorder:
    def __init__(self):
        self.prompts = []

    def generate(self, prompt, image=None, max_new_tokens=None):
        from vision import GenerationResult

        self.prompts.append(prompt)
        return GenerationResult(text="{}", input_tokens=0, output_tokens=0, inference_seconds=0.0)

    def load(self):
        pass


class Renderer:
    def __init__(self, build_path, config):
        self.build_path = build_path
        self.config = config
        self.scenes = {}

    def _scene(self, episode_id, directory):
        if episode_id in self.scenes:
            return self.scenes[episode_id]
        from main import create_scene

        scene = dict(self.config.get("scene", {}))
        entry = None
        for split in ("evaluation_episodes", "development_episodes"):
            for item in scene.get(split, []) or []:
                if item.get("id") == episode_id:
                    entry = item
        if entry is None:
            raise KeyError(f"scene episode {episode_id!r} not in resolved_config.json")
        for key in ("development_episodes", "evaluation_episodes", "development_seeds", "evaluation_seeds", "seeds"):
            scene.pop(key, None)
        scene.update({k: v for k, v in entry.items() if k not in {"id", "seed"}})
        controller, agent, _ = create_scene(
            build_path=self.build_path,
            scene_config=scene,
            seed=int(entry.get("seed", 0)),
            episode_directory=directory,
        )
        self.scenes[episode_id] = (controller, agent)
        return controller, agent

    def render(self, episode_id, state, directory):
        from main import _set_avatar_pose

        controller, agent = self._scene(episode_id, directory)
        current_pitch = float((agent.get_camera_angles() or {}).get("pitch", 0.0))
        pitch = float((state.get("camera_angles") or {}).get("pitch", current_pitch))
        _set_avatar_pose(
            controller,
            agent.avatar_id,
            state["position"],
            state["yaw"],
            pitch=pitch,
            current_pitch=current_pitch,
        )
        if hasattr(agent, "camera_pitch"):
            agent.camera_pitch = pitch
        return agent.get_current_view()

    def close(self):
        from main import shutdown_controller

        for controller, _ in self.scenes.values():
            shutdown_controller(controller)


def load_image(report_path, step, renderer, episode_id, scratch):
    relative = step.get("observation_image_path_relative")
    if relative:
        path = Path(report_path).resolve().parent / relative.replace("\\", "/")
        if path.exists():
            from PIL import Image

            return Image.open(path).convert("RGB"), "saved"
    if renderer is None:
        return None, "missing (pass --build-path to re-render)"
    return renderer.render(episode_id, step["state_before"], scratch), "rendered"


def raw_fields(text):
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def build_planner(config, report, backend, directory):
    from episode_worker import _create_planner

    condition = {
        "model_name": report["model"],
        "perceptual_interface": report["perceptual_interface"],
        "memory_mode": report["memory_mode"],
        "baseline": report.get("baseline", "primary"),
    }
    return _create_planner(config, condition, backend, directory)


def prepare(planner, config, report, step_index):
    goal = report["goal"]
    task_spec = task_spec_for(config, goal)
    contract = TaskContract.build(goal=goal, configured=task_spec)
    planner._active_task_contract = contract
    planner._active_allowed_actions = set(contract.allowed_actions)
    planner._active_turn_angle_degrees = float(report.get("turn_angle_degrees") or Planner.TURN_ANGLE_DEGREES)
    planner._sighting_handling = planner._sighting_handling_policy(task_spec)
    consistency = planner._completion_consistency_policy(task_spec)
    planner._completion_policy = planner._completion_policy_setting(task_spec, consistency)
    step = report["history"][step_index]
    safe = step.get("collision_safe_actions")
    planner._collision_safe_actions = set(safe) if safe else None
    history = report["history"][:step_index]
    return contract, step, history


def detect_run_wording(planner, decide_kwargs, logged_memory, logged_chars):
    real_backend = planner.backend
    try:
        for name, rules in WORDINGS.items():
            recorder = PromptRecorder()
            planner.backend = recorder
            planner.TARGET_VISIBILITY_RULES = rules
            planner._decide(memory_context=copy.deepcopy(logged_memory), **decide_kwargs)
            if recorder.prompts and len(recorder.prompts[-1]) == logged_chars:
                return name
    finally:
        planner.backend = real_backend
    print("    note: neither wording reproduces the logged prompt length; "
          "treating the current wording as the run's")
    return "with_partial_rule"


def replay(args):
    import tempfile

    rows = []
    renderer = None
    backend = PromptRecorder() if args.dry_run else None
    scratch = Path(tempfile.mkdtemp(prefix="replay_visibility_"))
    try:
        for report_path, wanted in zip(args.episode, args.steps):
            report = json.loads(Path(report_path).read_text(encoding="utf-8"))
            if report.get("perceptual_interface") != "direct_visual":
                print(f"skip {report_path}: only direct_visual frames are replayed")
                continue
            config = find_config(report_path, args.config)
            if backend is None:
                from episode_worker import _create_backend

                backend = _create_backend(config, report["model"])
                backend.load()
            if renderer is None and args.build_path and not args.dry_run:
                renderer = Renderer(args.build_path, config)
            elif renderer is not None:
                renderer.config = config
            steps = wanted or sorted(
                set(missed_frames(report)) | set(false_sighting_frames(report))
            )
            index_of = {item.get("step"): i for i, item in enumerate(report["history"])}
            label = Path(report_path).resolve().parent.name.split("__", 2)[-1]
            print(f"{label}: steps {steps}")
            for number in steps:
                if number not in index_of:
                    print(f"  step {number}: not in the log")
                    continue
                planner = build_planner(config, report, backend, scratch)
                contract, step, history = prepare(planner, config, report, index_of[number])
                logged_call = next(
                    (c for c in step.get("model_calls", []) if c.get("stage") == "status_then_action"),
                    {},
                )
                if args.dry_run:
                    image, image_source = None, "not needed (dry run)"
                else:
                    image, image_source = load_image(
                        report_path, step, renderer, scene_episode_id(report_path), scratch
                    )
                    if image is None:
                        print(f"  step {number}: image {image_source}; skipped")
                        continue
                decide_kwargs = dict(
                    image=image,
                    perception=None,
                    goal=report["goal"],
                    blocked_action=planner._get_blocked_action(history),
                    action_count=len([h for h in history if h.get("executed")]),
                    max_steps=int(config.get("max_steps", 20)),
                    current_state=step.get("state_before") or {},
                    recent_history=history,
                    task_contract=contract,
                )
                run_wording = detect_run_wording(
                    planner, decide_kwargs, step.get("memory_supplied"),
                    logged_call.get("prompt_characters"),
                )
                other = next(w for w in WORDINGS if w != run_wording)
                variants = [("memory", name, run_wording) for name in MEMORY_VARIANTS]
                variants.append(("wording", "logged", other))
                for factor, memory_name, wording in variants:
                    memory = memory_variant(step.get("memory_supplied"), memory_name)
                    if memory == "not_applicable" or (
                        memory_name != "logged"
                        and memory == step.get("memory_supplied")
                    ):
                        continue
                    planner.TARGET_VISIBILITY_RULES = WORDINGS[wording]
                    before = len(getattr(backend, "prompts", []))
                    decision, call = planner._decide(
                        memory_context=memory, **decide_kwargs
                    )
                    raw_text = (call or {}).get("raw_response", "")
                    raw = raw_fields(raw_text)
                    baseline = memory_name == "logged" and wording == run_wording
                    prompt_chars = (call or {}).get("prompt_characters")
                    if args.dry_run and len(backend.prompts) > before:
                        prompt_chars = len(backend.prompts[-1])
                    rows.append({
                        "episode": label,
                        "step": number,
                        "frame_type": frame_type(report, number),
                        "image": image_source,
                        "factor": "baseline" if baseline else factor,
                        "memory": memory_name,
                        "wording": wording,
                        "target_visible_raw": raw.get("target_visible"),
                        "target_visible_validated": decision.get("target_visible"),
                        "target_location": raw.get("target_location"),
                        "target_distance": raw.get("target_distance"),
                        "status": raw.get("status"),
                        "visual_evidence": raw.get("visual_evidence"),
                        "prompt_characters": prompt_chars,
                        "logged_prompt_characters": logged_call.get("prompt_characters") if baseline else "",
                        "reproduces_logged_response": (
                            (raw_text.strip() == (logged_call.get("raw_response") or "").strip())
                            if baseline and not args.dry_run else ""
                        ),
                        "raw_response": raw_text,
                    })
    finally:
        if renderer is not None:
            renderer.close()
    return rows


def summarize(rows, dry_run):
    if dry_run:
        print("\nBaseline prompt fidelity (built vs logged prompt length):")
        for row in rows:
            if row["factor"] == "baseline":
                same = row["prompt_characters"] == row["logged_prompt_characters"]
                print(f"  {row['episode']} step {row['step']}: {row['prompt_characters']} vs "
                      f"{row['logged_prompt_characters']} {'OK' if same else 'MISMATCH'}")
        return
    print("\nFrames flagged visible (validated) per variant, by frame type"
          " (missed: target truly visible, higher is better;"
          " false_sighting: target not visible, lower is better):")
    keys = []
    for row in rows:
        key = (row["memory"], row["wording"])
        if key not in keys:
            keys.append(key)
    for key in keys:
        parts = []
        for kind in ("missed", "false_sighting"):
            subset = [r for r in rows if (r["memory"], r["wording"]) == key
                      and r["frame_type"] == kind]
            if subset:
                flagged = sum(r["target_visible_validated"] is True for r in subset)
                parts.append(f"{kind} {flagged}/{len(subset)}")
        print(f"  memory={key[0]:31} wording={key[1]:20} " + "   ".join(parts))
    base = [r for r in rows if r["factor"] == "baseline"]
    reproduced = sum(r["reproduces_logged_response"] is True for r in base)
    print(f"\nBaseline reproduced the logged response on {reproduced}/{len(base)} frames.")


def main(argv=None):
    args = parse_arguments(argv)
    rows = replay(args)
    if not rows:
        print("Nothing replayed.")
        return 1
    with open(args.out, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summarize(rows, args.dry_run)
    print(f"\nWrote {len(rows)} rows to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
