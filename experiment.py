import csv
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import shutil
import time

from memory import EmbeddingExperienceStore, MemoryManager
from planner import CoELABaselinePlanner, Planner
from task_contract import TaskContract
from vision import VisionLanguageModel


def expand_config_sequence(config_path):
    path = Path(config_path)
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict) or "run_configs" not in data:
        return [path]
    entries = data["run_configs"]
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path}: run_configs must be a non-empty list.")
    resolved = []
    for entry in entries:
        child = Path(entry)
        if not child.is_absolute():
            child = path.parent / child
        if not child.is_file():
            raise ValueError(f"{path}: config not found: {entry}")
        with child.open("r", encoding="utf-8") as file:
            child_data = json.load(file)
        if isinstance(child_data, dict) and "run_configs" in child_data:
            raise ValueError(f"{path}: suites cannot contain other suites ({entry}).")
        resolved.append(child)
    return resolved


@dataclass(frozen=True)
class Condition:
    model_name: str
    perceptual_interface: str
    memory_mode: str
    baseline: str = "primary"


class ExperimentRunner:
    def __init__(
        self,
        config,
        episode_factory,
        episode_cleanup=None,
        isolated_episode_executor=None,
        isolated_sequence_executor=None,
    ):
        self.config = config
        self._validate_study_design()
        self.episode_factory = episode_factory
        self.episode_cleanup = episode_cleanup
        self.isolated_episode_executor = isolated_episode_executor
        self.isolated_sequence_executor = isolated_sequence_executor
        run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_directory = Path(
            config.get("output_directory", "experiment_results")
        ) / run_stamp
        self.output_directory.mkdir(parents=True, exist_ok=True)
        self.results = []
        self._write_json(self.output_directory / "resolved_config.json", config)

    def _validate_study_design(self):
        design = self.config.get("study_design", {})
        models = self.config.get("models", [])
        if design.get("require_single_model", False) and len(models) != 1:
            raise ValueError(
                "This study configuration requires exactly one VLM; "
                f"received {len(models)}."
            )
        if (
            design.get("exclude_coela_baseline", False)
            and self.config.get("coela_baseline", {}).get("enabled", False)
        ):
            raise ValueError(
                "CoELA is excluded by study_design but enabled in "
                "coela_baseline."
            )

    @classmethod
    def from_file(cls, config_path, episode_factory, episode_cleanup=None):
        with Path(config_path).open("r", encoding="utf-8") as file:
            return cls(json.load(file), episode_factory, episode_cleanup)

    def build_conditions(self):
        conditions = []
        interfaces = self.config.get(
            "perceptual_interfaces", ["direct_visual", "text_mediated"]
        )
        memory_modes = self.config.get(
            "memory_modes", ["none", "episodic", "semantic", "combined"]
        )
        for model in self.config["models"]:
            model_name = model["name"] if isinstance(model, dict) else model
            for interface in interfaces:
                for memory_mode in memory_modes:
                    conditions.append(
                        Condition(model_name, interface, memory_mode)
                    )
            if self.config.get("coela_baseline", {}).get("enabled", False):
                conditions.append(
                    Condition(
                        model_name,
                        "coela",
                        "combined",
                        baseline="coela_style_single_agent",
                    )
                )
        return conditions

    def run(self):
        run_primary_conditions = bool(
            self.config.get("run_primary_conditions", True)
        )
        all_conditions = self.build_conditions() if run_primary_conditions else []
        model_names = [
            item["name"] if isinstance(item, dict) else item
            for item in self.config["models"]
        ]
        episodes = self._episode_specs()
        tasks = self.config.get("tasks", [])
        sequential_enabled = self._sequential_extension_enabled()
        print(
            "Experiment plan: "
            f"{len(model_names)} models, {len(all_conditions)} conditions, "
            f"{len(tasks)} tasks, {len(episodes)} scene episodes "
            f"({len(all_conditions) * len(tasks) * len(episodes)} episodes)."
            + (
                " Primary condition sweep disabled; running enabled extensions only."
                if not run_primary_conditions
                else ""
            ),
            flush=True,
        )
        for model_name in model_names:
            conditions = [
                item for item in all_conditions if item.model_name == model_name
            ]
            if self.config.get("randomize_condition_order", True):
                order_seed = int(self.config.get("condition_order_seed", 2026))
                random.Random(f"{order_seed}:{model_name}").shuffle(conditions)
            requires_parent_backend = (
                bool(conditions)
                and self.isolated_episode_executor is None
            ) or (
                sequential_enabled
                and self.isolated_sequence_executor is None
            )
            backend = None
            loaded = False
            try:
                if requires_parent_backend:
                    backend = self._create_backend(model_name)
                    print(f"Loading model: {model_name}", flush=True)
                    backend.load()
                    loaded = True
                if self.isolated_episode_executor is not None and conditions:
                    print(
                        "Running isolated watchdog episodes for model: "
                        f"{model_name}",
                        flush=True,
                    )
                for condition in conditions:
                    try:
                        self._run_condition(
                            condition,
                            backend if self.isolated_episode_executor is None else None,
                        )
                    except Exception as error:
                        print(
                            f"Condition failed for {model_name} / "
                            f"{condition.perceptual_interface} / "
                            f"{condition.memory_mode}: {error}",
                            flush=True,
                        )
                        self._record_condition_errors(condition, error)
                if sequential_enabled:
                    first_sequence_error = None
                    for sequence_condition in self._sequential_conditions(
                        model_name
                    ):
                        try:
                            if self.isolated_sequence_executor is not None:
                                self._run_sequential_extension_isolated(
                                    model_name, sequence_condition
                                )
                            else:
                                self._run_sequential_extension(
                                    model_name, backend, sequence_condition
                                )
                        except Exception as error:
                            print(
                                "Sequential condition failed for "
                                f"{model_name} / "
                                f"{sequence_condition.perceptual_interface} / "
                                f"{sequence_condition.memory_mode}: {error}",
                                flush=True,
                            )
                            if first_sequence_error is None:
                                first_sequence_error = error
                    if first_sequence_error is not None:
                        raise first_sequence_error
            except Exception as error:
                if requires_parent_backend and not loaded:
                    print(
                        f"Model could not be loaded: {model_name}: {error}",
                        flush=True,
                    )
                    for condition in conditions:
                        self._record_condition_errors(condition, error)
                    if sequential_enabled:
                        for sequence_condition in self._sequential_conditions(
                            model_name
                        ):
                            self._record_sequential_errors(
                                model_name, error, sequence_condition
                            )
                else:
                    print(
                        f"Model-level error for {model_name}: {error}",
                        flush=True,
                    )
            finally:
                if loaded:
                    try:
                        backend.unload()
                    except Exception as error:
                        print(
                            f"Model unload warning for {model_name}: {error}",
                            flush=True,
                        )

        self._save_summary()
        return self.results

    def _episode_specs(self):
        scene = self.config.get("scene", {})
        split = self.config.get("run_split", "evaluation")
        configured = scene.get(f"{split}_episodes")
        if configured is not None:
            if not isinstance(configured, list) or not configured:
                raise ValueError(
                    f"scene.{split}_episodes must be a non-empty list."
                )
            episodes = []
            for index, entry in enumerate(configured, start=1):
                if not isinstance(entry, dict):
                    raise ValueError(
                        f"scene.{split}_episodes[{index - 1}] must be an object."
                    )
                episode_id = str(
                    entry.get("id") or f"{split}_{index:02d}"
                ).strip()
                seed = int(entry.get("seed", index - 1))
                resolved_scene = dict(scene)
                for key in (
                    "development_episodes",
                    "evaluation_episodes",
                    "development_seeds",
                    "evaluation_seeds",
                    "seeds",
                ):
                    resolved_scene.pop(key, None)
                resolved_scene.update(
                    {
                        key: value
                        for key, value in entry.items()
                        if key not in {"id", "seed"}
                    }
                )
                episodes.append(
                    {
                        "id": episode_id,
                        "seed": seed,
                        "scene_config": resolved_scene,
                    }
                )
            if self.config.get("study_design", {}).get(
                "require_explicit_scene_instances", False
            ):
                fingerprints = [
                    (
                        item["scene_config"].get("name"),
                        json.dumps(
                            item["scene_config"].get("initial_position", {}),
                            sort_keys=True,
                        ),
                        float(item["scene_config"].get("initial_yaw", 0)),
                    )
                    for item in episodes
                ]
                if len(fingerprints) != len(set(fingerprints)):
                    raise ValueError(
                        f"scene.{split}_episodes contains duplicate scene, "
                        "position, and heading combinations."
                    )
            return episodes

        seeds = scene.get(f"{split}_seeds", scene.get("seeds", [0]))
        if (
            self.config.get("study_design", {}).get(
                "require_explicit_scene_instances", False
            )
            and scene.get("name")
            and len(seeds) > 1
        ):
            raise ValueError(
                "A named preloaded scene does not vary merely because its "
                "seed changes. Configure explicit development_episodes or "
                "evaluation_episodes with distinct poses/headings."
            )
        return [
            {
                "id": f"seed_{seed}",
                "seed": int(seed),
                "scene_config": scene,
            }
            for seed in seeds
        ]

    def _run_condition(self, condition, backend):
        tasks = self.config.get("tasks", [])
        split = self.config.get("run_split", "evaluation")
        for episode in self._episode_specs():
            seed = episode["seed"]
            episode_label = episode["id"]
            for task in tasks:
                task_id = task.get("id", "task")
                episode_id = self._episode_id(
                    condition, task_id, episode_label
                )
                history_directory = self.output_directory / episode_id
                controller = None
                print(f"Starting episode: {episode_id}", flush=True)
                try:
                    self._cleanup_build_processes()
                    if self.isolated_episode_executor is not None:
                        outcome = self.isolated_episode_executor(
                            {
                                "config": self.config,
                                "condition": {
                                    "model_name": condition.model_name,
                                    "perceptual_interface": (
                                        condition.perceptual_interface
                                    ),
                                    "memory_mode": condition.memory_mode,
                                    "baseline": condition.baseline,
                                },
                                "scene_config": episode["scene_config"],
                                "seed": seed,
                                "history_directory": str(history_directory),
                                "task": task,
                                "episode_metadata": {
                                    "episode_id": episode_id,
                                    "task_id": task_id,
                                    "scene_episode_id": episode_label,
                                    "scene_seed": seed,
                                    "split": split,
                                    "evaluation_episode": split == "evaluation",
                                },
                            }
                        )
                        outcome_kind = outcome.get("outcome")
                        if outcome_kind == "completed":
                            report = outcome.get("report")
                            if not isinstance(report, dict):
                                raise RuntimeError(
                                    "Watchdog worker returned no episode report."
                                )
                            self.results.append(report)
                        elif outcome_kind == "timeout":
                            report = self._timeout_result(
                                condition=condition,
                                task=task,
                                seed=seed,
                                episode_id=episode_id,
                                scene_episode_id=episode_label,
                                timeout_seconds=outcome.get("timeout_seconds"),
                                duration_seconds=outcome.get("duration_seconds"),
                                checkpoint_path=outcome.get("checkpoint_path"),
                                partial_history_path=outcome.get(
                                    "partial_history_path"
                                ),
                                worker_exitcode=outcome.get("worker_exitcode"),
                            )
                            self.results.append(report)
                        else:
                            error_type = outcome.get("error_type", "RuntimeError")
                            message = outcome.get(
                                "message",
                                "Watchdog worker failed without a report.",
                            )
                            report = self._error_result(
                                condition,
                                task,
                                seed,
                                RuntimeError(f"{error_type}: {message}"),
                                episode_id,
                                scene_episode_id=episode_label,
                            )
                            report.update(
                                {
                                    "duration_seconds": outcome.get(
                                        "duration_seconds", 0
                                    ),
                                    "watchdog_checkpoint_path": outcome.get(
                                        "checkpoint_path"
                                    ),
                                    "partial_history_path": outcome.get(
                                        "partial_history_path"
                                    ),
                                    "watchdog_exitcode": outcome.get(
                                        "worker_exitcode"
                                    ),
                                }
                            )
                            self.results.append(report)
                        print(
                            f"Finished episode: {episode_id} | "
                            f"status={report.get('status')} | "
                            f"actions={report.get('actions_taken')}",
                            flush=True,
                        )
                        continue
                    controller, agent, scene_metadata = self._create_episode_with_retry(
                        scene_config=episode["scene_config"],
                        seed=seed,
                        episode_directory=history_directory,
                    )
                    planner = self._create_planner(
                        condition=condition,
                        backend=backend,
                        history_directory=history_directory,
                    )
                    report = planner.run_task(
                        agent=agent,
                        goal=task["goal"],
                        max_steps=int(
                            task.get(
                                "max_steps",
                                self.config.get("max_steps", 20),
                            )
                        ),
                        episode_metadata={
                            "episode_id": episode_id,
                            "task_id": task_id,
                            "scene_episode_id": episode_label,
                            "scene_seed": seed,
                            "split": split,
                            "evaluation_episode": split == "evaluation",
                            **scene_metadata,
                        },
                        success_evaluator=(
                            lambda history=None: agent.evaluate_success(
                                task.get("success_criteria"),
                                history=history,
                            )
                        ),
                        task_spec=task.get("task_spec"),
                    )
                    self.results.append(report)
                    print(
                        f"Finished episode: {episode_id} | "
                        f"status={report.get('status')} | "
                        f"actions={report.get('actions_taken')}",
                        flush=True,
                    )
                except Exception as error:
                    self.results.append(
                        self._error_result(
                            condition,
                            task,
                            seed,
                            error,
                            episode_id,
                            scene_episode_id=episode_label,
                        )
                    )
                    print(
                        f"Episode failed: {episode_id} | "
                        f"{type(error).__name__}: {error}",
                        flush=True,
                    )
                finally:
                    self._terminate_controller(controller)
                    self._cleanup_build_processes()
                    time.sleep(float(self.config.get("relaunch_delay_seconds", 1)))
                    self._save_summary()

    def _create_episode_with_retry(
        self, scene_config, seed, episode_directory, max_attempts=3
    ):
        attempts = max(1, int(self.config.get("tdw_start_retries", max_attempts)))
        last_error = None
        for attempt in range(1, attempts + 1):
            try:
                return self.episode_factory(
                    scene_config=scene_config,
                    seed=seed,
                    episode_directory=episode_directory,
                )
            except Exception as error:
                last_error = error
                text = str(error).lower()
                port_collision = (
                    "address in use" in text or "tcp://*:1071" in text
                )
                if not port_collision or attempt >= attempts:
                    raise
                print(
                    f"TDW port collision on attempt {attempt}/{attempts}; "
                    "cleaning up and retrying.",
                    flush=True,
                )
                self._cleanup_build_processes()
                time.sleep(min(10, attempt * 2))
        raise last_error

    def _record_condition_errors(self, condition, error):
        for episode in self._episode_specs():
            seed = episode["seed"]
            episode_label = episode["id"]
            for task in self.config.get("tasks", []):
                task_id = task.get("id", "task")
                episode_id = self._episode_id(
                    condition, task_id, episode_label
                )
                self.results.append(
                    self._error_result(
                        condition,
                        task,
                        seed,
                        error,
                        episode_id,
                        scene_episode_id=episode_label,
                    )
                )

    def _create_planner(self, condition, backend, history_directory):
        memory_context = self.config.get("memory_context", {})
        if not isinstance(memory_context, dict):
            raise ValueError("memory_context must be an object when configured.")
        memory = MemoryManager(
            episodic_window=int(self.config.get("episodic_window", 8)),
            learned_store=self._learned_store(),
            context_episodic_limit=int(
                memory_context.get("episodic_events", 4)
            ),
            context_object_limit=int(
                memory_context.get("semantic_objects", 12)
            ),
            context_pose_limit=int(
                memory_context.get("visited_poses", 8)
            ),
            context_fact_limit=int(
                memory_context.get("semantic_facts", 12)
            ),
            context_text_char_limit=int(
                memory_context.get("text_characters", 1000)
            ),
            context_prompt_char_limit=int(
                memory_context.get("prompt_characters", 3000)
            ),
        )
        common = {
            "model_name": condition.model_name,
            "backend": backend,
            "memory_manager": memory,
            "history_directory": history_directory,
            "perception_max_new_tokens": int(
                self.config.get("perception_max_new_tokens", 300)
            ),
            "perception_retry_max_new_tokens": int(
                self.config.get("perception_retry_max_new_tokens", 224)
            ),
            "decision_max_new_tokens": int(
                self.config.get("decision_max_new_tokens", 300)
            ),
            "completion_max_new_tokens": int(
                self.config.get("completion_max_new_tokens", 64)
            ),
            "save_step_images": bool(
                self.config.get("save_step_images", False)
            ),
        }
        if condition.baseline == "coela_style_single_agent":
            return CoELABaselinePlanner(**common)
        return Planner(
            perceptual_interface=condition.perceptual_interface,
            memory_mode=condition.memory_mode,
            baseline_name=condition.baseline,
            memory_guided_search=bool(
                self.config.get("memory_guided_search", True)
            ),
            **common,
        )

    def _create_backend(self, model_name):
        model_config = next(
            (
                item
                for item in self.config["models"]
                if isinstance(item, dict) and item["name"] == model_name
            ),
            {},
        )
        return VisionLanguageModel(
            model_name=model_name,
            use_4bit=model_config.get(
                "use_4bit", self.config.get("use_4bit", True)
            ),
            max_new_tokens=int(model_config.get("max_new_tokens", 300)),
            min_pixels=model_config.get("min_pixels"),
            max_pixels=model_config.get("max_pixels"),
            clear_cuda_cache_between_calls=bool(
                model_config.get("clear_cuda_cache_between_calls", False)
            ),
            generation_max_time_seconds=model_config.get(
                "generation_max_time_seconds"
            ),
            four_bit_compute_dtype=model_config.get(
                "four_bit_compute_dtype",
                self.config.get("four_bit_compute_dtype", "auto"),
            ),
            log_cuda_memory=bool(
                model_config.get(
                    "log_cuda_memory",
                    self.config.get("log_cuda_memory", True),
                )
            ),
            cuda_allocator_config=model_config.get(
                "cuda_allocator_config",
                self.config.get("cuda_allocator_config"),
            ),
        )

    def _learned_store(self):
        extension = self.config.get("extensions", {}).get("learned_memory", {})
        if not extension.get("enabled", False):
            return None
        return EmbeddingExperienceStore(
            directory=extension.get("directory", "learned_memory"),
            model_name=extension.get(
                "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            frozen=extension.get("frozen", True),
        )

    def _sequential_extension_enabled(self):
        extension = self.config.get("extensions", {}).get("sequential_tasks", {})
        tasks = extension.get("tasks", self.config.get("tasks", []))
        return bool(extension.get("enabled", False)) and len(tasks) >= 2

    VALID_SEQUENTIAL_INTERFACES = ("direct_visual", "text_mediated", "hybrid", "coela")

    def _sequential_conditions(self, model_name):
        extension = self.config.get("extensions", {}).get("sequential_tasks", {})
        default_memory = extension.get("memory_mode", "combined")
        configured = extension.get("conditions")
        if configured is None:
            entries = [
                {
                    "perceptual_interface": extension.get(
                        "perceptual_interface", "direct_visual"
                    ),
                    "memory_mode": default_memory,
                }
            ]
        else:
            if not isinstance(configured, list) or not configured:
                raise ValueError(
                    "extensions.sequential_tasks.conditions must be a "
                    "non-empty list."
                )
            entries = configured
        conditions = []
        seen = set()
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError(
                    "Each extensions.sequential_tasks.conditions entry must "
                    "be an object with perceptual_interface and memory_mode."
                )
            interface = entry.get("perceptual_interface")
            if interface not in self.VALID_SEQUENTIAL_INTERFACES:
                raise ValueError(
                    "extensions.sequential_tasks.conditions needs a "
                    "perceptual_interface of "
                    + ", ".join(self.VALID_SEQUENTIAL_INTERFACES)
                    + f"; got {interface!r}."
                )
            memory_mode = entry.get("memory_mode", default_memory)
            if (interface, memory_mode) in seen:
                raise ValueError(
                    "extensions.sequential_tasks.conditions lists "
                    f"{interface} / {memory_mode} more than once."
                )
            seen.add((interface, memory_mode))
            baseline = "sequential_extension"
            if interface == "coela":
                if memory_mode != "combined":
                    raise ValueError(
                        "The coela sequential condition always uses combined "
                        f"memory; got {memory_mode!r}."
                    )
                baseline = "coela_style_single_agent"
            conditions.append(
                Condition(
                    model_name,
                    interface,
                    memory_mode,
                    baseline=baseline,
                )
            )
        return conditions

    def _sequential_extension_details(self, model_name, condition=None):
        extension = self.config.get("extensions", {}).get("sequential_tasks", {})
        tasks = extension.get("tasks", self.config.get("tasks", []))
        if not isinstance(tasks, list) or len(tasks) < 2:
            raise ValueError(
                "extensions.sequential_tasks requires at least two tasks."
            )
        first_episode = self._episode_specs()[0]
        seed = int(extension.get("seed", first_episode["seed"]))
        if condition is None:
            condition = self._sequential_conditions(model_name)[0]
        return extension, tasks, first_episode, seed, condition

    @staticmethod
    def _sequence_precondition_failure(task, reports_by_task, memory):
        configured = task.get("sequential_preconditions", {})
        if configured in (None, ""):
            return None
        if not isinstance(configured, dict):
            raise ValueError(
                "task.sequential_preconditions must be an object when set."
            )
        required_task_ids = configured.get("requires_successful_task_ids", [])
        if isinstance(required_task_ids, str):
            required_task_ids = [required_task_ids]
        if not isinstance(required_task_ids, (list, tuple, set)):
            raise ValueError(
                "sequential_preconditions.requires_successful_task_ids must "
                "be a string or list of task IDs."
            )
        missing = []
        for task_id in required_task_ids:
            task_id = str(task_id).strip()
            report = reports_by_task.get(task_id)
            if report is None:
                missing.append(f"required stage {task_id!r} has no report")
            elif report.get("task_success") is not True:
                missing.append(
                    f"required stage {task_id!r} was not independently "
                    "successful"
                )

        landmark_task_id = configured.get("requires_landmark_task_id")
        if landmark_task_id not in (None, ""):
            landmark_task_id = str(landmark_task_id).strip()
            has_landmark = bool(
                hasattr(memory, "has_task_landmark")
                and memory.has_task_landmark(task_id=landmark_task_id)
            )
            if not has_landmark:
                missing.append(
                    f"no validated landmark is available from stage "
                    f"{landmark_task_id!r}"
                )
        if not missing:
            return None
        return {
            "requires_successful_task_ids": [
                str(task_id) for task_id in required_task_ids
            ],
            "requires_landmark_task_id": landmark_task_id,
            "unmet_requirements": missing,
        }

    @staticmethod
    def _apply_task_start_pose(agent, task, metadata):
        pose = task.get("start_pose")
        if pose in (None, "", {}):
            return None
        if not isinstance(pose, dict) or "x" not in pose or "z" not in pose:
            raise ValueError("task.start_pose must be an object with x and z.")
        if not hasattr(agent, "set_pose"):
            raise ValueError("task.start_pose is set but the agent cannot set_pose.")
        before = (
            agent.get_state_snapshot()
            if hasattr(agent, "get_state_snapshot")
            else None
        )
        agent.set_pose(
            {"x": pose["x"], "y": pose.get("y", 1.6), "z": pose["z"]},
            pose.get("yaw", 0.0),
        )
        applied = {
            "x": float(pose["x"]),
            "z": float(pose["z"]),
            "yaw": float(pose.get("yaw", 0.0)),
            "pose_before_reset": before,
        }
        metadata["start_pose_reset"] = applied
        return applied

    @staticmethod
    def _precondition_failed_result(
        condition,
        task,
        episode_metadata,
        failure,
    ):
        contract = TaskContract.build(
            goal=task.get("goal", ""),
            configured=task.get("task_spec"),
        )
        detail = "; ".join(failure.get("unmet_requirements", []))
        return {
            "goal": task.get("goal"),
            "task_contract": contract.as_dict(),
            "status": "precondition_failed",
            "reason": (
                "Dependent sequential stage was not run because its "
                f"precondition failed: {detail}."
            ),
            "precondition_failure": failure,
            "model": condition.model_name,
            "baseline": condition.baseline,
            "perceptual_interface": condition.perceptual_interface,
            "memory_mode": condition.memory_mode,
            "actions_taken": 0,
            "duration_seconds": 0.0,
            "hard_timeout": False,
            "task_success": None,
            "agent_declared_success": False,
            "controller_assisted_completion": False,
            "requires_manual_review": False,
            "perception_parser_applicable": (
                condition.perceptual_interface != "direct_visual"
            ),
            "ground_truth_evaluation": {
                "available": False,
                "success": None,
                "reason": "Skipped because a sequential precondition failed.",
            },
            "episode_metadata": dict(episode_metadata),
            "inference": {},
            "history": [],
        }

    def _sequence_task_metadata(
        self,
        episode_id,
        index,
        tasks,
        task,
        seed,
        scene_metadata,
    ):
        return {
            "episode_id": episode_id,
            "sequence_index": index,
            "sequence_length": len(tasks),
            "sequence_task_ids": [
                item.get("id", f"task_{task_index}")
                for task_index, item in enumerate(tasks)
            ],
            "task_id": task.get("id", f"task_{index}"),
            "scene_seed": seed,
            "sequential_extension": True,
            **scene_metadata,
        }

    def _run_sequential_extension(self, model_name, backend, condition=None):
        if not self._sequential_extension_enabled():
            return
        extension, tasks, first_episode, seed, condition = (
            self._sequential_extension_details(model_name, condition)
        )
        scene = first_episode["scene_config"]
        episode_id = self._episode_id(condition, "sequence", seed)
        directory = self.output_directory / episode_id
        controller = None
        reports_by_task = {}
        try:
            controller, agent, scene_metadata = self.episode_factory(
                scene_config=scene,
                seed=seed,
                episode_directory=directory,
            )
            planner = self._create_planner(condition, backend, directory)
            for index, task in enumerate(tasks):
                task_id = task.get("id", f"task_{index}")
                metadata = self._sequence_task_metadata(
                    episode_id,
                    index,
                    tasks,
                    task,
                    seed,
                    scene_metadata,
                )
                failure = self._sequence_precondition_failure(
                    task,
                    reports_by_task,
                    planner.memory,
                )
                if failure is not None:
                    report = self._precondition_failed_result(
                        condition,
                        task,
                        metadata,
                        failure,
                    )
                else:
                    max_steps = int(
                        task.get(
                            "max_steps",
                            extension.get(
                                "max_steps", self.config.get("max_steps", 20)
                            ),
                        )
                    )
                    self._apply_task_start_pose(agent, task, metadata)
                    report = planner.run_task(
                        agent=agent,
                        goal=task["goal"],
                        max_steps=max_steps,
                        reset_memory=index == 0,
                        episode_metadata=metadata,
                        success_evaluator=(
                            lambda history=None, task=task: agent.evaluate_success(
                                task.get("success_criteria"),
                                history=history,
                            )
                        ),
                        task_spec=task.get("task_spec"),
                    )
                self.results.append(report)
                reports_by_task[str(task_id)] = report
                planner.memory.record_task_boundary(
                    task_id=task_id,
                    goal=task["goal"],
                    task_contract=report.get("task_contract"),
                    status=report.get("status"),
                    task_success=report.get("task_success"),
                    agent_state=(
                        agent.get_state_snapshot()
                        if hasattr(agent, "get_state_snapshot")
                        else None
                    ),
                )
        finally:
            self._terminate_controller(controller)
            self._cleanup_build_processes()

    def _run_sequential_extension_isolated(self, model_name, condition=None):
        if not self._sequential_extension_enabled():
            return
        extension, tasks, first_episode, seed, condition = (
            self._sequential_extension_details(model_name, condition)
        )
        episode_id = self._episode_id(condition, "sequence", seed)
        directory = self.output_directory / episode_id
        outcome = self.isolated_sequence_executor(
            {
                "config": self.config,
                "model_name": model_name,
                "condition": {
                    "model_name": condition.model_name,
                    "perceptual_interface": condition.perceptual_interface,
                    "memory_mode": condition.memory_mode,
                    "baseline": condition.baseline,
                },
                "scene_config": first_episode["scene_config"],
                "seed": seed,
                "history_directory": str(directory),
                "tasks": tasks,
                "extension": extension,
                "episode_id": episode_id,
            }
        )
        self._cleanup_build_processes()
        reports = outcome.get("reports", [])
        if not isinstance(reports, list):
            reports = []
        self.results.extend(
            report for report in reports if isinstance(report, dict)
        )
        if outcome.get("outcome") == "completed":
            if len(reports) != len(tasks):
                raise RuntimeError(
                    "Sequential watchdog completed without a report for every stage."
                )
            return

        active_index = int(outcome.get("active_task_index", len(reports)))
        active_index = min(max(active_index, 0), len(tasks) - 1)
        active_task = tasks[active_index]
        metadata = self._sequence_task_metadata(
            episode_id,
            active_index,
            tasks,
            active_task,
            seed,
            {},
        )
        if outcome.get("outcome") == "timeout":
            report = self._timeout_result(
                condition=condition,
                task=active_task,
                seed=seed,
                episode_id=episode_id,
                scene_episode_id=first_episode["id"],
                timeout_seconds=outcome.get("timeout_seconds"),
                duration_seconds=outcome.get("duration_seconds"),
                checkpoint_path=outcome.get("checkpoint_path"),
                partial_history_path=outcome.get("partial_history_path"),
                worker_exitcode=outcome.get("worker_exitcode"),
            )
            report["episode_metadata"].update(metadata)
            self.results.append(report)
            return

        error = RuntimeError(
            f"{outcome.get('error_type', 'SequenceWorkerError')}: "
            f"{outcome.get('message', 'watchdog worker failed without a report.')}"
        )
        report = self._error_result(
            condition,
            active_task,
            seed,
            error,
            episode_id,
            scene_episode_id=first_episode["id"],
        )
        report["episode_metadata"].update(metadata)
        report["duration_seconds"] = outcome.get("duration_seconds", 0)
        report["watchdog_checkpoint_path"] = outcome.get("checkpoint_path")
        report["partial_history_path"] = outcome.get("partial_history_path")
        report["watchdog_exitcode"] = outcome.get("worker_exitcode")
        self.results.append(report)

    def _record_sequential_errors(self, model_name, error, condition=None):
        if not self._sequential_extension_enabled():
            return
        _, tasks, first_episode, seed, condition = self._sequential_extension_details(
            model_name, condition
        )
        episode_id = self._episode_id(condition, "sequence", seed)
        for index, task in enumerate(tasks):
            report = self._error_result(
                condition,
                task,
                seed,
                error,
                episode_id,
                scene_episode_id=first_episode["id"],
            )
            report["episode_metadata"].update(
                self._sequence_task_metadata(
                    episode_id,
                    index,
                    tasks,
                    task,
                    seed,
                    {},
                )
            )
            self.results.append(report)

    @staticmethod
    def _terminate_controller(controller):
        if controller is None:
            return
        try:
            controller.communicate({"$type": "terminate"})
        except Exception as error:
            print(f"Unity termination warning: {error}")
        finally:
            socket = getattr(controller, "socket", None)
            if socket is None:
                return
            context = getattr(socket, "context", None)
            try:
                socket.close(linger=0)
            except TypeError:
                try:
                    socket.close(0)
                except Exception:
                    pass
            except Exception:
                pass
            if context is not None:
                try:
                    context.destroy(linger=0)
                except Exception:
                    pass

    def _cleanup_build_processes(self):
        if self.episode_cleanup is None:
            return
        try:
            self.episode_cleanup()
        except Exception as error:
            print(f"TDW process cleanup warning: {error}")

    @staticmethod
    def _episode_id(condition, task_id, episode_label):
        safe_model = condition.model_name.replace("/", "_")
        safe_episode = "".join(
            character if character.isalnum() or character in {"-", "_"} else "_"
            for character in str(episode_label)
        )
        return (
            f"{safe_model}__{condition.baseline}__"
            f"{condition.perceptual_interface}__{condition.memory_mode}__"
            f"{task_id}__{safe_episode}"
        )

    @staticmethod
    def _error_result(
        condition,
        task,
        seed,
        error,
        episode_id,
        scene_episode_id=None,
    ):
        return {
            "goal": task.get("goal"),
            "status": "error",
            "reason": f"{type(error).__name__}: {error}",
            "model": condition.model_name,
            "baseline": condition.baseline,
            "perceptual_interface": condition.perceptual_interface,
            "memory_mode": condition.memory_mode,
            "actions_taken": 0,
            "duration_seconds": 0,
            "hard_timeout": False,
            "perception_parser_applicable": (
                condition.perceptual_interface != "direct_visual"
            ),
            "episode_metadata": {
                "episode_id": episode_id,
                "task_id": task.get("id"),
                "scene_episode_id": scene_episode_id,
                "scene_seed": seed,
            },
        }

    @staticmethod
    def _timeout_result(
        condition,
        task,
        seed,
        episode_id,
        scene_episode_id,
        timeout_seconds,
        duration_seconds,
        checkpoint_path=None,
        partial_history_path=None,
        worker_exitcode=None,
    ):
        limit = float(timeout_seconds or 0)
        elapsed = float(duration_seconds or limit)
        return {
            "goal": task.get("goal"),
            "status": "timeout",
            "reason": (
                "Episode exceeded the hard watchdog limit of "
                f"{limit:g} seconds and its worker process was terminated."
            ),
            "model": condition.model_name,
            "baseline": condition.baseline,
            "perceptual_interface": condition.perceptual_interface,
            "memory_mode": condition.memory_mode,
            "actions_taken": None,
            "duration_seconds": round(elapsed, 5),
            "hard_timeout": True,
            "timeout_seconds": limit,
            "task_success": None,
            "agent_declared_success": False,
            "controller_assisted_completion": False,
            "requires_manual_review": False,
            "perception_parser_applicable": (
                condition.perceptual_interface != "direct_visual"
            ),
            "watchdog_checkpoint_path": checkpoint_path,
            "partial_history_path": partial_history_path,
            "watchdog_exitcode": worker_exitcode,
            "episode_metadata": {
                "episode_id": episode_id,
                "task_id": task.get("id"),
                "scene_episode_id": scene_episode_id,
                "scene_seed": seed,
            },
        }

    def _save_summary(self):
        self._apply_manual_review_labels()
        self._write_json(self.output_directory / "all_results.json", self.results)
        fields = [
            "episode_id",
            "model",
            "baseline",
            "perceptual_interface",
            "memory_mode",
            "turn_angle_degrees",
            "goal",
            "status",
            "task_success",
            "agent_declared_success",
            "controller_assisted_completion",
            "actions_taken",
            "duration_seconds",
            "hard_timeout",
            "timeout_seconds",
            "watchdog_exitcode",
            "partial_history_path",
            "blocked_or_failed_actions",
            "invalid_model_outputs",
            "perception_parser_applicable",
            "perception_parse_failures",
            "guarded_action_replacements",
            "fallback_actions",
            "rejected_completions",
            "rejected_impossible_claims",
            "completion_inconsistencies",
            "completion_retry_calls",
            "completion_retries_resolved",
            "completion_promotions",
            "completion_resolution",
            "completion_policy",
            "simplified_rules",
            "controller_priority",
            "completion_claims",
            "completion_claims_rejected",
            "completion_claims_rejected_by_environment",
            "rejected_claim_rate",
            "target_approach_steps",
            "target_approach_replacements",
            "target_approach_persisted_steps",
            "target_overshoot_corrections",
            "target_detours",
            "impossible_status_ignored",
            "initial_scan_turns",
            "search_turn_oscillation_replacements",
            "target_loss_recoveries",
            "recent_sighting_reversals_allowed",
            "target_side_flip_rejections",
            "unlocated_sighting_approaches",
            "memory_guided_search_holds",
            "status_normalizations",
            "evidence_text_claims",
            "evidence_text_claims_accepted",
            "memory_destination_abandonments",
            "memory_destination_suspensions",
            "memory_destination_blocked_approaches",
            "distinct_cells_visited",
            "repeated_pose_heading_steps",
            "net_displacement_m",
            "memory_guided_steps_to_landmark",
            "memory_guided_steps_to_sighting",
            "target_reacquisition_calls",
            "target_reacquisition_confirmations",
            "evidence_conflict_checks",
            "evidence_conflict_confirmations",
            "object_id_visible_at_stop",
            "object_id_visible_pixels_at_stop",
            "first_object_id_visible_step",
            "object_id_visible_frames",
            "object_id_visible_frames_confirmed",
            "object_id_visible_frames_missed",
            "actions_before_first_object_id_sighting",
            "path_length_before_first_object_id_sighting",
            "environment_success_ever",
            "first_environment_success_step",
            "goal_reached_without_stop",
            "requires_manual_review",
            "immediately_repeated_actions",
            "inference_calls",
            "input_tokens",
            "output_tokens",
            "inference_seconds",
            "preprocess_seconds",
            "time_to_first_token_seconds",
            "decode_seconds",
            "max_memory_context_characters",
            "memory_context_budget_characters",
            "generation_timeouts",
            "generation_soft_limit_exceeded",
            "min_cuda_free_memory_mb",
            "max_cuda_memory_allocated_mb",
            "max_cuda_memory_reserved_mb",
            "max_cuda_memory_allocated_mb_after_cleanup",
            "max_cuda_memory_reserved_mb_after_cleanup",
            "reason",
        ]
        with (self.output_directory / "summary.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for result in self.results:
                row = {key: result.get(key) for key in fields}
                row["episode_id"] = result.get("episode_metadata", {}).get(
                    "episode_id"
                )
                inference = result.get("inference", {})
                row.update(
                    {
                        "inference_calls": inference.get("calls"),
                        "input_tokens": inference.get("input_tokens"),
                        "output_tokens": inference.get("output_tokens"),
                        "inference_seconds": inference.get("inference_seconds"),
                        "preprocess_seconds": inference.get("preprocess_seconds"),
                        "time_to_first_token_seconds": inference.get(
                            "time_to_first_token_seconds"
                        ),
                        "decode_seconds": inference.get("decode_seconds"),
                        "max_memory_context_characters": inference.get(
                            "max_memory_context_characters"
                        ),
                        "memory_context_budget_characters": inference.get(
                            "memory_context_budget_characters"
                        ),
                        "generation_timeouts": inference.get(
                            "generation_timeouts"
                        ),
                        "generation_soft_limit_exceeded": inference.get(
                            "generation_soft_limit_exceeded"
                        ),
                        "min_cuda_free_memory_mb": inference.get(
                            "min_cuda_free_memory_mb"
                        ),
                        "max_cuda_memory_allocated_mb": inference.get(
                            "max_cuda_memory_allocated_mb"
                        ),
                        "max_cuda_memory_reserved_mb": inference.get(
                            "max_cuda_memory_reserved_mb"
                        ),
                        "max_cuda_memory_allocated_mb_after_cleanup": inference.get(
                            "max_cuda_memory_allocated_mb_after_cleanup"
                        ),
                        "max_cuda_memory_reserved_mb_after_cleanup": inference.get(
                            "max_cuda_memory_reserved_mb_after_cleanup"
                        ),
                    }
                )
                writer.writerow(row)

        self._write_manual_review_sheet()

    def _manual_review_rows(self):
        rows = []
        mapping = []
        image_directory = self.output_directory / "manual_review_images"
        image_directory.mkdir(parents=True, exist_ok=True)
        for result in self.results:
            if not result.get("requires_manual_review"):
                continue
            if result.get("status") != "completed":
                continue
            evaluation = result.get("ground_truth_evaluation") or {}
            episode_id = result.get("episode_metadata", {}).get("episode_id")
            if not episode_id:
                continue
            review_id = self._review_id(episode_id)
            source_image = result.get("final_image_path")
            extension = Path(source_image).suffix if source_image else ".png"
            review_image = image_directory / f"{review_id}{extension or '.png'}"
            if source_image and Path(source_image).exists() and not review_image.exists():
                shutil.copy2(source_image, review_image)
            mapping.append(
                {
                    "review_id": review_id,
                    "episode_id": episode_id,
                }
            )
            rows.append(
                {
                    "review_id": review_id,
                    "image_path": str(
                        Path("manual_review_images") / review_image.name
                    ),
                    "rubric": evaluation.get("rubric"),
                    "manual_target_visible": "",
                    "reviewer_notes": "",
                }
            )
        mapping.sort(key=lambda item: item["review_id"])
        self._write_json(
            self.output_directory / "manual_review_mapping.json", mapping
        )
        rows.sort(key=lambda item: item["review_id"])
        return rows

    def _review_id(self, episode_id):
        salt = self.config.get("condition_order_seed", 2026)
        digest = hashlib.sha256(
            f"{salt}:{episode_id}".encode("utf-8")
        ).hexdigest()[:12]
        return f"review_{digest}"

    def _read_manual_review_labels(self):
        path = self.output_directory / "manual_review.csv"
        if not path.exists():
            return {}
        try:
            with path.open("r", newline="", encoding="utf-8") as file:
                return {
                    row.get("review_id"): row
                    for row in csv.DictReader(file)
                    if row.get("review_id")
                }
        except OSError:
            return {}

    def _apply_manual_review_labels(self):
        labels = self._read_manual_review_labels()
        for result in self.results:
            episode_id = result.get("episode_metadata", {}).get("episode_id")
            review_id = self._review_id(episode_id) if episode_id else None
            raw = str(
                labels.get(review_id, {}).get("manual_target_visible", "")
            ).strip().lower()
            if raw not in {"true", "false", "yes", "no", "1", "0"}:
                continue
            target_visible = raw in {"true", "yes", "1"}
            result["manual_target_visible"] = target_visible
            result["task_success"] = bool(
                result.get("agent_declared_success") and target_visible
            )

    def _write_manual_review_sheet(self):
        rows = self._manual_review_rows()
        if not rows:
            return
        existing = self._read_manual_review_labels()
        for row in rows:
            previous = existing.get(row["review_id"], {})
            row["manual_target_visible"] = previous.get(
                "manual_target_visible", ""
            )
            row["reviewer_notes"] = previous.get("reviewer_notes", "")
        fields = list(rows[0])
        with (self.output_directory / "manual_review.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _write_json(path, data):
        with Path(path).open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, ensure_ascii=False)
