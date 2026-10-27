import json
import inspect
import multiprocessing
import os
from pathlib import Path
import time
import traceback


def _atomic_write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False, default=str)
    os.replace(temporary, path)


def _read_json(path):
    try:
        with Path(path).open("r", encoding="utf-8") as file:
            value = json.load(file)
        return value if isinstance(value, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _read_json_list(path):
    try:
        with Path(path).open("r", encoding="utf-8") as file:
            value = json.load(file)
        return value if isinstance(value, list) else []
    except (OSError, json.JSONDecodeError):
        return []


class EpisodeWatchdog:
    def __init__(
        self,
        build_path,
        timeout_seconds,
        termination_grace_seconds=15,
    ):
        self.build_path = str(build_path)
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.termination_grace_seconds = max(
            0.0,
            float(termination_grace_seconds),
        )

    def __call__(self, payload):
        episode_directory = Path(payload["history_directory"]).resolve()
        episode_directory.mkdir(parents=True, exist_ok=True)
        result_path = episode_directory / "watchdog_result.json"
        checkpoint_path = episode_directory / "watchdog_checkpoint.json"
        partial_history_path = episode_directory / "partial_history.json"
        worker_payload = {
            **payload,
            "history_directory": str(episode_directory),
            "build_path": self.build_path,
            "result_path": str(result_path),
            "checkpoint_path": str(checkpoint_path),
            "partial_history_path": str(partial_history_path),
        }
        _atomic_write_json(
            checkpoint_path,
            {
                "phase": "queued",
                "timestamp": time.time(),
                "hard_timeout_seconds": self.timeout_seconds,
            },
        )

        context = multiprocessing.get_context("spawn")
        process = context.Process(
            target=run_episode_worker,
            args=(worker_payload,),
            name="tdw_episode_worker",
        )
        started = time.perf_counter()
        try:
            process.start()
        except Exception as error:
            return {
                "outcome": "error",
                "error_type": type(error).__name__,
                "message": str(error),
                "duration_seconds": round(time.perf_counter() - started, 5),
                "checkpoint_path": str(checkpoint_path),
                "partial_history_path": (
                    str(partial_history_path)
                    if partial_history_path.exists()
                    else None
                ),
                "worker_exitcode": None,
            }

        process.join(self.timeout_seconds)
        elapsed = time.perf_counter() - started
        if process.is_alive():
            process.terminate()
            process.join(self.termination_grace_seconds)
            if process.is_alive() and hasattr(process, "kill"):
                process.kill()
                process.join(self.termination_grace_seconds)
            elapsed = time.perf_counter() - started
            return {
                "outcome": "timeout",
                "timeout_seconds": self.timeout_seconds,
                "duration_seconds": round(elapsed, 5),
                "checkpoint_path": str(checkpoint_path),
                "partial_history_path": (
                    str(partial_history_path)
                    if partial_history_path.exists()
                    else None
                ),
                "worker_exitcode": process.exitcode,
            }

        worker_result = _read_json(result_path)
        if worker_result is None:
            return {
                "outcome": "error",
                "error_type": "WorkerResultMissing",
                "message": (
                    "The episode worker exited without writing its result "
                    f"(exit code {process.exitcode})."
                ),
                "duration_seconds": round(elapsed, 5),
                "checkpoint_path": str(checkpoint_path),
                "partial_history_path": (
                    str(partial_history_path)
                    if partial_history_path.exists()
                    else None
                ),
                "worker_exitcode": process.exitcode,
            }
        worker_result.setdefault("duration_seconds", round(elapsed, 5))
        worker_result.setdefault("checkpoint_path", str(checkpoint_path))
        worker_result.setdefault(
            "partial_history_path",
            str(partial_history_path) if partial_history_path.exists() else None,
        )
        worker_result["worker_exitcode"] = process.exitcode
        return worker_result


class SequenceWatchdog:
    def __init__(
        self,
        build_path,
        timeout_seconds,
        termination_grace_seconds=15,
    ):
        self.build_path = str(build_path)
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.termination_grace_seconds = max(
            0.0,
            float(termination_grace_seconds),
        )

    def __call__(self, payload):
        directory = Path(payload["history_directory"]).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        result_path = directory / "sequence_watchdog_result.json"
        checkpoint_path = directory / "sequence_watchdog_checkpoint.json"
        partial_history_path = directory / "sequence_partial_history.json"
        reports_path = directory / "sequence_completed_reports.json"
        worker_payload = {
            **payload,
            "history_directory": str(directory),
            "build_path": self.build_path,
            "result_path": str(result_path),
            "checkpoint_path": str(checkpoint_path),
            "partial_history_path": str(partial_history_path),
            "reports_path": str(reports_path),
        }
        _atomic_write_json(
            checkpoint_path,
            {
                "phase": "queued",
                "timestamp": time.time(),
                "hard_timeout_seconds": self.timeout_seconds,
                "active_task_index": 0,
            },
        )

        context = multiprocessing.get_context("spawn")
        process = context.Process(
            target=run_sequence_worker,
            args=(worker_payload,),
            name="tdw_sequence_worker",
        )
        started = time.perf_counter()
        try:
            process.start()
        except Exception as error:
            return {
                "outcome": "error",
                "error_type": type(error).__name__,
                "message": str(error),
                "reports": _read_json_list(reports_path),
                "active_task_index": 0,
                "duration_seconds": round(time.perf_counter() - started, 5),
                "checkpoint_path": str(checkpoint_path),
                "partial_history_path": None,
                "worker_exitcode": None,
            }

        process.join(self.timeout_seconds)
        elapsed = time.perf_counter() - started
        if process.is_alive():
            process.terminate()
            process.join(self.termination_grace_seconds)
            if process.is_alive() and hasattr(process, "kill"):
                process.kill()
                process.join(self.termination_grace_seconds)
            checkpoint = _read_json(checkpoint_path) or {}
            return {
                "outcome": "timeout",
                "timeout_seconds": self.timeout_seconds,
                "duration_seconds": round(
                    time.perf_counter() - started,
                    5,
                ),
                "reports": _read_json_list(reports_path),
                "active_task_index": int(
                    checkpoint.get("active_task_index", 0) or 0
                ),
                "checkpoint_path": str(checkpoint_path),
                "partial_history_path": (
                    str(partial_history_path)
                    if partial_history_path.exists()
                    else None
                ),
                "worker_exitcode": process.exitcode,
            }

        worker_result = _read_json(result_path)
        checkpoint = _read_json(checkpoint_path) or {}
        if worker_result is None:
            return {
                "outcome": "error",
                "error_type": "WorkerResultMissing",
                "message": (
                    "The sequence worker exited without writing its result "
                    f"(exit code {process.exitcode})."
                ),
                "reports": _read_json_list(reports_path),
                "active_task_index": int(
                    checkpoint.get("active_task_index", 0) or 0
                ),
                "duration_seconds": round(elapsed, 5),
                "checkpoint_path": str(checkpoint_path),
                "partial_history_path": (
                    str(partial_history_path)
                    if partial_history_path.exists()
                    else None
                ),
                "worker_exitcode": process.exitcode,
            }
        worker_result.setdefault("reports", _read_json_list(reports_path))
        worker_result.setdefault(
            "active_task_index",
            int(checkpoint.get("active_task_index", 0) or 0),
        )
        worker_result.setdefault("duration_seconds", round(elapsed, 5))
        worker_result.setdefault("checkpoint_path", str(checkpoint_path))
        worker_result.setdefault(
            "partial_history_path",
            str(partial_history_path) if partial_history_path.exists() else None,
        )
        worker_result["worker_exitcode"] = process.exitcode
        return worker_result


def _model_config(config, model_name):
    for item in config.get("models", []):
        if isinstance(item, dict) and item.get("name") == model_name:
            return item
    return {}


def _create_backend(config, model_name):
    from vision import VisionLanguageModel

    model_config = _model_config(config, model_name)
    return VisionLanguageModel(
        model_name=model_name,
        use_4bit=model_config.get("use_4bit", config.get("use_4bit", True)),
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
            config.get("four_bit_compute_dtype", "auto"),
        ),
        log_cuda_memory=bool(
            model_config.get(
                "log_cuda_memory",
                config.get("log_cuda_memory", True),
            )
        ),
        cuda_allocator_config=model_config.get(
            "cuda_allocator_config",
            config.get("cuda_allocator_config"),
        ),
    )


def _learned_store(config):
    extension = config.get("extensions", {}).get("learned_memory", {})
    if not extension.get("enabled", False):
        return None
    from memory import EmbeddingExperienceStore

    return EmbeddingExperienceStore(
        directory=extension.get("directory", "learned_memory"),
        model_name=extension.get(
            "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        frozen=extension.get("frozen", True),
    )


def _create_planner(config, condition, backend, history_directory):
    from memory import MemoryManager
    from planner import CoELABaselinePlanner, Planner

    memory_context = config.get("memory_context", {})
    if not isinstance(memory_context, dict):
        raise ValueError("memory_context must be an object when configured.")
    memory = MemoryManager(
        episodic_window=int(config.get("episodic_window", 8)),
        learned_store=_learned_store(config),
        context_episodic_limit=int(memory_context.get("episodic_events", 4)),
        context_object_limit=int(memory_context.get("semantic_objects", 12)),
        context_pose_limit=int(memory_context.get("visited_poses", 8)),
        context_fact_limit=int(memory_context.get("semantic_facts", 12)),
        context_text_char_limit=int(memory_context.get("text_characters", 1000)),
        context_prompt_char_limit=int(
            memory_context.get("prompt_characters", 3000)
        ),
    )
    common = {
        "model_name": condition["model_name"],
        "backend": backend,
        "memory_manager": memory,
        "history_directory": history_directory,
        "perception_max_new_tokens": int(
            config.get("perception_max_new_tokens", 300)
        ),
        "perception_retry_max_new_tokens": int(
            config.get("perception_retry_max_new_tokens", 224)
        ),
        "decision_max_new_tokens": int(
            config.get("decision_max_new_tokens", 300)
        ),
        "completion_max_new_tokens": int(
            config.get("completion_max_new_tokens", 64)
        ),
        "save_step_images": bool(config.get("save_step_images", False)),
    }
    if condition.get("baseline") == "coela_style_single_agent":
        return CoELABaselinePlanner(**common)
    return Planner(
        perceptual_interface=condition["perceptual_interface"],
        memory_mode=condition["memory_mode"],
        baseline_name=condition.get("baseline", "primary"),
        memory_guided_search=bool(config.get("memory_guided_search", True)),
        **common,
    )


def run_episode_worker(payload):
    result_path = Path(payload["result_path"])
    checkpoint_path = Path(payload["checkpoint_path"])
    partial_history_path = Path(payload["partial_history_path"])
    history_directory = Path(payload["history_directory"])
    started = time.perf_counter()
    controller = None
    backend = None
    shutdown_controller = None
    close_tdw_build_processes = None

    def checkpoint(phase, **details):
        _atomic_write_json(
            checkpoint_path,
            {
                "phase": phase,
                "timestamp": time.time(),
                "elapsed_seconds": round(time.perf_counter() - started, 5),
                **details,
            },
        )

    def save_partial_history(history):
        _atomic_write_json(
            partial_history_path,
            {
                "history": history,
                "last_completed_step": history[-1].get("step") if history else 0,
                "updated_at": time.time(),
            },
        )
        checkpoint(
            "running_task",
            last_completed_step=history[-1].get("step") if history else 0,
        )

    try:
        from main import (
            close_tdw_build_processes as close_tdw_build_processes_function,
            create_scene,
            shutdown_controller as shutdown_controller_function,
        )

        shutdown_controller = shutdown_controller_function
        close_tdw_build_processes = close_tdw_build_processes_function
        config = payload["config"]
        condition = payload["condition"]
        task = payload["task"]
        history_directory.mkdir(parents=True, exist_ok=True)

        checkpoint("creating_scene")
        controller, agent, scene_metadata = create_scene(
            build_path=payload["build_path"],
            scene_config=payload["scene_config"],
            seed=int(payload["seed"]),
            episode_directory=history_directory,
        )

        checkpoint("loading_model")
        backend = _create_backend(config, condition["model_name"])
        backend.load()
        planner = _create_planner(
            config,
            condition,
            backend,
            history_directory,
        )
        checkpoint("running_task")
        run_task_options = {
            "agent": agent,
            "goal": task["goal"],
            "max_steps": int(task.get("max_steps", config.get("max_steps", 20))),
            "episode_metadata": {
                **payload.get("episode_metadata", {}),
                **scene_metadata,
            },
            "success_evaluator": lambda history=None: agent.evaluate_success(
                task.get("success_criteria"),
                history=history,
            ),
            "task_spec": task.get("task_spec"),
        }
        parameters = inspect.signature(planner.run_task).parameters.values()
        if any(
            parameter.name == "progress_callback"
            or parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in parameters
        ):
            run_task_options["progress_callback"] = save_partial_history
        report = planner.run_task(**run_task_options)
        checkpoint("completed", status=report.get("status"))
        _atomic_write_json(
            result_path,
            {
                "outcome": "completed",
                "report": report,
                "duration_seconds": round(time.perf_counter() - started, 5),
            },
        )
    except Exception as error:
        checkpoint(
            "error",
            error_type=type(error).__name__,
            message=str(error),
        )
        _atomic_write_json(
            result_path,
            {
                "outcome": "error",
                "error_type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
                "duration_seconds": round(time.perf_counter() - started, 5),
            },
        )
    finally:
        if backend is not None:
            try:
                backend.unload()
            except Exception:
                pass
        if shutdown_controller is not None:
            try:
                shutdown_controller(controller)
            except Exception:
                pass
        if close_tdw_build_processes is not None:
            try:
                close_tdw_build_processes(payload["build_path"])
            except Exception:
                pass


def run_sequence_worker(payload):
    result_path = Path(payload["result_path"])
    checkpoint_path = Path(payload["checkpoint_path"])
    partial_history_path = Path(payload["partial_history_path"])
    reports_path = Path(payload["reports_path"])
    history_directory = Path(payload["history_directory"])
    started = time.perf_counter()
    controller = None
    backend = None
    shutdown_controller = None
    close_tdw_build_processes = None
    active_task_index = 0
    reports = []
    reports_by_task = {}

    def checkpoint(phase, **details):
        _atomic_write_json(
            checkpoint_path,
            {
                "phase": phase,
                "timestamp": time.time(),
                "elapsed_seconds": round(time.perf_counter() - started, 5),
                "active_task_index": active_task_index,
                **details,
            },
        )

    def persist_reports():
        _atomic_write_json(reports_path, reports)

    def save_partial_history(history, task_id, sequence_index):
        _atomic_write_json(
            partial_history_path,
            {
                "task_id": task_id,
                "sequence_index": sequence_index,
                "history": history,
                "last_completed_step": (
                    history[-1].get("step") if history else 0
                ),
                "updated_at": time.time(),
            },
        )
        checkpoint(
            "running_task",
            task_id=task_id,
            sequence_index=sequence_index,
            last_completed_step=history[-1].get("step") if history else 0,
        )

    try:
        from experiment import Condition, ExperimentRunner
        from main import (
            close_tdw_build_processes as close_tdw_build_processes_function,
            create_scene,
            shutdown_controller as shutdown_controller_function,
        )

        shutdown_controller = shutdown_controller_function
        close_tdw_build_processes = close_tdw_build_processes_function
        config = payload["config"]
        condition_data = payload["condition"]
        extension = payload.get("extension", {})
        tasks = payload.get("tasks", [])
        if not isinstance(tasks, list) or len(tasks) < 2:
            raise ValueError("Sequential watchdog requires at least two tasks.")
        history_directory.mkdir(parents=True, exist_ok=True)
        condition = Condition(
            condition_data["model_name"],
            condition_data["perceptual_interface"],
            condition_data["memory_mode"],
            baseline=condition_data.get("baseline", "sequential_extension"),
        )
        task_ids = [
            item.get("id", f"task_{index}")
            for index, item in enumerate(tasks)
        ]

        checkpoint("creating_scene")
        controller, agent, scene_metadata = create_scene(
            build_path=payload["build_path"],
            scene_config=payload["scene_config"],
            seed=int(payload["seed"]),
            episode_directory=history_directory,
        )
        checkpoint("loading_model")
        backend = _create_backend(config, condition.model_name)
        backend.load()
        planner = _create_planner(
            config,
            condition_data,
            backend,
            history_directory,
        )

        for index, task in enumerate(tasks):
            active_task_index = index
            task_id = task.get("id", f"task_{index}")
            metadata = {
                "episode_id": payload["episode_id"],
                "sequence_index": index,
                "sequence_length": len(tasks),
                "sequence_task_ids": task_ids,
                "task_id": task_id,
                "scene_seed": int(payload["seed"]),
                "sequential_extension": True,
                **scene_metadata,
            }
            checkpoint("before_task", task_id=task_id, sequence_index=index)
            failure = ExperimentRunner._sequence_precondition_failure(
                task,
                reports_by_task,
                planner.memory,
            )
            if failure is not None:
                report = ExperimentRunner._precondition_failed_result(
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
                            "max_steps", config.get("max_steps", 20)
                        ),
                    )
                )
                run_task_options = {
                    "agent": agent,
                    "goal": task["goal"],
                    "max_steps": max_steps,
                    "reset_memory": index == 0,
                    "episode_metadata": metadata,
                    "success_evaluator": (
                        lambda history=None, task=task: agent.evaluate_success(
                            task.get("success_criteria"),
                            history=history,
                        )
                    ),
                    "task_spec": task.get("task_spec"),
                }
                parameters = inspect.signature(planner.run_task).parameters.values()
                if any(
                    parameter.name == "progress_callback"
                    or parameter.kind == inspect.Parameter.VAR_KEYWORD
                    for parameter in parameters
                ):
                    run_task_options["progress_callback"] = (
                        lambda history, task_id=task_id, sequence_index=index:
                        save_partial_history(history, task_id, sequence_index)
                    )
                ExperimentRunner._apply_task_start_pose(agent, task, metadata)
                checkpoint("running_task", task_id=task_id, sequence_index=index)
                report = planner.run_task(**run_task_options)

            reports.append(report)
            reports_by_task[str(task_id)] = report
            persist_reports()
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
            active_task_index = index + 1
            checkpoint(
                "task_completed",
                task_id=task_id,
                sequence_index=index,
                status=report.get("status"),
                task_success=report.get("task_success"),
            )

        checkpoint("completed", completed_task_count=len(reports))
        _atomic_write_json(
            result_path,
            {
                "outcome": "completed",
                "reports": reports,
                "active_task_index": active_task_index,
                "duration_seconds": round(time.perf_counter() - started, 5),
            },
        )
    except Exception as error:
        checkpoint(
            "error",
            error_type=type(error).__name__,
            message=str(error),
        )
        persist_reports()
        _atomic_write_json(
            result_path,
            {
                "outcome": "error",
                "error_type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
                "reports": reports,
                "active_task_index": active_task_index,
                "duration_seconds": round(time.perf_counter() - started, 5),
            },
        )
    finally:
        if backend is not None:
            try:
                backend.unload()
            except Exception:
                pass
        if shutdown_controller is not None:
            try:
                shutdown_controller(controller)
            except Exception:
                pass
        if close_tdw_build_processes is not None:
            try:
                close_tdw_build_processes(payload["build_path"])
            except Exception:
                pass
