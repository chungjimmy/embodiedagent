import argparse
import inspect
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np

from dotenv import load_dotenv
from tdw.controller import Controller

from episode_worker import EpisodeWatchdog, SequenceWatchdog
from experiment import ExperimentRunner
from memory import MemoryManager
from movement import Agent
from planner import Planner
from scene import Scene
from task_contract import TaskContract
from vision import VisionLanguageModel


EXPERIMENT_TASKS = [
    {
        "id": "find_window",
        "goal": "Find a window and stop when one is clearly visible.",
        "task_spec": {
            "task_target": "window",
            "completion_mode": "target_visible",
            "completion_condition": (
                "At least one window is clearly identifiable in the current "
                "camera view."
            ),
            "requires_target_confirmation": True,
            "allowed_actions": [
                "move_forward",
                "move_backward",
                "turn_left",
                "turn_right",
            ],
        },
        "success_criteria": {
            "type": "manual_visual_review",
            "target_name": "window",
            "rubric": (
                "The saved final frame clearly contains at least one window."
            ),
        },
    },
]


def select_tasks(config):
    definitions = config.get("tasks") or EXPERIMENT_TASKS
    if not isinstance(definitions, list) or not definitions:
        raise ValueError("tasks must be a non-empty list when provided.")
    normalized = []
    for index, task in enumerate(definitions):
        if not isinstance(task, dict):
            raise ValueError(f"tasks[{index}] must be a JSON object.")
        goal = str(task.get("goal") or "").strip()
        if not goal:
            raise ValueError(f"tasks[{index}].goal must be non-empty.")
        task_id = str(task.get("id") or f"task_{index + 1}").strip()
        configured_spec = task.get("task_spec", task)
        contract = TaskContract.build(
            goal=goal,
            configured=configured_spec,
        )
        normalized_spec = (
            dict(configured_spec)
            if isinstance(configured_spec, dict)
            else {}
        )
        normalized_spec.update(contract.as_dict())
        normalized.append(
            {
                **task,
                "id": task_id,
                "goal": goal,
                "task_spec": normalized_spec,
            }
        )

    requested_ids = config.get("task_ids")
    if requested_ids is None:
        return normalized

    requested_ids = list(requested_ids)
    known_ids = {task["id"] for task in normalized}
    unknown_ids = sorted(set(requested_ids) - known_ids)
    if unknown_ids:
        raise ValueError(
            f"Unknown task_ids: {unknown_ids}. Available tasks: "
            f"{sorted(known_ids)}"
        )

    selected = [
        task for task in normalized if task["id"] in requested_ids
    ]
    if not selected:
        raise ValueError("task_ids must select at least one configured task.")
    return selected


def close_tdw_build_processes(build_path, timeout_seconds=10, tdw_port=1071):
    try:
        import psutil
    except ImportError as error:
        raise RuntimeError(
            "Reliable TDW cleanup requires psutil. Run: pip install psutil"
        ) from error

    target = Path(build_path).expanduser().resolve()

    def path_matches(executable):
        if not executable:
            return False
        executable_path = Path(executable).expanduser().resolve()
        if target.is_dir():
            return executable_path == target or target in executable_path.parents
        return executable_path == target

    def process_matches(process):
        try:
            info = process.as_dict(attrs=["exe", "cmdline", "name"])
        except (
            psutil.AccessDenied,
            psutil.NoSuchProcess,
            psutil.ZombieProcess,
        ):
            return False
        if path_matches(info.get("exe")):
            return True
        command_line = " ".join(info.get("cmdline") or [])
        target_text = os.path.normcase(str(target))
        return target_text in os.path.normcase(command_line)

    def port_processes():
        try:
            connections = psutil.net_connections(kind="tcp")
        except (psutil.AccessDenied, OSError):
            return []

        processes = {}
        for connection in connections:
            if not connection.laddr or connection.laddr.port != tdw_port:
                continue
            if not connection.pid or connection.pid == os.getpid():
                continue
            try:
                processes[connection.pid] = psutil.Process(connection.pid)
            except (psutil.AccessDenied, psutil.NoSuchProcess, psutil.ZombieProcess):
                continue
        return list(processes.values())

    matches_by_pid = {}
    for process in psutil.process_iter(["pid", "exe", "cmdline", "name"]):
        try:
            if process_matches(process):
                matches_by_pid[process.pid] = process
        except (
            psutil.AccessDenied,
            psutil.NoSuchProcess,
            psutil.ZombieProcess,
        ):
            continue

    for process in port_processes():
        matches_by_pid[process.pid] = process

    matches = list(matches_by_pid.values())

    if not matches:
        return

    print(f"Closing {len(matches)} remaining TDW build process(es)...")
    for process in matches:
        try:
            process.terminate()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass

    _, survivors = psutil.wait_procs(matches, timeout=timeout_seconds)
    for process in survivors:
        try:
            process.kill()
        except (psutil.AccessDenied, psutil.NoSuchProcess):
            pass
    if survivors:
        psutil.wait_procs(survivors, timeout=5)

    deadline = time.monotonic() + max(1, timeout_seconds)
    while time.monotonic() < deadline and port_processes():
        time.sleep(0.25)
    time.sleep(0.5)


def _supports_parameter(function, parameter):
    try:
        return parameter in inspect.signature(function).parameters
    except (TypeError, ValueError):
        return False


def close_controller_socket(controller):
    if controller is None:
        return
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


def shutdown_controller(controller):
    if controller is None:
        return
    try:
        controller.communicate({"$type": "terminate"})
    except Exception as error:
        print(f"TDW termination warning: {error}")
    finally:
        close_controller_socket(controller)


def _set_avatar_pose(
    controller,
    avatar_id,
    position,
    yaw,
    pitch=0.0,
    current_pitch=0.0,
):
    radians = math.radians(float(yaw))
    commands = [
            {
                "$type": "teleport_avatar_to",
                "avatar_id": avatar_id,
                "position": dict(position),
            },
            {
                "$type": "set_avatar_forward",
                "avatar_id": avatar_id,
                "forward": {
                    "x": math.sin(radians),
                    "y": 0,
                    "z": math.cos(radians),
                },
            },
        ]
    pitch = float(pitch)
    pitch_delta = pitch - float(current_pitch)
    if abs(pitch_delta) >= 1e-6:
        commands.append(
            {
                "$type": "rotate_sensor_container_by",
                "avatar_id": avatar_id,
                "axis": "pitch",
                "angle": pitch_delta,
            }
        )
    controller.communicate(commands)
    return pitch


def _target_segmentation_color(controller, object_id):
    from tdw.output_data import OutputData, SegmentationColors

    response = controller.communicate(
        {"$type": "send_segmentation_colors", "frequency": "once"}
    )
    for item in response:
        if OutputData.get_data_type_id(item) != "segm":
            continue
        segmentation = SegmentationColors(item)
        for index in range(segmentation.get_num()):
            if segmentation.get_object_id(index) != int(object_id):
                continue
            color = segmentation.get_object_color(index)
            return tuple(int(value) for value in list(color)[:3])
    return None


def _count_segmentation_pixels(image, color):
    return _segmentation_geometry(image, color)["visible_pixels"]


def _segmentation_geometry(image, color):
    pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
    expected = np.asarray(color, dtype=np.uint8)
    mask = np.all(pixels == expected, axis=-1)
    visible_pixels = int(np.count_nonzero(mask))
    height, width = mask.shape
    if visible_pixels == 0:
        return {
            "visible_pixels": 0,
            "bbox": None,
            "bbox_width": 0,
            "bbox_height": 0,
            "center_x_fraction": None,
            "center_y_fraction": None,
            "image_width": int(width),
            "image_height": int(height),
        }
    y_values, x_values = np.nonzero(mask)
    x_min = int(x_values.min())
    x_max = int(x_values.max())
    y_min = int(y_values.min())
    y_max = int(y_values.max())
    return {
        "visible_pixels": visible_pixels,
        "bbox": {
            "x_min": x_min,
            "y_min": y_min,
            "x_max": x_max,
            "y_max": y_max,
        },
        "bbox_width": x_max - x_min + 1,
        "bbox_height": y_max - y_min + 1,
        "center_x_fraction": round(((x_min + x_max) / 2) / width, 5),
        "center_y_fraction": round(((y_min + y_max) / 2) / height, 5),
        "image_width": int(width),
        "image_height": int(height),
    }


def _segmentation_geometry_passes(
    geometry,
    minimum_visible_pixels,
    minimum_bbox_width,
    minimum_bbox_height,
    edge_margin,
    minimum_vertical_center_fraction,
    maximum_vertical_center_fraction,
):
    bbox = geometry.get("bbox")
    if bbox is None:
        return False
    if geometry["visible_pixels"] < minimum_visible_pixels:
        return False
    if geometry["bbox_width"] < minimum_bbox_width:
        return False
    if geometry["bbox_height"] < minimum_bbox_height:
        return False
    if (
        bbox["x_min"] < edge_margin
        or bbox["y_min"] < edge_margin
        or bbox["x_max"] >= geometry["image_width"] - edge_margin
        or bbox["y_max"] >= geometry["image_height"] - edge_margin
    ):
        return False
    center_y = geometry.get("center_y_fraction")
    return (
        center_y is not None
        and minimum_vertical_center_fraction
        <= center_y
        <= maximum_vertical_center_fraction
    )


def verify_target_visibility_preflight(
    scene, agent, scene_config, initial_position, initial_yaw
):
    configured = scene_config.get("target_visibility_preflight", {})
    if configured in (None, False):
        return []
    if not isinstance(configured, dict):
        raise ValueError("scene.target_visibility_preflight must be an object.")
    if not bool(configured.get("enabled", True)):
        return []

    try:
        minimum_distance = float(configured.get("minimum_distance", 2.0))
        maximum_distance = float(configured.get("maximum_distance", 3.5))
        preferred_distance = float(configured.get("preferred_distance", 2.5))
        minimum_visible_pixels = int(configured.get("minimum_visible_pixels", 24))
        minimum_bbox_width = int(
            configured.get("minimum_bbox_width_pixels", 32)
        )
        minimum_bbox_height = int(
            configured.get("minimum_bbox_height_pixels", 32)
        )
        edge_margin = int(configured.get("edge_margin_pixels", 12))
        minimum_vertical_center_fraction = float(
            configured.get("minimum_vertical_center_fraction", 0.1)
        )
        maximum_vertical_center_fraction = float(
            configured.get("maximum_vertical_center_fraction", 0.9)
        )
        max_candidate_poses = int(configured.get("max_candidate_poses", 24))
    except (TypeError, ValueError) as error:
        raise ValueError(
            "scene.target_visibility_preflight values must be numeric."
        ) from error
    if min(
        minimum_visible_pixels,
        minimum_bbox_width,
        minimum_bbox_height,
    ) <= 0 or edge_margin < 0:
        raise ValueError(
            "target_visibility_preflight pixel thresholds must be positive."
        )
    if (
        minimum_distance <= 0
        or maximum_distance < minimum_distance
        or max_candidate_poses <= 0
    ):
        raise ValueError(
            "target_visibility_preflight distance range or candidate count is invalid."
        )
    if not (
        0
        <= minimum_vertical_center_fraction
        < maximum_vertical_center_fraction
        <= 1
    ):
        raise ValueError(
            "target_visibility_preflight vertical center fractions are invalid."
        )

    results = []
    initial_camera_pitch = float(agent.camera_pitch)
    tested_pitches = agent.get_supported_camera_pitches()
    try:
        for target in scene.get_task_objects():
            color = _target_segmentation_color(
                agent.controller, target["object_id"]
            )
            if color is None:
                raise RuntimeError(
                    "TDW did not return an instance-ID segmentation color for "
                    f"configured target {target['object_id']} ({target['name']})."
                )
            candidates = scene.target_view_candidates(
                target["x"],
                target["z"],
                minimum_distance=minimum_distance,
                maximum_distance=maximum_distance,
                preferred_distance=preferred_distance,
                clearance=scene.navigation_clearance,
                max_candidates=max_candidate_poses,
            )
            visible_from = None
            best_visible_pixels = 0
            best_pose = None
            best_geometry = None
            for candidate in candidates:
                for pitch in tested_pitches:
                    agent.camera_pitch = _set_avatar_pose(
                        agent.controller,
                        agent.avatar_id,
                        {
                            "x": candidate["x"],
                            "y": float(initial_position["y"]),
                            "z": candidate["z"],
                        },
                        candidate["yaw"],
                        pitch=pitch,
                        current_pitch=agent.camera_pitch,
                    )
                    id_pass = agent.get_render_pass("_id")
                    geometry = _segmentation_geometry(id_pass, color)
                    visible_pixels = geometry["visible_pixels"]
                    if visible_pixels > best_visible_pixels:
                        best_visible_pixels = visible_pixels
                        best_geometry = geometry
                        best_pose = {
                            "position": {
                                "x": candidate["x"],
                                "y": float(initial_position["y"]),
                                "z": candidate["z"],
                            },
                            "yaw": candidate["yaw"],
                            "pitch": pitch,
                            "distance": round(candidate["distance"], 3),
                        }
                    if not _segmentation_geometry_passes(
                        geometry=geometry,
                        minimum_visible_pixels=minimum_visible_pixels,
                        minimum_bbox_width=minimum_bbox_width,
                        minimum_bbox_height=minimum_bbox_height,
                        edge_margin=edge_margin,
                        minimum_vertical_center_fraction=(
                            minimum_vertical_center_fraction
                        ),
                        maximum_vertical_center_fraction=(
                            maximum_vertical_center_fraction
                        ),
                    ):
                        continue
                    visible_from = {
                        "position": {
                            "x": candidate["x"],
                            "y": float(initial_position["y"]),
                            "z": candidate["z"],
                        },
                        "yaw": candidate["yaw"],
                        "pitch": pitch,
                        "distance": round(candidate["distance"], 3),
                        "visible_pixels": visible_pixels,
                        "segmentation_geometry": geometry,
                    }
                    break
                if visible_from is not None:
                    break
            if visible_from is None:
                raise ValueError(
                    "Configured target is not visible from any tested reachable "
                    f"camera pose: {target['name']} (ID {target['object_id']}). "
                    f"Tested pitches: {tested_pitches}. Choose another target "
                    "location or adjust the fixed camera pitch. Target placement "
                    f"diagnostics: y={target.get('y')}, "
                    f"scale={target.get('scale_factor')}, "
                    f"grounding={target.get('grounding')}, "
                    f"candidate_poses={len(candidates)}, "
                    f"best_visible_pixels={best_visible_pixels}, "
                    f"best_pose={best_pose}, best_geometry={best_geometry}."
                )
            witness_image = agent.get_current_view()
            witness_path = agent.save_current_view(
                image=witness_image,
                prefix=f"preflight_target_{target['object_id']}",
            )
            visible_from["rgb_witness_path"] = witness_path
            results.append(
                {
                    "object_id": target["object_id"],
                    "name": target["name"],
                    "segmentation_color": color,
                    "visible_from": visible_from,
                    "grounding": target.get("grounding"),
                    "scale_factor": target.get("scale_factor"),
                }
            )
            print(
                f"✓ Target visibility preflight passed: {target['name']} "
                f"(ID: {target['object_id']}, "
                f"pitch: {visible_from['pitch']}°, "
                f"pixels: {visible_from['visible_pixels']})"
            )
    finally:
        agent.camera_pitch = _set_avatar_pose(
            agent.controller,
            agent.avatar_id,
            initial_position,
            initial_yaw,
            pitch=initial_camera_pitch,
            current_pitch=agent.camera_pitch,
        )
        if agent.capture is not None:
            agent.capture.set(frequency="never")
    return results


def create_scene(build_path, scene_config, seed, episode_directory):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass

    controller = None
    try:
        controller = Controller(launch_build=build_path)
        controller.communicate(
            {
                "$type": "set_screen_size",
                "width": int(scene_config.get("width", 1280)),
                "height": int(scene_config.get("height", 720)),
            }
        )

        turn_angle_degrees = float(
            scene_config.get("turn_angle_degrees", Agent.DEFAULT_TURN_ANGLE)
        )
        scene = Scene(
            controller,
            turn_angle_degrees=turn_angle_degrees,
        )
        scene_name = scene_config.get("name", "archviz_house")
        seed_applied = False
        if _supports_parameter(scene.create_room, "seed"):
            scene.create_room(scene_name=scene_name, seed=seed)
        else:
            scene.create_room(scene_name=scene_name)

        initial_position = scene_config.get(
            "initial_position", {"x": 5, "y": 1.6, "z": -5}
        )
        initial_yaw = float(scene_config.get("initial_yaw", 0))
        navigation = scene_config.get("navigation", {})
        navigation_enabled = bool(navigation.get("enabled", False))

        if _supports_parameter(scene.add_objects, "seed"):
            scene.add_objects(seed=seed)
        else:
            scene.add_objects()

        task_objects = scene_config.get("task_objects", [])
        if task_objects is None:
            task_objects = []
        if not isinstance(task_objects, list):
            raise ValueError("scene.task_objects must be a list when configured.")

        navigation_map_generated = False

        def generate_navigation_map():
            nonlocal navigation_map_generated
            scene.setup_navigation(
                cell_size=float(navigation.get("cell_size", 0.25)),
                raycast_y=float(
                    navigation.get("raycast_y", initial_position.get("y", 1.6))
                ),
                clearance=float(navigation.get("clearance", 0.35)),
                path_sample_spacing=float(
                    navigation.get("path_sample_spacing", 0.2)
                ),
            )
            navigation_map_generated = True

        uses_dynamic_target_placement = any(
            isinstance(item, dict) and not isinstance(item.get("position"), dict)
            for item in task_objects
        )
        if uses_dynamic_target_placement and not navigation_enabled:
            raise ValueError(
                "scene.task_objects uses dynamic placement but "
                "scene.navigation.enabled is false."
            )
        if uses_dynamic_target_placement:
            generate_navigation_map()
            task_objects = scene.resolve_task_object_positions(
                task_objects,
                initial_position=initial_position,
            )
        if task_objects:
            scene.add_configured_objects(task_objects)

        seed_applied = not scene.using_preloaded

        if navigation_enabled:
            navigation_clearance = float(navigation.get("clearance", 0.35))
            if not navigation_map_generated:
                generate_navigation_map()
            spawn_is_navigable = scene.is_navigable(
                initial_position["x"],
                initial_position["z"],
                clearance=navigation_clearance,
            )
            spawn_has_departure = scene.has_safe_departure(
                initial_position["x"],
                initial_position["z"],
                distance=Agent.DEFAULT_MOVE_STEP,
                clearance=navigation_clearance,
            )
            if not spawn_is_navigable or not spawn_has_departure:
                nearest = scene.nearest_navigable_position(
                    initial_position["x"],
                    initial_position["z"],
                    clearance=navigation_clearance,
                    min_departure=Agent.DEFAULT_MOVE_STEP,
                )
                failure_reason = (
                    "the spawn point itself does not clear the configured "
                    f"clearance={navigation_clearance:g}m"
                    if not spawn_is_navigable
                    else "the spawn point has no clear "
                    f"{Agent.DEFAULT_MOVE_STEP:g}m departure"
                )
                raise ValueError(
                    "Configured initial_position is not safely navigable: "
                    f"{failure_reason}. initial_position={initial_position}. "
                    "Suggested safe coordinate (verified navigable with a "
                    f"clear {Agent.DEFAULT_MOVE_STEP:g}m departure): {nearest}. "
                    "Map diagnostics: "
                    f"{scene.navigation_diagnostics()}."
                )
        camera_config = scene_config.get("camera", {})
        agent = Agent(
            controller,
            avatar_id=scene_config.get("avatar_id", "a"),
            initial_position=initial_position,
            scene=scene,
            initial_yaw=initial_yaw,
            initial_camera_pitch=float(
                camera_config.get("initial_pitch_degrees", 0)
            ),
            camera_pitch_step=float(
                camera_config.get("pitch_step_degrees", Agent.DEFAULT_LOOK_ANGLE)
            ),
            minimum_camera_pitch=float(
                camera_config.get(
                    "minimum_pitch_degrees",
                    Agent.DEFAULT_MIN_CAMERA_PITCH,
                )
            ),
            maximum_camera_pitch=float(
                camera_config.get(
                    "maximum_pitch_degrees",
                    Agent.DEFAULT_MAX_CAMERA_PITCH,
                )
            ),
            turn_angle=turn_angle_degrees,
            blank_frame_retry_count=int(
                camera_config.get("blank_frame_retry_count", 2)
            ),
            reject_blank_frames=bool(
                camera_config.get("reject_blank_frames", True)
            ),
        )
        agent.capture_directory = Path(episode_directory) / "captures"
        agent.setup_avatar()
        agent.setup_image_capture()
        target_visibility_preflight = verify_target_visibility_preflight(
            scene,
            agent,
            scene_config,
            initial_position,
            initial_yaw,
        )

        return controller, agent, {
            "scene_name": scene_name,
            "requested_seed": seed,
            "tdw_scene_seed_applied": seed_applied,
            "initial_position": initial_position,
            "initial_yaw": initial_yaw,
            "turn_angle_degrees": turn_angle_degrees,
            "navigation_occupancy_map_enabled": navigation_enabled,
            "navigation_diagnostics": (
                scene.navigation_diagnostics()
                if navigation_enabled
                else None
            ),
            "task_objects": scene.get_task_objects(),
            "target_visibility_preflight": target_visibility_preflight,
        }
    except Exception:
        shutdown_controller(controller)
        raise


def load_config(path):
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def resolve_scene_episode(config, split="development", index=0):
    scene = dict(config.get("scene", {}))
    episodes = scene.get(f"{split}_episodes")
    if episodes:
        entry = dict(episodes[index])
        seed = int(entry.pop("seed", index))
        entry.pop("id", None)
        for key in (
            "development_episodes",
            "evaluation_episodes",
            "development_seeds",
            "evaluation_seeds",
            "seeds",
        ):
            scene.pop(key, None)
        scene.update(entry)
        return scene, seed
    seeds = scene.get(f"{split}_seeds", scene.get("seeds", [0]))
    return scene, int(seeds[index])


def run_automated(config_path, build_path):
    config = load_config(config_path)
    config["tasks"] = select_tasks(config)

    close_tdw_build_processes(build_path)

    def episode_factory(scene_config, seed, episode_directory):
        return create_scene(
            build_path=build_path,
            scene_config=scene_config,
            seed=seed,
            episode_directory=episode_directory,
        )

    timeout_seconds = float(config.get("episode_timeout_seconds", 0) or 0)
    watchdog = None
    if timeout_seconds > 0:
        watchdog = EpisodeWatchdog(
            build_path=build_path,
            timeout_seconds=timeout_seconds,
            termination_grace_seconds=float(
                config.get("watchdog_termination_grace_seconds", 15)
            ),
        )
        print(
            "Episode watchdog enabled: each automated episode runs in an "
            f"isolated process with a {timeout_seconds:g}s hard limit.",
            flush=True,
        )

    sequence_watchdog = None
    sequential_extension = config.get("extensions", {}).get(
        "sequential_tasks", {}
    )
    sequence_timeout_seconds = float(
        sequential_extension.get("hard_timeout_seconds", 0) or 0
    )
    if sequential_extension.get("enabled", False) and sequence_timeout_seconds > 0:
        sequence_watchdog = SequenceWatchdog(
            build_path=build_path,
            timeout_seconds=sequence_timeout_seconds,
            termination_grace_seconds=float(
                sequential_extension.get(
                    "watchdog_termination_grace_seconds",
                    config.get("watchdog_termination_grace_seconds", 15),
                )
            ),
        )
        print(
            "Sequence watchdog enabled: all dependent stages share one "
            f"worker process with a {sequence_timeout_seconds:g}s hard limit.",
            flush=True,
        )

    runner = ExperimentRunner(
        config,
        episode_factory,
        episode_cleanup=lambda: close_tdw_build_processes(build_path),
        isolated_episode_executor=watchdog,
        isolated_sequence_executor=sequence_watchdog,
    )
    results = runner.run()
    print(f"Completed {len(results)} episodes.")
    print(f"Results: {runner.output_directory.resolve()}")


def run_interactive(config_path, build_path):
    config = load_config(config_path)
    model_entry = config["models"][0]
    model_name = model_entry["name"] if isinstance(model_entry, dict) else model_entry
    controller = None
    try:
        scene_config, development_seed = resolve_scene_episode(
            config, split="development"
        )
        controller, agent, metadata = create_scene(
            build_path=build_path,
            scene_config=scene_config,
            seed=development_seed,
            episode_directory="interactive_run",
        )
        backend = VisionLanguageModel(
            model_name=model_name,
            use_4bit=(
                model_entry.get("use_4bit", config.get("use_4bit", True))
                if isinstance(model_entry, dict)
                else config.get("use_4bit", True)
            ),
            max_new_tokens=(
                int(model_entry.get("max_new_tokens", 300))
                if isinstance(model_entry, dict)
                else 300
            ),
            min_pixels=(
                model_entry.get("min_pixels")
                if isinstance(model_entry, dict)
                else None
            ),
            max_pixels=(
                model_entry.get("max_pixels")
                if isinstance(model_entry, dict)
                else None
            ),
            clear_cuda_cache_between_calls=(
                bool(model_entry.get("clear_cuda_cache_between_calls", False))
                if isinstance(model_entry, dict)
                else False
            ),
            generation_max_time_seconds=(
                model_entry.get("generation_max_time_seconds")
                if isinstance(model_entry, dict)
                else None
            ),
            four_bit_compute_dtype=(
                model_entry.get(
                    "four_bit_compute_dtype",
                    config.get("four_bit_compute_dtype", "auto"),
                )
                if isinstance(model_entry, dict)
                else config.get("four_bit_compute_dtype", "auto")
            ),
            log_cuda_memory=(
                bool(
                    model_entry.get(
                        "log_cuda_memory",
                        config.get("log_cuda_memory", True),
                    )
                )
                if isinstance(model_entry, dict)
                else bool(config.get("log_cuda_memory", True))
            ),
            cuda_allocator_config=(
                model_entry.get(
                    "cuda_allocator_config",
                    config.get("cuda_allocator_config"),
                )
                if isinstance(model_entry, dict)
                else config.get("cuda_allocator_config")
            ),
        )
        interactive = config.get("interactive", {})
        memory_context = config.get("memory_context", {})
        if not isinstance(memory_context, dict):
            raise ValueError("memory_context must be an object when configured.")
        memory = MemoryManager(
            episodic_window=int(config.get("episodic_window", 8)),
            context_episodic_limit=int(memory_context.get("episodic_events", 4)),
            context_object_limit=int(memory_context.get("semantic_objects", 12)),
            context_pose_limit=int(memory_context.get("visited_poses", 8)),
            context_fact_limit=int(memory_context.get("semantic_facts", 12)),
            context_text_char_limit=int(memory_context.get("text_characters", 1000)),
            context_prompt_char_limit=int(
                memory_context.get("prompt_characters", 3000)
            ),
        )
        planner = Planner(
            model_name=model_name,
            backend=backend,
            perceptual_interface=interactive.get(
                "perceptual_interface", "text_mediated"
            ),
            memory_mode=interactive.get("memory_mode", "semantic"),
            memory_manager=memory,
            history_directory="interactive_run",
            perception_max_new_tokens=int(
                config.get("perception_max_new_tokens", 300)
            ),
            perception_retry_max_new_tokens=int(
                config.get("perception_retry_max_new_tokens", 224)
            ),
            decision_max_new_tokens=int(
                config.get("decision_max_new_tokens", 300)
            ),
            completion_max_new_tokens=int(
                config.get("completion_max_new_tokens", 64)
            ),
            save_step_images=bool(config.get("save_step_images", False)),
        )
        for task in select_tasks(config):
            planner.run_task(
                agent,
                task["goal"],
                max_steps=int(config.get("max_steps", 20)),
                episode_metadata={
                    "interactive": True,
                    "task_id": task["id"],
                    **metadata,
                },
                success_evaluator=lambda history=None, task=task: agent.evaluate_success(
                    task.get("success_criteria"),
                    history=history,
                ),
                task_spec=task.get("task_spec"),
            )
    finally:
        shutdown_controller(controller)
        close_tdw_build_processes(build_path)


DEFAULT_CONFIGS = ("experiment_config.json", "three_task_config.json")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=None,
        help=(
            "Experiment JSON file, or a suite file listing several configs "
            "under run_configs. When omitted, runs task 1 "
            f"({DEFAULT_CONFIGS[0]}) and then task 2 ({DEFAULT_CONFIGS[1]})."
        ),
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run one manual development condition instead of the full sweep",
    )
    return parser.parse_args()


def main():
    load_dotenv()
    arguments = parse_arguments()
    build_path = os.getenv("BUILD_PATH")
    if not build_path:
        raise ValueError("BUILD_PATH was not found in the .env file.")
    if arguments.interactive:
        run_interactive(arguments.config or DEFAULT_CONFIGS[0], build_path)
    elif arguments.config:
        from experiment import expand_config_sequence

        config_paths = expand_config_sequence(arguments.config)
        if len(config_paths) == 1:
            run_automated(str(config_paths[0]), build_path)
            return
        failures = []
        for number, config_path in enumerate(config_paths, start=1):
            print(f"\n===== Config {number}/{len(config_paths)}: {config_path} =====")
            try:
                run_automated(str(config_path), build_path)
            except Exception as error:
                print(f"Config failed: {config_path}: {error}", flush=True)
                failures.append(str(config_path))
        if failures:
            raise SystemExit("Some configs failed: " + ", ".join(failures))
    else:
        for number, config_path in enumerate(DEFAULT_CONFIGS, start=1):
            print(f"\n===== Task {number}: {config_path} =====")
            run_automated(config_path, build_path)


if __name__ == "__main__":
    main()
