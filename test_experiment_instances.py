import tempfile
import unittest
import csv
import importlib.util
import json
from pathlib import Path
import sys
import types
from unittest import mock

from PIL import Image

from experiment import Condition, ExperimentRunner, expand_config_sequence


class ExperimentInstanceTests(unittest.TestCase):
    def test_three_task_config_requires_exact_target_visibility(self):
        config_path = Path(__file__).with_name("three_task_config.json")
        with config_path.open("r", encoding="utf-8") as file:
            config = json.load(file)

        expected_ids = [1001, 1002, 1001]
        self.assertEqual(len(config["tasks"]), len(expected_ids))
        for task, expected_id in zip(config["tasks"], expected_ids):
            criteria = task["success_criteria"]["criteria"]
            visibility = [
                item
                for item in criteria
                if item.get("type") == "object_visibility"
            ]
            self.assertEqual(len(visibility), 1)
            self.assertEqual(visibility[0]["target_object_id"], expected_id)
            self.assertGreater(visibility[0]["minimum_visible_pixels"], 0)
            self.assertFalse(
                any(
                    item.get("type") == "minimum_successful_translations"
                    for item in criteria
                )
            )
            self.assertTrue(
                task["task_spec"]["target_reacquisition_after_turn"]
            )
            self.assertEqual(
                task["task_spec"]["target_approach_persistence_steps"], 1
            )

    def test_object_visibility_uses_exact_instance_id_pixels(self):
        def stub(name, **attributes):
            module = types.ModuleType(name)
            for key, value in attributes.items():
                setattr(module, key, value)
            return module

        tdw_package = stub("tdw")
        tdw_package.__path__ = []
        tdw_add_ons = stub("tdw.add_ons")
        tdw_add_ons.__path__ = []
        stubs = {
            "tdw": tdw_package,
            "tdw.add_ons": tdw_add_ons,
            "tdw.add_ons.image_capture": stub(
                "tdw.add_ons.image_capture", ImageCapture=object
            ),
        }
        movement_path = Path(__file__).with_name("movement.py")
        with mock.patch.dict(sys.modules, stubs):
            spec = importlib.util.spec_from_file_location(
                "movement_visibility_test", movement_path
            )
            movement_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(movement_module)

        target_color = (17, 34, 51)
        id_image = Image.new("RGB", (20, 20), (0, 0, 0))
        for x in range(10):
            for y in range(10):
                id_image.putpixel((x, y), target_color)
        agent = movement_module.Agent.__new__(movement_module.Agent)
        agent._segmentation_color = lambda object_id: target_color
        agent.get_render_pass = lambda pass_mask: id_image

        result = agent.evaluate_success(
            {
                "type": "object_visibility",
                "target_object_id": 1001,
                "minimum_visible_pixels": 64,
            }
        )

        self.assertTrue(result["available"])
        self.assertTrue(result["success"])
        self.assertEqual(result["visible_pixels"], 100)
        self.assertEqual(result["target_object_id"], 1001)

    def test_object_visibility_fails_closed_when_evaluator_breaks(self):
        def stub(name, **attributes):
            module = types.ModuleType(name)
            for key, value in attributes.items():
                setattr(module, key, value)
            return module

        tdw_package = stub("tdw")
        tdw_package.__path__ = []
        tdw_add_ons = stub("tdw.add_ons")
        tdw_add_ons.__path__ = []
        stubs = {
            "tdw": tdw_package,
            "tdw.add_ons": tdw_add_ons,
            "tdw.add_ons.image_capture": stub(
                "tdw.add_ons.image_capture", ImageCapture=object
            ),
        }
        movement_path = Path(__file__).with_name("movement.py")
        with mock.patch.dict(sys.modules, stubs):
            spec = importlib.util.spec_from_file_location(
                "movement_visibility_failure_test", movement_path
            )
            movement_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(movement_module)

        agent = movement_module.Agent.__new__(movement_module.Agent)
        agent._segmentation_color = lambda object_id: None

        result = agent.evaluate_success(
            {
                "type": "object_visibility",
                "target_object_id": 1001,
                "minimum_visible_pixels": 64,
            }
        )

        self.assertTrue(result["available"])
        self.assertFalse(result["success"])
        self.assertTrue(result["evaluator_error"])

    def test_main_task_selection_preserves_runtime_planner_policies(self):
        def stub(name, **attributes):
            module = types.ModuleType(name)
            for key, value in attributes.items():
                setattr(module, key, value)
            return module

        placeholder = type("Placeholder", (), {})
        tdw_package = stub("tdw")
        tdw_package.__path__ = []
        stubs = {
            "dotenv": stub("dotenv", load_dotenv=lambda: None),
            "tdw": tdw_package,
            "tdw.controller": stub("tdw.controller", Controller=placeholder),
            "episode_worker": stub(
                "episode_worker",
                EpisodeWatchdog=placeholder,
                SequenceWatchdog=placeholder,
            ),
            "experiment": stub("experiment", ExperimentRunner=placeholder),
            "memory": stub("memory", MemoryManager=placeholder),
            "movement": stub("movement", Agent=placeholder),
            "planner": stub("planner", Planner=placeholder),
            "scene": stub("scene", Scene=placeholder),
            "vision": stub("vision", VisionLanguageModel=placeholder),
        }
        main_path = Path(__file__).with_name("main.py")
        with mock.patch.dict(sys.modules, stubs):
            spec = importlib.util.spec_from_file_location(
                "main_task_selection_test", main_path
            )
            main_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(main_module)

        selected = main_module.select_tasks(
            {
                "tasks": [
                    {
                        "id": "pink",
                        "goal": "Find the pink box.",
                        "task_spec": {
                            "task_target": "box",
                            "target_attributes": ["pink"],
                            "completion_mode": "target_near",
                            "visible_target_approach": True,
                            "target_reacquisition_after_turn": True,
                            "target_approach_persistence_steps": 3,
                            "collision_aware_action_filter": True,
                            "completion_consistency": {
                                "recheck_on_environment_success": True
                            },
                        },
                    }
                ]
            }
        )
        normalized = selected[0]["task_spec"]
        self.assertTrue(normalized["visible_target_approach"])
        self.assertTrue(normalized["target_reacquisition_after_turn"])
        self.assertEqual(normalized["target_approach_persistence_steps"], 3)
        self.assertTrue(normalized["collision_aware_action_filter"])
        self.assertTrue(
            normalized["completion_consistency"][
                "recheck_on_environment_success"
            ]
        )

    def test_sequential_smoke_config_can_skip_primary_condition_sweep(self):
        class FakeBackend:
            def load(self):
                return None

            def unload(self):
                return None

        with tempfile.TemporaryDirectory() as directory:
            runner = ExperimentRunner(
                {
                    "models": ["fake"],
                    "tasks": [],
                    "output_directory": directory,
                    "run_primary_conditions": False,
                    "extensions": {"sequential_tasks": {"enabled": False}},
                },
                episode_factory=lambda **kwargs: None,
            )
            calls = []
            runner._create_backend = lambda model_name: FakeBackend()
            runner._run_condition = lambda condition, backend: calls.append(
                (condition, backend)
            )
            results = runner.run()

        self.assertEqual(calls, [])
        self.assertEqual(results, [])

    def test_sequential_extension_keeps_one_memory_across_three_stages(self):
        class FakeController:
            def communicate(self, command):
                return []

        class FakeAgent:
            def get_state_snapshot(self):
                return {
                    "position": {"x": 1.0, "y": 1.6, "z": 2.0},
                    "yaw": 0.0,
                }

        class FakeMemory:
            def __init__(self):
                self.boundaries = []

            def record_task_boundary(self, **kwargs):
                self.boundaries.append(kwargs)

        class FakePlanner:
            def __init__(self):
                self.memory = FakeMemory()
                self.calls = []

            def run_task(self, **kwargs):
                self.calls.append(kwargs)
                return {
                    "status": "completed",
                    "task_contract": kwargs["task_spec"],
                }

        tasks = [
            {
                "id": "vase",
                "goal": "Find the vase.",
                "task_spec": {"task_target": "vase"},
            },
            {
                "id": "box",
                "goal": "Find the box.",
                "max_steps": 12,
                "task_spec": {"task_target": "box"},
            },
            {
                "id": "return_vase",
                "goal": "Return to the vase.",
                "task_spec": {"task_target": "vase"},
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            runner = ExperimentRunner(
                {
                    "models": ["fake"],
                    "tasks": tasks,
                    "output_directory": directory,
                    "max_steps": 9,
                    "extensions": {
                        "sequential_tasks": {
                            "enabled": True,
                            "memory_mode": "combined",
                            "perceptual_interface": "text_mediated",
                        }
                    },
                },
                episode_factory=lambda **kwargs: (
                    FakeController(),
                    FakeAgent(),
                    {"scene_name": "fake"},
                ),
            )
            fake_planner = FakePlanner()
            runner._create_planner = lambda *args: fake_planner
            runner._terminate_controller = lambda controller: None
            runner._cleanup_build_processes = lambda: None
            runner._run_sequential_extension("fake", backend=object())

        self.assertEqual(
            [call["reset_memory"] for call in fake_planner.calls],
            [True, False, False],
        )
        self.assertEqual(
            [call["max_steps"] for call in fake_planner.calls], [9, 12, 9]
        )
        self.assertEqual(
            [boundary["task_id"] for boundary in fake_planner.memory.boundaries],
            ["vase", "box", "return_vase"],
        )

    def test_start_pose_moves_agent_before_only_that_stage(self):
        class FakeController:
            def communicate(self, command):
                return []

        class FakeAgent:
            def __init__(self):
                self.poses = []
                self.position = {"x": 1.0, "y": 1.6, "z": 2.0}
                self.yaw = 0.0

            def set_pose(self, position, yaw):
                self.poses.append((dict(position), yaw))
                self.position = dict(position)
                self.yaw = yaw

            def get_state_snapshot(self):
                return {"position": dict(self.position), "yaw": self.yaw}

        class FakeMemory:
            def __init__(self):
                self.resets = 0

            def record_task_boundary(self, **kwargs):
                return None

        class FakePlanner:
            def __init__(self, agent):
                self.memory = FakeMemory()
                self.agent = agent
                self.calls = []

            def run_task(self, **kwargs):
                self.calls.append(
                    (
                        dict(self.agent.position),
                        kwargs["reset_memory"],
                        dict(kwargs["episode_metadata"]),
                    )
                )
                return {"status": "completed", "task_success": True}

        tasks = [
            {"id": "a", "goal": "Find a.", "task_spec": {"task_target": "a"}},
            {"id": "b", "goal": "Find b.", "task_spec": {"task_target": "b"}},
            {
                "id": "back",
                "goal": "Return to a.",
                "task_spec": {"task_target": "a"},
                "start_pose": {"x": -3.0, "z": 4.0, "yaw": 90},
            },
        ]
        agent = FakeAgent()
        with tempfile.TemporaryDirectory() as directory:
            runner = ExperimentRunner(
                {
                    "models": ["fake"],
                    "tasks": tasks,
                    "output_directory": directory,
                    "max_steps": 5,
                    "extensions": {
                        "sequential_tasks": {
                            "enabled": True,
                            "memory_mode": "combined",
                            "perceptual_interface": "text_mediated",
                        }
                    },
                },
                episode_factory=lambda **kwargs: (
                    FakeController(),
                    agent,
                    {"scene_name": "fake"},
                ),
            )
            planner = FakePlanner(agent)
            runner._create_planner = lambda *args: planner
            runner._terminate_controller = lambda controller: None
            runner._cleanup_build_processes = lambda: None
            runner._run_sequential_extension("fake", backend=object())

        self.assertEqual(agent.poses, [({"x": -3.0, "y": 1.6, "z": 4.0}, 90)])
        positions = [call[0] for call in planner.calls]
        self.assertEqual(positions[0], {"x": 1.0, "y": 1.6, "z": 2.0})
        self.assertEqual(positions[2], {"x": -3.0, "y": 1.6, "z": 4.0})
        self.assertEqual([call[1] for call in planner.calls], [True, False, False])
        self.assertNotIn("start_pose_reset", planner.calls[1][2])
        reset = planner.calls[2][2]["start_pose_reset"]
        self.assertEqual((reset["x"], reset["z"], reset["yaw"]), (-3.0, 4.0, 90.0))
        self.assertEqual(
            reset["pose_before_reset"]["position"], {"x": 1.0, "y": 1.6, "z": 2.0}
        )

    def test_start_pose_requires_coordinates_and_capable_agent(self):
        with self.assertRaises(ValueError):
            ExperimentRunner._apply_task_start_pose(
                object(), {"start_pose": {"x": 1.0}}, {}
            )
        with self.assertRaises(ValueError):
            ExperimentRunner._apply_task_start_pose(
                object(), {"start_pose": {"x": 1.0, "z": 2.0}}, {}
            )
        self.assertIsNone(
            ExperimentRunner._apply_task_start_pose(object(), {"id": "x"}, {})
        )

    def test_fixed_return_config_resets_only_the_return_stage(self):
        import json

        with open("three_task_config_fixed_return.json") as handle:
            fixed = json.load(handle)
        with open("three_task_config.json") as handle:
            original = json.load(handle)
        poses = {task["id"]: task.get("start_pose") for task in fixed["tasks"]}
        self.assertIsNone(poses["locate_pink_box"])
        self.assertIsNone(poses["locate_yellow_box"])
        self.assertEqual(
            set(poses["return_to_pink_box"]), {"x", "y", "z", "yaw"}
        )
        self.assertNotEqual(
            fixed["output_directory"], original["output_directory"]
        )
        for task in fixed["tasks"]:
            task.pop("start_pose", None)
        fixed.pop("output_directory")
        original.pop("output_directory")
        self.assertEqual(fixed, original)

    def test_return_stage_is_marked_precondition_failed_after_bad_vase_stop(self):
        class FakeController:
            def communicate(self, command):
                return []

        class FakeAgent:
            def get_state_snapshot(self):
                return {
                    "position": {"x": 1.0, "y": 1.6, "z": 2.0},
                    "yaw": 0.0,
                }

        class FakeMemory:
            def record_task_boundary(self, **kwargs):
                return None

            def has_task_landmark(self, task_id=None):
                return False

        class FakePlanner:
            def __init__(self):
                self.memory = FakeMemory()
                self.calls = []

            def run_task(self, **kwargs):
                self.calls.append(kwargs["goal"])
                successful = kwargs["goal"] == "Find the box."
                return {
                    "status": "completed" if successful else "action_limit",
                    "task_success": successful,
                    "task_contract": kwargs["task_spec"],
                }

        task_spec = {
            "completion_mode": "target_near",
            "allowed_actions": ["move_forward", "turn_right"],
        }
        tasks = [
            {
                "id": "locate_vase",
                "goal": "Find the vase.",
                "task_spec": {"task_target": "vase", **task_spec},
            },
            {
                "id": "locate_box",
                "goal": "Find the box.",
                "task_spec": {"task_target": "box", **task_spec},
            },
            {
                "id": "return_to_vase",
                "goal": "Return to the vase.",
                "task_spec": {"task_target": "vase", **task_spec},
                "sequential_preconditions": {
                    "requires_successful_task_ids": ["locate_vase"],
                    "requires_landmark_task_id": "locate_vase",
                },
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            runner = ExperimentRunner(
                {
                    "models": ["fake"],
                    "tasks": tasks,
                    "output_directory": directory,
                    "extensions": {"sequential_tasks": {"enabled": True}},
                },
                episode_factory=lambda **kwargs: (
                    FakeController(),
                    FakeAgent(),
                    {"scene_name": "fake"},
                ),
            )
            planner = FakePlanner()
            runner._create_planner = lambda *args: planner
            runner._terminate_controller = lambda controller: None
            runner._cleanup_build_processes = lambda: None
            runner._run_sequential_extension("fake", backend=object())

        self.assertEqual(planner.calls, ["Find the vase.", "Find the box."])
        self.assertEqual(
            [result["status"] for result in runner.results],
            ["action_limit", "completed", "precondition_failed"],
        )
        self.assertIsNone(runner.results[-1]["task_success"])

    def test_explicit_scene_instances_override_ineffective_seed_only_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = ExperimentRunner(
                {
                    "models": ["fake"],
                    "tasks": [{"id": "task", "goal": "Do a task."}],
                    "output_directory": directory,
                    "run_split": "evaluation",
                    "scene": {
                        "name": "archviz_house",
                        "initial_position": {"x": 5, "y": 1.6, "z": -5},
                        "evaluation_episodes": [
                            {
                                "id": "heading_000",
                                "seed": 101,
                                "initial_yaw": 0,
                            },
                            {
                                "id": "heading_180",
                                "seed": 202,
                                "initial_yaw": 180,
                            },
                        ],
                    },
                },
                episode_factory=lambda **kwargs: None,
            )
            episodes = runner._episode_specs()

        self.assertEqual([item["id"] for item in episodes], [
            "heading_000",
            "heading_180",
        ])
        self.assertEqual(
            [item["scene_config"]["initial_yaw"] for item in episodes],
            [0, 180],
        )
        self.assertNotIn("evaluation_episodes", episodes[0]["scene_config"])

    def test_manual_review_label_is_applied_without_becoming_model_ground_truth(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = ExperimentRunner(
                {
                    "models": ["fake"],
                    "tasks": [],
                    "output_directory": directory,
                },
                episode_factory=lambda **kwargs: None,
            )
            runner.results = [
                {
                    "goal": "Find a window.",
                    "status": "completed",
                    "agent_declared_success": True,
                    "task_success": None,
                    "requires_manual_review": True,
                    "ground_truth_evaluation": {"rubric": "Window visible."},
                    "episode_metadata": {"episode_id": "episode_1"},
                    "final_image_path_relative": "captures/final.png",
                    "inference": {},
                }
            ]
            runner._save_summary()
            review_path = Path(runner.output_directory) / "manual_review.csv"
            with review_path.open("r", newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            rows[0]["manual_target_visible"] = "true"
            with review_path.open("w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            runner._save_summary()

        self.assertTrue(runner.results[0]["manual_target_visible"])
        self.assertTrue(runner.results[0]["task_success"])

    def test_hard_watchdog_timeout_is_not_recorded_as_a_normal_error(self):
        result = ExperimentRunner._timeout_result(
            condition=Condition("fake", "direct_visual", "none"),
            task={"id": "task", "goal": "Do a task."},
            seed=4,
            episode_id="episode_timeout",
            scene_episode_id="scene_1",
            timeout_seconds=600,
            duration_seconds=600.1,
            checkpoint_path="checkpoint.json",
            partial_history_path="partial_history.json",
            worker_exitcode=-15,
        )
        self.assertEqual(result["status"], "timeout")
        self.assertTrue(result["hard_timeout"])
        self.assertIsNone(result["task_success"])
        self.assertFalse(result["perception_parser_applicable"])


class SequentialConditionsTests(unittest.TestCase):
    TASKS = [
        {"id": "one", "goal": "Find one.", "task_spec": {"task_target": "box"}},
        {"id": "two", "goal": "Find two.", "task_spec": {"task_target": "box"}},
    ]

    def _runner(self, directory, sequential, **extra):
        return ExperimentRunner(
            {
                "models": ["fake"],
                "tasks": self.TASKS,
                "output_directory": directory,
                "run_primary_conditions": False,
                "extensions": {"sequential_tasks": {"enabled": True, **sequential}},
                **extra,
            },
            episode_factory=lambda **kwargs: None,
            **({"isolated_sequence_executor": extra.pop("executor")} if "executor" in extra else {}),
        )

    def test_without_a_conditions_list_the_single_condition_keys_still_apply(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = self._runner(
                directory,
                {"perceptual_interface": "text_mediated", "memory_mode": "semantic"},
            )
            conditions = runner._sequential_conditions("fake")
        self.assertEqual(
            [(c.perceptual_interface, c.memory_mode) for c in conditions],
            [("text_mediated", "semantic")],
        )

    def test_conditions_list_gives_one_sequence_per_entry_in_order(self):
        with tempfile.TemporaryDirectory() as directory:
            runner = self._runner(
                directory,
                {
                    "memory_mode": "combined",
                    "conditions": [
                        {"perceptual_interface": "text_mediated"},
                        {"perceptual_interface": "direct_visual", "memory_mode": "episodic"},
                    ],
                },
            )
            conditions = runner._sequential_conditions("fake")
        self.assertEqual(
            [(c.perceptual_interface, c.memory_mode, c.baseline) for c in conditions],
            [
                ("text_mediated", "combined", "sequential_extension"),
                ("direct_visual", "episodic", "sequential_extension"),
            ],
        )

    def test_bad_conditions_are_rejected(self):
        bad_values = [
            [],
            "direct_visual",
            [{"perceptual_interface": "telepathy"}],
            [{"memory_mode": "combined"}],
            ["direct_visual"],
            [
                {"perceptual_interface": "direct_visual", "memory_mode": "combined"},
                {"perceptual_interface": "direct_visual", "memory_mode": "combined"},
            ],
        ]
        for bad in bad_values:
            with tempfile.TemporaryDirectory() as directory:
                runner = self._runner(directory, {"conditions": bad})
                with self.assertRaises(ValueError, msg=repr(bad)):
                    runner._sequential_conditions("fake")

    def test_run_executes_every_condition_even_when_the_first_fails(self):
        class FakeBackend:
            def load(self):
                return None

            def unload(self):
                return None

        with tempfile.TemporaryDirectory() as directory:
            runner = self._runner(
                directory,
                {
                    "conditions": [
                        {"perceptual_interface": "text_mediated", "memory_mode": "combined"},
                        {"perceptual_interface": "direct_visual", "memory_mode": "combined"},
                    ]
                },
            )
            runner._create_backend = lambda model_name: FakeBackend()
            ran = []

            def fake_sequence(model_name, backend, condition=None):
                ran.append((condition.perceptual_interface, condition.memory_mode))
                if condition.perceptual_interface == "text_mediated":
                    raise RuntimeError("first sequence broke")

            runner._run_sequential_extension = fake_sequence
            runner.run()
        self.assertEqual(
            ran, [("text_mediated", "combined"), ("direct_visual", "combined")]
        )

    def test_isolated_route_gets_each_condition_and_its_own_directory(self):
        calls = []

        def executor(payload):
            calls.append(payload)
            return {"outcome": "completed", "reports": [{"status": "completed"}, {"status": "completed"}]}

        with tempfile.TemporaryDirectory() as directory:
            runner = ExperimentRunner(
                {
                    "models": ["fake"],
                    "tasks": self.TASKS,
                    "output_directory": directory,
                    "run_primary_conditions": False,
                    "extensions": {
                        "sequential_tasks": {
                            "enabled": True,
                            "seed": 101,
                            "conditions": [
                                {"perceptual_interface": "text_mediated", "memory_mode": "combined"},
                                {"perceptual_interface": "direct_visual", "memory_mode": "combined"},
                            ],
                        }
                    },
                },
                episode_factory=lambda **kwargs: None,
                isolated_sequence_executor=executor,
            )
            runner._cleanup_build_processes = lambda: None
            runner._episode_specs = lambda: [
                {"id": "scene_1", "seed": 7, "scene_config": {}}
            ]
            runner.run()
        self.assertEqual(
            [c["condition"]["perceptual_interface"] for c in calls],
            ["text_mediated", "direct_visual"],
        )
        self.assertEqual(len({c["episode_id"] for c in calls}), 2)
        self.assertEqual(len({c["history_directory"] for c in calls}), 2)
        self.assertEqual({c["seed"] for c in calls}, {101})
        self.assertEqual(len(runner.results), 4)

    def _shipped_sequence(self):
        config = json.loads(
            Path(__file__).with_name("three_task_config.json").read_text(encoding="utf-8")
        )
        directory = tempfile.mkdtemp()
        config["output_directory"] = directory
        runner = ExperimentRunner(config, episode_factory=lambda **kwargs: None)
        return config, runner

    def test_shipped_three_task_config_runs_every_text_memory_mode_and_coela(self):
        config, runner = self._shipped_sequence()
        conditions = runner._sequential_conditions("fake")
        self.assertEqual(
            [(c.perceptual_interface, c.memory_mode, c.baseline) for c in conditions],
            [
                ("text_mediated", "none", "sequential_extension"),
                ("text_mediated", "episodic", "sequential_extension"),
                ("text_mediated", "semantic", "sequential_extension"),
                ("text_mediated", "combined", "sequential_extension"),
                ("coela", "combined", "coela_style_single_agent"),
            ],
        )
        self.assertEqual(
            len({runner._episode_id(c, "sequence", 101) for c in conditions}), 5
        )
        for task in config["tasks"]:
            self.assertIs(task["task_spec"]["sighting_handling"], True)

    def test_the_coela_condition_builds_the_coela_planner(self):
        from planner import CoELABaselinePlanner
        _, runner = self._shipped_sequence()
        coela = runner._sequential_conditions("fake")[-1]
        planner = runner._create_planner(coela, backend=object(), history_directory=tempfile.mkdtemp())
        self.assertIsInstance(planner, CoELABaselinePlanner)
        self.assertEqual(planner.memory_mode, "combined")

    def test_coela_with_other_memory_is_refused(self):
        config, _ = self._shipped_sequence()
        config["extensions"]["sequential_tasks"]["conditions"] = [
            {"perceptual_interface": "coela", "memory_mode": "episodic"}
        ]
        runner = ExperimentRunner(config, episode_factory=lambda **kwargs: None)
        with self.assertRaises(ValueError):
            runner._sequential_conditions("fake")

if __name__ == "__main__":
    unittest.main()


class ConfigSuiteTests(unittest.TestCase):
    def test_plain_config_stands_for_itself(self):
        from pathlib import Path
        self.assertEqual(
            expand_config_sequence("experiment_config.json"),
            [Path("experiment_config.json")],
        )

    def test_suite_rejects_missing_and_nested_entries(self):
        import os
        with tempfile.TemporaryDirectory() as directory:
            missing = os.path.join(directory, "missing.json")
            with open(missing, "w") as handle:
                json.dump({"run_configs": ["nope.json"]}, handle)
            with self.assertRaises(ValueError):
                expand_config_sequence(missing)
            inner = os.path.join(directory, "inner.json")
            with open(inner, "w") as handle:
                json.dump({"run_configs": ["x.json"]}, handle)
            outer = os.path.join(directory, "outer.json")
            with open(outer, "w") as handle:
                json.dump({"run_configs": ["inner.json"]}, handle)
            with self.assertRaises(ValueError):
                expand_config_sequence(outer)
