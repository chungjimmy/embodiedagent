import importlib
import json
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image


class NavigationGridTests(unittest.TestCase):
    def _load_scene_class(self):
        tdw = types.ModuleType("tdw")
        controller = types.ModuleType("tdw.controller")
        tdw_utils = types.ModuleType("tdw.tdw_utils")
        output_data = types.ModuleType("tdw.output_data")
        controller.Controller = type("Controller", (), {})
        tdw_utils.TDWUtils = type("TDWUtils", (), {})
        output_data.OutputData = type("OutputData", (), {})
        output_data.Rigidbodies = type("Rigidbodies", (), {})
        modules = {
            "tdw": tdw,
            "tdw.controller": controller,
            "tdw.tdw_utils": tdw_utils,
            "tdw.output_data": output_data,
        }
        with patch.dict(sys.modules, modules):
            sys.modules.pop("scene", None)
            return importlib.import_module("scene").Scene

    def test_path_sampling_rejects_wall_crossing(self):
        Scene = self._load_scene_class()
        scene = Scene(controller=None)
        x_values = [-1.0, 0.0, 1.0]
        z_values = [-1.0, 0.0, 1.0]
        positions = np.zeros((3, 3, 2), dtype=float)
        for x_index, x in enumerate(x_values):
            for z_index, z in enumerate(z_values):
                positions[x_index, z_index] = [x, z]
        occupancy = np.zeros((3, 3), dtype=int)
        occupancy[1, 1] = 1
        scene.occupancy_map = types.SimpleNamespace(
            positions=positions,
            occupancy_map=occupancy,
        )
        scene.occupancy_cell_size = 1.0
        scene.navigation_clearance = 0.01
        scene.path_sample_spacing = 0.25

        self.assertFalse(scene.is_path_navigable(-1, 0, 1, 0))
        self.assertTrue(scene.is_path_navigable(-1, -1, 1, -1))

    def test_isolated_island_status_is_still_traversable(self):
        Scene = self._load_scene_class()
        scene = Scene(controller=None)
        positions = np.zeros((2, 2, 2), dtype=float)
        positions[0, 0] = [0.0, 0.0]
        positions[0, 1] = [0.0, 0.5]
        positions[1, 0] = [0.5, 0.0]
        positions[1, 1] = [0.5, 0.5]
        occupancy = np.full((2, 2), 3, dtype=int)
        scene.occupancy_map = types.SimpleNamespace(
            positions=positions,
            occupancy_map=occupancy,
        )
        scene.occupancy_cell_size = 0.5
        scene.navigation_clearance = 0.1
        scene.path_sample_spacing = 0.1

        self.assertTrue(scene.is_navigable(0.0, 0.0, clearance=0))
        self.assertTrue(scene.is_path_navigable(0.0, 0.0, 0.5, 0.5))
        self.assertIsNotNone(scene.nearest_navigable_position(0.25, 0.25))
        self.assertEqual(
            Scene._infer_occupancy_cell_size(positions, fallback=0.25),
            0.5,
        )

    def test_safe_departure_rejects_a_free_but_boxed_in_spawn(self):
        Scene = self._load_scene_class()
        scene = Scene(controller=None)
        positions = np.zeros((3, 3, 2), dtype=float)
        for x_index, x in enumerate((-1.0, 0.0, 1.0)):
            for z_index, z in enumerate((-1.0, 0.0, 1.0)):
                positions[x_index, z_index] = [x, z]
        occupancy = np.ones((3, 3), dtype=int)
        occupancy[1, 1] = 0
        scene.occupancy_map = types.SimpleNamespace(
            positions=positions,
            occupancy_map=occupancy,
        )
        scene.occupancy_cell_size = 1.0
        scene.navigation_clearance = 0.01
        scene.path_sample_spacing = 0.25

        self.assertTrue(scene.is_navigable(0.0, 0.0, clearance=0.01))
        self.assertFalse(
            scene.has_safe_departure(0.0, 0.0, distance=1.0, clearance=0.01)
        )
        self.assertIsNone(
            scene.nearest_navigable_position(
                0.0,
                0.0,
                clearance=0.01,
                min_departure=1.0,
            )
        )

    def test_old_tdw_constructor_receives_requested_cell_size(self):
        Scene = self._load_scene_class()

        class OldOccupancyMap:
            def __init__(self, cell_size=0.5):
                self.cell_size = cell_size
                self.positions = np.array(
                    [
                        [[0.0, 0.0], [0.0, cell_size]],
                        [[cell_size, 0.0], [cell_size, cell_size]],
                    ]
                )
                self.occupancy_map = np.full((2, 2), 3, dtype=int)

            def generate(self, ignore_objects=None):
                return None

        occupancy_module = types.ModuleType("tdw.add_ons.occupancy_map")
        occupancy_module.OccupancyMap = OldOccupancyMap
        controller = types.SimpleNamespace(
            add_ons=[],
            communicate=lambda commands: [],
        )
        scene = Scene(controller=controller)
        with patch.dict(
            sys.modules,
            {"tdw.add_ons.occupancy_map": occupancy_module},
        ):
            scene.setup_navigation(cell_size=0.25)

        self.assertEqual(controller.add_ons[0].cell_size, 0.25)
        self.assertEqual(scene.occupancy_cell_size, 0.25)
        self.assertEqual(scene.occupancy_status_counts, {3: 4})

    def test_blank_frame_diagnostic_is_narrow(self):
        tdw = types.ModuleType("tdw")
        add_ons = types.ModuleType("tdw.add_ons")
        image_capture = types.ModuleType("tdw.add_ons.image_capture")
        image_capture.ImageCapture = type("ImageCapture", (), {})
        with patch.dict(
            sys.modules,
            {
                "tdw": tdw,
                "tdw.add_ons": add_ons,
                "tdw.add_ons.image_capture": image_capture,
            },
        ):
            sys.modules.pop("movement", None)
            Agent = importlib.import_module("movement").Agent

        black = Image.new("RGB", (64, 64), color=(0, 0, 0))
        varied = Image.new("RGB", (64, 64), color=(255, 255, 255))
        for x in range(32):
            for y in range(64):
                varied.putpixel((x, y), (0, 0, 0))
        self.assertTrue(Agent._frame_quality(black)["appears_blank"])
        self.assertFalse(Agent._frame_quality(varied)["appears_blank"])

    def test_camera_pitch_is_bounded_and_safe_actions_respect_limits(self):
        tdw = types.ModuleType("tdw")
        add_ons = types.ModuleType("tdw.add_ons")
        image_capture = types.ModuleType("tdw.add_ons.image_capture")
        image_capture.ImageCapture = type("ImageCapture", (), {})
        with patch.dict(
            sys.modules,
            {
                "tdw": tdw,
                "tdw.add_ons": add_ons,
                "tdw.add_ons.image_capture": image_capture,
            },
        ):
            sys.modules.pop("movement", None)
            Agent = importlib.import_module("movement").Agent

        controller = types.SimpleNamespace(communicate=lambda commands: [])
        scene = types.SimpleNamespace(
            is_path_navigable=lambda *args, **kwargs: True,
            get_object_positions=lambda: [],
        )
        agent = Agent(
            controller=controller,
            avatar_id="a",
            initial_position={"x": 0.0, "y": 1.6, "z": 0.0},
            scene=scene,
            initial_camera_pitch=0,
            camera_pitch_step=15,
            minimum_camera_pitch=-30,
            maximum_camera_pitch=30,
        )
        self.assertTrue(agent.look_down())
        self.assertEqual(agent.camera_pitch, 15)
        self.assertTrue(agent.look_down())
        self.assertEqual(agent.camera_pitch, 30)
        self.assertFalse(agent.look_down())
        safe = agent.get_collision_safe_actions(
            allowed_actions=["look_up", "look_down", "turn_right"]
        )
        self.assertIn("look_up", safe)
        self.assertNotIn("look_down", safe)
        self.assertEqual(
            agent.get_supported_camera_pitches(),
            [0.0, -15.0, 15.0, -30.0, 30.0],
        )

    def test_set_pose_teleports_and_rejects_unnavigable_cells(self):
        tdw = types.ModuleType("tdw")
        add_ons = types.ModuleType("tdw.add_ons")
        image_capture = types.ModuleType("tdw.add_ons.image_capture")
        image_capture.ImageCapture = type("ImageCapture", (), {})
        with patch.dict(
            sys.modules,
            {
                "tdw": tdw,
                "tdw.add_ons": add_ons,
                "tdw.add_ons.image_capture": image_capture,
            },
        ):
            sys.modules.pop("movement", None)
            Agent = importlib.import_module("movement").Agent

        sent = []
        controller = types.SimpleNamespace(
            communicate=lambda commands: sent.append(commands) or []
        )
        scene = types.SimpleNamespace(
            has_navigation_map=lambda: True,
            is_navigable=lambda x, z, clearance=None: x < 10,
            is_path_navigable=lambda *args, **kwargs: True,
            get_object_positions=lambda: [],
        )
        agent = Agent(
            controller=controller,
            avatar_id="a",
            initial_position={"x": 0.0, "y": 1.6, "z": 0.0},
            scene=scene,
            initial_camera_pitch=0,
        )
        snapshot = agent.set_pose({"x": -3.0, "z": 4.0}, 270)
        self.assertEqual(snapshot["position"], {"x": -3.0, "y": 1.6, "z": 4.0})
        self.assertEqual(snapshot["yaw"], -90.0)
        types_sent = [c["$type"] for batch in sent if batch for c in batch]
        self.assertIn("teleport_avatar_to", types_sent)
        self.assertIn("set_avatar_forward", types_sent)
        with self.assertRaises(ValueError):
            agent.set_pose({"x": 20.0, "z": 0.0}, 0)
        self.assertEqual(agent.get_position()["x"], -3.0)

    def test_configured_object_is_grounded_from_measured_bounds(self):
        Scene = self._load_scene_class()
        commands = []
        scene = Scene(
            controller=types.SimpleNamespace(
                communicate=lambda command: commands.append(command) or []
            )
        )
        measured = iter(
            [
                {"bottom_y": -0.40, "top_y": 0.25},
                {"bottom_y": 0.52, "top_y": 1.17},
            ]
        )
        scene._object_bounds = lambda object_id: next(measured)

        corrected, grounding = scene._ground_configured_object(
            object_id=1001,
            label="bright pink box",
            position={"x": 1.0, "y": 0.0, "z": 2.0},
            placement={
                "ground_to_floor": True,
                "surface_y": 0.0,
                "resolved_surface_y": 0.50,
                "resolved_surface_hit_object": False,
                "resolved_surface_object_id": None,
                "ground_clearance": 0.02,
            },
        )

        self.assertAlmostEqual(corrected["y"], 0.92)
        self.assertAlmostEqual(grounding["surface_y"], 0.50)
        self.assertEqual(grounding["surface_source"], "downward_raycast")
        self.assertAlmostEqual(grounding["resolved_bottom_y"], 0.52)
        self.assertEqual(commands[-1]["$type"], "teleport_object")
        self.assertAlmostEqual(commands[-1]["position"]["y"], 0.92)

    def test_task_object_scale_accepts_scalar_and_rejects_nonpositive_values(self):
        Scene = self._load_scene_class()
        self.assertEqual(
            Scene._normalized_scale_factor(3.0, "scale"),
            {"x": 3.0, "y": 3.0, "z": 3.0},
        )
        with self.assertRaises(ValueError):
            Scene._normalized_scale_factor(0, "scale")

    def test_target_placement_skips_cells_without_enough_view_poses(self):
        Scene = self._load_scene_class()
        scene = Scene(controller=None)
        scene.has_safe_departure = lambda *args, **kwargs: True

        def view_candidates(x, z, **kwargs):
            count = 2 if x == 1.0 else 4
            return [{"x": x, "z": z}] * count

        scene.target_view_candidates = view_candidates
        selected = scene._select_task_object_candidate(
            candidates=[
                {"x": 1.0, "z": 0.0, "path_distance": 6.0},
                {"x": 2.0, "z": 0.0, "path_distance": 6.25},
            ],
            selected=[],
            minimum_path_distance=6.0,
            minimum_separation=0.0,
            minimum_initial_path_distance=0.0,
            initial_distances=None,
            minimum_departure=1.0,
            clearance=0.75,
            farthest=False,
            minimum_view_poses=4,
        )

        self.assertEqual(selected["x"], 2.0)
        self.assertEqual(selected["verified_view_poses"], 4)

    def _open_room_scene(self):
        Scene = self._load_scene_class()
        scene = Scene(controller=None)
        coordinates = [-3.0 + 0.5 * index for index in range(12)]
        positions = np.zeros((12, 12, 2), dtype=float)
        for x_index, x in enumerate(coordinates):
            for z_index, z in enumerate(coordinates):
                positions[x_index, z_index] = [x, z]
        scene.occupancy_map = types.SimpleNamespace(
            positions=positions,
            occupancy_map=np.zeros((12, 12), dtype=int),
        )
        scene.occupancy_cell_size = 0.5
        scene.navigation_clearance = 0.1
        scene.path_sample_spacing = 0.25
        return scene

    @staticmethod
    def _anchored_object(**placement):
        return {
            "object_id": 7,
            "model_name": "any_model",
            "radius": 0.25,
            "placement": {
                "strategy": "nearest_reachable",
                "minimum_path_distance": 0.0,
                "minimum_departure": 0.0,
                "clearance": 0.1,
                **placement,
            },
        }

    def test_placement_anchor_pins_target_to_the_cell_nearest_the_anchor(self):
        scene = self._open_room_scene()
        start = {"x": -2.5, "y": 1.6, "z": -2.5}

        anchored = scene.resolve_task_object_positions(
            [self._anchored_object(anchor={"x": 1.0, "z": 1.1})],
            initial_position=start,
        )[0]
        self.assertEqual(anchored["position"]["x"], 1.0)
        self.assertEqual(anchored["position"]["z"], 1.0)
        self.assertEqual(anchored["placement"]["resolved_anchor"], {"x": 1.0, "z": 1.1})

        unanchored = scene.resolve_task_object_positions(
            [self._anchored_object()],
            initial_position=start,
        )[0]
        self.assertEqual(
            (unanchored["position"]["x"], unanchored["position"]["z"]),
            (-2.5, -2.5),
        )

        other_start = {"x": 2.0, "y": 1.6, "z": 2.0}
        again = scene.resolve_task_object_positions(
            [self._anchored_object(anchor={"x": 1.0, "z": 1.1})],
            initial_position=other_start,
        )[0]
        self.assertEqual(again["position"], anchored["position"])

    def test_placement_anchor_is_validated(self):
        scene = self._open_room_scene()
        start = {"x": -2.5, "y": 1.6, "z": -2.5}
        with self.assertRaises(ValueError):
            scene.resolve_task_object_positions(
                [self._anchored_object(anchor={"x": 1.0})],
                initial_position=start,
            )
        with self.assertRaises(ValueError):
            scene.resolve_task_object_positions(
                [self._anchored_object(anchor={"x": "left", "z": 1.0})],
                initial_position=start,
            )
        with self.assertRaises(ValueError):
            scene.resolve_task_object_positions(
                [
                    self._anchored_object(
                        anchor={"x": 1.0, "z": 1.0},
                        reference_object_id=3,
                    )
                ],
                initial_position=start,
            )

    def test_three_task_config_uses_one_fixed_pitch_and_no_look_actions(self):
        path = Path(__file__).with_name("three_task_config.json")
        config = json.loads(path.read_text(encoding="utf-8"))
        camera = config["scene"]["camera"]
        self.assertEqual(
            camera["initial_pitch_degrees"],
            camera["minimum_pitch_degrees"],
        )
        self.assertEqual(
            camera["initial_pitch_degrees"],
            camera["maximum_pitch_degrees"],
        )
        for task in config["tasks"]:
            actions = task["task_spec"]["allowed_actions"]
            self.assertNotIn("look_up", actions)
            self.assertNotIn("look_down", actions)


if __name__ == "__main__":
    unittest.main()
