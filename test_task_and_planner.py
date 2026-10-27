import json
import tempfile
import unittest
from pathlib import Path

from memory import MemoryManager
from planner import Planner
from task_contract import TaskContract
from vision import GenerationResult


WINDOW_SPEC = {
    "task_target": "window",
    "completion_mode": "target_visible",
    "completion_condition": "A window is clearly visible in the current view.",
    "requires_target_confirmation": True,
    "excluded_target_terms": ["door", "mirror", "reflection"],
    "allowed_actions": [
        "move_forward",
        "move_backward",
        "turn_left",
        "turn_right",
    ],
}


class FakeBackend:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.prompts = []

    def generate(self, prompt, image=None, max_new_tokens=None):
        self.prompts.append(
            {
                "prompt": prompt,
                "image": image,
                "max_new_tokens": max_new_tokens,
            }
        )
        text = self.responses.pop(0)
        return GenerationResult(
            text=text,
            input_tokens=20,
            output_tokens=20,
            inference_seconds=0.01,
            prompt_characters=len(prompt),
            visual_input_present=image is not None,
        )

    def configuration(self):
        return {"model_name": "fake"}


class FakeAgent:
    def __init__(self, output_directory):
        self.output_directory = Path(output_directory)
        self.position = {"x": 0.0, "y": 1.6, "z": 0.0}
        self.yaw = 0.0
        self.turn_angle = 45.0

    def get_current_view(self):
        return object()

    def get_state_snapshot(self):
        return {
            "position": dict(self.position),
            "yaw": self.yaw,
            "camera_angles": {"yaw": 0, "pitch": 0},
            "held_object": None,
        }

    def execute_action(self, action):
        before = self.get_state_snapshot()
        if action == "turn_right":
            self.yaw += self.turn_angle
        elif action == "turn_left":
            self.yaw -= self.turn_angle
        elif action == "move_forward":
            self.position["z"] += 1
        elif action == "move_backward":
            self.position["z"] -= 1
        after = self.get_state_snapshot()
        return {
            "outcome": "succeeded",
            "success": True,
            "state_before": before,
            "state_after": after,
            "position_changed": before["position"] != after["position"],
            "held_object_changed": False,
        }

    def save_current_view(self, image=None, prefix="final"):
        return str((self.output_directory / f"{prefix}.png").resolve())


class TaskContractTests(unittest.TestCase):
    def test_contract_accepts_generic_target_aliases(self):
        contract = TaskContract.build(
            "Find the pink box.",
            {
                "task_target": "box",
                "target_aliases": ["cube", "block"],
                "completion_mode": "target_visible",
            },
        )
        self.assertEqual(contract.target_aliases, ("cube", "block"))

    def test_contract_accepts_vertical_camera_actions(self):
        contract = TaskContract.build(
            "Find the floor-level box.",
            {
                "task_target": "box",
                "completion_mode": "target_visible",
                "allowed_actions": ["look_up", "look_down", "turn_right"],
            },
        )
        self.assertEqual(
            contract.allowed_actions,
            ("look_up", "look_down", "turn_right"),
        )

    def test_contract_is_config_driven_and_generic(self):
        contract = TaskContract.build(
            "Locate the emergency exit and stop when it is visible.",
            {
                "task_target": "emergency exit",
                "completion_mode": "target_visible",
                "allowed_actions": ["turn_left", "turn_right"],
            },
        )
        self.assertEqual(contract.task_target, "emergency exit")
        self.assertEqual(contract.completion_mode, "target_visible")
        self.assertEqual(contract.allowed_actions, ("turn_left", "turn_right"))

    def test_perception_cannot_replace_configured_target(self):
        parsed = {
            "scene_summary": "A large glass window is on the right wall.",
            "task_target": "tree trunk",
            "completion_mode": "target_near",
            "objects": [
                {
                    "name": "large window",
                    "location": "right",
                    "distance": "far",
                    "attributes": ["glass panes", "white frame"],
                }
            ],
            "target_visible": False,
        }
        contract = TaskContract.build("Find a window.", WINDOW_SPEC)
        Planner._normalize_perception(parsed, "Find a window.", contract)
        self.assertEqual(parsed["task_target"], "window")
        self.assertEqual(parsed["completion_mode"], "target_visible")
        self.assertTrue(parsed["target_visible"])
        self.assertEqual(parsed["reported_task_spec"]["task_target"], "tree trunk")

    def test_contract_keeps_explicit_target_exclusions(self):
        contract = TaskContract.build("Find a window.", WINDOW_SPEC)
        self.assertEqual(
            contract.excluded_target_terms,
            ("door", "mirror", "reflection"),
        )

    def test_contract_keeps_required_target_attributes(self):
        contract = TaskContract.build(
            "Find the blue vase.",
            {
                "task_target": "vase",
                "target_attributes": ["blue"],
                "completion_mode": "target_visible",
            },
        )
        self.assertEqual(contract.target_attributes, ("blue",))

    def test_indirect_view_phrasing_is_rejected_despite_inserted_words(self):
        exclusions = ("door", "mirror", "reflection")
        rejected = (
            "A window is visible through the open door on the right side "
            "of the room.",
            "A window can be seen through the large glass door.",
            "A window is reflected in the ornate mirror.",
        )
        for evidence in rejected:
            self.assertTrue(
                Planner._direct_visual_evidence_uses_exclusion(
                    evidence, exclusions
                ),
                f"expected exclusion to fire for: {evidence!r}",
            )

    def test_direct_target_near_an_excluded_object_is_accepted(self):
        exclusions = ("door", "mirror", "reflection")
        accepted = (
            "There is a window beside a door.",
            "A window is visible next to the bookshelf, near a closed door.",
            "A large window is directly ahead, clearly visible.",
            "A window is clearly visible, showing curtains beside a door.",
            "A window is directly visible; the door is off to the left.",
            "A window is clearly visible in the image next to a door.",
        )
        for evidence in accepted:
            self.assertFalse(
                Planner._direct_visual_evidence_uses_exclusion(
                    evidence, exclusions
                ),
                f"did not expect exclusion to fire for: {evidence!r}",
            )


class MemoryContextTests(unittest.TestCase):
    def test_prompt_context_is_bounded_and_keeps_recent_facts(self):
        memory = MemoryManager(
            episodic_window=8,
            context_episodic_limit=2,
            context_object_limit=2,
            context_pose_limit=2,
            context_fact_limit=2,
            context_text_char_limit=32,
        )
        for step in range(1, 5):
            perception = {
                "scene_summary": f"scene {step}",
                "scene_text": "x" * 48,
                "objects": [
                    {
                        "id": f"object-{step}",
                        "name": f"object {step}",
                        "location": "right",
                        "attributes": ["visible"],
                    }
                ],
                "spatial_relations": [f"relation {step}"],
                "obstacles": [f"obstacle {step}"],
                "task_progress": f"progress {step}",
            }
            state = {
                "position": {"x": float(step), "y": 1.6, "z": 0.0},
                "yaw": 0.0,
            }
            memory.record_step(
                {
                    "step": step,
                    "action": "turn_right",
                    "outcome": "succeeded",
                    "reason": f"event {step}",
                },
                perception=perception,
                agent_state=state,
            )

        context = memory.get_context("combined")
        semantic = context["semantic_memory"]
        self.assertEqual(
            [event["step"] for event in context["episodic_memory"]], [3, 4]
        )
        self.assertEqual(
            [item["name"] for item in semantic["objects"]],
            ["object 3", "object 4"],
        )
        self.assertEqual(len(semantic["visited_poses"]), 2)
        self.assertEqual(semantic["spatial_relations"], ["relation 3", "relation 4"])
        self.assertEqual(semantic["obstacles"], ["obstacle 3", "obstacle 4"])
        self.assertNotIn("last_scene_text", semantic)
        self.assertNotIn("last_scene_summary", semantic)
        self.assertNotIn("task_progress", semantic)

    def test_completed_stage_records_only_agent_observed_landmark(self):
        memory = MemoryManager(episodic_window=4, context_fact_limit=2)
        memory.record_task_boundary(
            task_id="locate_vase",
            goal="Find the vase.",
            task_contract={
                "task_target": "vase",
                "completion_mode": "target_near",
            },
            status="completed",
            agent_state={
                "position": {"x": 2.0, "y": 1.6, "z": -3.0},
                "yaw": 90.0,
            },
        )
        context = memory.get_context("combined")
        landmarks = context["semantic_memory"]["task_landmarks"]

        self.assertEqual(len(landmarks), 1)
        self.assertEqual(landmarks[0]["task_target"], "vase")
        self.assertEqual(
            landmarks[0]["position"], {"x": 2.0, "y": 1.6, "z": -3.0}
        )
        self.assertEqual(
            context["episodic_memory"][-1]["event_type"], "task_boundary"
        )

    def test_prompt_context_has_a_hard_character_budget(self):
        memory = MemoryManager(
            episodic_window=8,
            context_episodic_limit=4,
            context_object_limit=6,
            context_pose_limit=4,
            context_fact_limit=6,
            context_text_char_limit=200,
            context_prompt_char_limit=600,
        )
        for step in range(1, 5):
            memory.record_step(
                {
                    "step": step,
                    "action": "turn_right",
                    "outcome": "succeeded",
                    "reason": "r" * 200,
                },
                perception={
                    "scene_summary": "s" * 200,
                    "scene_text": "t" * 200,
                    "objects": [
                        {
                            "name": f"object {step}",
                            "location": "right",
                            "attributes": ["visible"],
                        }
                    ],
                },
                agent_state={
                    "position": {"x": float(step), "y": 1.6, "z": 0.0},
                    "yaw": 0.0,
                },
            )

        memory.get_context("combined")
        stats = memory.context_stats()
        self.assertLessEqual(stats["characters"], 600)
        self.assertEqual(stats["budget_characters"], 600)
        self.assertTrue(stats["trimmed"])

    def test_failed_stage_does_not_create_a_return_landmark(self):
        memory = MemoryManager(episodic_window=4)
        memory.record_task_boundary(
            task_id="locate_vase",
            goal="Find the vase.",
            task_contract={"task_target": "vase"},
            status="completed",
            task_success=False,
            agent_state={
                "position": {"x": 2.0, "y": 1.6, "z": -3.0},
                "yaw": 90.0,
            },
        )
        self.assertFalse(memory.has_task_landmark(task_id="locate_vase"))


class PlannerContractTests(unittest.TestCase):
    def _planner(self, directory, response, memory_mode="none"):
        backend = FakeBackend(response)
        planner = Planner(
            model_name="fake",
            backend=backend,
            perceptual_interface="direct_visual",
            memory_mode=memory_mode,
            history_directory=directory,
        )
        return planner, backend

    def test_vertical_look_guard_requires_a_move_or_turn_between_adjustments(self):
        history = [
            {
                "executed": True,
                "action": "look_down",
                "outcome": "succeeded",
            }
        ]
        available, reason = Planner._guarded_actions(
            history,
            blocked_action=None,
            allowed_actions={
                "move_forward",
                "turn_right",
                "look_up",
                "look_down",
            },
        )
        self.assertNotIn("look_up", available)
        self.assertNotIn("look_down", available)
        self.assertIn("vertical camera adjustment", reason)

    def test_search_turn_reversal_is_replaced_but_route_correction_is_allowed(self):
        first_search = (
            '{"status":"continue","target_visible":false,'
            '"target_location":"unknown","target_correction":"unknown",'
            '"visual_evidence":"No target is visible.",'
            '"action":"turn_right","action_purpose":"search",'
            '"reason":"Scan right."}'
        )
        reverse_search = (
            '{"status":"continue","target_visible":false,'
            '"target_location":"unknown","target_correction":"unknown",'
            '"visual_evidence":"No target is visible.",'
            '"action":"turn_left","action_purpose":"search",'
            '"reason":"Scan left."}'
        )
        route_correction = (
            '{"status":"continue","target_visible":false,'
            '"target_location":"unknown","target_correction":"unknown",'
            '"visual_evidence":"The previous heading was incorrect.",'
            '"action":"turn_left","action_purpose":"route_correction",'
            '"reason":"Return from the wrong route."}'
        )

        with tempfile.TemporaryDirectory() as directory:
            planner, _ = self._planner(
                directory,
                first_search,
            )
            planner.backend.responses.append(reverse_search)
            search_report = planner.run_task(
                agent=FakeAgent(directory),
                goal="Find a window.",
                max_steps=2,
                task_spec=WINDOW_SPEC,
            )

        with tempfile.TemporaryDirectory() as directory:
            planner, _ = self._planner(
                directory,
                first_search,
            )
            planner.backend.responses.append(route_correction)
            correction_report = planner.run_task(
                agent=FakeAgent(directory),
                goal="Find a window.",
                max_steps=2,
                task_spec=WINDOW_SPEC,
            )

        self.assertEqual(
            [item["action"] for item in search_report["history"]],
            ["turn_right", "turn_right"],
        )
        self.assertEqual(
            search_report["search_turn_oscillation_replacements"],
            1,
        )
        self.assertEqual(
            [item["action"] for item in correction_report["history"]],
            ["turn_right", "turn_left"],
        )
        self.assertEqual(
            correction_report["history"][1]["action_purpose"],
            "route_correction",
        )

    def test_planner_uses_the_agents_configured_turn_angle(self):
        response = (
            '{"status":"continue","target_visible":false,'
            '"target_location":"unknown","target_correction":"unknown",'
            '"visual_evidence":"No target is visible.",'
            '"action":"turn_right","action_purpose":"search",'
            '"reason":"Scan right."}'
        )
        with tempfile.TemporaryDirectory() as directory:
            planner, backend = self._planner(directory, response)
            agent = FakeAgent(directory)
            agent.turn_angle = 30.0
            report = planner.run_task(
                agent=agent,
                goal="Find a window.",
                max_steps=1,
                task_spec=WINDOW_SPEC,
            )

        self.assertEqual(report["turn_angle_degrees"], 30.0)
        self.assertEqual(report["history"][0]["state_after"]["yaw"], 30.0)
        self.assertIn("Body turns are 30 degrees", backend.prompts[0]["prompt"])

    def test_decision_uses_locked_target_even_if_model_changes_it(self):
        response = (
            '{"status":"completed","task_target":"tree trunk",'
            '"completion_mode":"target_near","target_visible":true,'
            '"target_location":"right","target_distance":"far",'
            '"target_held":false,"visual_evidence":"A large glass window is '
            'clearly visible on the right wall.","action":null,'
            '"reason":"Window found.","impossible_basis":"none",'
            '"impossible_evidence":"","area_explored":false,'
            '"exploration_evidence":"","area_mapped":false,'
            '"mapping_evidence":""}'
        )
        with tempfile.TemporaryDirectory() as directory:
            planner, _ = self._planner(directory, response)
            contract = TaskContract.build("Find a window.", WINDOW_SPEC)
            decision, _ = planner._decide(
                image=object(),
                perception=None,
                goal="Find a window.",
                memory_context=None,
                blocked_action=None,
                action_count=0,
                max_steps=20,
                current_state={},
                recent_history=[],
                task_contract=contract,
            )
        self.assertEqual(decision["status"], "completed")
        self.assertEqual(decision["task_target"], "window")
        self.assertEqual(decision["completion_mode"], "target_visible")
        self.assertEqual(
            decision["reported_task_spec"]["task_target"], "tree trunk"
        )

    def test_visible_target_schema_omits_held_and_distance_fields(self):
        response = (
            '{"status":"completed","target_visible":true,'
            '"target_location":"right","visual_evidence":"A window with '
            'glass panes is clearly visible on the right.","action":null,'
            '"reason":"The visibility condition is met.",'
            '"impossible_basis":"none","impossible_evidence":"",'
            '"area_explored":false,"exploration_evidence":"",'
            '"area_mapped":false,"mapping_evidence":""}'
        )
        with tempfile.TemporaryDirectory() as directory:
            planner, backend = self._planner(directory, response)
            contract = TaskContract.build("Find a window.", WINDOW_SPEC)
            decision, _ = planner._decide(
                image=object(),
                perception=None,
                goal="Find a window.",
                memory_context=None,
                blocked_action=None,
                action_count=0,
                max_steps=20,
                current_state={},
                recent_history=[],
                task_contract=contract,
            )
        self.assertEqual(decision["status"], "completed")
        prompt = backend.prompts[0]["prompt"]
        self.assertNotIn('"target_held"', prompt)
        self.assertNotIn('"target_distance"', prompt)
        self.assertNotIn('"area_explored"', prompt)
        self.assertNotIn('"mapping_evidence"', prompt)

    def test_required_target_attribute_rejects_decor_lookalike(self):
        spec = {
            "task_target": "vase",
            "target_attributes": ["blue"],
            "completion_mode": "target_visible",
        }
        contract = TaskContract.build("Find the blue vase.", spec)
        decor = {
            "scene_summary": "A white vase is visible on the right.",
            "objects": [
                {
                    "name": "white vase",
                    "location": "right",
                    "attributes": ["white ceramic"],
                }
            ],
            "target_visible": True,
            "target_location": "right",
        }
        Planner._normalize_perception(decor, "Find the blue vase.", contract)
        self.assertFalse(decor["target_visible"])

        target = {
            "scene_summary": "A blue ceramic vase is clearly visible on the right wall.",
            "objects": [
                {
                    "name": "blue vase",
                    "location": "right",
                    "attributes": ["blue ceramic"],
                }
            ],
            "target_visible": True,
            "target_location": "right",
        }
        Planner._normalize_perception(target, "Find the blue vase.", contract)
        self.assertTrue(target["target_visible"])

    def test_configured_blue_bag_task_confirms_the_real_step_four_output(self):
        config = json.loads(
            Path(__file__).with_name("experiment_config.json").read_text(
                encoding="utf-8"
            )
        )
        task = config["tasks"][0]
        goal = task["goal"]
        contract = TaskContract.build(goal, task["task_spec"])

        seen = {
            "scene_summary": "A blue bag is on the floor near a bookshelf.",
            "objects": [
                {
                    "name": "blue bag",
                    "location": "left",
                    "distance": "near",
                    "attributes": ["blue"],
                },
                {"name": "bookshelf", "location": "right", "distance": "unknown", "attributes": []},
            ],
            "target_visible": True,
            "target_location": "left",
            "target_distance": "near",
        }
        Planner._normalize_perception(seen, goal, contract)
        self.assertTrue(seen["target_visible"])

        lookalike = {
            "scene_summary": "A brown paper bag stands on a wooden floor next to a white wall.",
            "objects": [
                {
                    "name": "paper bag",
                    "location": "center",
                    "distance": "medium",
                    "attributes": ["brown"],
                }
            ],
            "target_visible": True,
            "target_location": "center",
        }
        Planner._normalize_perception(lookalike, goal, contract)
        self.assertFalse(lookalike["target_visible"])

    def test_excluded_look_alike_cannot_confirm_target(self):
        parsed = {
            "scene_summary": "A glass door is visible on the left.",
            "objects": [
                {
                    "name": "window door",
                    "location": "left",
                    "distance": "medium",
                    "attributes": ["glass panes"],
                }
            ],
            "target_visible": True,
            "target_location": "left",
        }
        contract = TaskContract.build("Find a window.", WINDOW_SPEC)
        Planner._normalize_perception(parsed, "Find a window.", contract)
        self.assertFalse(parsed["target_visible"])
        self.assertEqual(
            parsed["target_confirmation_source"],
            "rejected_unsupported_visibility_claim",
        )

    def test_prompt_exposes_guarded_actions_and_tracks_replacement(self):
        response = (
            '{"status":"continue","target_visible":false,'
            '"target_location":"unknown","target_distance":"unknown",'
            '"target_held":false,"visual_evidence":"No window is visible.",'
            '"action":"move_forward","reason":"Continue.",'
            '"impossible_basis":"none","impossible_evidence":"",'
            '"area_explored":false,"exploration_evidence":"",'
            '"area_mapped":false,"mapping_evidence":""}'
        )
        with tempfile.TemporaryDirectory() as directory:
            planner, backend = self._planner(
                directory, response, memory_mode="episodic"
            )
            contract = TaskContract.build("Find a window.", WINDOW_SPEC)
            decision, _ = planner._decide(
                image=object(),
                perception=None,
                goal="Find a window.",
                memory_context=None,
                blocked_action="move_forward",
                action_count=1,
                max_steps=20,
                current_state={},
                recent_history=[],
                task_contract=contract,
            )
        prompt = backend.prompts[0]["prompt"]
        available_line = next(
            line for line in prompt.splitlines()
            if line.startswith("Available actions:")
        )
        self.assertNotIn("move_forward", available_line)
        self.assertNotEqual(decision["action"], "move_forward")
        self.assertTrue(decision["model_output_valid"])
        self.assertEqual(
            decision["controller_intervention"], "guarded_action_replacement"
        )

    def test_prompt_no_memory_hides_guard_but_controller_still_enforces(self):
        response = (
            '{"status":"continue","target_visible":false,'
            '"target_location":"unknown","target_distance":"unknown",'
            '"target_held":false,"visual_evidence":"No window is visible.",'
            '"action":"move_forward","reason":"Continue.",'
            '"impossible_basis":"none","impossible_evidence":"",'
            '"area_explored":false,"exploration_evidence":"",'
            '"area_mapped":false,"mapping_evidence":""}'
        )
        with tempfile.TemporaryDirectory() as directory:
            planner, backend = self._planner(directory, response)
            contract = TaskContract.build("Find a window.", WINDOW_SPEC)
            decision, _ = planner._decide(
                image=object(),
                perception=None,
                goal="Find a window.",
                memory_context=None,
                blocked_action="move_forward",
                action_count=1,
                max_steps=20,
                current_state={},
                recent_history=[],
                task_contract=contract,
            )
        prompt = backend.prompts[0]["prompt"]
        available_line = next(
            line for line in prompt.splitlines()
            if line.startswith("Available actions:")
        )
        self.assertIn("move_forward", available_line)
        self.assertIn("Action guard: none", prompt)
        self.assertNotEqual(decision["action"], "move_forward")
        self.assertTrue(decision["model_output_valid"])
        self.assertEqual(
            decision["controller_intervention"], "guarded_action_replacement"
        )

    def test_environment_success_does_not_auto_stop_agent(self):
        response = (
            '{"status":"continue","target_visible":false,'
            '"target_location":"unknown","target_distance":"unknown",'
            '"target_held":false,"visual_evidence":"No window is visible.",'
            '"action":"turn_right","reason":"Search.",'
            '"impossible_basis":"none","impossible_evidence":"",'
            '"area_explored":false,"exploration_evidence":"",'
            '"area_mapped":false,"mapping_evidence":""}'
        )
        with tempfile.TemporaryDirectory() as directory:
            planner, _ = self._planner(directory, response)
            report = planner.run_task(
                agent=FakeAgent(directory),
                goal="Find a window.",
                max_steps=1,
                task_spec=WINDOW_SPEC,
                success_evaluator=lambda: {
                    "available": True,
                    "success": True,
                    "reason": "test criterion",
                },
            )
        self.assertEqual(report["status"], "action_limit")
        self.assertFalse(report["task_success"])
        self.assertTrue(report["goal_reached_without_stop"])

    def test_any_step_environment_success_is_reported_after_agent_moves_away(self):
        response = (
            '{"status":"continue","target_visible":false,'
            '"target_location":"unknown","target_distance":"unknown",'
            '"target_held":false,"visual_evidence":"No window is visible.",'
            '"action":"turn_right","reason":"Search.",'
            '"impossible_basis":"none","impossible_evidence":"",'
            '"area_explored":false,"exploration_evidence":"",'
            '"area_mapped":false,"mapping_evidence":""}'
        )
        evaluations = [
            {"available": True, "success": True, "reason": "reached"},
            {"available": True, "success": False, "reason": "moved away"},
        ]

        def evaluator():
            return evaluations.pop(0)

        with tempfile.TemporaryDirectory() as directory:
            planner, _ = self._planner(directory, response)
            report = planner.run_task(
                agent=FakeAgent(directory),
                goal="Find a window.",
                max_steps=1,
                task_spec=WINDOW_SPEC,
                success_evaluator=evaluator,
            )

        self.assertFalse(report["ground_truth_evaluation"]["success"])
        self.assertTrue(report["environment_success_ever"])
        self.assertEqual(report["first_environment_success_step"], 1)
        self.assertTrue(report["goal_reached_without_stop"])

    def test_environment_success_alone_does_not_trigger_near_recheck(self):
        perception_response = (
            '{"scene_summary":"A pink box fills the right side.",'
            '"objects":[{"name":"pink box","location":"right",'
            '"distance":"unknown","attributes":["pink"]}],'
            '"target_visible":true,"target_location":"right",'
            '"target_distance":"unknown","task_progress":"Target found."}'
        )
        decision_response = (
            '{"status":"continue","target_visible":false,'
            '"target_location":"unknown","target_distance":"unknown",'
            '"target_held":false,"visual_evidence":"No target.",'
            '"action":"turn_right","reason":"Keep searching.",'
            '"impossible_basis":"none","impossible_evidence":"",'
            '"area_explored":false,"exploration_evidence":"",'
            '"area_mapped":false,"mapping_evidence":""}'
        )
        task_spec = {
            "task_target": "box",
            "target_aliases": ["cube", "block"],
            "target_attributes": ["pink"],
            "completion_mode": "target_near",
            "completion_condition": "The pink box is visible and near.",
            "requires_target_confirmation": True,
            "completion_consistency": {
                "retry_on_inconsistency": True,
                "promote_validated_perception": False,
                "require_environment_success_for_completion": True,
            },
            "allowed_actions": [
                "move_forward",
                "move_backward",
                "turn_left",
                "turn_right",
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            backend = FakeBackend(
                perception_response,
                decision_response,
            )
            planner = Planner(
                model_name="fake",
                backend=backend,
                perceptual_interface="text_mediated",
                memory_mode="none",
                history_directory=directory,
            )
            report = planner.run_task(
                agent=FakeAgent(directory),
                goal="Find the pink box.",
                max_steps=1,
                task_spec=task_spec,
                success_evaluator=lambda: {
                    "available": True,
                    "success": True,
                    "independent": True,
                    "reason": "nearest distance=1.8; threshold=2.0",
                },
            )

        self.assertEqual(report["status"], "action_limit")
        self.assertEqual(report["actions_taken"], 1)
        self.assertEqual(report["completion_retry_calls"], 0)
        self.assertTrue(report["goal_reached_without_stop"])
        self.assertEqual(len(backend.prompts), 2)

    def test_perception_recheck_never_includes_environment_reason(self):
        perception_response = (
            '{"scene_summary":"A pink box is visibly near the agent.",'
            '"objects":[{"name":"pink box","location":"center",'
            '"distance":"near","attributes":["pink"]}],'
            '"target_visible":true,"target_location":"center",'
            '"target_correction":"none","target_distance":"near",'
            '"task_progress":"Target found."}'
        )
        decision_response = (
            '{"status":"continue","target_visible":false,'
            '"target_location":"unknown","target_distance":"unknown",'
            '"action":"move_forward","action_purpose":"target_approach",'
            '"reason":"Approach the target."}'
        )
        completion_response = (
            '{"status":"completed","reason":"Current visual evidence '
            'satisfies the contract."}'
        )
        task_spec = {
            "task_target": "box",
            "target_aliases": ["cube", "block"],
            "target_attributes": ["pink"],
            "completion_mode": "target_near",
            "completion_condition": "The pink box is visible and near.",
            "requires_target_confirmation": True,
            "completion_consistency": {
                "retry_on_inconsistency": True,
                "promote_validated_perception": False,
                "require_environment_success_for_completion": True,
            },
            "allowed_actions": [
                "move_forward",
                "move_backward",
                "turn_left",
                "turn_right",
            ],
        }
        private_reason = "nearest_1001_distance=1.8; threshold=2.0"
        with tempfile.TemporaryDirectory() as directory:
            backend = FakeBackend(
                perception_response,
                decision_response,
                completion_response,
            )
            planner = Planner(
                model_name="fake",
                backend=backend,
                perceptual_interface="text_mediated",
                memory_mode="none",
                history_directory=directory,
            )
            report = planner.run_task(
                agent=FakeAgent(directory),
                goal="Find the pink box.",
                max_steps=1,
                task_spec=task_spec,
                success_evaluator=lambda: {
                    "available": True,
                    "success": True,
                    "independent": True,
                    "reason": private_reason,
                },
            )

        retry_prompt = backend.prompts[-1]["prompt"]
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["completion_retry_calls"], 1)
        self.assertNotIn(private_reason, retry_prompt)
        self.assertNotIn("Independent environment check", retry_prompt)
        self.assertEqual(
            report["history"][0]["finish_check"][
                "completion_inconsistency_source"
            ],
            "perception_contract",
        )

    def test_environment_failure_rejects_false_near_completion(self):
        perception_response = (
            '{"scene_summary":"A pink box is centered across the room.",'
            '"objects":[{"name":"pink box","location":"center",'
            '"distance":"near","attributes":["pink"]}],'
            '"target_visible":true,"target_location":"center",'
            '"target_correction":"none",'
            '"target_distance":"near","task_progress":"Target found."}'
        )
        decision_response = (
            '{"status":"completed","target_visible":true,'
            '"target_location":"center","target_distance":"near",'
            '"target_held":false,"visual_evidence":"The pink box is visible.",'
            '"action":null,"reason":"The box appears near.",'
            '"impossible_basis":"none","impossible_evidence":"",'
            '"area_explored":false,"exploration_evidence":"",'
            '"area_mapped":false,"mapping_evidence":""}'
        )
        task_spec = {
            "task_target": "box",
            "target_aliases": ["cube", "block"],
            "target_attributes": ["pink"],
            "completion_mode": "target_near",
            "requires_target_confirmation": True,
            "visible_target_approach": True,
            "completion_consistency": {
                "retry_on_inconsistency": True,
                "promote_validated_perception": False,
                "require_environment_success_for_completion": True,
            },
            "allowed_actions": [
                "move_forward",
                "move_backward",
                "turn_left",
                "turn_right",
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            backend = FakeBackend(perception_response, decision_response)
            planner = Planner(
                model_name="fake",
                backend=backend,
                perceptual_interface="text_mediated",
                memory_mode="none",
                history_directory=directory,
            )
            report = planner.run_task(
                agent=FakeAgent(directory),
                goal="Find the pink box.",
                max_steps=1,
                task_spec=task_spec,
                success_evaluator=lambda: {
                    "available": True,
                    "success": False,
                    "independent": True,
                    "reason": "nearest distance=6.37; threshold=2.0",
                },
            )

        self.assertEqual(report["status"], "action_limit")
        self.assertFalse(report["task_success"])
        self.assertEqual(report["rejected_completions"], 1)
        self.assertEqual(report["history"][0]["action"], "move_forward")
        self.assertEqual(
            report["history"][0]["completion_rejection_source"],
            "environment",
        )

    def test_visible_target_approach_moves_toward_centered_target(self):
        perception_response = (
            '{"scene_summary":"A yellow box is centered ahead.",'
            '"objects":[{"name":"yellow box","location":"center",'
            '"distance":"medium","attributes":["yellow"]}],'
            '"target_visible":true,"target_location":"center",'
            '"target_correction":"none",'
            '"target_distance":"medium","task_progress":"Target found."}'
        )
        decision_response = (
            '{"status":"continue","target_visible":false,'
            '"target_location":"unknown","target_distance":"unknown",'
            '"target_held":false,"visual_evidence":"No target.",'
            '"action":"turn_right","reason":"Scan right.",'
            '"impossible_basis":"none","impossible_evidence":"",'
            '"area_explored":false,"exploration_evidence":"",'
            '"area_mapped":false,"mapping_evidence":""}'
        )
        task_spec = {
            "task_target": "box",
            "target_aliases": ["cube", "block"],
            "target_attributes": ["yellow"],
            "completion_mode": "target_near",
            "requires_target_confirmation": True,
            "visible_target_approach": True,
            "allowed_actions": [
                "move_forward",
                "move_backward",
                "turn_left",
                "turn_right",
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            backend = FakeBackend(perception_response, decision_response)
            planner = Planner(
                model_name="fake",
                backend=backend,
                perceptual_interface="text_mediated",
                memory_mode="none",
                history_directory=directory,
            )
            report = planner.run_task(
                agent=FakeAgent(directory),
                goal="Find the yellow box.",
                max_steps=1,
                task_spec=task_spec,
                success_evaluator=lambda: {
                    "available": True,
                    "success": False,
                    "independent": True,
                    "reason": "nearest distance=3.3; threshold=2.0",
                },
            )

        self.assertEqual(report["history"][0]["requested_action"], "turn_right")
        self.assertEqual(report["history"][0]["action"], "move_forward")
        self.assertEqual(report["target_approach_steps"], 1)
        self.assertEqual(report["target_approach_replacements"], 1)

    def test_fully_visible_side_target_moves_without_turning(self):
        perception_response = (
            '{"scene_summary":"A yellow box is fully visible on the right.",'
            '"objects":[{"name":"yellow box","location":"right",'
            '"distance":"medium","attributes":["yellow"]}],'
            '"target_visible":true,"target_location":"right",'
            '"target_correction":"none","target_distance":"medium",'
            '"task_progress":"Target visible."}'
        )
        decision_response = (
            '{"status":"continue","target_visible":false,'
            '"target_location":"unknown","target_distance":"unknown",'
            '"action":"turn_right","action_purpose":"target_alignment",'
            '"reason":"Turn toward the target."}'
        )
        task_spec = {
            "task_target": "box",
            "target_aliases": ["cube", "block"],
            "target_attributes": ["yellow"],
            "completion_mode": "target_near",
            "requires_target_confirmation": True,
            "visible_target_approach": True,
            "allowed_actions": [
                "move_forward",
                "move_backward",
                "turn_left",
                "turn_right",
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            backend = FakeBackend(perception_response, decision_response)
            planner = Planner(
                model_name="fake",
                backend=backend,
                perceptual_interface="text_mediated",
                memory_mode="none",
                history_directory=directory,
            )
            report = planner.run_task(
                agent=FakeAgent(directory),
                goal="Find the yellow box.",
                max_steps=1,
                task_spec=task_spec,
                success_evaluator=lambda: {
                    "available": True,
                    "success": False,
                    "independent": True,
                    "reason": "target is not yet within two meters",
                },
            )

        step = report["history"][0]
        self.assertEqual(step["requested_action"], "turn_right")
        self.assertEqual(step["action"], "move_forward")
        self.assertEqual(step["action_purpose"], "target_approach")

    def test_target_overshoot_allows_one_correction_then_moves_forward(self):
        visible_right = (
            '{"scene_summary":"The yellow box is clipped on the right.",'
            '"objects":[{"name":"yellow box","location":"right",'
            '"distance":"medium","attributes":["yellow"]}],'
            '"target_visible":true,"target_location":"right",'
            '"target_correction":"right","target_distance":"medium",'
            '"task_progress":"Target visible."}'
        )
        visible_left = (
            '{"scene_summary":"The yellow box is clipped on the left.",'
            '"objects":[{"name":"yellow box","location":"left",'
            '"distance":"medium","attributes":["yellow"]}],'
            '"target_visible":true,"target_location":"left",'
            '"target_correction":"left","target_distance":"medium",'
            '"task_progress":"Target visible."}'
        )
        model_turn = (
            '{"status":"continue","target_visible":false,'
            '"target_location":"unknown","target_distance":"unknown",'
            '"action":"turn_right","action_purpose":"search",'
            '"reason":"Search."}'
        )
        task_spec = {
            "task_target": "box",
            "target_aliases": ["cube", "block"],
            "target_attributes": ["yellow"],
            "completion_mode": "target_near",
            "requires_target_confirmation": True,
            "visible_target_approach": True,
            "allowed_actions": [
                "move_forward",
                "move_backward",
                "turn_left",
                "turn_right",
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            backend = FakeBackend(
                visible_right,
                model_turn,
                visible_left,
                model_turn,
                visible_right,
                model_turn,
            )
            planner = Planner(
                model_name="fake",
                backend=backend,
                perceptual_interface="text_mediated",
                memory_mode="none",
                history_directory=directory,
            )
            report = planner.run_task(
                agent=FakeAgent(directory),
                goal="Find the yellow box.",
                max_steps=3,
                task_spec=task_spec,
                success_evaluator=lambda: {
                    "available": True,
                    "success": False,
                    "independent": True,
                    "reason": "target is not yet within two meters",
                },
            )

        self.assertEqual(
            [item["action"] for item in report["history"]],
            ["turn_right", "turn_left", "move_forward"],
        )
        self.assertEqual(
            [item["action_purpose"] for item in report["history"]],
            ["target_alignment", "overshoot_correction", "target_approach"],
        )
        self.assertEqual(report["target_overshoot_corrections"], 1)

    def test_side_target_requires_redetection_after_turn(self):
        visible_left = (
            '{"scene_summary":"A yellow box is visible on the left.",'
            '"objects":[{"name":"yellow box","location":"left",'
            '"distance":"unknown","attributes":["yellow"]}],'
            '"target_visible":true,"target_location":"left",'
            '"target_correction":"left",'
            '"target_distance":"unknown","task_progress":"Target visible."}'
        )
        no_target = (
            '{"scene_summary":"The target is temporarily unclear.",'
            '"objects":[],"target_visible":false,'
            '"target_location":"unknown","target_distance":"unknown",'
            '"task_progress":"Target not confirmed in this frame."}'
        )
        turn_wrong_way = (
            '{"status":"continue","target_visible":false,'
            '"target_location":"unknown","target_distance":"unknown",'
            '"action":"turn_right","reason":"Search right."}'
        )
        task_spec = {
            "task_target": "box",
            "target_aliases": ["cube", "block"],
            "target_attributes": ["yellow"],
            "completion_mode": "target_near",
            "requires_target_confirmation": True,
            "visible_target_approach": True,
            "target_approach_persistence_steps": 2,
            "allowed_actions": [
                "move_forward",
                "move_backward",
                "turn_left",
                "turn_right",
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            backend = FakeBackend(
                visible_left,
                turn_wrong_way,
                no_target,
                turn_wrong_way,
            )
            planner = Planner(
                model_name="fake",
                backend=backend,
                perceptual_interface="text_mediated",
                memory_mode="none",
                history_directory=directory,
            )
            report = planner.run_task(
                agent=FakeAgent(directory),
                goal="Find the yellow box.",
                max_steps=2,
                task_spec=task_spec,
                success_evaluator=lambda: {
                    "available": True,
                    "success": False,
                    "independent": True,
                    "reason": "nearest distance=6.1; threshold=2.0",
                },
            )

        self.assertEqual(
            [item["action"] for item in report["history"]],
            ["turn_left", "turn_right"],
        )
        self.assertFalse(report["history"][0]["target_approach_persisted"])
        self.assertFalse(report["history"][1]["target_approach_persisted"])
        self.assertEqual(report["target_approach_persisted_steps"], 0)

    def test_side_turn_uses_one_targeted_reacquisition_check(self):
        visible_left = (
            '{"scene_summary":"A yellow box is visible on the left.",'
            '"objects":[{"name":"yellow box","location":"left",'
            '"distance":"medium","attributes":["yellow"]}],'
            '"target_visible":true,"target_location":"left",'
            '"target_correction":"left",'
            '"target_distance":"medium","task_progress":"Target visible."}'
        )
        no_target = (
            '{"scene_summary":"The target is not confirmed in this frame.",'
            '"objects":[],"target_visible":false,'
            '"target_location":"unknown","target_distance":"unknown",'
            '"task_progress":"Target not confirmed."}'
        )
        reacquired_center = (
            '{"scene_summary":"The yellow cube is centered ahead.",'
            '"objects":[{"name":"yellow cube","location":"center",'
            '"distance":"medium","attributes":["yellow"]}],'
            '"target_visible":true,"target_location":"center",'
            '"target_correction":"none",'
            '"target_distance":"medium"}'
        )
        turn_wrong_way = (
            '{"status":"continue","target_visible":false,'
            '"target_location":"unknown","target_distance":"unknown",'
            '"action":"turn_right","reason":"Search right."}'
        )
        task_spec = {
            "task_target": "box",
            "target_aliases": ["cube", "block"],
            "target_attributes": ["yellow"],
            "completion_mode": "target_near",
            "requires_target_confirmation": True,
            "visible_target_approach": True,
            "target_reacquisition_after_turn": True,
            "target_approach_persistence_steps": 1,
            "allowed_actions": [
                "move_forward",
                "move_backward",
                "turn_left",
                "turn_right",
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            backend = FakeBackend(
                visible_left,
                turn_wrong_way,
                no_target,
                reacquired_center,
                turn_wrong_way,
            )
            planner = Planner(
                model_name="fake",
                backend=backend,
                perceptual_interface="text_mediated",
                memory_mode="none",
                history_directory=directory,
            )
            report = planner.run_task(
                agent=FakeAgent(directory),
                goal="Find the yellow box.",
                max_steps=2,
                task_spec=task_spec,
                success_evaluator=lambda: {
                    "available": True,
                    "success": False,
                    "independent": True,
                    "reason": "target is not yet within two meters",
                },
            )

        self.assertEqual(
            [item["action"] for item in report["history"]],
            ["turn_left", "move_forward"],
        )
        self.assertTrue(
            report["history"][1]["target_reacquisition_attempted"]
        )
        self.assertTrue(
            report["history"][1]["target_reacquisition_confirmed"]
        )
        self.assertEqual(report["target_reacquisition_calls"], 1)
        self.assertEqual(report["target_reacquisition_confirmations"], 1)
        self.assertEqual(backend.prompts[3]["max_new_tokens"], 96)

    def test_centered_target_allows_only_one_persisted_forward_step(self):
        visible_center = (
            '{"scene_summary":"A yellow box is centered ahead.",'
            '"objects":[{"name":"yellow box","location":"center",'
            '"distance":"medium","attributes":["yellow"]}],'
            '"target_visible":true,"target_location":"center",'
            '"target_correction":"none",'
            '"target_distance":"medium","task_progress":"Target visible."}'
        )
        no_target = (
            '{"scene_summary":"The target is temporarily unclear.",'
            '"objects":[],"target_visible":false,'
            '"target_location":"unknown","target_distance":"unknown",'
            '"task_progress":"Target not confirmed in this frame."}'
        )
        turn_right = (
            '{"status":"continue","target_visible":false,'
            '"target_location":"unknown","target_distance":"unknown",'
            '"action":"turn_right","reason":"Search right."}'
        )
        turn_left = (
            '{"status":"continue","target_visible":false,'
            '"target_location":"unknown","target_distance":"unknown",'
            '"action":"turn_left","reason":"Search left."}'
        )
        task_spec = {
            "task_target": "box",
            "target_aliases": ["cube", "block"],
            "target_attributes": ["yellow"],
            "completion_mode": "target_near",
            "requires_target_confirmation": True,
            "visible_target_approach": True,
            "target_approach_persistence_steps": 3,
            "allowed_actions": [
                "move_forward",
                "move_backward",
                "turn_left",
                "turn_right",
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            backend = FakeBackend(
                visible_center,
                turn_right,
                no_target,
                turn_right,
                no_target,
                turn_left,
            )
            planner = Planner(
                model_name="fake",
                backend=backend,
                perceptual_interface="text_mediated",
                memory_mode="none",
                history_directory=directory,
            )
            report = planner.run_task(
                agent=FakeAgent(directory),
                goal="Find the yellow box.",
                max_steps=3,
                task_spec=task_spec,
                success_evaluator=lambda: {
                    "available": True,
                    "success": False,
                    "independent": True,
                    "reason": "target criteria not yet satisfied",
                },
            )

        self.assertEqual(
            [item["action"] for item in report["history"]],
            ["move_forward", "move_forward", "turn_left"],
        )
        self.assertTrue(report["history"][1]["target_approach_persisted"])
        self.assertFalse(report["history"][2]["target_approach_persisted"])
        self.assertEqual(report["target_approach_persisted_steps"], 1)

    def test_optional_step_image_is_recorded_relative_to_episode(self):
        response = (
            '{"status":"continue","target_visible":false,'
            '"target_location":"unknown","visual_evidence":"No target.",'
            '"action":"turn_right","reason":"Search.",'
            '"impossible_basis":"none","impossible_evidence":""}'
        )
        with tempfile.TemporaryDirectory() as directory:
            backend = FakeBackend(response)
            planner = Planner(
                model_name="fake",
                backend=backend,
                perceptual_interface="direct_visual",
                memory_mode="none",
                history_directory=directory,
                save_step_images=True,
            )
            report = planner.run_task(
                agent=FakeAgent(directory),
                goal="Find a window.",
                max_steps=1,
                task_spec=WINDOW_SPEC,
            )

        self.assertEqual(
            report["history"][0]["observation_image_path_relative"],
            "step_001.png",
        )

    def test_text_perception_of_window_allows_explicit_stop(self):
        perception_response = (
            '{"scene_summary":"A large glass window is visible on the right '
            'wall.","objects":[{"name":"window","location":"right",'
            '"distance":"far","attributes":["glass panes","white frame"]}],'
            '"spatial_relations":[],"obstacles":[],"target_visible":true,'
            '"target_location":"right","target_distance":"far",'
            '"task_progress":"Target is visible."}'
        )
        decision_response = (
            '{"status":"completed","target_visible":true,'
            '"target_location":"right","target_distance":"far",'
            '"target_held":false,"visual_evidence":"Perception clearly '
            'identifies the window on the right.","action":null,'
            '"reason":"The fixed visibility condition is met.",'
            '"impossible_basis":"none","impossible_evidence":"",'
            '"area_explored":false,"exploration_evidence":"",'
            '"area_mapped":false,"mapping_evidence":""}'
        )
        with tempfile.TemporaryDirectory() as directory:
            backend = FakeBackend(perception_response, decision_response)
            planner = Planner(
                model_name="fake",
                backend=backend,
                perceptual_interface="text_mediated",
                memory_mode="none",
                history_directory=directory,
            )
            report = planner.run_task(
                agent=FakeAgent(directory),
                goal="Find a window.",
                max_steps=1,
                task_spec=WINDOW_SPEC,
            )
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["actions_taken"], 0)
        self.assertTrue(report["history"][0]["observation"]["target_visible"])

    def test_completion_inconsistency_is_rechecked_by_same_model(self):
        perception_response = (
            '{"scene_summary":"A large glass window is visible on the right '
            'wall.","objects":[{"name":"window","location":"right",'
            '"distance":"far","attributes":["glass panes","white frame"]}],'
            '"spatial_relations":[],"obstacles":[],"target_visible":true,'
            '"target_location":"right","target_distance":"far",'
            '"task_progress":"Window is visible."}'
        )
        decision_response = (
            '{"status":"continue","target_visible":false,'
            '"target_location":"unknown","visual_evidence":"A window is '
            'visible on the right.","action":"turn_right",'
            '"reason":"Keep searching.","impossible_basis":"none",'
            '"impossible_evidence":"","area_explored":false,'
            '"exploration_evidence":"","area_mapped":false,'
            '"mapping_evidence":""}'
        )
        completion_response = (
            '{"status":"completed","reason":"The visible window satisfies '
            'the fixed condition."}'
        )
        with tempfile.TemporaryDirectory() as directory:
            backend = FakeBackend(
                perception_response,
                decision_response,
                completion_response,
            )
            planner = Planner(
                model_name="fake",
                backend=backend,
                perceptual_interface="text_mediated",
                memory_mode="none",
                history_directory=directory,
            )
            report = planner.run_task(
                agent=FakeAgent(directory),
                goal="Find a window.",
                max_steps=1,
                task_spec=WINDOW_SPEC,
            )
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["actions_taken"], 0)
        self.assertEqual(report["completion_inconsistencies"], 1)
        self.assertEqual(report["completion_retry_calls"], 1)
        self.assertEqual(report["completion_retries_resolved"], 1)
        self.assertEqual(report["completion_promotions"], 0)
        self.assertEqual(backend.prompts[-1]["max_new_tokens"], 64)

    def test_optional_completion_promotion_is_separately_labeled(self):
        perception_response = (
            '{"scene_summary":"A large glass window is visible on the right '
            'wall.","objects":[{"name":"window","location":"right",'
            '"distance":"far","attributes":["glass panes","white frame"]}],'
            '"spatial_relations":[],"obstacles":[],"target_visible":true,'
            '"target_location":"right","target_distance":"far",'
            '"task_progress":"Window is visible."}'
        )
        decision_response = (
            '{"status":"continue","target_visible":false,'
            '"target_location":"unknown","visual_evidence":"A window is '
            'visible on the right.","action":"turn_right",'
            '"reason":"Keep searching.","impossible_basis":"none",'
            '"impossible_evidence":"","area_explored":false,'
            '"exploration_evidence":"","area_mapped":false,'
            '"mapping_evidence":""}'
        )
        completion_response = (
            '{"status":"continue","reason":"Continue searching."}'
        )
        promoted_spec = {
            **WINDOW_SPEC,
            "completion_consistency": {
                "retry_on_inconsistency": True,
                "promote_validated_perception": True,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            backend = FakeBackend(
                perception_response,
                decision_response,
                completion_response,
            )
            planner = Planner(
                model_name="fake",
                backend=backend,
                perceptual_interface="text_mediated",
                memory_mode="none",
                history_directory=directory,
            )
            report = planner.run_task(
                agent=FakeAgent(directory),
                goal="Find a window.",
                max_steps=1,
                task_spec=promoted_spec,
            )
        self.assertEqual(report["status"], "completed")
        self.assertTrue(report["controller_assisted_completion"])
        self.assertFalse(report["agent_declared_success"])
        self.assertEqual(report["completion_promotions"], 1)

    def test_manual_review_is_not_counted_as_automatic_success(self):
        response = (
            '{"status":"completed","target_visible":true,'
            '"target_location":"center","target_distance":"far",'
            '"target_held":false,"visual_evidence":"A large window with glass '
            'panes is clearly visible in the center.","action":null,'
            '"reason":"Window found.","impossible_basis":"none",'
            '"impossible_evidence":"","area_explored":false,'
            '"exploration_evidence":"","area_mapped":false,'
            '"mapping_evidence":""}'
        )
        with tempfile.TemporaryDirectory() as directory:
            planner, _ = self._planner(directory, response)
            report = planner.run_task(
                agent=FakeAgent(directory),
                goal="Find a window.",
                max_steps=1,
                task_spec=WINDOW_SPEC,
                success_evaluator=lambda: {
                    "available": False,
                    "success": None,
                    "requires_manual_review": True,
                    "rubric": "Window visible.",
                    "reason": "manual review",
                },
            )
        self.assertEqual(report["status"], "completed")
        self.assertTrue(report["agent_declared_success"])
        self.assertIsNone(report["task_success"])
        self.assertTrue(report["requires_manual_review"])

    def test_malformed_output_is_separate_from_controller_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            planner, _ = self._planner(directory, "not JSON")
            report = planner.run_task(
                agent=FakeAgent(directory),
                goal="Find a window.",
                max_steps=1,
                task_spec=WINDOW_SPEC,
            )
        self.assertEqual(report["invalid_model_outputs"], 1)
        self.assertEqual(report["guarded_action_replacements"], 0)
        self.assertEqual(report["fallback_actions"], 1)

    def test_direct_visual_parser_is_not_applicable(self):
        response = (
            '{"status":"continue","target_visible":false,'
            '"target_location":"unknown","target_evidence_source":"unknown",'
            '"visual_evidence":"No window is visible in the current view.",'
            '"action":"turn_right","reason":"Search another heading.",'
            '"impossible_basis":"none","impossible_evidence":"",'
            '"area_explored":false,"exploration_evidence":"",'
            '"area_mapped":false,"mapping_evidence":""}'
        )
        with tempfile.TemporaryDirectory() as directory:
            planner, _ = self._planner(directory, response)
            report = planner.run_task(
                agent=FakeAgent(directory),
                goal="Find a window.",
                max_steps=1,
                task_spec=WINDOW_SPEC,
            )
        self.assertFalse(report["perception_parser_applicable"])
        self.assertIsNone(report["perception_parse_failures"])
        self.assertEqual(
            report["history"][0]["observation"]["parser_result"],
            "not_applicable",
        )

    def test_direct_visual_excluded_lookalike_cannot_complete(self):
        response = (
            '{"status":"completed","target_visible":true,'
            '"target_location":"right",'
            '"target_evidence_source":"excluded_lookalike",'
            '"visual_evidence":"A window appears through a mirror reflection '
            'on the right.","action":null,'
            '"reason":"The window is visible.","impossible_basis":"none",'
            '"impossible_evidence":"","area_explored":false,'
            '"exploration_evidence":"","area_mapped":false,'
            '"mapping_evidence":""}'
        )
        with tempfile.TemporaryDirectory() as directory:
            planner, _ = self._planner(directory, response)
            report = planner.run_task(
                agent=FakeAgent(directory),
                goal="Find a window.",
                max_steps=1,
                task_spec=WINDOW_SPEC,
            )
        self.assertEqual(report["status"], "action_limit")
        self.assertEqual(report["rejected_completions"], 1)
        observation = report["history"][0]["observation"]
        self.assertEqual(observation["excluded_target_terms"], WINDOW_SPEC[
            "excluded_target_terms"
        ])
        self.assertFalse(observation["target_visible"])


class DictShapedAttributeTests(unittest.TestCase):
    def test_scene_text_uses_attribute_values_not_keys(self):
        perception = {
            "scene_summary": "A pink box sits near the wall.",
            "objects": [
                {
                    "name": "box",
                    "camera_view_location": "center",
                    "distance": "near",
                    "attributes": {"color": "pink"},
                }
            ],
        }
        scene_text = Planner._scene_text_from_perception(perception)
        self.assertIn("(pink)", scene_text)
        self.assertNotIn("(color)", scene_text)

    def test_nested_attribute_values_are_flattened(self):
        values = Planner._attribute_values(
            {
                "color": "pink",
                "appearance": ["large", {"finish": "glossy"}],
            }
        )
        self.assertEqual(values, ["pink", "large", "glossy"])

    def test_dict_attribute_satisfies_required_target_attribute(self):
        perception = {
            "task_target": "box",
            "target_attributes": ["pink"],
        }
        item = {
            "name": "box",
            "location": "center",
            "attributes": {"color": "pink"},
        }
        self.assertTrue(
            Planner._target_object_matches_goal(
                "Find the pink box.", item, perception
            )
        )

    def test_perception_normalization_converts_attribute_dict_to_values(self):
        contract = TaskContract.build(
            "Find the pink box.",
            {
                "task_target": "box",
                "completion_mode": "target_visible",
                "target_attributes": ["pink"],
            },
        )
        perception = {
            "scene_summary": "A pink box is visible.",
            "objects": [
                {
                    "name": "box",
                    "camera_view_location": "center",
                    "attributes": {"color": "pink"},
                }
            ],
            "target_visible": True,
            "target_location": "center",
        }
        changed = Planner._normalize_perception(
            perception, "Find the pink box.", contract
        )
        self.assertTrue(changed)
        self.assertEqual(perception["objects"][0]["attributes"], ["pink"])

class PerceptionRetryTests(unittest.TestCase):
    def test_complete_camera_view_location_alias_does_not_retry(self):
        response = (
            '{"scene_summary":"A pink cube is visible on the right.",'
            '"objects":[{"name":"pink cube",'
            '"camera_view_location":"right","distance":"medium",'
            '"attributes":{"color":"pink"}}],'
            '"target_visible":true,"target_location":"right",'
            '"target_correction":"none","target_distance":"medium",'
            '"task_progress":"Target visible."}'
        )
        contract = TaskContract.build(
            "Find the pink box.",
            {
                "task_target": "box",
                "target_aliases": ["cube", "block"],
                "target_attributes": ["pink"],
                "completion_mode": "target_near",
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            backend = FakeBackend(response)
            planner = Planner(
                model_name="fake",
                backend=backend,
                perceptual_interface="text_mediated",
                memory_mode="none",
                history_directory=directory,
            )
            perception, calls = planner._prepare_perception(
                image=object(),
                goal="Find the pink box.",
                step=1,
                agent=FakeAgent(directory),
                task_contract=contract,
            )

        self.assertEqual(len(calls), 1)
        self.assertTrue(calls[0]["selected"])
        self.assertEqual(perception["objects"][0]["location"], "right")
        self.assertEqual(perception["target_correction"], "none")

    def test_required_perception_schema_rejects_nested_object_recovery(self):
        truncated = (
            '{"scene_summary":"Pink target visible","objects":['
            '{"name":"coat rack","attributes":{"color":"black"}},'
            '{"name":"pink cube","attributes":{"color":"pink"}}],'
            '"target_visible":true,"target'
        )
        parsed, result = Planner._parse_json(
            truncated,
            required_keys=Planner.PERCEPTION_REQUIRED_KEYS,
        )
        self.assertEqual(parsed, {})
        self.assertEqual(result, "invalid_json_schema")

    def test_malformed_perception_retries_with_compact_schema(self):
        truncated = (
            '{"scene_summary":"Pink target visible","objects":['
            '{"name":"coat rack","attributes":{"color":"black"}},'
            '{"name":"pink cube","attributes":{"color":"pink"}}],'
            '"target_visible":true,"target'
        )
        retry = (
            '{"scene_summary":"A bright pink cube is centered.",'
            '"objects":[{"name":"cube","location":"center",'
            '"distance":"near","attributes":{"color":"pink"}}],'
            '"target_visible":true,"target_location":"center",'
            '"target_distance":"near","task_progress":"Target found."}'
        )
        contract = TaskContract.build(
            "Find the bright pink box.",
            {
                "task_target": "box",
                "target_aliases": ["cube", "block"],
                "target_attributes": ["pink"],
                "completion_mode": "target_near",
            },
        )
        with tempfile.TemporaryDirectory() as directory:
            backend = FakeBackend(truncated, retry)
            planner = Planner(
                model_name="fake",
                backend=backend,
                perceptual_interface="text_mediated",
                memory_mode="none",
                history_directory=directory,
                perception_retry_max_new_tokens=224,
            )
            perception, calls = planner._prepare_perception(
                image=object(),
                goal="Find the bright pink box.",
                step=1,
                agent=FakeAgent(directory),
                task_contract=contract,
            )

        self.assertEqual(len(calls), 2)
        self.assertFalse(calls[0]["selected"])
        self.assertTrue(calls[1]["selected"])
        self.assertEqual(backend.prompts[1]["max_new_tokens"], 224)
        self.assertEqual(perception["parser_result"], "perception_retry_valid")
        self.assertTrue(perception["target_visible"])
        self.assertEqual(perception["objects"][0]["attributes"], ["pink"])


class PlannerApproachRegressionTests(unittest.TestCase):
    SPEC = {
        "task_target": "box",
        "target_aliases": ["cube", "block"],
        "target_attributes": ["pink"],
        "completion_mode": "target_near",
        "requires_target_confirmation": True,
        "visible_target_approach": True,
        "collision_aware_action_filter": True,
        "allowed_actions": [
            "move_forward", "move_backward", "turn_left", "turn_right"
        ],
        "completion_consistency": {
            "retry_on_inconsistency": True,
            "require_environment_success_for_completion": True,
        },
    }

    @staticmethod
    def _view(location="center", correction="unknown", distance="unknown"):
        return json.dumps({
            "scene_summary": "A pink box is visible across the room.",
            "objects": [{
                "name": "pink box", "location": location,
                "distance": distance, "attributes": ["pink"],
            }],
            "target_visible": False,
            "target_location": "unknown",
            "target_correction": correction,
            "target_distance": distance,
        })

    SEARCH = json.dumps({
        "status": "continue", "action": "turn_right",
        "action_purpose": "search", "reason": "Continue searching.",
        "target_visible": False,
    })

    def _run(self, directory, responses, steps=1, agent=None, success=False):
        backend = FakeBackend(*responses)
        planner = Planner(
            model_name="fake", backend=backend,
            perceptual_interface="text_mediated", memory_mode="combined",
            history_directory=directory,
        )
        report = planner.run_task(
            agent=agent or FakeAgent(directory),
            goal="Find the pink box.", max_steps=steps, task_spec=self.SPEC,
            success_evaluator=lambda: {
                "available": True, "independent": True, "success": success,
                "reason": "private evaluator evidence",
            },
        )
        return report, backend

    def test_centered_target_does_not_need_a_redundant_correction_field(self):
        for missing in (False, True):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as d:
                view = json.loads(self._view())
                if missing:
                    view.pop("target_correction")
                report, _ = self._run(d, [json.dumps(view), self.SEARCH])
            step = report["history"][0]
            self.assertTrue(step["observation"]["target_visible"])
            self.assertEqual(step["observation"]["target_correction"], "none")
            self.assertEqual(step["action"], "move_forward")
            self.assertEqual(step["action_purpose"], "target_approach")
            self.assertEqual(report["completion_retry_calls"], 0)

    def test_unknown_side_alignment_approaches_instead_of_scanning(self):
        with tempfile.TemporaryDirectory() as d:
            report, _ = self._run(d, [self._view(location="right"), self.SEARCH])
        step = report["history"][0]
        self.assertEqual(step["observation"]["target_correction"], "unknown")
        self.assertEqual(step["action"], "move_forward")
        self.assertTrue(step["target_approach_applied"])

    def test_unconfirmed_center_claim_cannot_start_a_pursuit(self):
        view = json.loads(self._view())
        view["objects"] = []
        view["scene_summary"] = "No target can be confirmed in this view."
        view["target_location"] = "center"
        with tempfile.TemporaryDirectory() as d:
            report, _ = self._run(d, [json.dumps(view), self.SEARCH])
        self.assertFalse(report["history"][0]["target_approach_applied"])
        self.assertEqual(report["history"][0]["action"], "turn_right")

    def test_progressing_pursuit_is_not_interrupted_after_three_moves(self):
        with tempfile.TemporaryDirectory() as d:
            report, _ = self._run(
                d, [self._view(), self.SEARCH] * 5, steps=5
            )
        self.assertEqual(
            [h["action"] for h in report["history"]], ["move_forward"] * 5
        )
        self.assertTrue(all(h["position_changed"] for h in report["history"]))
        self.assertEqual(report["target_approach_steps"], 5)

    def test_corrective_turn_is_not_blocked_by_search_turn_streak(self):
        with tempfile.TemporaryDirectory() as d:
            planner = Planner(
                model_name="fake", backend=FakeBackend(),
                history_directory=d,
            )
            planner._active_allowed_actions = set(self.SPEC["allowed_actions"])
            perception = json.loads(self._view(location="left", correction="left"))
            contract = TaskContract.build("Find the pink box.", self.SPEC)
            planner._normalize_perception(perception, "Find the pink box.", contract)
            history = [{
                "executed": True, "action": "turn_right", "success": True,
                "action_purpose": "search", "position_changed": False,
            }] * 6
            decision = planner._apply_movement_policy(
                decision={"status": "continue", "action": "turn_right"},
                goal="Find the pink box.", perception=perception,
                history=history, blocked_action=None, visible_target_approach=True,
            )
        self.assertEqual(decision["action"], "turn_left")
        self.assertEqual(decision["action_purpose"], "target_alignment")

    def test_centered_approach_still_respects_collision_safety(self):
        class BlockedForwardAgent(FakeAgent):
            def get_collision_safe_actions(self, allowed_actions=None):
                return ["move_backward", "turn_left", "turn_right"]

        with tempfile.TemporaryDirectory() as d:
            report, _ = self._run(
                d, [self._view(), self.SEARCH], agent=BlockedForwardAgent(d)
            )
        step = report["history"][0]
        self.assertNotEqual(step["action"], "move_forward")
        self.assertNotIn("move_forward", step["collision_safe_actions"])

    def test_environment_success_does_not_change_approach_actions(self):
        actions = []
        for success in (False, True):
            with tempfile.TemporaryDirectory() as d:
                report, _ = self._run(d, [self._view(), self.SEARCH], success=success)
            actions.append(report["history"][0]["action"])
            self.assertEqual(report["completion_retry_calls"], 0)
        self.assertEqual(actions, ["move_forward", "move_forward"])

    def test_rejected_model_recheck_is_not_reported_as_resolved(self):
        with tempfile.TemporaryDirectory() as d:
            report, _ = self._run(d, [
                self._view(distance="near"), self.SEARCH,
                '{"status":"completed","reason":"The box appears near."}',
            ])
        self.assertEqual(report["completion_retry_calls"], 1)
        self.assertEqual(report["completion_retries_resolved"], 0)
        self.assertEqual(report["rejected_completions"], 1)
        self.assertEqual(report["history"][0]["completion_retry_status"], "completed")
        self.assertFalse(report["history"][0]["completion_retry_resolved"])
        self.assertEqual(report["history"][0]["action"], "move_forward")
        self.assertFalse(report["task_success"])

    def test_successful_perception_retry_is_a_recovery_not_a_failure(self):
        with tempfile.TemporaryDirectory() as d:
            report, _ = self._run(d, ['{', self._view(), self.SEARCH])
        self.assertEqual(report["perception_parse_failures"], 0)
        calls = report["history"][0]["model_calls"]
        self.assertEqual(calls[0]["parser_result"], "invalid_json")
        self.assertEqual(calls[1]["stage"], "perception_retry")
        self.assertEqual(calls[1]["parser_result"], "valid")


class LandmarkRelevanceTests(unittest.TestCase):
    def test_landmarks_filtered_by_goal_and_expressed_relative_to_agent(self):
        from memory import MemoryManager

        memory = MemoryManager()
        memory.semantic["task_landmarks"] = [
            {"task_id": "a", "task_target": "box", "target_attributes": ["pink"],
             "position": {"x": -11.27, "y": 1.6, "z": -6.03}, "yaw": -135.0},
            {"task_id": "b", "task_target": "box", "target_attributes": ["yellow"],
             "position": {"x": 0.0, "y": 1.6, "z": 0.0}, "yaw": 0.0},
        ]
        memory.semantic["agent_state"] = {
            "position": {"x": -12.69, "y": 1.6, "z": -2.61}, "yaw": -45.0,
        }
        context = memory.semantic_context_snapshot(
            task_target="box", target_attributes=["pink"]
        )
        landmarks = context["task_landmarks"]
        self.assertEqual(len(landmarks), 1)
        self.assertEqual(landmarks[0]["task_id"], "a")
        self.assertEqual(
            landmarks[0]["relative_to_agent"],
            {"distance_m": 3.7, "turn": "left", "turn_degrees": 158},
        )


class MemoryGuidedSearchTests(unittest.TestCase):
    def _planner(self, directory, mode="combined", enabled=True):
        return Planner(
            model_name="fake", backend=FakeBackend("{}"),
            perceptual_interface="text_mediated", memory_mode=mode,
            history_directory=directory, memory_guided_search=enabled,
        )

    @staticmethod
    def _context(turn, degrees, distance, **extra):
        rel = {"distance_m": distance, "turn": turn, "turn_degrees": degrees}
        rel.update(extra)
        return {"semantic_memory": {"task_landmarks": [
            {"task_target": "box", "relative_to_agent": rel}]}}

    def test_turns_then_moves_toward_matching_remembered_pose(self):
        avail = {"move_forward", "turn_left", "turn_right"}
        with tempfile.TemporaryDirectory() as d:
            p = self._planner(d)
            act, _ = p._memory_guided_action(
                self._context("left", 158, 3.7), avail, False)
            self.assertEqual(act, "turn_left")
            act, _ = p._memory_guided_action(
                self._context("right", 10, 3.7), avail, False)
            self.assertEqual(act, "move_forward")

    def test_no_override_when_target_visible_disabled_or_no_semantic_memory(self):
        avail = {"move_forward", "turn_left", "turn_right"}
        ctx = self._context("left", 90, 3.0)
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(self._planner(d)._memory_guided_action(
                ctx, avail, True))
            self.assertIsNone(self._planner(d, enabled=False)
                              ._memory_guided_action(ctx, avail, False))
            self.assertIsNone(self._planner(d, mode="episodic")
                              ._memory_guided_action(ctx, avail, False))
            self.assertIsNone(self._planner(d)._memory_guided_action(
                {"semantic_memory": {}}, avail, False))


class EvidenceConflictCheckTests(unittest.TestCase):
    WINDOW_SPEC = {
        "task_target": "window",
        "completion_mode": "target_visible",
        "completion_condition": "At least one window is clearly identifiable.",
        "requires_target_confirmation": True,
        "excluded_target_terms": ["door", "mirror", "reflection"],
        "allowed_actions": [
            "move_forward", "move_backward", "turn_left", "turn_right",
        ],
    }
    HEDGED = (
        '{"status":"continue","action":"turn_right","action_purpose":"search",'
        '"reason":"Continue searching.",'
        '"visual_evidence":"Large window with a garden view on the right wall.",'
        '"target_visible":false,"target_location":"right",'
        '"target_correction":"unknown","target_evidence_source":"unknown"}'
    )
    NEGATED = HEDGED.replace(
        "Large window with a garden view on the right wall.",
        "No window visible on the white wall ahead.",
    )
    CONFIRMS = (
        '{"target_visible":true,"target_location":"right",'
        '"evidence":"Large window with a garden view on the wall"}'
    )
    REJECTS = (
        '{"target_visible":false,"target_location":"unknown",'
        '"evidence":"A doorway with curtains"}'
    )

    def _run(self, responses, check, max_steps=1, interface="direct_visual"):
        spec = dict(self.WINDOW_SPEC)
        if check is not None:
            spec["evidence_conflict_check"] = check
        with tempfile.TemporaryDirectory() as directory:
            backend = FakeBackend(*responses)
            planner = Planner(
                model_name="fake",
                backend=backend,
                perceptual_interface=interface,
                memory_mode="none",
                history_directory=directory,
            )
            report = planner.run_task(
                agent=FakeAgent(directory),
                goal="Find a window and stop when one is clearly visible.",
                max_steps=max_steps,
                task_spec=spec,
            )
        return report, backend

    def test_confirmed_target_completes_the_task(self):
        report, backend = self._run([self.HEDGED, self.CONFIRMS], True)
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["evidence_conflict_checks"], 1)
        self.assertEqual(report["evidence_conflict_confirmations"], 1)
        self.assertEqual(len(backend.prompts), 2)
        self.assertEqual(backend.prompts[1]["max_new_tokens"], 72)
        self.assertIsNotNone(backend.prompts[1]["image"])
        self.assertFalse(report["controller_assisted_completion"])

    def test_rejected_check_keeps_the_original_decision(self):
        report, backend = self._run([self.HEDGED, self.REJECTS], True)
        self.assertEqual(report["status"], "action_limit")
        self.assertEqual(report["evidence_conflict_checks"], 1)
        self.assertEqual(report["evidence_conflict_confirmations"], 0)
        self.assertEqual(report["history"][0]["action"], "turn_right")
        self.assertTrue(report["history"][0]["evidence_conflict_check_attempted"])
        self.assertFalse(report["history"][0]["evidence_conflict_check_confirmed"])

    def test_check_is_off_unless_the_task_enables_it(self):
        report, backend = self._run([self.HEDGED], None)
        self.assertEqual(len(backend.prompts), 1)
        self.assertEqual(report["evidence_conflict_checks"], 0)

    def test_negated_mention_does_not_trigger_a_call(self):
        report, backend = self._run([self.NEGATED], True)
        self.assertEqual(len(backend.prompts), 1)
        self.assertEqual(report["evidence_conflict_checks"], 0)

    def test_call_cap_bounds_extra_inference(self):
        report, backend = self._run(
            [self.HEDGED, self.REJECTS, self.HEDGED],
            {"enabled": True, "max_calls": 1},
            max_steps=2,
        )
        self.assertEqual(report["evidence_conflict_checks"], 1)
        self.assertEqual(len(backend.prompts), 3)

    def test_confirmation_must_still_pass_the_exclusion_rule(self):
        through_door = (
            '{"target_visible":true,"target_location":"right",'
            '"evidence":"Window visible through the open door frame"}'
        )
        report, _ = self._run([self.HEDGED, through_door], True)
        self.assertEqual(report["status"], "action_limit")
        self.assertEqual(report["evidence_conflict_confirmations"], 0)

    def test_policy_validation(self):
        with self.assertRaises(ValueError):
            Planner._evidence_conflict_policy({"evidence_conflict_check": "yes"})
        with self.assertRaises(ValueError):
            Planner._evidence_conflict_policy(
                {"evidence_conflict_check": {"max_calls": 0}}
            )
        self.assertFalse(Planner._evidence_conflict_policy({})["enabled"])


class TargetLossRecoveryTests(unittest.TestCase):
    ACTIONS = {"move_forward", "move_backward", "turn_left", "turn_right"}

    @classmethod
    def _config_task(cls):
        config = json.loads(
            Path(__file__).with_name("experiment_config.json").read_text(
                encoding="utf-8"
            )
        )
        return config["tasks"][0]

    def _planner(self, directory):
        planner = Planner(
            model_name="fake", backend=FakeBackend(), history_directory=directory
        )
        planner._active_allowed_actions = set(self.ACTIONS)
        return planner

    def _perception(self, location=None, correction="unknown"):
        task = self._config_task()
        contract = TaskContract.build(task["goal"], task["task_spec"])
        if location is None:
            view = {
                "scene_summary": "A room with white walls and a wooden floor.",
                "objects": [
                    {"name": "wall", "location": "center", "distance": "near", "attributes": ["white"]}
                ],
                "target_visible": False,
                "target_location": "unknown",
                "target_correction": "unknown",
                "target_distance": "unknown",
            }
        else:
            view = {
                "scene_summary": "A blue bag is on the floor next to a cabinet.",
                "objects": [
                    {"name": "blue bag", "location": location, "distance": "near", "attributes": ["blue"]}
                ],
                "target_visible": True,
                "target_location": location,
                "target_correction": correction,
                "target_distance": "near",
            }
        Planner._normalize_perception(view, task["goal"], contract)
        return view

    def _step(self, planner, history, perception, requested, recovery, persistence=0):
        goal = self._config_task()["goal"]
        decision = planner._apply_movement_policy(
            decision={
                "status": "continue",
                "action": requested,
                "action_purpose": "search",
                "reason": "Search.",
            },
            goal=goal,
            perception=perception,
            history=history,
            blocked_action=None,
            visible_target_approach=True,
            target_approach_persistence_steps=persistence,
            target_loss_recovery=Planner._target_loss_recovery_policy(
                {"target_loss_recovery": recovery}
            ),
        )
        action = decision["action"]
        history.append({
            "executed": True,
            "action": action,
            "success": True,
            "action_purpose": decision.get("action_purpose"),
            "position_changed": action in Planner.TRANSLATIONAL_ACTIONS,
        })
        return decision

    def _replay_until_bag_is_lost(self, planner, history, recovery):
        self._step(planner, history, self._perception("center", "none"), "move_forward", recovery, 1)
        for _ in range(3):
            self._step(planner, history, self._perception("left"), "move_forward", recovery, 1)
        self.assertEqual(
            [h["action"] for h in history], ["move_forward"] * 4
        )
        return self._step(planner, history, self._perception(None), "turn_right", recovery, 1)

    def test_default_behavior_is_unchanged_and_keeps_driving_past_the_bag(self):
        with tempfile.TemporaryDirectory() as d:
            planner, history = self._planner(d), []
            lost = self._replay_until_bag_is_lost(planner, history, False)
        self.assertEqual(lost["action"], "move_forward")
        self.assertTrue(lost.get("target_approach_persisted"))
        self.assertFalse(lost.get("target_loss_recovery_applied", False))

    def test_lost_after_a_left_sighting_turns_left_once(self):
        with tempfile.TemporaryDirectory() as d:
            planner, history = self._planner(d), []
            lost = self._replay_until_bag_is_lost(planner, history, True)
            self.assertEqual(lost["action"], "turn_left")
            self.assertEqual(lost["action_purpose"], "target_alignment")
            self.assertTrue(lost["target_loss_recovery_applied"])
            self.assertEqual(
                lost["controller_intervention"], "target_loss_side_recovery"
            )
            state = planner._target_approach_state
            self.assertEqual(state["last_action"], "turn_left")
            self.assertEqual(state["last_correction"], "left")
            self.assertFalse(state["reacquisition_attempted"])
            again = self._step(planner, history, self._perception(None), "turn_right", True)
            self.assertFalse(again.get("target_loss_recovery_applied", False))

    def test_side_sightings_refresh_the_pursuit_state_when_enabled(self):
        for recovery, expected_last in ((False, "center"), (True, "left")):
            with self.subTest(recovery=recovery), tempfile.TemporaryDirectory() as d:
                planner, history = self._planner(d), []
                self._step(planner, history, self._perception("center", "none"), "move_forward", recovery, 1)
                for _ in range(3):
                    self._step(planner, history, self._perception("left"), "move_forward", recovery, 1)
                self.assertEqual(
                    planner._target_approach_state["last_location"], expected_last
                )

    def test_no_recovery_when_the_loss_follows_a_turn(self):
        with tempfile.TemporaryDirectory() as d:
            planner, history = self._planner(d), []
            self._step(planner, history, self._perception("left", "left"), "turn_left", True)
            self.assertEqual(history[-1]["action"], "turn_left")
            lost = self._step(planner, history, self._perception(None), "turn_right", True)
        self.assertFalse(lost.get("target_loss_recovery_applied", False))

    def test_no_recovery_after_a_centered_sighting_or_a_long_gap(self):
        with tempfile.TemporaryDirectory() as d:
            planner, history = self._planner(d), []
            self._step(planner, history, self._perception("center", "none"), "move_forward", True)
            lost = self._step(planner, history, self._perception(None), "turn_right", True)
            self.assertFalse(lost.get("target_loss_recovery_applied", False))
        with tempfile.TemporaryDirectory() as d:
            planner, history = self._planner(d), []
            self._step(planner, history, self._perception("left"), "move_forward", True)
            for _ in range(3):
                history.append({
                    "executed": True, "action": "move_forward", "success": True,
                    "action_purpose": "other", "position_changed": True,
                })
            lost = self._step(planner, history, self._perception(None), "turn_right", True)
        self.assertFalse(lost.get("target_loss_recovery_applied", False))

    def test_recovery_respects_the_collision_filter(self):
        with tempfile.TemporaryDirectory() as d:
            planner, history = self._planner(d), []
            self._step(planner, history, self._perception("left"), "move_forward", True)
            planner._collision_safe_actions = {"move_forward", "turn_right"}
            lost = self._step(planner, history, self._perception(None), "turn_right", True)
        self.assertNotEqual(lost["action"], "turn_left")
        self.assertFalse(lost.get("target_loss_recovery_applied", False))

    def _search_after_a_sighting(self, planner, history, recovery):
        self._step(planner, history, self._perception("center", "none"), "move_forward", recovery)
        first = self._step(planner, history, self._perception(None), "turn_right", recovery)
        second = self._step(planner, history, self._perception(None), "turn_left", recovery)
        third = self._step(planner, history, self._perception(None), "turn_left", recovery)
        return first, second, third

    def test_default_guard_still_replaces_the_reversal(self):
        with tempfile.TemporaryDirectory() as d:
            _, second, _ = self._search_after_a_sighting(self._planner(d), [], False)
        self.assertEqual(second["action"], "turn_right")
        self.assertTrue(second["search_turn_oscillation_replaced"])

    def test_one_reversal_is_allowed_after_a_recent_sighting(self):
        with tempfile.TemporaryDirectory() as d:
            first, second, third = self._search_after_a_sighting(
                self._planner(d), [], True
            )
        self.assertEqual(first["action"], "turn_right")
        self.assertEqual(second["action"], "turn_left")
        self.assertTrue(second["recent_sighting_reversal_allowed"])
        self.assertFalse(second.get("search_turn_oscillation_replaced", False))
        self.assertEqual(third["action"], "turn_left")
        self.assertFalse(third.get("recent_sighting_reversal_allowed", False))

    def test_the_reversal_allowance_is_bounded(self):
        with tempfile.TemporaryDirectory() as d:
            planner, history = self._planner(d), []
            self._search_after_a_sighting(planner, history, True)
            flip = self._step(planner, history, self._perception(None), "turn_right", True)
        self.assertEqual(flip["action"], "turn_left")
        self.assertTrue(flip["search_turn_oscillation_replaced"])

    def test_no_reversal_allowance_without_a_sighting_or_outside_the_window(self):
        with tempfile.TemporaryDirectory() as d:
            planner, history = self._planner(d), []
            self._step(planner, history, self._perception(None), "turn_right", True)
            blocked = self._step(planner, history, self._perception(None), "turn_left", True)
        self.assertEqual(blocked["action"], "turn_right")
        self.assertTrue(blocked["search_turn_oscillation_replaced"])
        with tempfile.TemporaryDirectory() as d:
            planner, history = self._planner(d), []
            self._step(planner, history, self._perception("center", "none"), "move_forward", True)
            for _ in range(7):
                history.append({
                    "executed": True, "action": "move_forward", "success": True,
                    "action_purpose": "other", "position_changed": True,
                })
            self._step(planner, history, self._perception(None), "turn_right", True)
            late = self._step(planner, history, self._perception(None), "turn_left", True)
        self.assertEqual(late["action"], "turn_right")
        self.assertTrue(late["search_turn_oscillation_replaced"])

    def test_policy_validation_and_default_off(self):
        self.assertFalse(Planner._target_loss_recovery_policy({})["enabled"])
        self.assertFalse(
            Planner._target_loss_recovery_policy({"target_loss_recovery": False})["enabled"]
        )
        self.assertTrue(
            Planner._target_loss_recovery_policy({"target_loss_recovery": True})["enabled"]
        )
        tuned = Planner._target_loss_recovery_policy(
            {"target_loss_recovery": {"reversal_window_actions": 3}}
        )
        self.assertTrue(tuned["enabled"])
        self.assertEqual(tuned["reversal_window_actions"], 3)
        self.assertEqual(tuned["max_actions_since_sighting"], 2)
        for bad in ("yes", {"enabled": "no"}, {"reversal_window_actions": 0},
                    {"max_actions_since_sighting": True}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                Planner._target_loss_recovery_policy({"target_loss_recovery": bad})

    def test_shipped_configs_state_the_option_explicitly_and_task_two_keeps_it_off(self):
        shipped = self._config_task()["task_spec"]
        self.assertIsInstance(shipped["target_loss_recovery"], bool)
        Planner._target_loss_recovery_policy(shipped)
        sequence = json.loads(
            Path(__file__).with_name("three_task_config.json").read_text(encoding="utf-8")
        )
        for task in sequence["tasks"]:
            spec = task.get("task_spec", task)
            self.assertFalse(Planner._target_loss_recovery_policy(spec)["enabled"])

    def test_full_run_reports_the_recovery(self):
        spec = {
            "task_target": "box", "target_aliases": ["cube", "block"],
            "target_attributes": ["pink"], "completion_mode": "target_near",
            "requires_target_confirmation": True, "visible_target_approach": True,
            "collision_aware_action_filter": True,
            "allowed_actions": ["move_forward", "move_backward", "turn_left", "turn_right"],
            "target_loss_recovery": True,
        }
        seen = json.dumps({
            "scene_summary": "A pink box is visible across the room.",
            "objects": [{"name": "pink box", "location": "left", "distance": "unknown", "attributes": ["pink"]}],
            "target_visible": True, "target_location": "left",
            "target_correction": "unknown", "target_distance": "unknown",
        })
        lost = json.dumps({
            "scene_summary": "A white wall.",
            "objects": [{"name": "wall", "location": "center", "distance": "near", "attributes": ["white"]}],
            "target_visible": False, "target_location": "unknown",
            "target_correction": "unknown", "target_distance": "unknown",
        })
        search = json.dumps({
            "status": "continue", "action": "turn_right", "action_purpose": "search",
            "reason": "Continue searching.", "target_visible": False,
        })
        with tempfile.TemporaryDirectory() as d:
            planner = Planner(
                model_name="fake", backend=FakeBackend(seen, search, lost, search),
                perceptual_interface="text_mediated", memory_mode="none",
                history_directory=d,
            )
            report = planner.run_task(
                agent=FakeAgent(d), goal="Find the pink box.", max_steps=2, task_spec=spec,
                success_evaluator=lambda: {
                    "available": True, "independent": True, "success": False,
                    "reason": "private evaluator evidence",
                },
            )
        self.assertEqual([h["action"] for h in report["history"]], ["move_forward", "turn_left"])
        self.assertEqual(report["target_loss_recoveries"], 1)
        self.assertTrue(report["history"][1]["target_loss_recovery_applied"])


class TargetLossRecoveryFullRunTests(unittest.TestCase):
    SPEC = {
        "task_target": "box", "target_aliases": ["cube", "block"],
        "target_attributes": ["pink"], "completion_mode": "target_near",
        "requires_target_confirmation": True, "visible_target_approach": True,
        "target_reacquisition_after_turn": True,
        "target_approach_persistence_steps": 1,
        "collision_aware_action_filter": True,
        "allowed_actions": ["move_forward", "move_backward", "turn_left", "turn_right"],
        "target_loss_recovery": True,
    }

    @staticmethod
    def _view(location):
        if location is None:
            return json.dumps({
                "scene_summary": "A white wall.",
                "objects": [{"name": "wall", "location": "center", "distance": "near", "attributes": ["white"]}],
                "target_visible": False, "target_location": "unknown",
                "target_correction": "unknown", "target_distance": "unknown",
            })
        return json.dumps({
            "scene_summary": "A pink box is visible across the room.",
            "objects": [{"name": "pink box", "location": location, "distance": "unknown", "attributes": ["pink"]}],
            "target_visible": True, "target_location": location,
            "target_correction": "unknown", "target_distance": "unknown",
        })

    SEARCH = json.dumps({
        "status": "continue", "action": "turn_right", "action_purpose": "search",
        "reason": "Continue searching.", "target_visible": False,
    })

    def _run(self, responses, steps):
        with tempfile.TemporaryDirectory() as d:
            planner = Planner(
                model_name="fake", backend=FakeBackend(*responses),
                perceptual_interface="text_mediated", memory_mode="none",
                history_directory=d,
            )
            return planner.run_task(
                agent=FakeAgent(d), goal="Find the pink box.", max_steps=steps,
                task_spec=self.SPEC,
                success_evaluator=lambda: {
                    "available": True, "independent": True, "success": False,
                    "reason": "private evaluator evidence",
                },
            )

    def test_target_reappearing_after_the_recovery_turn_is_approached(self):
        report = self._run(
            [
                self._view("left"), self.SEARCH,
                self._view(None), self.SEARCH,
                self._view("center"), self.SEARCH,
            ],
            steps=3,
        )
        self.assertEqual(
            [h["action"] for h in report["history"]],
            ["move_forward", "turn_left", "move_forward"],
        )
        self.assertEqual(report["target_loss_recoveries"], 1)
        self.assertEqual(report["target_reacquisition_calls"], 0)
        self.assertEqual(report["history"][2]["action_purpose"], "target_approach")

    def test_a_still_missing_target_gets_one_reacquisition_check_then_normal_search(self):
        report = self._run(
            [
                self._view("left"), self.SEARCH,
                self._view(None), self.SEARCH,
                self._view(None),
                self._view(None),
                self.SEARCH,
            ],
            steps=3,
        )
        actions = [h["action"] for h in report["history"]]
        self.assertEqual(actions[:2], ["move_forward", "turn_left"])
        self.assertEqual(report["target_reacquisition_calls"], 1)
        self.assertEqual(report["target_loss_recoveries"], 1)
        self.assertEqual(actions[2], "turn_right")


class SideFlipRejectionTests(unittest.TestCase):
    ACTIONS = TargetLossRecoveryTests.ACTIONS
    _config_task = TargetLossRecoveryTests._config_task
    _planner = TargetLossRecoveryTests._planner
    _perception = TargetLossRecoveryTests._perception
    _step = TargetLossRecoveryTests._step

    def _approach_with_left_sightings(self, planner, history, recovery):
        self._step(planner, history, self._perception("center", "none"), "move_forward", recovery, 1)
        for _ in range(3):
            self._step(planner, history, self._perception("left"), "move_forward", recovery, 1)

    def test_a_flipped_sighting_is_rejected_and_the_earlier_side_is_used(self):
        with tempfile.TemporaryDirectory() as d:
            planner, history = self._planner(d), []
            self._approach_with_left_sightings(planner, history, True)
            flipped = self._step(
                planner, history, self._perception("right"), "move_forward", True, 1
            )
        self.assertTrue(flipped["target_side_flip_rejected"])
        self.assertEqual(flipped["action"], "turn_left")
        self.assertTrue(flipped["target_loss_recovery_applied"])

    def test_without_the_option_a_flipped_sighting_is_trusted(self):
        with tempfile.TemporaryDirectory() as d:
            planner, history = self._planner(d), []
            self._approach_with_left_sightings(planner, history, False)
            flipped = self._step(
                planner, history, self._perception("right"), "move_forward", False, 1
            )
        self.assertFalse(flipped.get("target_side_flip_rejected", False))
        self.assertEqual(flipped["action"], "move_forward")

    def test_a_turn_between_sightings_makes_the_new_side_legitimate(self):
        with tempfile.TemporaryDirectory() as d:
            planner, history = self._planner(d), []
            self._approach_with_left_sightings(planner, history, True)
            history.append({
                "executed": True, "action": "turn_left", "success": True,
                "action_purpose": "target_alignment", "position_changed": False,
            })
            seen = self._step(planner, history, self._perception("right"), "move_forward", True, 1)
        self.assertFalse(seen.get("target_side_flip_rejected", False))
        self.assertEqual(seen["action"], "move_forward")
        self.assertNotIn("target_loss_recovery_applied", {k for k, v in seen.items() if v is True})

    def test_center_to_side_and_same_side_sightings_are_never_flips(self):
        with tempfile.TemporaryDirectory() as d:
            planner, history = self._planner(d), []
            self._step(planner, history, self._perception("center", "none"), "move_forward", True, 1)
            right = self._step(planner, history, self._perception("right"), "move_forward", True, 1)
            again = self._step(planner, history, self._perception("right"), "move_forward", True, 1)
        self.assertFalse(right.get("target_side_flip_rejected", False))
        self.assertFalse(again.get("target_side_flip_rejected", False))

    def test_an_old_sighting_outside_the_window_is_not_compared(self):
        with tempfile.TemporaryDirectory() as d:
            planner, history = self._planner(d), []
            self._step(planner, history, self._perception("left"), "move_forward", True, 1)
            for _ in range(Planner.SIDE_FLIP_WINDOW_ACTIONS):
                history.append({
                    "executed": True, "action": "move_forward", "success": True,
                    "action_purpose": "other", "position_changed": True,
                })
            seen = self._step(planner, history, self._perception("right"), "move_forward", True, 1)
        self.assertFalse(seen.get("target_side_flip_rejected", False))


class SightingHandlingTests(unittest.TestCase):
    SETTINGS = (
        "accept_unlocated", "request_horizontal_location",
        "request_distance_estimate", "ignore_contradictory_correction",
    )

    FLOOR = json.dumps({
        "status": "continue", "action": "turn_right", "action_purpose": "search",
        "reason": "scan right to find the blue bag",
        "visual_evidence": "blue bag visible on the floor",
        "target_visible": True, "target_location": "on the floor",
        "target_correction": "unknown", "target_distance": "unknown",
        "target_evidence_source": "direct view",
    })
    LEFT_WALL = json.dumps({
        "status": "continue", "action": "turn_right", "action_purpose": "search",
        "reason": "scan right to find blue bag",
        "visual_evidence": "blue bag visible on left wall",
        "target_visible": True, "target_location": "left wall",
        "target_correction": "turn_right", "target_distance": "near",
        "target_evidence_source": "direct_view",
    })
    RECHECK_COMPLETED = (
        '```json\n{\n  "status": "completed",\n  "reason": "The blue bag is '
        'visible on the left wall, which satisfies the completion condition."'
    )
    SEARCH = json.dumps({
        "status": "continue", "action": "turn_right", "action_purpose": "search",
        "reason": "search for blue bag, turn right",
        "visual_evidence": "window on right side of room",
        "target_visible": False, "target_location": "unknown",
        "target_correction": "turn right", "target_distance": "unknown",
        "target_evidence_source": "unknown",
    })
    NEGATED = json.dumps({
        "status": "continue", "action": "turn_right", "action_purpose": "search",
        "reason": "turn right", "visual_evidence": "Blue bag not visible, turned right.",
        "target_visible": True, "target_location": "floor",
        "target_correction": "unknown", "target_distance": "unknown",
        "target_evidence_source": "unknown",
    })
    DONE_FAR_AWAY = json.dumps({
        "status": "completed", "action": None, "action_purpose": "other",
        "reason": "bag is near", "visual_evidence": "blue bag visible on the floor",
        "target_visible": True, "target_location": "floor",
        "target_correction": "unknown", "target_distance": "near",
        "target_evidence_source": "direct_view",
    })

    TEXT_UNLOCATED = json.dumps({
        "scene_summary": "Blue handbag visible on the floor ahead.",
        "objects": [{"name": "handbag", "location": "unknown", "distance": "far",
                     "attributes": ["blue"]}],
        "target_visible": True, "target_location": "unknown",
        "target_correction": "unknown", "target_distance": "far",
        "task_progress": "bag seen",
    })
    TEXT_NEGATED = json.dumps({
        "scene_summary": "Blue bag not visible, turned right.",
        "objects": [], "target_visible": True, "target_location": "unknown",
        "target_correction": "unknown", "target_distance": "unknown",
        "task_progress": "searching",
    })
    TEXT_LEFT_RIGHT_CORRECTION = json.dumps({
        "scene_summary": "Blue handbag visible on the left wall shelf.",
        "objects": [{"name": "handbag", "location": "left", "distance": "far",
                     "attributes": ["blue"]}],
        "target_visible": True, "target_location": "left",
        "target_correction": "right", "target_distance": "far",
        "task_progress": "bag seen",
    })
    TEXT_NOTHING = json.dumps({
        "scene_summary": "A living room with a sofa and a window.",
        "objects": [], "target_visible": False, "target_location": "unknown",
        "target_correction": "unknown", "target_distance": "unknown",
        "task_progress": "searching",
    })
    TEXT_DECISION = json.dumps({
        "status": "continue", "action": "turn_right", "action_purpose": "search",
        "reason": "search", "target_visible": True, "target_location": "unknown",
        "target_correction": "unknown", "target_distance": "unknown",
        "target_evidence_source": "unknown",
    })

    @staticmethod
    def _task():
        config = json.loads(
            Path(__file__).with_name("experiment_config.json").read_text(
                encoding="utf-8"
            )
        )
        return config["tasks"][0]

    def _run(self, responses, option, steps, interface="direct_visual",
             policy="model_distance"):
        task = self._task()
        spec = dict(task["task_spec"])
        spec["completion_policy"] = policy
        spec["claim_on_evidence_text"] = False
        if option is None:
            spec.pop("sighting_handling", None)
        else:
            spec["sighting_handling"] = option
        backend = FakeBackend(*responses)
        with tempfile.TemporaryDirectory() as d:
            planner = Planner(
                model_name="fake", backend=backend, perceptual_interface=interface,
                memory_mode="none", history_directory=d,
            )
            report = planner.run_task(
                agent=FakeAgent(d), goal=task["goal"], max_steps=steps, task_spec=spec,
                success_evaluator=lambda: {
                    "available": True, "independent": True, "success": False,
                    "reason": "private evaluator evidence",
                },
            )
        return report, backend

    def _run_text(self, perception, option, decision=None):
        return self._run(
            [perception, decision or self.TEXT_DECISION], option, 1, "text_mediated"
        )

    def test_policy_defaults_to_off_and_true_enables_every_setting(self):
        off = Planner._sighting_handling_policy({})
        self.assertEqual(set(off), set(self.SETTINGS))
        self.assertFalse(any(off.values()))
        self.assertFalse(any(Planner._sighting_handling_policy(None).values()))
        on = Planner._sighting_handling_policy({"sighting_handling": True})
        self.assertTrue(all(on.values()))
        partial = Planner._sighting_handling_policy(
            {"sighting_handling": {"accept_unlocated": True}}
        )
        self.assertEqual(
            [key for key, value in partial.items() if value], ["accept_unlocated"]
        )

    def test_policy_rejects_malformed_values(self):
        for bad in ("yes", 1, ["accept_unlocated"], {"accept_unlocated": "yes"},
                    {"accept_unlocated": 1}, {"not_a_setting": True}):
            with self.assertRaises(ValueError):
                Planner._sighting_handling_policy({"sighting_handling": bad})

    def test_the_old_interface_specific_key_is_refused_not_silently_ignored(self):
        for value in (True, False, {"accept_unlocated": True}):
            with self.assertRaises(ValueError) as caught:
                Planner._sighting_handling_policy({"direct_visual_sightings": value})
            self.assertIn("sighting_handling", str(caught.exception))
        with self.assertRaises(ValueError):
            Planner._sighting_handling_policy(
                {"direct_visual_sightings": True, "sighting_handling": True}
            )

    def test_shipped_configs_turn_every_setting_on_for_every_task(self):
        shipped = self._task()["task_spec"]
        self.assertIs(shipped["sighting_handling"], True)
        self.assertNotIn("direct_visual_sightings", shipped)
        self.assertTrue(all(Planner._sighting_handling_policy(shipped).values()))
        for name in ("three_task_config.json", "experiment_config_smoke.json"):
            config = json.loads(
                Path(__file__).with_name(name).read_text(encoding="utf-8")
            )
            for task in config["tasks"]:
                spec = task.get("task_spec", task)
                self.assertNotIn("direct_visual_sightings", spec, (name, task.get("id")))
                self.assertTrue(
                    all(Planner._sighting_handling_policy(spec).values()),
                    (name, task.get("id")),
                )

    def test_every_setting_reads_the_same_for_every_interface(self):
        with tempfile.TemporaryDirectory() as d:
            planners = [
                Planner(
                    model_name="fake", backend=FakeBackend(),
                    perceptual_interface=interface, memory_mode="none",
                    history_directory=d,
                )
                for interface in ("text_mediated", "direct_visual")
            ]
            policy = Planner._sighting_handling_policy({"sighting_handling": True})
            for planner in planners:
                planner._sighting_handling = dict(policy)
                for key in self.SETTINGS:
                    self.assertTrue(planner._sighting_setting(key), key)

    @staticmethod
    def _source(evidence, location="unknown", allowed=True, visible=True):
        return {
            "task_target": "bag", "target_aliases": ["handbag", "purse", "satchel"],
            "target_attributes": ["blue"], "excluded_target_terms": ["brown"],
            "completion_mode": "target_near", "requires_target_confirmation": True,
            "target_visible": visible, "target_location": location,
            "visual_evidence": evidence, "scene_summary": evidence,
            "direct_visual_evidence": True, "unlocated_sighting_allowed": allowed,
        }

    def test_a_flagged_sighting_without_a_horizontal_position_needs_the_setting(self):
        goal = "Find the blue bag."
        text = "blue bag visible on the floor"
        self.assertTrue(Planner._has_textual_target_evidence(goal, self._source(text)))
        self.assertFalse(
            Planner._has_textual_target_evidence(goal, self._source(text, allowed=False))
        )

    def test_unlocated_sightings_still_need_the_flag_the_attribute_and_a_positive_statement(self):
        goal = "Find the blue bag."
        check = lambda **kw: Planner._has_textual_target_evidence(goal, self._source(**kw))
        self.assertFalse(check(evidence="blue bag visible on the floor", visible=False))
        self.assertFalse(check(evidence="bag visible on the floor right now"))
        self.assertFalse(check(evidence="blue bag visible"))
        self.assertFalse(check(evidence="Blue bag not visible, turned right."))
        self.assertFalse(check(evidence="rug and coat rack on left, window"))
        self.assertTrue(check(evidence="small blue handbag on the floor"))

    def test_a_located_sighting_is_unchanged_by_the_setting(self):
        goal = "Find the blue bag."
        source = self._source("blue bag visible on left wall", location="left")
        self.assertTrue(Planner._has_textual_target_evidence(goal, source))
        source["unlocated_sighting_allowed"] = False
        self.assertTrue(Planner._has_textual_target_evidence(goal, source))

    def test_floor_sighting_is_discarded_without_the_setting(self):
        report, _ = self._run([self.FLOOR], option=False, steps=1)
        step = report["history"][0]
        self.assertFalse(step["observation"]["target_visible"])
        self.assertEqual(step["action"], "turn_right")
        self.assertEqual(report["unlocated_sighting_approaches"], 0)

    def test_floor_sighting_is_approached_with_the_setting(self):
        report, _ = self._run([self.FLOOR], option=True, steps=1)
        step = report["history"][0]
        self.assertTrue(step["observation"]["target_visible"])
        self.assertEqual(step["action"], "move_forward")
        self.assertEqual(step["action_purpose"], "target_approach")
        self.assertTrue(step["unlocated_sighting_approach"])
        self.assertEqual(step["requested_action"], "turn_right")
        self.assertEqual(report["unlocated_sighting_approaches"], 1)

    def test_one_blind_forward_step_follows_then_normal_search_resumes(self):
        report, _ = self._run([self.FLOOR, self.SEARCH, self.SEARCH], option=True, steps=3)
        self.assertEqual(
            [h["action"] for h in report["history"]],
            ["move_forward", "move_forward", "turn_right"],
        )
        self.assertEqual(
            report["history"][1]["controller_intervention"], "target_approach_persistence"
        )

    def test_completion_claimed_too_far_away_keeps_approaching(self):
        report, _ = self._run([self.DONE_FAR_AWAY], option=True, steps=1)
        step = report["history"][0]
        self.assertEqual(step["action"], "move_forward")
        self.assertEqual(step["action_purpose"], "target_approach")

    def test_completion_claim_without_the_setting_still_becomes_a_search_turn(self):
        report, _ = self._run([self.DONE_FAR_AWAY], option=False, steps=1)
        self.assertEqual(report["history"][0]["action"], "turn_right")

    def test_contradictory_left_sighting_with_a_right_correction_approaches(self):
        with_setting, _ = self._run(
            [self.LEFT_WALL, self.RECHECK_COMPLETED], option=True, steps=1
        )
        step = with_setting["history"][0]
        self.assertEqual(step["action"], "move_forward")
        self.assertTrue(step["target_correction_contradiction_ignored"])
        without, _ = self._run(
            [self.LEFT_WALL, self.RECHECK_COMPLETED], option=False, steps=1
        )
        self.assertEqual(without["history"][0]["action"], "turn_right")
        self.assertFalse(without["history"][0]["target_correction_contradiction_ignored"])

    def test_a_negated_floor_response_is_not_approached(self):
        report, _ = self._run([self.NEGATED], option=True, steps=1)
        step = report["history"][0]
        self.assertFalse(step["observation"]["target_visible"])
        self.assertEqual(step["action"], "turn_right")

    def test_the_last_seen_side_still_drives_recovery_after_a_located_sighting(self):
        report, _ = self._run(
            [self.LEFT_WALL, self.RECHECK_COMPLETED, self.SEARCH], option=True, steps=2
        )
        self.assertEqual(
            [h["action"] for h in report["history"]], ["move_forward", "turn_left"]
        )
        self.assertEqual(
            report["history"][1]["controller_intervention"], "target_loss_side_recovery"
        )

    def test_text_perception_without_a_side_is_discarded_without_the_setting(self):
        report, _ = self._run_text(self.TEXT_UNLOCATED, option=False)
        step = report["history"][0]
        self.assertFalse(step["observation"]["target_visible"])
        self.assertEqual(step["action"], "turn_right")
        self.assertEqual(report["unlocated_sighting_approaches"], 0)

    def test_text_perception_without_a_side_is_approached_with_the_setting(self):
        report, _ = self._run_text(self.TEXT_UNLOCATED, option=True)
        step = report["history"][0]
        self.assertTrue(step["observation"]["target_visible"])
        self.assertEqual(step["action"], "move_forward")
        self.assertEqual(step["action_purpose"], "target_approach")
        self.assertTrue(step["unlocated_sighting_approach"])
        self.assertEqual(step["requested_action"], "turn_right")
        self.assertEqual(report["unlocated_sighting_approaches"], 1)

    def test_a_negated_text_perception_is_not_approached(self):
        report, _ = self._run_text(self.TEXT_NEGATED, option=True)
        step = report["history"][0]
        self.assertFalse(step["observation"]["target_visible"])
        self.assertEqual(step["action"], "turn_right")

    def test_text_perception_with_a_contradictory_correction_approaches(self):
        decision = json.dumps({
            "status": "continue", "action": "turn_right", "action_purpose": "search",
            "reason": "search", "target_visible": True, "target_location": "left",
            "target_correction": "right", "target_distance": "far",
            "target_evidence_source": "direct_view",
        })
        with_setting, _ = self._run_text(
            self.TEXT_LEFT_RIGHT_CORRECTION, option=True, decision=decision
        )
        step = with_setting["history"][0]
        self.assertEqual(step["action"], "move_forward")
        self.assertTrue(step["target_correction_contradiction_ignored"])
        without, _ = self._run_text(
            self.TEXT_LEFT_RIGHT_CORRECTION, option=False, decision=decision
        )
        self.assertEqual(without["history"][0]["action"], "turn_right")
        self.assertFalse(without["history"][0]["target_correction_contradiction_ignored"])

    @staticmethod
    def _decision_prompt(backend):
        return backend.prompts[-1]["prompt"]

    @staticmethod
    def _field_notes(prompt):
        marker = "Field notes:"
        start = prompt.index(marker)
        return prompt[start:prompt.index("\n", start)]

    def _decision_prompts(self, option):
        _, direct = self._run([self.SEARCH], option=option, steps=1)
        _, text = self._run(
            [self.TEXT_NOTHING, self.TEXT_DECISION], option=option, steps=1,
            interface="text_mediated",
        )
        return self._decision_prompt(direct), self._decision_prompt(text)

    def test_the_decision_prompt_lists_the_allowed_values_only_when_asked(self):
        for prompt in self._decision_prompts(True):
            self.assertIn('"target_location":"left|center|right|unknown"', prompt)
            self.assertIn('"target_distance":"near|medium|far|unknown"', prompt)
            self.assertIn("never a surface such as floor or wall", prompt)
        for prompt in self._decision_prompts(False):
            self.assertIn('"target_location":"unknown"', prompt)
            self.assertIn('"target_distance":"unknown"', prompt)
            self.assertNotIn("Field notes", prompt)

    def test_both_interfaces_get_the_same_field_notes(self):
        direct, text = self._decision_prompts(True)
        self.assertEqual(self._field_notes(direct), self._field_notes(text))

    def test_each_setting_adds_only_its_own_prompt_text(self):
        only_location = {"request_horizontal_location": True}
        only_distance = {"request_distance_estimate": True}
        for interface, responses in (
            ("direct_visual", [self.SEARCH]),
            ("text_mediated", [self.TEXT_NOTHING, self.TEXT_DECISION]),
        ):
            _, backend = self._run(responses, only_location, 1, interface)
            prompt = self._decision_prompt(backend)
            self.assertIn('"target_location":"left|center|right|unknown"', prompt, interface)
            self.assertIn("target_location is where", prompt, interface)
            self.assertIn('"target_distance":"unknown"', prompt, interface)
            self.assertNotIn("target_distance", self._field_notes(prompt), interface)
            _, backend = self._run(responses, only_distance, 1, interface)
            prompt = self._decision_prompt(backend)
            self.assertIn('"target_distance":"near|medium|far|unknown"', prompt, interface)
            self.assertNotIn("Field notes", prompt, interface)
            self.assertIn('"target_location":"unknown"', prompt, interface)
            self.assertNotIn("target_location is where", prompt, interface)

    def test_field_notes_never_state_the_evaluator_threshold(self):
        for prompt in self._decision_prompts(True):
            notes = self._field_notes(prompt).lower()
            self.assertNotRegex(notes, r"\d")
            self.assertNotIn("meter", notes)
            self.assertNotIn(" m ", notes)

    def test_the_setting_leaves_the_text_perception_prompt_unchanged(self):
        _, with_setting = self._run_text(self.TEXT_NOTHING, True)
        _, without = self._run_text(self.TEXT_NOTHING, False)
        self.assertEqual(
            with_setting.prompts[0]["prompt"], without.prompts[0]["prompt"]
        )


class CompletionPolicyTests(unittest.TestCase):
    DIRECT_MEDIUM_CONTINUE = json.dumps({
        "status": "continue", "action": "turn_right", "action_purpose": "search",
        "reason": "scan right side for the blue bag",
        "visual_evidence": "blue bag partially visible on right side",
        "target_visible": True, "target_location": "right",
        "target_correction": "unknown", "target_distance": "medium",
        "target_evidence_source": "unknown",
    })
    DIRECT_MEDIUM_COMPLETED = json.dumps({
        "status": "completed", "action": None, "action_purpose": "other",
        "reason": "the blue bag is right here",
        "visual_evidence": "blue bag visible on the right side",
        "target_visible": True, "target_location": "right",
        "target_correction": "unknown", "target_distance": "medium",
        "target_evidence_source": "direct_view",
    })
    TRUNCATED_RECHECK = (
        '```json\n{\n  "status": "completed",\n  "reason": "The blue bag is '
        'directly visible in the current camera view and is near the agent'
    )
    TEXT_MEDIUM = json.dumps({
        "scene_summary": "Blue handbag on the floor to the left.",
        "objects": [{"name": "handbag", "location": "left", "distance": "medium",
                     "attributes": ["blue"]}],
        "target_visible": True, "target_location": "left",
        "target_correction": "unknown", "target_distance": "medium",
        "task_progress": "bag seen",
    })
    TEXT_DECISION = json.dumps({
        "status": "continue", "action": "turn_right", "action_purpose": "search",
        "reason": "keep searching", "target_visible": True,
        "target_location": "left", "target_correction": "unknown",
        "target_distance": "medium", "target_evidence_source": "direct_view",
    })
    SEARCH = json.dumps({
        "status": "continue", "action": "turn_right", "action_purpose": "search",
        "reason": "search for the blue bag", "visual_evidence": "a wall",
        "target_visible": False, "target_location": "unknown",
        "target_correction": "unknown", "target_distance": "unknown",
        "target_evidence_source": "unknown",
    })

    @staticmethod
    def _config(name="experiment_config.json"):
        return json.loads(Path(__file__).with_name(name).read_text(encoding="utf-8"))

    @staticmethod
    def _evaluation(in_range, visible=True):
        return {
            "available": True, "independent": True,
            "success": bool(in_range and visible), "criterion_type": "all",
            "criteria": [
                {"criterion_type": "object_visibility", "success": visible,
                 "visible_pixels": 45020 if visible else 0, "target_object_id": 2001},
                {"criterion_type": "object_proximity", "success": in_range,
                 "distance": 1.8 if in_range else 5.5, "target_object_id": 2001},
            ],
        }

    def _run(self, responses, policy, steps=1, interface="direct_visual",
             in_range=True, keep_planner=False, text_claims=False):
        task = self._config()["tasks"][0]
        spec = dict(task["task_spec"])
        spec["completion_policy"] = policy
        spec["claim_on_evidence_text"] = text_claims
        backend = FakeBackend(*responses)
        with tempfile.TemporaryDirectory() as d:
            planner = Planner(
                model_name="fake", backend=backend, perceptual_interface=interface,
                memory_mode="episodic", history_directory=d,
                completion_max_new_tokens=64,
            )
            report = planner.run_task(
                agent=FakeAgent(d), goal=task["goal"], max_steps=steps, task_spec=spec,
                success_evaluator=lambda: self._evaluation(in_range),
            )
        return (report, backend, planner) if keep_planner else (report, backend)

    def test_code_default_is_the_original_rule(self):
        self.assertEqual(Planner.DEFAULT_COMPLETION_POLICY, "model_distance")
        gate = {"require_environment_success_for_completion": True}
        self.assertEqual(Planner._completion_policy_setting({}, gate), "model_distance")
        self.assertEqual(Planner._completion_policy_setting(None), "model_distance")
        self.assertEqual(
            Planner._completion_policy_setting({"completion_policy": "assisted"}, gate),
            "assisted",
        )

    def test_unknown_policy_and_assisted_without_the_gate_are_refused(self):
        with self.assertRaises(ValueError):
            Planner._completion_policy_setting({"completion_policy": "loose"})
        with self.assertRaises(ValueError):
            Planner._completion_policy_setting(
                {"completion_policy": "assisted"},
                {"require_environment_success_for_completion": False},
            )

    def test_shipped_configs_use_assisted_with_the_gate_and_a_64_token_recheck(self):
        for name in ("experiment_config.json", "experiment_config_smoke.json",
                     "three_task_config.json"):
            config = self._config(name)
            self.assertEqual(config["completion_max_new_tokens"], 64, name)
            for task in config["tasks"]:
                spec = task["task_spec"]
                self.assertEqual(spec["completion_policy"], "assisted", (name, task["id"]))
                self.assertTrue(
                    spec["completion_consistency"][
                        "require_environment_success_for_completion"
                    ],
                    (name, task["id"]),
                )

    def test_model_claim_with_medium_distance_is_accepted_only_when_assisted(self):
        assisted, _ = self._run([self.DIRECT_MEDIUM_COMPLETED], "assisted")
        self.assertEqual(assisted["status"], "completed")
        self.assertTrue(assisted["task_success"])
        original, _ = self._run([self.DIRECT_MEDIUM_COMPLETED], "model_distance")
        self.assertNotEqual(original["status"], "completed")
        self.assertEqual(original["completion_claims_rejected"], 1)

    def test_recheck_fires_on_a_medium_sighting_only_when_assisted(self):
        assisted, backend = self._run(
            [self.DIRECT_MEDIUM_CONTINUE, self.TRUNCATED_RECHECK], "assisted"
        )
        self.assertEqual(assisted["status"], "completed")
        self.assertTrue(assisted["task_success"])
        self.assertEqual(assisted["completion_resolution"], "recheck_resolved")
        self.assertEqual(
            assisted["history"][-1]["finish_check"]["completion_retry_parser_result"],
            "truncated_json_recovered",
        )
        self.assertEqual(len(backend.prompts), 2)
        original, backend = self._run(
            [self.DIRECT_MEDIUM_CONTINUE, self.TRUNCATED_RECHECK], "model_distance"
        )
        self.assertNotEqual(original["status"], "completed")
        self.assertEqual(len(backend.prompts), 1)

    def test_text_mediated_gets_the_same_treatment(self):
        responses = [self.TEXT_MEDIUM, self.TEXT_DECISION, self.TRUNCATED_RECHECK]
        assisted, _ = self._run(responses, "assisted", interface="text_mediated")
        self.assertEqual(assisted["status"], "completed")
        original, backend = self._run(responses, "model_distance",
                                      interface="text_mediated")
        self.assertNotEqual(original["status"], "completed")
        self.assertEqual(len(backend.prompts), 2)

    def test_the_gate_still_decides_proximity_under_assisted(self):
        report, _ = self._run(
            [self.DIRECT_MEDIUM_COMPLETED], "assisted", in_range=False
        )
        self.assertNotEqual(report["status"], "completed")
        step = report["history"][0]
        self.assertEqual(step["completion_rejection_source"], "environment")
        self.assertEqual(step["action"], "move_forward")
        self.assertEqual(report["completion_claims"], 1)
        self.assertEqual(report["completion_claims_rejected_by_environment"], 1)
        self.assertEqual(report["rejected_claim_rate"], 1.0)

    def test_claim_counts_include_the_accepted_claim(self):
        report, _ = self._run(
            [self.DIRECT_MEDIUM_COMPLETED, self.DIRECT_MEDIUM_COMPLETED],
            "assisted", steps=2,
        )
        self.assertEqual(report["completion_claims"], 1)
        self.assertEqual(report["completion_claims_rejected"], 0)
        self.assertEqual(report["rejected_claim_rate"], 0.0)
        self.assertEqual(report["completion_policy"], "assisted")

    def test_truncated_status_recovery(self):
        recover = Planner._recover_truncated_status
        self.assertEqual(recover(self.TRUNCATED_RECHECK), "completed")
        self.assertEqual(recover('```json\n{"status": "continue", "rea'), "continue")
        self.assertIsNone(recover('```json\n{"stat'))
        self.assertIsNone(recover('{"status": "compl'))

    def test_gate_rejection_wording_never_reaches_memory(self):
        report, _, planner = self._run(
            [self.DIRECT_MEDIUM_COMPLETED], "assisted", in_range=False,
            keep_planner=True,
        )
        self.assertIn("gate rejected", report["history"][0]["reason"])
        stored = list(planner.memory.episodic)[-1]
        self.assertEqual(stored["reason"], "the blue bag is right here")
        self.assertNotIn("gate", json.dumps(stored).lower())
        self.assertNotIn("gate", planner.memory.semantic["task_progress"].lower())

    def test_both_interfaces_receive_the_same_visibility_definition(self):
        rules = Planner.TARGET_VISIBILITY_RULES
        _, direct = self._run([self.SEARCH], "assisted")
        _, text = self._run(
            [self.TEXT_MEDIUM.replace("true", "false"), self.SEARCH], "assisted",
            interface="text_mediated", in_range=False,
        )
        self.assertIn(rules, direct.prompts[0]["prompt"])
        self.assertIn(rules, text.prompts[0]["prompt"])
        self.assertNotIn(rules, text.prompts[1]["prompt"])
        self.assertIn("Do not return target_visible false while also", rules)
        self.assertIn("even when it is far away", rules)

    def test_no_prompt_states_a_distance_in_meters(self):
        _, direct = self._run([self.SEARCH], "assisted")
        _, text = self._run(
            [self.TEXT_MEDIUM.replace("true", "false"), self.SEARCH], "assisted",
            interface="text_mediated", in_range=False,
        )
        for prompt in [p["prompt"] for p in direct.prompts + text.prompts]:
            self.assertNotRegex(prompt.lower(), r"two meters|\d+(\.\d+)?\s*(m|meters?)\b")

    def test_visible_frames_the_planner_missed_are_counted_offline(self):
        report, _ = self._run(
            [self.SEARCH, self.DIRECT_MEDIUM_CONTINUE, self.SEARCH],
            "model_distance", steps=2, in_range=False,
        )
        self.assertEqual(report["object_id_visible_frames"], 2)
        self.assertEqual(report["object_id_visible_frames_confirmed"], 1)
        self.assertEqual(report["object_id_visible_frames_missed"], 1)


class PartialVisibilityAndMemoryHoldTests(unittest.TestCase):
    LEFT = json.dumps({
        "status": "continue", "action": "move_forward",
        "action_purpose": "target_approach", "reason": "approach",
        "visual_evidence": "blue bag on the floor on the left side",
        "target_visible": True, "target_location": "left",
        "target_correction": "unknown", "target_distance": "far",
        "target_evidence_source": "direct_view",
    })

    def test_partial_rule_is_part_of_the_shared_definition(self):
        rules = Planner.TARGET_VISIBILITY_RULES
        self.assertTrue(rules.startswith(Planner.TARGET_VISIBILITY_RULES_BASE))
        self.assertIn(Planner.TARGET_PARTIAL_VISIBILITY_RULE, rules)
        self.assertIn("only partly in view", rules)
        self.assertIn("every required\nattribute", rules)
        self.assertIn("says nothing about\ndistance", rules)
        self.assertNotRegex(rules.lower(), r"\d+(\.\d+)?\s*(m|meters?)\b")

    def test_hold_setting_defaults_off_and_validates(self):
        self.assertEqual(Planner._memory_guided_hold_setting({}), 0)
        self.assertEqual(Planner._memory_guided_hold_setting(None), 0)
        self.assertEqual(
            Planner._memory_guided_hold_setting({"memory_guided_search_hold_actions": 2}), 2
        )
        for bad in (-1, 1.5, "2", True):
            with self.assertRaises(ValueError):
                Planner._memory_guided_hold_setting(
                    {"memory_guided_search_hold_actions": bad}
                )

    def test_shipped_configs_hold_for_two_actions(self):
        for name in ("experiment_config.json", "experiment_config_smoke.json",
                     "three_task_config.json"):
            config = CompletionPolicyTests._config(name)
            for task in config["tasks"]:
                self.assertEqual(
                    task["task_spec"]["memory_guided_search_hold_actions"], 2,
                    (name, task["id"]),
                )

    def _run(self, hold, memory_mode="semantic"):
        task = CompletionPolicyTests._config()["tasks"][0]
        spec = dict(task["task_spec"])
        spec["memory_guided_search_hold_actions"] = hold
        backend = FakeBackend(self.LEFT, *([CompletionPolicyTests.SEARCH] * 6))
        with tempfile.TemporaryDirectory() as d:
            planner = Planner(
                model_name="fake", backend=backend,
                perceptual_interface="direct_visual", memory_mode=memory_mode,
                history_directory=d,
            )
            return planner.run_task(
                agent=FakeAgent(d), goal=task["goal"], max_steps=6, task_spec=spec,
                success_evaluator=lambda: CompletionPolicyTests._evaluation(
                    False, visible=False
                ),
            )

    def test_memory_guided_search_waits_two_actions_after_a_sighting(self):
        without = self._run(0)
        held = self._run(2)
        guided = lambda report: [
            h.get("controller_intervention") == "memory_guided_search"
            for h in report["history"]
        ]
        self.assertEqual(guided(without)[:3], [False, False, True])
        self.assertEqual(
            [h["memory_guided_search_held"] for h in held["history"]][:4],
            [False, True, True, False],
        )
        self.assertEqual(guided(held)[:4], [False, False, False, True])
        self.assertEqual(held["memory_guided_search_holds"], 2)
        self.assertEqual(without["memory_guided_search_holds"], 0)

    def test_the_side_loss_recovery_still_acts_during_the_hold(self):
        held = self._run(2)
        self.assertEqual(
            held["history"][1]["controller_intervention"], "target_loss_side_recovery"
        )


class ReplayVisibilityHelperTests(unittest.TestCase):
    def test_memory_variants(self):
        from replay_visibility import memory_variant
        logged = {
            "memory_mode": "combined",
            "episodic_memory": [{"step": 1, "action": "turn_right",
                                 "target_visible": False, "reason": "search"}],
            "semantic_memory": {"objects": []},
        }
        self.assertEqual(memory_variant(logged, "logged"), logged)
        self.assertIsNone(memory_variant(logged, "none"))
        self.assertNotIn("semantic_memory", memory_variant(logged, "episodic_only"))
        self.assertNotIn("episodic_memory", memory_variant(logged, "semantic_only"))
        stripped = memory_variant(logged, "episodic_without_target_fields")
        self.assertEqual(stripped["episodic_memory"], [{"step": 1, "action": "turn_right"}])
        self.assertEqual(logged["episodic_memory"][0]["reason"], "search")
        self.assertEqual(
            memory_variant({"memory_mode": "episodic", "episodic_memory": []}, "semantic_only"),
            "not_applicable",
        )

    def test_missed_frames_are_visible_but_unconfirmed(self):
        from replay_visibility import missed_frames
        def step(n, visible, confirmed):
            return {"step": n, "observation": {"target_visible": confirmed},
                    "environment_success_check": {"criteria": [
                        {"criterion_type": "object_visibility", "success": visible}]}}
        report = {"history": [step(1, False, False), step(2, True, True),
                              step(3, True, False), step(4, False, True)]}
        self.assertEqual(missed_frames(report), [3])


class StatusNormalizationTests(unittest.TestCase):
    COMPLETE = '{\n  "status": "complete",\n  "reason": "The blue bag is visible and near."\n}'

    def test_synonym_mapping(self):
        self.assertEqual(Planner._normalize_status_word("complete"), ("completed", True))
        self.assertEqual(Planner._normalize_status_word("completed"), ("completed", False))
        self.assertEqual(Planner._normalize_status_word("continue"), ("continue", False))
        self.assertEqual(Planner._normalize_status_word("finished"), ("finished", False))
        self.assertEqual(
            Planner._recover_truncated_status('```json\n{"status": "complete", "rea'),
            "completed",
        )

    def test_a_complete_recheck_is_read_as_completed_and_counted(self):
        run = CompletionPolicyTests()
        report, backend = run._run(
            [CompletionPolicyTests.DIRECT_MEDIUM_CONTINUE, self.COMPLETE], "assisted"
        )
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["status_normalizations"], 1)
        calls = report["history"][-1]["model_calls"]
        self.assertTrue(any('"complete"' in c["raw_response"] for c in calls))


class MemoryDestinationTests(unittest.TestCase):
    @staticmethod
    def _planner(policy, memory_mode="combined"):
        planner = Planner(
            model_name="fake", backend=FakeBackend(), perceptual_interface="text_mediated",
            memory_mode=memory_mode, history_directory=tempfile.mkdtemp(),
        )
        planner._memory_guided_destination = policy
        planner._active_turn_angle_degrees = 45.0
        return planner

    @staticmethod
    def _context(landmark_turn="left", sighting_turn="right",
                 sighting_distance=3.0, landmark_distance=4.0, at_pose=False):
        return {"semantic_memory": {
            "task_landmarks": [{
                "task_id": "locate_pink_box", "position": {"x": 1.0, "z": 2.0},
                "relative_to_agent": {"distance_m": landmark_distance,
                                      "turn_degrees": 90, "turn": landmark_turn,
                                      "at_pose": at_pose},
            }],
            "target_sighting": {
                "step": 3, "agent_position": {"x": 5.0, "z": 5.0},
                "relative_to_agent": {"distance_m": sighting_distance,
                                      "turn_degrees": 90, "turn": sighting_turn,
                                      "at_pose": False},
            },
        }}

    ACTIONS = {"turn_left", "turn_right", "move_forward"}

    def _guide(self, planner, context, executed_count=10):
        decision = {}
        result = planner._memory_guided_action(
            memory_context=context, available=self.ACTIONS,
            perception_confirms_target=False, executed_count=executed_count,
            decision=decision,
        )
        return result, decision

    def test_latest_pose_lets_a_sighting_override_the_landmark(self):
        result, _ = self._guide(self._planner("latest_pose"), self._context())
        self.assertEqual(result[0], "turn_right")

    def test_landmark_first_ignores_a_stale_sighting(self):
        planner = self._planner("landmark_first")
        planner._last_confirmed_sighting_count = 2
        result, decision = self._guide(planner, self._context(), executed_count=10)
        self.assertEqual(result[0], "turn_left")
        self.assertEqual(decision["memory_guided_destination"], "landmark")

    def test_landmark_first_uses_a_fresh_sighting_for_reacquisition(self):
        planner = self._planner("landmark_first")
        planner._last_confirmed_sighting_count = 7
        result, decision = self._guide(planner, self._context(), executed_count=10)
        self.assertEqual(result[0], "turn_right")
        self.assertEqual(decision["memory_guided_destination"], "sighting")

    def test_a_pose_visited_without_success_is_abandoned(self):
        planner = self._planner("landmark_first")
        context = self._context(landmark_distance=0.4, at_pose=True)
        outcomes = [self._guide(planner, context) for _ in range(4)]
        self.assertEqual([o[0][0] if o[0] else None for o in outcomes],
                         ["turn_left", "turn_left", None, None])
        self.assertTrue(outcomes[2][1]["memory_destination_abandoned"])
        self.assertNotIn("memory_destination_abandoned", outcomes[3][1])

    def test_without_a_landmark_or_fresh_sighting_memory_does_not_steer(self):
        planner = self._planner("landmark_first", memory_mode="semantic")
        context = self._context()
        context["semantic_memory"]["task_landmarks"] = []
        result, _ = self._guide(planner, context)
        self.assertIsNone(result)

    def test_setting(self):
        self.assertEqual(Planner._memory_guided_destination_setting({}), "latest_pose")
        self.assertEqual(
            Planner._memory_guided_destination_setting(
                {"memory_guided_destination": "landmark_first"}),
            "landmark_first",
        )
        with self.assertRaises(ValueError):
            Planner._memory_guided_destination_setting({"memory_guided_destination": "x"})
        for name in ("experiment_config.json", "experiment_config_smoke.json",
                     "three_task_config.json"):
            for task in CompletionPolicyTests._config(name)["tasks"]:
                self.assertEqual(
                    task["task_spec"]["memory_guided_destination"], "landmark_first"
                )


class EvidenceTextClaimTests(unittest.TestCase):
    PARTIAL_IN_VIEW = json.dumps({
        "status": "continue", "action": "turn_right", "action_purpose": "search",
        "reason": "search for the blue bag",
        "visual_evidence": "blue bag partially visible on the right side",
        "target_visible": False, "target_location": "right",
        "target_correction": "unknown", "target_distance": "near",
        "target_evidence_source": "unknown",
    })
    COMPLETED_FLAG_OFF = json.dumps({
        "status": "completed", "action": None, "action_purpose": "other",
        "reason": "the bag is here",
        "visual_evidence": "blue bag on the left side of the image",
        "target_visible": False, "target_location": "left",
        "target_correction": "unknown", "target_distance": "near",
        "target_evidence_source": "unknown",
    })
    RECHECK_YES = '{"status": "completed", "reason": "The blue bag is visible."}'

    def _run(self, responses, text_claims=True, in_range=True, interface="direct_visual"):
        return CompletionPolicyTests()._run(
            responses, "assisted", interface=interface, in_range=in_range,
            text_claims=text_claims,
        )

    def _variant(self, **changes):
        data = json.loads(self.PARTIAL_IN_VIEW)
        data.update(changes)
        return json.dumps(data)

    def test_setting_requires_the_assisted_policy(self):
        self.assertFalse(Planner._claim_on_evidence_text_setting({}, "assisted"))
        self.assertTrue(Planner._claim_on_evidence_text_setting(
            {"claim_on_evidence_text": True}, "assisted"))
        with self.assertRaises(ValueError):
            Planner._claim_on_evidence_text_setting(
                {"claim_on_evidence_text": True}, "model_distance")
        with self.assertRaises(ValueError):
            Planner._claim_on_evidence_text_setting(
                {"claim_on_evidence_text": "yes"}, "assisted")

    def test_shipped_configs(self):
        for name in ("experiment_config.json", "experiment_config_smoke.json",
                     "three_task_config.json"):
            for task in CompletionPolicyTests._config(name)["tasks"]:
                self.assertIs(task["task_spec"]["claim_on_evidence_text"], True)
        self.assertEqual(CompletionPolicyTests._config()["max_steps"], 30)

    def test_text_names_the_target_in_range_and_the_episode_completes(self):
        report, backend = self._run([self.PARTIAL_IN_VIEW, self.RECHECK_YES])
        self.assertEqual(report["status"], "completed")
        self.assertTrue(report["task_success"])
        self.assertEqual(report["evidence_text_claims"], 1)
        self.assertEqual(report["evidence_text_claims_accepted"], 1)
        self.assertEqual(len(backend.prompts), 2)

    def test_without_the_setting_no_claim_is_made(self):
        report, backend = self._run(
            [self.PARTIAL_IN_VIEW, self.RECHECK_YES], text_claims=False
        )
        self.assertNotEqual(report["status"], "completed")
        self.assertEqual(len(backend.prompts), 1)
        self.assertEqual(report["evidence_text_claims"], 0)

    def test_the_gate_still_decides_and_a_rejected_text_claim_keeps_the_models_action(self):
        report, _ = self._run([self.PARTIAL_IN_VIEW, self.RECHECK_YES], in_range=False)
        self.assertNotEqual(report["status"], "completed")
        step = report["history"][0]
        self.assertEqual(step["completion_rejection_source"], "environment")
        self.assertTrue(step["evidence_text_claim"])
        self.assertEqual(step["action"], "turn_right")
        self.assertEqual(report["evidence_text_claims"], 1)
        self.assertEqual(report["evidence_text_claims_accepted"], 0)
        self.assertFalse(step["observation"]["target_visible"])

    def test_negated_or_attribute_free_text_never_claims(self):
        for evidence in ("blue bag not visible, turned right",
                         "bag partially visible on the right side",
                         "rug and cabinet visible"):
            report, backend = self._run(
                [self._variant(visual_evidence=evidence), self.RECHECK_YES]
            )
            self.assertNotEqual(report["status"], "completed", evidence)
            self.assertEqual(len(backend.prompts), 1, evidence)

    def test_the_models_own_completed_claim_with_the_flag_off_reaches_the_gate(self):
        accepted, _ = self._run([self.COMPLETED_FLAG_OFF])
        self.assertEqual(accepted["status"], "completed")
        refused, _ = self._run([self.COMPLETED_FLAG_OFF], text_claims=False)
        self.assertNotEqual(refused["status"], "completed")

    def test_text_mediated_gets_the_same_rule(self):
        perception = json.dumps({
            "scene_summary": "Blue handbag partially visible at the right edge.",
            "objects": [], "target_visible": False, "target_location": "unknown",
            "target_correction": "unknown", "target_distance": "unknown",
            "task_progress": "edge of frame",
        })
        decision = json.dumps({
            "status": "continue", "action": "turn_right", "action_purpose": "search",
            "reason": "search", "target_visible": False, "target_location": "unknown",
            "target_correction": "unknown", "target_distance": "unknown",
            "target_evidence_source": "unknown",
        })
        report, _ = self._run(
            [perception, decision, self.RECHECK_YES], interface="text_mediated"
        )
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["evidence_text_claims"], 1)


class DestinationProgressTests(unittest.TestCase):
    def _planner(self):
        return MemoryDestinationTests._planner("landmark_first")

    @staticmethod
    def _context(distance):
        context = MemoryDestinationTests._context(landmark_distance=distance)
        context["semantic_memory"]["target_sighting"] = None
        return context

    def _guide(self, planner, distance, executed):
        decision = {}
        result = planner._memory_guided_action(
            memory_context=self._context(distance),
            available=MemoryDestinationTests.ACTIONS,
            perception_confirms_target=False,
            executed_count=executed, decision=decision,
        )
        return result, decision

    def test_no_progress_suspends_then_resumes(self):
        planner = self._planner()
        results = [self._guide(planner, 5.2, n) for n in range(7)]
        self.assertTrue(all(r[0] for r in results[:6]))
        self.assertIsNone(results[6][0])
        self.assertTrue(results[6][1]["memory_destination_suspended"])
        self.assertIsNone(self._guide(planner, 5.2, 6 + Planner.DESTINATION_SUSPEND_ACTIONS - 1)[0])
        self.assertIsNotNone(self._guide(planner, 5.2, 6 + Planner.DESTINATION_SUSPEND_ACTIONS)[0])

    def test_progress_resets_the_patience(self):
        planner = self._planner()
        distance = 6.0
        for n in range(12):
            distance -= 0.6
            result, decision = self._guide(planner, max(distance, 1.2), n)
            self.assertNotIn("memory_destination_suspended", decision)

    def test_latest_pose_policy_is_unchanged(self):
        planner = MemoryDestinationTests._planner("latest_pose")
        for n in range(12):
            result, decision = self._guide(planner, 5.2, n)
            self.assertIsNotNone(result)
            self.assertNotIn("memory_destination_suspended", decision)


class StricterEvidenceTextTests(unittest.TestCase):
    @staticmethod
    def _spec(task_id):
        name = "three_task_config.json" if task_id != "find_blue_bag" else "experiment_config.json"
        for task in CompletionPolicyTests._config(name)["tasks"]:
            if task["id"] == task_id:
                return task["goal"], dict(task["task_spec"])
        raise KeyError(task_id)

    def _claims(self, task_id, evidence):
        goal, spec = self._spec(task_id)
        contract = TaskContract.build(goal=goal, configured=spec)
        decision = dict(contract.as_dict(), visual_evidence=evidence)
        return Planner._evidence_text_names_target(goal, None, decision)

    def test_direct_partial_sightings_still_count(self):
        self.assertTrue(self._claims("find_blue_bag", "blue bag partially visible on right side"))
        self.assertTrue(self._claims("find_blue_bag", "partial view of blue bag on right side"))
        self.assertTrue(self._claims("locate_pink_box", "Partial view of the pink box on the right side of the room."))

    def test_hypothetical_indirect_occluded_and_negated_text_does_not(self):
        cases = [
            ("locate_yellow_box", "Partial view of the pink cube, indicating the yellow box might be nearby."),
            ("locate_pink_box", "Partial view of the pink box through the window."),
            ("locate_yellow_box", "Curtains partially obstructing the view of the yellow box."),
            ("locate_yellow_box", "Curtain partially covering the yellow box."),
            ("find_blue_bag", "curtain partially obscuring blue bag"),
            ("find_blue_bag", "Blue bag not visible, moving towards it."),
            ("find_blue_bag", "brown bag on the floor"),
        ]
        for task_id, evidence in cases:
            self.assertFalse(self._claims(task_id, evidence), evidence)

    def test_target_and_attribute_must_be_in_the_same_sentence(self):
        self.assertFalse(self._claims(
            "locate_yellow_box", "A pink box on the shelf. Yellow curtains by the window."))


class BlockedApproachTests(unittest.TestCase):
    @staticmethod
    def _context(cell_x):
        context = MemoryDestinationTests._context(landmark_distance=5.2)
        context["semantic_memory"]["target_sighting"] = None
        relative = context["semantic_memory"]["task_landmarks"][0]["relative_to_agent"]
        relative["turn_degrees"] = 10
        context["semantic_memory"]["agent_state"] = {"position": {"x": cell_x, "z": 0.0}}
        return context

    def _guide(self, planner, executed, available, cell_x=0.0):
        decision = {}
        result = planner._memory_guided_action(
            memory_context=self._context(cell_x), available=available,
            perception_confirms_target=False, executed_count=executed,
            decision=decision,
        )
        return result, decision

    def test_blocked_forward_suspends_until_the_agent_moves_elsewhere(self):
        planner = MemoryDestinationTests._planner("landmark_first")
        turns_only = {"turn_left", "turn_right", "move_backward"}
        result, decision = self._guide(planner, 0, turns_only)
        self.assertIsNone(result)
        self.assertTrue(decision["memory_destination_blocked"])
        free = MemoryDestinationTests.ACTIONS
        later = Planner.DESTINATION_SUSPEND_ACTIONS + 1
        self.assertIsNone(self._guide(planner, later, free, cell_x=0.0)[0])
        self.assertEqual(self._guide(planner, later, free, cell_x=2.0)[0][0], "move_forward")


class NavigationProgressMetricTests(unittest.TestCase):
    def test_spinning_in_place_is_visible_in_the_metrics(self):
        def step(x, yaw):
            return {"state_before": {"position": {"x": x, "z": 0.0}, "yaw": yaw}}
        history = [step(0, y) for y in (0, 45, 90, 0, 45)] + [step(1.0, 45), step(2.0, 45)]
        metrics = Planner._navigation_progress(history)
        self.assertEqual(metrics["distinct_cells_visited"], 3)
        self.assertEqual(metrics["repeated_pose_heading_steps"], 2)
        self.assertEqual(metrics["net_displacement_m"], 2.0)


class HybridInterfaceTests(unittest.TestCase):
    PERCEPTION = json.dumps({
        "scene_summary": "Blue handbag on the floor ahead.",
        "objects": [{"name": "handbag", "location": "center", "distance": "medium",
                     "attributes": ["blue"]}],
        "target_visible": True, "target_location": "center",
        "target_correction": "none", "target_distance": "medium",
        "task_progress": "bag seen",
    })
    DECISION = json.dumps({
        "status": "continue", "action": "move_forward",
        "action_purpose": "target_approach", "reason": "approach",
        "target_visible": True, "target_location": "center",
        "target_correction": "none", "target_distance": "medium",
        "target_evidence_source": "direct_view",
    })
    RECHECK = '{"status": "completed", "reason": "visible"}'

    class ImageRecordingBackend(FakeBackend):
        def __init__(self, *responses):
            super().__init__(*responses)
            self.images = []

        def generate(self, prompt, image=None, max_new_tokens=None):
            self.images.append(image is not None)
            return super().generate(prompt, image=image, max_new_tokens=max_new_tokens)

    def _run(self, interface, in_range=True):
        task = CompletionPolicyTests._config()["tasks"][0]
        backend = self.ImageRecordingBackend(self.PERCEPTION, self.DECISION, self.RECHECK)
        with tempfile.TemporaryDirectory() as d:
            planner = Planner(model_name="fake", backend=backend,
                              perceptual_interface=interface, memory_mode="none",
                              history_directory=d)
            report = planner.run_task(
                agent=FakeAgent(d), goal=task["goal"], max_steps=1,
                task_spec=task["task_spec"],
                success_evaluator=lambda: CompletionPolicyTests._evaluation(in_range),
            )
        return report, backend

    def test_hybrid_is_a_valid_interface_everywhere(self):
        from experiment import ExperimentRunner
        self.assertIn("hybrid", ExperimentRunner.VALID_SEQUENTIAL_INTERFACES)
        config = CompletionPolicyTests._config("experiment_config_hybrid.json")
        self.assertEqual(config["perceptual_interfaces"], ["hybrid"])
        self.assertNotEqual(config["output_directory"], "experiment_results")
        base = CompletionPolicyTests._config()
        for key in ("tasks", "max_steps", "memory_modes", "scene",
                    "decision_max_new_tokens", "completion_max_new_tokens"):
            self.assertEqual(config[key], base[key], key)

    def test_same_calls_as_text_but_the_decision_and_recheck_see_the_image(self):
        hybrid, hybrid_backend = self._run("hybrid")
        text, text_backend = self._run("text_mediated")
        self.assertEqual(hybrid["status"], "completed")
        self.assertEqual(text["status"], "completed")
        self.assertEqual(len(hybrid_backend.prompts), len(text_backend.prompts), 3)
        self.assertEqual(text_backend.images, [True, False, False])
        self.assertEqual(hybrid_backend.images, [True, True, True])

    def test_decision_prompt_is_the_text_prompt_plus_the_image_note(self):
        _, hybrid_backend = self._run("hybrid")
        _, text_backend = self._run("text_mediated")
        self.assertEqual(hybrid_backend.prompts[0]["prompt"], text_backend.prompts[0]["prompt"])
        hybrid_decision = hybrid_backend.prompts[1]["prompt"]
        text_decision = text_backend.prompts[1]["prompt"]
        self.assertIn("The current RGB image is also attached", hybrid_decision)
        self.assertNotIn("The current RGB image is also attached", text_decision)
        self.assertNotIn(Planner.TARGET_VISIBILITY_RULES, hybrid_decision)
        self.assertNotIn('"visual_evidence"', hybrid_decision)


if __name__ == "__main__":
    unittest.main()
