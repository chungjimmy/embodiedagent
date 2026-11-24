import json
import tempfile
import unittest
from pathlib import Path

from planner import Planner
from task_contract import TaskContract
from test_task_and_planner import FakeAgent, FakeBackend


BOX_SPEC = {
    "task_target": "box",
    "target_aliases": ["cube", "block"],
    "target_attributes": ["yellow"],
    "excluded_target_terms": ["pink"],
    "completion_mode": "target_near",
    "requires_target_confirmation": True,
    "visible_target_approach": True,
    "sighting_handling": True,
    "allowed_actions": [
        "move_forward",
        "move_backward",
        "turn_left",
        "turn_right",
    ],
}

NOT_YET = {
    "available": True,
    "success": False,
    "independent": True,
    "reason": "target is not yet within two meters",
}


def perception(location, correction="unknown"):
    return (
        '{"scene_summary":"A yellow box is visible on the ' + location + '.",'
        '"objects":[{"name":"yellow box","location":"' + location + '",'
        '"distance":"medium","attributes":["yellow"]}],'
        '"target_visible":true,"target_location":"' + location + '",'
        '"target_correction":"' + correction + '","target_distance":"medium"}'
    )


NO_TARGET = (
    '{"scene_summary":"A hallway with a wooden table.",'
    '"objects":[{"name":"table","location":"center","distance":"medium",'
    '"attributes":["wooden"]}],"target_visible":false,'
    '"target_location":"unknown","target_distance":"unknown"}'
)

MODEL_SEARCH_RIGHT = (
    '{"target_visible":false,"target_location":"unknown",'
    '"target_distance":"unknown","status":"continue",'
    '"action":"turn_right","reason":"Search."}'
)

MODEL_SEARCH_LEFT = (
    '{"target_visible":false,"target_location":"unknown",'
    '"target_distance":"unknown","status":"continue",'
    '"action":"turn_left","reason":"Search."}'
)


def make_planner(directory, backend, interface="text_mediated", memory="none"):
    return Planner(
        model_name="fake",
        backend=backend,
        perceptual_interface=interface,
        memory_mode=memory,
        history_directory=directory,
    )


def spec(**extra):
    value = dict(BOX_SPEC)
    value.update(extra)
    return value


class SettingTests(unittest.TestCase):
    def test_defaults_are_off(self):
        self.assertFalse(Planner._simplified_rules_setting({}))
        self.assertEqual(Planner._controller_priority_setting({}), "default")
        self.assertFalse(Planner._simplified_rules_setting(None))

    def test_invalid_values_are_rejected(self):
        with self.assertRaises(ValueError):
            Planner._simplified_rules_setting({"simplified_rules": "yes"})
        with self.assertRaises(ValueError):
            Planner._controller_priority_setting(
                {"controller_priority": "planner_first"}
            )


class SimplifiedSchemaTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.contract = TaskContract.build("Find the yellow box.", BOX_SPEC)

    def tearDown(self):
        self.directory.cleanup()

    def test_direct_schema_puts_evidence_before_action(self):
        planner = make_planner(
            self.directory.name, FakeBackend(), interface="direct_visual"
        )
        planner._simplified_rules = True
        keys = list(planner._decision_response_schema(self.contract))
        self.assertEqual(keys[0], "visual_evidence")
        self.assertLess(keys.index("target_visible"), keys.index("status"))
        self.assertLess(keys.index("target_location"), keys.index("action"))
        self.assertLess(keys.index("status"), keys.index("action"))
        self.assertNotIn("target_correction", keys)
        self.assertNotIn("action_purpose", keys)

    def test_default_schema_is_unchanged(self):
        planner = make_planner(
            self.directory.name, FakeBackend(), interface="direct_visual"
        )
        keys = list(planner._decision_response_schema(self.contract))
        self.assertEqual(keys[:2], ["status", "action"])
        self.assertIn("target_correction", keys)


class SimplifiedPromptTests(unittest.TestCase):
    def run_one(self, interface, simplified, responses):
        with tempfile.TemporaryDirectory() as directory:
            backend = FakeBackend(*responses)
            planner = make_planner(directory, backend, interface=interface)
            planner.run_task(
                agent=FakeAgent(directory),
                goal="Find the yellow box.",
                max_steps=1,
                task_spec=spec(simplified_rules=simplified),
                success_evaluator=lambda: NOT_YET,
            )
        return backend.prompts

    def test_direct_prompt_is_shorter_and_has_no_correction(self):
        response = (
            '{"visual_evidence":"A hallway with a wooden table.",'
            '"target_visible":false,"target_location":"unknown",'
            '"target_distance":"unknown","status":"continue",'
            '"action":"turn_left","reason":"Search."}'
        )
        old = self.run_one("direct_visual", False, [response])[0]["prompt"]
        new = self.run_one("direct_visual", True, [response])[0]["prompt"]
        self.assertLess(len(new), len(old) * 0.6)
        self.assertNotIn("target_correction", new)
        self.assertIn("move_forward toward it", new)
        self.assertNotIn("impossible", new)
        self.assertTrue(new.index('"visual_evidence"') < new.index('"status"'))

    def test_text_prompts_drop_correction(self):
        prompts = self.run_one(
            "text_mediated", True, [NO_TARGET, MODEL_SEARCH_LEFT]
        )
        self.assertEqual(len(prompts), 2)
        for item in prompts:
            self.assertNotIn("target_correction", item["prompt"])
        self.assertIn("Scene summary from the current image", prompts[1]["prompt"])


class TargetFirstControllerTests(unittest.TestCase):
    def run_task(self, responses, steps, **extra):
        with tempfile.TemporaryDirectory() as directory:
            backend = FakeBackend(*responses)
            planner = make_planner(directory, backend)
            report = planner.run_task(
                agent=FakeAgent(directory),
                goal="Find the yellow box.",
                max_steps=steps,
                task_spec=spec(**extra),
                success_evaluator=lambda: NOT_YET,
            )
        return report

    def test_side_target_is_approached_forward_even_when_model_searches(self):
        for side in ("left", "right"):
            report = self.run_task(
                [perception(side, correction=side), MODEL_SEARCH_RIGHT],
                1,
                controller_priority="target_first",
            )
            self.assertEqual(report["history"][0]["action"], "move_forward")
            self.assertEqual(
                report["history"][0]["action_purpose"], "target_approach"
            )

    def test_default_controller_follows_model_turn_correction(self):
        report = self.run_task(
            [perception("left", correction="left"), MODEL_SEARCH_RIGHT],
            1,
            target_reacquisition_after_turn=False,
        )
        self.assertEqual(report["history"][0]["action"], "turn_left")

    def test_centered_target_moves_forward(self):
        report = self.run_task(
            [perception("center"), MODEL_SEARCH_RIGHT],
            1,
            controller_priority="target_first",
        )
        self.assertEqual(report["history"][0]["action"], "move_forward")

    def test_lost_side_target_turns_toward_last_side(self):
        report = self.run_task(
            [perception("left"), MODEL_SEARCH_RIGHT, NO_TARGET, MODEL_SEARCH_RIGHT],
            2,
            controller_priority="target_first",
        )
        self.assertEqual(
            [item["action"] for item in report["history"]],
            ["move_forward", "turn_left"],
        )

    def test_blocked_forward_turns_toward_side_target(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = FakeBackend(perception("right"), MODEL_SEARCH_LEFT)
            planner = make_planner(directory, backend)
            agent = FakeAgent(directory)
            planner._collision_safe_actions_for_agent = (
                lambda agent, allowed: set(allowed) - {"move_forward"}
            )
            report = planner.run_task(
                agent=agent,
                goal="Find the yellow box.",
                max_steps=1,
                task_spec=spec(
                    controller_priority="target_first",
                    collision_aware_action_filter=True,
                    target_reacquisition_after_turn=False,
                ),
                success_evaluator=lambda: NOT_YET,
            )
        self.assertEqual(report["history"][0]["action"], "turn_right")

    def blocked_forward_run(self, responses, steps, **extra):
        with tempfile.TemporaryDirectory() as directory:
            backend = FakeBackend(*responses)
            planner = make_planner(directory, backend)
            planner._collision_safe_actions_for_agent = (
                lambda agent, allowed: set(allowed) - {"move_forward"}
            )
            options = dict(
                controller_priority="target_first",
                collision_aware_action_filter=True,
                target_reacquisition_after_turn=False,
            )
            options.update(extra)
            report = planner.run_task(
                agent=FakeAgent(directory),
                goal="Find the yellow box.",
                max_steps=steps,
                task_spec=spec(**options),
                success_evaluator=lambda: NOT_YET,
            )
        return report

    def test_blocked_forward_with_rejected_claim_turns_instead_of_backing(self):
        claim = (
            '{"target_visible":true,"target_location":"center",'
            '"target_distance":"medium","status":"completed",'
            '"action":null,"reason":"Done."}'
        )
        check = '{"status":"completed","reason":"Visible."}'
        report = self.blocked_forward_run(
            [perception("center"), claim, check, check], 1
        )
        self.assertIn(
            report["history"][0]["action"], {"turn_left", "turn_right"}
        )

    def test_detour_keeps_one_side(self):
        backing = (
            '{"target_visible":true,"target_location":"center",'
            '"target_distance":"medium","status":"continue",'
            '"action":"move_backward","reason":"Back up."}'
        )
        report = self.blocked_forward_run(
            [perception("center"), backing, perception("center"), backing], 2
        )
        actions = [item["action"] for item in report["history"]]
        self.assertEqual(actions, ["turn_left", "turn_left"])
        self.assertEqual(report["target_detours"], 2)

    def test_default_controller_can_still_back_up(self):
        backing = (
            '{"target_visible":true,"target_location":"center",'
            '"target_distance":"medium","status":"continue",'
            '"action":"move_backward","reason":"Back up."}'
        )
        report = self.blocked_forward_run(
            [perception("center"), backing], 1, controller_priority="default"
        )
        self.assertEqual(report["history"][0]["action"], "move_backward")

    def test_impossible_is_ignored_without_a_check(self):
        impossible = (
            '{"target_visible":false,"target_location":"unknown",'
            '"target_distance":"unknown","status":"impossible",'
            '"action":"turn_left","reason":"Not here."}'
        )
        report = self.run_task(
            [NO_TARGET, impossible, NO_TARGET, impossible],
            2,
            simplified_rules=True,
        )
        self.assertEqual(report["status"], "action_limit")
        self.assertEqual(report["impossible_status_ignored"], 2)
        self.assertEqual(report["rejected_impossible_claims"], 0)
        for item in report["history"]:
            self.assertNotIn("Impossible was rejected", item["reason"])

    def test_blocked_side_target_goes_around_the_obstacle(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = FakeBackend(
                perception("right"),
                MODEL_SEARCH_RIGHT,
                NO_TARGET,
                MODEL_SEARCH_RIGHT,
                NO_TARGET,
                MODEL_SEARCH_RIGHT,
            )
            planner = make_planner(directory, backend)
            agent = FakeAgent(directory)
            calls = []

            def safe_actions(agent, allowed):
                calls.append(1)
                if len(calls) == 1:
                    return set(allowed) - {"move_forward", "move_backward"}
                return set(allowed)

            planner._collision_safe_actions_for_agent = safe_actions
            report = planner.run_task(
                agent=agent,
                goal="Find the yellow box.",
                max_steps=3,
                task_spec=spec(
                    controller_priority="target_first",
                    collision_aware_action_filter=True,
                    target_reacquisition_after_turn=True,
                ),
                success_evaluator=lambda: NOT_YET,
            )
        actions = [item["action"] for item in report["history"]]
        self.assertEqual(actions, ["turn_right", "move_forward", "turn_left"])
        self.assertEqual(
            [item["action_purpose"] for item in report["history"]],
            ["target_detour"] * 3,
        )
        self.assertFalse(
            any(item.get("target_reacquisition_attempted") for item in report["history"])
        )

    def test_close_range_recheck_after_walking_to_target(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = FakeBackend(
                perception("center"),
                MODEL_SEARCH_RIGHT,
                NO_TARGET,
                perception("left"),
                MODEL_SEARCH_RIGHT,
            )
            planner = make_planner(directory, backend)
            report = planner.run_task(
                agent=FakeAgent(directory),
                goal="Find the yellow box.",
                max_steps=2,
                task_spec=spec(
                    controller_priority="target_first", close_range_recheck=True
                ),
                success_evaluator=lambda: NOT_YET,
            )
            prompts = [item["prompt"] for item in backend.prompts]
        self.assertEqual(len(prompts), 5)
        self.assertIn("It may now be very close", prompts[3])
        second = report["history"][1]
        self.assertTrue(second["target_reacquisition_attempted"])
        self.assertTrue(second["target_reacquisition_confirmed"])
        self.assertEqual(second["action"], "move_forward")

    def test_no_close_range_recheck_without_target_first(self):
        with tempfile.TemporaryDirectory() as directory:
            backend = FakeBackend(
                perception("center"),
                MODEL_SEARCH_RIGHT,
                NO_TARGET,
                MODEL_SEARCH_RIGHT,
            )
            planner = make_planner(directory, backend)
            planner.run_task(
                agent=FakeAgent(directory),
                goal="Find the yellow box.",
                max_steps=2,
                task_spec=spec(),
                success_evaluator=lambda: NOT_YET,
            )
            prompts = [item["prompt"] for item in backend.prompts]
        self.assertFalse(any("It may now be very close" in item for item in prompts))

    def test_initial_scan_turns_before_walking(self):
        forward = (
            '{"target_visible":false,"target_location":"unknown",'
            '"target_distance":"unknown","status":"continue",'
            '"action":"move_forward","reason":"Explore."}'
        )
        report = self.run_task(
            [NO_TARGET, forward] * 9,
            9,
            initial_scan=True,
        )
        actions = [item["action"] for item in report["history"]]
        self.assertEqual(actions[:6], ["turn_right"] * 6)
        self.assertEqual(actions[6], "move_forward")
        self.assertEqual(report["initial_scan_turns"], 6)

    def test_initial_scan_stops_once_target_is_seen(self):
        forward = (
            '{"target_visible":false,"target_location":"unknown",'
            '"target_distance":"unknown","status":"continue",'
            '"action":"move_forward","reason":"Explore."}'
        )
        report = self.run_task(
            [NO_TARGET, forward, perception("center"), forward, NO_TARGET, forward],
            3,
            initial_scan=True,
        )
        actions = [item["action"] for item in report["history"]]
        self.assertEqual(actions[0], "turn_right")
        self.assertEqual(actions[1], "move_forward")
        self.assertNotEqual(report["history"][2].get("action_purpose"), "initial_scan")

    def test_initial_scan_is_off_by_default(self):
        forward = (
            '{"target_visible":false,"target_location":"unknown",'
            '"target_distance":"unknown","status":"continue",'
            '"action":"move_forward","reason":"Explore."}'
        )
        report = self.run_task([NO_TARGET, forward], 1)
        self.assertEqual(report["history"][0]["action"], "move_forward")

    def test_both_options_together(self):
        report = self.run_task(
            [perception("right"), MODEL_SEARCH_LEFT],
            1,
            simplified_rules=True,
            controller_priority="target_first",
        )
        self.assertEqual(report["history"][0]["action"], "move_forward")


class ShippedConfigTests(unittest.TestCase):
    def task_specs(self, name):
        config = json.loads(
            Path(__file__).with_name(name).read_text(encoding="utf-8")
        )
        return [task["task_spec"] for task in config["tasks"]]

    def test_final_configs_turn_all_options_on(self):
        for name in (
            "experiment_config_final.json",
            "three_task_config_final.json",
            "experiment_config_final_check.json",
            "experiment_config_smolvlm.json",
            "experiment_config_llama.json",
        ):
            for task_spec in self.task_specs(name):
                if "final" in name:
                    self.assertTrue(task_spec.get("initial_scan"))
                    self.assertFalse(task_spec.get("close_range_recheck", False))
                if task_spec.get("task_target") == "bag":
                    self.assertIn("suitcase", task_spec["target_aliases"])
                self.assertTrue(Planner._simplified_rules_setting(task_spec))
                self.assertEqual(
                    Planner._controller_priority_setting(task_spec),
                    "target_first",
                )

    def test_original_configs_keep_both_options_off(self):
        for name in (
            "experiment_config.json",
            "experiment_config_hybrid.json",
            "three_task_config.json",
            "three_task_config_fixed_return.json",
        ):
            for task_spec in self.task_specs(name):
                self.assertFalse(Planner._simplified_rules_setting(task_spec))
                self.assertEqual(
                    Planner._controller_priority_setting(task_spec), "default"
                )


class OtherModelConfigTests(unittest.TestCase):
    def test_other_model_configs_are_small_text_runs(self):
        from vision import MODEL_ALIASES

        for key in ("smolvlm", "llama"):
            config = json.loads(
                Path(__file__)
                .with_name(f"experiment_config_{key}.json")
                .read_text(encoding="utf-8")
            )
            self.assertEqual(config["models"][0]["name"], MODEL_ALIASES[key])
            self.assertNotIn("min_pixels", config["models"][0])
            self.assertEqual(config["perceptual_interfaces"], ["text_mediated"])
            self.assertEqual(config["memory_modes"], ["none", "combined"])
            self.assertTrue(config["output_directory"].startswith("experiment_results_"))


if __name__ == "__main__":
    unittest.main()
