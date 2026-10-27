from datetime import datetime
import inspect
import json
import math
from pathlib import Path
import re
import time

from memory import MemoryManager, normalize_memory_mode
from task_contract import TaskContract
from vision import VisionLanguageModel


class Planner:
    MOVE_DISTANCE_METERS = 1.0
    TURN_ANGLE_DEGREES = 45
    COMPLETION_MODES = {
        "target_visible",
        "target_near",
        "target_held",
        "state_reached",
        "other",
        "unknown",
    }
    TURN_ACTIONS = {"turn_left", "turn_right"}
    TARGET_CORRECTIONS = {"none", "left", "right", "unknown"}
    ACTION_PURPOSES = {
        "search",
        "target_alignment",
        "target_approach",
        "overshoot_correction",
        "route_correction",
        "collision_avoidance",
        "other",
    }
    CAMERA_ACTIONS = {"look_up", "look_down"}
    TRANSLATIONAL_ACTIONS = {"move_forward", "move_backward"}
    MAX_CONSECUTIVE_TURNS_WITHOUT_TRANSLATION = 6
    LOCAL_SCAN_HEADINGS_REQUIRED = 7
    ROAMING_POSITIONS_REQUIRED = 3
    ROAMING_HEADINGS_REQUIRED = 4
    IMPOSSIBLE_BASES = {"blocked", "infeasible", "none"}
    DEFAULT_COMPLETION_CONSISTENCY = {
        "retry_on_inconsistency": True,
        "promote_validated_perception": False,
        "require_environment_success_for_completion": False,
    }
    DEFAULT_EVIDENCE_CONFLICT_CHECK = {"enabled": False, "max_calls": 8}
    COMPLETION_POLICIES = ("model_distance", "assisted")
    MEMORY_GUIDED_DESTINATIONS = ("latest_pose", "landmark_first")
    DEFAULT_MEMORY_GUIDED_DESTINATION = "latest_pose"
    SIGHTING_REACQUISITION_ACTIONS = 4
    DESTINATION_MAX_DWELL = 2
    DESTINATION_PATIENCE = 6
    DESTINATION_SUSPEND_ACTIONS = 8
    DESTINATION_PROGRESS_METERS = 0.5
    TARGET_VISIBILITY_RULES_BASE = """Target visibility rules:
When the contract includes target_aliases, any listed alias is an acceptable
visible name for the configured target class. When it includes
target_attributes, a candidate can confirm the target only when the image
supports every listed attribute as well as the target class or one alias.
Determine attributes from the pixels before comparing them with the contract;
never copy a requested attribute into an observation merely because it appears
in the goal. For color attributes, report the color actually visible. A
candidate whose visible color matches an excluded_target_terms color is a
non-target, even when its shape or class matches.
Inspect the entire frame, including the lower edge and corners. Do not dismiss
the whole image as unclear merely because an object is partly cropped. A
partly cropped candidate can be reported when its class and required visual
attributes are still recognizable; otherwise describe it as ambiguous.
If the target has a clear location and its required visible attributes, set
target_visible to true, even when it is far away. Distance is not required to
confirm a visible target. Do not return target_visible false while also
describing that target as a clearly visible object. Use target_visible false
when the object is only an ambiguous or unsupported candidate. Never set
target_visible true from the task label alone: a true value requires a
description that names or clearly describes the target and where it appears.
A bare direction, placeholder, or one-word claim is not visual evidence.
target_location describes where the target appears across the image (left,
center, or right), but target_correction describes whether a horizontal body
turn is necessary: use none when the target is fully visible and forward
movement can approach it; use left or right only when it is clipped by, or very
close to, that image edge; otherwise use unknown.
Estimate target_distance from the current image by apparent size; do not
invent exact meters. near means the target is large in the frame (roughly an
eighth of the image or more); medium means a clearly visible but modest part of
the frame; far means a small part of the frame. If the target is visible,
always give near, medium, or far for target_distance, not unknown."""
    TARGET_PARTIAL_VISIBILITY_RULE = """A target that is only partly in view, for example cut off by the image
edge, is visible when the visible portion shows its class and every required
attribute: set target_visible to true. Partial visibility says nothing about
distance and does not by itself mean the task is complete."""
    TARGET_VISIBILITY_RULES = (
        TARGET_VISIBILITY_RULES_BASE + "\n" + TARGET_PARTIAL_VISIBILITY_RULE
    )
    DEFAULT_COMPLETION_POLICY = "model_distance"
    DEFAULT_TARGET_LOSS_RECOVERY = {
        "enabled": False,
        "max_actions_since_sighting": 2,
        "reversal_window_actions": 6,
    }
    DEFAULT_SIGHTING_HANDLING = {
        "accept_unlocated": False,
        "request_horizontal_location": False,
        "request_distance_estimate": False,
        "ignore_contradictory_correction": False,
    }
    _NEGATION_WORDS = frozenset(
        {"no", "not", "without", "isnt", "arent", "none", "unable", "cannot",
         "cant", "nothing", "absent"}
    )
    PERCEPTION_REQUIRED_KEYS = {
        "scene_summary",
        "objects",
        "target_visible",
    }

    EXECUTABLE_ACTIONS = {
        "move_forward": "w",
        "move_backward": "s",
        "turn_left": "a",
        "turn_right": "d",
        "look_up": "q",
        "look_down": "e",
        "pick_up": "p",
        "put_down": "o",
    }

    def __init__(
        self,
        model_name="Qwen/Qwen2.5-VL-3B-Instruct",
        backend=None,
        perceptual_interface="direct_visual",
        memory_mode="none",
        memory_manager=None,
        history_directory="task_history",
        use_4bit=True,
        episodic_window=8,
        min_steps_before_impossible=None,
        baseline_name="primary",
        perception_max_new_tokens=300,
        perception_retry_max_new_tokens=224,
        decision_max_new_tokens=300,
        completion_max_new_tokens=64,
        save_step_images=False,
        memory_guided_search=True,
    ):
        if perceptual_interface not in {
            "direct_visual", "text_mediated", "hybrid", "coela"
        }:
            raise ValueError(f"Unknown perceptual interface: {perceptual_interface}")
        self.model_name = model_name
        self.backend = backend or VisionLanguageModel(
            model_name=model_name,
            use_4bit=use_4bit,
        )
        self.perceptual_interface = perceptual_interface
        self.memory_mode = normalize_memory_mode(memory_mode)
        self.memory = memory_manager or MemoryManager(
            episodic_window=episodic_window
        )
        self.history_directory = Path(history_directory)
        self.history_directory.mkdir(parents=True, exist_ok=True)
        self.min_steps_before_impossible = None
        self.baseline_name = baseline_name
        self.perception_max_new_tokens = int(perception_max_new_tokens)
        self.perception_retry_max_new_tokens = int(
            perception_retry_max_new_tokens
        )
        self.decision_max_new_tokens = int(decision_max_new_tokens)
        self.completion_max_new_tokens = int(completion_max_new_tokens)
        self.save_step_images = bool(save_step_images)
        self.memory_guided_search = bool(memory_guided_search)
        self._active_task_contract = None
        self._active_allowed_actions = set(self.EXECUTABLE_ACTIONS)
        self._collision_safe_actions = None
        self._target_approach_state = None
        self._last_target_sighting = None
        self._sighting_handling = dict(self.DEFAULT_SIGHTING_HANDLING)
        self._completion_policy = self.DEFAULT_COMPLETION_POLICY
        self._memory_guided_hold_actions = 0
        self._last_confirmed_sighting_count = None
        self._memory_guided_destination = self.DEFAULT_MEMORY_GUIDED_DESTINATION
        self._destination_dwell = {}
        self._abandoned_destinations = set()
        self._destination_progress = {}
        self._suspended_destinations = {}
        self._claim_on_evidence_text = False
        self._active_turn_angle_degrees = float(self.TURN_ANGLE_DEGREES)

    def run_task(
        self,
        agent,
        goal,
        max_steps=20,
        episode_metadata=None,
        reset_memory=True,
        success_evaluator=None,
        task_spec=None,
        progress_callback=None,
    ):
        task_contract = TaskContract.build(goal=goal, configured=task_spec)
        self._active_task_contract = task_contract
        self._active_allowed_actions = set(task_contract.allowed_actions)
        try:
            configured_turn_angle = float(
                getattr(agent, "turn_angle", self.TURN_ANGLE_DEGREES)
            )
        except (TypeError, ValueError):
            configured_turn_angle = float(self.TURN_ANGLE_DEGREES)
        if not math.isfinite(configured_turn_angle) or configured_turn_angle <= 0:
            configured_turn_angle = float(self.TURN_ANGLE_DEGREES)
        self._active_turn_angle_degrees = configured_turn_angle
        collision_aware_action_filter = self._collision_aware_filter_enabled(
            task_spec
        )
        self._collision_safe_actions = None
        completion_consistency = self._completion_consistency_policy(task_spec)
        visible_target_approach = self._visible_target_approach_enabled(task_spec)
        target_approach_persistence_steps = (
            self._target_approach_persistence_steps(task_spec)
        )
        target_reacquisition_after_turn = (
            self._target_reacquisition_after_turn_enabled(task_spec)
        )
        evidence_conflict_policy = self._evidence_conflict_policy(task_spec)
        target_loss_recovery = self._target_loss_recovery_policy(task_spec)
        self._sighting_handling = self._sighting_handling_policy(task_spec)
        self._completion_policy = self._completion_policy_setting(
            task_spec, completion_consistency
        )
        self._memory_guided_hold_actions = self._memory_guided_hold_setting(
            task_spec
        )
        self._last_confirmed_sighting_count = None
        self._memory_guided_destination = self._memory_guided_destination_setting(
            task_spec
        )
        self._destination_dwell = {}
        self._abandoned_destinations = set()
        self._destination_progress = {}
        self._suspended_destinations = {}
        self._claim_on_evidence_text = self._claim_on_evidence_text_setting(
            task_spec, self._completion_policy
        )
        evidence_conflict_calls_used = 0
        self._target_approach_state = None
        self._last_target_sighting = None
        started_at = datetime.now()
        task_start = time.perf_counter()
        history = []
        final_status = "running"
        final_reason = ""
        final_image_path = None
        latest_ground_truth = None
        inference_totals = {
            "calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "inference_seconds": 0.0,
            "prompt_characters": 0,
            "max_memory_context_characters": 0,
            "memory_context_budget_characters": 0,
            "max_cuda_memory_allocated_mb": 0.0,
            "max_cuda_memory_reserved_mb": 0.0,
            "max_cuda_memory_allocated_mb_after_cleanup": 0.0,
            "max_cuda_memory_reserved_mb_after_cleanup": 0.0,
            "max_cuda_memory_allocated_mb_before": 0.0,
            "max_cuda_memory_reserved_mb_before": 0.0,
            "min_cuda_free_memory_mb": None,
            "cuda_total_memory_mb": 0.0,
            "preprocess_seconds": 0.0,
            "time_to_first_token_seconds": 0.0,
            "decode_seconds": 0.0,
            "outputs_hitting_token_limit": 0,
            "generation_timeouts": 0,
            "generation_soft_limit_exceeded": 0,
        }
        if reset_memory:
            self.memory.reset_episode(goal=goal)
        else:
            self.memory.goal = goal

        print(f"\nGoal: {goal}")
        print(
            f"Condition: {self.perceptual_interface} + {self.memory_mode} memory"
        )

        try:
            for decision_step in range(1, max_steps + 1):
                step_start = time.perf_counter()
                current_view = agent.get_current_view()
                step_image_path = None
                if self.save_step_images:
                    try:
                        saved_step_image = Path(
                            agent.save_current_view(
                                image=current_view,
                                prefix=f"step_{decision_step:03d}",
                            )
                        ).resolve()
                        try:
                            step_image_path = str(
                                saved_step_image.relative_to(
                                    self.history_directory.resolve()
                                )
                            )
                        except (OSError, ValueError):
                            step_image_path = saved_step_image.name
                    except Exception as error:
                        print(
                            "Step image warning: "
                            f"{type(error).__name__}: {error}"
                        )
                agent_state = self._agent_state(agent)
                self._collision_safe_actions = (
                    self._collision_safe_actions_for_agent(
                        agent,
                        self._active_allowed_actions,
                    )
                    if collision_aware_action_filter
                    else None
                )
                latest_ground_truth = self._evaluate_success(
                    success_evaluator,
                    history=history,
                )
                perception, perception_call = self._prepare_perception(
                    image=current_view,
                    goal=goal,
                    step=decision_step,
                    agent=agent,
                    task_contract=task_contract,
                )
                if perception is not None:
                    perception_calls = self._present_calls(perception_call)
                    if perception_calls:
                        for call_index, call in enumerate(perception_calls):
                            self._save_scene_interpretation(
                                step=decision_step,
                                source=(
                                    "perception"
                                    if call_index == 0
                                    else "perception_retry"
                                ),
                                interpretation=(
                                    perception
                                    if call_index == len(perception_calls) - 1
                                    else None
                                ),
                                model_call=call,
                            )
                    else:
                        self._save_scene_interpretation(
                            step=decision_step,
                            source="perception",
                            interpretation=perception,
                            model_call=None,
                        )
                self._add_metrics(inference_totals, perception_call)
                reacquisition_call = None
                reacquisition_attempted = False
                reacquisition_confirmed = False
                if (
                    target_reacquisition_after_turn
                    and self._should_reacquire_target(
                        perception=perception,
                        goal=goal,
                        history=history,
                    )
                ):
                    reacquisition_attempted = True
                    reacquired_perception, reacquisition_call = (
                        self._reacquire_target(
                            image=current_view,
                            goal=goal,
                            step=decision_step,
                            task_contract=task_contract,
                        )
                    )
                    self._target_approach_state[
                        "reacquisition_attempted"
                    ] = True
                    reacquisition_confirmed = self._perception_confirms_target(
                        goal, reacquired_perception
                    )
                    self._save_scene_interpretation(
                        step=decision_step,
                        source="target_reacquisition",
                        interpretation=reacquired_perception,
                        model_call=reacquisition_call,
                    )
                    self._add_metrics(inference_totals, reacquisition_call)
                    if reacquisition_confirmed:
                        perception = reacquired_perception
                if perception is not None:
                    self.memory.record_observation(
                        perception=perception,
                        agent_state=agent_state,
                        step=decision_step,
                    )
                memory_context = self.memory.get_context(
                    mode=self.memory_mode,
                    goal=goal,
                    task_target=task_contract.task_target,
                    target_attributes=task_contract.target_attributes,
                )
                memory_context_stats = self.memory.context_stats()
                inference_totals["max_memory_context_characters"] = max(
                    inference_totals["max_memory_context_characters"],
                    int(memory_context_stats.get("characters") or 0),
                )
                inference_totals["memory_context_budget_characters"] = int(
                    memory_context_stats.get("budget_characters") or 0
                )

                blocked_action = self._get_blocked_action(history)
                action_count = len(
                    [item for item in history if item.get("executed")]
                )
                decision, decision_call = self._decide(
                    image=current_view,
                    perception=perception,
                    goal=goal,
                    memory_context=memory_context,
                    memory_context_stats=memory_context_stats,
                    blocked_action=blocked_action,
                    action_count=action_count,
                    max_steps=max_steps,
                    current_state=agent_state,
                    recent_history=history,
                    task_contract=task_contract,
                )
                if self._collision_safe_actions is not None:
                    decision["collision_safe_actions"] = sorted(
                        self._collision_safe_actions
                    )
                decision["target_reacquisition_attempted"] = (
                    reacquisition_attempted
                )
                decision["target_reacquisition_confirmed"] = (
                    reacquisition_confirmed
                )

                if (
                    decision["status"] == "impossible"
                    and not self._impossible_is_justified(
                        decision=decision,
                        history=history,
                        blocked_action=blocked_action,
                        current_state=agent_state,
                        perception=perception,
                        turn_angle_degrees=self._active_turn_angle_degrees,
                    )
                ):
                    decision["original_status"] = "impossible"
                    decision["status"] = "continue"
                    available, _ = self._guarded_actions(
                        history,
                        blocked_action,
                        allowed_actions=self._current_action_options(),
                        turn_angle_degrees=self._active_turn_angle_degrees,
                    )
                    requested_action, _ = self._normalize_action(
                        decision.get("requested_action")
                    )
                    decision["action"] = (
                        requested_action
                        if requested_action in available
                        else self._choose_exploration_action(
                            history,
                            blocked_action,
                            allowed_actions=self._current_action_options(),
                            turn_angle_degrees=self._active_turn_angle_degrees,
                        )
                    )
                    decision["parser_result"] = "impossible_rejected_by_criteria"
                    decision["controller_intervention"] = (
                        "impossible_rejected"
                    )
                    decision["reason"] = (
                        "Impossible was rejected because the area was not yet "
                        "sufficiently explored or mapped; continuing with "
                        f"{decision['action']}."
                    )

                (
                    decision,
                    evidence_conflict_call,
                ) = self._check_evidence_conflict(
                    decision=decision,
                    perception=perception,
                    image=current_view,
                    goal=goal,
                    task_contract=task_contract,
                    policy=evidence_conflict_policy,
                    calls_used=evidence_conflict_calls_used,
                )
                if evidence_conflict_call is not None:
                    evidence_conflict_calls_used += 1
                    self._add_metrics(inference_totals, evidence_conflict_call)
                    self._save_scene_interpretation(
                        step=decision_step,
                        source="evidence_conflict_check",
                        interpretation=None,
                        model_call=evidence_conflict_call,
                    )

                policy_perception = perception
                if policy_perception is None:
                    policy_perception = self._direct_visual_observation(
                        decision=decision,
                        goal=goal,
                        step=decision_step,
                    )
                (
                    decision,
                    completion_check,
                    completion_check_call,
                ) = self._resolve_completion_inconsistency(
                    decision=decision,
                    evidence=policy_perception,
                    environment_evidence=latest_ground_truth,
                    image=current_view,
                    goal=goal,
                    task_contract=task_contract,
                    policy=completion_consistency,
                )
                decision = self._apply_movement_policy(
                    decision=decision,
                    goal=goal,
                    perception=policy_perception,
                    history=history,
                    blocked_action=blocked_action,
                    environment_evidence=latest_ground_truth,
                    visible_target_approach=visible_target_approach,
                    target_approach_persistence_steps=(
                        target_approach_persistence_steps
                    ),
                    environment_completion_enabled=completion_consistency[
                        "require_environment_success_for_completion"
                    ],
                    memory_context=memory_context,
                    target_loss_recovery=target_loss_recovery,
                )
                if isinstance(completion_check, dict) and completion_check.get(
                    "completion_inconsistency"
                ):
                    completion_check["resolved"] = bool(
                        decision.get("completion_retry_resolved")
                    )
                    completion_check["accepted"] = (
                        decision.get("status") == "completed"
                    )
                    completion_check["rejection_source"] = decision.get(
                        "completion_rejection_source"
                    )
                step_perception = perception
                if step_perception is None:
                    step_perception = self._direct_visual_observation(
                        decision=decision,
                        goal=goal,
                        step=decision_step,
                    )
                    self.memory.record_observation(
                        perception=step_perception,
                        agent_state=agent_state,
                        step=decision_step,
                    )
                self._save_scene_interpretation(
                    step=decision_step,
                    source="decision",
                    interpretation={
                        "status": decision.get("status"),
                        "scene_text": step_perception.get("scene_text")
                        if isinstance(step_perception, dict)
                        else None,
                        "selected_memory": memory_context,
                        "selected_memory_stats": memory_context_stats,
                        "action_trace": self._action_trace(history),
                        "task_target": decision.get("task_target"),
                        "target_attributes": decision.get(
                            "target_attributes", []
                        ),
                        "completion_mode": decision.get("completion_mode"),
                        "completion_condition": decision.get(
                            "completion_condition"
                        ),
                        "requires_target_confirmation": decision.get(
                            "requires_target_confirmation"
                        ),
                        "target_visible": decision.get("target_visible"),
                        "target_location": decision.get("target_location"),
                        "target_correction": decision.get(
                            "target_correction"
                        ),
                        "target_distance": decision.get("target_distance"),
                        "target_evidence_source": decision.get(
                            "target_evidence_source"
                        ),
                        "target_held": decision.get("target_held"),
                        "completion_inconsistency": decision.get(
                            "completion_inconsistency", False
                        ),
                        "completion_inconsistency_source": decision.get(
                            "completion_inconsistency_source"
                        ),
                        "completion_rejection_source": decision.get(
                            "completion_rejection_source"
                        ),
                        "completion_retry_parser_result": decision.get(
                            "completion_retry_parser_result"
                        ),
                        "model_reason": decision.get("model_reason"),
                    "status_normalized": bool(
                        decision.get("status_normalized")
                        or decision.get("completion_retry_status_normalized")
                    ),
                        "completion_retry_attempted": decision.get(
                            "completion_retry_attempted", False
                        ),
                        "completion_retry_status": decision.get(
                            "completion_retry_status"
                        ),
                        "completion_retry_resolved": decision.get(
                            "completion_retry_resolved", False
                        ),
                        "completion_promoted_by_perception": decision.get(
                            "completion_promoted_by_perception", False
                        ),
                        "visual_evidence": decision.get("visual_evidence"),
                        "requested_action": decision.get("requested_action"),
                        "selected_action": decision.get("action"),
                        "action_purpose": decision.get("action_purpose"),
                        "collision_safe_actions": decision.get(
                            "collision_safe_actions"
                        ),
                        "action_guard": decision.get("action_guard"),
                        "impossible_basis": decision.get("impossible_basis"),
                        "impossible_evidence": decision.get("impossible_evidence"),
                        "area_explored": decision.get("area_explored"),
                        "exploration_evidence": decision.get("exploration_evidence"),
                        "area_mapped": decision.get("area_mapped"),
                        "mapping_evidence": decision.get("mapping_evidence"),
                        "exploration_summary": self._exploration_summary(
                            history,
                            agent_state,
                            turn_angle_degrees=self._active_turn_angle_degrees,
                        ),
                        "mapping_summary": self._mapping_summary(
                            history,
                            step_perception,
                            decision.get("visual_evidence"),
                        ),
                        "impossible_rejected": decision.get("original_status")
                        == "impossible",
                        "reason": decision.get("reason"),
                        "parser_result": decision.get("parser_result"),
                        "movement_policy": decision.get("movement_policy"),
                        "target_approach_applied": decision.get(
                            "target_approach_applied", False
                        ),
                        "target_approach_persisted": decision.get(
                            "target_approach_persisted", False
                        ),
                        "target_reacquisition_attempted": decision.get(
                            "target_reacquisition_attempted", False
                        ),
                        "target_reacquisition_confirmed": decision.get(
                            "target_reacquisition_confirmed", False
                        ),
                    },
                    model_call=decision_call,
                )
                self._save_scene_interpretation(
                    step=decision_step,
                    source="completion_check",
                    interpretation=completion_check,
                    model_call=completion_check_call,
                )
                self._add_metrics(inference_totals, decision_call)
                self._add_metrics(inference_totals, completion_check_call)

                if decision["status"] in {"completed", "impossible"}:
                    final_status = decision["status"]
                    final_reason = decision["reason"]
                    history.append(
                        {
                            "step": decision_step,
                            "timestamp": datetime.now().isoformat(),
                            "observation": step_perception,
                            "observation_image_path_relative": step_image_path,
                            "memory_supplied": memory_context,
                            "memory_context_stats": memory_context_stats,
                            "environment_success_check": latest_ground_truth,
                            "finish_check": decision,
                            "action": None,
                            "executed": False,
                            "reason": decision["reason"],
                            "model_calls": self._present_calls(
                                perception_call,
                                reacquisition_call,
                                decision_call,
                                evidence_conflict_call,
                                completion_check_call,
                            ),
                            "duration_seconds": round(
                                time.perf_counter() - step_start, 5
                            ),
                        }
                    )
                    self._notify_progress(progress_callback, history)
                    break

                outcome = self._execute(agent, decision["action"])
                step_record = {
                    "step": decision_step,
                    "timestamp": datetime.now().isoformat(),
                    "observation": step_perception,
                    "observation_image_path_relative": step_image_path,
                    "memory_supplied": memory_context,
                    "memory_context_stats": memory_context_stats,
                    "environment_success_check": latest_ground_truth,
                    "finish_check": {
                        "status": decision["status"],
                        "evidence": decision["visual_evidence"],
                        "reason": decision["reason"],
                    },
                    "visual_evidence": decision["visual_evidence"],
                    "target_evidence_source": decision.get(
                        "target_evidence_source"
                    ),
                    "action": decision["action"],
                    "action_purpose": decision.get("action_purpose"),
                    "requested_action": decision.get("requested_action"),
                    "reason": decision["reason"],
                    "parser_result": decision["parser_result"],
                    "model_parse_result": decision.get("model_parse_result"),
                    "model_output_valid": decision.get("model_output_valid"),
                    "decision_raw_response": decision.get("raw_response"),
                    "action_guard": decision.get("action_guard"),
                    "collision_safe_actions": decision.get(
                        "collision_safe_actions"
                    ),
                    "movement_policy": decision.get("movement_policy"),
                    "controller_intervention": decision.get(
                        "controller_intervention"
                    ),
                    "completion_inconsistency": decision.get(
                        "completion_inconsistency", False
                    ),
                    "completion_inconsistency_source": decision.get(
                        "completion_inconsistency_source"
                    ),
                    "completion_rejection_source": decision.get(
                        "completion_rejection_source"
                    ),
                    "completion_retry_parser_result": decision.get(
                        "completion_retry_parser_result"
                    ),
                    "model_reason": decision.get("model_reason"),
                    "evidence_text_claim": decision.get("evidence_text_claim", False),
                    "status_normalized": bool(
                        decision.get("status_normalized")
                        or decision.get("completion_retry_status_normalized")
                    ),
                    "completion_retry_attempted": decision.get(
                        "completion_retry_attempted", False
                    ),
                    "completion_retry_status": decision.get(
                        "completion_retry_status"
                    ),
                    "completion_retry_resolved": decision.get(
                        "completion_retry_resolved", False
                    ),
                    "completion_promoted_by_perception": decision.get(
                        "completion_promoted_by_perception", False
                    ),
                    "fallback_used": decision.get("fallback_used", False),
                    "target_approach_applied": decision.get(
                        "target_approach_applied", False
                    ),
                    "target_approach_replaced_action": decision.get(
                        "target_approach_replaced_action", False
                    ),
                    "target_approach_persisted": decision.get(
                        "target_approach_persisted", False
                    ),
                    "target_overshoot_corrected": decision.get(
                        "target_overshoot_corrected", False
                    ),
                    "search_turn_oscillation_replaced": decision.get(
                        "search_turn_oscillation_replaced", False
                    ),
                    "target_loss_recovery_applied": decision.get(
                        "target_loss_recovery_applied", False
                    ),
                    "recent_sighting_reversal_allowed": decision.get(
                        "recent_sighting_reversal_allowed", False
                    ),
                    "target_side_flip_rejected": decision.get(
                        "target_side_flip_rejected", False
                    ),
                    "unlocated_sighting_approach": decision.get(
                        "unlocated_sighting_approach", False
                    ),
                    "memory_guided_search_held": decision.get(
                        "memory_guided_search_held", False
                    ),
                    "memory_guided_destination": decision.get(
                        "memory_guided_destination"
                    ),
                    "memory_destination_abandoned": decision.get(
                        "memory_destination_abandoned", False
                    ),
                    "memory_destination_suspended": decision.get(
                        "memory_destination_suspended", False
                    ),
                    "memory_destination_blocked": decision.get(
                        "memory_destination_blocked", False
                    ),
                    "target_correction_contradiction_ignored": decision.get(
                        "target_correction_contradiction_ignored", False
                    ),
                    "target_reacquisition_attempted": decision.get(
                        "target_reacquisition_attempted", False
                    ),
                    "target_reacquisition_confirmed": decision.get(
                        "target_reacquisition_confirmed", False
                    ),
                    "evidence_conflict_check_attempted": decision.get(
                        "evidence_conflict_check_attempted", False
                    ),
                    "evidence_conflict_check_confirmed": decision.get(
                        "evidence_conflict_check_confirmed", False
                    ),
                    "high_level_plan": decision.get("high_level_plan"),
                    "executed": True,
                    **outcome,
                    "model_calls": self._present_calls(
                        perception_call,
                        reacquisition_call,
                        decision_call,
                        evidence_conflict_call,
                        completion_check_call,
                    ),
                    "duration_seconds": round(
                        time.perf_counter() - step_start, 5
                    ),
                }
                history.append(step_record)
                self.memory.record_step(
                    step_record=self._memory_safe_step_record(
                        step_record, decision
                    ),
                    perception=step_perception,
                    agent_state=outcome.get("state_after"),
                )
                self._notify_progress(progress_callback, history)
                print(
                    f"Step {decision_step}: {decision['action']} -> "
                    f"{outcome['outcome']} ({decision['reason']})"
                )
            else:
                final_status = "action_limit"
                final_reason = f"The {max_steps}-action limit was reached."

        except KeyboardInterrupt:
            final_status = "cancelled"
            final_reason = "The task was cancelled by the user."
        except Exception as error:
            final_status = "error"
            final_reason = f"{type(error).__name__}: {error}"
        finally:
            try:
                final_image_path = agent.save_current_view(prefix="final")
            except Exception as error:
                final_reason += f" Final image error: {error}"

        ground_truth = self._evaluate_success(
            success_evaluator,
            history=history,
        )
        if ground_truth is None:
            ground_truth = latest_ground_truth
        report = self._build_report(
            goal=goal,
            status=final_status,
            reason=final_reason,
            started_at=started_at,
            duration=time.perf_counter() - task_start,
            history=history,
            inference_totals=inference_totals,
            final_image_path=final_image_path,
            episode_metadata=episode_metadata or {},
            ground_truth=ground_truth,
            task_contract=task_contract,
            completion_consistency=completion_consistency,
        )
        history_path = self.save_history(report)
        report["history_path"] = str(history_path)
        self.print_history(report)
        return report

    def _prepare_perception(self, image, goal, step, agent, task_contract=None):
        if self.perceptual_interface == "direct_visual":
            return None, None

        task_contract = task_contract or TaskContract.build(goal)
        prompt = f"""
You are a visual perception module for an embodied agent.
Goal: {goal}
Fixed task contract (do not reinterpret it):
{self._json_text(task_contract.as_dict(), compact=True)}
Inspect only the attached CURRENT RGB image. Do not copy this instruction or
reuse an older image. Replace every value with your own observation.

Return exactly one JSON object and no Markdown. Use this structure:
{{
  "scene_summary": "",
  "objects": [
    {{
      "name": "",
      "location": "left|center|right",
      "distance": "near|medium|far|unknown",
      "attributes": []
    }}
  ],
  "spatial_relations": [],
  "obstacles": [],
  "target_visible": false,
  "target_location": "unknown",
  "target_correction": "unknown",
  "target_distance": "unknown",
  "task_progress": ""
}}

The scene_summary is the concise natural-language scene text that will be
passed to the planning module. It must always be a complete, grounded sentence:
describe what is visible, or say that the current view is unclear and no target
is confirmed. Use empty lists, false, or "unknown" for the other fields when
the image does not provide evidence. Do not output
phrases such as "one short current-view summary", "short relation", or
"left|center|right" as an observation. The target and stopping condition are
owned by the fixed task contract. Report only current visual evidence for that
contract; do not substitute a different object or completion rule.
If an object matches excluded_target_terms in the fixed contract, describe it
as a non-target and never use it to satisfy the completion condition.
{self.TARGET_VISIBILITY_RULES}
Put observed appearance details in each object's attributes. When an object is
visible, add an item with a name, location, distance, and attributes. Use the
exact key location (camera_view_location is also accepted for compatibility).
Use location values left, center, or right, and distance values near, medium,
far, or unknown, judged the same way as target_distance.
Keep scene_summary grounded in the image rather than the task label. Do not
describe a requested target as present unless the current image supports it.
Describe the requested target only if its visible appearance supports that
conclusion.
task_progress is an observation note only: do not claim that the agent stopped,
completed the task, or performed an action.
Keep the response compact: scene_summary must be at most 20 words; put the
target candidate first; include at most 2 objects, 1 spatial_relation, and 1
obstacle; and keep task_progress to at most 10 words. Do not repeat the
contract or describe irrelevant decor.
""".strip()
        result = self.backend.generate(
            prompt=prompt,
            image=image,
            max_new_tokens=self.perception_max_new_tokens,
        )
        parsed, parser_result = self._parse_json(
            result.text,
            required_keys=self.PERCEPTION_REQUIRED_KEYS,
        )
        perception_calls = [
            self._call_record(
                "perception",
                result,
                parser_result=parser_result,
                selected=False,
            )
        ]

        if not self._is_perception_payload(parsed):
            retry_prompt = self._perception_retry_prompt(goal, task_contract)
            retry_result = self.backend.generate(
                prompt=retry_prompt,
                image=image,
                max_new_tokens=self.perception_retry_max_new_tokens,
            )
            retry_parsed, retry_parser_result = self._parse_json(
                retry_result.text,
                required_keys=self.PERCEPTION_REQUIRED_KEYS,
            )
            perception_calls.append(
                self._call_record(
                    "perception_retry",
                    retry_result,
                    parser_result=retry_parser_result,
                    selected=self._is_perception_payload(retry_parsed),
                )
            )
            if self._is_perception_payload(retry_parsed):
                parsed = retry_parsed
                parser_result = "perception_retry_valid"
            else:
                parsed = {}
                parser_result = "invalid_perception_after_retry"
        else:
            perception_calls[0]["selected"] = True

        if not self._is_perception_payload(parsed):
            parsed = {
                "scene_summary": (
                    "The current view could not be parsed reliably; no target "
                    "is confirmed."
                ),
                "objects": [],
                "spatial_relations": [],
                "obstacles": [],
                "task_target": "",
                "completion_mode": "unknown",
                "completion_condition": "",
                "requires_target_confirmation": False,
                "target_visible": False,
                "target_location": "unknown",
                "target_correction": "unknown",
                "target_distance": "unknown",
                "task_progress": "unknown",
            }
        else:
            parsed.setdefault("spatial_relations", [])
            parsed.setdefault("obstacles", [])
            parsed.setdefault("target_location", "unknown")
            parsed.setdefault("target_correction", "unknown")
            parsed.setdefault("target_distance", "unknown")
            parsed.setdefault("task_progress", "unknown")
        parser_result = self._clean_perception(parsed, parser_result)
        if parser_result == "template_echo":
            parsed["scene_summary"] = (
                "The current view does not provide a reliable scene summary; "
                "no target is confirmed."
            )
            parsed["objects"] = []
            parsed["spatial_relations"] = []
            parsed["obstacles"] = []
            parsed["target_visible"] = False
            parsed["target_location"] = "unknown"
            parsed["target_correction"] = "unknown"
        elif not isinstance(parsed.get("scene_summary"), str) or not parsed[
            "scene_summary"
        ].strip():
            parsed["scene_summary"] = (
                "The current view is unclear; no target is confirmed from the "
                "available visual evidence."
            )
        parsed["unlocated_sighting_allowed"] = self._sighting_setting(
            "accept_unlocated"
        )
        parsed["completion_policy"] = self._completion_policy
        if self._normalize_perception(parsed, goal, task_contract=task_contract):
            if parser_result == "valid":
                parser_result = "perception_normalized"
        parsed["parser_result"] = parser_result
        parsed["perception_parser_applicable"] = True
        parsed["perception_step"] = step
        parsed["scene_text"] = self._scene_text_from_perception(parsed)

        if self.perceptual_interface == "coela":
            parsed["coela_semantic_map"] = self._coela_semantic_map(agent, parsed)
        return parsed, perception_calls

    def _perception_retry_prompt(self, goal, task_contract):
        return f"""
Inspect the attached CURRENT RGB image again.
Goal: {goal}
Fixed task contract: {self._json_text(task_contract.as_dict(), compact=True)}

The previous response was not a complete top-level JSON object. Return exactly
one compact JSON object and no Markdown using this schema:
{{"scene_summary":"max 12 words","objects":[{{"name":"","location":"left|center|right","distance":"near|medium|far|unknown","attributes":[]}}],"target_visible":false,"target_location":"left|center|right|unknown","target_correction":"none|left|right|unknown","target_distance":"near|medium|far|unknown","task_progress":"max 8 words"}}

Include at most two visible objects. Use target_aliases as acceptable names for
the target class. Preserve visible target colors and identifying text in
attributes. Set target_visible true only when the current image supports the
target class or an alias and all required target_attributes. Determine visible
attributes from the pixels rather than copying them from the goal. A candidate
matching an excluded_target_terms color is not the target. Finish the JSON
object before adding any detail. target_location is descriptive. Set
target_correction to none when the target is fully visible and can be approached
without turning; use left or right only when the target is clipped by or very
close to that edge.
""".strip()

    def _reacquire_target(self, image, goal, step, task_contract):
        state = (
            self._target_approach_state
            if isinstance(self._target_approach_state, dict)
            else {}
        )
        prompt = f"""
Inspect only the attached CURRENT RGB image.
Goal: {goal}
Fixed task contract: {self._json_text(task_contract.as_dict(), compact=True)}

The previous frame contained a confirmed target on the
{state.get("last_location", "unknown")}, and the agent then executed
{state.get("last_action", "a turn")} to align with it. This is a verification
check, not evidence that the target remains visible. Ignore memory and inspect
the current pixels again.

Return exactly one compact JSON object and no Markdown:
{{"scene_summary":"max 12 words","objects":[{{"name":"","location":"left|center|right","distance":"near|medium|far|unknown","attributes":[]}}],"target_visible":false,"target_location":"left|center|right|unknown","target_correction":"none|left|right|unknown","target_distance":"near|medium|far|unknown"}}

Include at most one object. Set target_visible true only when the object class
or an allowed alias and every required target attribute are visibly supported.
Determine attributes from the image; do not copy them from the goal. If the
candidate is ambiguous, absent, or matches an excluded term, return false and
an empty objects list. Set target_correction to none when the target is fully
visible and forward motion can approach it; use left or right only when it is
clipped by or very close to that edge. Finish the JSON before adding detail.
""".strip()
        result = self.backend.generate(
            prompt=prompt,
            image=image,
            max_new_tokens=min(self.perception_retry_max_new_tokens, 96),
        )
        parsed, parser_result = self._parse_json(
            result.text,
            required_keys=self.PERCEPTION_REQUIRED_KEYS,
        )
        if not self._is_perception_payload(parsed):
            parsed = {
                "scene_summary": (
                    "The bounded target reacquisition check was inconclusive."
                ),
                "objects": [],
                "target_visible": False,
                "target_location": "unknown",
                "target_correction": "unknown",
                "target_distance": "unknown",
            }
            parser_result = "invalid_target_reacquisition"
        else:
            parsed.setdefault("target_location", "unknown")
            parsed.setdefault("target_correction", "unknown")
            parsed.setdefault("target_distance", "unknown")
            parser_result = self._clean_perception(parsed, parser_result)
            if parser_result == "template_echo":
                parsed = {
                    "scene_summary": (
                        "The target reacquisition response copied its schema."
                    ),
                    "objects": [],
                    "target_visible": False,
                    "target_location": "unknown",
                    "target_correction": "unknown",
                    "target_distance": "unknown",
                }
            else:
                parsed["unlocated_sighting_allowed"] = self._sighting_setting(
                    "accept_unlocated"
                )
                parsed["completion_policy"] = self._completion_policy
                self._normalize_perception(
                    parsed,
                    goal,
                    task_contract=task_contract,
                )
                parser_result = "target_reacquisition_valid"

        parsed["unlocated_sighting_allowed"] = self._sighting_setting(
            "accept_unlocated"
        )
        parsed["completion_policy"] = self._completion_policy
        self._normalize_perception(
            parsed,
            goal,
            task_contract=task_contract,
        )
        parsed["spatial_relations"] = []
        parsed["obstacles"] = []
        parsed["task_progress"] = (
            "Target reacquired."
            if self._perception_confirms_target(goal, parsed)
            else "Target not reacquired."
        )
        parsed["parser_result"] = parser_result
        parsed["perception_parser_applicable"] = True
        parsed["perception_step"] = step
        parsed["target_reacquisition"] = True
        parsed["scene_text"] = self._scene_text_from_perception(parsed)
        call = self._call_record(
            "target_reacquisition",
            result,
            parser_result=parser_result,
            selected=self._perception_confirms_target(goal, parsed),
        )
        return parsed, call

    def _decision_response_schema(self, task_contract):
        schema = {
            "status": "",
            "action": None,
            "action_purpose": "other",
            "reason": "",
        }
        if self.perceptual_interface == "direct_visual":
            schema["visual_evidence"] = ""
        if task_contract.completion_mode == "target_visible":
            schema.update(
                {
                    "target_visible": False,
                    "target_location": "unknown",
                    "target_correction": "unknown",
                    "target_evidence_source": "unknown",
                }
            )
        elif task_contract.completion_mode == "target_near":
            schema.update(
                {
                    "target_visible": False,
                    "target_location": "unknown",
                    "target_correction": "unknown",
                    "target_distance": "unknown",
                    "target_evidence_source": "unknown",
                }
            )
        elif task_contract.completion_mode == "target_held":
            schema["target_held"] = False
        if (
            "target_location" in schema
            and self._sighting_setting("request_horizontal_location")
        ):
            schema["target_location"] = "left|center|right|unknown"
        if (
            "target_distance" in schema
            and self._sighting_setting("request_distance_estimate")
        ):
            schema["target_distance"] = "near|medium|far|unknown"
        return schema

    def _sighting_field_rules(self):
        lines = []
        if self._sighting_setting("request_horizontal_location"):
            lines.append(
                "target_location is where the target appears across the "
                "image: left, center, or right. It is never a surface such "
                "as floor or wall."
            )
        if not lines:
            return ""
        return "\nField notes: " + " ".join(lines)

    def _decide(
        self,
        image,
        perception,
        goal,
        memory_context,
        blocked_action,
        action_count,
        max_steps,
        current_state,
        recent_history,
        task_contract=None,
        memory_context_stats=None,
    ):
        task_contract = task_contract or TaskContract.build(goal)
        decision_schema = self._decision_response_schema(task_contract)
        available, action_guard = self._guarded_actions(
            recent_history,
            blocked_action,
            allowed_actions=self._current_action_options(
                task_contract.allowed_actions
            ),
            turn_angle_degrees=self._active_turn_angle_degrees,
        )
        if self.memory_mode == "none":
            prompt_available = self._current_action_options(
                task_contract.allowed_actions
            )
            prompt_guard = ""
        else:
            prompt_available = available
            prompt_guard = action_guard
        observation, decision_image = self._observation_for_prompt(image, perception)
        camera_pitch = (
            (current_state.get("camera_angles") or {}).get("pitch", "unknown")
            if isinstance(current_state, dict)
            else "unknown"
        )
        camera_rule = ""
        if available & self.CAMERA_ACTIONS:
            camera_rule = (
                "\n8. Camera pitch is "
                f"{camera_pitch} degrees. Use look_down only when a low target "
                "may be below the frame, and look_up only when an elevated "
                "target may be above it. Do not use vertical looks for routine "
                "horizontal exploration; after one vertical look, move or turn "
                "before using another."
            )
        if self.perceptual_interface == "direct_visual":
            output_concision_rule = (
                "9. If continuing, choose exactly one available action and do not "
                "repeat a blocked action. Keep reason and visual_evidence concise "
                "and grounded.\n"
                "10. Keep the entire response compact: reason is at most 12 words "
                "and visual_evidence is at most 16 words. Do not repeat the goal, "
                "contract, memory, or available actions."
            )
        else:
            output_concision_rule = (
                "9. If continuing, choose exactly one available action and do not "
                "repeat a blocked action. Keep reason to at most 12 words. Do not "
                "restate the observation, goal, contract, memory, or available actions."
            )
        output_concision_rule += self._sighting_field_rules()
        visibility_rules = (
            self.TARGET_VISIBILITY_RULES + "\n"
            if self.perceptual_interface == "direct_visual"
            else ""
        )
        landmark_rule = ""
        if self.memory_mode in {"semantic", "combined"}:
            landmark_rule = (
                "8a. Memory may list task_landmarks or a target_sighting that "
                "match the goal, each with relative_to_agent: distance_m and "
                "the turn (left or right, in degrees) from the agent's "
                "current heading to that remembered pose. It is recomputed "
                "every step. If the goal target is not visible, turn toward "
                "it by about turn_degrees (one body turn is the configured "
                "turn angle), then move_forward until distance_m is small; "
                "when at_pose is true, scan from there. A remembered pose "
                "is a hint, not proof; stop only on current direct visual "
                "evidence.\n"
            )
        prompt = f"""
You control one embodied TDW agent for one step.
Goal: {goal}
Fixed contract (immutable): {self._json_text(task_contract.as_dict(), compact=True)}
Current observation:
{observation}
Selected memory: {self._json_text(memory_context, compact=True)}
Available actions: {", ".join(sorted(prompt_available))}
Action guard: {prompt_guard or "none"}

Rules:
1. The contract cannot be renamed or changed. Use only the current observation
   as visual proof; memory landmarks are navigation hints, not proof.
   A semantic target_sighting is an earlier image-grounded navigation hint:
   return to or scan from that recorded pose, but never treat it as proof that
   the target is visible now.
2. A target candidate must match the target class or one target_aliases entry,
   plus every listed target_attributes. An excluded_target_terms look-alike
   never satisfies the contract. For target-visible/near tasks, use
   target_evidence_source as one of direct_view, excluded_lookalike, or unknown.
3. In text-mediated mode, the scene summary is authoritative: do not upgrade
   an unconfirmed target. In direct-visual mode, inspect the attached RGB.
   In hybrid mode, the scene summary decides whether the target is
   confirmed; the attached RGB is context for choosing the action.
4. Choose completed only if the current evidence satisfies completion_mode.
   target_visible: direct current visibility is enough. target_near: it must
   be directly visible and near. Otherwise continue.
5. Body turns are {self._active_turn_angle_degrees:g} degrees. target_location
   describes where the target appears; it does not by itself require a turn.
   If target_correction is none, move forward when safe. Turn left or right
   only when target_correction explicitly requests it. If correction is
   unknown, use the current evidence without assuming every side sighting
   requires a full turn. A confirmed centered target can be approached
   forward even when target_correction was omitted or is unknown. If the
   target leaves view, use the recorded sighting
   as a navigation hint and scan near that pose.
6. If the target is not visible, scan the local headings once, then translate
   to a genuinely new position before repeating a broad scan. Do not keep
   revisiting the same headings at the same position.
7. Choose impossible only after the recorded search supports a concrete
   blocked or infeasible conclusion; an absent target, one failed action, or
   the action limit is not enough. State that concrete basis in reason.
{visibility_rules}{landmark_rule}For every continue action, set action_purpose to search, target_alignment,
target_approach, route_correction, collision_avoidance, or other. Use
route_correction only when current evidence or retained memory indicates that
the prior heading or route was wrong; generic scanning is search.
{camera_rule}
{output_concision_rule}

Return exactly one JSON object, no Markdown, with this schema:
{self._json_text(decision_schema, compact=True)}
status is completed, continue, or impossible. action is null unless status is
continue. Do not add fields from another completion mode.
""".strip()
        result = self.backend.generate(
            prompt=prompt,
            image=decision_image,
            max_new_tokens=self.decision_max_new_tokens,
        )
        data, parser_result = self._parse_decision_response(result.text)
        data = data if isinstance(data, dict) else {}
        model_parse_result = parser_result
        reported_task_spec = {
            key: data.get(key)
            for key in (
                "task_target",
                "target_name",
                "completion_mode",
                "completion_condition",
                "requires_target_confirmation",
                "excluded_target_terms",
                "target_aliases",
                "target_attributes",
            )
            if data.get(key) not in (None, "")
        }
        status = data.get("status")
        if isinstance(status, str):
            status = re.sub(r"[^a-z]+", "", status.strip().lower())
        status, status_normalized = self._normalize_status_word(status)
        status_was_valid = status in {"completed", "continue", "impossible"}
        if status not in {"completed", "continue", "impossible"}:
            status = "continue"
            parser_result = "invalid_status_defaulted_to_continue"
        model_status = status
        task_target = task_contract.task_target
        completion_mode = task_contract.completion_mode
        completion_condition = task_contract.completion_condition
        requires_target_confirmation = (
            task_contract.requires_target_confirmation
        )
        data["task_target"] = task_target
        data["completion_mode"] = completion_mode
        data["completion_condition"] = completion_condition
        data["requires_target_confirmation"] = requires_target_confirmation
        data["excluded_target_terms"] = list(
            task_contract.excluded_target_terms
        )
        data["target_aliases"] = list(task_contract.target_aliases)
        data["target_attributes"] = list(task_contract.target_attributes)
        unlocated_sighting_allowed = self._sighting_setting(
            "accept_unlocated"
        )
        data["unlocated_sighting_allowed"] = unlocated_sighting_allowed
        data["completion_policy"] = self._completion_policy
        target_visible = self._coerce_bool(data.get("target_visible"))
        target_location = self._normalize_location(data.get("target_location"))
        target_correction = self._candidate_target_correction(
            perception,
            data,
        )
        target_distance = self._normalize_distance(data.get("target_distance"))
        target_evidence_source = self._normalize_target_evidence_source(
            data.get("target_evidence_source")
        )
        target_held = self._coerce_bool(data.get("target_held"))
        area_explored = self._coerce_bool(data.get("area_explored")) is True
        exploration_evidence = data.get("exploration_evidence") or ""
        if not isinstance(exploration_evidence, str):
            exploration_evidence = ""
        exploration_evidence = exploration_evidence.strip()
        area_mapped = self._coerce_bool(data.get("area_mapped")) is True
        mapping_evidence = data.get("mapping_evidence") or ""
        if not isinstance(mapping_evidence, str):
            mapping_evidence = ""
        mapping_evidence = mapping_evidence.strip()
        impossible_evidence = data.get("impossible_evidence") or data.get(
            "feasibility_evidence"
        )
        if not isinstance(impossible_evidence, str):
            impossible_evidence = ""
        impossible_evidence = impossible_evidence.strip()
        impossible_basis = self._normalize_impossible_basis(
            data.get("impossible_basis")
        )
        if status == "impossible" and impossible_basis == "none":
            impossible_basis = self._infer_impossible_basis(
                impossible_evidence or data.get("reason")
            )
        if status == "impossible" and not impossible_evidence:
            impossible_evidence = str(data.get("reason") or "").strip()
        requested_action = data.get("action")
        action, was_normalized = self._normalize_action(requested_action)
        action_purpose = self._normalize_action_purpose(
            data.get("action_purpose")
        )
        action_purpose = self._infer_action_purpose(
            action=action,
            action_purpose=action_purpose,
            goal=goal,
            perception=perception,
            blocked_action=blocked_action,
        )
        controller_intervention = None
        fallback_used = False
        if was_normalized and parser_result == "valid":
            parser_result = "normalized_action"
        if status == "continue" and action not in available:
            fallback_used = True
            if action in self.EXECUTABLE_ACTIONS:
                controller_intervention = "guarded_action_replacement"
            else:
                parser_result = "invalid_action_fallback"
            action = self._fallback_action(
                available,
                blocked_action,
                self._recent_actions(recent_history),
            )
        if status != "continue":
            action = None
            action_purpose = "other"
        else:
            action_purpose = self._infer_action_purpose(
                action=action,
                action_purpose=action_purpose,
                goal=goal,
                perception=perception,
                blocked_action=blocked_action,
            )
        evidence_text_claim = False
        if (
            status == "completed"
            and self._claim_on_evidence_text
            and self._completion_requires_visible_target(
                goal, perception, target_visible, decision=data
            )
            and self._evidence_text_names_target(goal, perception, data)
        ):
            evidence_text_claim = True
        if (
            status == "completed"
            and not evidence_text_claim
            and self._completion_requires_visible_target(
                goal,
                perception,
                target_visible,
                decision=data,
            )
        ):
            status = "continue"
            action = self._choose_exploration_action(
                recent_history,
                blocked_action,
                preferred=self._candidate_turn(
                    self._candidate_target_location(perception, data)
                ),
                allowed_actions=available,
                turn_angle_degrees=self._active_turn_angle_degrees,
            )
            parser_result = "completion_rejected_by_perception"
            controller_intervention = "completion_rejected"
            action_purpose = "search"
        visual_evidence = data.get("visual_evidence") or data.get("evidence")
        if not visual_evidence:
            visual_evidence = "No visual evidence returned by the model."
        reason = self._decision_reason(
            data=data,
            status=status,
            action=action,
            requested_action=requested_action,
            parser_result=parser_result,
        )
        return {
            "status": status,
            "task_target": task_target,
            "completion_mode": completion_mode,
            "completion_condition": completion_condition,
            "requires_target_confirmation": requires_target_confirmation,
            "excluded_target_terms": list(task_contract.excluded_target_terms),
            "target_aliases": list(task_contract.target_aliases),
            "target_attributes": list(task_contract.target_attributes),
            "target_visible": target_visible,
            "target_location": target_location,
            "target_correction": target_correction,
            "target_distance": target_distance,
            "target_evidence_source": target_evidence_source,
            "target_held": target_held,
            "unlocated_sighting_allowed": unlocated_sighting_allowed,
            "completion_policy": self._completion_policy,
            "evidence_text_claim": evidence_text_claim,
            "visual_evidence": str(visual_evidence),
            "action": action,
            "action_purpose": action_purpose,
            "requested_action": requested_action,
            "reason": reason,
            "model_reason": str(data.get("reason") or "").strip(),
            "status_normalized": status_normalized,
            "impossible_basis": impossible_basis,
            "impossible_evidence": impossible_evidence,
            "area_explored": area_explored,
            "exploration_evidence": exploration_evidence,
            "area_mapped": area_mapped,
            "mapping_evidence": mapping_evidence,
            "parser_result": parser_result,
            "model_parse_result": model_parse_result,
            "model_output_valid": (
                model_parse_result in {"valid", "plain_text_recovered"}
                and status_was_valid
                and not (
                    model_status == "continue"
                    and self._normalize_action(requested_action)[0]
                    not in self.EXECUTABLE_ACTIONS
                )
            ),
            "reported_task_spec": reported_task_spec,
            "controller_intervention": controller_intervention,
            "fallback_used": fallback_used,
            "action_guard": action_guard,
            "raw_response": result.text,
        }, self._call_record(
            "status_then_action",
            result,
            memory_context_characters=int(
                (memory_context_stats or {}).get("characters") or 0
            ),
            memory_context_budget_characters=int(
                (memory_context_stats or {}).get("budget_characters") or 0
            ),
            memory_context_trimmed=bool(
                (memory_context_stats or {}).get("trimmed", False)
            ),
        )

    def _observation_for_prompt(self, image, perception):
        if self.perceptual_interface == "direct_visual":
            return (
                "The current RGB observation is attached. Inspect the image "
                "directly for this decision.",
                image,
            )

        scene_text = self._scene_text_from_perception(perception)
        target_confirmed = (
            isinstance(perception, dict)
            and self._coerce_bool(perception.get("target_visible")) is True
        )
        target_status = (
            "confirmed by perception"
            if target_confirmed
            else "not confirmed by perception"
        )
        observation = (
            "Authoritative scene summary generated from the current RGB image:\n"
            f"{scene_text}\n\n"
            f"Target status: {target_status}.\n"
            "Use this summary as the current visual evidence. Do not claim the "
            "target is visible when perception has not confirmed it, even if the "
            "task goal names that target. When perception confirms the target, "
            "do not downgrade that confirmation; apply the fixed completion "
            "condition to it."
        )
        if self.perceptual_interface == "hybrid":
            observation += (
                "\nThe current RGB image is also attached. Use it for spatial "
                "context when choosing an action; target confirmation still "
                "comes from the scene summary above."
            )
            return observation, image
        return observation, None

    @classmethod
    def _scene_text_from_perception(cls, perception):
        if not isinstance(perception, dict):
            return "No scene interpretation was produced for the current view."

        parts = []
        existing_scene_text = perception.get("scene_text")
        if isinstance(existing_scene_text, str) and existing_scene_text.strip():
            return existing_scene_text.strip()

        summary = perception.get("scene_summary")
        if isinstance(summary, str) and summary.strip():
            parts.append(summary.strip())

        objects = perception.get("objects") or []
        object_descriptions = []
        for item in objects:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "unknown object").strip()
            location = cls._normalize_location(
                item.get("location")
                or item.get("camera_view")
                or item.get("camera_view_location")
            )
            distance = cls._normalize_distance(item.get("distance"))
            attributes = cls._attribute_values(item.get("attributes"))
            description = name
            if location != "unknown":
                description += f" on the {location}"
            if distance != "unknown":
                description += f" at {distance} distance"
            if attributes:
                description += f" ({', '.join(attributes)})"
            object_descriptions.append(description)
        if object_descriptions:
            parts.append("Visible objects: " + "; ".join(object_descriptions) + ".")

        relations = [str(value).strip() for value in perception.get("spatial_relations", []) or [] if str(value).strip()]
        if relations:
            parts.append("Spatial relations: " + "; ".join(relations) + ".")

        obstacles = [str(value).strip() for value in perception.get("obstacles", []) or [] if str(value).strip()]
        if obstacles:
            parts.append("Obstacles: " + "; ".join(obstacles) + ".")

        if not parts:
            parts.append("No reliable visual scene description was returned for the current view.")

        target = cls._normalize_task_target(perception.get("task_target"))
        if target:
            if cls._coerce_bool(perception.get("target_visible")) is True:
                parts.append(
                    f"Perception confirms the task target {target} is visible."
                )
                correction = cls._normalize_target_correction(
                    perception.get("target_correction")
                )
                if correction == "none":
                    parts.append(
                        "Target alignment: no horizontal turn is needed."
                    )
                elif correction in {"left", "right"}:
                    parts.append(
                        f"Target alignment: a {correction} correction is needed."
                    )
            else:
                parts.append(
                    f"Perception does not confirm the task target {target} "
                    "in this view; references to it are not visual confirmation."
                )
        return " ".join(parts)

    @classmethod
    def _direct_visual_observation(cls, decision, goal, step):
        decision = decision if isinstance(decision, dict) else {}
        target = cls._normalize_task_target(
            decision.get("task_target") or cls._infer_goal_target(goal)
        )
        evidence = str(decision.get("visual_evidence") or "").strip()
        if not evidence:
            evidence = "No reliable visual evidence was returned for the current view."
        scene_text = f"Direct visual planner evidence: {evidence}"
        reported_visible = cls._coerce_bool(decision.get("target_visible")) is True
        target_visible = reported_visible and cls._source_confirms_target(
            goal, decision
        )
        return {
            "scene_summary": evidence,
            "scene_text": scene_text,
            "objects": [],
            "spatial_relations": [],
            "obstacles": [],
            "task_target": target,
            "completion_mode": decision.get("completion_mode", "unknown"),
            "completion_condition": decision.get("completion_condition", ""),
            "requires_target_confirmation": decision.get(
                "requires_target_confirmation", False
            ),
            "excluded_target_terms": decision.get(
                "excluded_target_terms", []
            ),
            "target_aliases": decision.get("target_aliases", []),
            "target_attributes": decision.get("target_attributes", []),
            "target_evidence_source": decision.get(
                "target_evidence_source", "unknown"
            ),
            "target_visible": target_visible,
            "target_location": decision.get("target_location", "unknown"),
            "target_correction": decision.get(
                "target_correction", "unknown"
            ),
            "target_distance": decision.get("target_distance", "unknown"),
            "target_held": decision.get("target_held", False),
            "task_progress": decision.get("reason", ""),
            "direct_visual_evidence": True,
            "unlocated_sighting_allowed": (
                decision.get("unlocated_sighting_allowed") is True
            ),
            "completion_policy": decision.get(
                "completion_policy", cls.DEFAULT_COMPLETION_POLICY
            ),
            "perception_parser_applicable": False,
            "parser_result": "not_applicable",
            "perception_step": step,
        }

    @staticmethod
    def _action_trace(history, limit=8):
        trace = []
        for item in history:
            if not item.get("executed"):
                continue
            trace.append(
                {
                    "step": item.get("step"),
                    "action": item.get("action"),
                    "outcome": item.get("outcome"),
                    "position_changed": item.get("position_changed"),
                    "reason": item.get("reason"),
                }
            )
        return trace[-limit:]

    @staticmethod
    def _coela_semantic_map(agent, perception):
        state = Planner._agent_state(agent)
        return {
            "scene_text": perception.get("scene_text", ""),
            "visible_objects": perception.get("objects", []),
            "visible_obstacles": perception.get("obstacles", []),
            "spatial_relations": perception.get("spatial_relations", []),
            "agent_state": state,
            "task_progress": perception.get("task_progress", "unknown"),
        }

    @staticmethod
    def _execute(agent, action):
        if hasattr(agent, "execute_action"):
            return agent.execute_action(action)
        before = Planner._agent_state(agent)
        result = agent.execute_command(Planner.EXECUTABLE_ACTIONS[action])
        after = Planner._agent_state(agent)
        succeeded = result is not False
        return {
            "outcome": "succeeded" if succeeded else "failed",
            "success": succeeded,
            "state_before": before,
            "state_after": after,
            "position_changed": before.get("position") != after.get("position"),
            "held_object_changed": before.get("held_object") != after.get("held_object"),
        }

    @staticmethod
    def _agent_state(agent):
        if hasattr(agent, "get_state_snapshot"):
            return agent.get_state_snapshot()
        return {
            "position": agent.get_position() if hasattr(agent, "get_position") else None,
            "yaw": agent.get_yaw() if hasattr(agent, "get_yaw") else None,
            "held_object": (
                agent.get_held_object() if hasattr(agent, "get_held_object") else None
            ),
        }

    @staticmethod
    def _evaluate_success(success_evaluator, history=None):
        if success_evaluator is None:
            return None
        try:
            try:
                parameters = inspect.signature(success_evaluator).parameters
                accepts_history = any(
                    parameter.kind
                    in {
                        inspect.Parameter.POSITIONAL_ONLY,
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        inspect.Parameter.VAR_POSITIONAL,
                    }
                    for parameter in parameters.values()
                )
            except (TypeError, ValueError):
                accepts_history = False
            result = (
                success_evaluator(history)
                if accepts_history
                else success_evaluator()
            )
        except Exception as error:
            return {
                "available": False,
                "success": None,
                "reason": f"Success evaluator error: {type(error).__name__}: {error}",
            }
        return result if isinstance(result, dict) else {
            "available": True,
            "success": bool(result),
            "reason": "Boolean environment-side success evaluator.",
        }

    @staticmethod
    def _get_blocked_action(history):
        executed = [item for item in history if item.get("executed")]
        if not executed:
            return None
        previous = executed[-1]
        if previous.get("outcome") in {"blocked", "failed", "failed_or_blocked"}:
            return previous.get("action")
        return None

    @classmethod
    def _normalize_impossible_basis(cls, value):
        if not isinstance(value, str):
            return "none"
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "blocked_or_failed": "blocked",
            "blocked_action": "blocked",
            "impossible": "infeasible",
            "unreachable": "infeasible",
        }
        normalized = aliases.get(normalized, normalized)
        return normalized if normalized in cls.IMPOSSIBLE_BASES else "none"

    @classmethod
    def _infer_impossible_basis(cls, evidence):
        if not isinstance(evidence, str):
            return "none"
        normalized = evidence.lower()
        if any(
            marker in normalized
            for marker in (
                "blocked",
                "failed",
                "no route",
                "no viable",
                "cannot progress",
            )
        ):
            return "blocked"
        if any(
            marker in normalized
            for marker in (
                "unreachable",
                "physically impossible",
                "cannot reach",
                "infeasible",
            )
        ):
            return "infeasible"
        return "none"

    @classmethod
    def _exploration_summary(
        cls,
        history,
        current_state=None,
        turn_angle_degrees=None,
    ):
        turn_angle = float(turn_angle_degrees or cls.TURN_ANGLE_DEGREES)
        headings = set()
        positions = set()
        successful_translations = 0
        executed_actions = 0

        def add_state(state):
            if not isinstance(state, dict):
                return
            yaw = state.get("yaw")
            if isinstance(yaw, (int, float)) and not isinstance(yaw, bool):
                heading_count = int(round(360 / turn_angle))
                heading = int(
                    round((float(yaw) % 360) / turn_angle
                )
                ) % heading_count
                headings.add(heading)
            position = state.get("position")
            if isinstance(position, dict):
                x = position.get("x")
                z = position.get("z")
                if isinstance(x, (int, float)) and isinstance(z, (int, float)):
                    positions.add((round(float(x), 1), round(float(z), 1)))

        for item in history:
            if not item.get("executed"):
                continue
            executed_actions += 1
            add_state(item.get("state_before"))
            add_state(item.get("state_after"))
            if (
                item.get("action") in cls.TRANSLATIONAL_ACTIONS
                and item.get("outcome") == "succeeded"
                and item.get("position_changed", True)
            ):
                successful_translations += 1
        add_state(current_state)

        return {
            "heading_count": len(headings),
            "position_count": len(positions),
            "successful_translations": successful_translations,
            "executed_actions": executed_actions,
            "headings": sorted(headings),
        }

    @classmethod
    def _heading_index_from_yaw(cls, yaw, turn_angle_degrees=None):
        if not isinstance(yaw, (int, float)) or isinstance(yaw, bool):
            return None
        turn_angle = float(turn_angle_degrees or cls.TURN_ANGLE_DEGREES)
        heading_count = int(round(360 / turn_angle))
        return int(round((float(yaw) % 360) / turn_angle)) % heading_count

    @classmethod
    def _last_heading_index(cls, history, turn_angle_degrees=None):
        for item in reversed(history):
            for key in ("state_after", "state_before"):
                state = item.get(key)
                if not isinstance(state, dict):
                    continue
                heading = cls._heading_index_from_yaw(
                    state.get("yaw"),
                    turn_angle_degrees=turn_angle_degrees,
                )
                if heading is not None:
                    return heading
        return None

    @classmethod
    def _next_heading_index(cls, heading, action, turn_angle_degrees=None):
        if heading is None or action not in cls.TURN_ACTIONS:
            return None
        turn_angle = float(turn_angle_degrees or cls.TURN_ANGLE_DEGREES)
        heading_count = int(round(360 / turn_angle))
        direction = 1 if action == "turn_right" else -1
        return (heading + direction) % heading_count

    @classmethod
    def _turn_reaches_unvisited_heading(
        cls,
        history,
        action,
        turn_angle_degrees=None,
    ):
        if action not in cls.TURN_ACTIONS:
            return False
        current_heading = cls._last_heading_index(
            history,
            turn_angle_degrees=turn_angle_degrees,
        )
        if current_heading is None:
            return True
        next_heading = cls._next_heading_index(
            current_heading,
            action,
            turn_angle_degrees=turn_angle_degrees,
        )
        visited = set(
            cls._exploration_summary(
                history,
                turn_angle_degrees=turn_angle_degrees,
            ).get("headings", [])
        )
        return next_heading not in visited

    @classmethod
    def _area_is_explored(
        cls,
        history,
        current_state=None,
        turn_angle_degrees=None,
    ):
        summary = cls._exploration_summary(
            history,
            current_state,
            turn_angle_degrees=turn_angle_degrees,
        )
        return (
            summary["heading_count"]
            >= cls._local_scan_headings_required(turn_angle_degrees)
            or (
                summary["position_count"] >= cls.ROAMING_POSITIONS_REQUIRED
                and summary["heading_count"] >= cls.ROAMING_HEADINGS_REQUIRED
            )
        )

    @classmethod
    def _local_scan_headings_required(cls, turn_angle_degrees=None):
        turn_angle = float(turn_angle_degrees or cls.TURN_ANGLE_DEGREES)
        camera_horizontal_fov = 90.0
        return max(
            1,
            int(math.ceil((360.0 - camera_horizontal_fov) / turn_angle)) + 1,
        )

    @classmethod
    def _mapping_summary(
        cls,
        history,
        current_perception=None,
        current_visual_evidence=None,
    ):
        mapped_views = 0
        entities = set()
        relation_count = 0
        obstacle_count = 0

        def add_observation(observation):
            nonlocal mapped_views, relation_count, obstacle_count
            if not isinstance(observation, dict):
                return
            scene_summary = str(
                observation.get("scene_summary")
                or observation.get("scene_text")
                or ""
            ).strip()
            if scene_summary.lower().startswith(
                (
                    "no reliable visual scene description",
                    "no scene interpretation was produced",
                )
            ):
                scene_summary = ""
            objects = observation.get("objects")
            relations = observation.get("spatial_relations")
            obstacles = observation.get("obstacles")
            meaningful_objects = (
                objects
                if isinstance(objects, list)
                else []
            )
            meaningful_relations = relations if isinstance(relations, list) else []
            meaningful_obstacles = obstacles if isinstance(obstacles, list) else []
            if not (
                scene_summary
                or meaningful_objects
                or meaningful_relations
                or meaningful_obstacles
            ):
                return
            mapped_views += 1
            relation_count += len(meaningful_relations)
            obstacle_count += len(meaningful_obstacles)
            for item in meaningful_objects:
                if isinstance(item, dict):
                    name = str(item.get("name") or "").strip().lower()
                else:
                    name = str(item).strip().lower()
                if name:
                    entities.add(name)

        def add_visual_evidence(evidence):
            nonlocal mapped_views
            if not isinstance(evidence, str):
                return
            normalized = evidence.strip().lower()
            if normalized and normalized not in {
                "no visual evidence returned by the model.",
                "no evidence supplied.",
            }:
                mapped_views += 1

        for item in history:
            if not item.get("executed"):
                continue
            observation = item.get("observation")
            add_observation(observation)
            if observation is None:
                add_visual_evidence(item.get("visual_evidence"))

        add_observation(current_perception)
        if current_perception is None:
            add_visual_evidence(current_visual_evidence)

        return {
            "mapped_views": mapped_views,
            "entity_count": len(entities),
            "relation_count": relation_count,
            "obstacle_count": obstacle_count,
        }

    @classmethod
    def _area_is_mapped(
        cls,
        history,
        current_perception=None,
        current_state=None,
        current_visual_evidence=None,
        turn_angle_degrees=None,
    ):
        mapping = cls._mapping_summary(
            history,
            current_perception,
            current_visual_evidence,
        )
        coverage = cls._exploration_summary(
            history,
            current_state,
            turn_angle_degrees=turn_angle_degrees,
        )
        return (
            mapping["mapped_views"] >= 3
            and (
                coverage["heading_count"] >= 3
                or coverage["position_count"] >= 2
            )
            and (
                mapping["entity_count"] >= 1
                or mapping["relation_count"] >= 1
                or mapping["obstacle_count"] >= 1
            )
        )

    @classmethod
    def _impossible_is_justified(
        cls,
        decision,
        history,
        blocked_action,
        perception=None,
        current_state=None,
        turn_angle_degrees=None,
    ):
        if not isinstance(decision, dict):
            return False

        evidence = decision.get("impossible_evidence")
        if not isinstance(evidence, str):
            evidence = decision.get("reason", "")
        evidence = str(evidence or "").strip()
        exploration_evidence = decision.get("exploration_evidence")
        if not isinstance(exploration_evidence, str):
            exploration_evidence = ""
        exploration_evidence = exploration_evidence.strip()
        if len(evidence) < 12 and len(exploration_evidence) < 12:
            return False

        normalized = evidence.lower()
        area_explored = cls._area_is_explored(
            history,
            current_state,
            turn_angle_degrees=turn_angle_degrees,
        )
        model_confirmed_exploration = (
            cls._coerce_bool(decision.get("area_explored")) is True
        )
        exploration_claim = model_confirmed_exploration or bool(
            exploration_evidence
        ) or any(
            marker in normalized
            for marker in (
                "area explored",
                "area was explored",
                "all headings",
                "all directions",
                "scanned the area",
                "multiple positions",
            )
        )
        mapping_evidence = decision.get("mapping_evidence")
        if not isinstance(mapping_evidence, str):
            mapping_evidence = ""
        mapping_evidence = mapping_evidence.strip()
        area_mapped = cls._area_is_mapped(
            history,
            current_perception=perception,
            current_state=current_state,
            current_visual_evidence=decision.get("visual_evidence"),
            turn_angle_degrees=turn_angle_degrees,
        )
        model_confirmed_mapping = (
            cls._coerce_bool(decision.get("area_mapped")) is True
        )
        mapping_claim = model_confirmed_mapping or bool(
            mapping_evidence
        ) or any(
            marker in normalized
            for marker in (
                "area mapped",
                "semantic map",
                "multiple views",
                "spatial relations",
                "mapped observations",
            )
        )

        if (area_explored and exploration_claim) or (
            area_mapped and mapping_claim
        ):
            return True
        return False

    def _apply_movement_policy(
        self,
        decision,
        goal,
        perception,
        history,
        blocked_action,
        environment_evidence=None,
        visible_target_approach=False,
        target_approach_persistence_steps=0,
        environment_completion_enabled=False,
        memory_context=None,
        target_loss_recovery=None,
    ):
        if not isinstance(decision, dict):
            return decision

        target_goal = self._requires_target_condition(
            goal, perception, decision
        )
        target_confirmed = self._target_is_confirmed(
            goal, perception, decision
        )
        if decision.get("evidence_text_claim") and decision.get("status") == "completed":
            target_confirmed = True
        environment_satisfied = bool(
            isinstance(environment_evidence, dict)
            and environment_evidence.get("available")
            and environment_evidence.get("independent", True)
            and environment_evidence.get("success")
        )
        environment_completion_required = bool(
            environment_completion_enabled
        )
        environment_blocks_completion = bool(
            environment_completion_required and not environment_satisfied
        )
        perception_confirms_target = self._perception_confirms_target(
            goal, perception
        )
        recovery_enabled = bool(
            isinstance(target_loss_recovery, dict)
            and target_loss_recovery.get("enabled")
        )
        executed_actions = [
            item
            for item in (history or [])
            if isinstance(item, dict) and item.get("executed")
        ]
        if recovery_enabled and perception_confirms_target:
            sighting_side = self._candidate_target_location(
                perception, decision
            )
            if self._is_side_flip_after_translation(
                self._last_target_sighting, sighting_side, executed_actions
            ):
                perception_confirms_target = False
                decision["target_side_flip_rejected"] = True
            else:
                self._last_target_sighting = {
                    "side": (
                        sighting_side
                        if sighting_side in {"left", "right"}
                        else None
                    ),
                    "executed_count": len(executed_actions),
                    "recovery_used": False,
                    "reversal_used": False,
                }
        if decision.get("status") != "continue" and not (
            decision.get("status") == "completed"
            and target_goal
            and (not target_confirmed or environment_blocks_completion)
        ):
            decision["action"] = None
            decision["action_purpose"] = "other"
            return decision
        available, _ = self._guarded_actions(
            history,
            blocked_action,
            allowed_actions=self._current_action_options(),
            turn_angle_degrees=self._active_turn_angle_degrees,
        )
        previous_pursuit = self._target_approach_state or {}
        if visible_target_approach and (
            perception_confirms_target
            or previous_pursuit.get("remaining", 0) > 0
        ):
            pursuit_actions = self._current_action_options() & (
                self.TURN_ACTIONS | self.TRANSLATIONAL_ACTIONS
            )
            if blocked_action:
                pursuit_actions.discard(blocked_action)
            restored = pursuit_actions - available
            available.update(pursuit_actions)
            if restored:
                guard = str(decision.get("action_guard") or "").strip()
                detail = (
                    "search repetition limits do not interrupt a validated "
                    "target approach; collision-safe actions restored: "
                    + ", ".join(sorted(restored))
                )
                decision["action_guard"] = (
                    f"{guard}; {detail}" if guard else detail
                )
        requested_action = decision.get("action")
        action, _ = self._normalize_action(requested_action)
        action_was_guarded = False
        if action not in available:
            action_was_guarded = True
            decision["controller_intervention"] = (
                "guarded_action_replacement"
            )
            decision["fallback_used"] = True
            action = self._fallback_action(
                available,
                blocked_action,
                self._recent_actions(history),
            )
        decision["action_purpose"] = self._infer_action_purpose(
            action=action,
            action_purpose=(
                "other"
                if action_was_guarded
                else decision.get("action_purpose")
            ),
            goal=goal,
            perception=perception,
            blocked_action=blocked_action,
        )

        policy_reason = None
        if (
            decision.get("status") == "completed"
            and target_goal
            and (not target_confirmed or environment_blocks_completion)
        ):
            decision["original_status"] = "completed"
            decision["status"] = "continue"
            decision["completion_retry_resolved"] = False
            decision["reason"] = (
                "The model claimed completion, but the independent completion "
                "gate rejected it."
                if environment_blocks_completion
                else "The model claimed completion without sufficient current "
                "target confirmation."
            )
            if environment_blocks_completion:
                decision["parser_result"] = (
                    "completion_rejected_by_environment"
                )
                decision["completion_rejection_source"] = "environment"
            elif decision.get("parser_result") == "valid":
                decision["parser_result"] = "completion_rejected_by_policy"
                decision["completion_rejection_source"] = "perception"
            decision["controller_intervention"] = "completion_rejected"
            decision["action_purpose"] = "search"
            action = self._choose_exploration_action(
                history,
                blocked_action,
                preferred=self._candidate_turn(
                    self._candidate_target_location(perception, decision)
                ),
                allowed_actions=self._current_action_options(),
                turn_angle_degrees=self._active_turn_angle_degrees,
            )
            fallback = decision.get("pre_claim_action")
            if (
                decision.get("evidence_text_claim")
                and not perception_confirms_target
                and fallback in available
            ):
                action = fallback
                decision["action_purpose"] = "search"
            if (
                environment_blocks_completion
                and perception_confirms_target
                and "move_forward" in available
                and (
                    self._candidate_target_location(perception, decision)
                    in {"center", "left", "right"}
                    or self._is_unlocated_sighting(perception, decision)
                )
            ):
                action = "move_forward"
                decision["action_purpose"] = "target_approach"
            if environment_blocks_completion:
                policy_reason = (
                    "completion was rejected because the independent environment "
                    "criteria are not yet satisfied; continue approaching or "
                    f"exploring with {action}"
                )
            else:
                policy_reason = (
                    "completion was rejected because the target was not confirmed; "
                    f"turning to inspect with {action}"
                )

        if decision.get("status") != "continue":
            return decision

        recovery_turn = None
        if (
            recovery_enabled
            and visible_target_approach
            and not perception_confirms_target
        ):
            recovery_turn = self._side_loss_recovery_turn(
                executed_actions, target_loss_recovery, blocked_action
            )
        approach_state_before = self._target_approach_state

        if (
            visible_target_approach
            and perception_confirms_target
        ):
            location = self._candidate_target_location(perception, decision)
            correction = self._candidate_target_correction(
                perception,
                decision,
            )
            if correction == "unknown" and location == "center":
                correction = "none"
            elif location == "center" and correction in {"left", "right"}:
                correction = "none"
            elif (
                location in {"left", "right"}
                and correction in {"left", "right"}
                and correction != location
                and self._sighting_setting("ignore_contradictory_correction")
            ):
                decision["target_correction_contradiction_ignored"] = True
                correction = "unknown"
            decision["target_correction"] = correction
            previous_state = (
                self._target_approach_state
                if isinstance(self._target_approach_state, dict)
                else {}
            )
            desired_action = None
            movement_purpose = None
            force_forward_next = False

            if previous_state.get("force_forward_after_correction"):
                desired_action = "move_forward"
                movement_purpose = "target_approach"
            elif correction == "none":
                desired_action = "move_forward"
                movement_purpose = "target_approach"
            elif correction in {"left", "right"}:
                desired_action = self._candidate_turn(correction)
                movement_purpose = "target_alignment"
                previous_turn = previous_state.get("last_action")
                if (
                    previous_state.get("action_purpose")
                    == "target_alignment"
                    and previous_turn
                    == self._opposite_turn(desired_action)
                    and self._last_successful_action(history) == previous_turn
                ):
                    movement_purpose = "overshoot_correction"
                    force_forward_next = True
            if desired_action in available:
                replaced = action != desired_action
                action = desired_action
                decision["action_purpose"] = movement_purpose
                decision["target_approach_applied"] = True
                decision["target_approach_replaced_action"] = replaced
                decision["target_overshoot_corrected"] = (
                    movement_purpose == "overshoot_correction"
                )
                if replaced:
                    if decision.get("controller_intervention"):
                        decision["secondary_controller_intervention"] = (
                            "target_approach_replacement"
                        )
                    else:
                        decision["controller_intervention"] = (
                            "target_approach_replacement"
                        )
                policy_reason = (
                    "the configured sequential target-approach policy used "
                    f"target_correction={correction} with {desired_action}"
                )
                self._target_approach_state = {
                    "remaining": (
                        min(target_approach_persistence_steps, 1)
                        if desired_action == "move_forward"
                        else 0
                    ),
                    "last_action": desired_action,
                    "last_location": location,
                    "last_correction": correction,
                    "action_purpose": movement_purpose,
                    "force_forward_after_correction": force_forward_next,
                    "reacquisition_attempted": False,
                }
            elif correction == "unknown":
                decision["action_purpose"] = self._infer_action_purpose(
                    action=action,
                    action_purpose=decision.get("action_purpose"),
                    goal=goal,
                    perception=perception,
                    blocked_action=blocked_action,
                )
                candidate_turn = self._candidate_turn(location)
                if (
                    location in {"left", "right"}
                    and "move_forward" in available
                    and action != "move_forward"
                ):
                    action = "move_forward"
                    decision["action_purpose"] = "target_approach"
                    decision["target_approach_applied"] = True
                    decision["target_approach_replaced_action"] = True
                    decision.setdefault(
                        "controller_intervention",
                        "target_approach_replacement",
                    )
                    policy_reason = (
                        "the target is confirmed at the side with an unknown "
                        "correction; approaching with move_forward instead of "
                        "a search scan"
                    )
                    self._target_approach_state = {
                        "remaining": 0,
                        "last_action": action,
                        "last_location": location,
                        "last_correction": "unknown",
                        "action_purpose": "target_approach",
                        "force_forward_after_correction": False,
                        "reacquisition_attempted": False,
                    }
                elif (
                    location == "unknown"
                    and self._is_unlocated_sighting(perception, decision)
                    and "move_forward" in available
                ):
                    replaced = action != "move_forward"
                    action = "move_forward"
                    decision["action_purpose"] = "target_approach"
                    decision["target_approach_applied"] = True
                    decision["target_approach_replaced_action"] = replaced
                    decision["unlocated_sighting_approach"] = True
                    if replaced:
                        decision.setdefault(
                            "controller_intervention",
                            "target_approach_replacement",
                        )
                    policy_reason = (
                        "the target is confirmed by the model's evidence text "
                        "but has no horizontal position; approaching with "
                        "move_forward as for a centered sighting"
                    )
                    self._target_approach_state = {
                        "remaining": min(
                            target_approach_persistence_steps, 1
                        ),
                        "last_action": action,
                        "last_location": location,
                        "last_correction": "unknown",
                        "action_purpose": "target_approach",
                        "force_forward_after_correction": False,
                        "reacquisition_attempted": False,
                        "unlocated": True,
                    }
                elif action == candidate_turn:
                    self._target_approach_state = {
                        "remaining": 0,
                        "last_action": action,
                        "last_location": location,
                        "last_correction": "unknown",
                        "action_purpose": "target_alignment",
                        "force_forward_after_correction": False,
                        "reacquisition_attempted": False,
                    }
        elif recovery_turn is not None:
            replaced = action != recovery_turn
            side = self._last_target_sighting["side"]
            self._last_target_sighting["recovery_used"] = True
            action = recovery_turn
            decision["action_purpose"] = "target_alignment"
            decision["target_loss_recovery_applied"] = True
            if replaced:
                if decision.get("controller_intervention"):
                    decision["secondary_controller_intervention"] = (
                        "target_loss_side_recovery"
                    )
                else:
                    decision["controller_intervention"] = (
                        "target_loss_side_recovery"
                    )
            policy_reason = (
                f"the target was last confirmed at the {side} of the frame "
                f"and is no longer visible, so turning {side} with "
                f"{recovery_turn} instead of continuing past it"
            )
            self._target_approach_state = {
                "remaining": 0,
                "last_action": recovery_turn,
                "last_location": side,
                "last_correction": side,
                "action_purpose": "target_alignment",
                "force_forward_after_correction": False,
                "reacquisition_attempted": False,
            }
        elif (
            visible_target_approach
            and isinstance(self._target_approach_state, dict)
            and self._target_approach_state.get("remaining", 0) > 0
            and (
                self._target_approach_state.get("last_location") == "center"
                or self._target_approach_state.get("unlocated") is True
            )
            and self._target_approach_state.get("last_action")
            == "move_forward"
            and (not history or history[-1].get("success", True))
        ):
            desired_action = "move_forward"
            if desired_action in available:
                replaced = action != desired_action
                action = desired_action
                self._target_approach_state["remaining"] -= 1
                self._target_approach_state["last_action"] = desired_action
                self._target_approach_state["action_purpose"] = (
                    "target_approach"
                )
                decision["target_approach_applied"] = True
                decision["target_approach_persisted"] = True
                decision["target_approach_replaced_action"] = replaced
                decision["action_purpose"] = "target_approach"
                if replaced:
                    if decision.get("controller_intervention"):
                        decision["secondary_controller_intervention"] = (
                            "target_approach_persistence"
                        )
                    else:
                        decision["controller_intervention"] = (
                            "target_approach_persistence"
                        )
                policy_reason = (
                    "the configured sequential target-approach policy "
                    "continued a recently confirmed pursuit with move_forward"
                )
            else:
                self._target_approach_state = None
        elif visible_target_approach and not perception_confirms_target:
            self._target_approach_state = None

        if (
            recovery_enabled
            and visible_target_approach
            and perception_confirms_target
            and self._target_approach_state is approach_state_before
        ):
            current_location = self._candidate_target_location(
                perception, decision
            )
            self._target_approach_state = {
                "remaining": 0,
                "last_action": action,
                "last_location": current_location,
                "last_correction": self._candidate_target_correction(
                    perception, decision
                ),
                "action_purpose": (
                    decision.get("action_purpose") or "target_approach"
                ),
                "force_forward_after_correction": False,
                "reacquisition_attempted": False,
            }

        if perception_confirms_target:
            self._last_confirmed_sighting_count = len(executed_actions)
        guided = None
        if self._memory_guided_hold_active(
            executed_actions, perception_confirms_target
        ):
            decision["memory_guided_search_held"] = True
        else:
            guided = self._memory_guided_action(
                memory_context=memory_context,
                available=available,
                perception_confirms_target=perception_confirms_target,
                executed_count=len(executed_actions),
                decision=decision,
            )
        if guided is not None:
            action, guided_reason = guided
            decision["action_purpose"] = "route_correction"
            decision["memory_guided_action"] = True
            if decision.get("controller_intervention"):
                decision["secondary_controller_intervention"] = (
                    "memory_guided_search"
                )
            else:
                decision["controller_intervention"] = "memory_guided_search"
            policy_reason = guided_reason

        if (
            not perception_confirms_target
            and action in self.TURN_ACTIONS
            and decision.get("action_purpose") == "search"
            and self._is_search_turn_reversal(history, action)
        ):
            replacement = action
            if recovery_enabled and self._recent_sighting_allows_reversal(
                executed_actions, target_loss_recovery
            ):
                self._last_target_sighting["reversal_used"] = True
                decision["recent_sighting_reversal_allowed"] = True
                policy_reason = (
                    "the target was confirmed within the last "
                    f"{target_loss_recovery['reversal_window_actions']} "
                    f"actions, so the reversing search turn {action} is "
                    "allowed once"
                )
            else:
                replacement = self._search_turn_replacement(
                    history=history,
                    requested_action=action,
                    available=available,
                    blocked_action=blocked_action,
                )
            if replacement != action:
                action = replacement
                decision["search_turn_oscillation_replaced"] = True
                if decision.get("controller_intervention"):
                    decision["secondary_controller_intervention"] = (
                        "search_turn_oscillation_replacement"
                    )
                else:
                    decision["controller_intervention"] = (
                        "search_turn_oscillation_replacement"
                    )
                decision["action_purpose"] = (
                    "search"
                    if replacement in self.TURN_ACTIONS
                    else "other"
                )
                policy_reason = (
                    "a generic search turn would immediately reverse the prior "
                    f"search turn, so the controller selected {replacement}"
                )

        if policy_reason:
            decision["movement_policy"] = policy_reason
            existing_reason = str(decision.get("reason") or "").strip()
            decision["reason"] = (
                f"{existing_reason} Movement policy: {policy_reason}"
                if existing_reason
                else f"Movement policy: {policy_reason}"
            )
        decision["action"] = action
        return decision

    def _memory_guided_action(
        self, memory_context, available, perception_confirms_target,
        executed_count=None, decision=None,
    ):
        if (
            not self.memory_guided_search
            or perception_confirms_target
            or self.memory_mode not in {"semantic", "combined"}
            or not isinstance(memory_context, dict)
        ):
            return None
        semantic = memory_context.get("semantic_memory")
        if not isinstance(semantic, dict):
            return None
        landmarks = [
            item
            for item in (semantic.get("task_landmarks") or [])
            if isinstance(item, dict)
        ]
        sighting = semantic.get("target_sighting")
        if not isinstance(sighting, dict):
            sighting = None
        if self._memory_guided_destination == "landmark_first":
            chosen = self._landmark_first_destination(
                landmarks, sighting, executed_count,
                agent_cell=self._agent_cell(semantic),
            )
            if chosen is None:
                return None
            kind, item = chosen
            relative = item["relative_to_agent"]
            key = self._destination_key(kind, item)
            if isinstance(decision, dict):
                decision["memory_guided_destination"] = kind
        else:
            candidates = landmarks + ([sighting] if sighting else [])
            relative = None
            key = None
            for item in reversed(candidates):
                if isinstance(item.get("relative_to_agent"), dict):
                    relative = item["relative_to_agent"]
                    break
            if relative is None:
                return None
        try:
            distance = float(relative.get("distance_m"))
            degrees = float(relative.get("turn_degrees"))
        except (TypeError, ValueError):
            return None
        side = relative.get("turn")
        if side not in {"left", "right"}:
            return None
        turn_angle = max(float(self._active_turn_angle_degrees), 1.0)
        turn_action = "turn_right" if side == "right" else "turn_left"
        needs_turn = degrees > turn_angle / 2.0
        if distance < 1.0 and not relative.get("at_pose"):
            return None
        if distance < 1.0 and key is not None:
            self._destination_dwell[key] = self._destination_dwell.get(key, 0) + 1
            if self._destination_dwell[key] > self.DESTINATION_MAX_DWELL:
                self._abandoned_destinations.add(key)
                if isinstance(decision, dict):
                    decision["memory_destination_abandoned"] = True
                return None
        if distance >= 1.0 and key is not None:
            best, stalled = self._destination_progress.get(key, (distance, -1))
            if distance < best - self.DESTINATION_PROGRESS_METERS:
                best, stalled = distance, 0
            else:
                stalled += 1
            self._destination_progress[key] = (min(best, distance), stalled)
            if stalled >= self.DESTINATION_PATIENCE:
                self._suspend_destination(key, executed_count, semantic, decision)
                return None
        if distance < 1.0:
            if needs_turn and turn_action in available:
                return turn_action, (
                    "memory-guided search: at the remembered pose, "
                    f"turning {side} to its recorded heading"
                )
            return None
        if needs_turn and turn_action in available:
            return turn_action, (
                f"memory-guided search: remembered pose is {distance:.1f} m "
                f"away, turning {side} toward it"
            )
        if (
            key is not None
            and not needs_turn
            and "move_forward" not in available
        ):
            self._suspend_destination(key, executed_count, semantic, decision,
                                      blocked=True)
            return None
        if not needs_turn and "move_forward" in available:
            return "move_forward", (
                f"memory-guided search: facing the remembered pose "
                f"({distance:.1f} m away), moving forward"
            )
        return None

    @staticmethod
    def _agent_cell(semantic):
        state = (semantic or {}).get("agent_state") or {}
        position = state.get("position") or {}
        try:
            return (round(float(position["x"])), round(float(position["z"])))
        except (KeyError, TypeError, ValueError):
            return None

    def _suspend_destination(self, key, executed_count, semantic, decision,
                             blocked=False):
        self._destination_progress.pop(key, None)
        self._suspended_destinations[key] = {
            "until": (executed_count or 0) + self.DESTINATION_SUSPEND_ACTIONS,
            "cell": self._agent_cell(semantic),
        }
        if isinstance(decision, dict):
            decision["memory_destination_suspended"] = True
            if blocked:
                decision["memory_destination_blocked"] = True

    def _destination_still_suspended(self, key, executed_count, agent_cell):
        entry = self._suspended_destinations.get(key)
        if not entry:
            return False
        if (executed_count or 0) < entry["until"]:
            return True
        if entry["cell"] is not None and agent_cell == entry["cell"]:
            return True
        self._suspended_destinations.pop(key, None)
        return False

    def _landmark_first_destination(self, landmarks, sighting, executed_count,
                                    agent_cell=None):
        since = None
        if executed_count is not None and self._last_confirmed_sighting_count is not None:
            since = executed_count - self._last_confirmed_sighting_count
        ordered = []
        if (
            sighting is not None
            and since is not None
            and since <= self.SIGHTING_REACQUISITION_ACTIONS
        ):
            ordered.append(("sighting", sighting))
        ordered.extend(("landmark", item) for item in reversed(landmarks))
        for kind, item in ordered:
            if not isinstance(item.get("relative_to_agent"), dict):
                continue
            key = self._destination_key(kind, item)
            if key in self._abandoned_destinations:
                continue
            if self._destination_still_suspended(key, executed_count, agent_cell):
                continue
            return kind, item
        return None

    @staticmethod
    def _destination_key(kind, item):
        position = item.get("position") or item.get("agent_position") or {}
        try:
            return (
                kind,
                round(float(position.get("x")) * 4) / 4,
                round(float(position.get("z")) * 4) / 4,
            )
        except (TypeError, ValueError, AttributeError):
            return (kind, str(item.get("task_id") or item.get("step")))

    @classmethod
    def _memory_guided_destination_setting(cls, task_spec):
        if not isinstance(task_spec, dict):
            return cls.DEFAULT_MEMORY_GUIDED_DESTINATION
        value = task_spec.get(
            "memory_guided_destination", cls.DEFAULT_MEMORY_GUIDED_DESTINATION
        )
        if value not in cls.MEMORY_GUIDED_DESTINATIONS:
            raise ValueError(
                "task_spec.memory_guided_destination must be one of: "
                + ", ".join(cls.MEMORY_GUIDED_DESTINATIONS)
            )
        return value

    @staticmethod
    def _visible_target_approach_enabled(task_spec):
        if not isinstance(task_spec, dict):
            return False
        value = task_spec.get("visible_target_approach", False)
        if not isinstance(value, bool):
            raise ValueError(
                "task_spec.visible_target_approach must be true or false."
            )
        return value

    @staticmethod
    def _opposite_turn(action):
        return {
            "turn_left": "turn_right",
            "turn_right": "turn_left",
        }.get(action)

    @staticmethod
    def _last_successful_action(history):
        for item in reversed(history or []):
            if not isinstance(item, dict) or not item.get("executed"):
                continue
            if item.get("success", True):
                return item.get("action")
            return None
        return None

    @classmethod
    def _is_search_turn_reversal(cls, history, action):
        for item in reversed(history or []):
            if not isinstance(item, dict) or not item.get("executed"):
                continue
            return bool(
                item.get("action") == cls._opposite_turn(action)
                and item.get("action_purpose") == "search"
                and not item.get("position_changed", False)
            )
        return False

    def _side_loss_recovery_turn(self, executed, policy, blocked_action):
        sighting = self._last_target_sighting
        if not isinstance(sighting, dict) or sighting.get("recovery_used"):
            return None
        turn = self._candidate_turn(sighting.get("side"))
        if turn is None or turn == blocked_action:
            return None
        since = executed[int(sighting.get("executed_count", 0)):]
        if not since or len(since) > int(policy["max_actions_since_sighting"]):
            return None
        for item in since:
            if (
                item.get("action") not in self.TRANSLATIONAL_ACTIONS
                or not item.get("success", True)
                or not item.get("position_changed", True)
            ):
                return None
        if turn not in self._current_action_options():
            return None
        return turn

    SIDE_FLIP_WINDOW_ACTIONS = 4

    @classmethod
    def _is_side_flip_after_translation(cls, previous, side, executed):
        if not isinstance(previous, dict):
            return False
        previous_side = previous.get("side")
        if (
            side not in {"left", "right"}
            or previous_side not in {"left", "right"}
            or side == previous_side
        ):
            return False
        since = executed[int(previous.get("executed_count", 0)):]
        if not since or len(since) > cls.SIDE_FLIP_WINDOW_ACTIONS:
            return False
        return all(
            item.get("action") in cls.TRANSLATIONAL_ACTIONS for item in since
        )

    def _recent_sighting_allows_reversal(self, executed, policy):
        sighting = self._last_target_sighting
        if not isinstance(sighting, dict) or sighting.get("reversal_used"):
            return False
        since = len(executed) - int(sighting.get("executed_count", 0))
        return 0 < since <= int(policy["reversal_window_actions"])

    @classmethod
    def _target_loss_recovery_policy(cls, task_spec):
        policy = dict(cls.DEFAULT_TARGET_LOSS_RECOVERY)
        if not isinstance(task_spec, dict):
            return policy
        configured = task_spec.get("target_loss_recovery", False)
        if isinstance(configured, bool):
            policy["enabled"] = configured
            return policy
        if not isinstance(configured, dict):
            raise ValueError(
                "task_spec.target_loss_recovery must be true, false, or an "
                "object."
            )
        enabled = configured.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError(
                "task_spec.target_loss_recovery.enabled must be true or false."
            )
        policy["enabled"] = enabled
        for key in ("max_actions_since_sighting", "reversal_window_actions"):
            value = configured.get(key, policy[key])
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(
                    f"task_spec.target_loss_recovery.{key} must be a "
                    "positive integer."
                )
            policy[key] = value
        return policy

    @classmethod
    def _sighting_handling_policy(cls, task_spec):
        policy = dict(cls.DEFAULT_SIGHTING_HANDLING)
        if not isinstance(task_spec, dict):
            return policy
        if "direct_visual_sightings" in task_spec:
            raise ValueError(
                "task_spec.direct_visual_sightings was renamed "
                "task_spec.sighting_handling and now applies to every "
                "perceptual interface."
            )
        configured = task_spec.get("sighting_handling", False)
        if isinstance(configured, bool):
            return {key: configured for key in policy}
        if not isinstance(configured, dict):
            raise ValueError(
                "task_spec.sighting_handling must be true, false, or an "
                "object."
            )
        unknown = set(configured) - set(policy)
        if unknown:
            raise ValueError(
                "task_spec.sighting_handling has unknown keys: "
                + ", ".join(sorted(unknown))
            )
        for key in policy:
            value = configured.get(key, False)
            if not isinstance(value, bool):
                raise ValueError(
                    f"task_spec.sighting_handling.{key} must be true or "
                    "false."
                )
            policy[key] = value
        return policy

    def _sighting_setting(self, key):
        return bool(self._sighting_handling.get(key))

    @classmethod
    def _claim_on_evidence_text_setting(cls, task_spec, completion_policy):
        if not isinstance(task_spec, dict):
            return False
        value = task_spec.get("claim_on_evidence_text", False)
        if not isinstance(value, bool):
            raise ValueError("task_spec.claim_on_evidence_text must be true or false.")
        if value and completion_policy != "assisted":
            raise ValueError(
                "task_spec.claim_on_evidence_text requires completion_policy 'assisted'."
            )
        return value

    @classmethod
    def _evidence_text_names_target(cls, goal, perception, decision):
        spec = cls._task_spec(perception, decision)
        target = spec["task_target"] or cls._infer_goal_target(goal)
        names = {
            token
            for value in [target] + list(spec.get("target_aliases", ()))
            for token in cls._content_tokens(value)
            if len(token) > 2
        }
        if not names:
            return False
        if isinstance(perception, dict):
            text = perception.get("scene_summary")
        elif isinstance(decision, dict):
            text = decision.get("visual_evidence")
        else:
            text = None
        required = cls._required_target_attribute_tokens(spec)
        excluded = {
            token
            for term in spec.get("excluded_target_terms", ())
            for token in cls._content_tokens(term)
        }
        return any(
            cls._sentence_is_current_sighting(sentence, names, required, excluded)
            for sentence in re.split(r"[.;!?\n]", str(text or ""))
        )

    _HYPOTHETICAL_WORDS = frozenset(
        {"might", "may", "could", "would", "should", "possibly", "possible",
         "potential", "potentially", "perhaps", "maybe", "likely", "probably",
         "seems", "suggest", "suggests", "suggesting", "indicate", "indicates",
         "indicating", "expected", "search", "searching", "looking", "toward",
         "towards", "nearby"}
    )
    _OCCLUSION_WORDS = frozenset(
        {"obstruct", "obstructs", "obstructing", "obstructed", "obscure",
         "obscures", "obscuring", "obscured", "cover", "covers", "covering",
         "covered", "block", "blocks", "blocking", "blocked", "hidden",
         "hiding", "behind"}
    )

    @classmethod
    def _sentence_is_current_sighting(cls, sentence, names, required, excluded):
        words = {
            w.replace("'", "")
            for w in re.findall(r"[a-z']+", str(sentence or "").lower())
        }
        if not words or not (words & names):
            return False
        if required and not required.issubset(cls._content_tokens(sentence)):
            return False
        blocked = (
            cls._NEGATION_WORDS
            | cls._HYPOTHETICAL_WORDS
            | cls._OCCLUSION_WORDS
            | cls._INDIRECT_VIEW_CUES
            | excluded
        )
        return not (words & blocked)

    @classmethod
    def _memory_guided_hold_setting(cls, task_spec):
        if not isinstance(task_spec, dict):
            return 0
        value = task_spec.get("memory_guided_search_hold_actions", 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                "task_spec.memory_guided_search_hold_actions must be a "
                "non-negative integer."
            )
        return value

    def _memory_guided_hold_active(self, executed_actions, perception_confirms_target):
        if (
            perception_confirms_target
            or self._memory_guided_hold_actions <= 0
            or self._last_confirmed_sighting_count is None
        ):
            return False
        since = len(executed_actions) - self._last_confirmed_sighting_count
        return 0 < since <= self._memory_guided_hold_actions

    STATUS_SYNONYMS = {
        "complete": "completed",
    }

    @classmethod
    def _normalize_status_word(cls, status):
        if isinstance(status, str) and status in cls.STATUS_SYNONYMS:
            return cls.STATUS_SYNONYMS[status], True
        return status, False

    @staticmethod
    def _recover_truncated_status(text):
        match = re.search(
            r'"status"\s*:\s*"(completed|complete|continue)"', str(text or ""), re.I
        )
        if not match:
            return None
        return Planner._normalize_status_word(match.group(1).lower())[0]

    @staticmethod
    def _memory_safe_step_record(step_record, decision):
        if step_record.get("completion_rejection_source") != "environment":
            return step_record
        model_reason = ""
        if isinstance(decision, dict):
            model_reason = str(decision.get("model_reason") or "").strip()
        safe = dict(step_record)
        safe["reason"] = model_reason or "Continue."
        return safe

    @classmethod
    def _completion_policy_setting(cls, task_spec, completion_consistency=None):
        if not isinstance(task_spec, dict):
            return cls.DEFAULT_COMPLETION_POLICY
        policy = task_spec.get("completion_policy", cls.DEFAULT_COMPLETION_POLICY)
        if policy not in cls.COMPLETION_POLICIES:
            raise ValueError(
                "task_spec.completion_policy must be one of: "
                + ", ".join(cls.COMPLETION_POLICIES)
            )
        if policy == "assisted" and not (
            isinstance(completion_consistency, dict)
            and completion_consistency.get(
                "require_environment_success_for_completion"
            )
        ):
            raise ValueError(
                "task_spec.completion_policy 'assisted' requires "
                "completion_consistency.require_environment_success_for_completion."
            )
        return policy

    @staticmethod
    def _assisted_completion(*sources):
        return any(
            isinstance(source, dict)
            and source.get("completion_policy") == "assisted"
            for source in sources
        )

    @classmethod
    def _is_unlocated_sighting(cls, perception, decision):
        return bool(
            isinstance(perception, dict)
            and perception.get("unlocated_sighting_allowed") is True
            and cls._candidate_target_location(perception, decision)
            == "unknown"
        )

    @classmethod
    def _search_turn_replacement(
        cls,
        history,
        requested_action,
        available,
        blocked_action,
    ):
        previous_action = cls._opposite_turn(requested_action)
        preference = [
            previous_action,
            "move_forward",
            "move_backward",
            requested_action,
        ]
        for candidate in preference:
            if candidate in available and candidate != blocked_action:
                return candidate
        return requested_action

    @staticmethod
    def _target_reacquisition_after_turn_enabled(task_spec):
        if not isinstance(task_spec, dict):
            return False
        value = task_spec.get("target_reacquisition_after_turn", False)
        if not isinstance(value, bool):
            raise ValueError(
                "task_spec.target_reacquisition_after_turn must be true or "
                "false."
            )
        return value

    @classmethod
    def _evidence_conflict_policy(cls, task_spec):
        policy = dict(cls.DEFAULT_EVIDENCE_CONFLICT_CHECK)
        if not isinstance(task_spec, dict):
            return policy
        configured = task_spec.get("evidence_conflict_check", False)
        if isinstance(configured, bool):
            policy["enabled"] = configured
            return policy
        if not isinstance(configured, dict):
            raise ValueError(
                "task_spec.evidence_conflict_check must be true, false, or an "
                "object."
            )
        enabled = configured.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError(
                "task_spec.evidence_conflict_check.enabled must be true or "
                "false."
            )
        max_calls = configured.get("max_calls", policy["max_calls"])
        if (
            isinstance(max_calls, bool)
            or not isinstance(max_calls, int)
            or max_calls < 1
        ):
            raise ValueError(
                "task_spec.evidence_conflict_check.max_calls must be a "
                "positive integer."
            )
        policy.update({"enabled": enabled, "max_calls": max_calls})
        return policy

    @classmethod
    def _text_names_target(cls, text, target_names):
        for sentence in re.split(r"[.;!?\n]", str(text or "")):
            words = re.findall(r"[a-z']+", sentence.lower())
            normalized = {w.replace("'", "") for w in words}
            if normalized & cls._NEGATION_WORDS:
                continue
            if any(name in normalized for name in target_names):
                return True
        return False

    def _evidence_conflict_candidate(self, decision, perception, task_contract):
        if decision.get("status") != "continue":
            return False
        if task_contract.completion_mode != "target_visible":
            return False
        names = {
            token
            for value in (task_contract.task_target, *task_contract.target_aliases)
            for token in re.findall(r"[a-z]+", str(value).lower())
            if len(token) > 2
        }
        if not names:
            return False
        texts = []
        if perception is None:
            texts.append(decision.get("visual_evidence"))
        else:
            texts.append(perception.get("scene_summary"))
            for item in perception.get("objects") or []:
                if isinstance(item, dict):
                    texts.append(str(item.get("name", "")))
        return any(self._text_names_target(text, names) for text in texts)

    def _check_evidence_conflict(
        self,
        decision,
        perception,
        image,
        goal,
        task_contract,
        policy,
        calls_used,
    ):
        if not policy["enabled"] or calls_used >= policy["max_calls"]:
            return decision, None
        if image is None or not self._evidence_conflict_candidate(
            decision, perception, task_contract
        ):
            return decision, None
        policy_source = (
            perception
            if perception is not None
            else self._direct_visual_observation(
                decision=decision, goal=goal, step=0
            )
        )
        if self._task_condition_satisfied(goal, policy_source, decision):
            return decision, None

        target = task_contract.task_target
        excluded = ", ".join(task_contract.excluded_target_terms) or "none"
        prompt = f"""
Inspect only the attached CURRENT RGB image. Ignore memory and earlier steps.
Fixed task contract: {self._json_text(task_contract.as_dict(), compact=True)}

An earlier answer mentioned the target but did not confirm it. This is a
verification check, not evidence that the target is visible.
Is at least one {target} clearly identifiable in this image right now?
Excluded look-alikes do not count: {excluded}.

Return exactly one compact JSON object and no Markdown:
{{"target_visible":false,"target_location":"left|center|right|unknown","evidence":"6 to 12 words naming what is seen"}}
Set target_visible true only when the pixels clearly show a {target}. A partly
hidden, doubtful, or excluded candidate is false.
""".strip()
        result = self.backend.generate(
            prompt=prompt,
            image=image,
            max_new_tokens=max(self.completion_max_new_tokens, 72),
        )
        data, parser_result = self._parse_json(result.text)
        data = data if isinstance(data, dict) else {}
        visible = self._coerce_bool(data.get("target_visible")) is True
        location = self._normalize_location(data.get("target_location"))
        evidence = str(data.get("evidence") or "").strip()
        source = {
            **task_contract.as_dict(),
            "target_visible": visible,
            "target_location": location,
            "visual_evidence": evidence,
            "scene_summary": evidence,
            "direct_visual_evidence": True,
        }
        confirmed = visible and self._source_confirms_target(goal, source)
        call = self._call_record(
            "evidence_conflict_check",
            result,
            parser_result=parser_result,
            selected=confirmed,
        )
        decision["evidence_conflict_check_attempted"] = True
        decision["evidence_conflict_check_confirmed"] = confirmed
        decision["evidence_conflict_check_evidence"] = evidence
        if confirmed:
            decision["status"] = "completed"
            decision["action"] = None
            decision["target_visible"] = True
            decision["target_location"] = location
            decision["visual_evidence"] = evidence
            decision["parser_result"] = "evidence_conflict_confirmed"
            decision["controller_intervention"] = "evidence_conflict_confirmed"
            decision["reason"] = (
                "Verification call confirmed the target the model had "
                "described but not flagged as visible."
            )
        return decision, call

    def _should_reacquire_target(self, perception, goal, history):
        if self.perceptual_interface == "direct_visual":
            return False
        if self._perception_confirms_target(goal, perception):
            return False
        state = self._target_approach_state
        if not isinstance(state, dict):
            return False
        if state.get("reacquisition_attempted"):
            return False
        alignment_side = state.get("last_correction")
        if alignment_side not in {"left", "right"}:
            alignment_side = state.get("last_location")
        if alignment_side not in {"left", "right"}:
            return False
        expected_action = self._candidate_turn(alignment_side)
        if state.get("last_action") != expected_action:
            return False
        executed = [
            item
            for item in history
            if isinstance(item, dict) and item.get("executed")
        ]
        if not executed:
            return False
        previous = executed[-1]
        return bool(
            previous.get("action") == expected_action
            and previous.get("success", True)
        )

    @staticmethod
    def _target_approach_persistence_steps(task_spec):
        if not isinstance(task_spec, dict):
            return 0
        value = task_spec.get("target_approach_persistence_steps", 0)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                "task_spec.target_approach_persistence_steps must be a "
                "non-negative integer."
            )
        return value

    @staticmethod
    def _collision_aware_filter_enabled(task_spec):
        if not isinstance(task_spec, dict):
            return False
        return bool(task_spec.get("collision_aware_action_filter", False))

    @staticmethod
    def _collision_safe_actions_for_agent(agent, allowed_actions):
        getter = getattr(agent, "get_collision_safe_actions", None)
        if not callable(getter):
            return None
        permitted = set(allowed_actions or [])
        try:
            safe_actions = getter(allowed_actions=sorted(permitted))
        except TypeError:
            safe_actions = getter()
        except Exception:
            return None
        if not isinstance(safe_actions, (list, tuple, set)):
            return None
        return {action for action in safe_actions if action in permitted}

    def _current_action_options(self, allowed_actions=None):
        permitted = set(
            self._active_allowed_actions
            if allowed_actions is None
            else allowed_actions
        )
        if self._collision_safe_actions is None:
            return permitted
        constrained = permitted & self._collision_safe_actions
        return constrained or permitted

    @classmethod
    def _completion_consistency_policy(cls, task_spec):
        policy = dict(cls.DEFAULT_COMPLETION_CONSISTENCY)
        if not isinstance(task_spec, dict):
            return policy
        configured = task_spec.get("completion_consistency", {})
        if configured in (None, ""):
            return policy
        if not isinstance(configured, dict):
            raise ValueError(
                "task_spec.completion_consistency must be an object when set."
            )
        aliases = {
            "retry_on_inconsistency": "retry_on_inconsistency",
            "promote_validated_perception": "promote_validated_perception",
            "recheck_on_environment_success": (
                "require_environment_success_for_completion"
            ),
            "require_environment_success_for_completion": (
                "require_environment_success_for_completion"
            ),
        }
        for configured_key, policy_key in aliases.items():
            if configured_key not in configured:
                continue
            value = configured[configured_key]
            if not isinstance(value, bool):
                raise ValueError(
                    "task_spec.completion_consistency."
                    f"{configured_key} must be true or false."
                )
            policy[policy_key] = value
        return policy

    def _resolve_completion_inconsistency(
        self,
        decision,
        evidence,
        environment_evidence,
        image,
        goal,
        task_contract,
        policy,
    ):
        if not isinstance(decision, dict) or decision.get("status") != "continue":
            return decision, None, None
        perception_condition_satisfied = self._task_condition_satisfied(
            goal, evidence, decision
        )
        evidence_text_trigger = (
            not perception_condition_satisfied
            and self._claim_on_evidence_text
            and self._evidence_text_names_target(
                goal,
                evidence if self.perceptual_interface != "direct_visual" else None,
                decision,
            )
        )
        if not perception_condition_satisfied and not evidence_text_trigger:
            return decision, None, None

        decision["completion_inconsistency"] = True
        decision["completion_inconsistency_source"] = (
            "evidence_text" if evidence_text_trigger else "perception_contract"
        )
        if evidence_text_trigger:
            decision["evidence_text_claim"] = True
        decision["pre_claim_action"] = decision.get("action")
        decision["completion_retry_attempted"] = False
        decision["completion_retry_resolved"] = False
        decision["completion_promoted_by_perception"] = False
        evidence_text = (
            evidence.get("scene_text")
            or evidence.get("scene_summary")
            or evidence.get("visual_evidence")
            or "Validated current evidence satisfies the completion condition."
        ) if isinstance(evidence, dict) else (
            "Validated current evidence satisfies the completion condition."
        )
        check = {
            "task_contract": task_contract.as_dict(),
            "validated_evidence": evidence_text,
            "initial_status": "continue",
            "completion_inconsistency": True,
            "inconsistency_source": decision[
                "completion_inconsistency_source"
            ],
            "retry_attempted": False,
            "promotion_enabled": bool(
                policy["promote_validated_perception"]
            ),
            "environment_evidence_used_for_retry": False,
        }

        retry_status = "continue"
        retry_call = None
        if policy["retry_on_inconsistency"]:
            prompt = f"""
You are verifying whether an embodied-agent task is complete.
Fixed task contract (authoritative and immutable):
{self._json_text(task_contract.as_dict(), compact=True)}
Validated current evidence:
{evidence_text}

Does this evidence satisfy the contract's completion condition right now?
Return exactly one JSON object and no Markdown:
{{"status":"completed|continue","reason":"brief evidence-based reason"}}
Return completed only if the stated completion condition is satisfied now.
Do not require the target to be held, approached, or examined further unless
the fixed completion condition explicitly requires it. Do not rename the
target or substitute an excluded look-alike.
""".strip()
            result = self.backend.generate(
                prompt=prompt,
                image=(
                    image
                    if self.perceptual_interface in {"direct_visual", "hybrid"}
                    else None
                ),
                max_new_tokens=self.completion_max_new_tokens,
            )
            retry_call = self._call_record("completion_check", result)
            retry_data, retry_parser_result = self._parse_json(result.text)
            retry_data = retry_data if isinstance(retry_data, dict) else {}
            if "status" not in retry_data:
                recovered_status = self._recover_truncated_status(result.text)
                if recovered_status is not None:
                    retry_data = {"status": recovered_status}
                    retry_parser_result = "truncated_json_recovered"
            raw_status = retry_data.get("status")
            if isinstance(raw_status, str):
                raw_status = re.sub(r"[^a-z]+", "", raw_status.strip().lower())
            raw_status, retry_status_normalized = self._normalize_status_word(
                raw_status
            )
            if retry_status_normalized:
                decision["completion_retry_status_normalized"] = True
            if raw_status in {"completed", "continue"}:
                retry_status = raw_status
            else:
                retry_parser_result = "invalid_completion_retry_status"
            decision["completion_retry_attempted"] = True
            decision["completion_retry_status"] = retry_status
            decision["completion_retry_parser_result"] = retry_parser_result
            check.update(
                {
                    "retry_attempted": True,
                    "retry_status": retry_status,
                    "retry_parser_result": retry_parser_result,
                    "retry_reason": str(retry_data.get("reason") or "").strip(),
                }
            )

        if retry_status == "completed":
            decision["status"] = "completed"
            decision["action"] = None
            decision["completion_retry_resolved"] = True
            decision["parser_result"] = "completion_retry_confirmed"
            decision["reason"] = (
                "Completion re-check confirmed that the validated current "
                "evidence satisfies the fixed task contract."
            )
            if task_contract.completion_mode == "target_visible":
                decision["target_visible"] = True
            check["resolved"] = True
            return decision, check, retry_call

        if policy["promote_validated_perception"]:
            decision["status"] = "completed"
            decision["action"] = None
            decision["completion_promoted_by_perception"] = True
            decision["controller_intervention"] = (
                "completion_promoted_by_perception"
            )
            decision["reason"] = (
                "Controller-assisted completion promotion: validated current "
                "evidence satisfied the fixed task contract after the model "
                "continued."
            )
            if task_contract.completion_mode == "target_visible":
                decision["target_visible"] = True
            check["promoted"] = True
            return decision, check, retry_call

        check["resolved"] = False
        follow_up = (
            "after the re-check"
            if decision["completion_retry_attempted"]
            else "while the completion retry was disabled"
        )
        decision["reason"] = (
            f"{str(decision.get('reason') or '').strip()} Completion "
            "inconsistency: validated current evidence satisfied the fixed "
            f"task contract, but the model continued {follow_up}."
        ).strip()
        return decision, check, retry_call

    @classmethod
    def _choose_exploration_action(
        cls,
        history,
        blocked_action,
        preferred=None,
        allowed_actions=None,
        turn_angle_degrees=None,
    ):
        available, _ = cls._guarded_actions(
            history,
            blocked_action,
            allowed_actions=allowed_actions,
            turn_angle_degrees=turn_angle_degrees,
        )
        if preferred in available and preferred != blocked_action:
            return preferred

        recent = cls._recent_actions(history)
        summary = cls._exploration_summary(
            history,
            turn_angle_degrees=turn_angle_degrees,
        )
        visited_headings = set(summary.get("headings", []))
        current_heading = cls._last_heading_index(
            history,
            turn_angle_degrees=turn_angle_degrees,
        )

        if (
            current_heading is not None
            and len(visited_headings)
            < cls._local_scan_headings_required(turn_angle_degrees)
        ):
            last_turn = next(
                (
                    action
                    for action in reversed(recent)
                    if action in cls.TURN_ACTIONS
                ),
                None,
            )
            turn_preference = (
                [last_turn, "turn_left" if last_turn == "turn_right" else "turn_right"]
                if last_turn
                else ["turn_right", "turn_left"]
            )
            for action in turn_preference:
                next_heading = cls._next_heading_index(
                    current_heading,
                    action,
                    turn_angle_degrees=turn_angle_degrees,
                )
                if (
                    action in available
                    and action != blocked_action
                    and next_heading not in visited_headings
                ):
                    return action

        if current_heading is None and not recent:
            for action in ("turn_right", "turn_left"):
                if action in available and action != blocked_action:
                    return action

        for action in ("move_forward", "move_backward"):
            if action in available and action != blocked_action:
                return action

        preference = ["turn_right", "turn_left", "move_forward", "move_backward"]
        if recent and recent[-1] == "turn_right":
            preference = ["turn_left", "move_forward", "move_backward", "turn_right"]
        elif recent and recent[-1] == "turn_left":
            preference = ["turn_right", "move_forward", "move_backward", "turn_left"]
        for action in preference:
            if action in available and action != blocked_action:
                return action
        return cls._fallback_action(
            available, blocked_action, recent_actions=recent
        )

    @staticmethod
    def _normalize_task_target(value):
        if isinstance(value, (list, tuple)):
            value = " ".join(str(item) for item in value)
        if not isinstance(value, str):
            return ""
        normalized = re.sub(r"\s+", " ", value.strip())
        if normalized.lower() in {"", "unknown", "none", "null"}:
            return ""
        return normalized

    @classmethod
    def _normalize_completion_mode(cls, value):
        if not isinstance(value, str):
            return "unknown"
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "visible": "target_visible",
            "object_visible": "target_visible",
            "target_found": "target_visible",
            "found": "target_visible",
            "near": "target_near",
            "close": "target_near",
            "target_close": "target_near",
            "held": "target_held",
            "carried": "target_held",
            "pickup": "target_held",
            "reached": "state_reached",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized in cls.COMPLETION_MODES:
            return normalized
        if "near" in normalized or "close" in normalized:
            return "target_near"
        if "visible" in normalized or "found" in normalized:
            return "target_visible"
        if "held" in normalized or "carry" in normalized:
            return "target_held"
        return "unknown"

    @staticmethod
    def _infer_goal_target(goal):
        text = re.sub(r"\s+", " ", str(goal or "").strip())
        if not text:
            return ""
        patterns = (
            r"\b(?:find|locate|identify|spot|search for|look for|go to|approach|"
            r"reach|navigate to|pick up|interact with)\s+(?:a|an|the|one)\s+"
            r"(.+?)(?=\s+(?:and|when|once|until|if|that|which)\b|[,.!?]|$)",
            r"\btarget\s*(?:is|:)\s*(.+?)(?=\s+(?:and|when|once|until|if)\b|[,.!?]|$)",
        )
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                target = Planner._normalize_task_target(match.group(1))
                if target:
                    return target
        return ""

    @staticmethod
    def _infer_goal_completion_mode(goal):
        text = str(goal or "").lower()
        if any(term in text for term in ("near", "close", "within", "approach")):
            return "target_near"
        if any(
            term in text
            for term in ("find", "locate", "identify", "spot", "visible", "see")
        ):
            return "target_visible"
        return "unknown"

    @classmethod
    def _task_spec(cls, *sources):
        spec = {
            "task_target": "",
            "completion_mode": "unknown",
            "completion_condition": "",
            "requires_target_confirmation": None,
            "excluded_target_terms": (),
            "target_aliases": (),
            "target_attributes": (),
        }
        for source in sources:
            if not isinstance(source, dict):
                continue
            target = cls._normalize_task_target(
                source.get("task_target")
                or source.get("target_name")
                or source.get("target")
            )
            if target and not spec["task_target"]:
                spec["task_target"] = target

            mode = cls._normalize_completion_mode(source.get("completion_mode"))
            if mode != "unknown" and spec["completion_mode"] == "unknown":
                spec["completion_mode"] = mode

            condition = source.get("completion_condition")
            if isinstance(condition, str) and condition.strip() and not spec[
                "completion_condition"
            ]:
                spec["completion_condition"] = condition.strip()

            required = cls._coerce_bool(
                source.get("requires_target_confirmation")
            )
            if required is True:
                spec["requires_target_confirmation"] = True
            elif (
                required is False
                and spec["requires_target_confirmation"] is None
            ):
                spec["requires_target_confirmation"] = False

            exclusions = source.get("excluded_target_terms", ())
            if isinstance(exclusions, str):
                exclusions = [exclusions]
            if isinstance(exclusions, (list, tuple, set)):
                for value in exclusions:
                    normalized = cls._normalize_task_target(value).lower()
                    if (
                        normalized
                        and normalized not in spec["excluded_target_terms"]
                    ):
                        spec["excluded_target_terms"] += (normalized,)

            aliases = (
                source.get("target_aliases")
                or source.get("target_class_aliases")
                or ()
            )
            if isinstance(aliases, str):
                aliases = [aliases]
            if isinstance(aliases, (list, tuple, set)):
                for value in aliases:
                    normalized = cls._normalize_task_target(value).lower()
                    if normalized and normalized not in spec["target_aliases"]:
                        spec["target_aliases"] += (normalized,)

            attributes = (
                source.get("target_attributes")
                or source.get("required_target_attributes")
                or source.get("visual_attributes")
                or ()
            )
            if isinstance(attributes, str):
                attributes = [attributes]
            if isinstance(attributes, (list, tuple, set)):
                for value in attributes:
                    normalized = cls._normalize_task_target(value).lower()
                    if normalized and normalized not in spec["target_attributes"]:
                        spec["target_attributes"] += (normalized,)

        if spec["completion_mode"] in {
            "target_visible",
            "target_near",
            "target_held",
        }:
            spec["requires_target_confirmation"] = True
        return spec

    @staticmethod
    def _content_tokens(value):
        ignored = {"a", "an", "the", "one", "target", "object", "item"}
        return {
            token
            for token in re.findall(r"[a-z0-9]+", str(value or "").lower())
            if token not in ignored
        }

    @classmethod
    def _attribute_values(cls, attributes):
        if attributes in (None, ""):
            return []
        if isinstance(attributes, dict):
            source = attributes.values()
        elif isinstance(attributes, (list, tuple, set)):
            source = attributes
        else:
            value = str(attributes).strip()
            return [value] if value else []

        values = []
        for item in source:
            for value in cls._attribute_values(item):
                if value not in values:
                    values.append(value)
        return values

    @classmethod
    def _required_target_attribute_tokens(cls, spec):
        tokens = set()
        for attribute in spec.get("target_attributes", ()):
            tokens.update(cls._content_tokens(attribute))
        return tokens

    @classmethod
    def _item_matches_excluded_target_terms(cls, item, exclusions):
        if not isinstance(item, dict) or not exclusions:
            return False
        attributes = cls._attribute_values(item.get("attributes"))
        item_tokens = cls._content_tokens(
            " ".join(
                [str(item.get("name") or "")]
                + [str(value) for value in attributes]
            )
        )
        return any(
            cls._content_tokens(term).issubset(item_tokens)
            for term in exclusions
            if cls._content_tokens(term)
        )

    @classmethod
    def _goal_has_target(cls, goal):
        return bool(cls._infer_goal_target(goal))

    @classmethod
    def _requires_target_condition(cls, goal, *sources):
        spec = cls._task_spec(*sources)
        if spec["requires_target_confirmation"] is not None:
            return spec["requires_target_confirmation"] is True
        mode = spec["completion_mode"]
        if mode == "unknown":
            mode = cls._infer_goal_completion_mode(goal)
        return mode in {"target_visible", "target_near", "target_held"}

    @classmethod
    def _target_is_confirmed(cls, goal, perception, decision):
        return cls._task_condition_satisfied(goal, perception, decision)

    @staticmethod
    def _last_executed_action(history):
        recent = Planner._recent_actions(history)
        return recent[-1] if recent else None

    @staticmethod
    def _candidate_turn(location):
        return {
            "left": "turn_left",
            "right": "turn_right",
        }.get(location)

    @classmethod
    def _candidate_target_location(cls, perception, decision):
        for source in (perception, decision):
            if not isinstance(source, dict):
                continue
            for key in ("candidate_target_location", "target_location"):
                location = cls._normalize_location(source.get(key))
                if location != "unknown":
                    return location
        return "unknown"

    @classmethod
    def _candidate_target_correction(cls, perception, decision):
        for source in (perception, decision):
            if not isinstance(source, dict):
                continue
            for key in (
                "target_correction",
                "target_turn",
                "alignment_correction",
                "horizontal_correction",
            ):
                correction = cls._normalize_target_correction(source.get(key))
                if correction != "unknown":
                    return correction
        return "unknown"

    @staticmethod
    def _normalize_location(value):
        if not isinstance(value, str):
            return "unknown"
        normalized = value.strip().lower()
        if "left" in normalized:
            return "left"
        if "right" in normalized:
            return "right"
        if "center" in normalized or "centre" in normalized:
            return "center"
        return "unknown"

    @staticmethod
    def _normalize_target_correction(value):
        if not isinstance(value, str):
            return "unknown"
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "none": "none",
            "no": "none",
            "no_turn": "none",
            "aligned": "none",
            "centered": "none",
            "centred": "none",
            "forward": "none",
            "left": "left",
            "turn_left": "left",
            "correct_left": "left",
            "left_edge": "left",
            "right": "right",
            "turn_right": "right",
            "correct_right": "right",
            "right_edge": "right",
            "unknown": "unknown",
            "unclear": "unknown",
        }
        return aliases.get(normalized, "unknown")

    @classmethod
    def _normalize_action_purpose(cls, value):
        if not isinstance(value, str):
            return "other"
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "explore": "search",
            "exploration": "search",
            "scan": "search",
            "align": "target_alignment",
            "alignment": "target_alignment",
            "approach": "target_approach",
            "correct": "route_correction",
            "correction": "route_correction",
            "backtrack": "route_correction",
            "avoid": "collision_avoidance",
            "obstacle_avoidance": "collision_avoidance",
        }
        normalized = aliases.get(normalized, normalized)
        return normalized if normalized in cls.ACTION_PURPOSES else "other"

    @classmethod
    def _infer_action_purpose(
        cls,
        action,
        action_purpose,
        goal,
        perception,
        blocked_action,
    ):
        normalized = cls._normalize_action_purpose(action_purpose)
        if normalized != "other":
            return normalized
        target_visible = cls._perception_confirms_target(goal, perception)
        if action in cls.TURN_ACTIONS:
            if target_visible:
                return "target_alignment"
            if blocked_action:
                return "collision_avoidance"
            return "search"
        if action in cls.TRANSLATIONAL_ACTIONS and target_visible:
            return "target_approach"
        return "other"

    @staticmethod
    def _normalize_target_evidence_source(value):
        if not isinstance(value, str):
            return "unknown"
        normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "direct": "direct_view",
            "directly_visible": "direct_view",
            "direct_view": "direct_view",
            "visible_directly": "direct_view",
            "excluded": "excluded_lookalike",
            "lookalike": "excluded_lookalike",
            "reflection": "excluded_lookalike",
            "mirror": "excluded_lookalike",
            "indirect": "excluded_lookalike",
            "excluded_lookalike": "excluded_lookalike",
        }
        return aliases.get(normalized, "unknown")

    @staticmethod
    def _normalize_distance(value):
        if not isinstance(value, str):
            return "unknown"
        normalized = value.strip().lower()
        if any(term in normalized for term in ("near", "close")):
            return "near"
        if any(term in normalized for term in ("medium", "moderate", "mid")):
            return "medium"
        if any(term in normalized for term in ("far", "distant")):
            return "far"
        return "unknown"

    @classmethod
    def _target_object_matches_goal(cls, goal, item, perception=None):
        if not isinstance(item, dict):
            return False
        spec = cls._task_spec(perception)
        target = spec["task_target"] or cls._infer_goal_target(goal)
        if not target:
            return False
        if cls._item_matches_excluded_target_terms(
            item, spec["excluded_target_terms"]
        ):
            return False
        name = str(item.get("name") or "").lower()
        attributes = cls._attribute_values(item.get("attributes"))
        item_tokens = cls._content_tokens(
            " ".join([name] + [str(value) for value in attributes])
        )
        required_attribute_tokens = cls._required_target_attribute_tokens(spec)
        if required_attribute_tokens and not required_attribute_tokens.issubset(
            item_tokens
        ):
            return False
        target_names = [target] + list(spec.get("target_aliases", ()))
        name_tokens = cls._content_tokens(name)
        for target_name in target_names:
            target_normalized = target_name.lower()
            if target_normalized in name or name in target_normalized:
                return True
            target_tokens = cls._content_tokens(target_name)
            if bool(name_tokens) and (
                name_tokens.issubset(target_tokens)
                or target_tokens.issubset(name_tokens)
            ):
                return True
        return False

    @classmethod
    def _has_strong_target_evidence(cls, goal, item, perception=None):
        if not cls._target_object_matches_goal(goal, item, perception):
            return False

        location = cls._normalize_location(
            item.get("location")
            or item.get("camera_view")
            or item.get("camera_view_location")
        )
        meaningful_attributes = cls._attribute_values(item.get("attributes"))
        return (
            location != "unknown"
            and bool(meaningful_attributes)
        )

    @classmethod
    def _has_textual_target_evidence(cls, goal, source):
        if not isinstance(source, dict):
            return False
        if cls._coerce_bool(source.get("target_visible")) is not True:
            return False

        location = cls._normalize_location(
            source.get("target_location")
            or source.get("candidate_target_location")
        )
        unlocated_ok = (
            location == "unknown"
            and source.get("unlocated_sighting_allowed") is True
        )
        if location == "unknown" and not unlocated_ok:
            return False

        spec = cls._task_spec(source)
        target = spec["task_target"] or cls._infer_goal_target(goal)
        target_token_sets = [
            cls._content_tokens(value)
            for value in [target] + list(spec.get("target_aliases", ()))
            if cls._content_tokens(value)
        ]
        if not target_token_sets:
            return False

        evidence_source = cls._normalize_target_evidence_source(
            source.get("target_evidence_source")
        )
        if evidence_source == "excluded_lookalike":
            return False

        evidence_values = (
            source.get("scene_summary"),
            source.get("visual_evidence"),
        )
        evidence = " ".join(
            value.strip()
            for value in evidence_values
            if isinstance(value, str) and value.strip()
        )
        evidence_tokens = cls._content_tokens(evidence)
        required_attribute_tokens = cls._required_target_attribute_tokens(spec)
        if len(evidence_tokens) < 4:
            return False
        if required_attribute_tokens and not required_attribute_tokens.issubset(
            evidence_tokens
        ):
            return False
        if (
            source.get("direct_visual_evidence") is True
            and cls._direct_visual_evidence_uses_exclusion(
                evidence,
                spec["excluded_target_terms"],
            )
        ):
            return False
        if not any(
            tokens.intersection(evidence_tokens) for tokens in target_token_sets
        ):
            return False
        if unlocated_ok:
            names = {
                token
                for tokens in target_token_sets
                for token in tokens
                if len(token) > 2
            }
            return cls._text_names_target(evidence, names)
        return True

    _INDIRECT_VIEW_CUES = frozenset(
        {"through", "via", "behind", "beyond", "reflected", "mirrored"}
    )

    @classmethod
    def _direct_visual_evidence_uses_exclusion(
        cls, evidence, exclusions, token_radius=4
    ):
        text = " ".join(str(evidence or "").lower().split())
        if not text or not exclusions:
            return False
        tokens = re.findall(r"[a-z0-9]+", text)
        exclusion_token_sets = [
            cls._content_tokens(term)
            for term in exclusions
            if cls._content_tokens(term)
        ]
        if not exclusion_token_sets:
            return False
        for index, token in enumerate(tokens):
            if token not in cls._INDIRECT_VIEW_CUES:
                continue
            nearby = set(
                tokens[max(0, index - token_radius) : index + token_radius + 1]
            )
            if any(term_tokens.issubset(nearby) for term_tokens in exclusion_token_sets):
                return True
        return False

    @classmethod
    def _source_confirms_target(cls, goal, source):
        if not isinstance(source, dict):
            return False
        if any(
            cls._has_strong_target_evidence(goal, item, source)
            for item in source.get("objects", []) or []
        ):
            return True
        return cls._has_textual_target_evidence(goal, source)

    @classmethod
    def _perception_confirms_target(cls, goal, perception):
        return cls._source_confirms_target(goal, perception)

    @classmethod
    def _target_distance(cls, goal, perception=None, decision=None):
        for source in (perception, decision):
            if not isinstance(source, dict):
                continue
            distance = cls._normalize_distance(source.get("target_distance"))
            if distance != "unknown":
                return distance
            for item in source.get("objects", []) or []:
                if cls._target_object_matches_goal(goal, item, source):
                    distance = cls._normalize_distance(item.get("distance"))
                    if distance != "unknown":
                        return distance
        return "unknown"

    @classmethod
    def _task_condition_satisfied(cls, goal, perception, decision=None):
        spec = cls._task_spec(perception, decision)
        mode = spec["completion_mode"]
        if mode == "unknown":
            mode = cls._infer_goal_completion_mode(goal)
        requires = spec["requires_target_confirmation"]
        if mode in {
            "target_visible",
            "target_near",
            "target_held",
        }:
            requires = True
        elif requires is None:
            requires = mode in {
                "target_visible",
                "target_near",
                "target_held",
            }
        if not requires:
            return True

        if isinstance(perception, dict):
            visible = cls._perception_confirms_target(goal, perception)
        else:
            visible = cls._source_confirms_target(goal, decision)

        if mode == "target_near":
            if cls._assisted_completion(perception, decision):
                return visible
            distance = cls._target_distance(
                goal,
                perception,
                None if isinstance(perception, dict) else decision,
            )
            return visible and distance == "near"
        if mode == "target_held":
            sources = (
                (perception,)
                if isinstance(perception, dict)
                else (decision,)
            )
            for source in sources:
                if isinstance(source, dict) and cls._coerce_bool(
                    source.get("target_held")
                ) is True:
                    return True
            return False
        if mode == "target_visible":
            return visible
        return True

    @staticmethod
    def _coerce_bool(value):
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "yes", "visible", "confirmed"}:
                return True
            if normalized in {"false", "no", "not visible", "unknown"}:
                return False
        return None

    @staticmethod
    def _fallback_action(available, blocked_action, recent_actions=None):
        recent_actions = recent_actions or []
        if blocked_action == "move_forward":
            preference = [
                "turn_right",
                "turn_left",
                "move_backward",
                "look_down",
                "look_up",
            ]
        elif recent_actions and recent_actions[-1] in {"turn_left", "turn_right"}:
            preference = [
                "move_forward",
                "move_backward",
                "turn_left",
                "turn_right",
                "look_down",
                "look_up",
            ]
        else:
            preference = [
                "move_forward",
                "turn_right",
                "turn_left",
                "move_backward",
                "look_down",
                "look_up",
            ]
        for action in preference:
            if action in available and action != blocked_action:
                return action
        return sorted(available)[0]

    @classmethod
    def _guarded_actions(
        cls,
        history,
        blocked_action,
        allowed_actions=None,
        turn_angle_degrees=None,
    ):
        available = set(allowed_actions or cls.EXECUTABLE_ACTIONS)
        restrictions = []
        if blocked_action:
            available.discard(blocked_action)
            restrictions.append(f"{blocked_action} was blocked or failed")

        recent = cls._recent_actions(history)
        if recent and recent[-1] in cls.CAMERA_ACTIONS:
            available.difference_update(cls.CAMERA_ACTIONS)
            restrictions.append(
                "a vertical camera adjustment just occurred; move or turn "
                "before another vertical look"
            )
        recent_steps = [
            item for item in history
            if item.get("executed") and item.get("action")
        ][-3:]
        translations_are_progressing = bool(
            recent
            and recent[-1] in cls.TRANSLATIONAL_ACTIONS
            and len(recent_steps) == 3
            and all(item.get("position_changed") for item in recent_steps)
        )
        if (
            len(recent) >= 3
            and len(set(recent[-3:])) == 1
            and not translations_are_progressing
        ):
            repeated = recent[-1]
            if repeated in cls.TURN_ACTIONS and cls._turn_reaches_unvisited_heading(
                history,
                repeated,
                turn_angle_degrees=turn_angle_degrees,
            ):
                restrictions.append(
                    f"{repeated} repeats only to reach an unvisited heading"
                )
            else:
                available.discard(repeated)
                restrictions.append(
                    f"{repeated} was selected three times consecutively"
                )

        turn_streak = 0
        for action in reversed(recent):
            if action not in {"turn_left", "turn_right"}:
                break
            turn_streak += 1
        if turn_streak >= cls.MAX_CONSECUTIVE_TURNS_WITHOUT_TRANSLATION:
            available.discard("turn_left")
            available.discard("turn_right")
            restrictions.append(
                "six turns occurred without translational movement; "
                "choose a useful non-turn action or report a justified impossible result"
            )

        if not available:
            available = set(allowed_actions or cls.EXECUTABLE_ACTIONS)
            if blocked_action:
                available.discard(blocked_action)
        return available, "; ".join(restrictions)

    @staticmethod
    def _recent_actions(history):
        return [
            item.get("action")
            for item in history
            if item.get("executed") and item.get("action")
        ]

    @staticmethod
    def _normalize_action(action):
        if not isinstance(action, str):
            return action, False
        normalized = action.strip().lower().replace("-", " ").replace("_", " ")
        normalized = re.sub(r"[^a-z0-9 ]+", "", normalized)
        normalized = "_".join(normalized.split())
        aliases = {
            "w": "move_forward",
            "forward": "move_forward",
            "s": "move_backward",
            "backward": "move_backward",
            "a": "turn_left",
            "left": "turn_left",
            "d": "turn_right",
            "right": "turn_right",
            "q": "look_up",
            "up": "look_up",
            "look_upward": "look_up",
            "camera_up": "look_up",
            "e": "look_down",
            "down": "look_down",
            "look_downward": "look_down",
            "camera_down": "look_down",
            "p": "pick_up",
            "pickup": "pick_up",
            "o": "put_down",
            "putdown": "put_down",
            "walk_forward": "move_forward",
            "walk_backward": "move_backward",
            "go_forward": "move_forward",
            "go_backward": "move_backward",
        }
        canonical = aliases.get(normalized, normalized)
        return canonical, canonical != action

    @staticmethod
    def _clean_perception(parsed, parser_result):
        template_markers = {
            "one short current-view summary",
            "short relation",
            "visible obstacle and location",
            "what the current view proves",
            "left|center|right and near|far",
            "left|center|right",
        }
        values = []
        for key in (
            "scene_summary",
            "task_progress",
            "target_location",
            "target_correction",
        ):
            values.append(str(parsed.get(key, "")).strip().lower())
        for item in parsed.get("objects", []) or []:
            if isinstance(item, dict):
                values.extend(str(value).strip().lower() for value in item.values())
        for key in ("spatial_relations", "obstacles"):
            values.extend(str(value).strip().lower() for value in parsed.get(key, []) or [])

        if any(value in template_markers for value in values):
            return "template_echo"
        return parser_result

    @classmethod
    def _normalize_perception(cls, parsed, goal, task_contract=None):
        task_contract = task_contract or TaskContract.build(goal)
        changed = False
        reported_task_spec = {
            key: parsed.get(key)
            for key in (
                "task_target",
                "target_name",
                "completion_mode",
                "completion_condition",
                "requires_target_confirmation",
                "excluded_target_terms",
                "target_aliases",
                "target_attributes",
            )
            if parsed.get(key) not in (None, "")
        }
        if reported_task_spec:
            parsed["reported_task_spec"] = reported_task_spec
        objects = []
        for item in parsed.get("objects", []) or []:
            if not isinstance(item, dict):
                continue
            normalized = dict(item)
            if "location" not in normalized:
                for key in ("camera_view", "camera_view_location"):
                    if normalized.get(key):
                        normalized["location"] = normalized[key]
                        changed = True
                        break
            normalized_attributes = cls._attribute_values(
                normalized.get("attributes")
            )
            if normalized.get("attributes") != normalized_attributes:
                normalized["attributes"] = normalized_attributes
                changed = True
            objects.append(normalized)
        if objects != parsed.get("objects", []):
            parsed["objects"] = objects
            changed = True

        task_target = task_contract.task_target
        completion_mode = task_contract.completion_mode
        completion_condition = task_contract.completion_condition
        requires_target_confirmation = (
            task_contract.requires_target_confirmation
        )
        excluded_target_terms = list(task_contract.excluded_target_terms)
        target_aliases = list(task_contract.target_aliases)
        target_attributes = list(task_contract.target_attributes)
        if task_target and parsed.get("task_target") != task_target:
            parsed["task_target"] = task_target
            changed = True
        if parsed.get("completion_mode") != completion_mode:
            parsed["completion_mode"] = completion_mode
            changed = True
        if completion_condition and parsed.get("completion_condition") != completion_condition:
            parsed["completion_condition"] = completion_condition
            changed = True
        if parsed.get("requires_target_confirmation") != requires_target_confirmation:
            parsed["requires_target_confirmation"] = requires_target_confirmation
            changed = True
        if parsed.get("excluded_target_terms") != excluded_target_terms:
            parsed["excluded_target_terms"] = excluded_target_terms
            changed = True
        if parsed.get("target_aliases") != target_aliases:
            parsed["target_aliases"] = target_aliases
            changed = True
        if parsed.get("target_attributes") != target_attributes:
            parsed["target_attributes"] = target_attributes
            changed = True

        normalized_target_visible = cls._coerce_bool(
            parsed.get("target_visible")
        )
        if normalized_target_visible is not None and parsed.get(
            "target_visible"
        ) != normalized_target_visible:
            parsed["target_visible"] = normalized_target_visible
            changed = True

        correction_value = parsed.get("target_correction")
        if correction_value in (None, ""):
            for alias in (
                "target_turn",
                "alignment_correction",
                "horizontal_correction",
            ):
                if parsed.get(alias) not in (None, ""):
                    correction_value = parsed.get(alias)
                    changed = True
                    break
        normalized_correction = cls._normalize_target_correction(
            correction_value
        )
        if parsed.get("target_correction") != normalized_correction:
            parsed["target_correction"] = normalized_correction
            changed = True

        target_objects = [
            item
            for item in objects
            if cls._target_object_matches_goal(goal, item, parsed)
        ]
        if target_objects:
            parsed["candidate_target_detected"] = True
            parsed["candidate_target_location"] = target_objects[0].get(
                "location", "unknown"
            )
            changed = True
            target_distance = cls._normalize_distance(
                parsed.get("target_distance")
            )
            strong_target = next(
                (
                    item
                    for item in target_objects
                    if cls._has_strong_target_evidence(goal, item, parsed)
                ),
                None,
            )
            if strong_target is not None:
                target_distance = cls._normalize_distance(
                    strong_target.get("distance")
                )
                parsed["target_visible"] = True
                parsed["target_confirmation_source"] = (
                    "structured_visual_evidence"
                )
                changed = True
            if target_distance != "unknown":
                parsed["target_distance"] = target_distance
                changed = True

            if parsed.get("target_visible") is True:
                location = cls._normalize_location(
                    target_objects[0].get("location")
                )
                if location != "unknown" and parsed.get("target_location") != location:
                    parsed["target_location"] = location
                    changed = True

        if parsed.get("target_visible") is True and not cls._source_confirms_target(
            goal, parsed
        ):
            parsed["target_visible"] = False
            parsed["target_confirmation_source"] = (
                "rejected_unsupported_visibility_claim"
            )
            changed = True
        if parsed.get("target_visible") is not True:
            if parsed.get("target_correction") != "unknown":
                parsed["target_correction"] = "unknown"
                changed = True
        elif (
            parsed.get("target_correction") == "unknown"
            and cls._normalize_location(parsed.get("target_location"))
            == "center"
        ):
            parsed["target_correction"] = "none"
            parsed["target_correction_source"] = "centered_visual_evidence"
            changed = True
        return changed

    @classmethod
    def _completion_requires_visible_target(
        cls,
        goal,
        perception,
        decision_target_visible=None,
        decision=None,
    ):
        spec = cls._task_spec(perception, decision)
        mode = spec["completion_mode"]
        if mode == "unknown":
            mode = cls._infer_goal_completion_mode(goal)
        requires = spec["requires_target_confirmation"]
        if mode in {
            "target_visible",
            "target_near",
            "target_held",
        }:
            requires = True
        elif requires is None:
            requires = mode in {
                "target_visible",
                "target_near",
                "target_held",
            }
        if not requires or mode not in {"target_visible", "target_near"}:
            return False
        if decision is None:
            decision = {"target_visible": decision_target_visible}
        return not cls._task_condition_satisfied(goal, perception, decision)

    @staticmethod
    def _decision_reason(
        data, status, action, requested_action, parser_result
    ):
        if parser_result in {
            "completion_rejected_by_perception",
            "completion_rejected_by_policy",
        }:
            return (
                "The decision model claimed completion, but the fixed task "
                "condition was not supported by current visual evidence; "
                f"continuing with action {action}."
            )
        for key in ("reason", "explanation", "rationale"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        if parser_result == "invalid_action_fallback":
            if requested_action is None:
                return (
                    f"The model did not provide an action; the planner selected "
                    f"the fallback action {action}."
                )
            return (
                f"The model returned unsupported action {requested_action!r}; "
                f"the planner selected the fallback action {action}."
            )
        if parser_result in {"json_not_found", "invalid_json"}:
            return (
                "The model response was not valid JSON; the planner selected "
                f"the fallback action {action}."
            )
        if parser_result == "template_echo":
            return "The perception model copied the response template instead of describing the image."
        return f"The model selected status {status} and action {action}."

    @classmethod
    def _is_perception_payload(cls, value):
        return (
            isinstance(value, dict)
            and isinstance(value.get("scene_summary"), str)
            and isinstance(value.get("objects"), list)
            and cls._coerce_bool(value.get("target_visible")) is not None
        )

    @staticmethod
    def _parse_json(text, required_keys=None):
        text = text or ""
        decoder = json.JSONDecoder()
        saw_object = False
        saw_schema_mismatch = False
        required_keys = set(required_keys or ())
        for index, character in enumerate(text):
            if character != "{":
                continue
            saw_object = True
            try:
                value, _ = decoder.raw_decode(text[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                if required_keys and not required_keys.issubset(value):
                    saw_schema_mismatch = True
                    continue
                return value, "valid"
        if saw_schema_mismatch:
            return {}, "invalid_json_schema"
        return {}, "invalid_json" if saw_object else "json_not_found"

    @classmethod
    def _parse_decision_response(cls, text):
        data, parser_result = cls._parse_json(text)
        if parser_result == "valid":
            return data, parser_result

        recovered = cls._recover_truncated_json_decision(text)
        if recovered is not None:
            return recovered, "truncated_json_recovered"

        raw = text or ""
        status_match = re.search(
            r"(?im)^\s*status\s*:\s*(completed|continue|impossible)\b",
            raw,
        )
        if status_match is None:
            return data, parser_result

        fields = {}
        for line in raw.splitlines():
            field_match = re.match(
                r"^\s*([A-Za-z][A-Za-z0-9 _-]*)\s*:\s*(.*?)\s*$",
                line,
            )
            if field_match is None:
                continue
            field_name = re.sub(
                r"[^a-z0-9]+", "_", field_match.group(1).lower()
            ).strip("_")
            fields[field_name] = field_match.group(2).strip()

        def field(*names):
            for name in names:
                normalized_name = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip(
                    "_"
                )
                if normalized_name in fields:
                    value = fields[normalized_name]
                    if len(value) >= 2 and value[0] == value[-1] in {'"', "'"}:
                        try:
                            parsed_value = json.loads(value)
                            if isinstance(parsed_value, str):
                                value = parsed_value
                        except json.JSONDecodeError:
                            value = value[1:-1]
                    return value
            return ""

        action_value = field("action")
        reason_value = field("reason", "explanation", "rationale")
        evidence_value = field("visual_evidence", "evidence")
        recovered = {
            "status": status_match.group(1).lower(),
            "action": action_value,
            "reason": reason_value,
            "visual_evidence": evidence_value,
        }
        for key in (
            "task_target",
            "target_name",
            "completion_mode",
            "completion_condition",
            "requires_target_confirmation",
            "target_visible",
            "target_location",
            "target_correction",
            "target_distance",
            "target_evidence_source",
            "target_held",
            "action_purpose",
            "impossible_basis",
            "impossible_evidence",
            "area_explored",
            "exploration_evidence",
            "area_mapped",
            "mapping_evidence",
        ):
            value = field(key)
            if value:
                recovered[key] = value
        if recovered["action"] in {"null", "none", "n/a"}:
            recovered["action"] = None
        return recovered, "plain_text_recovered"

    @staticmethod
    def _recover_truncated_json_decision(text):
        raw = text or ""
        start = raw.find("{")
        if start < 0:
            return None
        raw = raw[start:]
        if raw.rstrip().endswith("}"):
            return None

        fields = {}
        field_names = (
            "status",
            "action",
            "reason",
            "visual_evidence",
            "evidence",
            "target_visible",
            "target_location",
            "target_correction",
            "target_distance",
            "target_evidence_source",
            "target_held",
            "action_purpose",
            "impossible_basis",
            "impossible_evidence",
        )
        for field_name in field_names:
            escaped_name = re.escape(field_name)
            string_match = re.search(
                r'"'
                + escaped_name
                + r'"\s*:\s*"((?:\\.|[^"\\])*)"\s*(?=,|\}|$)',
                raw,
                flags=re.DOTALL,
            )
            if string_match is not None:
                try:
                    fields[field_name] = json.loads(
                        '"' + string_match.group(1) + '"'
                    )
                except json.JSONDecodeError:
                    pass
                continue

            scalar_match = re.search(
                r'"'
                + escaped_name
                + r'"\s*:\s*(true|false|null|-?\d+(?:\.\d+)?)\s*(?=,|\}|$)',
                raw,
                flags=re.IGNORECASE,
            )
            if scalar_match is not None:
                try:
                    fields[field_name] = json.loads(
                        scalar_match.group(1).lower()
                    )
                except json.JSONDecodeError:
                    pass

        status = fields.get("status")
        if not isinstance(status, str):
            return None
        status = re.sub(r"[^a-z]+", "", status.strip().lower())
        if status not in {"completed", "continue", "impossible"}:
            return None
        fields["status"] = status
        return fields

    @staticmethod
    def _json_text(value, compact=False):
        if value is None:
            return "None"
        options = {"ensure_ascii": False, "default": str}
        if compact:
            options["separators"] = (",", ":")
        else:
            options["indent"] = 2
        return json.dumps(value, **options)

    @staticmethod
    def _call_record(stage, result, **metadata):
        if result is None:
            return None
        record = {
            "stage": stage,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "inference_seconds": result.inference_seconds,
            "raw_response": result.text,
        }
        for field in (
            "prompt_characters",
            "visual_input_present",
            "preprocess_seconds",
            "time_to_first_token_seconds",
            "decode_seconds",
            "total_call_seconds",
            "prefill_stalled",
            "cuda_free_memory_mb_before",
            "cuda_free_memory_mb_after",
            "cuda_total_memory_mb",
            "cuda_memory_allocated_mb_before",
            "cuda_memory_reserved_mb_before",
            "cuda_memory_allocated_mb",
            "cuda_memory_reserved_mb",
            "cuda_free_memory_mb_after_cleanup",
            "cuda_memory_allocated_mb_after_cleanup",
            "cuda_memory_reserved_mb_after_cleanup",
            "generation_index",
            "max_new_tokens_requested",
            "output_hit_token_limit",
            "generation_timed_out",
            "generation_soft_limit_exceeded",
        ):
            if hasattr(result, field):
                record[field] = getattr(result, field)
        record.update(metadata)
        return record

    @staticmethod
    def _present_calls(*calls):
        present = []
        for call in calls:
            if call is None:
                continue
            if isinstance(call, (list, tuple)):
                present.extend(Planner._present_calls(*call))
            else:
                present.append(call)
        return present

    @staticmethod
    def _add_metrics(totals, call):
        if call is None:
            return
        if isinstance(call, (list, tuple)):
            for item in call:
                Planner._add_metrics(totals, item)
            return
        totals["calls"] += 1
        totals["input_tokens"] += call["input_tokens"]
        totals["output_tokens"] += call["output_tokens"]
        totals["inference_seconds"] += call["inference_seconds"]
        totals["prompt_characters"] += int(call.get("prompt_characters") or 0)
        totals["preprocess_seconds"] += float(
            call.get("preprocess_seconds") or 0
        )
        totals["time_to_first_token_seconds"] += float(
            call.get("time_to_first_token_seconds") or 0
        )
        totals["decode_seconds"] += float(call.get("decode_seconds") or 0)
        totals["max_cuda_memory_allocated_mb_before"] = max(
            totals["max_cuda_memory_allocated_mb_before"],
            float(call.get("cuda_memory_allocated_mb_before") or 0),
        )
        totals["max_cuda_memory_reserved_mb_before"] = max(
            totals["max_cuda_memory_reserved_mb_before"],
            float(call.get("cuda_memory_reserved_mb_before") or 0),
        )
        totals["cuda_total_memory_mb"] = max(
            totals["cuda_total_memory_mb"],
            float(call.get("cuda_total_memory_mb") or 0),
        )
        telemetry_available = float(call.get("cuda_total_memory_mb") or 0) > 0
        free_memory = call.get("cuda_free_memory_mb_before")
        if telemetry_available and free_memory is not None:
            current_minimum = totals["min_cuda_free_memory_mb"]
            totals["min_cuda_free_memory_mb"] = (
                float(free_memory)
                if current_minimum is None
                else min(current_minimum, float(free_memory))
            )
        free_memory_after = call.get("cuda_free_memory_mb_after")
        if telemetry_available and free_memory_after is not None:
            current_minimum = totals["min_cuda_free_memory_mb"]
            totals["min_cuda_free_memory_mb"] = (
                float(free_memory_after)
                if current_minimum is None
                else min(current_minimum, float(free_memory_after))
            )
        totals["max_cuda_memory_allocated_mb"] = max(
            totals["max_cuda_memory_allocated_mb"],
            float(call.get("cuda_memory_allocated_mb") or 0),
        )
        totals["max_cuda_memory_reserved_mb"] = max(
            totals["max_cuda_memory_reserved_mb"],
            float(call.get("cuda_memory_reserved_mb") or 0),
        )
        totals["max_cuda_memory_allocated_mb_after_cleanup"] = max(
            totals["max_cuda_memory_allocated_mb_after_cleanup"],
            float(call.get("cuda_memory_allocated_mb_after_cleanup") or 0),
        )
        totals["max_cuda_memory_reserved_mb_after_cleanup"] = max(
            totals["max_cuda_memory_reserved_mb_after_cleanup"],
            float(call.get("cuda_memory_reserved_mb_after_cleanup") or 0),
        )
        totals["outputs_hitting_token_limit"] += int(
            bool(call.get("output_hit_token_limit"))
        )
        totals["generation_timeouts"] += int(
            bool(call.get("generation_timed_out"))
        )
        totals["generation_soft_limit_exceeded"] += int(
            bool(call.get("generation_soft_limit_exceeded"))
        )

    @classmethod
    def _criterion_result(cls, evaluation, criterion_type):
        if not isinstance(evaluation, dict):
            return None
        if evaluation.get("criterion_type") == criterion_type:
            return evaluation
        for child in evaluation.get("criteria", []) or []:
            match = cls._criterion_result(child, criterion_type)
            if match is not None:
                return match
        return None

    @staticmethod
    def _navigation_progress(history):
        cells, seen, repeats, positions = set(), set(), 0, []
        for item in history or []:
            state = (item or {}).get("state_before") or {}
            position = state.get("position") or {}
            try:
                x, z = float(position["x"]), float(position["z"])
            except (KeyError, TypeError, ValueError):
                continue
            cell = (round(x * 2) / 2, round(z * 2) / 2)
            yaw = state.get("yaw")
            try:
                heading = round(float(yaw) / 45.0) % 8
            except (TypeError, ValueError):
                heading = None
            cells.add(cell)
            if (cell, heading) in seen:
                repeats += 1
            seen.add((cell, heading))
            positions.append((x, z))
        net = (
            math.dist(positions[0], positions[-1]) if len(positions) > 1 else 0.0
        )
        return {
            "distinct_cells_visited": len(cells),
            "repeated_pose_heading_steps": repeats,
            "net_displacement_m": round(net, 3),
        }

    @staticmethod
    def _translation_path_length(records):
        distance = 0.0
        for item in records:
            if not isinstance(item, dict) or not item.get("executed"):
                continue
            before = item.get("state_before", {}).get("position", {})
            after = item.get("state_after", {}).get("position", {})
            try:
                distance += math.hypot(
                    float(after["x"]) - float(before["x"]),
                    float(after["z"]) - float(before["z"]),
                )
            except (KeyError, TypeError, ValueError):
                continue
        return round(distance, 5)

    def _build_report(
        self,
        goal,
        status,
        reason,
        started_at,
        duration,
        history,
        inference_totals,
        final_image_path,
        episode_metadata,
        ground_truth,
        task_contract,
        completion_consistency,
    ):
        actions = [item for item in history if item.get("executed")]
        decisions = []
        for item in history:
            if item.get("executed"):
                decisions.append(item)
                continue
            finish = item.get("finish_check")
            if isinstance(finish, dict) and (
                finish.get("model_parse_result") is not None
                or finish.get("parser_result") is not None
            ):
                decisions.append(finish)
        repeated = sum(
            left.get("action") == right.get("action")
            for left, right in zip(actions, actions[1:])
        )
        evaluator_available = bool(
            isinstance(ground_truth, dict) and ground_truth.get("available")
        )
        successful_environment_steps = [
            item.get("step")
            for item in history
            if isinstance(item.get("environment_success_check"), dict)
            and item["environment_success_check"].get("available")
            and item["environment_success_check"].get("success")
        ]
        environment_success_ever = bool(successful_environment_steps) or bool(
            evaluator_available and ground_truth.get("success")
        )
        requires_manual_review = bool(
            isinstance(ground_truth, dict)
            and ground_truth.get("requires_manual_review")
            and status == "completed"
        )
        perception_parser_applicable = (
            self.perceptual_interface != "direct_visual"
        )
        controller_assisted_completion = any(
            bool(item.get("completion_promoted_by_perception"))
            for item in decisions
        )
        environment_trace = []
        for item in history:
            evaluation = item.get("environment_success_check")
            visibility = self._criterion_result(
                evaluation, "object_visibility"
            )
            proximity = self._criterion_result(
                evaluation, "object_proximity"
            )
            if visibility is None and proximity is None:
                continue
            environment_trace.append(
                {
                    "step": item.get("step"),
                    "target_object_id": (
                        visibility.get("target_object_id")
                        if isinstance(visibility, dict)
                        else proximity.get("target_object_id")
                        if isinstance(proximity, dict)
                        else None
                    ),
                    "object_id_visible": (
                        bool(visibility.get("success"))
                        if isinstance(visibility, dict)
                        else None
                    ),
                    "visible_pixels": (
                        visibility.get("visible_pixels")
                        if isinstance(visibility, dict)
                        else None
                    ),
                    "minimum_visible_pixels": (
                        visibility.get("minimum_visible_pixels")
                        if isinstance(visibility, dict)
                        else None
                    ),
                    "distance": (
                        proximity.get("distance")
                        if isinstance(proximity, dict)
                        else None
                    ),
                    "maximum_distance": (
                        proximity.get("max_distance")
                        if isinstance(proximity, dict)
                        else None
                    ),
                }
            )
        object_id_sighting_steps = [
            item["step"]
            for item in environment_trace
            if item.get("object_id_visible") and item.get("step") is not None
        ]
        first_object_id_visible_step = (
            min(object_id_sighting_steps)
            if object_id_sighting_steps
            else None
        )
        confirmed_by_step = {
            item.get("step"): bool(
                (item.get("observation") or {}).get("target_visible")
            )
            for item in history
            if isinstance(item, dict) and item.get("step") is not None
        }
        visible_steps = sorted(set(object_id_sighting_steps))
        object_id_visible_frames_confirmed = sum(
            confirmed_by_step.get(step, False) for step in visible_steps
        )
        records_before_first_sighting = [
            item
            for item in history
            if first_object_id_visible_step is not None
            and item.get("executed")
            and item.get("step", 0) < first_object_id_visible_step
        ]
        final_visibility = self._criterion_result(
            ground_truth, "object_visibility"
        )
        agent_declared_success = (
            status == "completed" and not controller_assisted_completion
        )
        navigation = self._navigation_progress(history)
        rejected_claims = sum(
            item.get("controller_intervention") == "completion_rejected"
            for item in decisions
        )
        completion_claims = rejected_claims + int(agent_declared_success)
        task_success = status == "completed"
        if evaluator_available:
            task_success = task_success and bool(ground_truth.get("success"))
        elif requires_manual_review:
            task_success = None if status == "completed" else False

        final_image_relative = None
        if final_image_path:
            try:
                final_image_relative = str(
                    Path(final_image_path).resolve().relative_to(
                        self.history_directory.resolve()
                    )
                )
            except (OSError, ValueError):
                final_image_relative = Path(final_image_path).name
        return {
            "goal": goal,
            "task_contract": task_contract.as_dict(),
            "completion_consistency": dict(completion_consistency),
            "status": status,
            "reason": reason,
            "model": self.model_name,
            "model_configuration": (
                self.backend.configuration()
                if hasattr(self.backend, "configuration")
                else {"model_name": self.model_name}
            ),
            "baseline": self.baseline_name,
            "perceptual_interface": self.perceptual_interface,
            "memory_mode": self.memory_mode,
            "turn_angle_degrees": self._active_turn_angle_degrees,
            "started_at": started_at.isoformat(),
            "finished_at": datetime.now().isoformat(),
            "duration_seconds": round(duration, 5),
            "actions_taken": len(actions),
            "blocked_or_failed_actions": sum(
                item.get("outcome") in {"blocked", "failed", "failed_or_blocked"}
                for item in actions
            ),
            "invalid_model_outputs": sum(
                item.get("model_output_valid") is False for item in decisions
            ),
            "truncated_json_recoveries": sum(
                item.get("model_parse_result") == "truncated_json_recovered"
                for item in decisions
            ),
            "guarded_action_replacements": sum(
                item.get("controller_intervention")
                == "guarded_action_replacement"
                for item in decisions
            ),
            "fallback_actions": sum(
                bool(item.get("fallback_used")) for item in decisions
            ),
            "rejected_completions": sum(
                item.get("controller_intervention") == "completion_rejected"
                for item in decisions
            ),
            **navigation,
            "completion_claims": completion_claims,
            "completion_claims_rejected": rejected_claims,
            "completion_claims_rejected_by_environment": sum(
                item.get("completion_rejection_source") == "environment"
                for item in decisions
            ),
            "rejected_claim_rate": (
                round(rejected_claims / completion_claims, 4)
                if completion_claims
                else None
            ),
            "completion_policy": self._completion_policy,
            "rejected_impossible_claims": sum(
                item.get("controller_intervention") == "impossible_rejected"
                for item in decisions
            ),
            "completion_inconsistencies": sum(
                bool(item.get("completion_inconsistency"))
                for item in decisions
            ),
            "completion_retry_calls": sum(
                bool(item.get("completion_retry_attempted"))
                for item in decisions
            ),
            "completion_retries_resolved": sum(
                bool(item.get("completion_retry_resolved"))
                for item in decisions
            ),
            "completion_promotions": sum(
                bool(item.get("completion_promoted_by_perception"))
                for item in decisions
            ),
            "completion_resolution": (
                "not_completed"
                if status != "completed"
                else "controller_promoted"
                if controller_assisted_completion
                else "recheck_resolved"
                if any(
                    bool(item.get("completion_retry_resolved"))
                    for item in decisions
                )
                else "first_pass"
            ),
            "target_approach_steps": sum(
                bool(item.get("target_approach_applied")) for item in decisions
            ),
            "target_approach_replacements": sum(
                bool(item.get("target_approach_replaced_action"))
                for item in decisions
            ),
            "target_approach_persisted_steps": sum(
                bool(item.get("target_approach_persisted"))
                for item in decisions
            ),
            "target_overshoot_corrections": sum(
                bool(item.get("target_overshoot_corrected"))
                for item in decisions
            ),
            "search_turn_oscillation_replacements": sum(
                bool(item.get("search_turn_oscillation_replaced"))
                for item in decisions
            ),
            "target_loss_recoveries": sum(
                bool(item.get("target_loss_recovery_applied"))
                for item in decisions
            ),
            "recent_sighting_reversals_allowed": sum(
                bool(item.get("recent_sighting_reversal_allowed"))
                for item in decisions
            ),
            "target_side_flip_rejections": sum(
                bool(item.get("target_side_flip_rejected"))
                for item in decisions
            ),
            "memory_destination_abandonments": sum(
                bool(item.get("memory_destination_abandoned"))
                for item in decisions
            ),
            "memory_destination_suspensions": sum(
                bool(item.get("memory_destination_suspended"))
                for item in decisions
            ),
            "memory_destination_blocked_approaches": sum(
                bool(item.get("memory_destination_blocked"))
                for item in decisions
            ),
            "memory_guided_steps_to_landmark": sum(
                item.get("memory_guided_destination") == "landmark"
                and item.get("controller_intervention") == "memory_guided_search"
                for item in decisions
            ),
            "memory_guided_steps_to_sighting": sum(
                item.get("memory_guided_destination") == "sighting"
                and item.get("controller_intervention") == "memory_guided_search"
                for item in decisions
            ),
            "evidence_text_claims": sum(
                bool(item.get("evidence_text_claim")) for item in decisions
            ),
            "evidence_text_claims_accepted": sum(
                bool(item.get("evidence_text_claim"))
                and item.get("status") == "completed"
                for item in decisions
            ),
            "status_normalizations": sum(
                bool(
                    item.get("status_normalized")
                    or item.get("completion_retry_status_normalized")
                )
                for item in decisions
            ),
            "memory_guided_search_holds": sum(
                bool(item.get("memory_guided_search_held"))
                for item in decisions
            ),
            "unlocated_sighting_approaches": sum(
                bool(item.get("unlocated_sighting_approach"))
                for item in decisions
            ),
            "action_purpose_counts": {
                purpose: sum(
                    item.get("action_purpose") == purpose for item in actions
                )
                for purpose in sorted(self.ACTION_PURPOSES)
                if any(
                    item.get("action_purpose") == purpose for item in actions
                )
            },
            "target_reacquisition_calls": sum(
                bool(item.get("target_reacquisition_attempted"))
                for item in decisions
            ),
            "target_reacquisition_confirmations": sum(
                bool(item.get("target_reacquisition_confirmed"))
                for item in decisions
            ),
            "evidence_conflict_checks": sum(
                bool(item.get("evidence_conflict_check_attempted"))
                for item in decisions
            ),
            "evidence_conflict_confirmations": sum(
                bool(item.get("evidence_conflict_check_confirmed"))
                for item in decisions
            ),
            "perception_parser_applicable": perception_parser_applicable,
            "perception_parse_failures": (
                sum(
                    isinstance(item.get("observation"), dict)
                    and item["observation"].get("parser_result")
                    not in {
                        "valid",
                        "perception_normalized",
                        "perception_retry_valid",
                        "target_reacquisition_valid",
                    }
                    for item in history
                )
                if perception_parser_applicable
                else None
            ),
            "immediately_repeated_actions": repeated,
            "inference": {
                **inference_totals,
                "preprocess_seconds": round(
                    inference_totals["preprocess_seconds"], 5
                ),
                "time_to_first_token_seconds": round(
                    inference_totals["time_to_first_token_seconds"], 5
                ),
                "decode_seconds": round(
                    inference_totals["decode_seconds"], 5
                ),
                "inference_seconds": round(
                    inference_totals["inference_seconds"], 5
                ),
            },
            "incorrect_impossible_requires_review": status == "impossible",
            "ground_truth_evaluation": ground_truth,
            "task_success": task_success,
            "agent_declared_success": agent_declared_success,
            "controller_assisted_completion": controller_assisted_completion,
            "success_evidence_source": (
                "environment_evaluator"
                if evaluator_available
                else "controller_promoted_perception"
                if controller_assisted_completion
                else "agent_stop_with_visual_evidence"
            ),
            "requires_manual_review": requires_manual_review,
            "environment_success_ever": environment_success_ever,
            "first_environment_success_step": (
                min(successful_environment_steps)
                if successful_environment_steps
                else None
            ),
            "goal_reached_without_stop": bool(
                environment_success_ever and status != "completed"
            ),
            "object_id_visible_at_stop": (
                bool(final_visibility.get("success"))
                if isinstance(final_visibility, dict)
                else None
            ),
            "object_id_visible_pixels_at_stop": (
                final_visibility.get("visible_pixels")
                if isinstance(final_visibility, dict)
                else None
            ),
            "first_object_id_visible_step": first_object_id_visible_step,
            "object_id_visible_frames": len(visible_steps),
            "object_id_visible_frames_confirmed": (
                object_id_visible_frames_confirmed
            ),
            "object_id_visible_frames_missed": (
                len(visible_steps) - object_id_visible_frames_confirmed
            ),
            "object_id_sighting_steps": object_id_sighting_steps,
            "actions_before_first_object_id_sighting": len(
                records_before_first_sighting
            ),
            "path_length_before_first_object_id_sighting": (
                self._translation_path_length(records_before_first_sighting)
                if first_object_id_visible_step is not None
                else None
            ),
            "environment_trace": environment_trace,
            "final_image_path": final_image_path,
            "final_image_path_relative": final_image_relative,
            "scene_interpretation_path": str(
                (self.history_directory / "scene_interpretations.txt").resolve()
            ),
            "scene_interpretation_path_relative": "scene_interpretations.txt",
            "episode_metadata": episode_metadata,
            "history": history,
        }

    def save_history(self, report):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        task_id = re.sub(r"[^a-zA-Z0-9_-]+", "_", report["goal"])[:40]
        path = self.history_directory / f"{task_id}_{timestamp}.json"
        with path.open("w", encoding="utf-8") as file:
            json.dump(report, file, indent=2, ensure_ascii=False)
        return path.resolve()

    @staticmethod
    def _notify_progress(progress_callback, history):
        if progress_callback is None:
            return
        try:
            progress_callback(history)
        except Exception as error:
            print(f"Progress checkpoint warning: {error}")

    def _save_scene_interpretation(
        self, step, source, interpretation, model_call
    ):
        if interpretation is None and model_call is None:
            return

        path = self.history_directory / "scene_interpretations.txt"
        try:
            with path.open("a", encoding="utf-8") as file:
                file.write(f"\n===== Step {step} | {source} =====\n")
                if model_call is not None:
                    file.write("Raw model response:\n")
                    file.write(
                        str(model_call.get("raw_response", "")).strip()
                    )
                    file.write("\n")
                if interpretation is not None:
                    file.write("Parsed interpretation:\n")
                    json.dump(
                        interpretation,
                        file,
                        indent=2,
                        ensure_ascii=False,
                    )
                    file.write("\n")
                if model_call is not None:
                    file.write(
                        f"Parser stage: {model_call.get('stage', 'unknown')}\n"
                    )
                file.flush()
        except OSError as error:
            print(f"Scene interpretation logging warning: {error}")

    @staticmethod
    def print_history(report):
        print("\n========== TASK RESULT ==========")
        print(f"Goal: {report['goal']}")
        print(f"Status: {report['status']}")
        print(f"Reason: {report['reason']}")
        print(f"Actions: {report['actions_taken']}")
        print(f"Duration: {report['duration_seconds']:.3f} seconds")
        print(f"History: {report.get('history_path')}")
        print("=================================\n")


class CoELABaselinePlanner(Planner):
    HIGH_LEVEL_PLANS = {
        "explore_left",
        "explore_right",
        "approach_target",
        "retreat",
        "pick_up_target",
        "put_down_target",
    }

    def __init__(self, *args, **kwargs):
        kwargs["perceptual_interface"] = "coela"
        kwargs["memory_mode"] = "combined"
        kwargs["baseline_name"] = "coela_style_single_agent"
        super().__init__(*args, **kwargs)

    def _decide(
        self,
        image,
        perception,
        goal,
        memory_context,
        blocked_action,
        action_count,
        max_steps,
        current_state,
        recent_history,
        task_contract=None,
        memory_context_stats=None,
    ):
        task_contract = task_contract or TaskContract.build(goal)
        prompt = f"""
You are the high-level planning module in a single-agent CoELA-style baseline.
Goal: {goal}
Fixed task contract: {self._json_text(task_contract.as_dict())}
Current generated scene text: {perception.get("scene_text", "")}
Current semantic map: {self._json_text(perception)}
Episodic and semantic memory: {self._json_text(memory_context)}

First determine status. Use completed only when the current state proves the
goal is achieved. Use impossible only when the semantic or episodic memory
shows that the local area has been explored or mapped and the goal cannot be
completed. The action limit never authorizes impossible by itself. Do not use
impossible merely because the target is not in the current view or the scene
is ambiguous.
Do not reinterpret the target, completion mode, or allowed actions in the
fixed task contract. A candidate object is not confirmation;
use the current semantic evidence and the inferred condition to decide
whether the goal is complete. Distance is only required when the inferred
completion condition is target_near.
Set area_explored true only when the episodic or semantic memory supports that
the local area has been checked from multiple headings or positions. If the
semantic observations cover multiple views, objects, obstacles, or spatial
relations, set area_mapped true and provide mapping_evidence.
Otherwise use continue and choose one high-level plan from:
{", ".join(sorted(self.HIGH_LEVEL_PLANS))}.
Return JSON with status before plan:
{{"status":"completed|continue|impossible","target_visible":false,"target_location":"unknown","target_distance":"unknown","target_held":false,"plan":"plan or null","visual_evidence":"current evidence","reason":"fresh reason","impossible_basis":"none|blocked|infeasible","impossible_evidence":"concrete evidence or empty","area_explored":false,"exploration_evidence":"","area_mapped":false,"mapping_evidence":""}}
""".strip()
        result = self.backend.generate(
            prompt=prompt,
            image=None,
            max_new_tokens=self.decision_max_new_tokens,
        )
        data, parser_result = self._parse_json(result.text)
        model_parse_result = parser_result
        data = data if isinstance(data, dict) else {}
        status = data.get("status")
        if status not in {"completed", "continue", "impossible"}:
            status = "continue"
            parser_result = "invalid_status_defaulted_to_continue"
        plan = data.get("plan")
        if status == "continue" and plan not in self.HIGH_LEVEL_PLANS:
            plan = "explore_right"
            parser_result = "invalid_plan_fallback"
        action = (
            self._translate_high_level_plan(
                plan, perception, blocked_action, recent_history
            )
            if status == "continue"
            else None
        )
        reason = str(data.get("reason", "")).strip()
        if not reason:
            reason = (
                f"The CoELA-style planner selected the {plan} plan."
                if plan
                else f"The CoELA-style planner selected status {status}."
            )
        impossible_evidence = data.get("impossible_evidence") or data.get(
            "feasibility_evidence"
        )
        if not isinstance(impossible_evidence, str):
            impossible_evidence = ""
        impossible_evidence = impossible_evidence.strip()
        if status == "impossible" and not impossible_evidence:
            impossible_evidence = reason
        impossible_basis = self._normalize_impossible_basis(
            data.get("impossible_basis")
        )
        if status == "impossible" and impossible_basis == "none":
            impossible_basis = self._infer_impossible_basis(impossible_evidence)
        area_explored = self._coerce_bool(
            data.get("area_explored")
        ) is True
        exploration_evidence = data.get("exploration_evidence") or ""
        if not isinstance(exploration_evidence, str):
            exploration_evidence = ""
        area_mapped = self._coerce_bool(
            data.get("area_mapped")
        ) is True
        mapping_evidence = data.get("mapping_evidence") or ""
        if not isinstance(mapping_evidence, str):
            mapping_evidence = ""
        perception_target_visible = self._coerce_bool(
            perception.get("target_visible")
            if isinstance(perception, dict)
            else None
        )
        decision_target_visible = self._coerce_bool(data.get("target_visible"))
        target_visible = (
            perception_target_visible is True
            or decision_target_visible is True
        )
        target_location = self._normalize_location(
            data.get("target_location")
            or (
                perception.get("target_location")
                if isinstance(perception, dict)
                else None
            )
        )
        target_distance = self._normalize_distance(
            data.get("target_distance")
            or (
                perception.get("target_distance")
                if isinstance(perception, dict)
                else None
            )
        )
        task_target = task_contract.task_target
        completion_mode = task_contract.completion_mode
        completion_condition = task_contract.completion_condition
        requires_target_confirmation = task_contract.requires_target_confirmation
        return {
            "status": status,
            "task_target": task_target,
            "completion_mode": completion_mode,
            "completion_condition": completion_condition.strip(),
            "requires_target_confirmation": requires_target_confirmation,
            "target_visible": target_visible,
            "target_location": target_location,
            "target_distance": target_distance,
            "target_held": self._coerce_bool(data.get("target_held")),
            "visual_evidence": str(
                data.get("visual_evidence", "No evidence supplied.")
            ),
            "action": action,
            "high_level_plan": plan,
            "reason": reason,
            "impossible_basis": impossible_basis,
            "impossible_evidence": impossible_evidence,
            "area_explored": area_explored,
            "exploration_evidence": exploration_evidence.strip(),
            "area_mapped": area_mapped,
            "mapping_evidence": mapping_evidence.strip(),
            "parser_result": parser_result,
            "model_parse_result": model_parse_result,
            "model_output_valid": model_parse_result == "valid",
            "fallback_used": parser_result == "invalid_plan_fallback",
            "raw_response": result.text,
        }, self._call_record("coela_status_then_plan", result)

    def _translate_high_level_plan(
        self, plan, perception, blocked_action, recent_history=None
    ):
        target_location = str(perception.get("target_location", "unknown")).lower()
        mapping = {
            "explore_left": "turn_left",
            "explore_right": "turn_right",
            "retreat": "move_backward",
            "pick_up_target": "pick_up",
            "put_down_target": "put_down",
        }
        if plan == "approach_target":
            if "left" in target_location:
                action = "turn_left"
            elif "right" in target_location:
                action = "turn_right"
            else:
                action = "move_forward"
        else:
            action = mapping.get(plan, "turn_right")
        if action == blocked_action:
            return self._fallback_action(
                self._current_action_options() - {blocked_action},
                blocked_action,
                self._recent_actions(recent_history or []),
            )
        guarded, _ = self._guarded_actions(
            recent_history or [],
            blocked_action,
            allowed_actions=self._current_action_options(),
            turn_angle_degrees=self._active_turn_angle_degrees,
        )
        if action not in guarded:
            return self._fallback_action(
                guarded,
                blocked_action,
                self._recent_actions(recent_history or []),
            )
        return action
