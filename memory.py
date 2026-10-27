from collections import deque
from copy import deepcopy
from datetime import datetime
import json
import math
from pathlib import Path


PRIMARY_MEMORY_MODES = {"none", "episodic", "semantic", "combined"}
OPTIONAL_MEMORY_MODES = {"learned", "combined_learned"}
VALID_MEMORY_MODES = PRIMARY_MEMORY_MODES | OPTIONAL_MEMORY_MODES
MEMORY_API_VERSION = 2


def normalize_memory_mode(mode):
    mode = (mode or "none").strip().lower()
    aliases = {
        "no_memory": "none",
        "short_term": "episodic",
        "long_term": "semantic",
    }
    mode = aliases.get(mode, mode)
    if mode not in VALID_MEMORY_MODES:
        raise ValueError(
            f"Unknown memory mode {mode!r}. Expected one of "
            f"{sorted(VALID_MEMORY_MODES)}."
        )
    return mode


class MemoryManager:
    def __init__(
        self,
        episodic_window=8,
        learned_store=None,
        context_episodic_limit=4,
        context_object_limit=12,
        context_pose_limit=8,
        context_fact_limit=12,
        context_text_char_limit=1000,
        context_prompt_char_limit=3000,
    ):
        if episodic_window < 1:
            raise ValueError("episodic_window must be at least 1.")
        self.episodic_window = int(episodic_window)
        self.context_episodic_limit = self._positive_limit(
            context_episodic_limit, "context_episodic_limit"
        )
        self.context_object_limit = self._positive_limit(
            context_object_limit, "context_object_limit"
        )
        self.context_pose_limit = self._positive_limit(
            context_pose_limit, "context_pose_limit"
        )
        self.context_fact_limit = self._positive_limit(
            context_fact_limit, "context_fact_limit"
        )
        self.context_text_char_limit = self._positive_limit(
            context_text_char_limit, "context_text_char_limit"
        )
        self.context_prompt_char_limit = self._positive_limit(
            context_prompt_char_limit, "context_prompt_char_limit"
        )
        if self.context_prompt_char_limit < 256:
            raise ValueError(
                "context_prompt_char_limit must be at least 256 characters."
            )
        self.learned_store = learned_store
        self._last_context_stats = {
            "characters": 0,
            "budget_characters": self.context_prompt_char_limit,
            "trimmed": False,
        }
        self.reset_episode()

    @staticmethod
    def _positive_limit(value, name):
        try:
            limit = int(value)
        except (TypeError, ValueError) as error:
            raise ValueError(f"{name} must be a positive integer.") from error
        if limit < 1:
            raise ValueError(f"{name} must be at least 1.")
        return limit

    def reset_episode(self, goal=None):
        self.goal = goal
        self.episodic = deque(maxlen=self.episodic_window)
        self.semantic = {
            "objects": {},
            "visited_poses": [],
            "blocked_directions": [],
            "spatial_relations": [],
            "obstacles": [],
            "agent_state": {},
            "task_progress": "No progress recorded.",
            "last_scene_summary": "",
            "last_scene_text": "",
            "task_target": "",
            "completion_mode": "unknown",
            "completion_condition": "",
            "target_attributes": [],
            "target_visible": False,
            "target_location": "unknown",
            "target_distance": "unknown",
            "target_sightings": [],
            "task_landmarks": [],
        }

    def record_observation(self, perception=None, agent_state=None, step=None):
        if agent_state:
            self.semantic["agent_state"] = deepcopy(agent_state)
            pose = {
                "step": step,
                "position": deepcopy(agent_state.get("position")),
                "yaw": agent_state.get("yaw"),
            }
            if pose["position"] is not None:
                previous = self.semantic["visited_poses"][-1:] or [None]
                if pose != previous[0]:
                    self.semantic["visited_poses"].append(pose)
                    self.semantic["visited_poses"] = self.semantic[
                        "visited_poses"
                    ][-50:]

        if not isinstance(perception, dict):
            return

        summary = perception.get("scene_summary")
        if summary:
            self.semantic["last_scene_summary"] = str(summary)

        scene_text = perception.get("scene_text") or summary
        if scene_text:
            self.semantic["last_scene_text"] = str(scene_text)

        for key in (
            "task_target",
            "target_attributes",
            "completion_mode",
            "completion_condition",
            "target_location",
            "target_distance",
        ):
            value = perception.get(key)
            if value not in (None, ""):
                self.semantic[key] = deepcopy(value)

        target_visible = perception.get("target_visible")
        if isinstance(target_visible, str):
            target_visible = target_visible.strip().lower() in {
                "true",
                "yes",
                "visible",
                "confirmed",
            }
        if target_visible is not None:
            self.semantic["target_visible"] = bool(target_visible)

        if bool(target_visible):
            self._record_target_sighting(
                perception=perception,
                agent_state=agent_state,
                step=step,
            )

        for item in perception.get("objects", []) or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "unknown object")).strip().lower()
            key = str(item.get("id") or name)
            self.semantic["objects"].pop(key, None)
            self.semantic["objects"][key] = {
                "name": name,
                "location": item.get("location", "unknown"),
                "attributes": item.get("attributes", []),
                "confidence": item.get("confidence"),
                "last_seen_step": step,
            }

        self._append_unique(
            "spatial_relations", perception.get("spatial_relations", [])
        )
        self._append_unique("obstacles", perception.get("obstacles", []))

        progress = perception.get("task_progress")
        if progress:
            self.semantic["task_progress"] = str(progress)

    def record_step(self, step_record, perception=None, agent_state=None):
        compact_event = {
            "step": step_record.get("step"),
            "action": step_record.get("action"),
            "outcome": step_record.get("outcome"),
            "reason": step_record.get("reason"),
            "position_changed": step_record.get("position_changed"),
            "held_object_changed": step_record.get("held_object_changed"),
            "scene_text": (
                perception.get("scene_text")
                if isinstance(perception, dict)
                else None
            ),
            "task_target": (
                perception.get("task_target")
                if isinstance(perception, dict)
                else None
            ),
            "target_attributes": (
                perception.get("target_attributes")
                if isinstance(perception, dict)
                else None
            ),
            "completion_mode": (
                perception.get("completion_mode")
                if isinstance(perception, dict)
                else None
            ),
            "completion_condition": (
                perception.get("completion_condition")
                if isinstance(perception, dict)
                else None
            ),
            "target_visible": (
                perception.get("target_visible")
                if isinstance(perception, dict)
                else None
            ),
            "target_location": (
                perception.get("target_location")
                if isinstance(perception, dict)
                else None
            ),
            "target_distance": (
                perception.get("target_distance")
                if isinstance(perception, dict)
                else None
            ),
        }
        self.episodic.append(compact_event)
        self.record_observation(
            perception=perception,
            agent_state=agent_state,
            step=step_record.get("step"),
        )

        if step_record.get("outcome") in {"blocked", "failed"}:
            action = step_record.get("action")
            if action in {"move_forward", "move_backward"}:
                fact = {
                    "step": step_record.get("step"),
                    "action": action,
                    "yaw": (agent_state or {}).get("yaw"),
                    "position": deepcopy((agent_state or {}).get("position")),
                }
                if fact not in self.semantic["blocked_directions"]:
                    self.semantic["blocked_directions"].append(fact)

        self.semantic["task_progress"] = str(
            step_record.get("reason") or self.semantic["task_progress"]
        )

    def record_task_boundary(
        self,
        task_id,
        goal,
        task_contract,
        status,
        agent_state=None,
        task_success=None,
    ):
        contract = task_contract if isinstance(task_contract, dict) else {}
        target = str(contract.get("task_target") or "").strip()
        position = None
        yaw = None
        if isinstance(agent_state, dict):
            raw_position = agent_state.get("position")
            if isinstance(raw_position, dict):
                try:
                    position = {
                        "x": float(raw_position["x"]),
                        "y": float(raw_position.get("y", 0.0)),
                        "z": float(raw_position["z"]),
                    }
                except (KeyError, TypeError, ValueError):
                    position = None
            yaw = agent_state.get("yaw")

        boundary = {
            "event_type": "task_boundary",
            "task_id": str(task_id),
            "goal": str(goal),
            "task_target": target,
            "completion_mode": contract.get("completion_mode"),
            "target_attributes": deepcopy(
                contract.get("target_attributes", [])
            ),
            "status": str(status),
            "task_success": task_success,
            "position": position,
            "yaw": yaw,
        }
        self.episodic.append(boundary)
        if (
            status == "completed"
            and task_success is not False
            and target
            and position is not None
        ):
            landmark = {
                "task_id": str(task_id),
                "task_target": target,
                "target_attributes": deepcopy(
                    contract.get("target_attributes", [])
                ),
                "completion_mode": contract.get("completion_mode"),
                "position": position,
                "yaw": yaw,
                "source": (
                    "independently_validated_completion_pose"
                    if task_success is True
                    else "agent_declared_completion_pose"
                ),
            }
            prior = self.semantic["task_landmarks"]
            prior[:] = [
                item
                for item in prior
                if not (
                    item.get("task_target") == target
                    and item.get("completion_mode")
                    == landmark["completion_mode"]
                    and item.get("target_attributes", [])
                    == landmark["target_attributes"]
                )
            ]
            prior.append(landmark)
            self.semantic["task_landmarks"] = prior[-50:]
            self.semantic["task_progress"] = (
                f"Completed {target} at the recorded agent pose."
            )
        else:
            self.semantic["task_progress"] = (
                f"Stage {task_id} ended with status {status}"
                + (
                    " without independent success validation."
                    if task_success is False
                    else "."
                )
            )

    def has_task_landmark(self, task_id=None, task_target=None):
        expected_task_id = (
            str(task_id).strip() if task_id not in (None, "") else None
        )
        expected_target = (
            str(task_target).strip().lower()
            if task_target not in (None, "")
            else None
        )
        for landmark in self.semantic.get("task_landmarks", []):
            if not isinstance(landmark, dict):
                continue
            if (
                expected_task_id
                and str(landmark.get("task_id")) != expected_task_id
            ):
                continue
            if (
                expected_target
                and str(landmark.get("task_target", "")).lower()
                != expected_target
            ):
                continue
            if isinstance(landmark.get("position"), dict):
                return True
        return False

    def get_context(
        self,
        mode,
        goal=None,
        learned_top_k=3,
        task_target=None,
        target_attributes=None,
    ):
        mode = normalize_memory_mode(mode)
        if mode == "none":
            self._last_context_stats = {
                "characters": 0,
                "budget_characters": self.context_prompt_char_limit,
                "trimmed": False,
            }
            return None

        context = {"memory_mode": mode}
        if mode in {"episodic", "combined", "combined_learned"}:
            context["episodic_memory"] = [
                self._episodic_event_for_context(event)
                for event in list(self.episodic)[-self.context_episodic_limit:]
            ]
        if mode in {"semantic", "combined", "combined_learned"}:
            context["semantic_memory"] = self.semantic_context_snapshot(
                task_target=task_target,
                target_attributes=target_attributes,
            )
        if mode in {"learned", "combined_learned"}:
            if self.learned_store is None:
                context["learned_memory"] = []
            else:
                context["learned_memory"] = self.learned_store.retrieve(
                    goal or self.goal or "", top_k=learned_top_k
                )
        return self._fit_context_to_prompt_budget(context)

    def context_stats(self):
        return dict(self._last_context_stats)

    @staticmethod
    def _context_json(context):
        return json.dumps(
            context,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )

    def _fit_context_to_prompt_budget(self, context):
        compact = deepcopy(context)
        original_characters = len(self._context_json(compact))
        budget = self.context_prompt_char_limit

        event_text_limit = min(self.context_text_char_limit, max(96, budget // 8))
        for event in compact.get("episodic_memory", []) or []:
            if not isinstance(event, dict):
                continue
            for key in ("reason", "scene_text", "completion_condition"):
                if key in event:
                    event[key] = self._truncate_text(event.get(key), event_text_limit)

        semantic = compact.get("semantic_memory")
        if isinstance(semantic, dict):
            semantic_text_limit = min(
                self.context_text_char_limit,
                max(96, budget // 8),
            )
            for key in ("task_progress", "last_scene_summary", "last_scene_text"):
                if key in semantic:
                    semantic[key] = self._truncate_text(
                        semantic.get(key), semantic_text_limit
                    )

        list_paths = (
            ("learned_memory",),
            ("semantic_memory", "objects"),
            ("semantic_memory", "spatial_relations"),
            ("semantic_memory", "obstacles"),
            ("episodic_memory",),
            ("semantic_memory", "visited_poses"),
            ("semantic_memory", "blocked_directions"),
        )
        while len(self._context_json(compact)) > budget:
            trimmed = False
            for path in list_paths:
                values = self._context_path(compact, path)
                if isinstance(values, list) and len(values) > 1:
                    del values[0]
                    trimmed = True
                    break
            if trimmed:
                continue

            if self._shrink_context_text(compact):
                continue
            break

        if len(self._context_json(compact)) > budget:
            compact = self._minimal_context(compact)
            while (
                len(self._context_json(compact)) > budget
                and self._shrink_context_text(compact)
            ):
                pass

        serialized = self._context_json(compact)
        if len(serialized) > budget:
            compact.pop("learned_memory", None)
            serialized = self._context_json(compact)
        if len(serialized) > budget:
            compact = {"memory_mode": compact.get("memory_mode", "unknown")}
            serialized = self._context_json(compact)
        self._last_context_stats = {
            "characters": len(serialized),
            "budget_characters": budget,
            "trimmed": len(serialized) < original_characters,
            "original_characters": original_characters,
        }
        return compact

    @staticmethod
    def _context_path(value, path):
        current = value
        for key in path:
            if not isinstance(current, dict) or key not in current:
                return None
            current = current[key]
        return current

    @staticmethod
    def _truncate_text(value, limit):
        text = str(value or "")
        if len(text) <= limit:
            return text
        return text[: max(1, limit - 1)].rstrip() + "…"

    def _shrink_context_text(self, context):
        candidates = []
        semantic = context.get("semantic_memory")
        if isinstance(semantic, dict):
            candidates.extend(
                (semantic, key)
                for key in ("last_scene_text", "last_scene_summary", "task_progress")
            )
        for event in context.get("episodic_memory", []) or []:
            if isinstance(event, dict):
                candidates.extend((event, key) for key in ("scene_text", "reason"))
        for mapping, key in candidates:
            value = mapping.get(key)
            if not isinstance(value, str) or len(value) <= 48:
                continue
            mapping[key] = self._truncate_text(value, max(48, len(value) // 2))
            return True
        return False

    @staticmethod
    def _minimal_context(context):
        minimal = {"memory_mode": context.get("memory_mode", "unknown")}
        episodic = context.get("episodic_memory")
        if isinstance(episodic, list) and episodic:
            minimal["episodic_memory"] = [episodic[-1]]
        semantic = context.get("semantic_memory")
        if isinstance(semantic, dict):
            minimal_semantic = {
                key: deepcopy(semantic.get(key))
                for key in (
                    "task_target",
                    "target_attributes",
                    "completion_mode",
                    "target_visible",
                    "target_location",
                    "target_distance",
                    "target_sighting",
                    "agent_state",
                )
                if key in semantic
            }
            landmarks = semantic.get("task_landmarks")
            if isinstance(landmarks, list) and landmarks:
                minimal_semantic["task_landmarks"] = [deepcopy(landmarks[-1])]
            blocked = semantic.get("blocked_directions")
            if isinstance(blocked, list) and blocked:
                minimal_semantic["blocked_directions"] = [deepcopy(blocked[-1])]
            minimal["semantic_memory"] = minimal_semantic
        return minimal

    def semantic_snapshot(self):
        snapshot = deepcopy(self.semantic)
        snapshot["objects"] = list(snapshot["objects"].values())
        return snapshot

    def semantic_context_snapshot(self, task_target=None, target_attributes=None):
        full_snapshot = self.semantic_snapshot()
        snapshot = {
            "objects": full_snapshot["objects"][-self.context_object_limit:],
            "visited_poses": full_snapshot["visited_poses"][
                -self.context_pose_limit:
            ],
            "blocked_directions": full_snapshot["blocked_directions"][
                -self.context_pose_limit:
            ],
            "spatial_relations": full_snapshot["spatial_relations"][
                -self.context_fact_limit:
            ],
            "obstacles": full_snapshot["obstacles"][-self.context_fact_limit:],
            "task_landmarks": self._relevant_landmarks(
                full_snapshot["task_landmarks"],
                full_snapshot.get("agent_state"),
                task_target,
                target_attributes,
            )[-self.context_fact_limit:],
            "agent_state": self._compact_agent_state(
                full_snapshot.get("agent_state")
            ),
        }
        target_sighting = self._latest_target_sighting(
            task_target=task_target,
            target_attributes=target_attributes,
        )
        if target_sighting is not None:
            relative = self._relative_to_agent(
                target_sighting.get("agent_position"),
                target_sighting.get("agent_yaw"),
                full_snapshot.get("agent_state"),
            )
            if relative:
                target_sighting["relative_to_agent"] = relative
            snapshot["target_sighting"] = target_sighting
        return snapshot

    def _relevant_landmarks(
        self, landmarks, agent_state, task_target=None, target_attributes=None
    ):
        target = str(task_target or "").strip().lower()
        required = [
            str(a).strip().lower()
            for a in (target_attributes or [])
            if str(a).strip()
        ]
        selected = []
        for landmark in landmarks or []:
            if not isinstance(landmark, dict):
                continue
            if target and str(landmark.get("task_target", "")).lower() != target:
                continue
            seen = " ".join(
                str(v).lower() for v in landmark.get("target_attributes", [])
            )
            if any(attribute not in seen for attribute in required):
                continue
            item = deepcopy(landmark)
            relative = self._relative_to_agent(
                landmark.get("position"), landmark.get("yaw"), agent_state
            )
            if relative:
                item["relative_to_agent"] = relative
            selected.append(item)
        return selected

    @staticmethod
    def _relative_to_agent(position, yaw, agent_state):
        if not isinstance(position, dict) or not isinstance(agent_state, dict):
            return None
        own = agent_state.get("position")
        if not isinstance(own, dict) or agent_state.get("yaw") is None:
            return None
        try:
            dx = float(position["x"]) - float(own["x"])
            dz = float(position["z"]) - float(own["z"])
            own_yaw = float(agent_state["yaw"])
        except (KeyError, TypeError, ValueError):
            return None

        def wrap(angle):
            return (angle + 180.0) % 360.0 - 180.0

        distance = math.hypot(dx, dz)
        if distance >= 1.0:
            turn = wrap(math.degrees(math.atan2(dx, dz)) - own_yaw)
            result = {"distance_m": round(distance, 1)}
        else:
            if yaw is None:
                return {"distance_m": round(distance, 1), "at_pose": True}
            try:
                turn = wrap(float(yaw) - own_yaw)
            except (TypeError, ValueError):
                return None
            result = {"distance_m": round(distance, 1), "at_pose": True}
        result["turn"] = "right" if turn > 0 else "left"
        result["turn_degrees"] = int(round(abs(turn)))
        return result

    def _record_target_sighting(self, perception, agent_state, step):
        if not isinstance(perception, dict):
            return
        source = str(perception.get("target_confirmation_source") or "").lower()
        if source.startswith("rejected"):
            return
        target = str(
            perception.get("task_target")
            or self.semantic.get("task_target")
            or ""
        ).strip().lower()
        if not target:
            return

        attributes = perception.get("target_attributes")
        if not isinstance(attributes, (list, tuple, set)):
            attributes = []
        normalized_attributes = []
        for attribute in attributes:
            text = str(attribute).strip().lower()
            if text and text not in normalized_attributes:
                normalized_attributes.append(text)

        try:
            step_value = int(step) if step is not None else None
        except (TypeError, ValueError):
            step_value = None
        sightings = self.semantic["target_sightings"]
        if any(
            item.get("step") == step_value
            and item.get("task_target") == target
            and item.get("target_attributes", []) == normalized_attributes
            for item in sightings
            if isinstance(item, dict)
        ):
            return

        position = None
        yaw = None
        if isinstance(agent_state, dict):
            raw_position = agent_state.get("position")
            if isinstance(raw_position, dict):
                try:
                    position = {
                        "x": float(raw_position["x"]),
                        "y": float(raw_position.get("y", 0.0)),
                        "z": float(raw_position["z"]),
                    }
                except (KeyError, TypeError, ValueError):
                    position = None
            yaw = agent_state.get("yaw")

        sightings.append(
            {
                "step": step_value,
                "task_target": target,
                "target_attributes": normalized_attributes,
                "agent_position": position,
                "agent_yaw": yaw,
                "target_location": perception.get("target_location", "unknown"),
                "target_distance": perception.get("target_distance", "unknown"),
                "scene_summary": self._truncate_text(
                    perception.get("scene_summary", ""),
                    min(self.context_text_char_limit, 96),
                ),
                "confirmation_source": source or "visual_observation",
            }
        )
        self.semantic["target_sightings"] = sightings[-50:]

    def _latest_target_sighting(self, task_target=None, target_attributes=None):
        target = str(task_target or "").strip().lower()
        required_attributes = []
        for attribute in target_attributes or []:
            text = str(attribute).strip().lower()
            if text and text not in required_attributes:
                required_attributes.append(text)

        for sighting in reversed(self.semantic.get("target_sightings", [])):
            if not isinstance(sighting, dict):
                continue
            if target and sighting.get("task_target") != target:
                continue
            observed_attributes = " ".join(
                str(value).lower()
                for value in sighting.get("target_attributes", [])
            )
            if any(attribute not in observed_attributes for attribute in required_attributes):
                continue
            return deepcopy(sighting)
        return None

    def _episodic_event_for_context(self, event):
        if not isinstance(event, dict):
            return {"event": str(event)}
        fields = (
            "event_type",
            "task_id",
            "step",
            "action",
            "outcome",
            "status",
            "task_success",
            "position_changed",
            "held_object_changed",
            "target_visible",
            "target_location",
            "target_distance",
            "position",
            "yaw",
        )
        compact = {
            key: deepcopy(event[key])
            for key in fields
            if key in event and event[key] is not None
        }
        if event.get("event_type") == "task_boundary":
            for key in ("task_target", "target_attributes"):
                if key in event and event[key] is not None:
                    compact[key] = deepcopy(event[key])
        reason = event.get("reason")
        if reason:
            compact["reason"] = self._truncate_text(reason, 48)
        return compact

    @staticmethod
    def _compact_agent_state(agent_state):
        if not isinstance(agent_state, dict):
            return {}
        compact = {}
        position = agent_state.get("position")
        if isinstance(position, dict):
            compact["position"] = deepcopy(position)
        if agent_state.get("yaw") is not None:
            compact["yaw"] = agent_state.get("yaw")
        return compact

    def _truncate_for_context(self, value):
        text = str(value or "")
        if len(text) <= self.context_text_char_limit:
            return text
        return text[: self.context_text_char_limit - 1].rstrip() + "…"

    def _append_unique(self, key, values):
        for value in values or []:
            if value not in self.semantic[key]:
                self.semantic[key].append(value)
        self.semantic[key] = self.semantic[key][-50:]


class EmbeddingExperienceStore:
    def __init__(
        self,
        directory="learned_memory",
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        frozen=False,
    ):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.data_path = self.directory / "experiences.json"
        self.model_name = model_name
        self.frozen = frozen
        self._encoder = None
        self.records = self._load()

    def _load(self):
        if not self.data_path.exists():
            return []
        with self.data_path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _get_encoder(self):
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as error:
                raise RuntimeError(
                    "Learned embedding memory requires sentence-transformers. "
                    "Install it only for the optional learned-memory experiment."
                ) from error
            self._encoder = SentenceTransformer(self.model_name)
        return self._encoder

    def add_experience(self, goal, actions, outcome, notes="", metadata=None):
        if self.frozen:
            raise RuntimeError(
                "The learned-memory store is frozen for evaluation and cannot "
                "accept new experiences."
            )
        text = self._experience_text(goal, actions, outcome, notes)
        embedding = self._get_encoder().encode(
            text, normalize_embeddings=True
        ).tolist()
        record = {
            "timestamp": datetime.now().isoformat(),
            "goal": goal,
            "actions": actions,
            "outcome": outcome,
            "notes": notes,
            "metadata": metadata or {},
            "text": text,
            "embedding": embedding,
        }
        self.records.append(record)
        self._save()

    def retrieve(self, query, top_k=3):
        if not self.records or not query:
            return []
        query_vector = self._get_encoder().encode(
            query, normalize_embeddings=True
        ).tolist()
        ranked = []
        for record in self.records:
            score = sum(
                left * right
                for left, right in zip(query_vector, record["embedding"])
            )
            result = {key: value for key, value in record.items() if key != "embedding"}
            result["similarity"] = round(float(score), 5)
            ranked.append(result)
        ranked.sort(key=lambda item: item["similarity"], reverse=True)
        return ranked[:top_k]

    def freeze(self):
        self.frozen = True
        metadata_path = self.directory / "metadata.json"
        with metadata_path.open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "frozen": True,
                    "frozen_at": datetime.now().isoformat(),
                    "model_name": self.model_name,
                    "experience_count": len(self.records),
                },
                file,
                indent=2,
            )

    def _save(self):
        with self.data_path.open("w", encoding="utf-8") as file:
            json.dump(self.records, file, indent=2, ensure_ascii=False)

    @staticmethod
    def _experience_text(goal, actions, outcome, notes):
        return (
            f"Goal: {goal}\nActions: {json.dumps(actions, ensure_ascii=False)}\n"
            f"Outcome: {outcome}\nLessons: {notes}"
        )
