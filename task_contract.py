from dataclasses import asdict, dataclass
import re


SUPPORTED_ACTIONS = (
    "move_forward",
    "move_backward",
    "turn_left",
    "turn_right",
    "look_up",
    "look_down",
    "pick_up",
    "put_down",
)

COMPLETION_MODES = {
    "target_visible",
    "target_near",
    "target_held",
    "state_reached",
    "other",
    "unknown",
}


def _normalize_target(value):
    if isinstance(value, (list, tuple)):
        value = " ".join(str(item) for item in value)
    if not isinstance(value, str):
        return ""
    value = re.sub(r"\s+", " ", value.strip())
    return "" if value.lower() in {"", "unknown", "none", "null"} else value


def _normalize_excluded_target_terms(value):
    if value in (None, ""):
        return ()
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple, set)):
        raise ValueError(
            "task_spec.excluded_target_terms must be a string or list of strings."
        )

    terms = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(
                "task_spec.excluded_target_terms must contain only strings."
            )
        normalized = _normalize_target(item).lower()
        if normalized and normalized not in terms:
            terms.append(normalized)
    return tuple(terms)


def _normalize_target_aliases(value):
    if value in (None, ""):
        return ()
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple, set)):
        raise ValueError(
            "task_spec.target_aliases must be a string or list of strings."
        )

    aliases = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(
                "task_spec.target_aliases must contain only strings."
            )
        normalized = _normalize_target(item).lower()
        if normalized and normalized not in aliases:
            aliases.append(normalized)
    return tuple(aliases)


def _normalize_target_attributes(value):
    if value in (None, ""):
        return ()
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple, set)):
        raise ValueError(
            "task_spec.target_attributes must be a string or list of strings."
        )

    attributes = []
    for item in value:
        if not isinstance(item, str):
            raise ValueError(
                "task_spec.target_attributes must contain only strings."
            )
        normalized = _normalize_target(item).lower()
        if normalized and normalized not in attributes:
            attributes.append(normalized)
    return tuple(attributes)


def infer_target(goal):
    text = re.sub(r"\s+", " ", str(goal or "").strip())
    patterns = (
        r"\b(?:find|locate|identify|spot|search for|look for|go to|approach|"
        r"reach|navigate to|pick up|interact with)\s+(?:a|an|the|one)\s+"
        r"(.+?)(?=\s+(?:and|when|once|until|if|that|which)\b|[,.!?]|$)",
        r"\btarget\s*(?:is|:)\s*(.+?)(?=\s+(?:and|when|once|until|if)\b|[,.!?]|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            target = _normalize_target(match.group(1))
            if target:
                return target
    return ""


def normalize_completion_mode(value):
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
    if normalized in COMPLETION_MODES:
        return normalized
    if "near" in normalized or "close" in normalized:
        return "target_near"
    if "visible" in normalized or "found" in normalized:
        return "target_visible"
    if "held" in normalized or "carry" in normalized:
        return "target_held"
    return "unknown"


def infer_completion_mode(goal):
    text = str(goal or "").lower()
    if any(term in text for term in ("near", "close", "within", "approach")):
        return "target_near"
    if any(
        term in text
        for term in ("find", "locate", "identify", "spot", "visible", "see")
    ):
        return "target_visible"
    if any(term in text for term in ("pick up", "hold", "carry")):
        return "target_held"
    return "unknown"


@dataclass(frozen=True)
class TaskContract:
    task_target: str
    completion_mode: str
    completion_condition: str
    requires_target_confirmation: bool
    allowed_actions: tuple
    excluded_target_terms: tuple
    target_aliases: tuple
    target_attributes: tuple

    @classmethod
    def build(cls, goal, configured=None):
        configured = configured if isinstance(configured, dict) else {}
        target = _normalize_target(
            configured.get("task_target")
            or configured.get("target_name")
            or configured.get("target")
        ) or infer_target(goal)
        mode = normalize_completion_mode(configured.get("completion_mode"))
        if mode == "unknown":
            mode = infer_completion_mode(goal)

        condition = configured.get("completion_condition")
        if not isinstance(condition, str) or not condition.strip():
            condition = str(goal or "").strip()

        requires = configured.get("requires_target_confirmation")
        if mode in {"target_visible", "target_near", "target_held"}:
            requires = True
        elif not isinstance(requires, bool):
            requires = False

        requested_actions = configured.get("allowed_actions", SUPPORTED_ACTIONS)
        if not isinstance(requested_actions, (list, tuple, set)):
            raise ValueError("task_spec.allowed_actions must be a list of actions.")
        allowed = tuple(dict.fromkeys(str(action) for action in requested_actions))
        unsupported = sorted(set(allowed) - set(SUPPORTED_ACTIONS))
        if unsupported:
            raise ValueError(
                f"Unsupported task actions: {unsupported}. Supported actions: "
                f"{list(SUPPORTED_ACTIONS)}"
            )
        if not allowed:
            raise ValueError("A task must allow at least one executable action.")
        if requires and not target:
            raise ValueError(
                "A target-based task requires task_spec.task_target (or a goal "
                "from which a target can be inferred)."
            )

        excluded_target_terms = _normalize_excluded_target_terms(
            configured.get("excluded_target_terms")
        )
        target_aliases = _normalize_target_aliases(
            configured.get("target_aliases")
            or configured.get("target_class_aliases")
        )
        target_attributes = _normalize_target_attributes(
            configured.get("target_attributes")
            or configured.get("required_target_attributes")
            or configured.get("visual_attributes")
        )

        return cls(
            task_target=target,
            completion_mode=mode,
            completion_condition=condition.strip(),
            requires_target_confirmation=bool(requires),
            allowed_actions=allowed,
            excluded_target_terms=excluded_target_terms,
            target_aliases=target_aliases,
            target_attributes=target_attributes,
        )

    def as_dict(self):
        data = asdict(self)
        data["allowed_actions"] = list(self.allowed_actions)
        data["excluded_target_terms"] = list(self.excluded_target_terms)
        data["target_aliases"] = list(self.target_aliases)
        data["target_attributes"] = list(self.target_attributes)
        return data
