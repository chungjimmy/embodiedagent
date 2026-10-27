import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

INTERFACE_ORDER = ["direct_visual", "text_mediated", "hybrid"]
MEMORY_ORDER = ["none", "episodic", "semantic", "combined"]

OUTCOME_COLORS = {
    "verified_success": "#2a9d4b",
    "completed_unreviewed": "#f2c14e",
    "false_stop": "#c0392b",
    "action_limit": "#8d99ae",
    "other_failure": "#4a4e69",
}
OUTCOME_LABELS = {
    "verified_success": "Completed and verified",
    "completed_unreviewed": "Completed, not yet verified",
    "false_stop": "Completed, verified wrong",
    "action_limit": "Hit action limit",
    "other_failure": "Timeout / error",
}
MEMORY_COLORS = {
    "none": "#4c78a8",
    "episodic": "#f58518",
    "semantic": "#54a24b",
    "combined": "#b279a2",
}
ACTION_COLORS = {
    "turn_left": "#a6cee3",
    "turn_right": "#1f78b4",
    "move_forward": "#33a02c",
    "move_backward": "#ff7f00",
    "look_up": "#cab2d6",
    "look_down": "#6a3d9a",
}
NEGATION = re.compile(
    r"\b(no|not|without|isn't|aren't|none|unable|cannot|can't|nothing|absent)\b", re.I
)
TRUE_WORDS = {"true", "yes", "y", "1", "pass", "visible", "window"}
FALSE_WORDS = {"false", "no", "n", "0", "fail", "none", "absent"}


def find_run_dir(arg):
    if arg:
        path = Path(arg)
        if path.is_file():
            path = path.parent
        if not (path / "all_results.json").exists():
            sys.exit(f"No all_results.json in {path}")
        return path
    if Path("all_results.json").exists():
        return Path(".")
    candidates = sorted(
        p.parent for p in Path("experiment_results").glob("*/all_results.json")
    )
    if not candidates:
        sys.exit("No run found. Pass the run folder: python visualize_results.py <folder>")
    return candidates[-1]


def parse_label(raw):
    text = str(raw or "").strip().lower()
    if not text:
        return None
    if text in TRUE_WORDS:
        return True
    if text in FALSE_WORDS:
        return False
    return "invalid"


def load_labels(run_dir):
    csv_path = run_dir / "manual_review.csv"
    map_path = run_dir / "manual_review_mapping.json"
    warnings = []
    if not csv_path.exists() or not map_path.exists():
        return {}, ["manual_review.csv or manual_review_mapping.json not found; all labels pending."]
    mapping = {m["review_id"]: m["episode_id"] for m in json.loads(map_path.read_text("utf-8"))}
    labels = {}
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            review_id = (row.get("review_id") or "").strip()
            if not review_id:
                continue
            label = parse_label(row.get("manual_target_visible"))
            if review_id not in mapping:
                warnings.append(f"{review_id}: not in mapping file, ignored.")
            elif label == "invalid":
                warnings.append(
                    f"{review_id}: unreadable label {row.get('manual_target_visible')!r}; use true/false."
                )
            elif label is not None:
                labels[mapping[review_id]] = label
    return labels, warnings


def mentions_target(entry, target):
    obs = entry.get("observation") or {}
    texts = [entry.get("visual_evidence"), obs.get("scene_summary")]
    for text in texts:
        if not text:
            continue
        for sentence in re.split(r"[.;]", str(text)):
            if re.search(target, sentence, re.I) and not NEGATION.search(sentence):
                return True
    for item in obs.get("objects") or []:
        if isinstance(item, dict) and re.search(target, str(item.get("name", "")), re.I):
            return True
    return False


def position_of(entry):
    for key in ("state_after", "state_before"):
        state = entry.get(key)
        if isinstance(state, dict) and isinstance(state.get("position"), dict):
            pos = state["position"]
            return pos.get("x"), pos.get("z")
    return None


def num(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


def build_episode(report, labels):
    meta = report.get("episode_metadata") or {}
    target = (report.get("task_contract") or {}).get("task_target") or "window"
    history = report.get("history") or []
    inference = report.get("inference") or {}
    episode_id = meta.get("episode_id") or report.get("episode_id") or ""
    place = meta.get("scene_episode_id") or episode_id.rsplit("__", 1)[-1]

    steps = []
    for entry in history:
        obs = entry.get("observation") or {}
        steps.append(
            {
                "step": entry.get("step"),
                "action": entry.get("action"),
                "requested": entry.get("requested_action"),
                "outcome": entry.get("outcome"),
                "guard": bool(entry.get("action_guard")),
                "intervention": entry.get("controller_intervention"),
                "oscillation": bool(entry.get("search_turn_oscillation_replaced")),
                "target_visible": bool(obs.get("target_visible")),
                "mention": mentions_target(entry, target),
                "completed": (entry.get("finish_check") or {}).get("status") == "completed",
                "pos": position_of(entry),
            }
        )
    conflict_steps = [
        s["step"] for s in steps if s["mention"] and not s["target_visible"] and not s["completed"]
    ]

    status = report.get("status")
    manual = labels.get(episode_id)
    task_success = report.get("task_success")
    if status == "completed":
        if manual is None and isinstance(task_success, bool):
            manual = task_success
        if manual is True:
            outcome = "verified_success"
        elif manual is False:
            outcome = "false_stop"
        else:
            outcome = "completed_unreviewed"
    elif status == "action_limit":
        outcome = "action_limit"
    else:
        outcome = "other_failure"

    start = meta.get("initial_position") or {}
    path = [(start.get("x"), start.get("z"))] if start else []
    path += [s["pos"] for s in steps if s["pos"]]
    moving = [s for s in steps if s["action"]]
    return {
        "episode_id": episode_id,
        "interface": report.get("perceptual_interface"),
        "memory": report.get("memory_mode"),
        "place": place,
        "status": status,
        "outcome": outcome,
        "actions": num(report.get("actions_taken")),
        "turns": sum(1 for s in moving if s["action"] in ("turn_left", "turn_right")),
        "forward": sum(1 for s in moving if s["action"] == "move_forward"),
        "blocked": num(report.get("blocked_or_failed_actions")),
        "guard_replacements": num(report.get("guarded_action_replacements")),
        "oscillation_replacements": num(report.get("search_turn_oscillation_replacements")),
        "target_loss_recoveries": num(report.get("target_loss_recoveries")),
        "unlocated_sighting_approaches": num(report.get("unlocated_sighting_approaches")),
        "invalid_outputs": num(report.get("invalid_model_outputs")),
        "parse_failures": num(report.get("perception_parse_failures")),
        "calls": num(inference.get("calls")),
        "input_tokens": num(inference.get("input_tokens")),
        "output_tokens": num(inference.get("output_tokens")),
        "seconds": num(report.get("duration_seconds")),
        "window_mention_steps": sum(1 for s in steps if s["mention"]),
        "conflict_steps": len(conflict_steps),
        "first_conflict_step": conflict_steps[0] if conflict_steps else "",
        "target_visible_steps": sum(1 for s in steps if s["target_visible"]),
        "manual_label": "" if manual is None else str(bool(manual)),
        "reason": report.get("reason", ""),
        "steps": steps,
        "path": [p for p in path if p and None not in p],
        "max_steps_hint": 0,
    }


def sort_key(ep):
    return (
        INTERFACE_ORDER.index(ep["interface"]) if ep["interface"] in INTERFACE_ORDER else 9,
        MEMORY_ORDER.index(ep["memory"]) if ep["memory"] in MEMORY_ORDER else 9,
        ep["place"],
    )


def condition(ep):
    return f"{ep['interface'].replace('_', ' ')}\n{ep['memory']}"


def short_condition(interface, memory):
    short = {"direct_visual": "direct", "text_mediated": "text", "hybrid": "hybrid"}
    return f"{short.get(interface, interface)} / {memory}"


def save(fig, out_dir, name, show):
    fig.savefig(out_dir / name, dpi=160, bbox_inches="tight")
    if not show:
        plt.close(fig)


def fig_outcome_grid(episodes, out_dir, show):
    places = sorted({e["place"] for e in episodes})
    conds = [(i, m) for i in INTERFACE_ORDER for m in MEMORY_ORDER
             if any(e["interface"] == i and e["memory"] == m for e in episodes)]
    lookup = {(e["interface"], e["memory"], e["place"]): e for e in episodes}
    fig, ax = plt.subplots(figsize=(1.6 * len(places) + 4.2, 0.62 * len(conds) + 2.2))
    for r, (interface, memory) in enumerate(conds):
        for c, place in enumerate(places):
            ep = lookup.get((interface, memory, place))
            if ep is None:
                continue
            ax.add_patch(plt.Rectangle((c, r), 0.94, 0.9, color=OUTCOME_COLORS[ep["outcome"]]))
            ax.text(c + 0.47, r + 0.45, str(ep["actions"]), ha="center", va="center",
                    fontsize=11, fontweight="bold", color="white")
            if ep["outcome"] == "completed_unreviewed" and ep["actions"] <= 1:
                ax.text(c + 0.47, r + 0.8, "≤1 action", ha="center", va="center",
                        fontsize=6.5, color="#333")
    ax.set_xlim(0, len(places))
    ax.set_ylim(len(conds), 0)
    ax.set_xticks([i + 0.47 for i in range(len(places))])
    ax.set_xticklabels([p.replace("eval_", "") for p in places])
    ax.set_yticks([i + 0.45 for i in range(len(conds))])
    ax.set_yticklabels([short_condition(i, m) for i, m in conds])
    ax.xaxis.tick_top()
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    present = [o for o in OUTCOME_COLORS if any(e["outcome"] == o for e in episodes)]
    ax.legend(handles=[Patch(color=OUTCOME_COLORS[o], label=OUTCOME_LABELS[o]) for o in present],
              loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=8)
    ax.set_title("Outcome per episode (number = actions used)", pad=28)
    save(fig, out_dir, "01_outcome_grid.png", show)


def fig_actions(episodes, out_dir, show):
    places = sorted({e["place"] for e in episodes})
    conds = [(i, m) for i in INTERFACE_ORDER for m in MEMORY_ORDER
             if any(e["interface"] == i and e["memory"] == m for e in episodes)]
    limit = max((e["actions"] for e in episodes if e["status"] == "action_limit"), default=None)
    fig, ax = plt.subplots(figsize=(max(8, 1.25 * len(conds) + 2), 4.5))
    width = 0.8 / max(1, len(places))
    greys = plt.cm.Blues([0.4 + 0.5 * k / max(1, len(places) - 1) for k in range(len(places))])
    for k, place in enumerate(places):
        for x, (interface, memory) in enumerate(conds):
            ep = next((e for e in episodes if (e["interface"], e["memory"], e["place"]) ==
                       (interface, memory, place)), None)
            if ep is None:
                continue
            hatch = "///" if ep["outcome"] == "action_limit" else None
            ax.bar(x - 0.4 + width * (k + 0.5), ep["actions"], width * 0.92,
                   color=OUTCOME_COLORS[ep["outcome"]], edgecolor="white", hatch=hatch)
            ax.text(x - 0.4 + width * (k + 0.5), ep["actions"] + 0.3, place[-1],
                    ha="center", fontsize=7, color="#444")
    if limit:
        ax.axhline(limit, color="#555", ls="--", lw=0.8)
        ax.text(len(conds) - 0.45, limit + 0.3, f"{limit}-action limit", ha="right", fontsize=8)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([short_condition(i, m).replace(" / ", "\n") for i, m in conds], fontsize=8)
    ax.set_ylabel("Actions used")
    ax.set_title("Actions used (letter = start position; hatched = never finished)")
    present = [o for o in OUTCOME_COLORS if any(e["outcome"] == o for e in episodes)]
    ax.legend(handles=[Patch(color=OUTCOME_COLORS[o], label=OUTCOME_LABELS[o]) for o in present],
              fontsize=7.5, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    save(fig, out_dir, "02_actions_to_finish.png", show)


def fig_cost(episodes, out_dir, show):
    conds = [(i, m) for i in INTERFACE_ORDER for m in MEMORY_ORDER
             if any(e["interface"] == i and e["memory"] == m for e in episodes)]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    metrics = [("input_tokens", "Input tokens / episode"),
               ("output_tokens", "Output tokens / episode"),
               ("seconds", "Wall time / episode (s)")]
    for ax, (key, title) in zip(axes, metrics):
        for x, (interface, memory) in enumerate(conds):
            group = [e for e in episodes if e["interface"] == interface and e["memory"] == memory]
            values = [e[key] for e in group]
            ax.bar(x, sum(values) / len(values), color=MEMORY_COLORS.get(memory, "#999"),
                   alpha=0.45, width=0.7)
            for e in group:
                ax.scatter(x, e[key], s=26, zorder=3,
                           color=OUTCOME_COLORS[e["outcome"]], edgecolor="black", linewidth=0.4)
        ax.set_xticks(range(len(conds)))
        ax.set_xticklabels([short_condition(i, m).replace(" / ", "\n") for i, m in conds], fontsize=7.5)
        ax.set_title(title, fontsize=10)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    handles = [Line2D([0], [0], marker="o", ls="", color=OUTCOME_COLORS[o], markeredgecolor="black",
                      label=OUTCOME_LABELS[o]) for o in OUTCOME_COLORS
               if any(e["outcome"] == o for e in episodes)]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.06))
    fig.suptitle("Cost per condition (bar = mean, dots = individual episodes)", y=1.02)
    save(fig, out_dir, "03_cost.png", show)


def fig_timeline(episodes, out_dir, show):
    rows = sorted(episodes, key=sort_key)
    max_len = max(len(e["steps"]) for e in rows)
    fig, ax = plt.subplots(figsize=(max(9, 0.52 * max_len + 5), 0.3 * len(rows) + 2.2))
    for r, ep in enumerate(rows):
        for s in ep["steps"]:
            c = (s["step"] or 1) - 1
            if s["completed"]:
                ax.add_patch(plt.Rectangle((c, r), 0.92, 0.84, color=OUTCOME_COLORS[ep["outcome"]]))
                ax.text(c + 0.46, r + 0.44, "STOP", ha="center", va="center", fontsize=5.5,
                        color="white", fontweight="bold")
                continue
            action = s["action"] or s["requested"] or ""
            color = ACTION_COLORS.get(action, "#dddddd")
            ax.add_patch(plt.Rectangle((c, r), 0.92, 0.84, color=color))
            if s["outcome"] and s["outcome"] != "succeeded":
                ax.add_patch(plt.Rectangle((c, r), 0.92, 0.84, fill=False, edgecolor="#d00000", lw=1.6))
            if s["requested"] and s["action"] and s["requested"] != s["action"]:
                ax.plot(c + 0.12, r + 0.12, "o", color="black", ms=2.8)
            if s["mention"]:
                ax.text(c + 0.46, r + 0.44, "W", ha="center", va="center", fontsize=7,
                        fontweight="bold", color="#b8860b" if not s["target_visible"] else "white")
            if s["target_visible"]:
                ax.add_patch(plt.Rectangle((c + 0.05, r + 0.05), 0.82, 0.74, fill=False,
                                           edgecolor="gold", lw=1.4))
    ax.set_xlim(0, max_len)
    ax.set_ylim(len(rows), 0)
    ax.set_xticks([i + 0.46 for i in range(max_len)])
    ax.set_xticklabels(range(1, max_len + 1), fontsize=7)
    ax.set_yticks([i + 0.42 for i in range(len(rows))])
    ax.set_yticklabels([f"{short_condition(e['interface'], e['memory'])}  {e['place'][-1]}" for e in rows],
                       fontsize=7.5)
    ax.set_xlabel("Step")
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    handles = [Patch(color=c, label=a) for a, c in ACTION_COLORS.items()
               if any(s["action"] == a for e in rows for s in e["steps"])]
    handles += [
        Line2D([0], [0], marker="s", ls="", markerfacecolor="none", markeredgecolor="#d00000", label="blocked/failed"),
        Line2D([0], [0], marker="o", ls="", color="black", ms=4, label="controller changed the action"),
        Line2D([0], [0], marker="$W$", ls="", color="#b8860b", ms=8, label="model text names target, target_visible=false"),
        Line2D([0], [0], marker="s", ls="", markerfacecolor="none", markeredgecolor="gold", label="target_visible=true"),
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, fontsize=7.5)
    ax.set_title("Step-by-step timeline (one row per episode; last letter = start position)", pad=34)
    save(fig, out_dir, "04_step_timeline.png", show)


def fig_conflicts(episodes, out_dir, show):
    conds = [(i, m) for i in INTERFACE_ORDER for m in MEMORY_ORDER
             if any(e["interface"] == i and e["memory"] == m for e in episodes)]
    fig, ax = plt.subplots(figsize=(max(8, 1.25 * len(conds) + 2), 4.3))
    for x, (interface, memory) in enumerate(conds):
        group = [e for e in episodes if e["interface"] == interface and e["memory"] == memory]
        steps = sum(len([s for s in e["steps"] if not s["completed"]]) for e in group)
        conflict = sum(e["conflict_steps"] for e in group)
        agreed = sum(1 for e in group for s in e["steps"]
                     if s["mention"] and s["target_visible"] and not s["completed"])
        ax.bar(x, 100 * conflict / steps if steps else 0, color="#e07a5f", width=0.7,
               label="names the target but target_visible=false" if x == 0 else None)
        ax.bar(x, 100 * agreed / steps if steps else 0,
               bottom=100 * conflict / steps if steps else 0, color="#81b29a", width=0.7,
               label="names the target and target_visible=true" if x == 0 else None)
        ax.text(x, 100 * (conflict + agreed) / steps + 1.2 if steps else 1,
                f"{conflict}/{steps}", ha="center", fontsize=7.5)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([short_condition(i, m).replace(" / ", "\n") for i, m in conds], fontsize=8)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    ax.set_ylabel("% of non-stop steps")
    ax.set_title("Steps where the model's own text names the target\n(label = conflicting steps / all non-stop steps)", fontsize=10, pad=14)
    ax.legend(frameon=False, fontsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    save(fig, out_dir, "05_decision_conflicts.png", show)


def fig_trajectories(episodes, out_dir, show):
    places = sorted({e["place"] for e in episodes})
    fig, axes = plt.subplots(1, len(places), figsize=(5.2 * len(places), 5), squeeze=False)
    for ax, place in zip(axes[0], places):
        group = [e for e in episodes if e["place"] == place and e["path"]]
        for ep in group:
            xs, zs = zip(*ep["path"])
            ax.plot(xs, zs, color=MEMORY_COLORS.get(ep["memory"], "#777"),
                    ls={"direct_visual": "-", "hybrid": "--"}.get(ep["interface"], ":"), lw=1.6, alpha=0.85)
            marker = "*" if ep["outcome"] != "action_limit" else "X"
            ax.scatter(xs[-1], zs[-1], marker=marker, s=90 if marker == "*" else 55,
                       color=MEMORY_COLORS.get(ep["memory"], "#777"), edgecolor="black", linewidth=0.5, zorder=4)
        if group:
            ax.scatter(*group[0]["path"][0], marker="o", s=70, color="black", zorder=5)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_title(place.replace("eval_", "start "))
        ax.set_xlabel("x (m)")
        ax.set_ylabel("z (m)")
        ax.grid(alpha=0.25)
    handles = [Line2D([0], [0], color=c, label=m) for m, c in MEMORY_COLORS.items()]
    handles += [
        Line2D([0], [0], color="#555", ls="-", label="direct visual"),
        Line2D([0], [0], color="#555", ls=":", label="text mediated"),
        Line2D([0], [0], marker="o", ls="", color="black", label="start"),
        Line2D([0], [0], marker="*", ls="", color="#555", markersize=11, label="stopped (completed)"),
        Line2D([0], [0], marker="X", ls="", color="#555", label="action limit"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.07))
    fig.suptitle("Where the agent went (turns in place do not move the path)", y=1.0)
    save(fig, out_dir, "06_trajectories.png", show)


def fig_interventions(episodes, out_dir, show):
    conds = [(i, m) for i in INTERFACE_ORDER for m in MEMORY_ORDER
             if any(e["interface"] == i and e["memory"] == m for e in episodes)]
    series = [
        ("guard_replacements", "Guard replaced the model's action", "#457b9d"),
        ("oscillation_replacements", "Search-turn reversal replaced", "#e9c46a"),
        ("target_loss_recoveries", "Turned back toward a lost target", "#2d6a4f"),
        ("unlocated_sighting_approaches", "Approached a sighting with no side", "#bc6c25"),
        ("blocked", "Blocked / failed move", "#d62828"),
        ("invalid_outputs", "Invalid model output", "#6d597a"),
        ("parse_failures", "Perception parse failure", "#2a9d8f"),
    ]
    fig, ax = plt.subplots(figsize=(max(8, 1.25 * len(conds) + 3), 4.3))
    for x, (interface, memory) in enumerate(conds):
        group = [e for e in episodes if e["interface"] == interface and e["memory"] == memory]
        bottom = 0.0
        for key, _label, color in series:
            value = sum(e[key] for e in group) / len(group)
            ax.bar(x, value, bottom=bottom, color=color, width=0.7)
            bottom += value
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels([short_condition(i, m).replace(" / ", "\n") for i, m in conds], fontsize=8)
    ax.set_ylabel("Mean count per episode")
    ax.set_title("Controller interventions and model failures")
    ax.legend(handles=[Patch(color=c, label=l) for _k, l, c in series], frameon=False, fontsize=8,
              loc="upper left", bbox_to_anchor=(1.01, 1))
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    save(fig, out_dir, "07_interventions.png", show)


def write_table(episodes, out_dir):
    columns = ["episode_id", "interface", "memory", "place", "status", "outcome", "manual_label",
               "actions", "turns", "forward", "blocked", "guard_replacements",
               "oscillation_replacements", "target_loss_recoveries",
               "unlocated_sighting_approaches", "invalid_outputs",
               "parse_failures", "calls",
               "input_tokens", "output_tokens", "seconds", "window_mention_steps",
               "target_visible_steps", "conflict_steps", "first_conflict_step", "reason"]
    with (out_dir / "episode_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for ep in sorted(episodes, key=sort_key):
            writer.writerow(ep)


def write_findings(episodes, warnings, out_dir):
    lines = []
    total = len(episodes)
    counts = Counter(e["outcome"] for e in episodes)
    verified = counts["verified_success"]
    pending = counts["completed_unreviewed"]
    lines.append(f"Episodes: {total}")
    for key in OUTCOME_COLORS:
        if counts[key]:
            lines.append(f"  {OUTCOME_LABELS[key]}: {counts[key]}")
    lines.append(
        f"Verified success rate: {verified}/{total} (lower bound); "
        f"{verified + pending}/{total} if every pending label turns out true (upper bound)."
    )
    lines.append("")
    lines.append("Per condition (declared completion / lower-upper bound on verified success):")
    by_cond = defaultdict(list)
    for e in episodes:
        by_cond[(e["interface"], e["memory"])].append(e)
    for interface in INTERFACE_ORDER:
        for memory in MEMORY_ORDER:
            group = by_cond.get((interface, memory))
            if not group:
                continue
            done = sum(1 for e in group if e["status"] == "completed")
            low = sum(1 for e in group if e["outcome"] == "verified_success")
            high = low + sum(1 for e in group if e["outcome"] == "completed_unreviewed")
            conflicts = sum(e["conflict_steps"] for e in group)
            out_tok = sum(e["output_tokens"] for e in group) / len(group)
            secs = sum(e["seconds"] for e in group) / len(group)
            lines.append(
                f"  {short_condition(interface, memory):22s} done {done}/{len(group)}  "
                f"verified {low}-{high}/{len(group)}  conflict steps {conflicts:3d}  "
                f"mean out tokens {out_tok:6.0f}  mean time {secs:5.0f}s"
            )
    lines.append("")
    lines.append("Episodes that did not finish:")
    for e in sorted(episodes, key=sort_key):
        if e["status"] == "completed":
            continue
        if e["status"] == "timeout":
            lines.append(
                f"  {short_condition(e['interface'], e['memory']):22s} {e['place']:11s} "
                f"watchdog timeout after {e['seconds']:.0f}s (no history in all_results.json; "
                "see the episode folder's partial_history.json)"
            )
            continue
        lines.append(
            f"  {short_condition(e['interface'], e['memory']):22s} {e['place']:11s} "
            f"turns {e['turns']:2d} forward {e['forward']:2d} blocked {e['blocked']} "
            f"target-named steps {e['window_mention_steps']:2d} "
            f"(target_visible=true on {e['target_visible_steps']}) conflicts {e['conflict_steps']}"
        )
    lines.append("")
    lines.append("Completed episodes that need a human label to count:")
    for e in sorted(episodes, key=sort_key):
        if e["outcome"] == "completed_unreviewed":
            note = "  <- finished within one action; navigation barely tested" if e["actions"] <= 1 else ""
            lines.append(f"  {short_condition(e['interface'], e['memory']):22s} {e['place']:11s} "
                         f"actions {e['actions']}{note}")
    if warnings:
        lines.append("")
        lines.append("Warnings:")
        lines.extend(f"  {w}" for w in warnings)
    text = "\n".join(lines)
    (out_dir / "findings.txt").write_text(text + "\n", encoding="utf-8")
    return text


def main():
    parser = argparse.ArgumentParser(description="Visualize a finished experiment run.", formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", nargs="?", help="folder containing all_results.json")
    parser.add_argument("--out", help="output folder (default: <run_dir>/figures)")
    parser.add_argument("--show", action="store_true", help="also open the figures in windows")
    args = parser.parse_args()
    if args.show:
        matplotlib.use("TkAgg", force=True)

    run_dir = find_run_dir(args.run_dir)
    out_dir = Path(args.out) if args.out else run_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    reports = json.loads((run_dir / "all_results.json").read_text("utf-8"))
    labels, warnings = load_labels(run_dir)
    episodes = [build_episode(r, labels) for r in reports]
    known = {e["episode_id"] for e in episodes}
    warnings += [f"Label for unknown episode {k} ignored." for k in labels if k not in known]
    if not episodes:
        sys.exit("all_results.json contains no episodes.")

    summary_path = run_dir / "summary.csv"
    if summary_path.exists():
        with summary_path.open(newline="", encoding="utf-8-sig") as handle:
            rows = {r["episode_id"]: r for r in csv.DictReader(handle)}
        for e in episodes:
            row = rows.get(e["episode_id"])
            if e["status"] == "timeout":
                continue
            if row and str(row.get("actions_taken")) != str(e["actions"]):
                warnings.append(f"{e['episode_id']}: summary.csv and all_results.json disagree on actions.")

    fig_outcome_grid(episodes, out_dir, args.show)
    fig_actions(episodes, out_dir, args.show)
    fig_cost(episodes, out_dir, args.show)
    fig_timeline(episodes, out_dir, args.show)
    fig_conflicts(episodes, out_dir, args.show)
    fig_trajectories(episodes, out_dir, args.show)
    fig_interventions(episodes, out_dir, args.show)
    write_table(episodes, out_dir)
    print(write_findings(episodes, warnings, out_dir))
    print(f"\nFigures and tables written to {out_dir.resolve()}")
    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
