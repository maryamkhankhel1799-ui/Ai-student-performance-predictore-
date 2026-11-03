"""Rule-based study planner (works without any API key)."""
import pandas as pd

DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
ACTIVITIES = [
    "Learn concepts and write short notes",
    "Solve practice questions",
    "Timed past-paper practice",
    "Flashcards and self-quiz",
    "Fix mistakes from earlier practice",
    "Mixed practice and quick revision",
]


def allocate_hours(marks, daily_hours):
    """Give weaker subjects more of the daily study time."""
    daily_hours = max(float(daily_hours), 0.5)
    weights = {s: (100 - m) + 10 for s, m in marks.items()}
    total = sum(weights.values())
    return {s: round(daily_hours * w / total, 2) for s, w in weights.items()}


def weekly_plan(marks, daily_hours):
    """Build a 7-day plan as a table. Weakest subjects appear more often."""
    daily_hours = max(float(daily_hours), 0.5)
    ordered = [s for s, _ in sorted(marks.items(), key=lambda kv: kv[1])]
    cycle = ordered + ordered[:2]
    rows = []
    for i, day in enumerate(DAYS[:6]):
        focus = cycle[i % len(cycle)]
        second = ordered[(ordered.index(focus) + 1) % len(ordered)]
        rows.append({
            "Day": day,
            "Main focus": focus,
            "Second subject": second if second != focus else "Revision",
            "Hours": round(daily_hours, 1),
            "Activity": ACTIVITIES[i],
        })
    rows.append({
        "Day": "Sunday",
        "Main focus": "Weekly review",
        "Second subject": "Rest and plan next week",
        "Hours": round(daily_hours / 2, 1),
        "Activity": "Review mistakes, light reading, and sleep well",
    })
    return pd.DataFrame(rows)
