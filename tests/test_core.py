"""Quick checks that need no API key. Run: python tests/test_core.py"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules import analysis, embeddings, llm, planner, predictor  # noqa: E402


def run():
    low = predictor.predict_score(50, 60, 1, 2.0)
    high = predictor.predict_score(90, 95, 6, 3.8)
    assert 0 <= low < high <= 100, "higher inputs should give a higher score"

    df = pd.DataFrame({"Subject": ["Math", "", "Art"], "Marks": [45, 90, "abc"]})
    marks = analysis.clean_marks(df)
    assert marks == {"Math": 45.0}, marks
    assert analysis.weak_subjects({"A": 40, "B": 80, "C": 55}) == [("A", 40), ("C", 55)]

    sample = {"Math": 45.0, "Physics": 58.0, "English": 80.0}
    hours = planner.allocate_hours(sample, 3)
    assert abs(sum(hours.values()) - 3) < 0.05
    assert len(planner.weekly_plan(sample, 3)) == 7
    assert len(planner.weekly_plan({"Only": 70.0}, 2)) == 7

    assert len(embeddings.top_resources("weak in math problems", 3)) == 3
    assert llm.parse_marks_json('Here: {"Math": "78", "Art": 101, "Bad": "x"}') == {"Math": 78.0, "Art": 100.0}

    for fig in (analysis.marks_bar(sample), analysis.factors_chart(70, 85, 3, 3.0),
                analysis.whatif_chart(70, 85, 3, 3.0), analysis.time_donut(hours)):
        assert fig is not None
    print("All checks passed ✅")


if __name__ == "__main__":
    run()
