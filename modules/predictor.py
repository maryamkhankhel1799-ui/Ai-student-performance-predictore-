"""Score prediction with a tiny linear model built with NumPy only."""
import numpy as np

_cache = {}


def make_training_data(n=500, seed=42):
    """Create practice data that follows common-sense study patterns."""
    rng = np.random.default_rng(seed)
    marks = rng.uniform(30, 100, n)
    attendance = rng.uniform(40, 100, n)
    hours = rng.uniform(0, 10, n)
    gpa = rng.uniform(1.5, 4.0, n)
    noise = rng.normal(0, 3, n)
    score = 8 + 0.40 * marks + 0.15 * attendance + 1.8 * hours + 5.0 * gpa + noise
    X = np.column_stack([marks, attendance, hours, gpa])
    return X, np.clip(score, 0, 100)


def _weights():
    if "w" not in _cache:
        X, y = make_training_data()
        A = np.column_stack([np.ones(len(X)), X])
        _cache["w"], *_ = np.linalg.lstsq(A, y, rcond=None)
    return _cache["w"]


def predict_score(avg_marks, attendance, study_hours, prev_gpa):
    """Return the predicted overall score (0-100)."""
    x = np.array([1.0, avg_marks, attendance, study_hours, prev_gpa])
    return float(np.clip(x @ _weights(), 0, 100))


def performance_level(score):
    """Return (label, emoji, advice) for a score."""
    if score >= 85:
        return "Excellent", "🌟", "Outstanding! Keep your routine and aim higher."
    if score >= 70:
        return "Good", "😊", "Solid progress. Strengthen your weaker subjects."
    if score >= 55:
        return "Average", "📈", "You can improve with a steady plan and more practice."
    return "Needs Support", "🚨", "Start early, ask teachers for help, and follow the plan."


def estimated_gpa(score):
    """Convert a 0-100 score to an approximate 4.0 GPA."""
    return round(score / 100 * 4, 2)
