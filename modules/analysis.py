"""Data cleaning, weak-subject detection and colorful Matplotlib charts."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from modules.predictor import predict_score

PALETTE = ["#6C5CE7", "#00B8D9", "#FF6B6B", "#FDCB6E",
           "#00B894", "#E17055", "#0984E3", "#A29BFE"]


def clean_marks(df):
    """Turn the marks table into {subject: marks} with valid numbers only."""
    marks = {}
    if df is None or len(df) == 0:
        return marks
    for _, row in df.iterrows():
        name = str(row.get("Subject", "")).strip()
        value = pd.to_numeric(row.get("Marks"), errors="coerce")
        if name and name.lower() != "nan" and pd.notna(value):
            marks[name] = float(np.clip(value, 0, 100))
    return marks


def weak_subjects(marks, threshold=60):
    """Subjects below the threshold, weakest first."""
    weak = [(s, m) for s, m in marks.items() if m < threshold]
    return sorted(weak, key=lambda item: item[1])


def _level_color(mark):
    if mark < 50:
        return "#FF6B6B"
    if mark < 60:
        return "#FDCB6E"
    if mark < 75:
        return "#00B8D9"
    return "#00B894"


def _style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)


def marks_bar(marks, threshold=60):
    fig, ax = plt.subplots(figsize=(7, 4))
    names, values = list(marks.keys()), list(marks.values())
    bars = ax.bar(names, values, color=[_level_color(v) for v in values], edgecolor="white")
    ax.axhline(threshold, color="#6C5CE7", linestyle="--", linewidth=1.5,
               label=f"Weak below {threshold}")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 1.5, f"{value:.0f}",
                ha="center", fontweight="bold")
    ax.set_ylim(0, 112)
    ax.set_ylabel("Marks")
    ax.set_title("Marks by subject", fontweight="bold")
    ax.legend(loc="upper right")
    _style(ax)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    return fig


def factors_chart(avg, attendance, study_hours, prev_gpa):
    labels = ["Average marks", "Attendance", "Study time (8h = 100)", "Previous GPA (4.0 = 100)"]
    values = [avg, attendance, min(study_hours / 8 * 100, 100), prev_gpa / 4 * 100]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.barh(labels, values, color=PALETTE[:4], edgecolor="white")
    ax.axvline(75, color="#2D3436", linestyle="--", linewidth=1, label="Target: 75")
    for bar, value in zip(bars, values):
        ax.text(value + 1.5, bar.get_y() + bar.get_height() / 2, f"{value:.0f}", va="center")
    ax.set_xlim(0, 115)
    ax.set_title("Your strengths at a glance", fontweight="bold")
    ax.legend(loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig


def whatif_chart(avg, attendance, study_hours, prev_gpa):
    extra = np.arange(0, 4.5, 0.5)
    scores = [predict_score(avg, attendance, min(study_hours + e, 12), prev_gpa) for e in extra]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(extra, scores, color="#6C5CE7", marker="o", linewidth=2.5)
    ax.fill_between(extra, scores, min(scores) - 2, color="#A29BFE", alpha=0.3)
    ax.set_xlabel("Extra study hours per day")
    ax.set_ylabel("Predicted score (%)")
    ax.set_title("What if you studied more?", fontweight="bold")
    _style(ax)
    fig.tight_layout()
    return fig


def time_donut(hours_by_subject):
    names, values = list(hours_by_subject.keys()), list(hours_by_subject.values())
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(names))]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.pie(values, labels=names, colors=colors, autopct="%1.0f%%", startangle=90,
           wedgeprops=dict(width=0.45, edgecolor="white"))
    ax.set_title("Daily study time split", fontweight="bold")
    fig.tight_layout()
    return fig
