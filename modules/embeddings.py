"""Embedding model: match a student's needs to study resources.

Uses sentence-transformers (all-MiniLM-L6-v2) if installed. Otherwise it falls
back to a small built-in NumPy hashing embedding so the app always works.
"""
import re
import zlib

import numpy as np

RESOURCES = [
    {"title": "Pomodoro focus sessions", "text": "study time focus distraction short sessions 25 minutes breaks concentration not enough study time"},
    {"title": "Active recall", "text": "memorize remember exam self-quiz flashcards test yourself revision low marks"},
    {"title": "Spaced repetition", "text": "forget revise review schedule memory long term revision exams"},
    {"title": "Maths and physics problem practice", "text": "math maths physics chemistry numerical problems practice solve formulas past papers weak in"},
    {"title": "Better writing and essays", "text": "english writing essay grammar reading vocabulary language literature weak in"},
    {"title": "Build strong foundations", "text": "basics concepts fundamentals low marks understand topic concept map improve overall performance fix foundations"},
    {"title": "Attendance habits", "text": "low attendance miss class lectures routine punctual catch up notes"},
    {"title": "Exam stress and sleep", "text": "stress anxiety sleep rest health exam pressure calm"},
    {"title": "Study group learning", "text": "group study friends explain teach discuss doubts questions"},
    {"title": "Weekly time blocking", "text": "timetable schedule plan time management weekly routine organize improve overall performance"},
    {"title": "Computer science practice", "text": "computer science programming coding practice projects debug algorithms weak in"},
    {"title": "Ask for help early", "text": "teacher tutor doubts help mentor stuck needs support low marks"},
]

_state = {"model": None, "tried": False, "vectors": None}


def _load_model():
    if not _state["tried"]:
        _state["tried"] = True
        try:
            from sentence_transformers import SentenceTransformer
            _state["model"] = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            _state["model"] = None
    return _state["model"]


def _hash_embed(texts, dim=256):
    vectors = np.zeros((len(texts), dim))
    for i, text in enumerate(texts):
        for word in re.findall(r"[a-z]+", text.lower()):
            vectors[i, zlib.crc32(word.encode()) % dim] += 1
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)


def embed(texts):
    model = _load_model()
    if model is not None:
        return np.asarray(model.encode(texts, normalize_embeddings=True))
    return _hash_embed(texts)


def backend_name():
    return "sentence-transformers (all-MiniLM-L6-v2)" if _load_model() else "built-in hashing fallback"


def top_resources(query, k=3):
    """Return the k resources most similar to the query."""
    if _state["vectors"] is None:
        _state["vectors"] = embed([r["title"] + " " + r["text"] for r in RESOURCES])
    scores = _state["vectors"] @ embed([query])[0]
    best = np.argsort(scores)[::-1][:k]
    return [RESOURCES[i] for i in best]
