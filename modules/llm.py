"""Groq API helpers: language model, coding model and vision model."""
import base64
import json
import os
import re

LANGUAGE_MODEL = os.getenv("GROQ_LANGUAGE_MODEL", "llama-3.3-70b-versatile")
CODE_MODEL = os.getenv("GROQ_CODE_MODEL", "llama-3.3-70b-versatile")
VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct")


def get_client(api_key):
    if not api_key or not api_key.strip():
        raise ValueError("Please enter your Groq API key in the sidebar.")
    from groq import Groq  # imported here so the rest of the app works without it

    return Groq(api_key=api_key.strip())


def _chat(client, model, messages, temperature=0.4, max_tokens=1200):
    response = client.chat.completions.create(
        model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
    )
    return response.choices[0].message.content.strip()


def generate_recommendations(client, profile):
    """Language model: personalized advice and study plan."""
    marks_text = ", ".join(f"{s}: {m:.0f}" for s, m in profile["marks"].items())
    weak_text = ", ".join(s for s, _ in profile["weak"]) or "none"
    tips = "\n".join(f"- {r['title']}: {r['text']}" for r in profile["resources"])
    prompt = (
        f"Student: {profile['name']}\n"
        f"Marks (out of 100): {marks_text}\n"
        f"Attendance: {profile['attendance']}%\n"
        f"Study hours per day: {profile['study_hours']}\n"
        f"Previous GPA (4.0 scale): {profile['gpa']}\n"
        f"Predicted score: {profile['score']:.1f}% ({profile['level']})\n"
        f"Weak subjects: {weak_text}\n"
        f"Relevant study techniques:\n{tips}\n\n"
        "Write, in simple friendly language and Markdown:\n"
        "1. A 2-sentence summary of how the student is doing.\n"
        "2. Specific tips for each weak subject (or the lowest subject if none are weak).\n"
        "3. A 7-day personalized study plan as a short list.\n"
        "4. Three daily habits to improve.\n"
        "5. One motivating closing line.\n"
        "Keep it under 450 words. Do not promise results."
    )
    messages = [
        {"role": "system", "content": "You are a supportive academic coach for students."},
        {"role": "user", "content": prompt},
    ]
    return _chat(client, LANGUAGE_MODEL, messages)


def code_assistant(client, language, question):
    """Coding model: help students write small programs."""
    messages = [
        {"role": "system", "content": (
            f"You are a patient coding tutor. Answer with clean, commented {language} code "
            "in a Markdown code block, followed by a short beginner-friendly explanation.")},
        {"role": "user", "content": question},
    ]
    return _chat(client, CODE_MODEL, messages, temperature=0.2, max_tokens=1500)


def parse_marks_json(text):
    """Pull a {subject: marks} dictionary out of the model's reply."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("The AI did not return any marks. Try a clearer image.")
    data = json.loads(match.group(0))
    marks = {}
    for subject, value in data.items():
        try:
            marks[str(subject).strip()] = max(0.0, min(100.0, float(value)))
        except (TypeError, ValueError):
            continue
    if not marks:
        raise ValueError("No valid subject marks were found in the image.")
    return marks


def read_marksheet(client, image_bytes, mime_type="image/jpeg"):
    """Vision model: read subject marks from a photo of a report card."""
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": (
                "Read this report card or marksheet. Reply with ONLY a JSON object that maps "
                "each subject name to its marks out of 100 (convert to a percentage if needed). "
                'Example: {"Math": 78, "English": 84}. Ignore totals and ranks.')},
            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded}"}},
        ],
    }]
    reply = _chat(client, VISION_MODEL, messages, temperature=0.0, max_tokens=500)
    return parse_marks_json(reply)
