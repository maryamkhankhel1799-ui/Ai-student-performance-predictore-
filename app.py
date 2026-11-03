"""AI Student Performance Predictor - main Streamlit app."""
import os

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from modules import analysis, embeddings, llm, planner, predictor

load_dotenv()
st.set_page_config(page_title="AI Student Performance Predictor", page_icon="🎓", layout="wide")

st.markdown("""
<style>
.hero{background:linear-gradient(120deg,#6C5CE7,#00B8D9);padding:26px 30px;border-radius:20px;
      color:white;margin-bottom:18px}
.hero h1{margin:0;font-size:2.1rem;color:white}
.hero p{margin:6px 0 0;font-size:1.05rem;color:#f1f5ff}
[data-testid="stMetric"]{background:white;border-radius:16px;padding:14px 16px;
      box-shadow:0 4px 14px rgba(108,92,231,.15);border-left:6px solid #6C5CE7}
.stTabs [data-baseweb="tab"]{font-weight:700;font-size:1rem}
.stButton>button{border-radius:12px;font-weight:700}
</style>
""", unsafe_allow_html=True)


def default_key():
    key = os.getenv("GROQ_API_KEY", "")
    if not key:
        try:
            key = st.secrets.get("GROQ_API_KEY", "")
        except Exception:
            key = ""
    return key


def show_fig(fig):
    st.pyplot(fig)
    plt.close(fig)


# ---------- session state ----------
if "marks_source" not in st.session_state:
    st.session_state.marks_source = pd.DataFrame({
        "Subject": ["Math", "Physics", "Chemistry", "English", "Computer Science"],
        "Marks": [72.0, 58.0, 65.0, 80.0, 88.0],
    })
if "ai_text" not in st.session_state:
    st.session_state.ai_text = ""
flash = st.session_state.pop("flash", None)
if flash:
    st.toast(flash)

st.markdown('<div class="hero"><h1>🎓 AI Student Performance Predictor</h1>'
            '<p>Enter your numbers, see your predicted performance, and get a personal study plan.</p></div>',
            unsafe_allow_html=True)

# ---------- sidebar ----------
with st.sidebar:
    st.header("👤 Student details")
    name = st.text_input("Name", "Student")
    attendance = st.slider("Attendance (%)", 0, 100, 85)
    study_hours = st.slider("Study hours per day", 0.0, 12.0, 3.0, 0.5)
    prev_gpa = st.number_input("Previous GPA (0-4)", 0.0, 4.0, 3.0, 0.1)
    st.header("🔑 Groq API")
    api_key = st.text_input("Groq API key", value=default_key(), type="password",
                            help="Get a free key at console.groq.com")
    with st.expander("🧠 AI models in use"):
        st.write(f"**Language:** {llm.LANGUAGE_MODEL}")
        st.write(f"**Coding:** {llm.CODE_MODEL}")
        st.write(f"**Vision:** {llm.VISION_MODEL}")
        st.write(f"**Embedding:** {embeddings.backend_name()}")

tab_predict, tab_insights, tab_coach, tab_scan, tab_code = st.tabs(
    ["📝 Predict", "📊 Insights & Plan", "🤖 AI Coach", "🖼️ Scan Marksheet", "💻 Code Helper"])

marks, avg, score = {}, 0.0, 0.0
weak = []

# ---------- tab 1: predict ----------
with tab_predict:
    st.subheader("Enter your subject marks (out of 100)")
    st.caption("Click a cell to edit. Use the empty bottom row to add a subject.")
    edited = st.data_editor(
        st.session_state.marks_source,
        num_rows="dynamic",
        column_config={"Marks": st.column_config.NumberColumn(min_value=0, max_value=100, step=1)},
    )
    marks = analysis.clean_marks(edited)
    if not marks:
        st.warning("Add at least one subject with marks to see your prediction.")
    else:
        avg = sum(marks.values()) / len(marks)
        score = predictor.predict_score(avg, attendance, study_hours, prev_gpa)
        label, emoji, advice = predictor.performance_level(score)
        weak = analysis.weak_subjects(marks)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Predicted score", f"{score:.1f}%")
        c2.metric("Performance", f"{emoji} {label}")
        c3.metric("Average marks", f"{avg:.1f}")
        c4.metric("Estimated GPA", f"{predictor.estimated_gpa(score)}")
        st.info(advice)

        st.subheader("⚠️ Weak subjects")
        if weak:
            for subject, mark in weak:
                st.error(f"**{subject}**: {mark:.0f}/100 needs extra attention")
        else:
            lowest = min(marks, key=marks.get)
            st.success(f"No subject is below 60. Your lowest is **{lowest}** ({marks[lowest]:.0f}). Keep improving it!")
        st.caption("This is an estimate from a simple practice model, not a guarantee.")

# ---------- tab 2: insights ----------
with tab_insights:
    if not marks:
        st.info("Enter your marks in the Predict tab first.")
    else:
        left, right = st.columns(2)
        with left:
            show_fig(analysis.marks_bar(marks))
            show_fig(analysis.whatif_chart(avg, attendance, study_hours, prev_gpa))
        with right:
            show_fig(analysis.factors_chart(avg, attendance, study_hours, prev_gpa))
            show_fig(analysis.time_donut(planner.allocate_hours(marks, max(study_hours, 1))))
        st.subheader("📅 Your weekly study plan")
        st.dataframe(planner.weekly_plan(marks, max(study_hours, 1)), hide_index=True)

# ---------- tab 3: AI coach ----------
with tab_coach:
    if not marks:
        st.info("Enter your marks in the Predict tab first.")
    else:
        parts = []
        if weak:
            parts.append("weak in " + ", ".join(s for s, _ in weak))
        else:
            parts.append("improve overall performance")
        if attendance < 75:
            parts.append("low attendance")
        if study_hours < 2:
            parts.append("not enough study time")
        if avg < 60:
            parts.append("low marks fix foundations")
        resources = embeddings.top_resources(" ".join(parts), 3)

        st.subheader("🔎 Best-matching study techniques")
        st.caption(f"Found with the embedding model: {embeddings.backend_name()}")
        for r in resources:
            st.markdown(f"- **{r['title']}**")

        st.subheader("✨ Personalized AI recommendations")
        if st.button("Generate AI recommendations", type="primary"):
            if not api_key:
                st.warning("Enter your Groq API key in the sidebar first.")
            else:
                profile = {
                    "name": name, "marks": marks, "attendance": attendance,
                    "study_hours": study_hours, "gpa": prev_gpa, "score": score,
                    "level": predictor.performance_level(score)[0],
                    "weak": weak, "resources": resources,
                }
                with st.spinner("Your AI coach is thinking..."):
                    try:
                        client = llm.get_client(api_key)
                        st.session_state.ai_text = llm.generate_recommendations(client, profile)
                    except Exception as error:
                        st.error(f"Could not get recommendations: {error}")
        if st.session_state.ai_text:
            st.markdown(st.session_state.ai_text)

# ---------- tab 4: scan marksheet ----------
with tab_scan:
    st.subheader("Upload a photo of your report card")
    st.caption("The vision model reads your subjects and marks and fills the Predict table.")
    upload = st.file_uploader("Report card image", type=["png", "jpg", "jpeg"])
    scanned = None
    if upload is not None:
        st.image(upload.getvalue(), caption="Your upload", width=450)
        if st.button("🔍 Read marks with AI vision", type="primary"):
            if not api_key:
                st.warning("Enter your Groq API key in the sidebar first.")
            else:
                with st.spinner("Reading your marksheet..."):
                    try:
                        client = llm.get_client(api_key)
                        scanned = llm.read_marksheet(client, upload.getvalue(), upload.type or "image/jpeg")
                    except Exception as error:
                        st.error(f"Could not read the image: {error}")
    if scanned:
        st.session_state.marks_source = pd.DataFrame(
            {"Subject": list(scanned.keys()), "Marks": list(scanned.values())})
        st.session_state.flash = "Marks loaded! Check them in the Predict tab."
        st.rerun()

# ---------- tab 5: code helper ----------
with tab_code:
    st.subheader("Coding assistant")
    st.caption("Ask for small programs, for example a GPA calculator or a study timer.")
    language = st.selectbox("Language", ["Python", "JavaScript", "SQL", "Java", "C++"])
    question = st.text_area("What do you want to build?",
                            "Write a function that calculates my GPA from a list of marks.")
    if st.button("Generate code", type="primary"):
        if not api_key:
            st.warning("Enter your Groq API key in the sidebar first.")
        elif not question.strip():
            st.warning("Please type a question.")
        else:
            with st.spinner("Writing code..."):
                try:
                    client = llm.get_client(api_key)
                    st.markdown(llm.code_assistant(client, language, question))
                except Exception as error:
                    st.error(f"Could not generate code: {error}")

st.divider()
st.caption("Predictions and AI advice are general guidance, not official academic results.")
