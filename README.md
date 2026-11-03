# 🎓 AI Student Performance Predictor

A colorful Streamlit app that predicts a student's academic performance, finds weak subjects, draws performance graphs and creates a personalized study plan with Groq-powered AI.

## Features

- **Inputs:** subject marks, attendance, study hours per day, previous GPA
- **Prediction:** predicted score, performance level and estimated GPA (NumPy linear model)
- **Weak subjects:** automatic detection (below 60) with a lowest-subject fallback
- **Graphs (Matplotlib):** marks by subject, strengths at a glance, "what if I study more", study-time donut
- **Study plan:** a 7-day plan that gives weaker subjects more time (works with no API key)
- **AI models:**

| Role | Used for | Default model |
|---|---|---|
| Language model | Personalized recommendations and plan | `llama-3.3-70b-versatile` (Groq) |
| Coding model | Code Helper tab | `llama-3.3-70b-versatile` (Groq, coding prompt) |
| Vision model | Reads marks from a report card photo | `meta-llama/llama-4-scout-17b-16e-instruct` (Groq) |
| Embedding model | Matches study techniques to the student | `all-MiniLM-L6-v2` if installed, else a built-in NumPy fallback |

Groq does not offer an embeddings endpoint, so embeddings run locally. Model names on Groq change over time. If a model is retired, set a new one in `.env` (see below).

## Project structure

```
ai-student-performance-predictor/
├── app.py                 # Streamlit interface
├── requirements.txt
├── README.md
├── .env.example           # copy to .env and add your key
├── .gitignore
├── .streamlit/config.toml # colors and theme
├── modules/
│   ├── predictor.py       # score prediction (NumPy)
│   ├── analysis.py        # cleaning, weak subjects, charts (Pandas, Matplotlib)
│   ├── planner.py         # weekly study plan
│   ├── embeddings.py      # embedding model + resource matching
│   └── llm.py             # Groq: language, coding, vision
└── tests/test_core.py     # quick checks, no API key needed
```

## Run it locally

1. Install Python 3.10 or newer.
2. Open a terminal in the project folder and create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   ```
3. Install the packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Get a free API key at https://console.groq.com, then:
   ```bash
   cp .env.example .env             # Windows: copy .env.example .env
   ```
   Open `.env` and paste your key after `GROQ_API_KEY=`. You can also paste it into the app sidebar.
5. Start the app:
   ```bash
   streamlit run app.py
   ```
6. Optional checks: `python tests/test_core.py`
7. Optional better embeddings (large download): `pip install sentence-transformers`

Prediction, charts, weak subjects and the weekly plan work without a key. The AI Coach, Scan Marksheet and Code Helper tabs need one.

## Deploy free on Streamlit Community Cloud

1. Push the project to a GitHub repository.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app**, choose your repository, branch `main`, and main file `app.py`.
4. Open **Advanced settings → Secrets** and add:
   ```toml
   GROQ_API_KEY = "your_groq_api_key_here"
   ```
5. Click **Deploy**. You get a public link in a few minutes.

Keep `sentence-transformers` out of `requirements.txt` when deploying, since it is large. The fallback embedding works fine.

## Troubleshooting

- **Model not found or decommissioned:** pick a current model from the Groq console and set `GROQ_LANGUAGE_MODEL`, `GROQ_CODE_MODEL` or `GROQ_VISION_MODEL` in `.env`.
- **Invalid API key:** check for extra spaces and that the key is from console.groq.com.
- **Marksheet not read well:** use a bright, straight, close-up photo.
- **`ModuleNotFoundError`:** activate your virtual environment and run `pip install -r requirements.txt` again.

## Disclaimer

Predictions come from a simple practice model trained on generated data, and AI advice is general guidance. Neither is an official result or a guarantee.
