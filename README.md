# Fake Job Detection — Web App

A Flask website that predicts whether a job posting is Real or Fake, combining
a trained ML model (TF-IDF + MLP neural network) with a weighted rule-based
scam-phrase check.

## Run locally

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
python train_model.py   # trains model.pkl, vectorizer.pkl, metrics.json (only needed once, or to retrain)
python app.py
```

Open http://localhost:5000

`model.pkl`, `vectorizer.pkl`, and `metrics.json` are already generated and
committed, so `python train_model.py` is optional unless you want to retrain
on updated data.

## Deploy for free

### Option A — Render.com (recommended, easiest)

1. Push this `web/` folder to a GitHub repo.
2. Go to https://render.com → New → Web Service → connect the repo.
3. Render will detect `render.yaml` automatically (Blueprint), or set manually:
   - Build command: `pip install -r requirements.txt && python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"`
   - Start command: `gunicorn app:app`
4. Choose the **Free** plan and deploy. You'll get a URL like
   `https://fake-job-detection.onrender.com`.

Note: Render's free web services spin down after 15 minutes of inactivity
and take ~30–60s to wake back up on the next visit — normal for a free demo.

### Option B — PythonAnywhere (free, no sleep, no card)

1. Create a free account at https://www.pythonanywhere.com
2. Upload this `web/` folder (or `git clone` it) via a Bash console.
3. `pip install --user -r requirements.txt` and run the nltk download command above.
4. In the **Web** tab, create a new Flask web app pointing at `app.py`
   (`app` is the Flask instance name).
5. Reload the app — your site is live at `https://<username>.pythonanywhere.com`.

### Option C — Hugging Face Spaces (free, good for ML demos)

1. Create a new Space → SDK: **Docker**.
2. Add a `Dockerfile`:
   ```dockerfile
   FROM python:3.11-slim
   WORKDIR /app
   COPY . .
   RUN pip install -r requirements.txt && \
       python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
   EXPOSE 7860
   CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]
   ```
3. Push this folder + the Dockerfile to the Space's git repo.

## Storage note

Prediction history is stored in `prediction_history.json` on the server's
local disk. Free hosting tiers (Render, Spaces) typically use **ephemeral**
storage, so history resets on redeploy/restart — expected for a free demo,
not a bug. For persistent history, swap the JSON file for a hosted database
(e.g. free tier of Supabase/Neon Postgres) later.

## What changed vs. the original desktop app

- **Reliability fix**: the old logic treated a single matched keyword
  (even generic phrases like "work from home") as an automatic Fake verdict,
  regardless of what the ML model predicted. It's now a weighted blend
  (65% ML probability + 35% rule signal), so one generic phrase can no
  longer flip a legitimate posting to Fake. See `detection.py`.
- **Real accuracy**: `train_model.py` now does a stratified train/test split
  and reports measured accuracy/precision/recall/F1 on a held-out set
  (`metrics.json`), instead of a hardcoded "94.12%" string.
- **Class imbalance handling**: the training set (only) is balanced via
  oversampling of the minority (fraudulent) class; the test set stays at the
  real ~5% fraud rate for a trustworthy accuracy number.
