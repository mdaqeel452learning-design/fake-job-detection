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
  regardless of what the ML model predicted. ML probability and the rule
  score are now combined as a probabilistic OR (`1 - (1-ml)*(1-rule)`), not
  a weighted average: either a confident ML judgment or overwhelming rule
  evidence (several severe scam phrases together) can independently flag
  Fake, but a single weak/generic phrase still can't decide anything alone.
  A weighted average was tried first but had a structural flaw — capping
  the rule weight low enough to stop one phrase from dominating also meant
  maxed-out rule evidence could never cross the threshold on its own if the
  ML model happened to disagree, even for blatant multi-red-flag scam text.
  See `detection.py`.
- **Real accuracy**: `train_model.py` now does a stratified train/test split
  and reports measured accuracy/precision/recall/F1 on a held-out set
  (`metrics.json`), instead of a hardcoded "94.12%" string.
- **Class imbalance handling**: the training set (only) is balanced via
  oversampling of the minority (fraudulent) class; the test set stays at the
  real ~5% fraud rate for a trustworthy accuracy number.
- **Fully automatic, text-only**: the training data's strongest fraud
  predictor by far is `has_company_logo`/`has_company_profile` (82%/84%
  present in real postings vs. 33%/32% in fake ones) — but that's a visual
  attribute of the original listing page, never present in pasted text, so
  it can't be used here without asking the user extra questions. Two
  automatically-extractable text proxies (salary/pay mention, ALL-CAPS
  ratio) were tried and measured with a standalone AUC of only 0.593
  (barely above chance) — too weak to reliably help, so the model stays
  plain TF-IDF text. Prediction requires nothing but the pasted text;
  there's no manual input of any kind. Note: exact accuracy/precision/
  recall numbers in `metrics.json` can shift by a couple of points between
  retrains even with the same code, due to normal floating-point
  non-determinism in neural network training — check `metrics.json` for
  the actual numbers behind the currently-deployed model.
- **Vagueness signal**: some scam postings are too thin to catch by wording
  alone — no named employer, no real way to apply, barely any detail.
  `vagueness_check()` in `detection.py` flags three text-derivable signals
  (unusually short, no application method found, no identifiable company
  name found), each weighted the same as a low-severity scam phrase and
  fed into the same rule score. A posting missing all three can tip to
  Fake; a single missing signal alone can't, since plenty of legitimate
  short gig postings exist too.
