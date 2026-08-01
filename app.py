import json
import os
from datetime import datetime, timezone

from flask import Flask, jsonify, render_template, request

from detection import Detector, METRICS_PATH

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HISTORY_PATH = os.path.join(_BASE_DIR, "prediction_history.json")
MAX_HISTORY = 500

app = Flask(__name__)
detector = Detector()


def load_metrics():
    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"accuracy": None, "precision": None, "recall": None, "f1_score": None}


def load_history():
    try:
        with open(HISTORY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_history(history):
    try:
        with open(HISTORY_PATH, "w", encoding="utf-8") as f:
            json.dump(history[-MAX_HISTORY:], f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def compute_stats(history):
    total = len(history)
    fake = sum(1 for x in history if x.get("is_fake"))
    real = total - fake
    avg_conf = (sum(float(x.get("confidence") or 0) for x in history) / total) if total else 0.0
    return {
        "total": total,
        "fake": fake,
        "real": real,
        "fake_pct": round((fake / total * 100), 1) if total else 0.0,
        "real_pct": round((real / total * 100), 1) if total else 0.0,
        "avg_confidence": round(avg_conf, 2),
    }


@app.route("/")
def index():
    metrics = load_metrics()
    history = load_history()
    stats = compute_stats(history)
    return render_template("index.html", metrics=metrics, stats=stats)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(silent=True) or {}
    text = (payload.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Please enter a job description."}), 400

    def tri_state(value):
        if value in (True, "yes", "true", "1", 1):
            return True
        if value in (False, "no", "false", "0", 0):
            return False
        return None

    has_logo = tri_state(payload.get("has_logo"))
    has_profile = tri_state(payload.get("has_profile"))
    has_salary = tri_state(payload.get("has_salary"))

    result = detector.predict(text, has_logo=has_logo, has_profile=has_profile, has_salary=has_salary)

    history = load_history()
    history.append(
        {
            "text": text,
            "is_fake": result["is_fake"],
            "confidence": result["confidence"],
            "ml_probability": result["ml_probability"],
            "matched_patterns": result["matched_patterns"],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    save_history(history)

    result["stats"] = compute_stats(history)
    return jsonify(result)


@app.route("/history")
def history_page():
    history = list(reversed(load_history()))
    return render_template("history.html", history=history)


@app.route("/api/history/clear", methods=["POST"])
def clear_history():
    save_history([])
    return jsonify({"ok": True})


@app.route("/statistics")
def statistics_page():
    history = load_history()
    stats = compute_stats(history)
    metrics = load_metrics()
    recent = list(reversed(history[-10:]))
    return render_template("statistics.html", stats=stats, metrics=metrics, recent=recent, history=history)


@app.route("/about")
def about_page():
    metrics = load_metrics()
    return render_template("about.html", metrics=metrics)


@app.route("/how-it-works")
def how_it_works_page():
    return render_template("how_it_works.html")


@app.route("/contact")
def contact_page():
    return render_template("contact.html")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)
