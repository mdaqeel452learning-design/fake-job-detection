import json
import os

from flask import Flask, jsonify, render_template, request

import storage
from detection import Detector, METRICS_PATH
from storage import append_history, clear_history, load_history, new_history_item

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)
detector = Detector()


def load_metrics():
    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"accuracy": None, "precision": None, "recall": None, "f1_score": None}


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

    result = detector.predict(text)

    append_history(new_history_item(text, result))
    history = load_history()

    result["stats"] = compute_stats(history)
    return jsonify(result)


@app.route("/history")
def history_page():
    history = list(reversed(load_history()))
    return render_template("history.html", history=history)


@app.route("/api/history/clear", methods=["POST"])
def clear_history_route():
    clear_history()
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


@app.route("/api/debug/storage")
def debug_storage():
    # No secrets exposed - just whether the DB connected and, if not, why.
    return jsonify({
        "db_ready": storage._db_ready,
        "db_error": storage._db_error,
        "history_count": len(load_history()),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True, use_reloader=False)
