"""Shared text preprocessing and hybrid fake-job scoring logic.

Used by both train_model.py (to build the model) and app.py (to serve
predictions), so training and inference always agree on how text is
prepared and how the final decision is made.
"""
import os
import re
import pickle

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(_BASE_DIR, "model.pkl")
VECTORIZER_PATH = os.path.join(_BASE_DIR, "vectorizer.pkl")
METRICS_PATH = os.path.join(_BASE_DIR, "metrics.json")

try:
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer
except Exception:  # nltk not installed / corpora not downloaded
    stopwords = None
    WordNetLemmatizer = None

_FALLBACK_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "he", "in", "is", "it", "its", "of", "on", "that", "the", "to", "was",
    "were", "will", "with",
}


def _safe_stopwords():
    if stopwords is None:
        return set(_FALLBACK_STOPWORDS)
    try:
        return set(stopwords.words("english"))
    except Exception:
        return set(_FALLBACK_STOPWORDS)


def _safe_lemmatizer():
    if WordNetLemmatizer is None:
        return None
    try:
        lemmatizer = WordNetLemmatizer()
        lemmatizer.lemmatize("test")  # forces corpus load / raises if missing
        return lemmatizer
    except Exception:
        return None


lemmatizer = _safe_lemmatizer()
stop_words = _safe_stopwords()


def preprocess_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-zA-Z]", " ", text)
    words = text.split()
    if lemmatizer is not None:
        words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    else:
        words = [w for w in words if w not in stop_words]
    return " ".join(words)


# Strong, hard-to-fake-innocently scam indicators.
HIGH_SEVERITY_PATTERNS = [
    "registration fee",
    "processing fee",
    "verification fee",
    "documentation fee",
    "application fee",
    "interview fee",
    "service fee",
    "joining fee",
    "security fee",
    "refundable fee",
    "pay a fee",
    "small fee",
    "security deposit",
    "deposit required",
    "send money",
    "money to secure",
    "wire transfer",
    "western union",
    "money gram",
    "buy a starter kit",
    "starter kit",
    "investment required",
    "whatsapp only",
    "contact via telegram",
    "telegram only",
    "no interview",
    "no company name",
    "pay before you start",
    "training fee",
]

# Regex versions catch the same intent (pay a fee before something happens)
# even when the exact phrase isn't in the literal list above — e.g. "a
# document verification fee ... must be paid before scheduling" doesn't
# contain any of the literal phrases word-for-word, but is the same scam.
HIGH_SEVERITY_REGEX = [
    re.compile(r"\bfee\b[^.]{0,60}\b(before|prior to)\b", re.IGNORECASE),
    re.compile(r"\b(pay|paid|payment)\b[^.]{0,60}\bfee\b[^.]{0,60}\b(confirm|secure|schedule|book)\b", re.IGNORECASE),
]

# Weaker indicators: common in real postings too, so they only nudge the
# score rather than deciding it on their own.
LOW_SEVERITY_PATTERNS = [
    "no experience",
    "no prior experience",
    "anyone can apply",
    "earn money",
    "easy money",
    "guaranteed income",
    "limited slots",
    "first come",
    "referral code",
    "apply fast",
    "immediate joining",
    "urgently hiring",
    "urgent hiring",
    "joining bonus",
    "weekly payment",
    "no targets",
    "no pressure",
    "simple data entry",
    "basic typing",
    "hours daily and earn",
    "whatsapp",
]

# Rule score (0..RULE_SCORE_CAP) beyond which the rule component saturates
# at 1.0.
RULE_SCORE_CAP = 6.0

FAKE_THRESHOLD = 0.5

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_URL_RE = re.compile(r"https?://|www\.", re.IGNORECASE)
_PHONE_RE = re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b")
_APPLY_PHRASE_RE = re.compile(
    r"\bapply\s+(at|here|online|via|through|now)\b|send\s+(your\s+)?resume|"
    r"click\s+here|visit\s+our|application\s+form",
    re.IGNORECASE,
)
_COMPANY_SUFFIX_RE = re.compile(
    r"\b[A-Z][\w&]*(?:\s[A-Z][\w&]*){0,3}\s"
    r"(Inc\.?|LLC|L\.L\.C\.|Corp\.?|Corporation|Company|Co\.|Ltd\.?|Group|"
    r"Technologies|Solutions|Enterprises|Industries)\b"
)
_COMPANY_LABEL_RE = re.compile(r"\b(company|employer|organization)\s*:\s*\S+", re.IGNORECASE)
VAGUE_WORD_COUNT_THRESHOLD = 60


def vagueness_check(text: str):
    """Detects postings that are short and give no way to verify who's
    hiring or how to actually apply. Each flag is a weak, independent
    nudge (same weight as a low-severity scam phrase) — never decisive
    alone — since plenty of real short gig postings exist too."""
    text = text or ""
    flags = []
    score = 0.0

    word_count = len(text.split())
    if word_count < VAGUE_WORD_COUNT_THRESHOLD:
        score += 1.0
        flags.append("Posting is unusually short on detail")

    has_apply_method = bool(
        _EMAIL_RE.search(text) or _URL_RE.search(text) or _PHONE_RE.search(text) or _APPLY_PHRASE_RE.search(text)
    )
    if not has_apply_method:
        score += 1.0
        flags.append("No clear way to apply (no email, link, or contact method)")

    has_company_name = bool(_COMPANY_SUFFIX_RE.search(text) or _COMPANY_LABEL_RE.search(text))
    if not has_company_name:
        score += 1.0
        flags.append("No identifiable company name mentioned")

    return score, flags


_NEGATION_CUES = ("no ", "not ", "without ", "never ", "exempt from", "free of", "does not", "won't", "will not")
_NEGATION_WINDOW = 30
_SENTENCE_BOUNDARY_RE = re.compile(r"[.!?\n]")


def _is_negated(text_lower: str, match_start: int) -> bool:
    """Checks for a negation cue right before a match, in the SAME clause
    only — so "exempt from any application fee" or "no registration fee
    required" don't score as scam signals just because the phrase appears,
    but an unrelated "No experience." in a prior sentence can't wrongly
    negate a later, separate match like "immediate joining"."""
    preceding = text_lower[:match_start]
    boundaries = [m.end() for m in _SENTENCE_BOUNDARY_RE.finditer(preceding)]
    clause_start = boundaries[-1] if boundaries else 0
    start = max(clause_start, match_start - _NEGATION_WINDOW)
    context = text_lower[start:match_start]
    return any(cue in context for cue in _NEGATION_CUES)


def rule_based_check(text: str):
    """Returns (raw_score, matched_patterns) for a piece of job text."""
    text_lower = (text or "").lower()
    matched = []
    score = 0.0
    for pattern in HIGH_SEVERITY_PATTERNS:
        idx = text_lower.find(pattern)
        if idx != -1 and not _is_negated(text_lower, idx):
            score += 2.0
            matched.append(pattern)
    for pattern in LOW_SEVERITY_PATTERNS:
        idx = text_lower.find(pattern)
        if idx != -1 and not _is_negated(text_lower, idx):
            score += 1.0
            matched.append(pattern)
    for regex in HIGH_SEVERITY_REGEX:
        m = regex.search(text_lower)
        if m and not _is_negated(text_lower, m.start()):
            score += 2.0
            matched.append(m.group(0).strip())

    vague_score, vague_flags = vagueness_check(text)
    score += vague_score
    matched.extend(vague_flags)

    return score, matched


def _load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# --- Template/duplicate-spam detection ---------------------------------
# Separate from the fraud-scam verdict on purpose. Mass-templated postings
# (the same boilerplate sentence submitted repeatedly with one word swapped)
# are a real job-board abuse pattern, but they are NOT the same thing as a
# scam designed to steal money/data — a single well-written posting can't
# be judged "templated" in isolation, since that requires comparing it
# against *other* submissions. Folding this into the ML training labels
# (tried and reverted) taught the model "well-formatted job posting =
# fraud", which wrongly flagged ordinary real postings. Comparing against
# recent submission history avoids that: it only fires on actual repetition.
DUPLICATE_SIMILARITY_THRESHOLD = 0.85
DUPLICATE_MIN_MATCHES = 2
DUPLICATE_HISTORY_WINDOW = 200


def duplicate_signal(text: str, history_texts, vectorizer) -> dict:
    """Compares `text` against recent prior submissions using the same
    TF-IDF vectorizer the model uses. sklearn's TfidfVectorizer L2-normalizes
    rows by default, so a plain dot product between two vectors already IS
    their cosine similarity — no extra library needed."""
    cleaned = preprocess_text(text)
    if not cleaned.strip() or not history_texts:
        return {"is_template_spam": False, "similar_count": 0}

    recent = history_texts[-DUPLICATE_HISTORY_WINDOW:]
    cleaned_history = [preprocess_text(t) for t in recent]
    vectors = vectorizer.transform([cleaned] + cleaned_history)
    target = vectors[0]
    others = vectors[1:]

    similarities = (others @ target.T).toarray().ravel()
    similar_count = int((similarities >= DUPLICATE_SIMILARITY_THRESHOLD).sum())

    return {
        "is_template_spam": similar_count >= DUPLICATE_MIN_MATCHES,
        "similar_count": similar_count,
    }


class Detector:
    """Loads the trained model/vectorizer once and scores job text."""

    def __init__(self, model_path=MODEL_PATH, vectorizer_path=VECTORIZER_PATH):
        self.model = _load_pickle(model_path)
        self.vectorizer = _load_pickle(vectorizer_path)

    def predict(self, text: str) -> dict:
        cleaned = preprocess_text(text)
        vector = self.vectorizer.transform([cleaned])

        proba = self.model.predict_proba(vector)[0]
        ml_prob_fake = float(proba[1]) if len(proba) > 1 else float(proba[0])

        raw_rule_score, matched = rule_based_check(text)
        rule_component = min(raw_rule_score / RULE_SCORE_CAP, 1.0)

        # Combine as a probabilistic OR, not a weighted average: either a
        # confident ML judgment OR overwhelming rule evidence (several
        # severe scam phrases) can independently push a posting to Fake.
        # A weighted average can't do this — with rule weight kept low
        # enough that one generic phrase can't dominate, maxed-out rule
        # evidence alone was structurally incapable of crossing the
        # threshold whenever the ML model happened to disagree, even for
        # blatant scam text with 4+ matched phrases. A single weak/generic
        # phrase still can't decide anything on its own here, since it only
        # produces a small rule_component (e.g. ~0.17 for one low-severity
        # match), which needs substantial ML agreement to cross 0.5.
        final_score = 1.0 - (1.0 - ml_prob_fake) * (1.0 - rule_component)
        is_fake = final_score >= FAKE_THRESHOLD

        return {
            "is_fake": bool(is_fake),
            "confidence": round(final_score * 100, 2) if is_fake else round((1 - final_score) * 100, 2),
            "fake_probability": round(final_score * 100, 2),
            "ml_probability": round(ml_prob_fake * 100, 2),
            "rule_score": raw_rule_score,
            "matched_patterns": matched,
        }
