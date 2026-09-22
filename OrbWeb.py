"""
Agricultural Variety Identifier — Flask + HTML web version.

    pip install flask opencv-python numpy
    python web_app.py

Then open http://127.0.0.1:5000 in your browser.
"""
from __future__ import annotations

import base64
import json
import shutil
import time
from pathlib import Path
from typing import Dict, List, Tuple

try:
    from flask import Flask, jsonify, request, Response
except ImportError:
    raise SystemExit("Flask is required. Install with:  pip install flask")

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
APP_NAME = "Agricultural Variety Identifier"
APP_SHORT_NAME = "AgriVariety"
APP_VERSION = "3.2.0"

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
REFS_DIR = DATA_DIR / "references"
CAPS_DIR = DATA_DIR / "captures"
BANK_FILE = DATA_DIR / "bank.json"
LOG_FILE = DATA_DIR / "log.jsonl"

# Rename these to whatever varieties you want to recognise.
# The app is generic — it works for any crop, fruit, vegetable, or seed.
VARIETIES: Tuple[str, ...] = (
    "Variety One",
    "Variety Two",
    "Variety Three",
    "Variety Four",
)

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
ORB_FEATURES = 1000
MATCH_CEILING = 64
LOWE_RATIO = 0.75
WORK_WIDTH = 480


# --------------------------------------------------------------------------- #
# Filesystem helpers
# --------------------------------------------------------------------------- #
def ensure_dirs() -> None:
    REFS_DIR.mkdir(parents=True, exist_ok=True)
    CAPS_DIR.mkdir(parents=True, exist_ok=True)


def load_bank() -> Dict[str, List[dict]]:
    if not BANK_FILE.exists():
        return {v: [] for v in VARIETIES}
    try:
        raw = json.loads(BANK_FILE.read_text(encoding="utf-8"))
    except Exception:
        raw = {}
    for v in VARIETIES:
        raw.setdefault(v, [])
    return raw


def save_bank(bank: Dict[str, List[dict]]) -> None:
    BANK_FILE.write_text(
        json.dumps(bank, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def append_log(entry: dict) -> None:
    with LOG_FILE.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")


def read_log(limit: int = 200) -> List[dict]:
    if not LOG_FILE.exists():
        return []
    rows: List[dict] = []
    for line in LOG_FILE.read_text(encoding="utf-8").splitlines()[-limit:]:
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return list(reversed(rows))


def _wipe_dir_contents(folder: Path) -> int:
    """Delete every child of `folder` but keep the folder itself. Returns count."""
    if not folder.exists():
        return 0
    count = 0
    for child in folder.iterdir():
        try:
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
            count += 1
        except Exception:
            pass
    return count


def reset_all(scope: str = "all") -> dict:
    """
    scope:
        "all"    → references + captures + log + bank  (full factory reset)
        "bank"   → references + bank only              (keep log & captures)
        "log"    → log only                            (keep bank)
        "caps"   → captures only                       (keep bank & log)
    """
    result = {"scope": scope, "removed": {}}

    if scope in ("all", "bank"):
        result["removed"]["references_files"] = _wipe_dir_contents(REFS_DIR)
        save_bank({v: [] for v in VARIETIES})

    if scope in ("all", "log"):
        if LOG_FILE.exists():
            try:
                LOG_FILE.unlink()
                result["removed"]["log"] = 1
            except Exception:
                result["removed"]["log"] = 0

    if scope in ("all", "caps"):
        result["removed"]["capture_files"] = _wipe_dir_contents(CAPS_DIR)

    ensure_dirs()
    return result


# --------------------------------------------------------------------------- #
# ORB engine
# --------------------------------------------------------------------------- #
class ORBEngine:
    def __init__(self, n_features: int = ORB_FEATURES):
        self.orb = cv2.ORB_create(nfeatures=n_features, fastThreshold=12)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def _prepare(self, image_bgr: np.ndarray) -> np.ndarray:
        h, w = image_bgr.shape[:2]
        if w > WORK_WIDTH:
            scale = WORK_WIDTH / w
            image_bgr = cv2.resize(image_bgr, (int(w * scale), int(h * scale)))
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        return cv2.equalizeHist(gray)

    def _detect(self, gray: np.ndarray):
        kps, desc = self.orb.detectAndCompute(gray, None)
        return list(kps or []), desc

    def match(self, query_bgr, reference_bgr, label="reference") -> dict:
        q_gray = self._prepare(query_bgr)
        r_gray = self._prepare(reference_bgr)
        q_kps, q_desc = self._detect(q_gray)
        r_kps, r_desc = self._detect(r_gray)

        good: List[cv2.DMatch] = []
        if q_desc is not None and r_desc is not None and len(q_desc) and len(r_desc):
            knn = self.bf.knnMatch(q_desc, r_desc, k=2)
            for pair in knn:
                if len(pair) < 2:
                    continue
                m, n = pair
                if m.distance < LOWE_RATIO * n.distance and m.distance < MATCH_CEILING:
                    good.append(m)

        denom = max(1, min(len(q_kps), len(r_kps)))
        ratio = len(good) / denom
        absolute = min(1.0, len(good) / 40.0)
        score = 0.5 * ratio + 0.5 * absolute

        composite = self._draw_composite(q_gray, r_gray, q_kps, r_kps, good, label)

        return {
            "label": label,
            "score": score,
            "good_matches": len(good),
            "kp_query": len(q_kps),
            "kp_reference": len(r_kps),
            "match_ratio": ratio,
            "composite_bgr": composite,
        }

    @staticmethod
    def _to_bgr(gray: np.ndarray) -> np.ndarray:
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def _draw_composite(self, q_gray, r_gray, q_kps, r_kps, good, label) -> np.ndarray:
        q_vis = cv2.drawKeypoints(
            self._to_bgr(q_gray), q_kps, None,
            color=(0, 200, 255),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
        r_vis = cv2.drawKeypoints(
            self._to_bgr(r_gray), r_kps, None,
            color=(0, 200, 255),
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
        matched = cv2.drawMatches(
            q_vis, q_kps, r_vis, r_kps, good, None,
            matchColor=(80, 240, 120),
            singlePointColor=(255, 140, 0),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        h, w = matched.shape[:2]
        header = np.full((52, w, 3), (26, 28, 34), dtype=np.uint8)
        cv2.putText(
            header,
            f"Query kp: {len(q_kps)}   Reference kp: {len(r_kps)}   "
            f"Good matches: {len(good)}   Variety: {label}",
            (14, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (240, 240, 240), 1, cv2.LINE_AA,
        )
        composite = np.vstack([header, matched])
        mid_x = composite.shape[1] // 2
        cv2.line(composite, (mid_x, 52), (mid_x, composite.shape[0]), (200, 200, 200), 2)
        return composite


ENGINE = ORBEngine()


def bgr_to_data_url(bgr: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        return ""
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


# --------------------------------------------------------------------------- #
# Flask app
# --------------------------------------------------------------------------- #
app = Flask(__name__)


@app.route("/")
def index():
    return Response(HTML_PAGE, mimetype="text/html")


@app.route("/api/bank")
def api_bank():
    bank = load_bank()
    total = sum(len(v) for v in bank.values())
    present = sum(1 for v in bank.values() if v)
    per_variety = {v: len(bank.get(v, [])) for v in VARIETIES}
    return jsonify({
        "total": total,
        "varieties_present": present,
        "per_variety": per_variety,
        "varieties": list(VARIETIES),
    })


@app.route("/api/enroll", methods=["POST"])
def api_enroll():
    variety = (request.form.get("variety") or "").strip()
    if variety not in VARIETIES:
        return jsonify({"ok": False, "error": "Choose one of the registered varieties."}), 400

    file = request.files.get("file")
    if file is None or not file.filename:
        return jsonify({"ok": False, "error": "No photo uploaded."}), 400

    suffix = Path(file.filename).suffix.lower()
    if suffix not in IMAGE_SUFFIXES:
        return jsonify({"ok": False, "error": f"Unsupported file type: {suffix}"}), 400

    stamp = time.strftime("%Y%m%d-%H%M%S")
    dest_dir = REFS_DIR / variety.replace(" ", "_")
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{stamp}_{Path(file.filename).name}"
    file.save(str(dest))

    bank = load_bank()
    bank.setdefault(variety, []).append({
        "path": str(dest),
        "created_at": time.time(),
        "name": Path(file.filename).name,
    })
    save_bank(bank)

    total = sum(len(v) for v in bank.values())
    return jsonify({
        "ok": True,
        "variety": variety,
        "total": total,
        "message": f"Saved under {variety}. Total references: {total}.",
    })


@app.route("/api/identify", methods=["POST"])
def api_identify():
    file = request.files.get("file")
    if file is None or not file.filename:
        return jsonify({"ok": False, "error": "No photo uploaded."}), 400

    bank = load_bank()
    if not any(bank.values()):
        return jsonify({"ok": False, "error": "No enrolled references yet."}), 400

    tmp_dir = DATA_DIR / "tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / f"q_{int(time.time() * 1000)}_{Path(file.filename).name}"
    file.save(str(tmp_path))

    try:
        query = cv2.imread(str(tmp_path), cv2.IMREAD_COLOR)
        if query is None:
            return jsonify({"ok": False, "error": "Could not read the image."}), 400

        best = None
        for variety, entries in bank.items():
            for entry in entries:
                ref_img = cv2.imread(entry["path"], cv2.IMREAD_COLOR)
                if ref_img is None:
                    continue
                result = ENGINE.match(query, ref_img, label=variety)
                if best is None or result["score"] > best["score"]:
                    best = result

        if best is None:
            return jsonify({"ok": False, "error": "No readable reference images."}), 400

        stamp = time.strftime("%Y%m%d-%H%M%S")
        cap_path = CAPS_DIR / f"{stamp}_{best['label'].replace(' ', '_')}.jpg"
        cv2.imwrite(str(cap_path), query)

        append_log({
            "timestamp": time.time(),
            "label": best["label"],
            "score": round(best["score"], 4),
            "good_matches": best["good_matches"],
            "query_path": str(cap_path),
        })

        return jsonify({
            "ok": True,
            "label": best["label"],
            "score": best["score"],
            "good_matches": best["good_matches"],
            "kp_query": best["kp_query"],
            "kp_reference": best["kp_reference"],
            "match_ratio": best["match_ratio"],
            "composite": bgr_to_data_url(best["composite_bgr"]),
            "query_path": str(cap_path),
        })
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass


@app.route("/api/log")
def api_log():
    rows = read_log(limit=200)
    return jsonify({"entries": rows})


@app.route("/api/reset", methods=["POST"])
def api_reset():
    scope = "all"
    if request.is_json:
        scope = (request.json or {}).get("scope", "all")
    else:
        scope = request.form.get("scope", "all")

    if scope not in ("all", "bank", "log", "caps"):
        return jsonify({"ok": False, "error": f"Unknown scope: {scope}"}), 400

    result = reset_all(scope)
    return jsonify({
        "ok": True,
        "scope": scope,
        "removed": result["removed"],
        "message": _reset_message(scope, result["removed"]),
    })


def _reset_message(scope: str, removed: dict) -> str:
    if scope == "all":
        refs = removed.get("references_files", 0)
        caps = removed.get("capture_files", 0)
        return (
            f"Factory reset complete — cleared {refs} reference file(s), "
            f"{caps} capture(s), and the log."
        )
    if scope == "bank":
        refs = removed.get("references_files", 0)
        return f"Reference bank cleared — removed {refs} file(s)."
    if scope == "log":
        return "Identification log cleared."
    if scope == "caps":
        caps = removed.get("capture_files", 0)
        return f"Captures cleared — removed {caps} file(s)."
    return "Reset complete."


# --------------------------------------------------------------------------- #
# HTML + CSS + JS (single page)
# --------------------------------------------------------------------------- #
HTML_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Agricultural Variety Identifier</title>
<style>
  :root {
    --bg: #07090d;
    --surface: rgba(22, 26, 34, 0.62);
    --surface-2: rgba(34, 39, 52, 0.85);
    --border: rgba(255, 255, 255, 0.07);
    --border-hi: rgba(255, 255, 255, 0.14);
    --primary: #4ADE80;
    --primary-dk: #22A559;
    --accent: #60A5FA;
    --danger: #f87171;
    --danger-dk: #b91c1c;
    --text: #f5f7fa;
    --muted: #9ba3af;
    --radius: 22px;
    --radius-sm: 14px;
    --shadow: 0 20px 60px -20px rgba(0, 0, 0, 0.7);
  }

  * { box-sizing: border-box; }

  html, body {
    margin: 0;
    padding: 0;
    min-height: 100vh;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 "Helvetica Neue", Arial, sans-serif;
    color: var(--text);
    background: var(--bg);
    -webkit-font-smoothing: antialiased;
    overflow-x: hidden;
  }

  body::before {
    content: "";
    position: fixed;
    inset: 0;
    z-index: -2;
    background:
      radial-gradient(900px 600px at 15% -10%, rgba(74, 222, 128, 0.14), transparent 60%),
      radial-gradient(800px 700px at 100% 20%, rgba(96, 165, 250, 0.12), transparent 60%),
      radial-gradient(700px 600px at 50% 110%, rgba(74, 222, 128, 0.10), transparent 60%),
      #07090d;
  }

  body::after {
    content: "";
    position: fixed;
    inset: 0;
    z-index: -1;
    background-image:
      linear-gradient(rgba(255,255,255,0.02) 1px, transparent 1px),
      linear-gradient(90deg, rgba(255,255,255,0.02) 1px, transparent 1px);
    background-size: 48px 48px;
    mask-image: radial-gradient(ellipse at center, black 40%, transparent 80%);
  }

  /* ---------- Header ---------- */
  header {
    position: sticky;
    top: 0;
    z-index: 50;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 16px 28px;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    background: rgba(7, 9, 13, 0.72);
    border-bottom: 1px solid var(--border);
  }

  .brand {
    display: flex;
    align-items: center;
    gap: 12px;
    font-weight: 700;
    font-size: 17px;
    letter-spacing: 0.2px;
  }

  .brand .glyph {
    width: 38px;
    height: 38px;
    display: grid;
    place-items: center;
    border-radius: 12px;
    background: linear-gradient(135deg, var(--primary), var(--primary-dk));
    box-shadow: 0 8px 24px -8px rgba(74, 222, 128, 0.6);
    font-size: 20px;
  }

  .brand .sub {
    color: var(--muted);
    font-weight: 500;
    font-size: 12px;
    margin-top: 2px;
  }

  nav {
    display: flex;
    gap: 4px;
    padding: 4px;
    border-radius: 999px;
    background: var(--surface);
    border: 1px solid var(--border);
  }

  nav button {
    appearance: none;
    border: 0;
    background: transparent;
    color: var(--muted);
    font-family: inherit;
    font-size: 13.5px;
    font-weight: 600;
    padding: 9px 18px;
    border-radius: 999px;
    cursor: pointer;
    transition: all 0.22s ease;
  }

  nav button:hover { color: var(--text); }

  nav button.active {
    background: linear-gradient(135deg, var(--primary), var(--primary-dk));
    color: #05130a;
    box-shadow: 0 6px 20px -8px rgba(74, 222, 128, 0.8);
  }

  /* ---------- Main ---------- */
  main {
    max-width: 1080px;
    margin: 0 auto;
    padding: 32px 28px 80px;
  }

  section.tab { display: none; animation: fadeUp 0.35s ease both; }
  section.tab.active { display: block; }

  @keyframes fadeUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
  }

  .hero h1 {
    font-size: clamp(28px, 4vw, 42px);
    line-height: 1.1;
    margin: 0 0 12px;
    background: linear-gradient(180deg, #ffffff 30%, #a8b1bd 100%);
    -webkit-background-clip: text;
    background-clip: text;
    color: transparent;
    letter-spacing: -0.02em;
  }

  .hero p {
    color: var(--muted);
    font-size: 16px;
    max-width: 640px;
    line-height: 1.6;
    margin: 0 0 28px;
  }

  /* ---------- Cards ---------- */
  .grid {
    display: grid;
    gap: 18px;
  }

  .grid.cols-2 { grid-template-columns: repeat(2, 1fr); }
  @media (max-width: 760px) {
    .grid.cols-2 { grid-template-columns: 1fr; }
  }

  .card {
    position: relative;
    padding: 24px;
    border-radius: var(--radius);
    background: var(--surface);
    border: 1px solid var(--border);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    box-shadow: var(--shadow);
    transition: transform 0.28s ease, border-color 0.28s ease;
  }

  .card.hoverable:hover {
    transform: translateY(-3px);
    border-color: var(--border-hi);
  }

  .card.danger {
    border-color: rgba(248, 113, 113, 0.22);
    background: linear-gradient(180deg, rgba(248, 113, 113, 0.05), rgba(22, 26, 34, 0.62));
  }

  .card.danger .icon-badge {
    background: rgba(248, 113, 113, 0.14);
    color: var(--danger);
    border: 1px solid rgba(248, 113, 113, 0.3);
  }

  .card h3 {
    margin: 0 0 8px;
    font-size: 18px;
    letter-spacing: -0.01em;
  }

  .card p {
    margin: 0 0 18px;
    color: var(--muted);
    font-size: 14px;
    line-height: 1.55;
  }

  .card .icon-badge {
    width: 44px;
    height: 44px;
    display: grid;
    place-items: center;
    border-radius: 14px;
    margin-bottom: 16px;
    font-size: 20px;
  }

  .icon-badge.green {
    background: rgba(74, 222, 128, 0.14);
    color: var(--primary);
    border: 1px solid rgba(74, 222, 128, 0.25);
  }

  .icon-badge.blue {
    background: rgba(96, 165, 250, 0.14);
    color: var(--accent);
    border: 1px solid rgba(96, 165, 250, 0.25);
  }

  /* ---------- Buttons ---------- */
  .btn {
    appearance: none;
    font-family: inherit;
    font-size: 14.5px;
    font-weight: 600;
    padding: 12px 22px;
    border-radius: 14px;
    border: 1px solid transparent;
    cursor: pointer;
    transition: transform 0.15s ease, box-shadow 0.22s ease, background 0.22s ease;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    letter-spacing: 0.1px;
  }

  .btn:active { transform: scale(0.98); }
  .btn:disabled { opacity: 0.5; cursor: not-allowed; }

  .btn.primary {
    background: linear-gradient(135deg, var(--primary), var(--primary-dk));
    color: #05130a;
    box-shadow: 0 10px 30px -12px rgba(74, 222, 128, 0.9);
  }
  .btn.primary:hover:not(:disabled) {
    box-shadow: 0 14px 36px -10px rgba(74, 222, 128, 0.95);
  }

  .btn.ghost {
    background: var(--surface-2);
    color: var(--text);
    border-color: var(--border);
  }
  .btn.ghost:hover:not(:disabled) {
    border-color: var(--border-hi);
  }

  .btn.danger {
    background: linear-gradient(135deg, var(--danger), var(--danger-dk));
    color: #ffffff;
    box-shadow: 0 10px 30px -12px rgba(248, 113, 113, 0.9);
  }
  .btn.danger:hover:not(:disabled) {
    box-shadow: 0 14px 36px -10px rgba(248, 113, 113, 0.95);
  }

  .btn.outline-danger {
    background: rgba(248, 113, 113, 0.08);
    color: var(--danger);
    border: 1px solid rgba(248, 113, 113, 0.35);
  }
  .btn.outline-danger:hover:not(:disabled) {
    background: rgba(248, 113, 113, 0.15);
    border-color: var(--danger);
  }

  .btn.full { width: 100%; }

  /* ---------- Drop zone ---------- */
  .dropzone {
    position: relative;
    border: 2px dashed rgba(255, 255, 255, 0.12);
    border-radius: var(--radius-sm);
    padding: 34px 20px;
    text-align: center;
    cursor: pointer;
    transition: all 0.25s ease;
    background: rgba(255, 255, 255, 0.015);
    overflow: hidden;
  }

  .dropzone:hover, .dropzone.drag {
    border-color: var(--primary);
    background: rgba(74, 222, 128, 0.06);
  }

  .dropzone .dz-icon {
    font-size: 30px;
    margin-bottom: 10px;
    opacity: 0.85;
  }

  .dropzone .dz-title {
    font-weight: 600;
    font-size: 14.5px;
    margin-bottom: 4px;
  }

  .dropzone .dz-sub {
    color: var(--muted);
    font-size: 12.5px;
  }

  .dropzone input[type="file"] {
    display: none;
  }

  .preview-wrap {
    margin-top: 14px;
    border-radius: var(--radius-sm);
    overflow: hidden;
    border: 1px solid var(--border);
    background: #000;
    display: none;
  }

  .preview-wrap.visible { display: block; }

  .preview-wrap img {
    display: block;
    width: 100%;
    max-height: 340px;
    object-fit: contain;
  }

  /* ---------- Variety pills ---------- */
  .pills {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
  }

  .pill {
    appearance: none;
    font-family: inherit;
    font-size: 13.5px;
    font-weight: 600;
    padding: 10px 18px;
    border-radius: 999px;
    border: 1px solid var(--border);
    background: var(--surface-2);
    color: var(--text);
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .pill:hover { border-color: var(--border-hi); }

  .pill.active {
    background: linear-gradient(135deg, var(--primary), var(--primary-dk));
    color: #05130a;
    border-color: transparent;
    box-shadow: 0 6px 20px -8px rgba(74, 222, 128, 0.8);
  }

  .pill .count {
    display: inline-block;
    margin-left: 8px;
    font-size: 11px;
    padding: 1px 7px;
    border-radius: 999px;
    background: rgba(0, 0, 0, 0.25);
    color: inherit;
  }

  .pill:not(.active) .count {
    background: rgba(255, 255, 255, 0.08);
    color: var(--muted);
  }

  /* ---------- Layout helpers ---------- */
  .row { display: flex; gap: 12px; flex-wrap: wrap; }
  .row.end { justify-content: flex-end; }
  .spacer { flex: 1; }

  .section-title {
    font-size: 13px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--muted);
    font-weight: 700;
    margin: 0 0 12px;
  }

  /* ---------- Stats ---------- */
  .stats {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 12px;
  }

  .stat {
    padding: 16px;
    border-radius: var(--radius-sm);
    background: var(--surface-2);
    border: 1px solid var(--border);
  }

  .stat .label {
    color: var(--muted);
    font-size: 11.5px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    font-weight: 700;
    margin-bottom: 6px;
  }

  .stat .value {
    font-size: 22px;
    font-weight: 700;
    letter-spacing: -0.01em;
  }

  /* ---------- Score ring ---------- */
  .score-block {
    display: flex;
    gap: 24px;
    align-items: center;
    flex-wrap: wrap;
  }

  .ring {
    position: relative;
    width: 148px;
    height: 148px;
    flex-shrink: 0;
  }

  .ring svg { transform: rotate(-90deg); }

  .ring .bg { stroke: rgba(255, 255, 255, 0.06); }

  .ring .fg {
    stroke: url(#ringGrad);
    stroke-linecap: round;
    transition: stroke-dashoffset 1s cubic-bezier(0.22, 1, 0.36, 1);
  }

  .ring .center {
    position: absolute;
    inset: 0;
    display: grid;
    place-items: center;
    flex-direction: column;
    text-align: center;
  }

  .ring .pct {
    font-size: 30px;
    font-weight: 800;
    letter-spacing: -0.02em;
    line-height: 1;
  }

  .ring .sub {
    color: var(--muted);
    font-size: 11px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 700;
    margin-top: 6px;
  }

  .score-meta h2 {
    margin: 0 0 6px;
    font-size: 24px;
    letter-spacing: -0.01em;
  }

  .score-meta .match-label {
    color: var(--primary);
    font-size: 12px;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 700;
    margin-bottom: 8px;
  }

  .score-meta p {
    color: var(--muted);
    margin: 0;
    font-size: 13.5px;
    line-height: 1.5;
  }

  /* ---------- Composite viewer ---------- */
  .composite {
    border-radius: var(--radius-sm);
    overflow: hidden;
    border: 1px solid var(--border);
    background: #000;
    cursor: zoom-in;
    position: relative;
  }

  .composite img {
    display: block;
    width: 100%;
    max-height: 420px;
    object-fit: contain;
  }

  .composite::after {
    content: "Click to zoom";
    position: absolute;
    bottom: 10px;
    right: 10px;
    font-size: 11px;
    padding: 4px 10px;
    border-radius: 999px;
    background: rgba(0,0,0,0.6);
    color: var(--muted);
    border: 1px solid var(--border);
    opacity: 0;
    transition: opacity 0.2s ease;
  }
  .composite:hover::after { opacity: 1; }

  /* ---------- Log list ---------- */
  .log-list { display: flex; flex-direction: column; gap: 10px; }

  .log-item {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 14px 18px;
    border-radius: var(--radius-sm);
    background: var(--surface);
    border: 1px solid var(--border);
    transition: border-color 0.2s ease, transform 0.2s ease;
  }

  .log-item:hover {
    border-color: var(--border-hi);
    transform: translateX(2px);
  }

  .log-item .dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: var(--primary);
    box-shadow: 0 0 12px rgba(74, 222, 128, 0.7);
    flex-shrink: 0;
  }

  .log-item .main {
    flex: 1;
    min-width: 0;
  }

  .log-item .title {
    font-weight: 600;
    font-size: 14.5px;
    margin-bottom: 2px;
  }

  .log-item .sub {
    color: var(--muted);
    font-size: 12.5px;
  }

  .log-item .right {
    text-align: right;
    flex-shrink: 0;
  }

  .log-item .score {
    font-weight: 700;
    color: var(--primary);
    font-size: 15px;
  }

  .log-item .when {
    color: var(--muted);
    font-size: 11.5px;
    margin-top: 2px;
  }

  .empty {
    text-align: center;
    padding: 60px 20px;
    color: var(--muted);
    border: 1px dashed var(--border);
    border-radius: var(--radius);
  }

  .empty .emoji { font-size: 34px; display: block; margin-bottom: 12px; opacity: 0.6; }

  /* ---------- Toast ---------- */
  #toast {
    position: fixed;
    bottom: 24px;
    left: 50%;
    transform: translateX(-50%) translateY(80px);
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 14px 22px;
    border-radius: 14px;
    background: rgba(20, 24, 32, 0.95);
    border: 1px solid var(--border-hi);
    color: var(--text);
    font-size: 14px;
    font-weight: 500;
    box-shadow: 0 20px 50px -12px rgba(0, 0, 0, 0.9);
    backdrop-filter: blur(20px);
    opacity: 0;
    pointer-events: none;
    transition: all 0.35s cubic-bezier(0.22, 1, 0.36, 1);
    z-index: 100;
    max-width: 90vw;
  }

  #toast.show {
    opacity: 1;
    transform: translateX(-50%) translateY(0);
  }

  #toast .t-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--primary);
    box-shadow: 0 0 10px var(--primary);
  }
  #toast.error .t-dot { background: var(--danger); box-shadow: 0 0 10px var(--danger); }

  /* ---------- Modal ---------- */
  .modal {
    position: fixed;
    inset: 0;
    display: none;
    align-items: center;
    justify-content: center;
    background: rgba(0, 0, 0, 0.85);
    z-index: 200;
    padding: 30px;
    backdrop-filter: blur(8px);
  }
  .modal.show { display: flex; animation: fadeUp 0.25s ease both; }
  .modal img {
    max-width: 100%;
    max-height: 100%;
    border-radius: var(--radius-sm);
    box-shadow: 0 30px 80px -20px rgba(0, 0, 0, 0.9);
  }
  .modal .close-hint {
    position: absolute;
    top: 22px;
    right: 26px;
    color: var(--muted);
    font-size: 13px;
  }

  /* ---------- Confirm dialog ---------- */
  .confirm {
    position: fixed;
    inset: 0;
    display: none;
    align-items: center;
    justify-content: center;
    background: rgba(0, 0, 0, 0.75);
    z-index: 250;
    padding: 30px;
    backdrop-filter: blur(10px);
  }
  .confirm.show { display: flex; animation: fadeUp 0.22s ease both; }

  .confirm-card {
    width: 100%;
    max-width: 440px;
    background: linear-gradient(180deg, #1a1f29, #12151b);
    border: 1px solid var(--border-hi);
    border-radius: var(--radius);
    padding: 26px;
    box-shadow: 0 30px 90px -20px rgba(0, 0, 0, 0.95);
  }

  .confirm-card .c-icon {
    width: 46px;
    height: 46px;
    display: grid;
    place-items: center;
    border-radius: 14px;
    background: rgba(248, 113, 113, 0.14);
    color: var(--danger);
    border: 1px solid rgba(248, 113, 113, 0.35);
    font-size: 22px;
    margin-bottom: 16px;
  }

  .confirm-card h3 {
    margin: 0 0 8px;
    font-size: 19px;
  }

  .confirm-card p {
    color: var(--muted);
    font-size: 14px;
    line-height: 1.55;
    margin: 0 0 22px;
  }

  .confirm-card .row { justify-content: flex-end; }

  /* ---------- Spinner ---------- */
  .spinner {
    width: 16px;
    height: 16px;
    border: 2px solid rgba(5, 19, 10, 0.3);
    border-top-color: #05130a;
    border-radius: 50%;
    animation: spin 0.7s linear infinite;
  }
  .spinner.danger {
    border: 2px solid rgba(255,255,255,0.3);
    border-top-color: #fff;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* ---------- Responsive ---------- */
  @media (max-width: 640px) {
    header { padding: 12px 16px; flex-direction: column; gap: 12px; }
    main { padding: 22px 16px 60px; }
    nav button { padding: 8px 12px; font-size: 12.5px; }
    .card { padding: 18px; }
  }
</style>
</head>
<body>

<header>
  <div class="brand">
    <div class="glyph">🌾</div>
    <div>
      <div>AgriVariety</div>
      <div class="sub">Agricultural Variety Identifier · ORB v3.2.0</div>
    </div>
  </div>
  <nav>
    <button data-tab="home" class="active">Home</button>
    <button data-tab="enroll">Enrol</button>
    <button data-tab="identify">Identify</button>
    <button data-tab="log">Log</button>
  </nav>
</header>

<main>

  <!-- ==================== HOME ==================== -->
  <section id="tab-home" class="tab active">
    <div class="hero">
      <h1>Identify any agricultural variety<br/>with computer vision.</h1>
      <p>Enrol reference photos for each variety you want to recognise — crops,
         fruits, vegetables, seeds, or anything else — then match any new photo
         using ORB keypoint matching. Everything runs locally on your machine.</p>
    </div>

    <div class="grid cols-2">
      <div class="card hoverable">
        <div class="icon-badge green">📸</div>
        <h3>Enrol reference photos</h3>
        <p>Add a labelled photo for each variety you want to recognise.
           More photos per variety means better matching.</p>
        <button class="btn primary full" onclick="switchTab('enroll')">
          Open Enrolment →
        </button>
      </div>

      <div class="card hoverable">
        <div class="icon-badge blue">🔍</div>
        <h3>Identify a variety</h3>
        <p>Upload any sample photo — we compare it side by side with the best
           match and show you the keypoint alignment.</p>
        <button class="btn primary full" onclick="switchTab('identify')">
          Start Identification →
        </button>
      </div>
    </div>

    <div class="card" style="margin-top: 22px;">
      <p class="section-title">Reference bank</p>
      <div class="stats" id="home-stats">
        <div class="stat">
          <div class="label">Enrolled photos</div>
          <div class="value" id="stat-total">—</div>
        </div>
        <div class="stat">
          <div class="label">Varieties covered</div>
          <div class="value" id="stat-covered">—</div>
        </div>
        <div class="stat">
          <div class="label">Registered varieties</div>
          <div class="value" id="stat-registered">—</div>
        </div>
      </div>
      <div style="margin-top: 18px;" class="pills" id="home-pills"></div>
    </div>

    <!-- ==================== DANGER ZONE ==================== -->
    <div class="card danger" style="margin-top: 22px;">
      <div class="icon-badge">⚠️</div>
      <h3>Danger zone — reset &amp; start anew</h3>
      <p>
        Wipe the stored data so you can start a fresh session. Choose what to
        clear — the reference bank, the identification log, captured query
        images, or all of it.
      </p>

      <div class="row" style="gap: 10px;">
        <button class="btn outline-danger" onclick="confirmReset('bank')">
          Clear reference bank
        </button>
        <button class="btn outline-danger" onclick="confirmReset('log')">
          Clear log
        </button>
        <button class="btn outline-danger" onclick="confirmReset('caps')">
          Clear captures
        </button>
        <div class="spacer"></div>
        <button class="btn danger" onclick="confirmReset('all')">
          Reset everything
        </button>
      </div>
    </div>
  </section>

  <!-- ==================== ENROLL ==================== -->
  <section id="tab-enroll" class="tab">
    <div class="hero" style="margin-bottom: 22px;">
      <h1 style="font-size: 30px;">Enrol a reference photo</h1>
      <p>Upload a clear sample photo and pick its variety. It will be added
         to the matching bank instantly.</p>
    </div>

    <div class="grid cols-2">
      <div class="card">
        <p class="section-title">1 · Choose a photo</p>
        <div class="dropzone" id="enroll-dz">
          <div class="dz-icon">📁</div>
          <div class="dz-title">Drop a photo here or click to browse</div>
          <div class="dz-sub">JPG · PNG · BMP · WEBP</div>
          <input type="file" id="enroll-file" accept="image/*" />
        </div>
        <div class="preview-wrap" id="enroll-preview-wrap">
          <img id="enroll-preview" alt="Preview" />
        </div>
      </div>

      <div class="card">
        <p class="section-title">2 · Pick the variety</p>
        <div class="pills" id="enroll-pills"></div>

        <div style="margin-top: 26px;">
          <button class="btn primary full" id="enroll-save" disabled>
            Save reference
          </button>
        </div>
      </div>
    </div>
  </section>

  <!-- ==================== IDENTIFY ==================== -->
  <section id="tab-identify" class="tab">
    <div class="hero" style="margin-bottom: 22px;">
      <h1 style="font-size: 30px;">Identify a variety</h1>
      <p>Upload a query photo. We match it against every enrolled reference
         and pick the closest variety.</p>
    </div>

    <div class="card">
      <p class="section-title">Query photo</p>
      <div class="dropzone" id="identify-dz">
        <div class="dz-icon">🌱</div>
        <div class="dz-title">Drop a sample photo here or click to browse</div>
        <div class="dz-sub">JPG · PNG · BMP · WEBP</div>
        <input type="file" id="identify-file" accept="image/*" />
      </div>
      <div class="preview-wrap" id="identify-preview-wrap">
        <img id="identify-preview" alt="Preview" />
      </div>
      <div style="margin-top: 18px;" class="row end">
        <button class="btn primary" id="identify-run" disabled>
          Run identification
        </button>
      </div>
    </div>

    <div id="identify-result" style="display: none; margin-top: 22px;">
      <div class="card">
        <p class="section-title">Result</p>
        <div class="score-block">
          <div class="ring">
            <svg width="148" height="148" viewBox="0 0 148 148">
              <defs>
                <linearGradient id="ringGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                  <stop offset="0%" stop-color="#4ADE80" />
                  <stop offset="100%" stop-color="#22A559" />
                </linearGradient>
              </defs>
              <circle class="bg" cx="74" cy="74" r="64" fill="none" stroke-width="12" />
              <circle class="fg" id="ring-fg" cx="74" cy="74" r="64" fill="none"
                      stroke-width="12" stroke-dasharray="402.1"
                      stroke-dashoffset="402.1" />
            </svg>
            <div class="center">
              <div class="pct" id="score-pct">0%</div>
              <div class="sub">Match score</div>
            </div>
          </div>
          <div class="score-meta">
            <div class="match-label">Best match</div>
            <h2 id="match-label">—</h2>
            <p id="match-desc">Comparing keypoints…</p>
          </div>
        </div>
      </div>

      <div class="card" style="margin-top: 18px;">
        <p class="section-title">Keypoint alignment — query (left) vs. best reference (right)</p>
        <div class="composite" id="composite-wrap">
          <img id="composite-img" alt="Composite" />
        </div>
      </div>

      <div class="card" style="margin-top: 18px;">
        <p class="section-title">Details</p>
        <div class="stats">
          <div class="stat">
            <div class="label">Good matches</div>
            <div class="value" id="det-good">—</div>
          </div>
          <div class="stat">
            <div class="label">Query keypoints</div>
            <div class="value" id="det-qkp">—</div>
          </div>
          <div class="stat">
            <div class="label">Reference keypoints</div>
            <div class="value" id="det-rkp">—</div>
          </div>
          <div class="stat">
            <div class="label">Match ratio</div>
            <div class="value" id="det-ratio">—</div>
          </div>
        </div>
      </div>
    </div>
  </section>

  <!-- ==================== LOG ==================== -->
  <section id="tab-log" class="tab">
    <div class="hero" style="margin-bottom: 22px;">
      <h1 style="font-size: 30px;">Identification log</h1>
      <p>Every identification you've run, newest first.</p>
    </div>

    <div class="row end" style="margin-bottom: 14px;">
      <button class="btn outline-danger" onclick="confirmReset('log')">
        Clear log
      </button>
    </div>

    <div id="log-container" class="log-list"></div>
  </section>

</main>

<div id="toast">
  <div class="t-dot"></div>
  <div id="toast-text">Ready</div>
</div>

<div class="modal" id="modal">
  <div class="close-hint">Click anywhere to close</div>
  <img id="modal-img" alt="Zoomed" />
</div>

<!-- Confirm reset dialog -->
<div class="confirm" id="confirm">
  <div class="confirm-card">
    <div class="c-icon">⚠️</div>
    <h3 id="confirm-title">Reset everything?</h3>
    <p id="confirm-body">
      This will permanently delete stored data. This cannot be undone.
    </p>
    <div class="row">
      <button class="btn ghost" id="confirm-cancel">Cancel</button>
      <button class="btn danger" id="confirm-ok">
        Yes, reset
      </button>
    </div>
  </div>
</div>

<script>
// ---------------------------------------------------------------------------
// Tab navigation
// ---------------------------------------------------------------------------
const tabs = document.querySelectorAll("nav button");
const sections = document.querySelectorAll("section.tab");

function switchTab(name) {
  tabs.forEach(b => b.classList.toggle("active", b.dataset.tab === name));
  sections.forEach(s => s.classList.toggle("active", s.id === "tab-" + name));
  if (name === "home") refreshHome();
  if (name === "enroll") refreshEnrollPills();
  if (name === "log") refreshLog();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

tabs.forEach(b => b.addEventListener("click", () => switchTab(b.dataset.tab)));

// ---------------------------------------------------------------------------
// Toast
// ---------------------------------------------------------------------------
let toastTimer = null;
function toast(msg, isError = false) {
  const el = document.getElementById("toast");
  document.getElementById("toast-text").textContent = msg;
  el.classList.toggle("error", isError);
  el.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove("show"), 2800);
}

// ---------------------------------------------------------------------------
// Modal (zoom)
// ---------------------------------------------------------------------------
const modal = document.getElementById("modal");
const modalImg = document.getElementById("modal-img");
modal.addEventListener("click", () => modal.classList.remove("show"));

// ---------------------------------------------------------------------------
// Confirm reset dialog
// ---------------------------------------------------------------------------
const confirmEl = document.getElementById("confirm");
const confirmTitle = document.getElementById("confirm-title");
const confirmBody = document.getElementById("confirm-body");
const confirmOk = document.getElementById("confirm-ok");
const confirmCancel = document.getElementById("confirm-cancel");

const RESET_COPY = {
  all: {
    title: "Reset everything?",
    body: "This will permanently delete every enrolled reference photo, " +
          "every captured query image, and the full identification log. " +
          "You'll start from a completely clean slate.",
    btn: "Yes, reset everything",
  },
  bank: {
    title: "Clear the reference bank?",
    body: "This will delete every enrolled reference photo. The log and " +
          "captured query images are kept. You'll need to re-enrol before " +
          "you can identify again.",
    btn: "Yes, clear bank",
  },
  log: {
    title: "Clear the log?",
    body: "This will delete every entry from the identification log. " +
          "Enrolled references and captures are kept.",
    btn: "Yes, clear log",
  },
  caps: {
    title: "Clear captured query images?",
    body: "This will delete every captured query image saved during " +
          "identification. Enrolled references and the log are kept.",
    btn: "Yes, clear captures",
  },
};

let pendingScope = null;

function confirmReset(scope) {
  pendingScope = scope;
  const copy = RESET_COPY[scope] || RESET_COPY.all;
  confirmTitle.textContent = copy.title;
  confirmBody.textContent = copy.body;
  confirmOk.textContent = copy.btn;
  confirmEl.classList.add("show");
}

confirmCancel.onclick = () => {
  confirmEl.classList.remove("show");
  pendingScope = null;
};

confirmOk.onclick = async () => {
  if (!pendingScope) return;
  const scope = pendingScope;
  confirmOk.disabled = true;
  confirmOk.innerHTML = '<div class="spinner danger"></div> Resetting…';

  try {
    const r = await fetch("/api/reset", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ scope }),
    });
    const d = await r.json();
    if (!d.ok) throw new Error(d.error || "Reset failed");
    toast(d.message);
    confirmEl.classList.remove("show");
    pendingScope = null;
    refreshHome();
    refreshEnrollPills();
    refreshLog();
    if (scope === "all" || scope === "bank") {
      document.getElementById("identify-result").style.display = "none";
    }
  } catch (e) {
    toast(e.message, true);
  } finally {
    confirmOk.disabled = false;
  }
};

confirmEl.addEventListener("click", e => {
  if (e.target === confirmEl) {
    confirmEl.classList.remove("show");
    pendingScope = null;
  }
});

// ---------------------------------------------------------------------------
// Home stats
// ---------------------------------------------------------------------------
async function refreshHome() {
  try {
    const r = await fetch("/api/bank");
    const d = await r.json();
    document.getElementById("stat-total").textContent = d.total;
    document.getElementById("stat-covered").textContent =
      `${d.varieties_present} / ${d.varieties.length}`;
    document.getElementById("stat-registered").textContent = d.varieties.length;

    const pills = document.getElementById("home-pills");
    pills.innerHTML = "";
    d.varieties.forEach(v => {
      const el = document.createElement("div");
      el.className = "pill" + (d.per_variety[v] ? " active" : "");
      el.style.cursor = "default";
      el.innerHTML = `${v} <span class="count">${d.per_variety[v]}</span>`;
      pills.appendChild(el);
    });
  } catch (e) {
    console.error(e);
  }
}

// ---------------------------------------------------------------------------
// Enrol
// ---------------------------------------------------------------------------
let enrollFile = null;
let enrollVariety = null;

function setupDropzone(dzId, inputId, previewId, wrapId, onSet) {
  const dz = document.getElementById(dzId);
  const input = document.getElementById(inputId);
  const preview = document.getElementById(previewId);
  const wrap = document.getElementById(wrapId);

  dz.addEventListener("click", () => input.click());

  dz.addEventListener("dragover", e => {
    e.preventDefault();
    dz.classList.add("drag");
  });
  dz.addEventListener("dragleave", () => dz.classList.remove("drag"));
  dz.addEventListener("drop", e => {
    e.preventDefault();
    dz.classList.remove("drag");
    if (e.dataTransfer.files.length) {
      handleFile(e.dataTransfer.files[0], preview, wrap, onSet);
    }
  });

  input.addEventListener("change", e => {
    if (e.target.files.length) {
      handleFile(e.target.files[0], preview, wrap, onSet);
    }
  });
}

function handleFile(file, preview, wrap, onSet) {
  if (!file.type.startsWith("image/")) {
    toast("Please choose an image file.", true);
    return;
  }
  const url = URL.createObjectURL(file);
  preview.src = url;
  wrap.classList.add("visible");
  onSet(file);
}

setupDropzone("enroll-dz", "enroll-file", "enroll-preview",
              "enroll-preview-wrap", f => {
  enrollFile = f;
  updateEnrollButton();
});

setupDropzone("identify-dz", "identify-file", "identify-preview",
              "identify-preview-wrap", f => {
  window._identifyFile = f;
  document.getElementById("identify-run").disabled = false;
});

function updateEnrollButton() {
  document.getElementById("enroll-save").disabled =
    !(enrollFile && enrollVariety);
}

async function refreshEnrollPills() {
  try {
    const r = await fetch("/api/bank");
    const d = await r.json();
    const container = document.getElementById("enroll-pills");
    container.innerHTML = "";
    d.varieties.forEach(v => {
      const b = document.createElement("button");
      b.className = "pill" + (enrollVariety === v ? " active" : "");
      b.innerHTML = `${v} <span class="count">${d.per_variety[v]}</span>`;
      b.onclick = () => {
        enrollVariety = v;
        updateEnrollButton();
        refreshEnrollPills();
      };
      container.appendChild(b);
    });
  } catch (e) {
    console.error(e);
  }
}

document.getElementById("enroll-save").onclick = async () => {
  if (!enrollFile || !enrollVariety) return;
  const btn = document.getElementById("enroll-save");
  btn.disabled = true;
  btn.innerHTML = '<div class="spinner"></div> Saving…';

  const fd = new FormData();
  fd.append("file", enrollFile);
  fd.append("variety", enrollVariety);

  try {
    const r = await fetch("/api/enroll", { method: "POST", body: fd });
    const d = await r.json();
    if (!d.ok) throw new Error(d.error || "Upload failed");
    toast(d.message);
    enrollFile = null;
    document.getElementById("enroll-preview-wrap").classList.remove("visible");
    document.getElementById("enroll-file").value = "";
    updateEnrollButton();
    refreshEnrollPills();
  } catch (e) {
    toast(e.message, true);
    btn.disabled = false;
  } finally {
    btn.innerHTML = "Save reference";
  }
};

// ---------------------------------------------------------------------------
// Identify
// ---------------------------------------------------------------------------
document.getElementById("identify-run").onclick = async () => {
  const file = window._identifyFile;
  if (!file) return;

  const btn = document.getElementById("identify-run");
  btn.disabled = true;
  btn.innerHTML = '<div class="spinner"></div> Running…';

  const fd = new FormData();
  fd.append("file", file);

  try {
    const r = await fetch("/api/identify", { method: "POST", body: fd });
    const d = await r.json();
    if (!d.ok) throw new Error(d.error || "Identification failed");

    showResult(d);
    toast(`Matched: ${d.label}`);
  } catch (e) {
    toast(e.message, true);
  } finally {
    btn.disabled = false;
    btn.innerHTML = "Run identification";
  }
};

function showResult(d) {
  const wrap = document.getElementById("identify-result");
  wrap.style.display = "block";

  const pct = Math.round(d.score * 100);
  document.getElementById("score-pct").textContent = pct + "%";

  const C = 402.1;
  const offset = C - (C * Math.min(1, d.score));
  const ring = document.getElementById("ring-fg");
  ring.style.strokeDashoffset = C;
  requestAnimationFrame(() => {
    setTimeout(() => { ring.style.strokeDashoffset = offset; }, 30);
  });

  document.getElementById("match-label").textContent = d.label;
  document.getElementById("match-desc").textContent =
    `${d.good_matches} good keypoint matches between the query and the best reference photo.`;

  document.getElementById("det-good").textContent = d.good_matches;
  document.getElementById("det-qkp").textContent = d.kp_query;
  document.getElementById("det-rkp").textContent = d.kp_reference;
  document.getElementById("det-ratio").textContent =
    (d.match_ratio * 100).toFixed(1) + "%";

  const img = document.getElementById("composite-img");
  img.src = d.composite;
  document.getElementById("composite-wrap").onclick = () => {
    modalImg.src = d.composite;
    modal.classList.add("show");
  };

  wrap.scrollIntoView({ behavior: "smooth", block: "start" });
}

// ---------------------------------------------------------------------------
// Log
// ---------------------------------------------------------------------------
async function refreshLog() {
  const container = document.getElementById("log-container");
  if (!container) return;
  container.innerHTML = "";
  try {
    const r = await fetch("/api/log");
    const d = await r.json();
    if (!d.entries || d.entries.length === 0) {
      container.innerHTML = `
        <div class="empty">
          <span class="emoji">📭</span>
          No identification events yet.<br/>
          Run an identification to see it appear here.
        </div>`;
      return;
    }
    d.entries.forEach(e => {
      const when = new Date(e.timestamp * 1000).toLocaleString();
      const pct = (e.score * 100).toFixed(1);
      const el = document.createElement("div");
      el.className = "log-item";
      el.innerHTML = `
        <div class="dot"></div>
        <div class="main">
          <div class="title">${escapeHtml(e.label)}</div>
          <div class="sub">Good matches: ${e.good_matches}</div>
        </div>
        <div class="right">
          <div class="score">${pct}%</div>
          <div class="when">${when}</div>
        </div>`;
      container.appendChild(el);
    });
  } catch (e) {
    console.error(e);
  }
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, c => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[c]));
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
refreshHome();
refreshEnrollPills();
</script>
</body>
</html>
"""


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ensure_dirs()
    print(f"\n  🌾  {APP_NAME} v{APP_VERSION}")
    print(f"      Open http://127.0.0.1:5000 in your browser\n")
    app.run(host="127.0.0.1", port=5000, debug=False)