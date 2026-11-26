#!/usr/bin/env python3
"""Institution logo fetcher: metadata.json institutes -> <outdir>/<slug>.<ext>.

Resolution strategy (pure HTTP, zero LLM calls; marks all missing when offline):

  Primary path — Wikidata P154 (official logo image):
    1. Expand aliases and generate institution candidate queries (excluding bare geographic segments).
    2. Query wbsearchentities (top 5 items) and verify exact normalized label match.
    3. Fetch Wikidata claim P154 (logo image) and download from Wikimedia Commons.
    No weak fallback: misses are marked "missing" rather than substituting campus photos or flags.

  Flag guardrail (all paths):
    Any candidate filename or URL containing "flag" (case-insensitive) is rejected.

Usage:
    python3 fetch_logos.py <metadata.json> <outdir>

Outputs:
    <outdir>/<slug>.<png|jpg|svg|gif>   One file per resolved institution
    Updates metadata.json in place with `"logos": {"Institute A": "institute-a.png", ...}`

Exit codes:
    0 = completed (best-effort; all-missing is still exit 0)
    2 = invalid arguments or unreadable metadata.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

# Wikimedia API etiquette: provide a descriptive User-Agent header.
UA = "DP-P2P-pipeline/fetch_logos (paper2assets; https://github.com/; contact via repo)"
TIMEOUT = 15.0
REQUEST_INTERVAL = 0.5  # global rate-limit interval in seconds (see _pace)

WD_API = "https://www.wikidata.org/w/api.php"
COMMONS_FILEPATH = "https://commons.wikimedia.org/wiki/Special:FilePath/{fn}"
WIKI_SUMMARY = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"

# Offline map from common abbreviations/aliases to canonical institution names.
ALIASES = {
    "mit": "Massachusetts Institute of Technology",
    "cmu": "Carnegie Mellon University",
    "uc berkeley": "University of California, Berkeley",
    "ucb": "University of California, Berkeley",
    "ucla": "University of California, Los Angeles",
    "ucsd": "University of California, San Diego",
    "uc san diego": "University of California, San Diego",
    "uc merced": "University of California, Merced",
    "ucsb": "University of California, Santa Barbara",
    "uc santa barbara": "University of California, Santa Barbara",
    "uiuc": "University of Illinois Urbana-Champaign",
    "nyu": "New York University",
    "hkust": "Hong Kong University of Science and Technology",
    "cuhk": "Chinese University of Hong Kong",
    "pku": "Peking University",
    "thu": "Tsinghua University",
    "sjtu": "Shanghai Jiao Tong University",
    "nudt": "National University of Defense Technology",
    "ethz": "ETH Zurich",
    "eth zurich": "ETH Zurich",
    "epfl": "EPFL",
    "kaist": "KAIST",
    "msra": "Microsoft Research",
    "microsoft research asia": "Microsoft Research",
    "google research": "Google",
    "google deepmind": "Google DeepMind",
    "deepmind": "Google DeepMind",
    "meta ai": "Meta Platforms",
    "fair": "Meta Platforms",
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "nvidia": "Nvidia",
    "ibm research": "IBM",
}

# Substring keywords identifying institution segments (excludes bare geographic tokens).
INSTITUTION_TOKENS = (
    "universi",       # university / université / universität / universidad
    "institut",       # institute / institut / institution
    "college", "school", "faculty", "academy", "academia", "polytechnic",
    "laborator", "laboratory", "labs", "research", "hospital", "center", "centre",
    "corporation", "technolog", "company", " inc", " ltd", "gmbh",
    # Major industry and AI labs
    "microsoft", "google", "meta", "apple", "nvidia", "amazon", "openai",
    "anthropic", "deepmind", "ibm", "intel", "samsung", "tencent", "alibaba",
    "baidu", "huawei", "bytedance", "polixir",
)


def slugify(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", name.lower()).strip("-")
    return s or "logo"


SAFE_ENTITY_TOKENS = (
    "universi", "institut", "college", "academy", "polytechnic",
    "corporation", "technolog", "company", " inc", " ltd", "gmbh",
    "microsoft", "google", "meta", "apple", "nvidia", "amazon", "openai",
    "anthropic", "deepmind", "ibm", "intel", "samsung", "tencent", "alibaba",
    "baidu", "huawei", "bytedance", "polixir",
)


def _has_institution_token(seg: str) -> bool:
    s = seg.lower()
    return any(tok in s for tok in INSTITUTION_TOKENS)


def _safe_entity_segment(seg: str) -> bool:
    """Reject generic school/faculty/lab fragments that can match another campus."""
    s = seg.lower()
    return any(tok in s for tok in SAFE_ENTITY_TOKENS)


def _alias(text: str) -> str:
    return ALIASES.get(text.strip().lower(), text.strip())


def query_candidates(name: str) -> list[str]:
    """Generate ordered, deduplicated institution query candidates for Wikidata lookup."""
    raw = name.strip()
    out: list[str] = []
    seen: set[str] = set()

    def push(s: str) -> None:
        s = s.strip(" ,;&/-—.").strip()
        if len(s) < 2:
            return
        key = s.lower()
        if key not in seen:
            seen.add(key)
            out.append(s)

    # 1. Full-name alias
    if raw.lower() in ALIASES:
        push(ALIASES[raw.lower()])

# 2. Full original name first so campus-specific names are not over-generalized.
    push(raw)

    # 3. Keep only clear institution-level segments; drop generic School/Faculty/Lab parts.
    segs = [s.strip() for s in re.split(r"\s*[,&/;]\s*|\s+[-—]\s+", raw) if s.strip()]
    for seg in segs:
        if _safe_entity_segment(seg):
            push(_alias(seg))
            push(seg)

    # 4. Name with parenthetical remarks removed
    push(re.sub(r"\s*\([^)]*\)", "", raw).strip())

    return out


def is_flag(text: str) -> bool:
    """Flag guardrail: reject filenames or URLs containing "flag" (case-insensitive)."""
    return "flag" in (text or "").lower()


def detect_ext(data: bytes) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if data[:3] == b"\xff\xd8\xff":
        return ".jpg"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return ".gif"
    head = data[:256].lstrip()
    if head[:5] in (b"<?xml", b"<svg ") or head[:4] == b"<svg":
        return ".svg"
    return ".png"


def is_image(data: bytes | None) -> bool:
    if not data or len(data) < 64:
        return False
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return True
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return True
    if data[:3] == b"\xff\xd8\xff":
        return True
    head = data[:256].lstrip().lower()
    return head[:4] == b"<svg" or head[:5] == b"<?xml"


# ── Network layer: global rate limit + 429 exponential backoff ────────────
_last_req = [0.0]


def _pace() -> None:
    """Ensure at least REQUEST_INTERVAL seconds between HTTP requests."""
    dt = time.time() - _last_req[0]
    if dt < REQUEST_INTERVAL:
        time.sleep(REQUEST_INTERVAL - dt)
    _last_req[0] = time.time()


def _open(url: str, accept_json: bool) -> bytes | None:
    """Perform an HTTP GET with User-Agent, rate limiting, and 429 backoff (2/4/8s)."""
    headers = {"User-Agent": UA}
    if accept_json:
        headers["Accept"] = "application/json"
    for attempt in range(4):
        _pace()
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
                if resp.status != 200:
                    return None
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 3:
                time.sleep(2 ** (attempt + 1))  # 2, 4, 8s exponential backoff
                continue
            return None
        except Exception:
            return None
    return None


def fetch_json(url: str) -> dict | None:
    raw = _open(url, accept_json=True)
    if raw is None:
        return None
    try:
        return json.loads(raw.decode("utf-8", errors="replace"))
    except Exception:
        return None


def fetch_bytes(url: str) -> bytes | None:
    return _open(url, accept_json=False)


def check_network() -> bool:
    """Lightweight connectivity probe so offline runs fail fast instead of timing out per institute."""
    return fetch_bytes("https://www.wikidata.org/") is not None


# ── Wikidata primary path ────────────────────────────────────────────────
def _entity_key(text: str) -> str:
    """Normalize entity labels for strict candidate matching."""
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


def wbsearch_entities(query: str, limit: int = 5) -> list[dict]:
    """Return Wikidata candidates with labels for semantic identity checks."""
    url = WD_API + "?" + urllib.parse.urlencode({
        "action": "wbsearchentities", "search": query, "language": "en",
        "uselang": "en", "type": "item", "format": "json", "limit": str(limit),
    })
    doc = fetch_json(url)
    if not doc:
        return []
    return [
        {"id": c["id"], "label": str(c.get("label") or "")}
        for c in doc.get("search", []) if c.get("id")
    ]


def p154_files(qid: str) -> list[str]:
    """Fetch Wikimedia Commons filenames from Wikidata claim P154 (logo image)."""
    url = WD_API + "?" + urllib.parse.urlencode({
        "action": "wbgetclaims", "entity": qid, "property": "P154", "format": "json",
    })
    doc = fetch_json(url)
    if not doc:
        return []
    out: list[str] = []
    for c in doc.get("claims", {}).get("P154", []):
        fn = c.get("mainsnak", {}).get("datavalue", {}).get("value")
        if isinstance(fn, str) and fn.strip():
            out.append(fn.strip())
    return out


def commons_url(fn: str) -> str:
    return COMMONS_FILEPATH.format(fn=urllib.parse.quote(fn.replace(" ", "_")))


_GENERIC_LOGO_WORDS = {
    "the", "of", "and", "for", "university", "institute", "institution",
    "college", "academy", "school", "research", "logo", "seal", "wordmark",
}
_LOGO_ACRONYMS = {"microsoftresearch": {"msr"}}


def logo_filename_matches(label: str, filename: str) -> bool:
    """Guard against incorrect P154 claims using names and trusted acronyms."""
    label_tokens = [
        token for token in re.findall(r"[a-z0-9]+", label.lower())
        if token not in _GENERIC_LOGO_WORDS and len(token) >= 3
    ]
    file_key = _entity_key(filename)
    file_tokens = set(re.findall(r"[a-z0-9]+", filename.lower()))
    if any(token in file_tokens for token in label_tokens):
        return True
    words = [
        token for token in re.findall(r"[a-z0-9]+", label.lower())
        if token not in {"the", "of", "and", "for"}
    ]
    acronym = "".join(token[0] for token in words if token)
    if len(acronym) >= 2 and acronym in file_key:
        return True
    return any(
        alias in file_key
        for alias in _LOGO_ACRONYMS.get(_entity_key(label), set())
    )


def try_wikidata(name: str, log=lambda s: None) -> dict | None:
    """Primary lookup: candidate query -> top 5 entities -> P154 -> Commons download."""
    for q in query_candidates(name):
        query_key = _entity_key(q)
        for candidate in wbsearch_entities(q, limit=5):
            qid = candidate["id"]
            if _entity_key(candidate.get("label", "")) != query_key:
                log(
                    f"    Entity label mismatch, rejected: {q!r} -> "
                    f"{candidate.get('label')!r} ({qid})"
                )
                continue
            for fn in p154_files(qid):
                if is_flag(fn):  # flag guardrail (P154)
                    log(f"    Guardrail rejected (P154 flag): {qid} -> {fn}")
                    continue
                if not logo_filename_matches(candidate.get("label", ""), fn):
                    log(
                        f"    P154 filename does not match entity, rejected: "
                        f"{candidate.get('label')!r} ({qid}) -> {fn!r}"
                    )
                    continue
                data = fetch_bytes(commons_url(fn))
                if is_image(data):
                    return {"data": data, "source": commons_url(fn),
                            "title": fn, "via": "P154"}
    return None


# ── Wikipedia page summary fallback (retained unused by default) ─────────
def try_wikipedia_summary(name: str, log=lambda s: None) -> dict | None:
    """Fallback helper: query Wikipedia REST summary page thumbnail."""
    for q in query_candidates(name):
        api_title = urllib.parse.quote(q.replace(" ", "_"))
        doc = fetch_json(WIKI_SUMMARY.format(title=api_title))
        if not doc or doc.get("type") == "disambiguation":
            continue
        img = (doc.get("originalimage") or {}).get("source") \
            or (doc.get("thumbnail") or {}).get("source")
        if not img:
            continue
        if is_flag(img):  # flag guardrail (fallback)
            log(f"    Guardrail rejected (fallback flag): {q} -> {img}")
            continue
        data = fetch_bytes(img)
        if is_image(data):
            return {"data": data, "source": img,
                    "title": doc.get("title") or q, "via": "wikipedia_summary"}
    return None


def fetch_logo_for(name: str, log=lambda s: None) -> dict | None:
    """Fetch only Wikidata P154 marks; missing is safer than a wrong page image."""
    return try_wikidata(name, log)


def load_institutes(meta: dict) -> list[str]:
    institutes = meta.get("institutes") or []
    if isinstance(institutes, str):
        institutes = [s.strip() for s in re.split(r"[;,]", institutes) if s.strip()]
    if not isinstance(institutes, list):
        return []
    return [str(x).strip() for x in institutes if str(x).strip()]


def clean_stale_assets(outdir: Path, keep: set[str]) -> None:
    """Remove prior auto-fetch files that are no longer referenced by metadata."""
    if not outdir.is_dir():
        return
    for path in outdir.iterdir():
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg", ".gif"}:
            if path.name not in keep:
                path.unlink()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("metadata", type=Path, help="Path to metadata.json containing institutes list")
    ap.add_argument("outdir", type=Path,
                    help="Output directory for downloaded logo files")
    ap.add_argument("--add-logo", action="append", default=[], metavar="NAME=URL",
                    help="Manual override (repeatable): skip automatic lookup and download the given "
                         "NAME=URL through the same validation/guardrail pipeline into metadata.logos")
    args = ap.parse_args()

    if not args.metadata.exists():
        print(f"[fetch_logos] metadata.json not found: {args.metadata}", file=sys.stderr)
        return 2
    try:
        meta = json.loads(args.metadata.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[fetch_logos] Failed to parse metadata.json: {e}", file=sys.stderr)
        return 2
    if not isinstance(meta, dict):
        print("[fetch_logos] Top-level metadata.json must be a JSON object", file=sys.stderr)
        return 2

    institutes = load_institutes(meta)
    args.outdir.mkdir(parents=True, exist_ok=True)

    if args.add_logo:
        # Manual override mode: process only specified NAME=URL entries.
        logos = meta.get("logos") if isinstance(meta.get("logos"), dict) else {}
        patched: dict[str, str] = {}
        for spec in args.add_logo:
            name, sep, url = spec.partition("=")
            name, url = name.strip(), url.strip()
            if not sep or not name or not url.startswith(("http://", "https://")):
                print(f"[fetch_logos] Invalid --add-logo value (expected NAME=http(s)://...): {spec!r}",
                      file=sys.stderr)
                continue
            if is_flag(url):
                print(f"[fetch_logos]   x {name}  rejected by flag guardrail", file=sys.stderr)
                continue
            data = fetch_bytes(url)
            if not is_image(data):
                print(f"[fetch_logos]   x {name}  download failed or non-image: {url}", file=sys.stderr)
                continue
            out_path = args.outdir / f"{slugify(name)}{detect_ext(data)}"
            out_path.write_bytes(data)
            logos[name] = out_path.name
            patched[name] = out_path.name
            print(f"[fetch_logos]   v {name} -> {out_path.name}  (path=add-logo, source={url})",
                  file=sys.stderr)
        meta["logos"] = logos
        args.metadata.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
                                 encoding="utf-8")
        print(json.dumps({"logos": patched}, indent=2, ensure_ascii=False))
        return 0

    if not institutes:
        print("[fetch_logos] No institutes found in metadata.json; skipping", file=sys.stderr)
        meta["logos"] = {}
        clean_stale_assets(args.outdir, set())
        args.metadata.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({"logos": {}}, indent=2))
        return 0

    online = check_network()
    if not online:
        print("[fetch_logos] Network unreachable (wikidata.org); marking all logos missing", file=sys.stderr)

    result: dict[str, str] = {}
    n_ok = 0
    for name in institutes:
        def log(s: str, _n=name) -> None:
            print(f"[fetch_logos] {_n!r}\n{s}", file=sys.stderr)
        fetched = fetch_logo_for(name, log) if online else None
        if not fetched:
            result[name] = "missing"
            print(f"[fetch_logos]   x {name}  (path=missing)", file=sys.stderr)
            continue
# Derive filename slug from title after stripping any existing image extension.
        stem = re.sub(r"\.(svg|png|jpe?g|gif)$", "", fetched["title"], flags=re.IGNORECASE)
        slug = slugify(stem)
        ext = detect_ext(fetched["data"])
        out_path = args.outdir / f"{slug}{ext}"
        out_path.write_bytes(fetched["data"])
        result[name] = out_path.name
        n_ok += 1
        print(f"[fetch_logos]   v {name} -> {out_path.name}  "
              f"(path={fetched['via']}, source={fetched['source']})", file=sys.stderr)

    meta["logos"] = result
    clean_stale_assets(
        args.outdir,
        {filename for filename in result.values() if filename != "missing"},
    )
    args.metadata.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"[fetch_logos] {n_ok}/{len(institutes)} resolved; updated {args.metadata}", file=sys.stderr)
    print(json.dumps({"logos": result}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
