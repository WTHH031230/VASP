#!/usr/bin/env python3
"""Deterministic scholarly-API enrichment for metadata.json (zero LLM, best-effort).

Fills missing objective bibliographic fields (arxiv_id, venue, paper_url) by exact
normalized title match across three public APIs:
  1. arXiv API (Atom XML): arxiv_id and abs URL.
  2. Semantic Scholar Graph API: venue+year and externalIds.ArXiv.
  3. DBLP Publication Search API: authoritative CS short venue names and electronic edition URLs.

Fill-only contract:
  - Existing non-empty fields are never overwritten, except placeholder preprint venues
    matching /arxiv|preprint/i (replaced with API venue or cleared to "").
  - Writes an audit block `meta["_enrich"]` only when at least one source matches.
  - Any network/parse failure emits a stderr warning and exits 0 without blocking the pipeline.

Usage: python3 enrich_metadata.py <metadata.json> [--timeout 15]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

UA = "Mozilla/5.0 (DP-P2P pipeline enrich_metadata/1.0)"
ARXIV_API = "http://export.arxiv.org/api/query"
S2_API = "https://api.semanticscholar.org/graph/v1/paper/search"
DBLP_API = "https://dblp.org/search/publ/api"
ATOM_NS = "{http://www.w3.org/2005/Atom}"

# Offline map from verbose venue names to standard short abbreviations (longest match first).
VENUE_MAP = [
    ("neural information processing systems", "NeurIPS"),
    ("international conference on machine learning", "ICML"),
    ("international conference on learning representations", "ICLR"),
    ("annual meeting of the association for computational linguistics", "ACL"),
    ("association for computational linguistics", "ACL"),
    ("empirical methods in natural language processing", "EMNLP"),
    ("computer vision and pattern recognition", "CVPR"),
    ("international conference on computer vision", "ICCV"),
    ("european conference on computer vision", "ECCV"),
    ("aaai conference on artificial intelligence", "AAAI"),
    ("international joint conference on artificial intelligence", "IJCAI"),
    ("knowledge discovery and data mining", "KDD"),
    ("international conference on artificial intelligence and statistics", "AISTATS"),
    ("conference on robot learning", "CoRL"),
    ("conference on uncertainty in artificial intelligence", "UAI"),
]

# Placeholder preprint venue patterns eligible for replacement by a real API venue.
FORBIDDEN_VENUE_RE = re.compile(r"arxiv|preprint", re.IGNORECASE)


def _warn(msg: str) -> None:
    """Log a best-effort warning without raising or changing the exit code."""
    print(f"[enrich_metadata] WARNING: {msg}", file=sys.stderr)


def norm_title(s: str) -> str:
    """Normalize title (lowercase alphanumeric only) for strict exact-title matching."""
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def canon_venue(raw_venue: str, year) -> str | None:
    """Canonicalize a venue (+ optional year) to "<ShortName> YYYY"; return None for arXiv/empty."""
    v = (raw_venue or "").strip()
    if not v or v.lower() in ("arxiv.org", "arxiv"):
        return None
    low = v.lower()
    short = None
    for pat, name in VENUE_MAP:
        if pat in low:
            short = name
            break
    label = short if short else v  # keep original venue string when no mapping matches
    ys = str(year).strip() if year not in (None, "") else ""
    if ys and ys not in label:
        label = f"{label} {ys}"
    return label


def http_get(url: str, timeout: float, accept: str) -> bytes | None:
    """HTTP GET helper; logs a warning and returns None on any error/timeout."""
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": accept})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                _warn(f"HTTP {resp.status}: {url}")
                return None
            return resp.read()
    except Exception as e:  # gracefully degrade on HTTP/network/timeout errors
        _warn(f"Request failed ({e!r}): {url}")
        return None


def _extract_arxiv_id(id_url: str) -> str | None:
    """Extract versionless arxiv_id from an Atom <id> URL (e.g. .../2201.13419v1)."""
    m = re.search(r"arxiv\.org/abs/(.+)$", id_url or "")
    if not m:
        return None
    return re.sub(r"v\d+$", "", m.group(1).strip())


def query_arxiv(title: str, timeout: float) -> dict:
    """Source 1: arXiv API. Returns {"matched": bool, "arxiv_id": str | None}."""
    out = {"matched": False, "arxiv_id": None}
    q = f'ti:"{title}"'
    url = ARXIV_API + "?" + urllib.parse.urlencode(
        {"search_query": q, "start": 0, "max_results": 5})
    raw = http_get(url, timeout, "application/atom+xml")
    if raw is None:
        return out
    try:
        root = ET.fromstring(raw)
    except Exception as e:
        _warn(f"Failed to parse arXiv Atom XML: {e!r}")
        return out
    want = norm_title(title)
    for entry in root.findall(ATOM_NS + "entry"):
        t_el = entry.find(ATOM_NS + "title")
        cand_title = (t_el.text or "") if t_el is not None else ""
        if norm_title(cand_title) != want:
            continue  # require exact normalized title match
        id_el = entry.find(ATOM_NS + "id")
        aid = _extract_arxiv_id(id_el.text if id_el is not None else "")
        out["matched"] = True
        out["arxiv_id"] = aid
        return out
    return out


def query_s2(title: str, timeout: float) -> dict:
    """Source 2: Semantic Scholar API. Returns {"matched", "venue", "arxiv_id"}."""
    out = {"matched": False, "venue": None, "arxiv_id": None}
    url = S2_API + "?" + urllib.parse.urlencode(
        {"query": title, "fields": "title,venue,year,externalIds", "limit": 5})
    raw = http_get(url, timeout, "application/json")
    if raw is None:
        return out
    try:
        doc = json.loads(raw.decode("utf-8", errors="replace"))
    except Exception as e:
        _warn(f"Failed to parse Semantic Scholar JSON: {e!r}")
        return out
# Prefer exact-title matches that carry a canonical conference/journal venue over arXiv records.
    want = norm_title(title)
    hits = [item for item in (doc.get("data") or [])
            if isinstance(item, dict) and norm_title(item.get("title", "")) == want]
    if not hits:
        return out
    out["matched"] = True
    best = next((h for h in hits
                 if canon_venue(h.get("venue", ""), h.get("year")) is not None), hits[0])
    out["venue"] = canon_venue(best.get("venue", ""), best.get("year"))
    for h in (best, *hits):  # take arxiv_id from best match first, then other hits
        ext = h.get("externalIds") or {}
        aid = ext.get("ArXiv") if isinstance(ext, dict) else None
        if aid and str(aid).strip():
            out["arxiv_id"] = str(aid).strip()
            break
    return out


def _norm_author(s: str) -> str:
    """Strip DBLP author disambiguation digits (e.g. "0001") before normalizing."""
    return re.sub(r"\s+\d+$", "", str(s)).strip().lower()


def _dblp_hits(query: str, timeout: float) -> list[dict]:
    url = DBLP_API + "?" + urllib.parse.urlencode({"q": query, "format": "json", "h": 15})
    raw = http_get(url, timeout, "application/json")
    if raw is None:
        return []
    try:
        doc = json.loads(raw.decode("utf-8", errors="replace"))
        return ((doc.get("result") or {}).get("hits") or {}).get("hit") or []
    except Exception as e:
        _warn(f"Failed to parse DBLP JSON: {e!r}")
        return []


def _dblp_venue_of(info: dict) -> str | None:
    v = info.get("venue", "")
    if isinstance(v, list):  # DBLP occasionally returns a list
        v = v[0] if v else ""
    if str(v).strip().lower() == "corr":  # CoRR is the arXiv mirror, not a published venue
        return None
    return canon_venue(str(v), info.get("year"))


def _dblp_authors_of(info: dict) -> set[str]:
    a = (info.get("authors") or {}).get("author") or []
    if isinstance(a, dict):
        a = [a]
    return {_norm_author(x.get("text", "")) for x in a if isinstance(x, dict) and x.get("text")}


def _dblp_ee_of(info: dict) -> str | None:
    """Extract the first http(s) electronic edition (ee) URL from a DBLP hit."""
    ee = info.get("ee", "")
    cands = ee if isinstance(ee, list) else [ee]
    for c in cands:
        c = str(c).strip()
        if c.startswith(("http://", "https://")):
            return c
    return None


def query_dblp(title: str, timeout: float, authors: list[str] | None = None) -> dict:
    """Source 3: DBLP API.

    Two-round matching:
      Round 1: Exact normalized title match.
      Round 2 (fallback if Round 1 yields no venue and title has a "System:" prefix):
        Match same normalized colon prefix plus >= 50% author overlap to handle camera-ready
        retitling without colliding on unrelated papers sharing a short name.
    """
    out = {"matched": False, "venue": None, "ee": None}
    want = norm_title(title)
    for h in _dblp_hits(title, timeout):
        info = (h or {}).get("info") or {}
        if norm_title(str(info.get("title", ""))) != want:
            continue
        out["matched"] = True
        out["ee"] = out["ee"] or _dblp_ee_of(info)
        cv = _dblp_venue_of(info)
        if cv:  # prefer published conference record over arXiv mirror
            out["venue"] = cv
            out["ee"] = _dblp_ee_of(info) or out["ee"]
            return out

    prefix = title.split(":", 1)[0].strip() if ":" in title else ""
    ours = {_norm_author(a) for a in (authors or []) if str(a).strip()}
    if out["venue"] is None and len(norm_title(prefix)) >= 4 and ours:
        need = max(2, (len(ours) + 1) // 2)
        for h in _dblp_hits(prefix, timeout):
            info = (h or {}).get("info") or {}
            cand_title = str(info.get("title", ""))
            if ":" not in cand_title or norm_title(cand_title.split(":", 1)[0]) != norm_title(prefix):
                continue
            if len(ours & _dblp_authors_of(info)) < need:
                continue
            cv = _dblp_venue_of(info)
            if cv:
                _warn(f"DBLP prefix+author match accepted retitled record: {cand_title!r} -> {cv}")
                out["matched"] = True
                out["venue"] = cv
                out["ee"] = _dblp_ee_of(info) or out["ee"]
                return out
    return out


def enrich(meta: dict, timeout: float) -> tuple[dict, bool]:
    """Enrich metadata dict in place; return (audit, changed). Leaves file untouched if no source matches."""
    title = (meta.get("title") or "").strip()
    audit = {"arxiv_matched": False, "s2_matched": False, "dblp_matched": False,
             "filled": [], "replaced_venue": None}
    if not title:
        _warn("metadata.title is empty; skipping enrichment")
        return audit, False

    ax = query_arxiv(title, timeout)
    s2 = query_s2(title, timeout)
    db = query_dblp(title, timeout, authors=meta.get("authors") or [])
    audit["arxiv_matched"] = ax["matched"]
    audit["s2_matched"] = s2["matched"]
    audit["dblp_matched"] = db["matched"]
    if not (ax["matched"] or s2["matched"] or db["matched"]):
        return audit, False  # no source matched; leave file unchanged

    # arxiv_id: prefer arXiv API, fall back to Semantic Scholar externalIds.ArXiv (fill-only).
    arxiv_id = ax["arxiv_id"] or s2["arxiv_id"]
    if arxiv_id and not (meta.get("arxiv_id") or "").strip():
        meta["arxiv_id"] = arxiv_id
        audit["filled"].append("arxiv_id")

# venue: fill-only (except replacing preprint placeholders); prefer S2, fall back to DBLP.
    cur_venue = (meta.get("venue") or "").strip()
    api_venue = s2["venue"] or db["venue"]  # canonicalized or None
    if not cur_venue:
        if api_venue:
            meta["venue"] = api_venue
            audit["filled"].append("venue")
    elif FORBIDDEN_VENUE_RE.search(cur_venue):
        # Placeholder preprint value: replace with API venue or clear to empty string
        audit["replaced_venue"] = cur_venue
        meta["venue"] = api_venue if api_venue else ""

# paper_url: fill-only; prefer arXiv abs URL, fall back to DBLP electronic edition URL.
    have_arxiv_id = (meta.get("arxiv_id") or "").strip()
    if not (meta.get("paper_url") or "").strip():
        if have_arxiv_id:
            meta["paper_url"] = f"https://arxiv.org/abs/{have_arxiv_id}"
            audit["filled"].append("paper_url")
        elif db.get("ee"):
            meta["paper_url"] = db["ee"]
            audit["filled"].append("paper_url")

    return audit, True


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("metadata", type=Path, help="Path to metadata.json (enriched in place)")
    ap.add_argument("--timeout", type=float, default=15.0, help="Per-request HTTP timeout in seconds")
    args = ap.parse_args()

    if not args.metadata.exists():
        _warn(f"metadata.json not found: {args.metadata}")
        return 0  # best-effort: do not block pipeline
    try:
        meta = json.loads(args.metadata.read_text(encoding="utf-8"))
    except Exception as e:
        _warn(f"Failed to parse metadata.json; leaving unchanged: {e!r}")
        return 0
    if not isinstance(meta, dict):
        _warn("Top-level metadata.json is not a JSON object; skipping")
        return 0

    audit, changed = enrich(meta, args.timeout)
    if not changed:
        print(f"[enrich_metadata] No source matched; file unchanged: {args.metadata}")
        return 0

    meta["_enrich"] = audit
    args.metadata.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[enrich_metadata] arxiv={audit['arxiv_matched']} s2={audit['s2_matched']} "
          f"dblp={audit.get('dblp_matched')} "
          f"filled={audit['filled'] or 'none'} "
          f"replaced_venue={audit['replaced_venue'] or 'none'} -> {args.metadata}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
