#!/usr/bin/env python3
"""QR code generator: metadata.json links -> <outdir>/paper.png (+ code.png).

Classifies links by destination rather than field name so duplicate links across
paper_url / code_url / project_url deduplicate cleanly into at most two slots.

Classification (kind / label):
    Exact code-hosting domains (github.com, gitlab.com,
      bitbucket.org)                           -> kind=code    label="Code"
    Project page domains (e.g. *.github.io)    -> kind=project label="Project"
    Paper domains (arxiv.org, doi.org,
      openreview.net, acm, ieee, springer,
      aclanthology, mlr.press, neurips.cc, ...) -> kind=paper  label="Paper"
    Everything else                            -> kind=project label="Project"

Slots (at most 2 images):
    slot0 = paper.png  (prefers paper, falls back to project; sole link lands here)
    slot1 = code.png   (prefers code, else remaining link)
    Filenames are stable slot names; the human-readable label in the manifest follows
    the link's true destination classification.

Usage:
    python3 make_qr.py <metadata.json> <outdir>

Outputs:
    <outdir>/paper.png   slot0 (when at least one link exists)
    <outdir>/code.png    slot1 (when a second distinct link exists)
    Updates metadata.json in place with logical bundle paths ("assets/qr/<file>")
    and prints {"qr": [...]} JSON to stdout.

Exit codes:
    0 = at least one real QR code generated
    1 = no usable link, invalid URL, or rendering failure
    2 = CLI/input error (missing or invalid metadata.json)
"""
from __future__ import annotations

import argparse
import json
import sys
import unicodedata
import urllib.parse
from pathlib import Path

# Logical bundle prefix used in manifest paths for downstream poster composition.
QR_BUNDLE_PREFIX = "assets/qr"

# Destination host -> kind classification. Code requires hosting root; github.io is a project page.
_CODE_HOSTS = ("github.com", "gitlab.com", "bitbucket.org")
_PAPER_HOSTS = (
    "arxiv", "doi.org", "openreview", "acm.org", "ieee",
    "springer", "aclanthology", "semanticscholar",
    "mlr.press", "neurips.cc", "proceedings",  # PMLR / NeurIPS proceedings pages
)


def render_real_qr(url: str, out_path: Path) -> bool:
    try:
        import qrcode
    except ImportError:
        print("[make_qr] qrcode package not installed; refusing to emit unscannable placeholder", file=sys.stderr)
        return False
    try:
        qr = qrcode.QRCode(
            version=None,  # auto-select smallest version that fits the URL
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=10,
            border=2,
        )
        qr.add_data(url)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(str(out_path))
        return True
    except Exception as e:
        print(f"[make_qr] qrcode render failed ({url!r}): {e}", file=sys.stderr)
        return False


def render_qr(url: str, label: str, out_path: Path) -> bool:
    if not render_real_qr(url, out_path):
        return False
    print(f"[make_qr] {label} -> {out_path.name}  (real QR, url={url})", file=sys.stderr)
    return True


def normalize_url(raw: str) -> str:
    """Return a safe HTTP(S) URL, normalizing PDF ligatures and Unicode paths."""
    value = unicodedata.normalize("NFKC", (raw or "").strip())
    try:
        parts = urllib.parse.urlsplit(value)
    except ValueError:
        return ""
    if parts.scheme.lower() not in {"http", "https"} or not parts.hostname:
        return ""
    try:
        host = parts.hostname.encode("idna").decode("ascii")
    except UnicodeError:
        return ""
    port = f":{parts.port}" if parts.port is not None else ""
    netloc = host + port
    path = urllib.parse.quote(parts.path, safe="/%:@!$&'()*+,;=-._~")
    query = urllib.parse.quote(parts.query, safe="=&%:@!$'()*+,;/?-._~")
    return urllib.parse.urlunsplit((parts.scheme.lower(), netloc, path, query, ""))


def classify_url(url: str) -> tuple[str, str]:
    """Classify a URL by destination and return (kind, label); empty strings map to project."""
    value = (url or "").strip()
    if not value:
        return "project", "Project"
    try:
        host = (urllib.parse.urlsplit(value).hostname or "").lower()
    except ValueError:
        host = ""
    if any(host == domain or host.endswith(f".{domain}") for domain in _CODE_HOSTS):
        return "code", "Code"
    if any(h in host for h in _PAPER_HOSTS):
        return "paper", "Paper"
    return "project", "Project"


def _norm_key(url: str) -> str:
    """Deduplication key: scheme-insensitive, stripped trailing slash, lowercased."""
    u = url.strip().lower()
    for pre in ("https://", "http://"):
        if u.startswith(pre):
            u = u[len(pre):]
            break
    return u.rstrip("/")


def collect_urls(meta: dict) -> list[dict]:
    """Collect deduplicated (url, kind, label) entries from metadata.

    Synthesizes an arXiv URL when paper_url is empty and arxiv_id is present.
    Deduplicates by _norm_key, preserving the first observed raw URL string.
    """
    paper_url = normalize_url(str(meta.get("paper_url") or ""))
    if not paper_url:
        arxiv_id = str(meta.get("arxiv_id") or "").strip()
        if arxiv_id:
            paper_url = normalize_url(f"https://arxiv.org/abs/{arxiv_id}")
    code_url = normalize_url(str(meta.get("code_url") or ""))
    project_url = normalize_url(str(meta.get("project_url") or ""))

    seen: dict[str, dict] = {}
    for url in (paper_url, code_url, project_url):
        if not url:
            continue
        key = _norm_key(url)
        if key in seen:
            continue
        kind, label = classify_url(url)
        seen[key] = {"url": url, "kind": kind, "label": label, "_key": key}
    return list(seen.values())


def assign_slots(urls: list[dict]) -> list[dict]:
    """Assign deduplicated links into at most 2 slots and return manifest entries with paths.

    slot0 (paper.png) prefers paper then project (any single link lands in slot0);
    slot1 (code.png) prefers code, otherwise takes the next remaining link.
    """
    papers = [u for u in urls if u["kind"] == "paper"]
    projects = [u for u in urls if u["kind"] == "project"]

    slot0 = (papers or projects or urls or [None])[0]
    if slot0 is None:
        return []
    remaining = [u for u in urls if u["_key"] != slot0["_key"]]
    codes = [u for u in remaining if u["kind"] == "code"]
    slot1 = (codes or remaining or [None])[0]

    slot_names = ("paper", "code")
    plan = [slot0] + ([slot1] if slot1 is not None else [])
    results = []
    for i, entry in enumerate(plan):
        results.append({
            "kind": entry["kind"],
            "url": entry["url"],
            "path": f"{QR_BUNDLE_PREFIX}/{slot_names[i]}.png",
            "label": entry["label"],
            "_file": f"{slot_names[i]}.png",
        })
    return results


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("metadata", type=Path, help="Path to metadata.json (paper_url/code_url/project_url/arxiv_id)")
    ap.add_argument("outdir", type=Path,
                    help="Output directory for generated QR PNG files")
    args = ap.parse_args()

    if not args.metadata.exists():
        print(f"[make_qr] metadata.json not found: {args.metadata}", file=sys.stderr)
        return 2
    try:
        meta = json.loads(args.metadata.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[make_qr] Failed to parse metadata.json: {e}", file=sys.stderr)
        return 2
    if not isinstance(meta, dict):
        print("[make_qr] Top-level metadata.json must be a JSON object", file=sys.stderr)
        return 2

    urls = collect_urls(meta)
    if not urls:
        print("[make_qr] No usable links found; nothing to generate", file=sys.stderr)
        print(json.dumps({"qr": []}, indent=2, ensure_ascii=False))
        return 1

    plan = assign_slots(urls)
    args.outdir.mkdir(parents=True, exist_ok=True)
    results = []
    written: set[str] = set()
    for entry in plan:
        out_path = args.outdir / entry["_file"]
        if render_qr(entry["url"], entry["label"], out_path):
            written.add(entry["_file"])
            results.append({k: entry[k] for k in ("kind", "url", "path", "label")})

    # Remove any stale second-slot file left over from a previous run.
    for name in ("paper.png", "code.png"):
        stale = args.outdir / name
        if name not in written and stale.exists():
            stale.unlink()

    # Write the QR manifest back into metadata.json in place.
    if results:
        try:
            meta_now = json.loads(args.metadata.read_text(encoding="utf-8"))
            meta_now["qr"] = results
            args.metadata.write_text(
                json.dumps(meta_now, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        except Exception as e:
            print(f"[make_qr] WARNING: failed to update qr manifest ({args.metadata}): {e}", file=sys.stderr)

    print(json.dumps({"qr": results}, indent=2, ensure_ascii=False))
    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
