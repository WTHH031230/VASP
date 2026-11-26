#!/usr/bin/env python3
"""segment_roles: single-pass LLM line-role segmentation and structured element extraction.

Role taxonomy:
- `figure_caption`: carries `"num"` and spans all continuation lines (one caption = one unit).
- `subfig_caption`: carries `"parent"` ((a)/(b) sub-captions belong to the figure).
- `algorithm`: pseudocode block (acts as a cropping wall).
- `figure_content`: in-figure text (plot labels, legends, teaser annotations).
Downstream cropping (`caption_cut.py`) resolves anchors and walls purely by role lookup.

The same call also extracts structured representations of Tables, Equations, and Algorithms
plus raw paper metadata while keeping the numbered-line input format fixed.

Outputs:
  <outdir>/segments2.json             Role segment list consumed by caption_cut
  <outdir>/structured_elements.json   Structured tables/equations/algorithms
  <outdir>/metadata_candidate.json    Raw paper metadata extracted in the same pass
  <outdir>/source_lines.json          Original text, page index, and bbox per source_line_id
  <outdir>/roles_viz/pageN.png        Debug overlay (only with --viz)

Reuses a complete semantic cache when present; otherwise performs a single strict-schema call
(at most one retry on empty or malformed output). Raw prompts, schemas, responses, and usage
are recorded under `attempts/semantic-NNN/`.
Usage: python segment_roles.py <paper_dir> <outdir> [--model gpt-5.6-sol --effort medium --viz]
"""
import argparse, json, re, sys
from collections import Counter
from pathlib import Path
import fitz
from PIL import Image, ImageDraw, ImageFont

_D = Path(__file__).resolve().parent
_PIPE = _D.parent.parent
for _p in (_D, _D.parent, _PIPE): sys.path.insert(0, str(_p))
import textgap
from llm.codex_exec import call_codex
from llm.claude_cli import call_claude, is_claude_model
from outer_rule_candidates import (
    extract_outer_boundary_candidates,
    normalize_page_boundaries,
    prompt_candidates,
)

Z=2.0
STRUCTURED_SCHEMA="dp-p2p.structured-elements.v1"
EXPLICIT_FIGURE_CAPTION_RE=re.compile(
    rf"^\s*(?:Figure|Fig\.?)\s+(?P<num>{textgap.FIGURE_ID_RE})\s*[:：]",
    re.IGNORECASE,
)
try: FONT=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",14)
except Exception: FONT=ImageFont.load_default()

PROMPT = """Below are the numbered text lines of an academic paper (content pages in reading order; bibliography pages are omitted, and appendix/supplement pages may follow the body). Assign EVERY line id to exactly one role. Output contiguous ranges.

Roles:
- "title_author": paper title + author/affiliation block
- "page_header": page-level chrome printed at the top, such as a repeated running
  title, author running head, publisher/conference banner, or preprint notice. This
  is outside the paper body. It is NOT the paper title/byline, a section heading,
  or a title rendered inside a Figure.
- "page_footer": page-level chrome printed at the bottom, such as a repeated
  copyright/publisher footer or running footer. Keep a bare page number as
  page_number. This is NOT a body footnote or acknowledgment.
- "heading": section/subsection/run-in heading (+ "level":"section"|"subsection"|"runin")
- "body": main paragraph prose
- "figure_caption": the caption of a NUMBERED figure — the line starting "Figure N"/"Fig. N" AND all its continuation lines (the whole caption paragraph, which may include a bold title sentence). Add "num": N as an integer for plain numeric labels, or the exact string for appendix labels such as "A.1" or "S2". If one caption's lines are interleaved with another segment, emit multiple figure_caption segments with the same "num".
- "subfig_caption": a sub-figure caption such as "(a) ...", "(b) Illustration of ...", "(c) UNIMAX" — it labels ONE panel of a figure; the figure's numbered main caption is a separate figure_caption (usually below, on the same page). Include its continuation lines. Add "parent" using the same integer-or-exact-string convention (null only if truly undeterminable).
- "table_title": the caption/title of a NUMBERED table ("Table N: ..."), including continuation lines. Add "num" using the same integer-or-exact-string convention.
- "equation_caption": the number/label of a standalone display equation (e.g. "(3)")
- "algorithm": lines of a pseudocode/algorithm listing (an "Algorithm N ..." float), INCLUDING its "Algorithm N" title line. This is NOT figure content.
- "figure_content": text rendered INSIDE a figure or table graphic: axis tick values and axis titles, legend entries, plot titles sitting directly above plot panels, node/arrow labels, words or symbols inside diagrams/teasers, table cell values. NOT captions, NOT pseudocode.
- "footnote": footnotes / acknowledgments
- "references": reference/bibliography list
- "page_number": a page number
- "other": anything else

Critical distinctions (get these right):
- A repeated line such as "Published as a conference paper at ICLR 2025" at the
  physical top of pages -> page_header, even when there is no horizontal rule.
- The full paper title and author block on page 1 -> title_author, never
  page_header merely because they are near the top.
- "Figure 5: NPC2s achieve higher log-likelihoods than ..." and its following wrapped lines -> one figure_caption segment, num=5.
- "(a) t = 1" / "(b) “a batter swings at a pitch”" -> subfig_caption with parent = the figure whose "Figure N" caption appears nearby (read the context).
- "GOODCMNIST-color", "0.0200", "concept", "Epoch 1", "hardware efficiency", a lone "?" inside a teaser -> figure_content.
- A prose paragraph that merely DISCUSSES a figure -> body, not caption.

In the SAME output, reconstruct the paper's numbered tables, display equations, and
algorithms from these lines so a poster renderer can typeset them natively.

Structured-element rules:
- Copy every string and number faithfully from the supplied lines. NEVER invent a
  missing cell, symbol, result, variable definition, or algorithm step.
- source_line_ids must contain only ids from the input and must cover the evidence
  used for that element.
- tables: include num, caption, headers, rows, footnotes, emphasis, source_line_ids.
  headers and rows are arrays of strings. For multi-level headers, use one flattened
  unambiguous string per final column. If row/column structure cannot be recovered
  faithfully, set status="unresolved", keep headers/rows empty, and list the relevant
  source_line_ids instead of guessing.
- equations: include label, latex, context, source_line_ids, status. Set latex=null
  and status="unresolved" if the extracted text is insufficient for exact LaTeX.
- algorithms: include num, title, inputs, outputs, steps, source_line_ids, status.
  Steps must preserve source order and technical symbols; do not summarize away
  control flow. Use status="unresolved" rather than completing missing text.
- Return [] for a category that is absent.

In the SAME output, extract bibliographic metadata:
- title: full printed paper title, repairing only obvious line-break hyphenation.
- authors: author display names in byline order.
- institutes: deduplicated affiliations in first-appearance order.
- author_index_map: map every author name to 1-indexed positions in institutes,
  following the numeric superscripts in the byline. For a single-institute byline
  with no superscripts, map every author to [1]. Never invent an affiliation.
- venue: real conference/journal short name plus year, only when evidenced by the
  supplied lines (for example "NeurIPS 2023"). A workshop uses its parent
  conference. Never use "arXiv", "arXiv preprint", or "Preprint" as venue.
- paper_url, code_url, project_url: only URLs explicitly present in the lines;
  otherwise "".
- arxiv_id: this paper's own YYMM.NNNNN id only when present in the paper's
  first-page text; never take an id from references.
Missing strings must be "" and missing lists must be []. Do not guess.


Output ONE JSON object, no fences:
{
  "segments": [
    {"start":0,"end":3,"role":"title_author"},
    {"start":57,"end":60,"role":"figure_caption","num":2}
  ],
  "tables": [
    {
      "num":2,
      "caption":"...",
      "headers":["Method","Accuracy"],
      "rows":[["Baseline","82.1"],["Ours","87.4"]],
      "footnotes":[],
      "emphasis":{},
      "source_line_ids":[70,71,72],
      "status":"ok"
    }
  ],
  "equations": [],
  "algorithms": [],
  "metadata": {
    "title": "...",
    "authors": ["First Last", "Second Author"],
    "author_index_map": {"First Last": [1], "Second Author": [2]},
    "institutes": ["Institute A", "Institute B"],
    "venue": "NeurIPS 2023",
    "paper_url": "",
    "code_url": "",
    "arxiv_id": "",
    "project_url": ""
  }
}
Lines:
__LINES__
"""

OUTER_BOUNDARY_PROMPT = """

In the SAME JSON object, select only page-outer content-boundary rules from the
candidate list below. These are long horizontal or vertical PDF strokes near a
page side. For every listed content page, return one record with top, bottom,
left, and right entries.

The numbered input is separated by explicit PDF-page markers. Independently of
the rule selection below, label text-only running chrome with page_header or
page_footer in segments. When a horizontal candidate and page_header/page_footer
belong to the same chrome, select the candidate here as well. Downstream geometry
will combine the exact painted text bbox and the exact rule bbox and use their
body-facing edge as one boundary; do not relabel either one as body.

A rule is a valid outer content boundary only when it separates the paper's
usable body/appendix content from page chrome or the physical page exterior:
- top: usable content begins immediately below this horizontal rule;
- bottom: usable content ends immediately above this horizontal rule;
- left: usable content begins immediately to the right of this vertical rule;
- right: usable content ends immediately to the left of this vertical rule.

Do NOT select a plot axis, Figure frame, table rule/grid, equation bar,
algorithm box, section-title underline, decorative rule inside the content, or
a line separating two internal blocks. Repeated running-header/footer rules are
good evidence. A page can legitimately have no rule on any or all sides.

Use confidence="high" only when the candidate is clearly the requested outer
boundary. Use medium/low for plausible but ambiguous candidates; downstream
geometry will conservatively ignore anything below high. Never manufacture an
id, and never reuse a candidate for another side.

Add this top-level field:
"page_outer_boundaries": [
  {
    "page": 3,
    "top": {"candidate_id": "p3_t1", "confidence": "high", "reason": "running-header separator"},
    "bottom": {"candidate_id": null, "confidence": "low", "reason": "no outer rule"},
    "left": {"candidate_id": null, "confidence": "low", "reason": "no outer rule"},
    "right": {"candidate_id": null, "confidence": "low", "reason": "no outer rule"}
  }
]

Content pages that require one record:
__CONTENT_PAGES__

Outer-boundary candidates (bbox is in PDF points; ratios are relative to page):
__OUTER_BOUNDARY_CANDIDATES__
"""

STRUCTURED_RULE_PROMPT = """

Also in the SAME JSON object, select only high-confidence LOCAL top/bottom rule
pairs that bound one numbered Table, Algorithm, or genuinely rule-delimited
Equation reconstructed in your tables/equations/algorithms output.

This is deliberately a narrow separator task, not Figure-content ownership:
- A valid pair consists of the exact outer top and bottom horizontal rules of
  one structured element. The content of that element lies between the rules.
- Prefer the outermost pair enclosing the complete element, not an internal
  table row rule, header separator, fraction bar, or algorithm step underline.
- Do NOT select plot axes, Figure borders, panel frames, legends, decorative
  rules, section underlines, page-header/footer rules, or a pair spanning two
  different elements.
- Tables and ruled algorithms are common. Ordinary display equations usually
  have no such pair; select an equation only when the supplied candidate clearly
  encloses a genuinely boxed or rule-delimited equation.
- Candidate context uses input line ids. Cross-reference the element number and
  source_line_ids in your own structured output. Never manufacture an id.
- Use confidence="high" only when both boundary rules and the element identity
  are clear. Medium/low selections are retained for audit but ignored by crop
  geometry. Return [] when no pair is reliable.

Add this top-level field:
"structured_rule_pairs": [
  {
    "element_type": "table",
    "element_num": 2,
    "page": 5,
    "pair_candidate_id": "p5_sp3",
    "confidence": "high",
    "reason": "outer booktabs rules enclosing Table 2"
  }
]

Local structured-rule pair candidates (all bboxes are exact PDF points):
__STRUCTURED_RULE_PAIR_CANDIDATES__
"""

def _even_backslash_runs(text):
    """Make LaTeX backslash runs JSON-safe without changing already-even runs."""
    return re.sub(
        r"\\+",
        lambda match: match.group(0) + ("\\" if len(match.group(0)) % 2 else ""),
        text,
    )


def _repair_json_latex(candidate):
    """Repair model-emitted LaTeX escapes, then other JSON-invalid slash runs."""
    repaired=re.sub(
        r'("latex"\s*:\s*")(.*?)("\s*,\s*"context")',
        lambda match: match.group(1)+_even_backslash_runs(match.group(2))+match.group(3),
        candidate,
        flags=re.S,
    )
    def fix_invalid_run(match):
        run=match.group(0)
        following=repaired[match.end():match.end()+1]
        valid_next=following in {'"', '\\', '/', 'b', 'f', 'n', 'r', 't', 'u'}
        return run+("\\" if len(run)%2 and not valid_next else "")
    return re.sub(r"\\+",fix_invalid_run,repaired)


def pj(raw):
    if not raw: return None
    s=re.sub(r"^```(?:json)?|```$","",raw.strip(),flags=re.M).strip(); m=re.search(r"\{.*\}",s,re.S)
    if not m: return None
    candidate=m.group(0)
    try: return json.loads(candidate)
    except json.JSONDecodeError:
        # A complete response can still be invalid JSON when LaTeX uses raw
        # backslashes (for example ``\\widetilde`` or ``\\,``).
        try: return json.loads(_repair_json_latex(candidate))
        except json.JSONDecodeError: return None


def _valid_source_ids(raw, n_lines):
    """Keep only line IDs present in the current numbered text, deduplicated in order."""
    if not isinstance(raw, list):
        return []
    out=[]; seen=set()
    for value in raw:
        if isinstance(value, bool):
            continue
        if isinstance(value, str) and value.strip().isdigit():
            value=int(value.strip())
        if isinstance(value, int) and 0 <= value < n_lines and value not in seen:
            seen.add(value); out.append(value)
    return out


def normalize_structured(obj, n_lines):
    """Preserve structured model content while normalizing top-level shape and source_line_ids."""
    out={"schema_version":STRUCTURED_SCHEMA}
    for key in ("tables","equations","algorithms"):
        items=obj.get(key) if isinstance(obj,dict) else None
        cleaned=[]
        for item in items if isinstance(items,list) else []:
            if not isinstance(item,dict):
                continue
            entry=dict(item)
            entry["source_line_ids"]=_valid_source_ids(entry.get("source_line_ids"),n_lines)
            status=entry.get("status")
            entry["status"]=status if status in ("ok","partial","unresolved") else "partial"
            cleaned.append(entry)
        out[key]=cleaned
    return out


def _figure_num(raw):
    """Return the legacy int-or-string representation used by segments2."""
    raw=str(raw).strip()
    return int(raw) if raw.isdigit() else raw


def _same_figure_num(left, right):
    """Compare model and source labels without changing persisted spelling."""
    if left is None or right is None:
        return False
    if isinstance(left,int) or str(left).strip().isdigit():
        if isinstance(right,int) or str(right).strip().isdigit():
            return int(left)==int(right)
    return str(left).strip().casefold()==str(right).strip().casefold()


def _caption_continuation_end(source_doc, anchor_pos, max_rows=24):
    """Find tightly connected continuation rows of a missed explicit caption.

    This recovery path runs only when a literal colon-form Figure anchor was
    not labelled as a caption. Adjacent rows must remain on the same page and
    in the same painted paragraph band. Overlapping math fragments are
    accepted; a real paragraph gap, new float anchor, or column jump stops it.
    """
    anchor=source_doc[anchor_pos]
    anchor_box=fitz.Rect(anchor["bbox"])
    page=anchor.get("page")
    section=anchor.get("section","body")
    top=anchor_box.y0
    bottom=anchor_box.y1
    left=anchor_box.x0
    right=anchor_box.x1
    line_height=max(1.0,anchor_box.height)
    end=int(anchor["id"])
    previous_id=end
    for item in source_doc[anchor_pos+1:anchor_pos+1+max_rows]:
        line_id=item.get("id")
        if not isinstance(line_id,int) or line_id!=previous_id+1:
            break
        if item.get("page")!=page or item.get("section","body")!=section:
            break
        text=str(item.get("text",""))
        if EXPLICIT_FIGURE_CAPTION_RE.match(text):
            break
        if re.match(r"^\s*(?:Table|Algorithm)\s+[A-Za-z0-9.]+\s*[:：]",text,re.I):
            break
        if textgap._section_heading_kind(text) is not None:
            break
        try:
            box=fitz.Rect(item["bbox"])
        except Exception:
            break
        if box.y1 < top-0.55*line_height:
            break
        if box.y0-bottom > max(4.5,0.38*line_height):
            break
        horizontal_gap=max(left-box.x1,box.x0-right,0.0)
        if horizontal_gap > max(18.0,1.5*line_height):
            break
        end=line_id
        previous_id=line_id
        bottom=max(bottom,box.y1)
        left=min(left,box.x0)
        right=max(right,box.x1)
    return end


def _replace_segment_range(segments, start, end, replacement):
    """Replace one inclusive line range while preserving unaffected metadata."""
    out=[]
    for raw in segments:
        if not isinstance(raw,dict):
            continue
        seg=dict(raw)
        seg_start=seg.get("start")
        seg_end=seg.get("end")
        if not isinstance(seg_start,int) or not isinstance(seg_end,int):
            out.append(seg)
            continue
        if seg_end < start or seg_start > end:
            out.append(seg)
            continue
        if seg_start < start:
            left=dict(seg); left["end"]=start-1; out.append(left)
        if seg_end > end:
            right=dict(seg); right["start"]=end+1; out.append(right)
    out.append(dict(replacement,start=start,end=end))
    out.sort(key=lambda item:(item.get("start",10**18),item.get("end",10**18)))
    return out


def enforce_explicit_figure_caption_invariants(segments, source_doc):
    """Recover only high-confidence colon-form Figure anchors missed by LLM."""
    corrected=[dict(item) for item in segments if isinstance(item,dict)]
    audit=[]
    for pos,item in enumerate(source_doc):
        match=EXPLICIT_FIGURE_CAPTION_RE.match(str(item.get("text","")))
        if not match:
            continue
        line_id=item.get("id")
        if not isinstance(line_id,int):
            continue
        current=None
        for seg in corrected:
            if (isinstance(seg.get("start"),int) and isinstance(seg.get("end"),int)
                    and seg["start"]<=line_id<=seg["end"]):
                current=seg
        num=_figure_num(match.group("num"))
        if (current is not None and current.get("role")=="figure_caption"
                and _same_figure_num(current.get("num"),num)):
            continue
        end=_caption_continuation_end(source_doc,pos)
        corrected=_replace_segment_range(
            corrected,line_id,end,{"role":"figure_caption","num":num})
        audit.append({
            "kind":"explicit_figure_caption_recovery",
            "start":line_id,
            "end":end,
            "num":num,
            "page":item.get("page"),
            "source_text":item.get("text",""),
            "previous_role":current.get("role") if current else None,
            "previous_num":current.get("num") if current else None,
        })
    return corrected,audit


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("paper"); ap.add_argument("out")
    ap.add_argument("--model",default="gpt-5.6-sol"); ap.add_argument("--effort",default="medium")
    ap.add_argument("--backend",default="auto"); ap.add_argument("--provider",default="openai-compat")
    ap.add_argument("--include-appendix",action="store_true",
                    help="Include appendix/supplementary pages in the semantic pass; References are always skipped")
    ap.add_argument("--timeout", type=int, default=1800, help="Per-call LLM timeout in seconds")
    ap.add_argument("--max-attempts", type=int, choices=(1, 2), default=2,
                    help="Maximum total LLM attempts (retries only on empty/malformed output, max 2)")
    ap.add_argument("--viz",action="store_true")
    ap.add_argument("--force",action="store_true",help="Ignore existing semantic cache and re-invoke the model")
    a=ap.parse_args(); pdir=Path(a.paper); od=Path(a.out); od.mkdir(parents=True,exist_ok=True)
    doc=fitz.open(pdir/"paper.pdf")
    content_lines=textgap.content_document_lines(
        doc,include_appendix=a.include_appendix)
    page_indices=[pi for pi,rows in enumerate(content_lines) if rows]
    lines=[]
    line_sections={}
    for pi in page_indices:
        for r,t,section in content_lines[pi]:
            g=len(lines); lines.append((g,pi,r,t)); line_sections[g]=section
    cache=od/"segments2.json"; structured_path=od/"structured_elements.json"
    source_path=od/"source_lines.json"
    metadata_candidate_path=od/"metadata_candidate.json"
    outer_candidates_path=od/"outer_boundary_candidates.json"
    outer_boundaries_path=od/"page_outer_boundaries.json"
    source_doc=[
        {
            "id":g,
            "page":pi+1,
            "bbox":[r.x0,r.y0,r.x1,r.y1],
            "text":t,
            **({"section":line_sections[g]} if a.include_appendix else {}),
        }
        for g,pi,r,t in lines
    ]
    old_source=None
    if source_path.exists():
        try: old_source=json.loads(source_path.read_text(encoding="utf-8"))
        except Exception: pass
    source_matches=old_source==source_doc
    outer_candidates=extract_outer_boundary_candidates(doc,source_doc)
    outer_candidates_doc={
        "schema_version":"dp-p2p.outer-boundary-candidates.v1",
        "candidates":outer_candidates,
    }
    old_outer_candidates=None
    if outer_candidates_path.exists():
        try:
            old_outer_candidates=json.loads(
                outer_candidates_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    outer_candidates_match=old_outer_candidates==outer_candidates_doc
    if (not a.force and cache.exists() and structured_path.exists()
            and metadata_candidate_path.exists() and outer_boundaries_path.exists()
            and source_matches and outer_candidates_match):
        segs=json.loads(cache.read_text())
        structured=json.loads(structured_path.read_text())
        print("Loaded semantic cache")
    else:
        clean=lambda s: re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]","",s)
        numbered_parts=[]
        for pi in page_indices:
            numbered_parts.append(f"\n=== PDF PAGE {pi + 1} ===")
            numbered_parts.extend(
                f"[{g}] {clean(t)}" for g,p,_r,t in lines if p==pi)
        numbered="\n".join(numbered_parts)
        boundary_prompt=(OUTER_BOUNDARY_PROMPT
            .replace("__CONTENT_PAGES__",json.dumps(
                [pi+1 for pi in page_indices],ensure_ascii=False))
            .replace("__OUTER_BOUNDARY_CANDIDATES__",json.dumps(
                prompt_candidates(outer_candidates),
                ensure_ascii=False,separators=(",",":"))))
        from semantic_runner import extract_semantic
        result = extract_semantic(
            source_doc, outer_candidates, [pi+1 for pi in page_indices],
            numbered, boundary_prompt, od, a, sys.modules[__name__])
        if result["state"] != "accepted":
            print(f"!! semantic failed: {result}; raw attempts are preserved in {od / 'attempts'}")
            return 1
        segs = json.loads(cache.read_text(encoding="utf-8"))
        structured = json.loads(structured_path.read_text(encoding="utf-8"))
        print(f"accepted: attempts={result['attempts']} segments={len(segs)} "
              f"tables={len(structured['tables'])} equations={len(structured['equations'])} "
              f"algorithms={len(structured['algorithms'])}")
    role={}
    for s in segs:
        for g in range(s.get("start",0),s.get("end",-1)+1):
            if 0<=g<len(lines): role[g]=(s.get("role","?"),s.get("num",s.get("parent")))
    print("Roles:",dict(Counter(r for r,_ in role.values())))
    # Quick sanity check on main figure caption numbering: anchor count vs unique numbers
    nums=[n for s in segs if s.get("role")=="figure_caption" and (n:=s.get("num")) is not None]
    subp=[s.get("parent") for s in segs if s.get("role")=="subfig_caption"]
    print(f"figure_caption segments:{sum(1 for s in segs if s.get('role')=='figure_caption')} nums:{sorted(set(nums),key=str)}  subfig parents:{sorted(set((p for p in subp if p is not None)),key=str)}")

    if not a.viz:
        print(f"Outputs: {od}/segments2.json + structured_elements.json + metadata_candidate.json + source_lines.json + page_outer_boundaries.json"); return
    (od/"roles_viz").mkdir(exist_ok=True)
    per={}
    for g,pi,r,t in lines: per.setdefault(pi,[]).append((g,r,t))
    RC={"figure_caption":(0,90,255),"subfig_caption":(0,180,220),"table_title":(150,0,200),"equation_caption":(0,170,90),"figure_content":(255,170,0)}
    for pi in page_indices:
        page=doc[pi]; pix=page.get_pixmap(matrix=fitz.Matrix(Z,Z),alpha=False)
        img=Image.frombytes("RGB",(pix.width,pix.height),pix.samples).convert("RGBA"); ov=Image.new("RGBA",img.size,(0,0,0,0)); dr=ImageDraw.Draw(ov)
        for g,r,t in per.get(pi,[]):
            if role.get(g,("",None))[0]=="body": dr.rectangle([r.x0*Z,r.y0*Z,r.x1*Z,r.y1*Z],fill=(120,120,120,45))
        img=Image.alpha_composite(img,ov).convert("RGB"); d2=ImageDraw.Draw(img)
        for g,r,t in per.get(pi,[]):
            ro,nm=role.get(g,("",None))
            if ro in RC:
                cc=RC[ro]; d2.rectangle([r.x0*Z,r.y0*Z,r.x1*Z,r.y1*Z],outline=cc,width=3)
                tag=ro.split("_")[0]+(f"{nm}" if nm is not None else "")
                d2.text((r.x0*Z,max(0,r.y0*Z-15)),tag,fill=cc,font=FONT)
        img.save(od/"roles_viz"/f"page{pi+1}.png")
    print(f"Outputs: {od}/roles_viz  (blue=figure_caption N, cyan=subfig_caption, purple=table_title N, orange=figure_content)")

if __name__=="__main__": raise SystemExit(main())
