#!/usr/bin/env python3
"""caption_cut: caption-anchored deterministic figure cropper.

Stage 2 of SemParser (consuming Stage 1 line roles from segment_roles.py):
- Anchors: figure_caption segments grouped by num; subfig_caption/figure_content belong to the figure.
- Walls: determined purely by role lookup (body/heading/caption/algorithm/...).
- Region claiming with connected-component ownership fallback on collapse.
- Page-level invariant arbitration (R3) between claim and ownership candidates.
- Cross-page orphan adoption for floating figures whose caption appears at the top of the next page.
- Self-calibrated thresholds via median body line height `med_lh`; repeated headers/footers excluded.

Outputs: <outdir>/{figures/,appendix_figures/,overlay/,report.json}
Usage: python caption_cut.py <paper_dir> <outdir> [--seg <segments2.json>]
"""
import argparse, json, math, re, sys, shutil, statistics, os
from collections import Counter, deque
from pathlib import Path
import fitz
from PIL import Image, ImageDraw, ImageFont

_D = Path(__file__).resolve().parent
for _p in (_D, _D.parent, _D.parent.parent): sys.path.insert(0, str(_p))
import textgap
from family_interface import family_maps, family_name, split_label, caption_identity, namespaced_num
from deterministic_table_bands import (
    derive_table_bands,
    exclude_table_owned_ink,
    index_active_table_bands,
)

Z=2.0
try: FONT=ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",14)
except Exception: FONT=ImageFont.load_default()
KMAP={"figure_caption":"figure","table_title":"table","equation_caption":"equation"}
RZ=4.0; NEAR_WHITE=247

WALL_ROLES = {"heading","title_author","body","footnote","page_number","page_header",
              "page_footer","references","references_start","figure_caption","table_title",
              "equation_caption","algorithm"}
FIG_PART   = {"figure_content","subfig_caption"}          # part of a figure: neither wall nor boundary

def _peaks(c,tol=6,mf=0.08):
    tot=sum(c.values()) or 1; u=dict(c); cl=[]
    while u:
        cx=max(u,key=lambda k:u[k]); mem=[k for k in list(u) if abs(k-cx)<=tol]; cl.append((cx,sum(u[k] for k in mem)))
        for k in mem: del u[k]
    cl=[x for x in cl if x[1]>=mf*tot]; cl.sort(key=lambda t:-t[1]); return cl

def role_maps(segs, line_ids):
    """Map semantic ranges onto their stable ``source_lines.json`` IDs."""
    valid = set(line_ids)
    role, num = {}, {}
    for segment in segs:
        for line_id in range(segment.get("start", 0), segment.get("end", -1) + 1):
            if line_id not in valid:
                continue
            role[line_id] = segment.get("role", "?")
            value = segment.get("num", segment.get("parent"))
            if value is not None:
                num[line_id] = value
    return role, num


def painted_text_envelopes(page):
    """Return (logical, painted) bboxes for PDF text drawing operations.

    ``get_texttrace`` provides the logical text operation and its ``seqno``;
    ``get_bboxlog`` provides the ink actually painted by that operation.
    Invalid or unrelated drawing records are ignored.
    """
    try:
        bboxlog = page.get_bboxlog()
        traces = page.get_texttrace()
    except Exception:
        return []
    envelopes = []
    for trace in traces:
        seqno = trace.get("seqno")
        if not isinstance(seqno, int) or not (0 <= seqno < len(bboxlog)):
            continue
        kind, bbox = bboxlog[seqno]
        if kind not in {"fill-text", "stroke-text"}:
            continue
        logical = fitz.Rect(trace.get("bbox", ()))
        painted = fitz.Rect(bbox) & page.rect
        values = tuple(logical) + tuple(painted)
        if (logical.is_empty or painted.is_empty
                or not all(math.isfinite(value) for value in values)):
            continue
        if (logical & painted).is_empty:
            continue
        envelopes.append((logical, painted))
    return envelopes


def expand_wall_to_painted_ink(rect, envelopes):
    """Expand a logical text wall only by verified PDF glyph overflow.

    Edge matching associates a source line with the text operation that
    produced it. It is not padding: only outward differences present in
    ``get_bboxlog`` are applied, and interior lines of multi-line drawing
    operations remain unchanged.
    """
    wall = fitz.Rect(rect)
    for logical, painted in envelopes:
        overlap_x = max(0.0, min(wall.x1, logical.x1) - max(wall.x0, logical.x0))
        overlap_y = max(0.0, min(wall.y1, logical.y1) - max(wall.y0, logical.y0))
        if overlap_x <= 0 or overlap_y <= 0:
            continue
        edge_tol = max(0.25, 0.05 * min(wall.height, logical.height))
        max_overflow = max(wall.height, logical.height)
        if overlap_y >= 0.5 * min(wall.height, logical.height):
            if abs(wall.x0 - logical.x0) <= edge_tol:
                overflow = logical.x0 - painted.x0
                if 0 < overflow <= max_overflow:
                    wall.x0 = painted.x0
            if abs(wall.x1 - logical.x1) <= edge_tol:
                overflow = painted.x1 - logical.x1
                if 0 < overflow <= max_overflow:
                    wall.x1 = painted.x1
        if overlap_x >= 0.5 * min(wall.width, logical.width):
            if abs(wall.y0 - logical.y0) <= edge_tol:
                overflow = logical.y0 - painted.y0
                if 0 < overflow <= max_overflow:
                    wall.y0 = painted.y0
            if abs(wall.y1 - logical.y1) <= edge_tol:
                overflow = painted.y1 - logical.y1
                if 0 < overflow <= max_overflow:
                    wall.y1 = painted.y1
    return wall


def make_text_wall(logical, painted=None, page_index=None, line_id=None,
                   role=None, text=""):
    """Keep semantic ownership geometry separate from painted geometry."""
    logical = fitz.Rect(logical)
    painted = fitz.Rect(painted) if painted is not None else fitz.Rect(logical)
    return {
        "logical": logical,
        "painted": painted,
        "page_index": page_index,
        "line_id": line_id,
        "role": role,
        "text": text,
    }


def wall_logical(wall):
    """Return the semantic bbox used only for relation / ownership tests."""
    if isinstance(wall, dict):
        return fitz.Rect(wall.get("logical", wall.get("painted", ())))
    return fitz.Rect(wall)


def wall_painted(wall):
    """Return the best non-raster painted bbox for a wall."""
    if isinstance(wall, dict):
        return fitz.Rect(wall.get("painted", wall.get("logical", ())))
    return fitz.Rect(wall)


def _rect_edge(rect, side):
    return {
        "top": rect.y0,
        "bottom": rect.y1,
        "left": rect.x0,
        "right": rect.x1,
    }[side]


def rendered_text_edge(logical, painted, side, gray, median_line_height):
    """Measure one text-wall edge from the already cached 4x page raster.

    The logical bbox identifies *which* semantic text owns the wall.  Raster
    components may move only the requested outward edge, within half a body
    line height.  Components that run out of that window are ignored, so an
    adjacent Figure cannot pull the wall through a white gap.  No padding is
    added: the returned coordinate is an observed pixel-cell edge.
    """
    if side not in {"top", "bottom", "left", "right"}:
        raise ValueError(f"invalid wall side: {side}")
    logical = fitz.Rect(logical)
    painted = fitz.Rect(painted)
    d, width, height = gray
    if logical.is_empty or width <= 0 or height <= 0:
        return _rect_edge(logical, side), {
            "status": "empty_logical", "accepted_components": 0,
            "rejected_open_components": 0,
        }

    outward = max(1.0 / RZ, 0.5 * float(median_line_height))
    inward = min(outward, max(1.0, 0.25 * float(median_line_height)))
    ortho = max(0.5, 0.10 * float(median_line_height))
    if side == "right":
        window = fitz.Rect(logical.x1 - inward, logical.y0 - ortho,
                           logical.x1 + outward, logical.y1 + ortho)
    elif side == "left":
        window = fitz.Rect(logical.x0 - outward, logical.y0 - ortho,
                           logical.x0 + inward, logical.y1 + ortho)
    elif side == "top":
        window = fitz.Rect(logical.x0 - ortho, logical.y0 - outward,
                           logical.x1 + ortho, logical.y0 + inward)
    else:
        window = fitz.Rect(logical.x0 - ortho, logical.y1 - inward,
                           logical.x1 + ortho, logical.y1 + outward)

    x0 = max(0, math.floor(window.x0 * RZ))
    y0 = max(0, math.floor(window.y0 * RZ))
    x1 = min(width, math.ceil(window.x1 * RZ))
    y1 = min(height, math.ceil(window.y1 * RZ))
    if x1 <= x0 or y1 <= y0:
        return _rect_edge(logical, side), {
            "status": "empty_window", "accepted_components": 0,
            "rejected_open_components": 0,
        }

    ink = set()
    for py in range(y0, y1):
        offset = py * width
        for px in range(x0, x1):
            if d[offset + px] < NEAR_WHITE:
                ink.add((px, py))

    components = []
    while ink:
        start = ink.pop()
        queue = deque([start])
        min_x = max_x = start[0]
        min_y = max_y = start[1]
        while queue:
            px, py = queue.popleft()
            min_x = min(min_x, px); max_x = max(max_x, px)
            min_y = min(min_y, py); max_y = max(max_y, py)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    neighbour = (px + dx, py + dy)
                    if neighbour in ink:
                        ink.remove(neighbour)
                        queue.append(neighbour)
        components.append((min_x, min_y, max_x, max_y))

    logical_edge = _rect_edge(logical, side)
    painted_edge = _rect_edge(painted, side)
    edge_px = logical_edge * RZ
    painted_px = painted_edge * RZ
    accepted = []
    rejected_open = 0
    for min_x, min_y, max_x, max_y in components:
        if side == "right":
            crosses = min_x < edge_px and max_x + 1 > edge_px
            painted_seed = (0 < painted_edge-logical_edge <= outward
                            and max_x+1 > edge_px
                            and max_x+1 <= painted_px+1.5)
            open_outward = max_x >= x1 - 1
        elif side == "left":
            crosses = min_x < edge_px and max_x + 1 > edge_px
            painted_seed = (0 < logical_edge-painted_edge <= outward
                            and min_x < edge_px
                            and min_x >= painted_px-1.5)
            open_outward = min_x <= x0
        elif side == "top":
            crosses = min_y < edge_px and max_y + 1 > edge_px
            painted_seed = (0 < logical_edge-painted_edge <= outward
                            and min_y < edge_px
                            and min_y >= painted_px-1.5)
            open_outward = min_y <= y0
        else:
            crosses = min_y < edge_px and max_y + 1 > edge_px
            painted_seed = (0 < painted_edge-logical_edge <= outward
                            and max_y+1 > edge_px
                            and max_y+1 <= painted_px+1.5)
            open_outward = max_y >= y1 - 1
        if not (crosses or painted_seed):
            continue
        if open_outward:
            rejected_open += 1
            continue
        accepted.append((min_x, min_y, max_x, max_y))

    if side == "right":
        observed = max(((component[2] + 1) / RZ for component in accepted),
                       default=logical_edge)
        coordinate = max(logical_edge, observed)
    elif side == "left":
        observed = min((component[0] / RZ for component in accepted),
                       default=logical_edge)
        coordinate = min(logical_edge, observed)
    elif side == "top":
        observed = min((component[1] / RZ for component in accepted),
                       default=logical_edge)
        coordinate = min(logical_edge, observed)
    else:
        observed = max(((component[3] + 1) / RZ for component in accepted),
                       default=logical_edge)
        coordinate = max(logical_edge, observed)
    return coordinate, {
        "status": "measured" if accepted else "logical_fallback",
        "accepted_components": len(accepted),
        "rejected_open_components": rejected_open,
        "logical_edge": round(logical_edge, 4),
        "painted_edge": round(painted_edge, 4),
        "rendered_edge": round(coordinate, 4),
        "search_window": [round(value, 4) for value in window],
    }


def wall_edge(wall, side, resolver=None):
    """Resolve an edge while keeping raw PDF-rule walls backward compatible."""
    if resolver is not None and isinstance(wall, dict):
        return resolver(wall, side)
    return _rect_edge(wall_painted(wall), side)


def acols(doc,page_indices,page_rows=None):
    a,b=Counter(),Counter()
    for pi in page_indices:
        rows=page_rows.get(pi,[]) if page_rows is not None else textgap.page_lines(doc[pi])
        for r,t in rows:
            if r.width>=30: a[round(r.x0)]+=1; b[round(r.x1)]+=1
    L=_peaks(a);R=_peaks(b);y0,y1=doc[0].rect.y0,doc[0].rect.y1
    n=max(1,min(len(L),len(R),2));Ls=sorted(x for x,_ in L[:n]);Rs=sorted(x for x,_ in R[:n])
    return [fitz.Rect(l,y0,rr,y1) for l,rr in zip(Ls,Rs)] if len(Ls)==len(Rs) and Ls else None

def _capnum(s):
    m=re.search(r'(?:Figure|Fig|Table|Tab)\.?\s*((?:[A-Z]\.?)?\d+(?:\.\d+)*)',s,re.I)
    return m.group(1) if m else None

def dens(page,rect):
    r=fitz.Rect(rect)&page.rect
    if r.width<2 or r.height<2: return 0.0
    pix=page.get_pixmap(matrix=fitz.Matrix(1.5,1.5),clip=r,alpha=False)
    d=Image.frombytes("RGB",(pix.width,pix.height),pix.samples).convert("L").getdata()
    return sum(1 for v in d if v<245)/max(1,len(d))

def has_ink(page,rect,thr=0.004): return dens(page,rect)>=thr

def ink_gap(page,cbox,direction,Ytop,Ybot,thr=0.03):
    """Distance from a caption to the nearest ink in the given direction."""
    for dd in range(2,500,3):
        if direction=="up":
            y=cbox.y0-dd
            if y-3<=Ytop: return 9999
            band=fitz.Rect(cbox.x0,y-3,cbox.x1,y)
        else:
            y=cbox.y1+dd
            if y+3>=Ybot: return 9999
            band=fitz.Rect(cbox.x0,y,cbox.x1,y+3)
        if dens(page,band)>thr: return dd
    return 9999

def _proj(page,rect,axis,z=2.0,thr=245):
    r=fitz.Rect(rect)&page.rect
    if r.width<3 or r.height<3: return [],0,(r.y0 if axis=="y" else r.x0),z
    pix=page.get_pixmap(matrix=fitz.Matrix(z,z),clip=r,alpha=False)
    img=Image.frombytes("RGB",(pix.width,pix.height),pix.samples).convert("L"); w,h=img.size; d=list(img.getdata())
    if axis=="y": return [sum(1 for c in range(w) if d[row*w+c]<thr) for row in range(h)], w, r.y0, z
    return [sum(1 for row in range(h) if d[row*w+c]<thr) for c in range(w)], h, r.x0, z

def find_hsplit(page,rect):
    counts,w,base,z=_proj(page,rect,"y"); n=len(counts)
    if n<10: return None
    tol=max(1,int(0.006*w)); lo,hi=int(0.10*n),int(0.90*n); best=None; run=None
    for i in range(lo,hi):
        if counts[i]<=tol: run=[i,i] if run is None else [run[0],i]
        else:
            if run and run[1]-run[0]>=int(4*z) and (not best or run[1]-run[0]>best[1]-best[0]): best=run
            run=None
    if run and run[1]-run[0]>=int(4*z) and (not best or run[1]-run[0]>best[1]-best[0]): best=run
    return base+((best[0]+best[1])/2)/z if best else None

def find_vsplit(page,y0,y1,lo_pt,hi_pt):
    if hi_pt-lo_pt<3 or y1-y0<3: return None
    counts,h,base,z=_proj(page,fitz.Rect(lo_pt,y0,hi_pt,y1),"x")
    if not counts: return None
    return base+min(range(len(counts)),key=lambda k:counts[k])/z

def content_margins(doc, hb, page_indices, page_rows=None):
    pages=list(page_indices)
    bot={}
    for pi in pages:
        ph=doc[pi].rect.height
        rows=page_rows.get(pi,[]) if page_rows is not None else textgap.page_lines(doc[pi])
        for r,txt in rows:
            if r.y1 < ph*0.86: continue
            t=re.sub(r'\d+','#',txt.strip().lower())
            if t: bot.setdefault(t,[]).append(r.y0)
    thr=max(2,int(len(pages)*0.3))
    reps=[min(ys) for ys in bot.values() if len(ys)>=thr]
    footer_top=min(reps) if reps else doc[0].rect.height
    tops=[]; bots=[]
    for pi in pages:
        rows=page_rows.get(pi,[]) if page_rows is not None else textgap.page_lines(doc[pi])
        cl=[r for r,txt in rows if r.y0>hb+2 and r.y1<footer_top-2 and txt.strip()]
        if not cl: continue
        tops.append(round(min(r.y0 for r in cl))); bots.append(round(max(r.y1 for r in cl)))
    Ytop=Counter(tops).most_common(1)[0][0] if tops else (hb+3)
    Ybot=Counter(bots).most_common(1)[0][0] if bots else (doc[0].rect.height-40)
    return float(Ytop), float(Ybot), float(footer_top)

def claim(cbox, direction, walls, Ytop, Ybot, CL, CR,
          edge_resolver=None, anchor_edge=None):
    """Claim a region on one side of a caption anchor.

    Logical wall boxes decide whether a wall is above/below/left/right and
    overlaps the anchor.  Painted/rendered edges decide only the final numeric
    boundary.  This prevents raster ink from changing semantic ownership.
    """
    x0,x1=cbox.x0,cbox.x1
    for _ in range(3):
        if direction=="up":
            y1=(anchor_edge if anchor_edge is not None else cbox.y0)
            ab=[wall_edge(w,"bottom",edge_resolver) for w in walls
                if wall_logical(w).y1<=cbox.y0+1
                and min(wall_logical(w).x1,x1)-max(wall_logical(w).x0,x0)>0]
            y0=max(ab+[Ytop])
        else:
            y0=(anchor_edge if anchor_edge is not None else cbox.y1)
            bl=[wall_edge(w,"top",edge_resolver) for w in walls
                if wall_logical(w).y0>=cbox.y1-1
                and min(wall_logical(w).x1,x1)-max(wall_logical(w).x0,x0)>0]
            y1=min(bl+[Ybot])
        if y1-y0<2: return fitz.Rect(x0,y0,x1,max(y0+2,y1))
        lefts=[wall_edge(w,"right",edge_resolver) for w in walls
               if wall_logical(w).x1<=cbox.x0+1
               and min(wall_logical(w).y1,y1)-max(wall_logical(w).y0,y0)>0]
        x0=max(lefts+[CL])
        rights=[wall_edge(w,"left",edge_resolver) for w in walls
                if wall_logical(w).x0>=cbox.x1-1
                and min(wall_logical(w).y1,y1)-max(wall_logical(w).y0,y0)>0]
        x1=min(rights+[CR])
        if x1-x0<2: return fitz.Rect(x0,y0,max(x0+2,x1),y1)
    return fitz.Rect(x0,y0,x1,y1)


def canonical_element_num(value):
    """Normalize a semantic element number for exact pair-to-anchor matching."""
    if value is None:
        return ""
    text=str(value).strip().casefold()
    text=re.sub(
        r"^(?:table|tab\.?|algorithm|alg\.?|equation|eq\.?)\s*", "", text)
    return text.strip(" \t\r\n:：.()[]{}")


def clip_figure_against_structured_bands(rect, caption, records):
    """Keep ownership/fallback candidates from crossing a Table band.

    The band is only a wall: it does not choose Figure direction or create a
    Figure candidate.  A band in another column is ignored.
    """
    clipped=fitz.Rect(rect)
    for record in records:
        band=record["band"]
        overlap_x=max(0.0,min(caption.x1,band.x1)-max(caption.x0,band.x0))
        if overlap_x < 0.20*min(max(caption.width,1e-6),max(band.width,1e-6)):
            continue
        caption_y=(caption.y0+caption.y1)/2
        if caption_y>=band.y1 and clipped.y0<band.y1<clipped.y1:
            clipped.y0=band.y1
        elif caption_y<=band.y0 and clipped.y0<band.y0<clipped.y1:
            clipped.y1=band.y0
    return clipped

def _ovfrac(a,b):
    ix=max(0.0,min(a.x1,b.x1)-max(a.x0,b.x0)); iy=max(0.0,min(a.y1,b.y1)-max(a.y0,b.y0))
    ar=(a.x1-a.x0)*(a.y1-a.y0); return (ix*iy)/ar if ar>0 else 0.0

def content_ink_bbox(page, region, exclude):
    """Locate visual content inside region as the union bbox of raster + vector elements,
    excluding page-spanning hairlines and blocks inside excluded regions."""
    pw,ph=page.rect.width,page.rect.height; boxes=[]
    try:
        for info in page.get_image_info(): boxes.append(fitz.Rect(info["bbox"]))
    except Exception: pass
    for dd in page.get_drawings():
        r=dd["rect"]
        if r.width<2 or r.height<2: continue
        if r.height<3 and r.width>0.5*pw: continue      # wide horizontal rule (table/header hairline)
        if r.width<3 and r.height>0.5*ph: continue      # tall vertical hairline
        boxes.append(r)
    keep=[]
    for r in boxes:
        c=fitz.Rect(r)&region
        if c.width<3 or c.height<3: continue
        if any(_ovfrac(c,e)>0.55 for e in exclude): continue
        keep.append(c)
    if not keep: return None
    return fitz.Rect(min(r.x0 for r in keep),min(r.y0 for r in keep),max(r.x1 for r in keep),max(r.y1 for r in keep))

def ink_elements(page,Ytop,Ybot):
    """Non-text ink primitives on a page (raster images + vector drawings)."""
    els=[]; pw,ph=page.rect.width,page.rect.height
    try:
        for info in page.get_image_info(): els.append(fitz.Rect(info["bbox"]))
    except Exception: pass
    for dd in page.get_drawings():
        r=dd["rect"]
        if r.width<2 or r.height<2: continue
        if r.height<3 and r.width>0.5*pw: continue      # wide horizontal hairline (not figure body)
        els.append(r)
    return [e for e in els if e.y1>Ytop-2 and e.y0<Ybot+2 and (e&page.rect).width>1]  # inside content band

def _cdist(a,b):
    ax,ay=(a.x0+a.x1)/2,(a.y0+a.y1)/2; bx,by=(b.x0+b.x1)/2,(b.y0+b.y1)/2; return ((ax-bx)**2+(ay-by)**2)**0.5

def resolve_overlaps(items, page, fullw=400.0, split_prior=None,
                     split_events=None, split_context="claim"):
    """Resolve overlapping ownership rectangles.

    ``split_prior`` is the legacy body left/right estimate. It is deliberately
    *not* a crop boundary: for horizontally adjacent Figures it may replace an
    outer claim edge only when that edge has fallen all the way back to the
    physical page edge. The resulting inner split is retained, then the prior
    is discarded before white-edge refinement and final clipping.
    """
    prior=None
    if split_prior is not None:
        try:
            prior_left=float(split_prior[0]); prior_right=float(split_prior[1])
            prior_left=max(float(page.rect.x0),prior_left)
            prior_right=min(float(page.rect.x1),prior_right)
            if (math.isfinite(prior_left) and math.isfinite(prior_right)
                    and prior_left<prior_right):
                prior=(prior_left,prior_right)
        except (TypeError,ValueError,IndexError):
            prior=None
    # ── Merge overlapping fragments that do not carry two distinct element numbers ──
    i=0
    while i < len(items):
        j=i+1
        while j < len(items):
            ra=items[i][3]; rb=items[j][3]
            if min(ra.x1,rb.x1)-max(ra.x0,rb.x0)>2 and min(ra.y1,rb.y1)-max(ra.y0,rb.y0)>2 and items[i][1]==items[j][1]:
                n1=caption_identity(items[i][2]); n2=caption_identity(items[j][2])
                if not (n1 and n2 and n1!=n2):                 # same element fragment (not two distinct numbers)
                    ca,cb=items[i][0],items[j][0]
                    items[i][0]=fitz.Rect(min(ca.x0,cb.x0),min(ca.y0,cb.y0),max(ca.x1,cb.x1),max(ca.y1,cb.y1))
                    items[i][3]=fitz.Rect(min(ra.x0,rb.x0),min(ra.y0,rb.y0),max(ra.x1,rb.x1),max(ra.y1,rb.y1))
                    if not n1 and n2: items[i][2]=items[j][2]
                    items.pop(j); continue
            j+=1
        i+=1
    # ── Split remaining overlaps between two differently numbered elements ──
    for _ in range(3):
        for i in range(len(items)):
            for j in range(i+1,len(items)):
                ra=items[i][3]; rb=items[j][3]
                ox=min(ra.x1,rb.x1)-max(ra.x0,rb.x0); oy=min(ra.y1,rb.y1)-max(ra.y0,rb.y0)
                if ox<=2 or oy<=2: continue
                ca=items[i][0]; cb=items[j][0]
                wa=ca.x1-ca.x0; wb=cb.x1-cb.x0
                capx=min(ca.x1,cb.x1)-max(ca.x0,cb.x0)        # horizontal overlap between the two captions
                if capx < 0.3*min(wa,wb):                     # side-by-side captions -> split along x
                    li,ri=(i,j) if ca.x0<cb.x0 else (j,i)
                    lc=items[li][0]; rc=items[ri][0]; lr=items[li][3]; rr=items[ri][3]
                    cL=(lc.x0+lc.x1)/2; cR=(rc.x0+rc.x1)/2
                    pair_is_figures=(items[li][1]=="figure_caption"
                                     and items[ri][1]=="figure_caption")
                    left_reference=lr.x0; right_reference=rr.x1
                    used_prior_left=used_prior_right=False
                    # A claim edge at the physical page edge is not evidence of
                    # the Figure's outer edge. Substitute the old body estimate
                    # only for this symmetry calculation; a real text/rule wall
                    # or any other non-page claim edge remains authoritative.
                    if prior is not None and pair_is_figures:
                        prior_left,prior_right=prior
                        if (abs(lr.x0-page.rect.x0)<=1.0
                                and prior_left<cL-2):
                            left_reference=prior_left
                            used_prior_left=True
                        if (abs(rr.x1-page.rect.x1)<=1.0
                                and prior_right>cR+2):
                            right_reference=prior_right
                            used_prior_right=True
                    lin=2*cL-left_reference
                    rin=2*cR-right_reference
                    raw_lin,raw_rin=lin,rin
                    symmetry_valid=(lr.x0+4<lin<=rin<rr.x1-4)
                    strategy=("legacy_body_prior_symmetry"
                              if used_prior_left or used_prior_right
                              else "claim_edge_symmetry")
                    if not symmetry_valid:                    # fall back to whitespace valley or midpoint
                        sx=find_vsplit(page,max(lr.y0,rr.y0),min(lr.y1,rr.y1),lc.x1,rc.x0) if rc.x0-lc.x1>=3 else None
                        if sx is None or not (lc.x1-1<=sx<=rc.x0+1): sx=(lc.x1+rc.x0)/2
                        lin=rin=sx
                        strategy="raster_gap_fallback"
                    if split_events is not None:
                        split_events.append({
                            "context":split_context,
                            "pair_is_figures":pair_is_figures,
                            "left_label":items[li][2],
                            "right_label":items[ri][2],
                            "left_caption_bbox":[round(v,4) for v in lc],
                            "right_caption_bbox":[round(v,4) for v in rc],
                            "left_claim_bbox":[round(v,4) for v in lr],
                            "right_claim_bbox":[round(v,4) for v in rr],
                            "split_prior":([round(prior[0],4),round(prior[1],4)]
                                           if prior is not None else None),
                            "left_reference":round(left_reference,4),
                            "right_reference":round(right_reference,4),
                            "used_prior_left":used_prior_left,
                            "used_prior_right":used_prior_right,
                            "raw_inner_edges":[round(raw_lin,4),round(raw_rin,4)],
                            "symmetry_valid":symmetry_valid,
                            "final_inner_edges":[round(lin,4),round(rin,4)],
                            "strategy":strategy,
                        })
                    items[li][3]=fitz.Rect(lr.x0,lr.y0,min(lr.x1,lin),lr.y1)
                    items[ri][3]=fitz.Rect(max(rr.x0,rin),rr.y0,rr.x1,rr.y1)
                else:                                         # vertically stacked -> split along y
                    ti,bi=(i,j) if ca.y0<cb.y0 else (j,i)
                    tr=items[ti][3]; br=items[bi][3]
                    if tr.y1 < br.y1-8:                        # upper element already bounded by a wall
                        sy=tr.y1
                    else:                                     # nearly identical claim regions -> search whitespace valley
                        # Search horizontal whitespace only within the x-intersection of the two captions.
                        hx0=max(min(ca.x0,cb.x0),max(tr.x0,br.x0)); hx1=min(max(ca.x1,cb.x1),min(tr.x1,br.x1))
                        capx0=max(ca.x0,cb.x0); capx1=min(ca.x1,cb.x1)
                        if capx1-capx0>10: hx0,hx1=capx0,capx1
                        sy=find_hsplit(page,fitz.Rect(hx0,max(tr.y0,br.y0),hx1,min(tr.y1,br.y1)))
                        if sy is None: sy=(max(tr.y0,br.y0)+min(tr.y1,br.y1))/2
                    items[ti][3]=fitz.Rect(tr.x0,tr.y0,tr.x1,min(tr.y1,sy))
                    items[bi][3]=fitz.Rect(br.x0,max(br.y0,sy),br.x1,br.y1)
    return items

def page_gray(page):
    pix=page.get_pixmap(matrix=fitz.Matrix(RZ,RZ),alpha=False)
    return list(Image.frombytes("RGB",(pix.width,pix.height),pix.samples).convert("L").getdata()), pix.width, pix.height

def refine_edges(bbox, gray, top_lim=0.0, bot_lim=1e9, left_lim=0.0, right_lim=1e9):
    d,W,H=gray
    # Exact PDF walls define ownership; raster search is rounded inward.
    tl=max(0,math.ceil(top_lim*RZ)); bl=min(H,math.floor(bot_lim*RZ))
    ll=max(0,math.ceil(left_lim*RZ)); rl=min(W,math.floor(right_lim*RZ))
    bx0=min(max(math.ceil(bbox.x0*RZ),0),W-1); bx1=min(max(math.floor(bbox.x1*RZ),1),W)
    by0=min(max(math.ceil(bbox.y0*RZ),0),H-1); by1=min(max(math.floor(bbox.y1*RZ),1),H)
    # Malformed / contradictory walls occasionally yield an empty ownership
    # interval. Preserve bbox-local behavior on that axis instead of creating
    # an inverted rectangle.
    own_x=max(bx0,ll)<min(bx1,rl)
    own_y=max(by0,tl)<min(by1,bl)
    if own_x:
        x0=max(bx0,ll); x1=min(bx1,rl)
    else:
        x0,x1=bx0,bx1; ll,rl=0,W
    if own_y:
        y0=max(by0,tl); y1=min(by1,bl)
    else:
        y0,y1=by0,by1; tl,bl=0,H
    if x1<=x0: x1=min(W,x0+1)
    if y1<=y0: y1=min(H,y0+1)
    hw=lambda y: all(d[y*W+x]>=NEAR_WHITE for x in range(x0,x1))
    vw=lambda x: all(d[y*W+x]>=NEAR_WHITE for y in range(y0,y1))
    while y0<y1-2 and hw(y0): y0+=1
    while y1>y0+2 and hw(y1-1): y1-=1
    while x0<x1-2 and vw(x0): x0+=1
    while x1>x0+2 and vw(x1-1): x1-=1
    while y0>tl and not hw(y0): y0-=1
    while y1<bl and not hw(y1-1): y1+=1
    while x0>ll and not vw(x0): x0-=1
    while x1<rl and not vw(x1-1): x1+=1
    # Keep the original floating-point walls authoritative for the PDF clip.
    final_left=left_lim if own_x else 0.0; final_right=right_lim if own_x else W/RZ
    final_top=top_lim if own_y else 0.0; final_bottom=bot_lim if own_y else H/RZ
    return fitz.Rect(max(x0/RZ,final_left),max(y0/RZ,final_top),
                     min(x1/RZ,final_right),min(y1/RZ,final_bottom))

def refine_trim(bbox, gray):
    """Trim pure-white margins inward only (no outward expansion) for ownership seeds."""
    d,W,H=gray
    x0=min(max(int(bbox.x0*RZ),0),W-1); x1=min(max(int(bbox.x1*RZ),1),W)
    y0=min(max(int(bbox.y0*RZ),0),H-1); y1=min(max(int(bbox.y1*RZ),1),H)
    if x1<=x0: x1=x0+1
    if y1<=y0: y1=y0+1
    hw=lambda y: all(d[y*W+x]>=NEAR_WHITE for x in range(x0,x1))
    vw=lambda x: all(d[y*W+x]>=NEAR_WHITE for y in range(y0,y1))
    while y0<y1-2 and hw(y0): y0+=1
    while y1>y0+2 and hw(y1-1): y1-=1
    while x0<x1-2 and vw(x0): x0+=1
    while x1>x0+2 and vw(x1-1): x1-=1
    return fitz.Rect(x0/RZ,y0/RZ,x1/RZ,y1/RZ)

def render_plain(page,bbox,out,z=textgap.ZOOM):
    pix=page.get_pixmap(matrix=fitz.Matrix(z,z),clip=fitz.Rect(bbox),alpha=False)
    Image.frombytes("RGB",(pix.width,pix.height),pix.samples).save(out); return pix.width,pix.height

def _ov(a,b): return max(0.0,min(a.x1,b.x1)-max(a.x0,b.x0))*max(0.0,min(a.y1,b.y1)-max(a.y0,b.y0))

def _area(r): return max(0.0,r.x1-r.x0)*max(0.0,r.y1-r.y0)

def nontext_ink(page, text_rects, Ytop, Ybot):
    """Filter out vector underlines/highlights/link boxes overlapping text lines from ink primitives."""
    out=[]
    for e in ink_elements(page,Ytop,Ybot):
        if any(_ov(e,tr)>0.5*_area(e) for tr in text_rects): continue
        out.append(e)
    return out

def header_bottom(lines, role, page_count, ph):
    """Bottom y of repeated running headers/watermarks (same text/y across >=30% of pages),
    excluding figure_content and subfig_caption lines."""
    top={}
    for g,pi,r,txt in lines:
        if r.y0 > ph*0.14: continue
        if role.get(g) in ("figure_content","subfig_caption","figure_caption","table_title","heading","title_author"): continue
        t=re.sub(r'\d+','#',txt.strip().lower())
        if t: top.setdefault((t,round(r.y0)),[]).append(r.y1)
    thr=max(2,int(page_count*0.3))
    reps=[max(ys) for k,ys in top.items() if len(ys)>=thr]
    return max(reps) if reps else 0.0

def ink_components(pool, page, cell):
    """Group ink primitives into connected components via grid quantization (cell=med_lh) + 1-cell dilation + BFS."""
    import collections
    W=int(page.rect.width/cell)+2; H=int(page.rect.height/cell)+2
    grid={}
    for idx,e in enumerate(pool):
        for gx in range(max(0,int(e.x0/cell)-1), min(W,int(e.x1/cell)+2)):
            for gy in range(max(0,int(e.y0/cell)-1), min(H,int(e.y1/cell)+2)):
                grid.setdefault((gx,gy),[]).append(idx)
    seen=set(); comps=[]
    for start in grid:
        if start in seen: continue
        q=collections.deque([start]); seen.add(start); members=[]
        while q:
            c=q.popleft(); members+=grid[c]
            for dx in (-1,0,1):
                for dy in (-1,0,1):
                    nb=(c[0]+dx,c[1]+dy)
                    if nb in grid and nb not in seen: seen.add(nb); q.append(nb)
        idxs=set(members)
        comps.append(fitz.Rect(min(pool[i].x0 for i in idxs),min(pool[i].y0 for i in idxs),
                               max(pool[i].x1 for i in idxs),max(pool[i].y1 for i in idxs)))
    return comps


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("paper"); ap.add_argument("out"); ap.add_argument("--seg",default=None)
    ap.add_argument("--source-lines", default=None,
                    help="Stable source_lines.json paired with segments2 (default: sibling file)")
    ap.add_argument("--outer-boundaries", default=None,
                    help="Per-page outer boundary annotations (default: sibling page_outer_boundaries.json)")
    ap.add_argument("--structured-elements", default=None,
                    help="Structured elements from the semantic pass (default: sibling structured_elements.json)")
    ap.add_argument("--disable-table-bands",action="store_true",
                    help="Ablation: disable deterministic Table bands and ownership isolation while keeping semantic cache")
    ap.add_argument("--disable-outer-boundaries",action="store_true",
                    help="Ablation: disable outer boundary rules and text chrome geometry while keeping semantic cache")
    ap.add_argument("--include-appendix",action="store_true",
                    help="Also crop appendix/supplementary pages; References are always skipped")
    a=ap.parse_args()
    pdir=Path(a.paper); slug=pdir.name; od=Path(a.out)
    for sub in ("figures","appendix_figures","overlay"):
        if (od/sub).exists(): shutil.rmtree(od/sub)
        (od/sub).mkdir(parents=True,exist_ok=True)
    seg=Path(a.seg) if a.seg else od/"segments2.json"
    if not seg.exists(): print("!! Missing role-segmentation cache (run segment_roles.py first):",seg); sys.exit(1)
    segs=json.loads(seg.read_text())
    outer_path = Path(a.outer_boundaries) if a.outer_boundaries else seg.with_name("page_outer_boundaries.json")
    try:
        outer_payload = json.loads(outer_path.read_text()) if outer_path.exists() else {}
    except Exception as exc:
        print(f"!! Invalid outer-boundaries file; using physical PDF page edges: {exc}", file=sys.stderr)
        outer_payload = {}
    if a.disable_outer_boundaries:
        outer_payload = {}
    doc=fitz.open(pdir/"paper.pdf")
    structured_path=(Path(a.structured_elements) if a.structured_elements
                     else seg.with_name("structured_elements.json"))
    try:
        structured_payload=(json.loads(structured_path.read_text())
                            if structured_path.exists() else {})
    except Exception as exc:
        print(f"!! Invalid structured_elements.json; disabling deterministic Table bands: {exc}",
              file=sys.stderr)
        structured_payload={}
    sectioned=textgap.sectioned_document_lines(doc)
    allowed_sections={"body","appendix"} if a.include_appendix else {"body"}
    content_lines=[
        [(r,t,section) for r,t,section in rows if section in allowed_sections]
        for rows in sectioned
    ]
    page_indices=[pi for pi,rows in enumerate(content_lines) if rows]
    page_set=set(page_indices)
    painted_envelopes={pi:painted_text_envelopes(doc[pi]) for pi in page_indices}
    def painted_wall(pi, rect):
        return expand_wall_to_painted_ink(
            rect, painted_envelopes.get(pi, ()))
    page_rows={pi:[(r,t) for r,t,_section in content_lines[pi]]
               for pi in page_indices}
    excluded_walls={pi:[fitz.Rect(r) for r,_t,section in sectioned[pi]
                        if section not in allowed_sections]
                    for pi in page_indices}
    all_text_rects={pi:[fitz.Rect(r) for r,_t,_section in sectioned[pi]]
                    for pi in page_indices}
    page_sections={pi:{section for _r,_t,section in content_lines[pi]}
                   for pi in page_indices}
    cols=acols(doc,page_indices,page_rows)
    if not cols: cols=[fitz.Rect(doc[0].rect.x0+40,doc[0].rect.y0,doc[0].rect.x1-40,doc[0].rect.y1)]
    hb=textgap.detect_header_band(doc)
    Ytop,Ybot,footer_top=content_margins(doc,hb,page_indices,page_rows)
    CL=min(c.x0 for c in cols); CR=max(c.x1 for c in cols)
    # Split-only prior. This must never be substituted into per-page legal
    # bounds, wall masks, white-edge search limits, or final crop clipping.
    legacy_body_lr_split_prior=(float(CL),float(CR))
    if len(cols)==2:
        mid=(cols[0].x1+cols[1].x0)/2
        cut_cols=[fitz.Rect(cols[0].x0,cols[0].y0,mid,cols[0].y1),fitz.Rect(mid,cols[1].y0,cols[1].x1,cols[1].y1)]
    else: cut_cols=list(cols)

    source_path = Path(a.source_lines) if a.source_lines else seg.with_name("source_lines.json")
    semantic_rows = None
    source_doc = []
    if source_path.exists():
        try:
            source_doc = json.loads(source_path.read_text(encoding="utf-8"))
            if not isinstance(source_doc,list):
                raise ValueError("source_lines root is not a list")
            semantic_rows = textgap.load_source_document_lines(
                source_path, include_appendix=a.include_appendix, page_count=len(doc))
        except Exception as exc:
            print(f"!! Invalid source_lines.json; falling back to live PDF line numbering: {exc}", file=sys.stderr)
            source_doc = []
    if semantic_rows:
        lines=[(g,pi,r,t) for g,pi,r,t,section in semantic_rows if pi in page_set]
        line_sections={g:section for g,pi,r,t,section in semantic_rows if pi in page_set}
    else:
        if not source_path.exists():
            print(f"!! Missing {source_path}; role alignment may drift across PyMuPDF versions", file=sys.stderr)
        lines=[]
        line_sections={}
        for pi in page_indices:
            for r,t,section in content_lines[pi]:
                g=len(lines)
                lines.append((g,pi,r,t))
                line_sections[g]=section
    role,num=role_maps(segs,(g for g,pi,r,t in lines))
    families=family_maps(segs,(g for g,pi,r,t in lines))
    per={}
    for g,pi,r,t in lines: per.setdefault(pi,[]).append((g,r,t))
    table_band_doc=derive_table_bands(
        doc,source_doc,segs,structured_payload,allowed_pages=page_set)
    (od/"deterministic_table_bands.json").write_text(
        json.dumps(table_band_doc,ensure_ascii=False,indent=2),encoding="utf-8")
    if a.disable_table_bands:
        table_bands_by_page,table_band_by_element={},{}
    else:
        table_bands_by_page,table_band_by_element=index_active_table_bands(
            table_band_doc,doc)
    # Text header/footer chrome comes from the semantic pass; geometry uses rendered PDF bboxes.
    # Merge contiguous LLM segments before determining orientation and nearest page edge.
    text_chrome_by_page={}
    line_by_id={g:(pi,fitz.Rect(r),t) for g,pi,r,t in lines}
    if not a.disable_outer_boundaries:
        for segment in segs:
            semantic_role=segment.get("role")
            if semantic_role not in {"page_header","page_footer"}:
                continue
            segment_by_page={}
            for g in range(segment.get("start",0),segment.get("end",-1)+1):
                item=line_by_id.get(g)
                if item is None:
                    continue
                pi,r,t=item
                segment_by_page.setdefault(pi,[]).append((g,r,t))
            for pi,members in segment_by_page.items():
                painted=None
                for _g,r,_t in members:
                    line_painted=painted_wall(pi,r)
                    if line_painted.is_empty:
                        line_painted=fitz.Rect(r)
                    painted=(painted|line_painted) if painted is not None else line_painted
                if painted is None or painted.is_empty:
                    continue
                page_rect=doc[pi].rect
                if painted.width>=painted.height:
                    top_gap=abs(painted.y0-page_rect.y0)
                    bottom_gap=abs(page_rect.y1-painted.y1)
                    side="top" if top_gap<=bottom_gap else "bottom"
                    text_axis="horizontal"
                else:
                    left_gap=abs(painted.x0-page_rect.x0)
                    right_gap=abs(page_rect.x1-painted.x1)
                    side="left" if left_gap<=right_gap else "right"
                    text_axis="vertical"
                page_entry=text_chrome_by_page.setdefault(pi,{})
                entry=page_entry.get(side)
                member_ids=[g for g,_r,_t in members]
                member_texts=[t for _g,_r,t in members]
                if entry is None:
                    page_entry[side]={
                        "bbox":painted,
                        "line_ids":member_ids,
                        "texts":member_texts,
                        "roles":[semantic_role],
                        "axes":[text_axis],
                    }
                else:
                    entry["bbox"]|=painted
                    entry["line_ids"].extend(member_ids)
                    entry["texts"].extend(member_texts)
                    entry["roles"].append(semantic_role)
                    entry["axes"].append(text_axis)
    # Self-calibrated unit: median body line height
    bh=[r.y1-r.y0 for g,pi,r,t in lines if role.get(g)=="body"]
    med_lh=statistics.median(bh) if bh else 10.0
    gcache={}
    rendered_edge_cache={}
    rendered_edge_events=[]

    def resolved_text_edge(wall, side):
        """Lazy per-line edge measurement on the one shared page raster."""
        logical=wall_logical(wall)
        painted=wall_painted(wall)
        pi=wall.get("page_index") if isinstance(wall,dict) else None
        if pi is None or pi not in page_set:
            return _rect_edge(painted,side)
        key=(pi,wall.get("line_id"),tuple(round(v,4) for v in logical),side)
        if key in rendered_edge_cache:
            return rendered_edge_cache[key]
        if pi not in gcache:
            gcache[pi]=page_gray(doc[pi])
        coordinate,detail=rendered_text_edge(
            logical,painted,side,gcache[pi],med_lh)
        rendered_edge_cache[key]=coordinate
        detail.update({
            "page":pi+1,
            "line_id":wall.get("line_id"),
            "role":wall.get("role"),
            "side":side,
            "logical_bbox":[round(v,4) for v in logical],
            "painted_bbox":[round(v,4) for v in painted],
            "text":wall.get("text","")[:240],
        })
        rendered_edge_events.append(detail)
        return coordinate

    Ytop=max(Ytop, header_bottom(lines,role,len(page_indices),doc[0].rect.height)+1)   # exclude repeated running header
    diagnostic_legacy_bounds=(Ytop,Ybot,CL,CR)

    # Per-page valid bounds: only high-confidence outer rules replace physical page edges;
    # missing or low-confidence sides default to physical PDF bounds. Multi-column dividers stay independent.
    selected_pages={}
    for item in outer_payload.get("pages",[]) if isinstance(outer_payload,dict) else []:
        if not isinstance(item,dict):
            continue
        try: selected_pages[int(item.get("page"))]=item
        except (TypeError,ValueError): continue
    page_outer_bounds={}
    page_outer_hard_bounds={}
    application=[]
    for pi in page_indices:
        page=doc[pi]
        page_bounds=(float(page.rect.y0),float(page.rect.y1),
                     float(page.rect.x0),float(page.rect.x1))
        top,bottom,left,right=page_bounds
        sources={side:"page_edge" for side in ("top","bottom","left","right")}
        selection=selected_pages.get(pi+1,{})
        applied={}
        for side in ("top","bottom","left","right"):
            value=selection.get(side,{}) if isinstance(selection,dict) else {}
            bbox=value.get("bbox") if isinstance(value,dict) and value.get("active") is True else None
            try:
                rule=fitz.Rect(bbox) if isinstance(bbox,list) and len(bbox)==4 else None
            except Exception:
                rule=None
            if rule is None or rule.is_empty or not all(math.isfinite(v) for v in tuple(rule)):
                continue
            if side=="top":
                top=max(page.rect.y0,rule.y1)
            elif side=="bottom":
                bottom=min(page.rect.y1,rule.y0)
            elif side=="left":
                left=max(page.rect.x0,rule.x1)
            else:
                right=min(page.rect.x1,rule.x0)
            sources[side]=value.get("candidate_id") or "outer_rule"
            applied[side]={
                "candidate_id":value.get("candidate_id"),
                "bbox":[round(v,4) for v in rule],
                "confidence":value.get("confidence"),
                "reason":value.get("reason",""),
            }
        # Text chrome shares the same page side with rule lines as a pure boundary clamp,
        # never participating in claim direction or ownership.
        # top→y1、bottom→y0、left→x1、right→x0。
        for side in ("top","bottom","left","right"):
            chrome=text_chrome_by_page.get(pi,{}).get(side)
            if not chrome:
                continue
            chrome_rect=fitz.Rect(chrome["bbox"])&page.rect
            if chrome_rect.is_empty or not all(math.isfinite(v) for v in tuple(chrome_rect)):
                continue
            if side=="top":
                top=max(top,chrome_rect.y1)
                effective_coordinate=top
            elif side=="bottom":
                bottom=min(bottom,chrome_rect.y0)
                effective_coordinate=bottom
            elif side=="left":
                left=max(left,chrome_rect.x1)
                effective_coordinate=left
            else:
                right=min(right,chrome_rect.x0)
                effective_coordinate=right
            semantic_roles=sorted(set(chrome["roles"]))
            semantic_source="+".join(semantic_roles)
            prior_source=sources[side]
            sources[side]=(semantic_source if prior_source=="page_edge"
                           else f"{prior_source}+{semantic_source}")
            entry=applied.setdefault(side,{
                "candidate_id":None,
                "bbox":None,
                "confidence":"semantic",
                "reason":"text-only page chrome",
            })
            entry.update({
                "text_roles":semantic_roles,
                "text_axes":sorted(set(chrome["axes"])),
                "text_bbox":[round(v,4) for v in chrome_rect],
                "text_line_ids":chrome["line_ids"],
                "text":chrome["texts"],
                "effective_coordinate":round(effective_coordinate,4),
            })
        # Outer boundaries may not cross a fixed two-column center divider; fall back to physical edge if so.
        if len(cols)==2:
            if left>=mid-med_lh:
                left=float(page.rect.x0)
                sources["left"]="page_edge_invalid_against_mid"
                applied.pop("left",None)
            if right<=mid+med_lh:
                right=float(page.rect.x1)
                sources["right"]="page_edge_invalid_against_mid"
                applied.pop("right",None)
        if top>=bottom-med_lh:
            if "top" in applied:
                top=float(page.rect.y0)
                sources["top"]="page_edge_invalid_pair"
                applied.pop("top",None)
            if top>=bottom-med_lh and "bottom" in applied:
                bottom=float(page.rect.y1)
                sources["bottom"]="page_edge_invalid_pair"
                applied.pop("bottom",None)
        if left>=right-med_lh:
            if "left" in applied:
                left=float(page.rect.x0)
                sources["left"]="page_edge_invalid_pair"
                applied.pop("left",None)
            if left>=right-med_lh and "right" in applied:
                right=float(page.rect.x1)
                sources["right"]="page_edge_invalid_pair"
                applied.pop("right",None)
# Fall back to physical page edges if combined bounds become degenerate.
        if top>=bottom-med_lh or left>=right-med_lh:
            top,bottom,left,right=page_bounds
            for side in tuple(applied):
                sources[side]="page_edge_invalid_rectangle"
            applied.clear()
        page_outer_bounds[pi]=(float(top),float(bottom),float(left),float(right))
        hard_top,hard_bottom,hard_left,hard_right=(
            float(top),float(bottom),float(left),float(right))
        page_outer_hard_bounds[pi]=(hard_top,hard_bottom,hard_left,hard_right)
        application.append({
            "page":pi+1,
            "page_edge":{"top":page.rect.y0,"bottom":page.rect.y1,
                         "left":page.rect.x0,"right":page.rect.x1},
            "legacy_body_bounds":{"top":diagnostic_legacy_bounds[0],
                                  "bottom":diagnostic_legacy_bounds[1],
                                  "left":diagnostic_legacy_bounds[2],
                                  "right":diagnostic_legacy_bounds[3],
                                  "usage":"left/right only as adjacent-Figure caption-symmetry prior; never a crop/refinement boundary"},
            "effective":{"top":top,"bottom":bottom,"left":left,"right":right},
            "hard_clip":{"top":hard_top,"bottom":hard_bottom,
                         "left":hard_left,"right":hard_right},
            "source":sources,
            "applied":applied,
            "fixed_column_midpoint":mid if len(cols)==2 else None,
        })
    application_doc={
        "schema_version":"dp-p2p.page-outer-boundary-application.v2",
        "policy":"high-confidence-rule-unioned-with-semantic-text-chrome; missing-side-uses-page-edge; legacy-body-left-right-only-for-adjacent-Figure-symmetry",
        "pages":application,
        "active_side_count":sum(len(item["applied"]) for item in application),
        "text_chrome_side_count":sum(
            sum("text_roles" in value for value in item["applied"].values())
            for item in application),
    }
    (od/"page_outer_boundary_application.json").write_text(
        json.dumps(application_doc,ensure_ascii=False,indent=2),encoding="utf-8")
    print(f"Outer-boundary fallback=PDF edges; body-column prior X[{CL:.0f},{CR:.0f}] "
          f"(diagnostic Y[{Ytop:.0f},{Ybot:.0f}]), cols={[[round(c.x0),round(c.x1)] for c in cols]}, "
          f"active_sides={application_doc['active_side_count']} "
          f"(text_chrome={application_doc['text_chrome_side_count']}), med_lh={med_lh:.1f}")

    flags=[]   # R2 invariant flag records
    elements=[]  # (pi, rect, kind, label, clb, crb, trimonly, section)
    xpage_cand=[]  # cross-page float candidates: (pi, label, cbox, section)
    matched_table_band_ids=set()
    immutable_table_elements={}
    split_prior_events=[]
    for pi in page_indices:
        rows=per.get(pi,[]); page=doc[pi]
        if not rows: continue
        pYtop,pYbot,pCL,pCR=page_outer_bounds.get(pi,(doc[pi].rect.y0,doc[pi].rect.y1,doc[pi].rect.x0,doc[pi].rect.x1))
        hYtop,hYbot,hCL,hCR=page_outer_hard_bounds.get(
            pi,(page.rect.y0,page.rect.y1,page.rect.x0,page.rect.x1))
        claim_rect=fitz.Rect(pCL,pYtop,pCR,pYbot)&page.rect
        legal_rect=fitz.Rect(hCL,hYtop,hCR,hYbot)&page.rect
        if len(cols)==2:
            pcut_cols=[
                fitz.Rect(pCL,page.rect.y0,mid,page.rect.y1),
                fitz.Rect(mid,page.rect.y0,pCR,page.rect.y1),
            ]
        else:
            pcut_cols=[fitz.Rect(pCL,page.rect.y0,pCR,page.rect.y1)]
        # ── Anchors: group figure_caption and table_title segments by num ──
        def units(kind):
            u={}; unnum=[]
            for g,r,t in rows:
                if role.get(g)!=kind: continue
                n=num.get(g)
                if n is None: unnum.append((g,r,t)); continue
                key=(families.get(g,family_name(role=kind)),n)
                u[key]=(u[key]|fitz.Rect(r)) if key in u else fitz.Rect(r)
            out=[(f"{family} {n}",bx) for (family,n),bx in u.items()]
            for g,r,t in unnum:                                  # unnumbered captions: keep each as anchor
                out.append((t.strip()[:14] or f"u{g}",fitz.Rect(r)))
            return out
        def anchor_section(cbox,kind):
            candidate_roles={kind}
            if kind=="figure_caption": candidate_roles.add("subfig_caption")
            for g,r,_t in rows:
                if role.get(g) in candidate_roles and _ov(r,cbox)>0:
                    return line_sections[g]
            return next(iter(page_sections.get(pi,{"body"})))
        fig_units=units("figure_caption"); tab_units=units("table_title")
        if not fig_units:                                        # no main caption on page: promote subfig captions as anchors
            subs={}
            for g,r,t in rows:
                if role.get(g)=="subfig_caption":
                    parent=num.get(g,f"s{g}")
                    family=families.get(g,"Figure")
                    key=(f"Figure {parent}" if isinstance(parent,int) else str(parent)) if family=="Figure" else f"{family} {parent}"
                    subs[key]=(subs[key]|fitz.Rect(r)) if key in subs else fitz.Rect(r)
            fig_units=[(f"Figure {k}" if isinstance(k,int) else str(k),bx) for k,bx in subs.items()]
        if not fig_units and not tab_units: continue
        # ── Walls: logical bbox determines ownership; painted bbox determines coordinates ──
        walls=[make_text_wall(
                   r,painted_wall(pi,r),pi,g,role.get(g),t)
               for g,r,t in rows if role.get(g) in WALL_ROLES]
        walls += [make_text_wall(r,painted_wall(pi,r),pi,None,
                                 "excluded_section","")
                  for r in excluded_walls.get(pi,[])]
        local_table_bands=table_bands_by_page.get(pi,[])
        local_table_walls=[record[side]
                           for record in local_table_bands
                           for side in ("top","bottom")]
        figure_walls=walls+local_table_walls
        units_all=fig_units+tab_units
        label_roles={**{label:"figure_caption" for label,_ in fig_units},
                     **{label:"table_title" for label,_ in tab_units}}
        table_record_by_label={}
        _own_cache={}
        def anchor_boundary(cbox,kind,side):
            members=[]
            for wall in walls:
                logical=wall_logical(wall)
                if (wall.get("role")!=kind or _ov(logical,cbox)<=0):
                    continue
                members.append(resolved_text_edge(wall,side))
            if not members:
                return _rect_edge(cbox,side)
            return (min(members) if side in {"top","left"}
                    else max(members))
        def owned_seed(cbox):
            """Ownership fallback seed: ink + figure text -> connected components -> nearest-caption union."""
            if "comps" not in _own_cache:
                trs=all_text_rects.get(pi,[])
                pool=list(nontext_ink(page,trs,pYtop,pYbot))
                pool+=[fitz.Rect(r) for g,r,t in rows if role.get(g) in FIG_PART]
                pool=exclude_table_owned_ink(pool,local_table_bands)
                _own_cache["comps"]=ink_components(pool,page,max(2.0,med_lh/2)) if pool else []
            mine=[c for c in _own_cache["comps"]
                  if min(range(len(units_all)),key=lambda i:_cdist(c,units_all[i][1]))==next(i for i,(l,b) in enumerate(units_all) if b is cbox)]
            if not mine: return None
            return fitz.Rect(min(c.x0 for c in mine),min(c.y0 for c in mine),max(c.x1 for c in mine),max(c.y1 for c in mine))&page.rect
        # ── Claim regions ──
        items=[]
        for label,cbox in tab_units:
            identity=split_label(label)
            table_num=canonical_element_num(namespaced_num(identity[1],identity[0])) if identity else canonical_element_num(_capnum(label) or label)
            band_record=table_band_by_element.get((pi,"table",table_num))
            if band_record is not None:
                # Semantic membership plus exact PDF geometry gives an
                # immutable Table body band.  No direction or candidate rank
                # participates in this branch.
                rect=fitz.Rect(band_record["band"])
                band_id=f"p{pi+1}:table:{table_num}"
                matched_table_band_ids.add(band_id)
                table_record_by_label[label]=band_record
                immutable_table_elements[(pi,label)]=fitz.Rect(rect)
            else:
                gu=ink_gap(page,cbox,"up",pYtop,pYbot); gd=ink_gap(page,cbox,"down",pYtop,pYbot)
                if abs(gu-gd)<=6:
                    cy=(cbox.y0+cbox.y1)/2
                    others=[(oc.y0+oc.y1)/2 for ol,oc in fig_units+tab_units if oc is not cbox and min(oc.x1,cbox.x1)-max(oc.x0,cbox.x0)>0]
                    dA=min([cy-y for y in others if y<cy],default=1e9); dB=min([y-cy for y in others if y>cy],default=1e9)
                    dirn="up" if dB<dA else "down"
                else: dirn="up" if gu<gd else "down"
                anchor_side="top" if dirn=="up" else "bottom"
                rect=claim(cbox,dirn,walls,pYtop,pYbot,pCL,pCR,
                           resolved_text_edge,
                           anchor_boundary(cbox,"table_title",anchor_side))
                if not has_ink(page,rect):
                    other="down" if dirn=="up" else "up"
                    anchor_side="top" if other=="up" else "bottom"
                    rect=claim(cbox,other,walls,pYtop,pYbot,pCL,pCR,
                               resolved_text_edge,
                               anchor_boundary(cbox,"table_title",anchor_side))
            items.append([cbox,"table_title",label,rect])
        for label,cbox in fig_units:
            ru=claim(cbox,"up",figure_walls,pYtop,pYbot,pCL,pCR,
                     resolved_text_edge,
                     anchor_boundary(cbox,"figure_caption","top"))
            if has_ink(page,ru): rect=ru
            else:
                rd=claim(cbox,"down",figure_walls,pYtop,pYbot,pCL,pCR,
                         resolved_text_edge,
                         anchor_boundary(cbox,"figure_caption","bottom"))
                rect=rd if has_ink(page,rd) else ru
            if rect.height<med_lh or rect.width<med_lh:          # collapsed region -> ownership fallback
                cbx=owned_seed(cbox)
                if cbx is None:                                   # fallback to content bbox if no component matched
                    excl=[oc for ol,oc in fig_units+tab_units if oc is not cbox]
                    cbx=content_ink_bbox(page,claim_rect,excl)
                if cbx and cbx.height>=med_lh and cbx.width>=med_lh and _area(cbx)>_area(rect)*2:
                    rect=cbx; flags.append({"page":pi+1,"kind":"fallback_content_layer","detail":label})
            items.append([cbox,"figure_caption",label,rect])
        resolve_overlaps(
            items,page,pCR-pCL,
            split_prior=legacy_body_lr_split_prior,
            split_events=split_prior_events,
            split_context="claim")
        for item in items:
            if item[1]=="table_title" and item[2] in table_record_by_label:
                item[3]=fitz.Rect(table_record_by_label[item[2]]["band"])
            elif item[1]=="figure_caption":
                item[3]=clip_figure_against_structured_bands(
                    item[3],item[0],local_table_bands)
        # ── Page-level deterministic invariant scoring ──
        units_all=fig_units+tab_units
        def _score(cand):
            """Score candidate crops as ((total_violations, missing_figures), valid_elements); figure completeness takes priority."""
            ok=[(c,k,l,r) for c,k,l,r in cand if r.width>8 and r.height>8 and has_ink(page,r)]
            v=len(cand)-len(ok)
            mf=sum(1 for c,k,l,r in cand if k=="figure_caption" and not(r.width>8 and r.height>8 and has_ink(page,r)))
            for i in range(len(ok)):
                for j in range(i+1,len(ok)):
                    if _ov(ok[i][3],ok[j][3])>0.01*min(_area(ok[i][3]),_area(ok[j][3])): v+=1
            for lab,cb in units_all:
                for c,k,l,r in ok:
                    if l!=lab and _ov(r,cb)>0.5*_area(cb): v+=1
            return (v,mf),ok
        cand_claim=[(c,k,l,fitz.Rect(r)) for c,k,l,r in items]
        v_claim,ok_claim=_score(cand_claim)
        chosen,ok_chosen,trimonly = cand_claim,ok_claim,False
        if v_claim>(0,0) and len(units_all)>0:
            # R3 arbitration: compare claim vs connected-component ownership when claim has violations
            pool=list(nontext_ink(page,all_text_rects.get(pi,[]),pYtop,pYbot))
            pool+=[fitz.Rect(r) for g,r,t in rows if role.get(g) in FIG_PART]
            pool=exclude_table_owned_ink(pool,local_table_bands)
            comps=ink_components(pool,page,max(2.0,med_lh/2)) if pool else []   # dilated adjacency ~= 1x line height
            owned={i:[] for i in range(len(units_all))}
            for e in comps:
                bi=min(range(len(units_all)),key=lambda i:_cdist(e,units_all[i][1]))
                owned[bi].append(e)
            cand_own=[]
            lab2ck={l:(c,k) for c,k,l,r in cand_claim}
            for i,(lab,cb) in enumerate(units_all):
                if lab in table_record_by_label:
                    seed=fitz.Rect(table_record_by_label[lab]["band"])
                else:
                    es=owned[i]
                    if es: seed=fitz.Rect(min(e.x0 for e in es),min(e.y0 for e in es),max(e.x1 for e in es),max(e.y1 for e in es))
                    else: seed=fitz.Rect(0,0,0,0)
                c,k=lab2ck.get(lab,(cb,label_roles[lab]))
                cand_own.append([c,k,lab,seed&page.rect])
            resolve_overlaps(
                cand_own,page,pCR-pCL,
                split_prior=legacy_body_lr_split_prior,
                split_events=split_prior_events,
                split_context="ownership")                    # split residual overlaps via whitespace/caption symmetry
            for item in cand_own:
                if item[1]=="table_title" and item[2] in table_record_by_label:
                    item[3]=fitz.Rect(table_record_by_label[item[2]]["band"])
                elif item[1]=="figure_caption":
                    item[3]=clip_figure_against_structured_bands(
                        item[3],item[0],local_table_bands)
            v_own,ok_own=_score(cand_own)
            if v_own<v_claim:
                chosen,ok_chosen,trimonly=cand_own,ok_own,True
                flags.append({"page":pi+1,"kind":"R3_ownership_adopted","detail":f"claim_violations={v_claim}->ownership_violations={v_own}"})
        # Record residual I1 on the chosen candidate (I2/I3 checked after refinement)
        chosen_labels={l for c,k,l,r in ok_chosen}
        for c,k,l,r in chosen:
            if l not in chosen_labels:
                flags.append({"page":pi+1,"kind":"I1_missing_element","detail":f"{l} empty or no ink"})
                if k=="figure_caption" and not any(
                        wall_logical(w).y1<=c.y0+1
                        and min(wall_logical(w).x1,c.x1)
                        -max(wall_logical(w).x0,c.x0)>0
                        for w in figure_walls):
                    xpage_cand.append((pi,l,c,anchor_section(c,k)))  # top-of-page caption may match float on previous page
        for cbox,kind,label,rect in ok_chosen:
            if kind=="figure_caption":
                rect=clip_figure_against_structured_bands(
                    rect,cbox,local_table_bands)
            rect=fitz.Rect(rect)&legal_rect
            clb,crb=hCL,hCR
            if not trimonly:
                ci,_=textgap.col_of(cbox,cols)
                if len(cols)==2 and ci in (0,1):
                    crosses = cbox.x0 < cols[0].x1-2 and cbox.x1 > cols[1].x0+2
                    if not crosses:
                        clb,crb=pcut_cols[ci].x0,pcut_cols[ci].x1
                        rect=fitz.Rect(max(rect.x0,clb),rect.y0,min(rect.x1,crb),rect.y1)
            elements.append((pi,rect,kind,label,clb,crb,trimonly,anchor_section(cbox,kind)))

    split_prior_application={
        "schema_version":"dp-p2p.caption-split-prior.v1",
        "policy":"legacy body left/right may replace only a physical-page outer claim edge during adjacent-Figure caption symmetry; discarded before refinement and final crop",
        "split_prior":[round(v,4) for v in legacy_body_lr_split_prior],
        "event_count":len(split_prior_events),
        "legacy_prior_event_count":sum(
            event["used_prior_left"] or event["used_prior_right"]
            for event in split_prior_events),
        "events":split_prior_events,
    }
    (od/"caption_split_prior_application.json").write_text(
        json.dumps(split_prior_application,ensure_ascii=False,indent=2),
        encoding="utf-8")

    table_band_application={
        "schema_version":"dp-p2p.deterministic-structured-band-application.v2",
        "policy":"semantic Table/Algorithm membership plus exact PDF rules forms local walls; Tables remain immutable outputs, Algorithms only wall/clip adjacent Figures",
        "disabled_for_ablation":a.disable_table_bands,
        "derived_active_count":table_band_doc.get("active_count",0),
        "active_table_count":sum(
            item.get("active") is True and item.get("element_type")=="table"
            for item in table_band_doc.get("bands",[])),
        "active_algorithm_count":sum(
            item.get("active") is True and item.get("element_type")=="algorithm"
            for item in table_band_doc.get("bands",[])),
        "local_wall_count":2*table_band_doc.get("active_count",0),
        "matched_table_band_ids":sorted(matched_table_band_ids),
        "conflict_count":table_band_doc.get("conflict_count",0),
    }
    (od/"deterministic_table_band_application.json").write_text(
        json.dumps(table_band_application,ensure_ascii=False,indent=2),
        encoding="utf-8")
    print(f"Deterministic structural bands: Table={table_band_application['active_table_count']} / "
          f"Algorithm={table_band_application['active_algorithm_count']} / "
          f"matched_Table={len(table_band_application['matched_table_band_ids'])} / "
          f"conflicts={table_band_application['conflict_count']}")

    # ── Cross-page orphan adoption for floating figures separated from top-of-page captions ──
    for cpi,label,cbox,section in xpage_cand:
        if cpi==0: continue
        prev=cpi-1
        if prev not in page_set or section not in page_sections.get(prev,set()): continue
        ppage=doc[prev]
        pYtop,pYbot,pCL,pCR=page_outer_bounds.get(prev,(ppage.rect.y0,ppage.rect.y1,ppage.rect.x0,ppage.rect.x1))
        hYtop,hYbot,hCL,hCR=page_outer_hard_bounds.get(
            prev,(ppage.rect.y0,ppage.rect.y1,ppage.rect.x0,ppage.rect.x1))
        legal_rect=fitz.Rect(hCL,hYtop,hCR,hYbot)&ppage.rect
        pool=list(nontext_ink(ppage,all_text_rects.get(prev,[]),pYtop,pYbot))
        pool+=[fitz.Rect(r) for g,r,t in per.get(prev,[]) if role.get(g) in FIG_PART]
        pool=exclude_table_owned_ink(pool,table_bands_by_page.get(prev,[]))
        comps=ink_components(pool,ppage,max(2.0,med_lh/2)) if pool else []
        prev_els=[r for (ppi,r,k,l,cl,cr,to,sec) in elements if ppi==prev]
        orph=[c for c in comps if c.width>=med_lh and c.height>=med_lh
              and not any(_ov(c,e)>0.5*_area(c) for e in prev_els)]
        if not orph: continue
        orph.sort(key=_area,reverse=True)
        main=orph[0]                                       # largest orphan ink component is the floating figure
        band=[c for c in orph if min(c.y1,main.y1)-max(c.y0,main.y0)>0]   # include sibling panels in the same y-band
        seed=fitz.Rect(min(c.x0 for c in band),min(c.y0 for c in band),
                       max(c.x1 for c in band),max(c.y1 for c in band))&legal_rect
        if seed.width>8 and seed.height>8 and has_ink(ppage,seed):
            elements.append((prev,seed,"figure_caption",label,hCL,hCR,True,section))
            flags[:]=[f for f in flags if not(f["kind"]=="I1_missing_element" and f["page"]==cpi+1 and f["detail"].startswith(label))]
            flags.append({"page":prev+1,"kind":"cross_page_adopted","detail":f"{label} (caption p{cpi+1}, figure p{prev+1})"})

    print(f"\n=== Cropped {len(elements)} visual elements ===")
    # Refinement boundary = rendered PDF bbox of wall roles (no fixed pixel padding)
    tmask_by_page={
        pi:[make_text_wall(r,painted_wall(pi,r),pi,g,role.get(g),t)
            for g,r,t in per.get(pi,[])
            if role.get(g) not in (FIG_PART|{"other"})]
           +[make_text_wall(r,painted_wall(pi,r),pi,None,
                            "excluded_section","")
             for r in excluded_walls.get(pi,[])]
           +[record[side] for record in table_bands_by_page.get(pi,[])
             for side in ("top","bottom")]
        for pi in page_indices}
    pe={}; used=set(); final=[]; manifest=[]
    for idx,(pi,rect0,kindc,label,clb,crb,trimonly,section) in enumerate(elements):
        kind=KMAP.get(kindc,"?")
        pYtop,pYbot,pCL,pCR=page_outer_bounds.get(pi,(doc[pi].rect.y0,doc[pi].rect.y1,doc[pi].rect.x0,doc[pi].rect.x1))
        hYtop,hYbot,hCL,hCR=page_outer_hard_bounds.get(
            pi,(doc[pi].rect.y0,doc[pi].rect.y1,doc[pi].rect.x0,doc[pi].rect.x1))
        legal_rect=fitz.Rect(hCL,hYtop,hCR,hYbot)&doc[pi].rect
        rect0=fitz.Rect(rect0)&legal_rect
        if pi not in gcache: gcache[pi]=page_gray(doc[pi])
        gray=gcache[pi]
        immutable_table=immutable_table_elements.get((pi,label))
        if immutable_table is not None:
            rect=fitz.Rect(immutable_table)&legal_rect
        elif trimonly:                                          # ownership seed: trim whitespace inward only
            rect=refine_trim(rect0,gray)
        else:
            tm=tmask_by_page.get(pi,[])
            ovx=lambda r: (min(wall_logical(r).x1,rect0.x1)
                           -max(wall_logical(r).x0,rect0.x0)>0)
            ovy=lambda r: (min(wall_logical(r).y1,rect0.y1)
                           -max(wall_logical(r).y0,rect0.y0)>0)
            tl=max([wall_edge(r,"bottom",resolved_text_edge) for r in tm
                    if wall_logical(r).y1<=rect0.y0+2 and ovx(r)],
                   default=hYtop)
            bl=min([wall_edge(r,"top",resolved_text_edge) for r in tm
                    if wall_logical(r).y0>=rect0.y1-2 and ovx(r)],
                   default=hYbot)
            ll=max([wall_edge(r,"right",resolved_text_edge) for r in tm
                    if wall_logical(r).x1<=rect0.x0+2 and ovy(r)],
                   default=hCL)
            rl=min([wall_edge(r,"left",resolved_text_edge) for r in tm
                    if wall_logical(r).x0>=rect0.x1-2 and ovy(r)],
                   default=hCR)
            tl=max(tl,hYtop); bl=min(bl,hYbot)
            ll=max(ll,clb,hCL); rl=min(rl,crb,hCR)
            rect=refine_edges(rect0,gray,tl,bl,ll,rl)
        rect=fitz.Rect(rect)&legal_rect
        safe=re.sub(r"[^A-Za-z0-9]+","",label) or f"e{idx}"
        while f"p{pi+1}_{safe}" in used: safe+="_"
        used.add(f"p{pi+1}_{safe}")
        pe.setdefault(pi,[]).append((rect,kind,label))
        final.append((pi,rect,kind,label))
        fn=f"page{pi+1}_{kind}_{safe}.png"
        asset_dir="appendix_figures" if section=="appendix" else "figures"
        try: w,h=render_plain(doc[pi],rect,od/asset_dir/fn)
        except Exception: w=h=0
        manifest.append({"page":pi+1,"section":section,
                         "priority":"secondary" if section=="appendix" else "primary",
                         "kind":kind,"label":label,
                         "family":split_label(label)[0] if split_label(label) else family_name(role=kind),
                         "bbox":[round(rect.x0,2),round(rect.y0,2),round(rect.x1,2),round(rect.y1,2)],
                         "degen":False,"file":f"{asset_dir}/{fn}","render_wh":[w,h]})
        print(f"  p{pi+1} [{section:8}/{kind:8}] {label:14} [{rect.x0:.0f},{rect.y0:.0f},{rect.x1:.0f},{rect.y1:.0f}] {w}x{h}")

    rendered_edge_doc={
        "schema_version":"dp-p2p.rendered-text-wall-edges.v1",
        "policy":"semantic logical bbox decides ownership; cached 4x raster adjusts only an observed outward wall edge; no fixed padding",
        "render_scale":RZ,
        "threshold":NEAR_WHITE,
        "max_outward_search":round(0.5*med_lh,4),
        "event_count":len(rendered_edge_events),
        "changed_edge_count":sum(
            abs(event.get("rendered_edge",0)-event.get("logical_edge",0))>1e-6
            for event in rendered_edge_events),
        "open_component_rejection_count":sum(
            event.get("rejected_open_components",0)
            for event in rendered_edge_events),
        "events":rendered_edge_events,
    }
    (od/"rendered_text_wall_edges.json").write_text(
        json.dumps(rendered_edge_doc,ensure_ascii=False,indent=2),
        encoding="utf-8")

    # ── R2 invariants ──
    capbox={}   # (pi,label)->cbox for I3 check
    for pi in page_indices:
        for g,r,t in per.get(pi,[]):
            if role.get(g) in ("figure_caption","table_title") and num.get(g) is not None:
                k=(pi,f"{families.get(g,family_name(role=role[g]))} {num[g]}")
                capbox[k]=(capbox[k]|fitz.Rect(r)) if k in capbox else fitz.Rect(r)
    bypage={}
    for pi,rect,kind,label in final: bypage.setdefault(pi,[]).append((rect,kind,label))
    for pi,els in bypage.items():
        pYtop,pYbot,pCL,pCR=page_outer_bounds.get(pi,(doc[pi].rect.y0,doc[pi].rect.y1,doc[pi].rect.x0,doc[pi].rect.x1))
        claim_rect=fitz.Rect(pCL,pYtop,pCR,pYbot)&doc[pi].rect
        for i in range(len(els)):
            for j in range(i+1,len(els)):
                ra,_,la=els[i]; rb,_,lb=els[j]
                if _ov(ra,rb) > 0.01*min(_area(ra),_area(rb)):
                    flags.append({"page":pi+1,"kind":"I2_overlap","detail":f"{la} × {lb}"})
        for (k_pi,k_lab),cb in capbox.items():
            if k_pi!=pi: continue
            for rect,kind,label in els:
                if label!=k_lab and _ov(rect,cb) > 0.5*_area(cb):
                    flags.append({"page":pi+1,"kind":"I3_swallows_caption","detail":f"{label} overlaps caption {k_lab}"})
        try: infos=doc[pi].get_image_info()
        except Exception: infos=[]
        for info in infos:
            rr=fitz.Rect(info["bbox"])
            if (rr&claim_rect).is_empty: continue
            if min(rr.width,rr.height) <= med_lh: continue        # ignore small inline icons
            if not any(_ov(rr,rect) > 0.5*_area(rr) for rect,_,_ in els):
                flags.append({"page":pi+1,"kind":"I4_raster_uncovered","detail":f"raster[{rr.x0:.0f},{rr.y0:.0f},{rr.x1:.0f},{rr.y1:.0f}]"})
    for m in manifest:                                   # mark manifest entries referenced by page flags
        mk=[fl["kind"] for fl in flags if fl["page"]==m["page"] and m["label"] in fl.get("detail","")]
        m["flags"]=mk; m["quality"]="flagged" if any(k.startswith("I") for k in mk) else "ok"
    (od/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    (od/"report.json").write_text(json.dumps({"slug":slug,"flags":flags},ensure_ascii=False,indent=2))
    print(f"INVARIANTS: {len(flags)} flags" + (f" → {od/'report.json'}" if flags else " ✓"))
    for f in flags[:12]: print("   ",f)

    for pi in page_indices:
        page=doc[pi]; pix=page.get_pixmap(matrix=fitz.Matrix(Z,Z),alpha=False)
        img=Image.frombytes("RGB",(pix.width,pix.height),pix.samples); dr=ImageDraw.Draw(img)
        pYtop,pYbot,pCL,pCR=page_outer_bounds.get(pi,(doc[pi].rect.y0,doc[pi].rect.y1,doc[pi].rect.x0,doc[pi].rect.x1))
        dr.rectangle([pCL*Z,pYtop*Z,pCR*Z,pYbot*Z],outline=(20,100,230),width=3)
        for rect,kind,label in pe.get(pi,[]):
            # ``pe`` intentionally contains pre-filter diagnostic candidates; a
            # rejected candidate may have a reversed/empty rectangle. It is not
            # a crop result and must not make optional overlay rendering fail.
            if rect.x1<rect.x0 or rect.y1<rect.y0 or rect.is_empty:
                continue
            dr.rectangle([rect.x0*Z,rect.y0*Z,rect.x1*Z,rect.y1*Z],outline=(230,20,20),width=4)
            dr.text((rect.x0*Z+3,rect.y0*Z+3),f"{kind}:{label}",fill=(230,20,20),font=FONT)
        img.save(od/"overlay"/f"page{pi+1}.png")
    print(f"Outputs: {od}")

if __name__=="__main__": main()
