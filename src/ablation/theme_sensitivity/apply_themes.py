#!/usr/bin/env python3
"""Generate 8 visual theme variants (T1-T8) for the 20-paper VASP (GPT-5.6-Sol) ablation set.

For Reviewer P6ve Weakness 1 (Visual Theme Ablation on Visual Presentation Scores):
- Holds content, asset extraction (SemParser), narrative organization, and spatial coordinates
  fixed to VASP (GPT-5.6-Sol) (`pi-full100-sol-20260925`).
- Varies the 6 visual styling dimensions across 8 themes:
  * t1_paper2poster   : Paper2Poster style (Royal Blue banner, blue pill headers, blue boxed panels)
  * t2_posterforest   : PosterForest style (Deep Navy & Slate multi-card hierarchy, dark navy callouts)
  * t3_postermeld     : PosterMELD style (Warm Terracotta/Amber academic banner, cream canvas)
  * t4_postergen      : PosterGen style (Forest Green & Sage cards, left-accent green strips, soft shadows)
  * t5_researchstudio : ResearchStudio style (Minimalist Swiss Charcoal/Zinc hairline Bento grid)
  * t6_p2p            : P2P style (Editorial Crimson/Burgundy & Gold, serif headings, open flow)
  * t7_monochrome     : Monochrome Wireframe control (Pure achromatic grayscale, 1px sharp wireframe)
  * t8_beamer         : LaTeX Beamer Classic (Madrid/Boadilla Royal Blue rounded title bars, classical serif)
- Preserves box-model geometry (zero added border widths or padding; all frames/bars use inset box-shadow)
  and verifies nodeRects, hardOverflow, and outside against t0_original.
"""

from __future__ import annotations

import argparse
import colorsys
import json
import re
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SOURCE_EXP = DEFAULT_REPO_ROOT / "results" / "exp" / "pi-full100-sol-20260925"
DEFAULT_K1_PAPERS = DEFAULT_REPO_ROOT / "results" / "exp" / "pi-headroom10-20260925" / "papers.json"
DEFAULT_E1_PAPERS = DEFAULT_REPO_ROOT / "results" / "exp" / "pi-headroom10-final-20260925" / "ext" / "papers.json"
DEFAULT_T0_K1_VB = DEFAULT_REPO_ROOT / "results" / "exp" / "pi-headroom10-final-20260925" / "evaluation" / "vb" / "results" / "k1"
DEFAULT_T0_E1_VB = DEFAULT_REPO_ROOT / "results" / "exp" / "pi-headroom10-final-20260925" / "eval_ext" / "vb" / "results" / "e1"

CANVAS_WIDTH = 2544
CANVAS_HEIGHT = 1696

SANS_STACK = 'Inter, "Liberation Sans", "Nimbus Sans", "DejaVu Sans", "Helvetica Neue", Arial, sans-serif'
GEOM_SANS_STACK = '"Liberation Sans", Inter, "Nimbus Sans", "DejaVu Sans", "Helvetica Neue", Arial, sans-serif'
MONO_SANS_STACK = '"Liberation Sans", "DejaVu Sans", Arial, Helvetica, sans-serif'
SERIF_STACK = '"Liberation Serif", "Nimbus Roman", "Century Schoolbook L", "DejaVu Serif", Georgia, "Times New Roman", serif'


@dataclass(frozen=True)
class ThemeSpec:
    theme_id: str
    label: str
    category: str
    description: str
    outer_bg: str
    canvas_bg: str
    card_bg: str
    tint_bg: str
    border_color: str
    ink_color: str
    muted_text: str
    primary_color: str
    primary_dark: str
    accent_color: str
    table_header_bg: str
    table_stripe_bg: str
    keep_dark_surfaces: bool
    body_font: str
    heading_font: str
    heading_letter_spacing: str
    extra_css: str


# Note on CSS specificity and box-model invariance:
# 1. All D5 generic node rules wrap exclusions in `:where(...)` and exclude
#    `:not([data-takeaway-block="1"]):not([data-dark-surface="1"])` so D6 and dark-block
#    overrides always win cleanly.
# 2. No override adds `border-width` or `padding`; all accent bars, top/bottom rules, and
#    card frames are rendered via `box-shadow: inset ...` or `border-color`, guaranteeing
#    0px change to element content boxes.
THEME_SPECS: tuple[ThemeSpec, ...] = (
    ThemeSpec(
        theme_id="t1_paper2poster",
        label="Paper2Poster Theme",
        category="baseline_theme",
        description="Classic blue academic banner, solid blue section bars, 2px blue boxed panels, sans-serif.",
        outer_bg="#dbe7f6",
        canvas_bg="#f4f8fd",
        card_bg="#ffffff",
        tint_bg="#eaf2fc",
        border_color="#90b4e2",
        ink_color="#102542",
        muted_text="#3d5a80",
        primary_color="#1e5aa8",
        primary_dark="#153e75",
        accent_color="#0284c7",
        table_header_bg="#dbeafe",
        table_stripe_bg="#eff6ff",
        keep_dark_surfaces=False,
        body_font=SANS_STACK,
        heading_font=SANS_STACK,
        heading_letter_spacing="-0.02em",
        extra_css="""
/* D3: Masthead - Solid Royal-Blue Banner */
.poster .masthead {
  background: #1e5aa8 !important;
  color: #ffffff !important;
  border-color: #0f3460 !important;
  border-radius: 10px !important;
  box-shadow: -14px -10px 0 10px #1e5aa8, 14px -10px 0 10px #1e5aa8, inset 0 -4px 0 0 #0f3460 !important;
}
.poster .masthead * {
  color: #e0f2fe !important;
}
.poster .masthead h1,
.poster .masthead .title-block h1 {
  color: #ffffff !important;
}
.poster .masthead .venue,
.poster .masthead .eyebrow,
.poster .masthead .kicker {
  color: #bae6fd !important;
}
.poster .masthead .brand-cluster,
.poster .masthead [data-node-id="branding"] {
  background: #ffffff !important;
  color: #102542 !important;
  border-color: #93c5fd !important;
  border-radius: 10px !important;
}
.poster .masthead .brand-cluster *,
.poster .masthead [data-node-id="branding"] * {
  color: #102542 !important;
}
/* D4: Section Headers - Solid Blue Pill Bar */
.poster .chapter {
  background: #1e5aa8 !important;
  color: #ffffff !important;
  border-color: transparent !important;
  border-radius: 6px !important;
  box-shadow: -6px 0 0 4px #1e5aa8, 6px 0 0 4px #1e5aa8 !important;
}
.poster .chapter * {
  color: #ffffff !important;
}
.poster .chapter [data-chapter-num="1"] {
  background: #0f3460 !important;
  color: #ffffff !important;
  border-radius: 4px !important;
}
/* D5: Container Borders & Surfaces - Boxed Blue Panels */
.poster :where([data-node-id]:not(.masthead):not(.masthead *):not([data-takeaway-block="1"]):not([data-dark-surface="1"])) {
  background: #ffffff !important;
  color: #102542 !important;
  border-color: #3b82f6 !important;
  border-radius: 8px !important;
  box-shadow: inset 0 0 0 2px #3b82f6 !important;
}
/* D6: Takeaway Block - Outlined Blue Box */
.poster [data-takeaway-block="1"] {
  background: #e0f2fe !important;
  color: #102542 !important;
  border-color: #1e5aa8 !important;
  box-shadow: inset 8px 0 0 0 #1e5aa8, inset 0 0 0 2px #1e5aa8 !important;
  border-radius: 8px !important;
}
.poster [data-takeaway-block="1"] * {
  color: #102542 !important;
}
""",
    ),
    ThemeSpec(
        theme_id="t2_posterforest",
        label="PosterForest Theme",
        category="baseline_theme",
        description="Deep Navy & Slate hierarchical multi-card layout with numbered badge headers and dark navy callout blocks.",
        outer_bg="#d5dde7",
        canvas_bg="#edf2f7",
        card_bg="#ffffff",
        tint_bg="#e2e8f0",
        border_color="#94a3b8",
        ink_color="#0f172a",
        muted_text="#475569",
        primary_color="#1b365d",
        primary_dark="#0f2342",
        accent_color="#2b6cb0",
        table_header_bg="#e2e8f0",
        table_stripe_bg="#f1f5f9",
        keep_dark_surfaces=True,
        body_font=GEOM_SANS_STACK,
        heading_font=GEOM_SANS_STACK,
        heading_letter_spacing="-0.02em",
        extra_css="""
/* D3: Masthead - Light Navy-Tinted Header Card with Deep Navy Bottom Rule */
.poster .masthead {
  background: #e2e8f0 !important;
  border-color: #1b365d !important;
  border-radius: 10px !important;
  box-shadow: -12px -8px 0 8px #e2e8f0, 12px -8px 0 8px #e2e8f0, inset 0 -4px 0 0 #1b365d !important;
}
.poster .masthead h1,
.poster .masthead .title-block h1 {
  color: #0f172a !important;
}
.poster .masthead .venue,
.poster .masthead .eyebrow {
  color: #1b365d !important;
}
/* D4: Section Headers - Numbered Navy Pill Badge + Navy Title */
.poster .chapter {
  background: #dbe4f0 !important;
  color: #0f172a !important;
  border-color: #1b365d !important;
  border-radius: 8px 8px 0 0 !important;
  box-shadow: -6px 0 0 4px #dbe4f0, 6px 0 0 4px #dbe4f0, inset 0 -2px 0 0 #1b365d !important;
}
.poster .chapter * {
  color: #0f172a !important;
}
.poster .chapter [data-chapter-num="1"] {
  background: #1b365d !important;
  color: #ffffff !important;
  border-radius: 6px !important;
}
/* D5: Container Borders & Surfaces - Hierarchical Multi-Card */
.poster :where([data-node-id]:not(.masthead):not(.masthead *):not([data-dark-surface="1"]):not([data-takeaway-block="1"])) {
  background: #ffffff !important;
  color: #0f172a !important;
  border-color: #94a3b8 !important;
  border-radius: 10px !important;
  box-shadow: inset 0 0 0 1.5px #94a3b8, 0 2px 6px rgba(15, 23, 42, 0.06) !important;
}
.poster [data-dark-surface="1"] {
  background: #1b365d !important;
  color: #ffffff !important;
  border-radius: 10px !important;
}
.poster [data-dark-surface="1"] :where(h1, h2, h3, h4, p, li, div, span, b, strong, i, sub, sup) {
  color: #f8fafc !important;
}
/* D6: Takeaway Block - Solid Dark Navy Block */
.poster [data-takeaway-block="1"] {
  background: #1b365d !important;
  color: #ffffff !important;
  border-color: #38bdf8 !important;
  box-shadow: inset 8px 0 0 0 #38bdf8 !important;
  border-radius: 10px !important;
}
.poster [data-takeaway-block="1"] * {
  color: #ffffff !important;
}
""",
    ),
    ThemeSpec(
        theme_id="t3_postermeld",
        label="PosterMELD Theme",
        category="baseline_theme",
        description="Warm Terracotta/Coral & Amber palette on warm cream canvas with full-width warm section bars.",
        outer_bg="#eadcd2",
        canvas_bg="#fbf7f2",
        card_bg="#fffdf9",
        tint_bg="#f5ebe1",
        border_color="#d6b8a4",
        ink_color="#2a1810",
        muted_text="#6b4c3e",
        primary_color="#b84a39",
        primary_dark="#8f3527",
        accent_color="#d97706",
        table_header_bg="#f5e6d8",
        table_stripe_bg="#fdf3e7",
        keep_dark_surfaces=False,
        body_font=SANS_STACK,
        heading_font=SANS_STACK,
        heading_letter_spacing="-0.02em",
        extra_css="""
/* D3: Masthead - Warm Terracotta-Tinted Banner */
.poster .masthead {
  background: #f3e4d8 !important;
  border-color: #b84a39 !important;
  border-radius: 8px !important;
  box-shadow: -12px -8px 0 8px #f3e4d8, 12px -8px 0 8px #f3e4d8, inset 0 -4px 0 0 #b84a39 !important;
}
.poster .masthead h1,
.poster .masthead .title-block h1 {
  color: #2a1810 !important;
}
.poster .masthead .venue,
.poster .masthead .eyebrow {
  color: #b84a39 !important;
}
.poster .masthead .authors,
.poster .masthead .subtitle,
.poster .masthead .affiliations {
  color: #5c3a2e !important;
}
/* D4: Section Headers - Full-Width Warm Terracotta Bar */
.poster .chapter {
  background: #b84a39 !important;
  color: #ffffff !important;
  border-color: transparent !important;
  border-radius: 6px !important;
  box-shadow: -6px 0 0 4px #b84a39, 6px 0 0 4px #b84a39 !important;
}
.poster .chapter * {
  color: #ffffff !important;
}
.poster .chapter [data-chapter-num="1"] {
  background: #8f3527 !important;
  color: #fde68a !important;
  border-radius: 4px !important;
}
/* D5: Container Borders & Surfaces - Warm Outlined Cards */
.poster :where([data-node-id]:not(.masthead):not(.masthead *):not([data-takeaway-block="1"]):not([data-dark-surface="1"])) {
  background: #fffdf9 !important;
  color: #2a1810 !important;
  border-color: #d99b82 !important;
  border-radius: 8px !important;
  box-shadow: inset 0 0 0 1.5px #d99b82 !important;
}
/* D6: Takeaway Block - Warm Tinted Callout Card */
.poster [data-takeaway-block="1"] {
  background: #f7e6d9 !important;
  color: #2a1810 !important;
  border-color: #b84a39 !important;
  box-shadow: inset 8px 0 0 0 #b84a39, inset 0 0 0 1.5px #b84a39 !important;
  border-radius: 8px !important;
}
.poster [data-takeaway-block="1"] * {
  color: #2a1810 !important;
}
""",
    ),
    ThemeSpec(
        theme_id="t4_postergen",
        label="PosterGen Theme",
        category="baseline_theme",
        description="Forest Green & Sage palette, split green header, left-accent section strips, rounded soft-shadow cards.",
        outer_bg="#d5e6de",
        canvas_bg="#f3f8f5",
        card_bg="#ffffff",
        tint_bg="#e6f2ec",
        border_color="#9cc5b3",
        ink_color="#12261e",
        muted_text="#3f6254",
        primary_color="#1f6f54",
        primary_dark="#144d3a",
        accent_color="#d97706",
        table_header_bg="#dff0e8",
        table_stripe_bg="#edf7f2",
        keep_dark_surfaces=False,
        body_font=SANS_STACK,
        heading_font=SANS_STACK,
        heading_letter_spacing="-0.02em",
        extra_css="""
/* D3: Masthead - Split Forest Green Header with Warm Amber Accent Rule */
.poster .masthead {
  background: #1f6f54 !important;
  color: #ffffff !important;
  border-color: #d97706 !important;
  border-radius: 10px !important;
  box-shadow: -14px -10px 0 10px #1f6f54, 14px -10px 0 10px #1f6f54, inset 0 -5px 0 0 #d97706 !important;
}
.poster .masthead * {
  color: #d1fae5 !important;
}
.poster .masthead h1,
.poster .masthead .title-block h1 {
  color: #ffffff !important;
}
.poster .masthead .venue,
.poster .masthead .eyebrow,
.poster .masthead .kicker {
  color: #fde68a !important;
}
.poster .masthead .brand-cluster,
.poster .masthead [data-node-id="branding"] {
  background: #ffffff !important;
  color: #12261e !important;
  border-radius: 10px !important;
}
.poster .masthead .brand-cluster *,
.poster .masthead [data-node-id="branding"] * {
  color: #12261e !important;
}
/* D4: Section Headers - Left-Accent Green Strip */
.poster .chapter {
  background: #e3f1ea !important;
  color: #12261e !important;
  border-color: #9cc5b3 !important;
  border-radius: 0 8px 8px 0 !important;
  box-shadow: -6px 0 0 4px #e3f1ea, inset 7px 0 0 0 #1f6f54, inset 0 -1px 0 0 #9cc5b3 !important;
}
.poster .chapter * {
  color: #144d3a !important;
}
.poster .chapter [data-chapter-num="1"] {
  background: #1f6f54 !important;
  color: #ffffff !important;
  border-radius: 5px !important;
}
/* D5: Container Borders & Surfaces - Rounded Soft-Shadow Cards */
.poster :where([data-node-id]:not(.masthead):not(.masthead *):not([data-takeaway-block="1"]):not([data-dark-surface="1"])) {
  background: #ffffff !important;
  color: #12261e !important;
  border-color: #a7d7c5 !important;
  border-radius: 12px !important;
  box-shadow: inset 0 0 0 1.5px #a7d7c5, 0 4px 10px rgba(20, 77, 58, 0.07) !important;
}
/* D6: Takeaway Block - Soft Green Highlight Card */
.poster [data-takeaway-block="1"] {
  background: #dff0e8 !important;
  color: #12261e !important;
  border-color: #1f6f54 !important;
  box-shadow: inset 8px 0 0 0 #1f6f54, inset 0 0 0 1.5px #1f6f54 !important;
  border-radius: 12px !important;
}
.poster [data-takeaway-block="1"] * {
  color: #12261e !important;
}
""",
    ),
    ThemeSpec(
        theme_id="t5_researchstudio",
        label="ResearchStudio Theme",
        category="baseline_theme",
        description="Minimalist Swiss Neo-Grotesque Charcoal & Zinc Bento grid with 1px hairline borders on pure white.",
        outer_bg="#e4e4e7",
        canvas_bg="#fafafa",
        card_bg="#ffffff",
        tint_bg="#f4f4f5",
        border_color="#d4d4d8",
        ink_color="#18181b",
        muted_text="#52525b",
        primary_color="#27272a",
        primary_dark="#09090b",
        accent_color="#2563eb",
        table_header_bg="#f4f4f5",
        table_stripe_bg="#fafafa",
        keep_dark_surfaces=False,
        body_font=SANS_STACK,
        heading_font=SANS_STACK,
        heading_letter_spacing="-0.025em",
        extra_css="""
/* D3: Masthead - Minimalist Open White Header with Crisp Black Rule */
.poster .masthead {
  background: #ffffff !important;
  border-color: #18181b !important;
  border-radius: 0 !important;
  box-shadow: inset 0 -2px 0 0 #18181b !important;
}
.poster .masthead h1,
.poster .masthead .title-block h1 {
  color: #09090b !important;
}
.poster .masthead .venue,
.poster .masthead .eyebrow {
  color: #2563eb !important;
}
/* D4: Section Headers - Minimalist Micro-Badge + 1.5px Hairline */
.poster .chapter {
  background: transparent !important;
  color: #18181b !important;
  border-color: #18181b !important;
  border-radius: 0 !important;
  box-shadow: inset 0 -1.5px 0 0 #18181b !important;
}
.poster .chapter * {
  color: #18181b !important;
}
.poster .chapter [data-chapter-num="1"] {
  background: #18181b !important;
  color: #ffffff !important;
  border-radius: 3px !important;
}
/* D5: Container Borders & Surfaces - 1px Hairline Bento Grid */
.poster :where([data-node-id]:not(.masthead):not(.masthead *):not([data-takeaway-block="1"]):not([data-dark-surface="1"])) {
  background: #ffffff !important;
  color: #18181b !important;
  border-color: #d4d4d8 !important;
  border-radius: 6px !important;
  box-shadow: inset 0 0 0 1px #d4d4d8 !important;
}
/* D6: Takeaway Block - Clean Bento Summary Cell */
.poster [data-takeaway-block="1"] {
  background: #f4f4f5 !important;
  color: #18181b !important;
  border-color: #18181b !important;
  box-shadow: inset 6px 0 0 0 #18181b, inset 0 0 0 1.5px #18181b !important;
  border-radius: 6px !important;
}
.poster [data-takeaway-block="1"] * {
  color: #18181b !important;
}
""",
    ),
    ThemeSpec(
        theme_id="t6_p2p",
        label="P2P Theme",
        category="baseline_theme",
        description="Editorial Crimson/Burgundy & Warm Gold palette, serif headings, open borderless flow.",
        outer_bg="#e8e0d8",
        canvas_bg="#faf7f2",
        card_bg="#ffffff",
        tint_bg="#f5eee6",
        border_color="#d9cbbd",
        ink_color="#1f1618",
        muted_text="#5f4b4e",
        primary_color="#8c1d40",
        primary_dark="#63122b",
        accent_color="#b4831e",
        table_header_bg="#f4e8ec",
        table_stripe_bg="#fbf6f7",
        keep_dark_surfaces=False,
        body_font=SANS_STACK,
        heading_font=SERIF_STACK,
        heading_letter_spacing="-0.025em",
        extra_css=f"""
/* D3: Masthead - Editorial Top-Accent Rule & Open Serif Title */
.poster .masthead {{
  background: transparent !important;
  border-color: #8c1d40 !important;
  border-radius: 0 !important;
  box-shadow: 0 -10px 0 0 #faf7f2, 0 -17px 0 0 #8c1d40, inset 0 -2px 0 0 #8c1d40 !important;
}}
.poster .masthead h1,
.poster .masthead .title-block h1 {{
  color: #1f1618 !important;
  font-family: {SERIF_STACK} !important;
}}
.poster .masthead .venue,
.poster .masthead .eyebrow {{
  color: #8c1d40 !important;
}}
/* D4: Section Headers - Crimson Underline + Editorial Serif Header */
.poster .chapter {{
  background: transparent !important;
  color: #1f1618 !important;
  border-color: #8c1d40 !important;
  border-radius: 0 !important;
  box-shadow: inset 0 -2.5px 0 0 #8c1d40 !important;
  font-family: {SERIF_STACK} !important;
}}
.poster .chapter * {{
  color: #1f1618 !important;
  font-family: {SERIF_STACK} !important;
}}
.poster .chapter [data-chapter-num="1"] {{
  background: transparent !important;
  color: #8c1d40 !important;
  font-weight: 900 !important;
}}
/* D5: Container Borders & Surfaces - Borderless Open Flow with Subtle Divider Rules */
.poster :where([data-node-id]:not(.masthead):not(.masthead *):not([data-takeaway-block="1"]):not([data-dark-surface="1"])) {{
  background: transparent !important;
  color: #1f1618 !important;
  border-color: #d9cbbd !important;
  border-radius: 0 !important;
  box-shadow: inset 0 -1px 0 0 #d9cbbd !important;
}}
.poster figure[data-node-id],
.poster [data-node-kind="figure"] {{
  background: #ffffff !important;
  box-shadow: inset 0 0 0 1px #d9cbbd !important;
}}
/* D6: Takeaway Block - Crimson Left-Bar Editorial Callout */
.poster [data-takeaway-block="1"] {{
  background: #f5e9ec !important;
  color: #1f1618 !important;
  border-color: #8c1d40 !important;
  border-radius: 0 !important;
  box-shadow: inset 8px 0 0 0 #8c1d40, inset 0 0 0 1px #8c1d40 !important;
}}
.poster [data-takeaway-block="1"] * {{
  color: #1f1618 !important;
}}
""",
    ),
    ThemeSpec(
        theme_id="t7_monochrome",
        label="Monochrome Wireframe",
        category="ablation_control",
        description="Pure achromatic grayscale palette (zero hue saturation), 1px rectangular wireframe boxes, zero rounded corners or shadows.",
        outer_bg="#dcdcdc",
        canvas_bg="#f7f7f7",
        card_bg="#ffffff",
        tint_bg="#eeeeee",
        border_color="#888888",
        ink_color="#111111",
        muted_text="#4a4a4a",
        primary_color="#222222",
        primary_dark="#000000",
        accent_color="#333333",
        table_header_bg="#e5e5e5",
        table_stripe_bg="#f2f2f2",
        keep_dark_surfaces=False,
        body_font=MONO_SANS_STACK,
        heading_font=MONO_SANS_STACK,
        heading_letter_spacing="-0.02em",
        extra_css="""
/* D3: Masthead - Plain White with 2px Black Rule */
.poster *,
.poster *::before,
.poster *::after {
  border-radius: 0 !important;
  box-shadow: none !important;
}
.poster .masthead {
  background: #ffffff !important;
  border-color: #111111 !important;
  box-shadow: inset 0 -2px 0 0 #111111 !important;
}
.poster .masthead * {
  color: #111111 !important;
}
/* D4: Section Headers - Plain Gray Wireframe Bar */
.poster .chapter {
  background: #e8e8e8 !important;
  color: #111111 !important;
  border-color: #111111 !important;
  box-shadow: inset 0 1px 0 0 #111111, inset 0 -2px 0 0 #111111 !important;
}
.poster .chapter [data-chapter-num="1"],
.poster .chapter [data-chapter-title="1"],
.poster .chapter * {
  background: transparent !important;
  color: #111111 !important;
}
/* D5: Container Borders & Surfaces - 1px Sharp Rectangular Wireframe */
.poster :where([data-node-id]:not(.masthead):not(.masthead *):not([data-takeaway-block="1"]):not([data-dark-surface="1"])) {
  background: #ffffff !important;
  color: #111111 !important;
  border-color: #666666 !important;
  box-shadow: inset 0 0 0 1px #666666 !important;
}
/* D6: Takeaway Block - Plain Gray Box with 2px Black Outline */
.poster [data-takeaway-block="1"] {
  background: #eaeaea !important;
  color: #111111 !important;
  border-color: #111111 !important;
  box-shadow: inset 6px 0 0 0 #111111, inset 0 0 0 2px #111111 !important;
}
.poster [data-takeaway-block="1"] * {
  color: #111111 !important;
}
""",
    ),
    ThemeSpec(
        theme_id="t8_beamer",
        label="LaTeX Beamer Classic",
        category="academic_traditional",
        description="Classic LaTeX Beamer Madrid/Boadilla Royal Blue (#003366) rounded header blocks and classical serif typography.",
        outer_bg="#d8e0ea",
        canvas_bg="#f5f7fa",
        card_bg="#ffffff",
        tint_bg="#e8eef5",
        border_color="#7b93b0",
        ink_color="#0a192f",
        muted_text="#3b526f",
        primary_color="#003366",
        primary_dark="#002244",
        accent_color="#8b1e24",
        table_header_bg="#dce6f2",
        table_stripe_bg="#eef3f9",
        keep_dark_surfaces=False,
        body_font=SERIF_STACK,
        heading_font=SERIF_STACK,
        heading_letter_spacing="-0.025em",
        extra_css="""
/* D3: Masthead - Classic Beamer Madrid Rounded Blue Title Box */
.poster .masthead {
  background: #003366 !important;
  color: #ffffff !important;
  border-color: #001a33 !important;
  border-radius: 12px !important;
  box-shadow: -14px -10px 0 10px #003366, 14px -10px 0 10px #003366, inset 0 -4px 0 0 #001a33 !important;
}
.poster .masthead * {
  color: #e2e8f0 !important;
}
.poster .masthead h1,
.poster .masthead .title-block h1 {
  color: #ffffff !important;
}
.poster .masthead .venue,
.poster .masthead .eyebrow,
.poster .masthead .kicker {
  color: #ffd166 !important;
}
.poster .masthead .brand-cluster,
.poster .masthead [data-node-id="branding"] {
  background: #ffffff !important;
  color: #0a192f !important;
  border-radius: 8px !important;
}
.poster .masthead .brand-cluster *,
.poster .masthead [data-node-id="branding"] * {
  color: #0a192f !important;
}
/* D4: Section Headers - Beamer Block Title Bar */
.poster .chapter {
  background: #204a77 !important;
  color: #ffffff !important;
  border-color: #003366 !important;
  border-radius: 8px 8px 0 0 !important;
  box-shadow: -6px 0 0 4px #204a77, 6px 0 0 4px #204a77, inset 0 -2px 0 0 #003366 !important;
}
.poster .chapter * {
  color: #ffffff !important;
}
.poster .chapter [data-chapter-num="1"] {
  background: #003366 !important;
  color: #ffd166 !important;
  border-radius: 5px !important;
}
/* D5: Container Borders & Surfaces - Beamer Framed Block */
.poster :where([data-node-id]:not(.masthead):not(.masthead *):not([data-takeaway-block="1"]):not([data-dark-surface="1"])) {
  background: #ffffff !important;
  color: #0a192f !important;
  border-color: #204a77 !important;
  border-radius: 8px !important;
  box-shadow: inset 0 0 0 1.5px #204a77 !important;
}
/* D6: Takeaway Block - Beamer Alert/Example Block */
.poster [data-takeaway-block="1"] {
  background: #e6eef7 !important;
  color: #0a192f !important;
  border-color: #003366 !important;
  box-shadow: inset 0 6px 0 0 #003366, inset 0 0 0 1.5px #003366 !important;
  border-radius: 8px !important;
}
.poster [data-takeaway-block="1"] * {
  color: #0a192f !important;
}
""",
    ),
)

THEME_BY_ID: dict[str, ThemeSpec] = {spec.theme_id: spec for spec in THEME_SPECS}

HEX_COLOR_RE = re.compile(r"#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6})\b")
RGB_COLOR_RE = re.compile(
    r"rgba?\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})(?:\s*,\s*([0-9.]+))?\s*\)"
)
CHAPTER_NUM_RE = re.compile(r"^\s*0?\d+\.?\s*$")
LEGEND_SELECTOR_RE = re.compile(r"(?:^|[\s,>+~])(?:\.legend\b|\.violet\b|\.green\b)", re.I)


def parse_hex_rgb(token: str) -> tuple[int, int, int]:
    raw = token.lstrip("#")
    if len(raw) == 3:
        raw = "".join(ch * 2 for ch in raw)
    return int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16)


def relative_luminance(r: int, g: int, b: int) -> float:
    return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0


def map_rgb_to_theme(r: int, g: int, b: int, theme: ThemeSpec) -> str:
    lum = relative_luminance(r, g, b)
    h, _l, s = colorsys.rgb_to_hls(r / 255.0, g / 255.0, b / 255.0)
    hue_deg = (h * 360.0) % 360.0

    if lum >= 0.965:
        return theme.card_bg
    if lum >= 0.91:
        return theme.canvas_bg
    if lum >= 0.82:
        return theme.tint_bg
    # Check chromatic saturation BEFORE mid-light border luminance so bright saturated
    # accents like amber (#f1a323, L=0.67, S=0.88) map to accent_color, not border_color!
    if s >= 0.32 and 0.16 <= lum < 0.82:
        if (12.0 <= hue_deg <= 68.0) or (hue_deg >= 335.0):
            return theme.accent_color
        return theme.primary_color
    if lum >= 0.65:
        return theme.border_color
    if lum >= 0.34:
        return theme.muted_text
    if lum >= 0.18:
        return theme.primary_dark if s >= 0.25 else theme.ink_color
    return theme.ink_color


def remap_css_colors(css_text: str, theme: ThemeSpec) -> str:
    """Remap colors per CSS rule block while preserving figure-legend swatch rules."""
    def replace_hex(match: re.Match[str]) -> str:
        r, g, b = parse_hex_rgb(match.group(0))
        return map_rgb_to_theme(r, g, b, theme)

    def replace_rgb(match: re.Match[str]) -> str:
        r = min(255, int(match.group(1)))
        g = min(255, int(match.group(2)))
        b = min(255, int(match.group(3)))
        alpha = match.group(4)
        mapped = map_rgb_to_theme(r, g, b, theme)
        if alpha is None:
            return mapped
        mr, mg, mb = parse_hex_rgb(mapped)
        return f"rgba({mr}, {mg}, {mb}, {alpha})"

    def remap_declarations(decls: str) -> str:
        out = HEX_COLOR_RE.sub(replace_hex, decls)
        out = RGB_COLOR_RE.sub(replace_rgb, out)
        # Also normalize named 'white' / 'black' in background/color declarations
        out = re.sub(
            r"(background(?:-color)?\s*:\s*)white\b",
            rf"\1{theme.card_bg}",
            out,
            flags=re.I,
        )
        return out

    # Process rule by rule so legend swatch rules (.legend, .violet, .green) stay intact
    # except in monochrome control theme where everything is grayscale.
    result: list[str] = []
    pos = 0
    for match in re.finditer(r"([^{}]+)\{([^{}]*)\}", css_text):
        result.append(css_text[pos : match.start()])
        sel = match.group(1)
        body = match.group(2)
        if theme.theme_id != "t7_monochrome" and LEGEND_SELECTOR_RE.search(sel):
            result.append(match.group(0))
        else:
            result.append(f"{sel}{{{remap_declarations(body)}}}")
        pos = match.end()
    result.append(css_text[pos:])
    return "".join(result)


def find_dark_background_selectors(css_text: str) -> list[str]:
    """Identify CSS selectors in original CSS that have a dark surface or white foreground text."""
    selectors: list[str] = []
    for block in re.finditer(r"([^{}@]+)\{([^{}]+)\}", css_text):
        sel_group = block.group(1).strip()
        props = block.group(2)
        is_dark_block = False

        for bg_match in re.finditer(
            r"background(?:-color)?\s*:\s*([^;]+)",
            props,
            flags=re.I,
        ):
            val = bg_match.group(1)
            hex_m = HEX_COLOR_RE.search(val)
            rgb_m = RGB_COLOR_RE.search(val)
            if hex_m:
                r, g, b = parse_hex_rgb(hex_m.group(0))
                if relative_luminance(r, g, b) < 0.48:
                    is_dark_block = True
            elif rgb_m:
                r, g, b = int(rgb_m.group(1)), int(rgb_m.group(2)), int(rgb_m.group(3))
                if relative_luminance(r, g, b) < 0.48:
                    is_dark_block = True
            elif re.search(r"\b(?:black|navy)\b", val, re.I):
                is_dark_block = True

        if re.search(r"(?:^|;)\s*color\s*:\s*(?:#fff(?:fff)?|white)\b", props, re.I):
            is_dark_block = True

        if is_dark_block:
            for part in sel_group.split(","):
                part = part.strip()
                if not part or ":" in part or part.startswith("@"):
                    continue
                if part in {".masthead", ".chapter", "th"} or ".chapter" in part or ".masthead" in part:
                    continue
                selectors.append(part)
    return sorted(set(selectors))


def annotate_html_hooks(
    html_text: str,
    design_state: dict[str, Any],
    dark_selectors: list[str],
    theme_id: str,
) -> str:
    """Add semantic data-* hooks to poster.html using pure stdlib regex (no bs4 re-serialization)."""
    if "class=\"poster\"" not in html_text and "class='poster'" not in html_text:
        raise ValueError("poster.html is missing .poster root")

    # 1. Add data-theme to .poster root tag
    html_out = re.sub(
        r"(<[a-zA-Z0-9]+\b[^>]*\bclass\s*=\s*['\"][^'\"]*\bposter\b[^'\"]*['\"][^>]*)(>)",
        rf'\1 data-theme="{theme_id}"\2',
        html_text,
        count=1,
    )

    # 2. Map data-node-id -> data-node-kind and data-takeaway-block="1"
    node_kinds: dict[str, str] = {}
    for node in design_state.get("nodes") or []:
        nid = str(node.get("id") or "")
        kind = str(node.get("kind") or "")
        if nid and kind:
            node_kinds[nid] = kind

    has_takeaway = False

    def enrich_node_tag(match: re.Match[str]) -> str:
        nonlocal has_takeaway
        tag_open = match.group(0)
        nid = match.group(1)
        kind = node_kinds.get(nid, "")
        extra_attrs = ""
        if kind and "data-node-kind=" not in tag_open:
            extra_attrs += f' data-node-kind="{kind}"'
        cls_m = re.search(r"\bclass\s*=\s*['\"]([^'\"]*)['\"]", tag_open)
        classes = (cls_m.group(1) if cls_m else "").lower()
        if kind == "conclusion" or nid in {"takeaway", "conclusion"} or "takeaway" in classes:
            if "data-takeaway-block=" not in tag_open:
                extra_attrs += ' data-takeaway-block="1"'
            has_takeaway = True
        if not extra_attrs:
            return tag_open
        return tag_open[:-1] + extra_attrs + ">"

    html_out = re.sub(
        r"<[a-zA-Z0-9]+\b[^>]*\bdata-node-id\s*=\s*['\"]([^'\"]+)['\"][^>]*>",
        enrich_node_tag,
        html_out,
    )

    if not has_takeaway:
        html_out = re.sub(
            r"(<[a-zA-Z0-9]+\b[^>]*\bclass\s*=\s*['\"][^'\"]*\btakeaway(?:-box)?\b[^'\"]*['\"][^>]*)(>)",
            r'\1 data-takeaway-block="1"\2',
            html_out,
            count=1,
        )

    # 3. Tag dark surface classes with data-dark-surface="1" (when not takeaway)
    dark_class_names = {
        sel.lstrip(".")
        for sel in dark_selectors
        if re.fullmatch(r"\.[a-zA-Z0-9_-]+", sel)
    }
    if dark_class_names:
        def enrich_dark_tag(match: re.Match[str]) -> str:
            tag_open = match.group(0)
            if "data-takeaway-block=" in tag_open or "data-dark-surface=" in tag_open:
                return tag_open
            cls_tokens = set(match.group(1).split())
            if cls_tokens & dark_class_names:
                return tag_open[:-1] + ' data-dark-surface="1">'
            return tag_open

        html_out = re.sub(
            r"<[a-zA-Z0-9]+\b[^>]*\bclass\s*=\s*['\"]([^'\"]+)['\"][^>]*>",
            enrich_dark_tag,
            html_out,
        )

    # 4. Tag the numeric badge element inside .chapter with data-chapter-num="1"
    #    (Never wrap bare text nodes in <span>, preserving p2's .chapter span layout!)
    html_out = re.sub(
        r"(<(?:div|header|h2|section)\b[^>]*\bclass\s*=\s*['\"][^'\"]*\bchapter\b[^'\"]*['\"][^>]*>\s*<(?:b|span|strong)\b)([^>]*>\s*0?\d+\.?\s*</(?:b|span|strong)>)",
        r'\1 data-chapter-num="1"\2',
        html_out,
    )

    return html_out


def build_themed_css(
    original_css: str,
    theme: ThemeSpec,
    dark_selectors: list[str],
    font_scale: float = 1.0,
) -> str:
    remapped = remap_css_colors(original_css, theme)
    dark_reset_css = ""
    if not theme.keep_dark_surfaces:
        sel_list = [f".poster {sel}" for sel in dark_selectors] + ['.poster [data-dark-surface="1"]']
        child_list = [f".poster {sel} *" for sel in dark_selectors] + ['.poster [data-dark-surface="1"] *']
        joined = ", ".join(sel_list)
        joined_children = ", ".join(child_list)
        dark_reset_css = f"""
/* Light-surface normalization for originally dark callout blocks and nested chips */
{joined} {{
  background: {theme.tint_bg} !important;
  color: {theme.ink_color} !important;
  border-color: {theme.primary_color} !important;
  box-shadow: inset 0 0 0 1.5px {theme.primary_color} !important;
}}
{joined_children} {{
  background-color: transparent !important;
  color: {theme.ink_color} !important;
}}
"""
    scale_css = ""
    if font_scale < 0.999:
        scale_css = """
/* Minor font-metric tightness compensation to prevent line-wrap overflow on Arial-authored posters */
html, body, .poster, .poster h1, .poster h2, .poster h3, .poster h4, .poster .chapter, .poster .chapter [data-chapter-title="1"] {
  font-family: "Liberation Sans", Arial, "Helvetica Neue", sans-serif !important;
}
"""
    base_override = f"""
/* === VASP THEME ABLATION OVERRIDE: {theme.theme_id} ({theme.label}) === */
html, body {{
  background: {theme.outer_bg} !important;
  color: {theme.ink_color} !important;
  font-family: {theme.body_font} !important;
}}
.poster {{
  background: {theme.canvas_bg} !important;
  color: {theme.ink_color} !important;
  font-family: {theme.body_font} !important;
}}
.poster h1, .poster h2, .poster h3, .poster h4, .poster .chapter, .poster .chapter [data-chapter-title="1"] {{
  font-family: {theme.heading_font} !important;
  letter-spacing: {theme.heading_letter_spacing} !important;
}}
.poster table {{
  background: {theme.card_bg} !important;
  border-top-color: {theme.primary_color} !important;
  border-bottom-color: {theme.primary_color} !important;
}}
.poster th {{
  background: {theme.table_header_bg} !important;
  color: {theme.ink_color} !important;
}}
.poster td {{
  border-bottom-color: {theme.border_color} !important;
  color: {theme.ink_color} !important;
}}
.poster tr.proposed td, .poster tr.highlight td {{
  background: {theme.tint_bg} !important;
  color: {theme.ink_color} !important;
}}
{dark_reset_css}
{theme.extra_css}
{scale_css}
"""
    return remapped + "\n" + base_override


def load_ablation_papers(k1_path: Path, e1_path: Path, limit: int | None = None) -> list[dict[str, str]]:
    k1 = json.loads(k1_path.read_text(encoding="utf-8"))
    e1 = json.loads(e1_path.read_text(encoding="utf-8"))
    papers: list[dict[str, str]] = []
    for title in k1:
        papers.append({"title": str(title), "subset": "k1"})
    for title in e1:
        papers.append({"title": str(title), "subset": "e1"})
    if len(papers) != 20 or len({p["title"] for p in papers}) != 20:
        raise ValueError(f"Expected 20 unique ablation papers, got {len(papers)}")
    if limit is not None:
        if limit < 1 or limit > len(papers):
            raise ValueError(f"--limit must be between 1 and {len(papers)}, got {limit}")
        papers = papers[:limit]
    return papers


def copy_assets_tree(src_assets: Path, dst_assets: Path) -> None:
    if dst_assets.exists() or dst_assets.is_symlink():
        if dst_assets.is_symlink() or dst_assets.is_file():
            dst_assets.unlink()
        else:
            shutil.rmtree(dst_assets)
    shutil.copytree(src_assets, dst_assets, symlinks=False)


def compute_node_rect_drift(
    t0_rects: dict[str, dict[str, float]],
    themed_rects: dict[str, dict[str, float]],
) -> float:
    max_drift = 0.0
    for nid, r0 in t0_rects.items():
        r1 = themed_rects.get(nid)
        if not r1:
            continue
        for key in ("x", "y", "w", "h"):
            drift = abs(float(r1.get(key, 0.0)) - float(r0.get(key, 0.0)))
            if drift > max_drift:
                max_drift = drift
    return round(max_drift, 2)


def render_one_theme_paper(task: dict[str, Any]) -> dict[str, Any]:
    repo_root = Path(task["repo_root"])
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from src.viscomposer.core.render_html import render_bundle  # type: ignore

    paper_title: str = task["title"]
    subset: str = task["subset"]
    theme_id: str = task["theme_id"]
    src_final = Path(task["src_final"])
    target_final = Path(task["target_final"])
    posters_dir = Path(task["posters_dir"])
    t0_overflow_count: int = task["t0_overflow_count"]
    t0_node_rects: dict[str, dict[str, float]] = task["t0_node_rects"]
    force_render: bool = task["force_render"]

    out_png = posters_dir / f"{paper_title}.png"
    val_file = target_final / "validation.json"
    if not force_render and out_png.is_file() and val_file.is_file():
        validation = json.loads(val_file.read_text(encoding="utf-8"))
        hard_overflow = len(validation.get("hardOverflow") or [])
        max_drift = compute_node_rect_drift(t0_node_rects, validation.get("nodeRects") or {})
        return {
            "theme_id": theme_id,
            "title": paper_title,
            "subset": subset,
            "poster_png": str(out_png),
            "target_final": str(target_final),
            "status": validation.get("status"),
            "hard_overflow_count": hard_overflow,
            "t0_overflow_count": t0_overflow_count,
            "outside_count": len(validation.get("outside") or []),
            "missing_images_count": len(validation.get("missingImages") or []),
            "max_node_rect_drift_px": max_drift,
            "used_tight_compensation": False,
            "cached_render": True,
        }

    target_final.mkdir(parents=True, exist_ok=True)
    posters_dir.mkdir(parents=True, exist_ok=True)

    html_raw = (src_final / "poster.html").read_text(encoding="utf-8")
    css_raw = (src_final / "style.css").read_text(encoding="utf-8")
    design_state = json.loads((src_final / "design_state.json").read_text(encoding="utf-8"))
    src_assets = src_final / "assets"
    if not src_assets.is_dir():
        src_assets = src_final.parent / "workspace" / "assets"
    copy_assets_tree(src_assets, target_final / "assets")
    (target_final / "design_state.json").write_text(
        json.dumps(design_state, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    theme = THEME_BY_ID[theme_id]
    dark_selectors = find_dark_background_selectors(css_raw)
    annotated_html = annotate_html_hooks(html_raw, design_state, dark_selectors, theme_id)
    themed_css = build_themed_css(css_raw, theme, dark_selectors, font_scale=1.0)

    bundle = {
        "html": annotated_html,
        "css": themed_css,
        "design_plan": design_state,
    }
    validation = render_bundle(
        bundle,
        target_final,
        export_pdf=False,
        canvas_width=CANVAS_WIDTH,
        canvas_height=CANVAS_HEIGHT,
    )
    hard_overflow = len(validation.get("hardOverflow") or [])
    used_tight_compensation = False
    if hard_overflow > t0_overflow_count:
        themed_css_tight = build_themed_css(css_raw, theme, dark_selectors, font_scale=0.98)
        bundle["css"] = themed_css_tight
        validation = render_bundle(
            bundle,
            target_final,
            export_pdf=False,
            canvas_width=CANVAS_WIDTH,
            canvas_height=CANVAS_HEIGHT,
        )
        hard_overflow = len(validation.get("hardOverflow") or [])
        used_tight_compensation = True

    shutil.copy2(target_final / "poster.png", out_png)
    max_drift = compute_node_rect_drift(t0_node_rects, validation.get("nodeRects") or {})

    return {
        "theme_id": theme_id,
        "title": paper_title,
        "subset": subset,
        "poster_png": str(out_png),
        "target_final": str(target_final),
        "status": validation.get("status"),
        "hard_overflow_count": hard_overflow,
        "t0_overflow_count": t0_overflow_count,
        "outside_count": len(validation.get("outside") or []),
        "missing_images_count": len(validation.get("missingImages") or []),
        "max_node_rect_drift_px": max_drift,
        "used_tight_compensation": used_tight_compensation,
        "cached_render": False,
    }


def stage_t0_original(
    papers: list[dict[str, str]],
    source_exp: Path,
    exp_dir: Path,
    t0_k1_vb: Path,
    t0_e1_vb: Path,
) -> tuple[dict[str, int], dict[str, dict[str, dict[str, float]]]]:
    t0_posters_dir = exp_dir / "posters" / "t0_original"
    t0_stored_vb_out = exp_dir / "vb" / "results" / "t0_stored_api"
    t0_posters_dir.mkdir(parents=True, exist_ok=True)
    t0_stored_vb_out.mkdir(parents=True, exist_ok=True)

    t0_overflow_by_title: dict[str, int] = {}
    t0_rects_by_title: dict[str, dict[str, dict[str, float]]] = {}
    staged_vb_count = 0

    for item in papers:
        title = item["title"]
        subset = item["subset"]
        src_run = (source_exp / "runs" / "g1" / title).resolve()
        src_final = src_run / "final"
        if not src_final.is_dir():
            raise FileNotFoundError(f"Missing source final dir for {title}: {src_final}")

        dst_final = exp_dir / "runs" / "t0_original" / title / "final"
        dst_final.mkdir(parents=True, exist_ok=True)
        for fname in ("poster.html", "style.css", "design_state.json", "poster.png", "validation.json"):
            sfile = src_final / fname
            if not sfile.is_file():
                sfile = src_run / fname
            if sfile.is_file():
                shutil.copy2(sfile, dst_final / fname)
        shutil.copy2(dst_final / "poster.png", t0_posters_dir / f"{title}.png")

        val_path = dst_final / "validation.json"
        if val_path.is_file():
            val = json.loads(val_path.read_text(encoding="utf-8"))
            t0_overflow_by_title[title] = len(val.get("hardOverflow") or [])
            t0_rects_by_title[title] = val.get("nodeRects") or {}
        else:
            t0_overflow_by_title[title] = 0
            t0_rects_by_title[title] = {}

        vb_src = (t0_k1_vb if subset == "k1" else t0_e1_vb) / f"{title}.json"
        if vb_src.is_file():
            rep = json.loads(vb_src.read_text(encoding="utf-8"))
            if rep.get("status") == "completed" and not rep.get("errors"):
                shutil.copy2(vb_src, t0_stored_vb_out / f"{title}.json")
                staged_vb_count += 1

    print(
        f"[apply_themes] Staged 20 t0_original posters and {staged_vb_count}/20 stored reference VB reports "
        f"(in vb/results/t0_stored_api)",
        flush=True,
    )
    return t0_overflow_by_title, t0_rects_by_title


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=DEFAULT_REPO_ROOT)
    parser.add_argument("--source-exp", type=Path, default=DEFAULT_SOURCE_EXP)
    parser.add_argument("--k1-papers", type=Path, default=DEFAULT_K1_PAPERS)
    parser.add_argument("--e1-papers", type=Path, default=DEFAULT_E1_PAPERS)
    parser.add_argument("--t0-k1-vb", type=Path, default=DEFAULT_T0_K1_VB)
    parser.add_argument("--t0-e1-vb", type=Path, default=DEFAULT_T0_E1_VB)
    parser.add_argument("--exp-dir", type=Path, required=True)
    parser.add_argument(
        "--themes",
        nargs="+",
        choices=list(THEME_BY_ID),
        default=list(THEME_BY_ID),
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of papers (e.g. 1 for smoke/quota check)")
    parser.add_argument("--force-render", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    exp_dir = args.exp_dir.resolve()
    exp_dir.mkdir(parents=True, exist_ok=True)

    papers = load_ablation_papers(args.k1_papers, args.e1_papers, limit=args.limit)
    print(f"[apply_themes] Loaded {len(papers)} ablation paper(s)", flush=True)

    t0_overflow_by_title, t0_rects_by_title = stage_t0_original(
        papers=papers,
        source_exp=args.source_exp.resolve(),
        exp_dir=exp_dir,
        t0_k1_vb=args.t0_k1_vb.resolve(),
        t0_e1_vb=args.t0_e1_vb.resolve(),
    )

    tasks: list[dict[str, Any]] = []
    for theme_id in args.themes:
        for item in papers:
            title = item["title"]
            src_run = (args.source_exp / "runs" / "g1" / title).resolve()
            tasks.append(
                {
                    "repo_root": str(args.repo_root.resolve()),
                    "title": title,
                    "subset": item["subset"],
                    "theme_id": theme_id,
                    "src_final": str(src_run / "final"),
                    "target_final": str(exp_dir / "runs" / theme_id / title / "final"),
                    "posters_dir": str(exp_dir / "posters" / theme_id),
                    "t0_overflow_count": t0_overflow_by_title.get(title, 0),
                    "t0_node_rects": t0_rects_by_title.get(title, {}),
                    "force_render": args.force_render,
                }
            )

    print(
        f"[apply_themes] Rendering {len(tasks)} themed posters across {len(args.themes)} themes "
        f"with {args.workers} Playwright workers...",
        flush=True,
    )
    records: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        future_map = {pool.submit(render_one_theme_paper, task): task for task in tasks}
        done_count = 0
        for fut in as_completed(future_map):
            rec = fut.result()
            records.append(rec)
            done_count += 1
            print(
                f"[{done_count}/{len(tasks)}] {rec['theme_id']} | {rec['title'][:42]} "
                f"-> status={rec['status']} overflow={rec['hard_overflow_count']}/{rec['t0_overflow_count']} "
                f"max_drift={rec['max_node_rect_drift_px']}px",
                flush=True,
            )

    records.sort(key=lambda r: (r["theme_id"], r["title"]))
    manifest = {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_experiment": str(args.source_exp.resolve()),
        "canvas_size": [CANVAS_WIDTH, CANVAS_HEIGHT],
        "paper_count": len(papers),
        "papers": papers,
        "themes": [
            {
                "theme_id": "t0_original",
                "label": "VASP Default (Original)",
                "category": "original",
                "description": "Unmodified VASP (GPT-5.6-Sol) posters from pi-full100-sol-20260925.",
            },
            *[
                {
                    "theme_id": spec.theme_id,
                    "label": spec.label,
                    "category": spec.category,
                    "description": spec.description,
                }
                for spec in THEME_SPECS
                if spec.theme_id in args.themes
            ],
        ],
        "renders": records,
    }
    manifest_path = exp_dir / "inputs" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[apply_themes] Wrote manifest to {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
