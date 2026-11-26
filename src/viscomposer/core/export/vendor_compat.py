"""Compatibility loader for the pinned ResearchStudio HTML-to-PPTX source."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType


_SOURCE_PATCHES = (
    (
        "const LINE_THRESHOLD = Math.max(20, lhPx * 0.5);",
        "// Accept wrapped small text whose rounded baseline delta is 20px.\n"
        "            const LINE_THRESHOLD = Math.max(8, lhPx * 0.45);",
    ),
    (
        """              let pseudoX = r.left + padLeft;
              let pseudoY = r.top + padTop;
              if (cs.display === 'flex' || cs.display === 'inline-flex') {""",
        """              let pseudoX = r.left + padLeft;
              let pseudoY = r.top + padTop;
              if (pcs.position === 'absolute') {
                // Respect explicit insets for numbered ::before badges.
                const left = parseFloat(pcs.left);
                const top = parseFloat(pcs.top);
                if (Number.isFinite(left)) pseudoX = r.left + left;
                if (Number.isFinite(top)) pseudoY = r.top + top;
              } else if (cs.display === 'flex' || cs.display === 'inline-flex') {""",
    ),
    (
        """            return boxy >= 2;
          };""",
        """            if (boxy >= 2) return true;

            // A flex/grid row may consist only of text-bearing inline tags.
            // Grid/flex blockifies those children and positions each one in a
            // distinct cell, but the tag-name-only inline test above would
            // otherwise merge the full row into a single PPT textbox.  That
            // loses the CSS gap and can make the second cell cross a column
            // boundary.  Treat two visible, spatially distinct children as a
            // composite row so each leaf keeps its browser bbox.
            const visible = [...node.children].filter(c => {
              const ccs = getComputedStyle(c);
              const cr = c.getBoundingClientRect();
              return ccs.display !== 'none' && ccs.visibility !== 'hidden'
                  && parseFloat(ccs.opacity || '1') > 0
                  && cr.width >= 1 && cr.height >= 1
                  && c.textContent.trim().length > 0;
            });
            if (visible.length < 2) return false;
            const rects = visible.map(c => c.getBoundingClientRect());
            return rects.some((a, i) => rects.slice(i + 1).some(b =>
              Math.abs(a.left - b.left) > 1 || Math.abs(a.top - b.top) > 1));
          };""",
    ),
    (
        """              if (tag !== 'span') return false;
              if (el.children.length > 0) return false;   // leaf text only""",
        """              // Inline semantic leaves can become independent
              // grid/flex items even when they are not <span> (for example
              // <b>Label</b><span>Value</span>).  Emit any ordinary inline
              // leaf whose text is not owned by an emitting ancestor.
              if (!['span', 'b', 'strong', 'em', 'i', 'small', 'label', 'a']
                    .includes(tag)) return false;
              if (el.children.length > 0) return false;   // leaf text only""",
    ),
    (
        """                } else {
                  collectRuns(node, runs);
                }
              }""",
        """                } else {
                  // CSS margins between adjacent inline children are visual
                  // whitespace even when the HTML source has no literal space
                  // (e.g. <strong>4,128</strong><span>training images</span>).
                  // Preserve that separation in the editable PPT text flow.
                  const ml = parseFloat(cs.marginLeft) || 0;
                  const prev = runs.length ? runs[runs.length - 1] : null;
                  if (ml > 0 && prev && prev.text.trimEnd() === prev.text) {
                    runs.push({
                      text: ' ',
                      font_family: cs.fontFamily,
                      font_size_px: parseFloat(cs.fontSize),
                      font_weight: cs.fontWeight,
                      font_style: cs.fontStyle,
                      color: toRGB(cs.color),
                      letter_spacing_px: 0,
                      text_transform: cs.textTransform,
                      vertical_align: 'baseline',
                      bg_color: '',
                    });
                  }
                  collectRuns(node, runs);
                }
              }""",
    ),
    (
        """              const pd = getComputedStyle(par).display;
              if (!['flex', 'inline-flex', 'grid', 'inline-grid'].includes(pd)) return false;""",
        """              // The independent leaf can be nested one wrapper below
              // the grid/flex item.  The ancestor ownership scan below is the
              // authoritative duplicate guard: it stops at an emitting inline
              // ancestor, or emits when it reaches a composite layout row.
              // Requiring the immediate parent itself to be flex/grid loses
              // labels such as a tag stacked above a heading inside a grid cell.""",
    ),
)


def patched_source(path: Path) -> str:
    """Apply narrowly scoped fixes while leaving the third-party clone clean."""

    source = path.read_text(encoding="utf-8")
    for old, new in _SOURCE_PATCHES:
        occurrences = source.count(old)
        if occurrences != 1:
            raise RuntimeError(
                "ResearchStudio converter changed: expected one compatibility "
                f"patch target, found {occurrences}: {old[:60]!r}"
            )
        source = source.replace(old, new, 1)
    return source


def load_patched_module(path: Path) -> ModuleType:
    """Compile the local trusted converter with reproducible compatibility fixes."""

    module = ModuleType("researchstudio_html_to_pptx")
    module.__file__ = str(path)
    module.__package__ = ""
    exec(compile(patched_source(path), str(path), "exec"), module.__dict__)
    return module
