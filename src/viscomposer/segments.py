"""Partition an existing prompt text into module-owned blocks.

A block starts at a unique anchor substring of the original text and runs to the
next block's anchor, so the blocks cover the text exactly. Under an arm, a block is
kept verbatim when its owners are plugged in; otherwise it is re-rendered for the
active modules (mixed sentences) or dropped. The full arm therefore reproduces the
original text by construction, and every arm differs only in owned blocks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from src.viscomposer.arms import Arm

ALL = "all"   # keep only when every owner is plugged in
ANY = "any"   # keep when at least one owner is plugged in


@dataclass(frozen=True)
class Block:
    anchor: str
    owners: tuple[str, ...] = ()   # () = base, always kept
    mode: str = ALL
    render: Callable[[Arm], str] | None = None   # partial arms; None = drop

    def kept(self, arm: Arm) -> bool:
        if not self.owners:
            return True
        return arm.has(*self.owners) if self.mode == ALL else arm.any(*self.owners)


@dataclass(frozen=True)
class Partition:
    name: str
    text: str
    pieces: tuple[tuple[Block, str, bool], ...]   # (block, original text, starts a line)

    def compose(self, arm: Arm) -> str:
        if arm.is_full:
            return self.text
        out: list[str] = []
        for block, original, line_start in self.pieces:
            if block.kept(arm):
                out.append(original)
            elif block.render is not None:
                out.append(block.render(arm))
            elif not line_start:
                # Mid-paragraph block: keep its trailing separator so the paragraph
                # break or sentence space it carried survives.
                out.append(original[len(original.rstrip()):])
        return tidy("".join(out))

    def ownership(self) -> list[dict[str, object]]:
        return [
            {"anchor": block.anchor, "owners": list(block.owners), "mode": block.mode,
             "mixed": block.render is not None, "chars": len(original)}
            for block, original, _ in self.pieces
        ]


def tidy(text: str) -> str:
    """Normalize whitespace left by removed blocks (never applied to the full arm)."""
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"(?<=\S) {2,}(?=\S)", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def partition(name: str, text: str, blocks: list[Block]) -> Partition:
    """Locate every anchor exactly once, in order, starting at offset 0."""
    starts: list[int] = []
    for block in blocks:
        count = text.count(block.anchor)
        if count != 1:
            raise ValueError(f"{name}: anchor must occur exactly once, found {count}: {block.anchor!r}")
        starts.append(text.index(block.anchor))
    if starts[0] != 0:
        raise ValueError(f"{name}: first block must start at offset 0")
    if starts != sorted(starts):
        raise ValueError(f"{name}: anchors are out of order")
    ends = starts[1:] + [len(text)]
    pieces = tuple(
        (block, text[start:end], start == 0 or text[start - 1] == "\n")
        for block, start, end in zip(blocks, starts, ends)
    )
    assert "".join(piece for _, piece, _ in pieces) == text
    return Partition(name, text, pieces)


def listing(items: list[str]) -> str:
    """Join clauses as 'a', 'a and b', or 'a, b, and c'."""
    if len(items) <= 1:
        return "".join(items)
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"
