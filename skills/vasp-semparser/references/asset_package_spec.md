# VASP Asset Package Specification

`semparser_cli.py package` outputs the exact standardized directory consumed by `vasp-viscomposer`:

```text
<asset_dir>/
├── manifest.json                      # Package status ("COMPLETE"/"INCOMPLETE"), counts, SHA-256
└── assets/
    ├── figures/                       # Lossless high-DPI body Figure PNG crops
    ├── appendix_figures/              # Secondary appendix Figure PNG crops (if --include-appendix)
    ├── logos/                         # Best-effort fetched institution/venue marks
    ├── qr/                            # Generated Paper / Code / Project QR PNGs
    └── meta/
        ├── text.txt                   # Extracted reading-order full paper text
        ├── metadata.json              # Title, authors, institutes, venue, URLs, logos, QR info
        ├── captions.json              # Normalized label -> caption text mapping
        ├── captions_full.json         # Lossless full caption record list
        ├── figures.json               # Body figure geometry, bbox, page, column metadata
        ├── appendix_figures.json      # Appendix figure geometry metadata
        ├── segments2.json             # Line-role segmentation
        ├── source_lines.json          # Numbered PDF lines with page & bbox coordinates
        └── structured_elements.json   # Structured tables, equations, and algorithms
```

This contract decouples Stage 1 (`vasp-semparser`) from Stage 2 (`vasp-viscomposer`). You can freely edit, replace, add, or remove files inside `<asset_dir>/assets/` before passing `<asset_dir>` to `vasp-viscomposer` or any downstream tool.
