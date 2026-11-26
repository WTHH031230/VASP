# VASP: Visual-anchored Agent for Scientific Poster Generation

Official codebase for **VASP: Visual-anchored Agent for Scientific Poster Generation**.

## Repository Structure

```text
VASP/
├── src/
│   ├── semparser/        # Stage 1: Semantics-directed asset parsing & role-constrained figure extraction
│   ├── viscomposer/      # Stage 2: Vision-centric closed-loop layout synthesis (including src/viscomposer/pi/)
│   ├── ablation/         # Decoupled ablations (wo_geom, wo_sv, wo_iter_sv) & CSS theme sensitivity (T0–T8)
│   ├── benchmarks/       # figure_crop, figure_completeness, vision_bench, information_coverage (+ paper_grounding_audit), vlm_judge, paperquiz, cross_family (Vertex AI)
│   └── baselines/        # Baseline adapters (Paper2Poster, PosterGen, PosterForest, PosterMELD, P2P, ResearchStudio, PaperX)
├── configs/
│   ├── pi/               # Provider template (models.example.json) & settings.json
│   └── schemas/          # JSON schemas for VisComposer, Visual Presentation, and Key-Info Coverage
├── scripts/              # Shell entrypoints for end-to-end generation, ablations, and benchmarks
└── requirements.txt
```

Benchmark annotations (100-paper index, 948 figure boxes, 100-figure completeness sample, 700 frozen reference information units, and 12 visual criteria metadata) are hosted on Hugging Face at [WTHH031230/VASP-data](https://huggingface.co/datasets/WTHH031230/VASP-data):

```bash
hf download WTHH031230/VASP-data --repo-type dataset --local-dir ./VASP-data
```

Source paper PDFs and official posters come from the publicly released Paper2Poster dataset.

## Setup

```bash
conda create -n vasp python=3.12 -y
conda activate vasp

pip install -r requirements.txt
playwright install chromium

# Install Node.js runtime dependencies (latest Pi CLI & Google Vertex AI auth; requires Node.js >= 20):
bash scripts/setup_pi.sh
```

## Supported Execution Backends

Poster generation (`run_vasp.sh`, SemParser, VisComposer) supports 5 backends (`pi-api`, `pi-chatgpt`, `codex`, `claude`, `agy`), and benchmark evaluation (`eval_benchmark.sh`, `src/benchmarks/cross_family/`) additionally supports **Google Vertex AI API** (`vertex`):

| `--backend` / `--designer` | Description | Prerequisite |
|---|---|---|
| `pi-api` | Pi agent via OpenAI-compatible API | Copy `configs/pi/models.example.json` → `configs/pi/models.json`, set `OPENAI_API_KEY` |
| `pi-chatgpt` | Pi agent via ChatGPT OAuth login | Run `pi /login` once (stores OAuth session under `PI_CODING_AGENT_DIR`) |
| `codex` | OpenAI Codex CLI (`codex exec` / `resume`) | `codex` installed & authenticated |
| `claude` | Anthropic Claude Code CLI (`claude -p` / `--resume`) | `claude` installed & authenticated |
| `agy` | Antigravity CLI (`agy -p` / `--conversation`) | `agy` installed & authenticated |
| `vertex` *(benchmarks)* | Google Vertex AI REST API (`call.mjs`) | Set `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION` (default `global`), and `GOOGLE_APPLICATION_CREDENTIALS` |

## Usage

### End-to-End (Stage 1 → Stage 2)
```bash
bash scripts/run_vasp.sh \
  --pdf /path/to/paper.pdf \
  --out ./outputs/my_paper \
  --backend pi-api \
  --model gpt-5.6-sol \
  --effort medium \
  --arm full \
  --max-revisions 2
```

### Stage 1 Only (SemParser)
```bash
bash scripts/run_semparser.sh <data_dir> <out_root> --include-appendix --backend codex --model gpt-5.6-sol --effort medium
```

### Stage 2 Only (VisComposer & Modular Ablations)
```bash
bash scripts/run_viscomposer.sh \
  --arm full \
  --asset-dir <semparser_paper_dir> \
  --output-dir <output_dir> \
  --designer pi-api \
  --model gpt-5.6-sol \
  --effort medium \
  --max-revisions 2
```

### Ablations & Sensitivity Analyses

All ablation and sensitivity variants reported in the paper are unified under `scripts/run_ablation.sh` (and `scripts/run_viscomposer.sh`):

| Paper Variant | Command |
|---|---|
| **w/o SemParser** | `bash scripts/run_ablation.sh no_assets --asset-dir <dir> --rs-root <rs_assets_root> --output-dir <out>` |
| **w/o anchored composition (template)** | `bash scripts/run_ablation.sh no_vf_template --asset-dir <dir> --output-dir <out>` |
| **w/o anchored composition (text-first)** | `bash scripts/run_ablation.sh text_first --asset-dir <dir> --output-dir <out>` |
| **w/o semantic verification** *(w/o iter. SV & GV)* | `bash scripts/run_ablation.sh no_sem --asset-dir <dir> --output-dir <out>` |
| **w/o refinement loop** | `bash scripts/run_ablation.sh no_basic --asset-dir <dir> --output-dir <out>` |
| **w/o iterative SV** *(init. SV + GV)* | `bash scripts/run_ablation.sh wo_iter_sv --asset-dir <dir> --output-dir <out>` |
| **w/o all SV** *(GV only)* | `bash scripts/run_ablation.sh wo_sv --asset-dir <dir> --output-dir <out>` |
| **w/o GV** *(init. + iter. SV)* | `bash scripts/run_ablation.sh wo_geom --asset-dir <dir> --output-dir <out>` |
| **CSS Theme Sensitivity (T0–T8)** | `bash scripts/run_ablation.sh theme_sensitivity --help` |
| **Reasoning Effort & Revision Rounds** | `bash scripts/run_viscomposer.sh --arm full --effort {low,medium,high,xhigh,max} --max-revisions {1,2,4,8,16} ...` |

> **Note on `no_assets` (w/o SemParser):** This ablation replaces SemParser's crops/captions/text with ResearchStudio's PDF asset extraction. First generate per-paper ResearchStudio assets using `src/baselines/research-studio-real/`, then point `--rs-root` (or `VASP_RS_ROOT`) to that root directory.

### Benchmarks (Multi-Backend Unified Runner, Vertex AI API & Figure Crop)
```bash
# Figure extraction (Table 2, body scope)
bash scripts/eval_figure_crop.sh <prediction_root> <output_dir> --gt-root <VASP-data/figure_annotations_948> --scope body

# Visual Presentation (reports equal-group paper Eq. 2 & Excl. N/A scores), Key-Info Coverage, VLM-as-Judge, and PaperQuiz
bash scripts/eval_benchmark.sh vision_bench --poster poster.png --out vb.json --backend codex --model gpt-5.6-sol
bash scripts/eval_benchmark.sh information_coverage --poster poster.png --reference-units <VASP-data/key_info_coverage_700/reference_information_units_full100.json> --out ic.json --backend pi-chatgpt
bash scripts/eval_benchmark.sh vlm_judge --poster poster.png --out vj.json --backend claude --model claude-opus-5
bash scripts/eval_benchmark.sh paperquiz --poster poster.png --qa-json o3_qa.json --out pq.json --backend vertex --model gemini-4-argon
```

> **Note on `src/benchmarks/cross_family/`:** For general poster evaluation via Google Vertex AI API, use `scripts/eval_benchmark.sh --backend vertex` above. The scripts under `src/benchmarks/cross_family/` are batch orchestration scripts for reproducing the 900-poster cross-family tables in Appendix C and expect the paper's `results/exp/` directory layout.

## License

This project is released under the [MIT License](LICENSE).
