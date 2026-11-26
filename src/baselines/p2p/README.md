# P2P Codex Adapter

This adapter runs the upstream `multimodal-art-projection/P2P` entry point.
Its four DocLayout requests (`figure`, `table`, `figurecaption`, and
`tablecaption`), confidence-threshold pairing, crop caches, content planning,
HTML layout, and blank-space feedback are preserved. The local detector is a
transport replacement for upstream's Modal endpoint and uses the exact
`doclayout_yolo_ft.pt` object (SHA256
`9a2ee0220fe3d9ad31b47e1d9f1282f46959a54e4618fce9cffcc9715b8286e2`).
Only LangChain model constructors are routed to `codex exec --ephemeral`.

Upstream's Modal file refers to the nonexistent package/version/path pair
`doclayout-yolo==0.0.2` and `path2doclayout_yolo_ft.pt`; the same repository's
requirements use the published `doclayout-yolo==0.0.3`, and the official model
repository names the unchanged object `models/Layout/YOLO/doclayout_yolo_ft.pt`.

```bash
bash scripts/baseline/start_p2p_detector.sh
python \
  -m src.baseline.p2p.run dataset/<paper>/paper.pdf \
  results/p2p/gpt5.6-sol-strict/<paper> \
  --detector-url http://127.0.0.1:8019/ \
  --model gpt-5.6-sol --effort medium
```

Each run preserves `poster.png`, `poster.html`, `poster.json`, Codex traces, and
`adapter_run.json` with wall time, tokens, and API-equivalent cost.
