You are constructing reference information units for an academic-poster information-coverage benchmark.

You receive exactly one attached image: an author-made OFFICIAL academic poster. Use only the visible poster. Do not inspect the paper, repository files, generated posters, or outside sources. Do not call tools.

Extract exactly SEVEN core information units that collectively represent the main information the poster's authors chose to communicate.

Selection principles:

1. Follow this poster's own emphasis. Different posters may emphasize a method, theoretical statements, equations, datasets, empirical findings, applications, or limitations. Do not force a fixed section taxonomy.
2. Prefer central messages over minor implementation details, isolated hyperparameters, acknowledgements, citations, QR codes, or decorative text.
3. Each unit must be independently judgeable in another poster. Avoid vague units such as “the method works well.”
4. Keep units semantically distinct. Do not split one claim into multiple near-duplicates merely to reach seven.
5. Conversely, do not combine unrelated claims into one unit. When some compression is necessary, combine only facts that form one coherent message.
6. Use the visible poster as the complete evidence boundary. Do not add facts remembered from the paper or inferred from scientific plausibility.
7. Balance motivation/problem, method/theory, and results only when the official poster itself emphasizes all of them.

Required fields for each unit:

- `unit_id`: use `<poster_id>-U1` through `<poster_id>-U7` in the order a reader would naturally understand the poster.
- `information_type`: a concise role such as `background`, `problem`, `method`, `theory`, `dataset`, `result`, `limitation`, or another poster-appropriate type.
- `topic`: a short label for the unit.
- `local_context`: two or three sentences defining terms and situating the claim so a later judge will not misinterpret it.
- `statement`: one concise, self-contained formulation of the core information.
- `coverage_requirement`: the minimum semantic content a generated poster must visibly communicate to count this unit as covered. Make this tolerant of paraphrases and compression. Do not require every number, example, dataset name, or qualifier unless it is central to the claim.
- `official_evidence`: a short description of the exact visible section, phrase, equation, diagram, plot, or table supporting the unit. Do not merely write “official poster.”

Quality guardrails:

- A statement may include an exact number only when the number is prominently emphasized and materially changes the message.
- A coverage requirement must not be stricter or broader than the statement.
- If a result table is dense, extract its clear comparative conclusion rather than several individual cells.
- Formulas and diagrams count as information, but describe the semantic message they convey.
- Correct obvious OCR spelling internally, but do not invent unreadable content.

Return only JSON conforming to the supplied schema.
