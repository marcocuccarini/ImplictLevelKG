# Chain Extractor Subagent

Extracts a multi-hop concept chain linking a hate-speech/stereotype target to the
implied stereotype statement, for the ImplictLevelKG project. Used to augment the
local knowledge graph (kg_en.ttl / kg_it.ttl) with intermediate reasoning steps
instead of a flat target -> stereotype edge.

## Instructions

You will be given a CSV file path (via /tasklet/... absolute path) containing rows with
columns: unique_id, source, text, target, target_en, implied_statement, label.

For EACH row, read the `text` (the original post) and `implied_statement` (the hidden
stereotype meaning), and produce a short **chain of intermediate concepts** that a
reader's mind plausibly passes through to get from the `target` to the
`implied_statement`. Think of it like hops in a reasoning graph, e.g.:

  target: migrante
  text: "...arriva in italia con un barcone e poi va nel mare..."
  implied_statement: "spero che affondino tutti in mare"
  chain: ["arriva in Italia", "con un barcone", "nel mare"]

Rules:
- The chain must have between 1 and 4 steps (concept phrases), ordered from closest-to-target
  to closest-to-the-implied-statement.
- Each step should be a short phrase (2-5 words) grounded in words/ideas actually present in
  the `text`, when possible. Do not invent unrelated content.
- If the text has no clear intermediate concepts (implied_statement follows directly from
  target with no chain of reasoning), return a chain with just 1 step summarizing the key
  connecting concept.
- Keep chain step language in the SAME language as the input text (English rows -> English
  steps, Italian rows -> Italian steps).
- Do not translate or modify `target`, `target_en`, `implied_statement`, or `label`.

Output a SINGLE JSON array (no markdown fences, no commentary) with one object per input row:
```json
[
  {
    "unique_id": "...",
    "target": "...",
    "target_en": "...",
    "chain": ["step 1", "step 2", "step 3"],
    "implied_statement": "..."
  },
  ...
]
```

Process ALL rows in the input file. Write the JSON array to
`/tmp/work/chain_output.json` using the sandbox (run_command / write via python), AND
also return the same JSON array as your final message (if the file is large, you may
truncate the final message but the file must contain the complete output).

If anything is unclear or a row is malformed, skip it and note the skipped unique_id in
a top-level `"_errors"` field of the JSON rather than failing the whole batch.
