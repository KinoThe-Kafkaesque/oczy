# First-run provenance — G2 v4-r20 oracle screen, first completed run

Record hygiene for kanban `t_738c4a6a` (follow-up to holistic probe
`t_c1be32f2`, finding F3). This file authenticates the sibling
`first_run_results.json`; it changes no score, verdict, or frozen record.

- Source bytes: `/home/nyanpasu/.local/state/oczy/diagnostics/2026-09-11-organ-identity/g2-v4r20/results.json`
  (diagnostic host; outside the repo, ephemeral).
- SHA-256: `36f768a20bf985995ef3cea006b4ce7476ceb512049d9e0a5cc369044a302284`
- Size: 40119 bytes. Rows: 79.
- Identity binding: the file's `probe_source_sha256` is
  `b2d32b08e12e38f707c52a5b99021e8e09063ddfd576af43611dc357e1638603`,
  the exact revision prefix recorded in the frozen gate log
  `experiments_logs/2026-10-05_r20_g2_v4_r20_oracle_screen.md`
  ("first completed screen (144.4 s, `probe_source_sha256` `b2d32b08…`)").
  `seconds` in-file is 144.4, matching the same log line.
- Relation to the recorded run: `results.json` (SHA-256 `3962fa7b…`,
  `probe_source_sha256` `0cc7cc5e…`, 156.9 s) differs from this file ONLY in
  `probe_source_sha256`, `seconds`, `truncation_accounting`, and the per-row
  `truncation_heuristic` flags (51 rows). All 79 `generated` strings, all
  `correct` flags, `gate_passed`, `scores`, and `oracle_by_family` are
  identical. First-run truncation counts: no_context 16/32, teaching 24/32,
  oracle 15/15; recorded run: 3/32, 0/32, 1/15.

## Re-derive (diff procedure)

```bash
cd /home/nyanpasu/Desktop/code/kinoSoft/oczy
A=experiments_logs/artifacts/2026-10-05_r20_g2_v4_r20_oracle_screen
sha256sum $A/first_run_results.json $A/results.json
# expect 36f768a2… and 3962fa7b… respectively
.venv/bin/python - "$A/first_run_results.json" "$A/results.json" <<'EOF'
import json, sys
a = json.load(open(sys.argv[1])); b = json.load(open(sys.argv[2]))
ga = [r["generated"] for r in a["rows"]]; gb = [r["generated"] for r in b["rows"]]
ca = [r["correct"] for r in a["rows"]]; cb = [r["correct"] for r in b["rows"]]
assert len(ga) == len(gb) == 79 and ga == gb and ca == cb
assert a["gate_passed"] == b["gate_passed"] == True
assert a["scores"] == b["scores"] and a["oracle_by_family"] == b["oracle_by_family"]
top = [k for k in set(a) | set(b) if a.get(k) != b.get(k)]
rowkeys = {k for x, y in zip(a["rows"], b["rows"]) for k in set(x) | set(y) if x.get(k) != y.get(k)}
print("top-level diffs:", sorted(top))
print("row-level diffs:", sorted(rowkeys))
assert sorted(top) == ["probe_source_sha256", "rows", "seconds", "truncation_accounting"]
assert sorted(rowkeys) == ["truncation_heuristic"]
print("OK: generations byte-identical; differences confined to the descriptive heuristic + run metadata")
EOF
```
