# Changelog — Bugfix-Set für experiments/

## Behobene Bugs

### 1. Faithfulness-Metrik immer 0.0 / 1.0 (s10, s11)

**Symptom:** In `aggregated/.../pollution_effect.csv` und
`aggregated/.../long_context_effect.csv` waren `mean_context_overlap`
über alle `pollution_ratio`/`top_k`-Stufen exakt 0.0, und `mean_hallucination`
exakt 1.0 — auch bei Antworten, die nachweislich gut im Kontext geerdet waren.

**Ursache:** `context_overlap(answer, contexts)` und `hallucination_score`
erwarten als zweites Argument ein **Iterable von Kontext-Strings**. Die Aufrufe
in `s10_context_pollution.py` und `s11_long_context.py` haben die Chunks vorher
mit `"\n".join(...)` zu einem einzelnen String konkateniert. In der Funktion
iteriert `for c in contexts` dann über die *Zeichen* dieses Strings; jedes
einzelne Zeichen wird gegen die Regex `[A-Za-z][A-Za-z0-9\-]+|\d+` (mindestens
2 Zeichen) getokenisiert → liefert nichts → `ctx_tokens` bleibt leer → Rückgabe
`0.0`, und damit `hallucination = 1.0 − 0.0 = 1.0` — konstant.

**Fix:**
* `experiments/suites/s10_context_pollution.py`, Z. 113–114: `context_chunks`
  direkt übergeben (keine Konkatenation).
* `experiments/suites/s11_long_context.py`, Z. 91–93: `ctx_chunks` direkt
  übergeben, lokale Variable `ctx_blob` entfernt.

**Verifikation:** Auf einer echten Antwort aus dem vorhandenen
`raw/.../generation_records.jsonl` liefert die reparierte Metrik
`overlap = 0.58 / hallucination = 0.42` statt der konstanten 0.0/1.0.

### 2. `bootstrap_ci()`-TypeError in s09_stability

**Symptom:**
```
Suite s09_stability failed after 0.24s:
bootstrap_ci() got an unexpected keyword argument 'n_boot'
```

**Ursache:** Doppelter Aufruffehler in
`experiments/suites/s09_stability.py`, Z. 90:
1. Das Keyword heißt in `metrics/stability_metrics.py` `n_resamples`, nicht
   `n_boot`.
2. `bootstrap_ci` gibt ein **3-Tupel** `(mean, lower, upper)` zurück, der
   Aufrufer entpackt aber nur in zwei Variablen `lo, hi = ...` — selbst nach
   Korrektur des Keywords wäre das ein `ValueError: too many values to unpack`.

**Fix:** Auf 3-Tupel-Unpacking umgestellt und korrektes Keyword:
```python
_, lo, hi = (
    bootstrap_ci(per_repeat_recall, n_resamples=200)
    if per_repeat_recall else (0.0, 0.0, 0.0)
)
```

Damit produziert s09 wieder echte Konfidenzintervalle für die Recall-Varianz.

## Neu hinzugefügt

### `experiments/tools/rescore_generation.py`

Offline-Re-Scoring-Werkzeug. Liest die bestehenden
`raw/<run_id>/<suite>/generation_records.jsonl` ein, rechnet
`context_overlap` und `hallucination_score` mit der reparierten Metrik nochmal
über die *gespeicherten* Antworten neu, und schreibt die aggregierten CSVs
(`pollution_effect.csv`, `long_context_effect.csv` und die `per_query_*.csv`)
neu — **ohne** dass die Generierung erneut über Ollama laufen muss.

Aufruf im Container:
```
docker exec -i rag-benchmark-runner \
  python -m experiments.tools.rescore_generation \
    --run-id run_20260527_144558_fbf83b \
    --output-root /opt/experiment-outputs
```

Das Original-`raw`-File wird vor dem Überschreiben nach
`generation_records.pre_rescore.jsonl` gesichert.

**Einschränkung:** Wenn das Original-`raw`-File keine `retrieved_ids` je Zeile
enthält (aktuell nicht der Fall), nutzt das Tool das `expected_answer`-Feld
als Fallback-Kontext. Das genügt für eine sinnvolle Re-Bewertung der
Faithfulness gegenüber dem Gold-Chunk, ist aber nicht *exakt* dasselbe wie der
zur Generierzeit tatsächlich gesehene Kontext (insbesondere bei
`pollution_ratio > 0`, wo Distraktoren eingemischt waren). Für absolut
identische Werte muss s10 einmal mit der gefixten Codebasis durchlaufen — dann
ist der Lauf vollständig konsistent und das Tool obsolet.

## Was unverändert bleibt

* Die Metrik-Funktionen selbst (`metrics/faithfulness_metrics.py`,
  `metrics/stability_metrics.py`) — sie waren korrekt.
* Sämtliche Retrieval-, Latenz- und Embedding-Metriken sowie alle anderen
  Suiten. Die bisherigen Retrieval-Zahlen aus `run_20260527_144558_fbf83b`
  bleiben gültig.
