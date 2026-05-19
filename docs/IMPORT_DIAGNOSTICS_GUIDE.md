# Import Diagnostics Guide

## Overview

The Import Wizard displays engineering-grade diagnostics after every import.
The diagnostics panel shows exactly what happened to the imported data — what
was repaired, what was dropped, which channels were included, and whether the
resulting waveform can be trusted.

The diagnostics are aggregated from existing backend results without
re-reading or recomputing anything from the source file.

---

## Diagnostics Sections

### Data Summary

Counts visible immediately after import:

| Field | Meaning |
|---|---|
| Rows imported | Rows retained in the normalized output |
| Rows dropped | Rows removed due to unrecoverable timestamps |
| Duplicate timestamps | Duplicate time values retained in the output |
| Invalid values | NaN count across all data columns |
| Data completeness | Percentage of source rows retained |
| Analog channels | Analog channels in the DisturbanceRecord |
| Digital channels | Digital channels in the DisturbanceRecord |
| Excluded columns | Source columns removed by user decision |
| User overrides applied | Columns where user changed name, type, or unit |

**Row drops always appear prominently** — data loss is never hidden.

---

### Timestamp

| Field | Meaning |
|---|---|
| Column | The source column used as the time axis |
| Strategy | How the time axis was produced |
| Format | strptime format string used for parsing |
| Format source | `auto-detected` or `user override` |
| Detection confidence | How confident the profiler was in the timestamp column |
| Repair actions | What the repair engine actually did |

#### Repair strategies

| Strategy | Description |
|---|---|
| No repair needed | Timestamps already valid — no transformation applied |
| Parsed using auto-detected format | Profiler detected the strptime format automatically |
| Parsed using user-supplied format | User explicitly specified the format string |
| Missing timestamps interpolated | NaT/blank timestamps filled by interpolation |
| Timestamps reconstructed from sampling interval | Time axis rebuilt from constant sampling rate |
| Date and time columns merged | Separate date and time columns combined |
| Excel date serial numbers converted | Excel numeric dates converted to datetime |
| Timestamps converted to target timezone | Timezone conversion applied |

---

### Channel Classification

Shows how confidently the auto-classifier assigned a semantic type (VOLTAGE,
CURRENT, MW, etc.) to each column.

| Field | Meaning |
|---|---|
| Confidence | Overall classification quality: High / Medium / Low |
| Low-confidence channels | Channels where confidence < 60% |

Low-confidence channels are imported as-is but may need manual review.

---

### Validation Messages

Grouped by severity:

- **ERROR** — blocking issues (import could not complete cleanly)
- **WARNING** — non-blocking issues (import completed but with caveats)
- **INFO** — informational notes about processing decisions

Common warnings:
- Unknown column type (`PLAN_UNKNOWN_COLUMN`) — column preserved as-is
- Sidecar already exists (`EXPORT_SIDECAR_EXISTS`) — not overwritten

---

### Export

Appears after the normalized file is saved, or as guidance before saving:

- Metadata sidecar purpose and content summary
- Rows and format written
- Whether the audit sidecar was written

**The metadata sidecar is important**: it preserves the repair strategy,
canonical column names, original source names, confidence scores, and user
overrides. Without it, a re-import cannot reproduce exactly the same result.

---

### Performance Guidance

Appears only for large datasets (> 100,000 rows):

- Recommendation to export as Parquet or Feather for faster reload
- Note that the waveform viewer uses viewport rendering

---

## Confidence Meaning

### Timestamp confidence

| Label | Range | Meaning |
|---|---|---|
| High | ≥ 85% | Profiler strongly identified this column as a timestamp |
| Medium | 60–84% | Plausible timestamp column, minor ambiguity |
| Low | < 60% | Uncertain identification — verify column selection |
| N/A | — | No candidate was detected |

### Channel classification confidence

| Label | Meaning |
|---|---|
| High | All or most channels classified with high confidence |
| Medium | Mixed confidence — some channels may need review |
| Low | Many channels have uncertain classification |
| N/A | No dataset available |

---

## Repair Summary Meaning

Repair actions describe what the backend actually did, not what was planned:

- `N rows removed due to unrecoverable timestamps` — rows where the
  timestamp could not be parsed and had no valid repair path. These rows
  are permanently absent from the waveform.
- `N duplicate timestamps detected` — identical time values were present
  in the source. These are retained but may affect analysis.
- `Missing timestamps were interpolated` — NaT gaps were filled using
  linear interpolation between valid neighbours.
- `Timestamps reconstructed from X s sampling interval` — the entire time
  axis was rebuilt from a constant sampling rate, replacing the original
  column values.
- `User-specified format applied: <format>` — the user overrode the
  auto-detected strptime format.

---

## Row-Drop Reporting

Row drops are always shown prominently in the diagnostics panel with the
`← data loss` marker. The `data_completeness` percentage shows how much
of the source was retained.

If rows were dropped, verify:
1. The timestamp column selection is correct.
2. The format string correctly parses all values.
3. Missing-value rows in the source are expected.

---

## Operational Guidance Philosophy

The diagnostics are written for power-system engineers, not developers:

- Engineering units and channel names, not internal codes
- Repair actions in plain language, not strategy enum values
- Data loss explicitly flagged, not buried in logs
- Confidence indicators for timestamp and channel classification
- Export guidance tied to actual file size and available engines

---

## Known Limitations

- Diagnostics are generated once per import. Editing column mappings after
  the fact requires re-running the import.
- The large-file threshold (100,000 rows) is based on row count, not file
  size in bytes. A wide CSV may warrant guidance at lower row counts.
- Parquet/Feather guidance is always shown for large files regardless of
  whether the engines are installed. The export dialog will report a clear
  error if they are absent.
- Classification confidence reflects the auto-classifier. User overrides
  are counted but do not retroactively update the confidence score shown.
