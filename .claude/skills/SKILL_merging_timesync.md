# SKILL: Multi-Source Merging & Time Synchronisation

## Trigger
Load this skill when implementing:
- src/models/merged_session.py
- src/engine/time_sync_engine.py
- src/ui/merge_manager.py
- src/ui/sync_panel.py
- anything involving loading multiple files simultaneously

---

## CORE CONCEPT

A MergedSession holds multiple sources (COMTRADE bays, PMU CSV files) on a
single shared time axis. Each source retains its own data — nothing is merged
into a single array. Time alignment is achieved by applying a per-source offset
that is added to all display time lookups for that source.

Golden rule: source data is NEVER modified. Only the display offset changes.

---

## DATA MODELS

### MergedSession (models/merged_session.py)
```python
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

@dataclass
class SessionSource:
    source_id:         str         # unique UUID
    source_type:       str         # 'COMTRADE' | 'PMU_CSV' | 'CSV' | 'EXCEL'
    record:            object      # DisturbanceRecord or PmuRecord
    label:             str         # display name e.g. "Substation A — BEN32"
    colour_theme:      str         # hex base colour for this source
    time_offset_s:     float = 0.0 # seconds to ADD to this source's timestamps
    gps_synchronised:  bool = False
    alignment_quality: str = 'MANUAL'  # 'CONFIRMED'|'ESTIMATED'|'MANUAL'
    visible:           bool = True

@dataclass
class MergedSession:
    session_id:           str
    session_name:         str
    reference_source_id:  str              # anchor — offset=0 always
    sources:              list[SessionSource] = field(default_factory=list)
    time_offsets:         dict[str, float] = field(default_factory=dict)
    alignment_method:     str = 'MANUAL'   # 'GPS_AUTO'|'CROSSCORR'|'MANUAL'|'HYBRID'

    def add_source(self, source: SessionSource) -> None:
        self.sources.append(source)
        self.time_offsets[source.source_id] = source.time_offset_s

    def get_display_time(self, source_id: str, raw_time_s: float) -> float:
        """Convert source-local time to merged session display time."""
        return raw_time_s + self.time_offsets.get(source_id, 0.0)

    def get_source_time(self, source_id: str, display_time_s: float) -> float:
        """Convert merged display time back to source-local time."""
        return display_time_s - self.time_offsets.get(source_id, 0.0)

    @property
    def overlap_window(self) -> tuple[float, float] | None:
        """Returns (start, end) of time covered by ALL sources, or None."""
        if len(self.sources) < 2:
            return None
        starts, ends = [], []
        for src in self.sources:
            rec = src.record
            t0 = self.get_display_time(src.source_id, float(rec.time_array[0]))
            t1 = self.get_display_time(src.source_id, float(rec.time_array[-1]))
            starts.append(t0)
            ends.append(t1)
        overlap_start = max(starts)
        overlap_end   = min(ends)
        return (overlap_start, overlap_end) if overlap_start < overlap_end else None
```

---

## TIME SYNC ENGINE (engine/time_sync_engine.py)

### Stage 1 — GPS Detection
```python
def detect_gps_sync(record) -> tuple[bool, str]:
    """
    Returns (is_gps_synced, confidence).
    A record is GPS-synced if its start_time has microsecond precision
    and the year is not a placeholder (9999).
    """
    if record.start_time.year == 9999:
        return False, 'PLACEHOLDER_TIMESTAMP'
    if record.start_time.microsecond != 0:
        return True, 'CONFIRMED'
    # Check HDR file for GPS sync mentions
    if hasattr(record, 'header_text') and record.header_text:
        hdr = record.header_text.upper()
        if any(k in hdr for k in ['GPS', 'IRIG', 'PTP', 'IEEE 1588', 'SYNCED']):
            return True, 'CONFIRMED_HDR'
    return False, 'UNCERTAIN'
```

### Stage 2 — Absolute Alignment
```python
from datetime import timedelta

def compute_absolute_offset(reference_record, source_record) -> float:
    """
    Returns offset in seconds to apply to source_record so its timestamps
    align with reference_record. offset is ADDED to source display times.
    """
    ref_start = reference_record.start_time
    src_start = source_record.start_time
    delta = (ref_start - src_start).total_seconds()
    return delta   # positive = source starts after reference
```

### Stage 3 — Cross-Correlation Alignment
```python
from scipy.signal import correlate
import numpy as np

def cross_correlate_alignment(
    ref_data: np.ndarray, ref_sample_rate: float,
    src_data: np.ndarray, src_sample_rate: float,
    window_duration_s: float = 1.0
) -> tuple[float, float]:
    """
    Estimates time offset between two records using cross-correlation.
    Returns (estimated_offset_s, correlation_quality_0_to_1).

    IMPORTANT: result is a SUGGESTION — must be presented to user for confirmation.
    Never apply automatically.
    """
    # Resample both to common rate for correlation only
    common_rate = min(ref_sample_rate, src_sample_rate)
    n_samples = int(window_duration_s * common_rate)

    # Take fault window (first n_samples around each record's midpoint)
    ref_mid = len(ref_data) // 2
    src_mid = len(src_data) // 2
    ref_win = ref_data[ref_mid:ref_mid + n_samples]
    src_win = src_data[src_mid:src_mid + n_samples]

    # Normalise
    ref_norm = (ref_win - np.mean(ref_win)) / (np.std(ref_win) + 1e-10)
    src_norm = (src_win - np.mean(src_win)) / (np.std(src_win) + 1e-10)

    # Cross-correlate
    xcorr = correlate(ref_norm, src_norm, mode='full')
    lags  = np.arange(-(len(src_norm)-1), len(ref_norm))
    best_lag  = lags[np.argmax(xcorr)]
    quality   = float(np.max(xcorr) / len(ref_norm))  # 0=no correlation, 1=perfect
    offset_s  = best_lag / common_rate

    return offset_s, min(quality, 1.0)
```

### Full Alignment Pipeline
```python
def align_session(session: MergedSession) -> MergedSession:
    """
    Stages 1-3. Stage 4 (user confirmation) handled by MergeManagerDialog.
    Returns session with offsets populated but NOT yet confirmed.
    """
    ref_src = next(s for s in session.sources if s.source_id == session.reference_source_id)
    ref_rec = ref_src.record

    for src in session.sources:
        if src.source_id == session.reference_source_id:
            session.time_offsets[src.source_id] = 0.0
            src.alignment_quality = 'CONFIRMED'
            continue

        gps_ok, _ = detect_gps_sync(src.record)
        ref_gps, _ = detect_gps_sync(ref_rec)

        if gps_ok and ref_gps:
            # Stage 2: absolute time alignment
            offset = compute_absolute_offset(ref_rec, src.record)
            session.time_offsets[src.source_id] = offset
            src.alignment_quality = 'CONFIRMED'
            src.gps_synchronised = True
        else:
            # Stage 3: cross-correlation estimate
            try:
                ch_ref = ref_rec.analogue_channels[0].raw_data
                ch_src = src.record.analogue_channels[0].raw_data
                offset, quality = cross_correlate_alignment(
                    ch_ref, ref_rec.sample_rate,
                    ch_src, src.record.sample_rate
                )
                session.time_offsets[src.source_id] = offset
                src.alignment_quality = 'ESTIMATED'
            except Exception:
                session.time_offsets[src.source_id] = 0.0
                src.alignment_quality = 'MANUAL'

    return session
```

---

## CANVAS RENDERING RULES FOR MERGED VIEW

```python
# Source group colours (assign in order, avoid same colour for adjacent sources)
SOURCE_COLOURS = ['#2E75B6', '#C05020', '#208040', '#8030A0', '#C09010']

# Each source group rendered with its own colour theme
# Channels within a group keep standard phase colours (R/Y/B)
# but with a subtle tint from the source colour

# Shared time axis: ALL plots linked to same ViewBox X range
# Plot.setXLink(reference_plot) for every plot in every source group

# Overlap region indicator on the time ruler
overlap = session.overlap_window
if overlap:
    region = pg.LinearRegionItem(
        values=overlap,
        brush=pg.mkBrush('#00FF0015'),  # very subtle green
        movable=False
    )
    time_ruler.addItem(region)

# Source group header bar — QLabel with coloured background
header = QLabel(f"  {source.label}  |  {source.alignment_quality}")
header.setStyleSheet(f"background: {source.colour_theme}40; color: #DDDDDD; padding: 4px;")
```

---

## MERGE MANAGER DIALOG REQUIREMENTS

```python
# The dialog must:
# 1. Show all loaded sources in a table (label, type, GPS status, quality badge)
# 2. Show per-source offset in ms (editable QDoubleSpinBox, range ±60000ms)
# 3. Show a miniature waveform preview panel updating in real-time as offsets change
# 4. Show overlap window duration in the dialog footer
# 5. Have Confirm and Cancel buttons
# 6. On Confirm: commit offsets to session.time_offsets and close
# 7. On Cancel: discard all offsets, session not modified

# Quality badge colours:
QUALITY_COLOURS = {
    'CONFIRMED':  '#44BB44',   # green
    'ESTIMATED':  '#FFAA00',   # amber
    'MANUAL':     '#888888',   # grey
}
```
