"""
src/ui/help_dialog.py

User Guide dialog — Help > User Guide (F1).

Two-panel layout:
  Left  — QListWidget topic navigator
  Right — QTextBrowser with self-contained HTML per topic

Architecture: Presentation layer only (LAW 1 — no engine/parser imports).
"""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QListWidget,
    QListWidgetItem,
    QSplitter,
    QTextBrowser,
    QVBoxLayout,
)
from PyQt6.QtCore import Qt


# ── Shared CSS injected into every page ──────────────────────────────────────

_CSS = """
<style>
body  { font-family: 'Segoe UI', Arial, sans-serif; font-size: 9pt;
        color: #DDDDDD; background-color: #1E1E1E; margin: 14px 18px; }
h2    { color: #5BA4FF; margin-top: 0; margin-bottom: 6px;
        border-bottom: 1px solid #3A3A3A; padding-bottom: 5px; font-size: 13pt; }
h3    { color: #80C8FF; margin-top: 14px; margin-bottom: 3px; font-size: 10pt; }
h4    { color: #AACCEA; margin-top: 10px; margin-bottom: 2px; font-size: 9pt; }
p     { margin-top: 3px; margin-bottom: 6px; line-height: 1.55; }
code  { background-color: #2A2A2A; color: #F0C060; padding: 1px 5px;
        font-family: Consolas, 'Courier New', monospace; font-size: 8.5pt; }
ul    { margin-top: 2px; margin-bottom: 8px; padding-left: 18px; }
ol    { margin-top: 2px; margin-bottom: 8px; padding-left: 20px; }
li    { margin-bottom: 4px; line-height: 1.5; }
table { border-collapse: collapse; width: 100%; margin: 8px 0 12px 0; }
th    { background-color: #2D3A4A; color: #80C8FF;
        text-align: left; padding: 5px 10px; font-weight: bold; }
td    { padding: 4px 10px; border-bottom: 1px solid #282828; color: #CCCCCC; }
tr:nth-child(even) td { background-color: #222222; }
.tip  { background-color: #182818; border-left: 3px solid #55BB55;
        padding: 7px 12px; margin: 10px 0; color: #BBEEBB; }
.note { background-color: #192030; border-left: 3px solid #5BA4FF;
        padding: 7px 12px; margin: 10px 0; }
.warn { background-color: #281818; border-left: 3px solid #FF5555;
        padding: 7px 12px; margin: 10px 0; color: #FFBBBB; }
hr    { border: none; border-top: 1px solid #333; margin: 12px 0; }
b     { color: #EEEEEE; }
</style>
"""


def _page(title: str, body: str) -> str:
    return f'<html><head>{_CSS}</head><body><h2>{title}</h2>{body}</body></html>'


# ── Topic pages ───────────────────────────────────────────────────────────────

_GETTING_STARTED = _page("Getting Started", """
<p>PowerWave Analyst is an offline desktop tool for analysing power system disturbance
records. It loads COMTRADE, PMU CSV and other formats and renders waveforms on a
shared time canvas for fault investigation.</p>

<h3>Quick start — 3 steps</h3>
<ol>
  <li>Open one or more disturbance record files: <code>File &gt; Open</code>
      or <b>Ctrl+O</b>.</li>
  <li>Channels appear automatically in stacked plots (Voltage/Current, Freq/Power, etc.)
      grouped by signal type. Check or uncheck channels in the file panel on the left.</li>
  <li>Right-click any plot for zoom options. Drag the <b>C1 / C2 cursors</b> to read
      instantaneous values in the Measurements panel on the right.</li>
</ol>

<h3>Supported file formats</h3>
<table>
  <tr><th>Format</th><th>Extension</th><th>Notes</th></tr>
  <tr><td>COMTRADE 1991 / 1999 / 2013</td><td>.cfg + .dat</td>
      <td>BEN32, NARI, ABB, Siemens, GE, SEL and all standard-compliant IEDs</td></tr>
  <tr><td>PMU CSV (Malaysian grid)</td><td>.csv</td>
      <td>Header row starting with <code>ID:</code>; time column
          <code>Time (Asia/Singapore)</code></td></tr>
  <tr><td>Generic CSV</td><td>.csv</td><td>Auto-detects delimiter, time column, units</td></tr>
  <tr><td>Excel</td><td>.xlsx / .xls</td><td>First sheet with numeric time column</td></tr>
</table>

<div class="tip">
<b>Tip:</b> You can open multiple files of different formats simultaneously.
Each file's channels are overlaid on the same canvas so you can compare
fault records from different bays or substations.
</div>
""")


_WINDOW_LAYOUT = _page("Window Layout", """
<h3>Main areas</h3>
<table>
  <tr><th>Area</th><th>Location</th><th>Purpose</th></tr>
  <tr><td><b>Canvas</b></td><td>Centre</td>
      <td>Waveform plots arranged in stacks (Voltage/Current, Freq/Power, DC Field,
          Mechanical, Generic). Each canvas group has an independent time axis.</td></tr>
  <tr><td><b>File panel</b></td><td>Centre-left</td>
      <td>Tree of loaded files and their channels. Check/uncheck to show/hide.
          Right-click for per-file and per-channel actions.</td></tr>
  <tr><td><b>Offset strip</b></td><td>Bottom of canvas area</td>
      <td>One row per loaded file with a time-offset slider for fine alignment.</td></tr>
  <tr><td><b>Measurements</b></td><td>Right dock</td>
      <td>C1/C2 cursor timestamps, per-channel values at each cursor, and ΔX.</td></tr>
  <tr><td><b>RMS Converter</b></td><td>Bottom dock (hidden)</td>
      <td>Cycle-by-cycle RMS engine. Open with <code>Tools &gt; RMS Converter</code>
          or <b>Ctrl+R</b>.</td></tr>
</table>

<h3>Signal stacks</h3>
<p>Channels are sorted into five stacks automatically by signal role:</p>
<table>
  <tr><th>Stack</th><th>Left Y-axis</th><th>Right Y-axis</th></tr>
  <tr><td>0 — Voltage / Current</td><td>Voltage (kV or pu)</td><td>Current (A/kA)</td></tr>
  <tr><td>1 — Frequency / Power</td><td>Frequency (Hz)</td><td>Active power (MW)</td></tr>
  <tr><td>2 — DC Field</td><td>Field voltage (V)</td><td>Field current (A)</td></tr>
  <tr><td>3 — Mechanical</td><td>Speed (RPM)</td><td>Valve / gate (%)</td></tr>
  <tr><td>4 — Generic / Other</td><td colspan="2">All unclassified analogue channels</td></tr>
</table>
<p>Digital (binary) channels appear below all analogue stacks in a shared panel.</p>

<h3>Canvas group sections</h3>
<p>When files with significantly different timestamps are loaded, they are placed in
separate <b>canvas group sections</b> — each with its own time axis. See
<b>Canvas Groups</b> for details.</p>
""")


_CANVAS_GROUPS = _page("Canvas Groups", """
<p>A <b>canvas group</b> is a set of files that share a common time axis. Files in the
same group are time-aligned to each other; files in different groups have independent
axes and appear in separate canvas sections.</p>

<h3>Automatic grouping rules</h3>
<ol>
  <li>When a file is loaded, its start timestamp (UTC epoch) is compared to every
      existing group.</li>
  <li>If the file's epoch is within the <b>grouping threshold</b> (default: 1 hour)
      of an existing group, it joins that group.</li>
  <li>If no matching group exists, a new section is created below the existing ones.</li>
  <li>Files with <b>no valid timestamp</b> (epoch ≤ 86400, i.e. before 1970-01-02)
      always get their own group — they cannot be auto-matched.</li>
</ol>

<div class="note">
<b>Group threshold</b> can be changed in
<code>Edit &gt; Preferences &gt; Calculation &gt; Grouping threshold</code>.
Changing it will regroup all currently loaded files immediately.
</div>

<h3>Moving a file to a different group</h3>
<p>Right-click the <b>file row</b> in the file panel → <b>Move to Group…</b> →
choose the target group or <i>New group (separate canvas)</i>.</p>

<h3>Detaching a group to a floating window</h3>
<p>Click the <b>⊟</b> button in the group header to detach the entire canvas section
to a separate floating window — useful when you have a second monitor. Click the
same button (now labelled <b>⊞</b>) inside the floating window to re-attach it.</p>

<h3>When files land in separate groups by mistake</h3>
<p>The most common cause is a <b>broken or missing timestamp</b>. For PMU files with
GPS faults the time column shows a broken format (e.g. <code>12:00.0</code> — only
one colon). The parser detects this and sets the epoch to zero, forcing a new group.</p>
<p><b>Fix:</b> right-click the file → <b>Set Start Time…</b> → enter the correct
event date and UTC time. Once the epoch is valid and within the threshold of another
group, use <b>Move to Group…</b> to place them together, then use the offset slider
or Auto-align for fine-tuning.</p>
""")


_LOADING_FILES = _page("Loading Files", """
<h3>Opening a file</h3>
<p>Use <code>File &gt; Open</code> (<b>Ctrl+O</b>). Multiple files can be opened in
sequence; each is added to the canvas without removing the others.</p>

<h3>COMTRADE files</h3>
<p>Select either the <code>.cfg</code> or the <code>.dat</code> file — the parser
automatically locates its companion. Both files must be in the same directory.</p>

<h3>PMU CSV files (Malaysian grid format)</h3>
<p>These files begin with a metadata row (<code>ID: NNN, Station Name: …</code>) and
have a time column named <code>Time (Asia/Singapore)</code>. They are recognised
automatically. Voltage magnitudes are in raw Volts and are divided by 1000 for kV
display.</p>

<div class="warn">
<b>PMU GPS fault — broken timestamps:</b> Some PMU devices report time as
<code>MM:SS.s</code> (one colon, no hours) when the GPS lock is lost. PowerWave
detects this and shows the <b>PMU Import</b> dialog asking for a manual anchor time.
Enter the event date and approximate time in UTC; the parser then reconstructs a
synthetic timeline at 50 samples/second from that anchor.
</div>

<h3>Import dialog — PMU files</h3>
<p>When a PMU file is loaded with quality issues, a dialog lists the detected problems
(Blockers / Warnings). For broken timestamps you must supply an anchor time before the
file can be used. Clicking <b>Cancel</b> loads the file with epoch = 0, placing it in
its own canvas group with a synthetic timeline starting at t = 0.</p>

<h3>Removing a file</h3>
<p>Right-click the file row in the file panel → <b>Remove file</b>. All its curves are
removed from the canvas and its offset row is deleted from the strip.</p>
""")


_CHANNELS_MODES = _page("Channels & Display Modes", """
<h3>File and channel tree</h3>
<p>The file panel shows one top-level item per loaded file. Expand it to see all
analogue channels grouped by signal role. Check the box next to any channel to
show or hide it on the canvas.</p>

<h3>Per-channel display mode</h3>
<p>Each analogue channel has three display modes, selectable by right-clicking the
channel row:</p>
<table>
  <tr><th>Mode</th><th>What is shown</th></tr>
  <tr><td><b>Raw</b></td><td>Instantaneous sample values directly from the file
      (default). For COMTRADE files this is the primary-side physical value after
      applying multiplier and offset.</td></tr>
  <tr><td><b>RMS</b></td><td>Cycle-by-cycle RMS envelope computed in the background.
      The first toggle may take a moment; result is cached. Not available for
      TREND-mode channels (sample rate &lt; 200 Hz).</td></tr>
  <tr><td><b>Value (locked)</b></td><td>Locks the display to the instantaneous value
      at C1 cursor position (useful for comparing snapshots).</td></tr>
</table>

<h3>Waveform vs Trend display</h3>
<p>Display mode is determined automatically by sample rate:</p>
<table>
  <tr><th>Sample rate</th><th>Display</th><th>Notes</th></tr>
  <tr><td>≥ 200 Hz</td><td>Continuous line (WAVEFORM)</td>
      <td>Decimated to max 2000 points per render. RMS available.</td></tr>
  <tr><td>&lt; 200 Hz</td><td>Scatter dots (TREND)</td>
      <td>Already cycle-averaged or slower. No RMS toggle. No FFT.</td></tr>
</table>

<h3>Per-channel scatter override</h3>
<p>Any waveform-mode channel can be switched to scatter display: right-click the
channel row → <b>Display as Scatter</b> (toggle). This persists until unchecked.</p>

<h3>Voltage convention (per file)</h3>
<p>Right-click a <b>file row</b> → <b>Voltage Convention</b> to specify whether the
voltage channel values in that file are line-to-line or line-to-earth. This affects
the PU divisor when PU mode is active. The nominal kV spinbox always expects the
<b>line-to-line</b> value regardless of convention.</p>
""")


_TIME_OFFSET = _page("Time Alignment & Offset", """
<p>When two or more files are loaded into the <b>same canvas group</b>, they are
initially positioned according to their embedded timestamps. For most well-formed
COMTRADE or PMU files this is sufficient. The offset slider provides sub-minute
fine-tuning on top of the timestamp alignment.</p>

<h3>How file time positioning works</h3>
<p>Each file has a <b>start epoch</b> (UTC seconds since 1970-01-01). When a group
is rendered, the earliest epoch in the group becomes <b>t = 0</b>. Every other file
is shifted right by <code>(its epoch − group epoch)</code> seconds. The offset slider
adds a further adjustment on top of that.</p>

<p>Formula: <code>t_displayed = t_raw + (file_epoch − group_epoch) + offset_s</code></p>

<h3>The offset slider</h3>
<p>The Offset strip at the bottom of the canvas area has one row per loaded file.
Each row contains:</p>
<ul>
  <li><b>− / + buttons</b> — step left or right by one step increment</li>
  <li><b>Slider</b> — drag to adjust; range is ±3000 steps</li>
  <li><b>Offset label</b> — current offset in milliseconds</li>
  <li><b>Step spinner</b> — size of one slider step (default 10 ms)</li>
  <li><b>Freq selector</b> — nominal frequency used for RMS calculation (50 / 60 Hz)</li>
</ul>

<h4>Maximum offset range</h4>
<table>
  <tr><th>Step size</th><th>Max range (±)</th><th>Use case</th></tr>
  <tr><td>10 ms (default)</td><td>30 s</td><td>Sub-cycle fine-tuning</td></tr>
  <tr><td>100 ms</td><td>300 s (5 min)</td><td>Moderate alignment</td></tr>
  <tr><td>1 000 ms</td><td>3 000 s (50 min)</td><td>Large time gaps</td></tr>
  <tr><td>10 000 ms</td><td>30 000 s (8.3 h)</td><td>Cross-timezone adjustment</td></tr>
</table>

<div class="note">
<b>Important:</b> The offset slider only affects files in the <b>same canvas group</b>.
If two files are in different group sections (different canvases), their sliders operate
on independent time axes and cannot align them with each other. You must first put both
files in the same group — see <b>Canvas Groups</b>.
</div>

<div class="warn">
<b>Common mistake — files in separate groups:</b>
If the offset slider appears to have no effect, check whether the two files are in the
same canvas group section. A PMU file with broken GPS timestamps will always land in its
own group (epoch = 0). Fix: right-click the file → <b>Set Start Time…</b> → enter
the correct event time, then use <b>Move to Group…</b> to merge them.
</div>

<h3>Auto-align from frequency (recommended for PMU vs COMTRADE)</h3>
<p>Right-click a file row → <b>Auto-align from Frequency</b> then select the reference
file. PowerWave computes the FFT cross-correlation of both frequency channels and
applies the resulting lag as the offset automatically. This is the most reliable method
when both files contain a frequency channel recording the same event.</p>

<h3>Setting start time manually</h3>
<p>Right-click a file row → <b>Set Start Time…</b> to open the time-entry dialog.
Enter the event date and UTC time. This corrects the file's start epoch, which may
cause it to be re-grouped with other files if the new epoch is within the grouping
threshold.</p>

<h3>COMTRADE timezone offset</h3>
<p>COMTRADE files store <b>local wall-clock time</b> — there is no timezone field in
the standard. If your substation uses a local timezone (e.g. MYT/SGT UTC+8), set
<code>Edit &gt; Preferences &gt; Calculation &gt; COMTRADE timezone offset</code>
to the correct UTC offset. PowerWave then subtracts this offset to convert the
stored timestamp to UTC before grouping and alignment.</p>
<p>Example: A Malaysian substation COMTRADE file timestamped 18:04 MYT is actually
10:04 UTC. Without the offset set to UTC+8, it will appear 8 hours away from a PMU
CSV file already in UTC.</p>
""")


_CURSORS = _page("Cursors & Measurements", """
<h3>Enabling cursors</h3>
<p>Right-click any waveform plot → <b>Enable C1 Cursor</b> or <b>Enable C2 Cursor</b>.
A vertical line appears across all stacks in that canvas group. Right-click again to
disable.</p>
<ul>
  <li><b>C1</b> — gold vertical line</li>
  <li><b>C2</b> — cyan vertical line</li>
</ul>

<h3>Moving cursors</h3>
<p>Click and drag the coloured vertical line. The cursor snaps to the nearest sample
on mouse release. You can also left-click anywhere on the plot to jump C1 to that
position.</p>

<h3>Readout overlay</h3>
<p>A small floating panel attached to C1 shows the times of both cursors and ΔX
(time difference C2 − C1). Drag this panel to reposition it if it overlaps waveforms.</p>

<h3>Measurements panel</h3>
<p>The <b>Measurements</b> dock on the right shows:</p>
<ul>
  <li>C1 and C2 absolute timestamps</li>
  <li>One row per visible channel with values at C1, C2, and their difference (Δ)</li>
  <li>Unit column</li>
</ul>
<p>Font size of the panel is configurable under
<code>Edit &gt; Preferences &gt; Display &gt; Panel text size</code>.</p>

<h3>Zoom and pan</h3>
<table>
  <tr><th>Action</th><th>Effect</th></tr>
  <tr><td>Mouse wheel</td><td>Zoom in/out on X axis (time)</td></tr>
  <tr><td>Right-click drag</td><td>Pan the time axis</td></tr>
  <tr><td>Right-click → Zoom to Fit (this group)</td><td>Reset X to show all data
      in the current canvas group</td></tr>
  <tr><td>Right-click → Zoom to Fit (all groups)</td><td>Reset X in every group
      independently</td></tr>
</table>
""")


_PHASOR = _page("Phasor Display", """
<p>The Phasor Display shows instantaneous phasor arrows at the C1 cursor position
for all visible voltage and current channels.</p>

<h3>Opening the phasor view</h3>
<p>Click the <b>Phasor Display</b> button in the canvas group header toolbar. A
floating dialog opens showing:</p>
<ul>
  <li>A polar arrow canvas (magnitude = radius, angle = direction)</li>
  <li>A value table (channel name, magnitude, angle)</li>
</ul>
<p>The dialog is moveable and resizable. Close it with the window close button.</p>

<div class="note">
<b>Note:</b> Phasor display currently uses the raw instantaneous sample at the cursor
for COMTRADE waveform files. For PMU CSV files it uses the pre-computed positive
sequence magnitude and angle directly from the file columns.
</div>
""")


_PU_MODE = _page("Voltage PU Mode", """
<p><b>Per-unit (PU)</b> mode normalises voltage channels to the nominal system voltage,
making it easy to compare per-unit voltage across different voltage levels simultaneously.</p>

<h3>Enabling PU mode</h3>
<p>Click the <b>PU</b> button in the canvas group toolbar. All voltage channels switch
to per-unit values. The Y-axis range becomes ±2.0 pu (configurable in Preferences).</p>

<h3>Nominal kV spinbox</h3>
<p>Each file has a <b>Nominal kV</b> spinbox in the channel tree (rightmost column).
Always enter the <b>line-to-line</b> nominal voltage (e.g. 275 for a 275 kV bus)
regardless of what the file contains.</p>

<h3>Voltage convention</h3>
<p>Right-click a file row → <b>Voltage Convention</b> to specify how the file's
voltage values are stored:</p>
<table>
  <tr><th>Convention</th><th>Badge</th><th>PU divisor</th></tr>
  <tr><td>Line-to-Line (default)</td><td><code>[L-L]</code></td>
      <td>Nominal kV directly</td></tr>
  <tr><td>Line-to-Earth (phase-to-earth)</td><td><code>[L-E]</code></td>
      <td>Nominal kV ÷ √3</td></tr>
</table>

<div class="tip">
<b>Example:</b> A 275 kV COMTRADE file storing line-to-earth voltages.
Set Nominal kV = 275, convention = Line-to-Earth. The divisor becomes 275 ÷ √3 ≈ 158.8 kV,
so a healthy phase reading of ~158.8 kV shows as 1.00 pu.
</div>

<p>The PU Y-axis range (default ±2.0 pu) can be changed in
<code>Edit &gt; Preferences &gt; Calculation &gt; PU Y-axis range</code>.</p>
""")


_RMS_CONVERTER = _page("RMS Converter", """
<p>The RMS Converter is a dedicated analysis dock for computing cycle-by-cycle RMS
envelopes, merging multiple files onto a common time axis, and exporting to CSV or
Excel.</p>

<h3>Opening the RMS Converter</h3>
<p><code>Tools &gt; RMS Converter</code> or <b>Ctrl+R</b>. The dock appears at the
bottom of the window. It can be undocked and resized independently.</p>

<h3>Loading files</h3>
<p>Use the <b>Add File</b> button inside the dock. Files loaded here are independent
of the main canvas — the same file can be open in both.</p>

<h3>RMS calculation</h3>
<p>Cycle-by-cycle RMS is computed using a sliding window of exactly one nominal cycle
(1/f seconds). The nominal frequency per file is set with the Freq selector in the
offset strip row.</p>

<h3>Offset and alignment</h3>
<p>Each file in the RMS Converter has its own offset row (same design as the main
canvas). Time alignment rules are identical — offset only works between files in the
same dock, and all files in the dock share one time axis.</p>

<h3>PU mode</h3>
<p>The <b>PU</b> button in the RMS Converter toolbar applies PU normalisation to
voltage channels using the nominal kV spinbox value for each file. The PU Y-range
setting from Preferences applies here too.</p>

<h3>Dual cursors</h3>
<p>C1 and C2 cursors in the RMS Converter are independent of the main canvas cursors.
The readout overlay shows the same information.</p>

<h3>Export</h3>
<p>Click <b>Export CSV</b> or <b>Export Excel</b> to save the merged RMS table to a
file. The export contains one row per common time point with one column per channel
across all loaded files.</p>
""")


_PREFERENCES = _page("Preferences", """
<p>Open with <code>Edit &gt; Preferences</code> or <b>Ctrl+,</b>.</p>

<h3>Calculation</h3>
<table>
  <tr><th>Setting</th><th>Default</th><th>Effect</th></tr>
  <tr><td>Nominal frequency</td><td>50 Hz</td>
      <td>Used for RMS window size and phasor calculation</td></tr>
  <tr><td>RMS merge tolerance</td><td>10 ms</td>
      <td>Maximum time gap when merging RMS timestamps across files</td></tr>
  <tr><td>PU Y-axis range</td><td>±2.0 pu</td>
      <td>Y-axis limits when PU mode is active (main canvas and RMS Converter)</td></tr>
  <tr><td>COMTRADE timezone offset</td><td>UTC (0 h)</td>
      <td>UTC offset of COMTRADE wall-clock timestamps. Set to UTC+8 for Malaysian /
          Singapore substations (MYT/SGT). Reload files after changing.</td></tr>
  <tr><td>Grouping threshold</td><td>1.0 h</td>
      <td>Files within this time window of each other are placed in the same canvas
          group. Changing this immediately regroups all loaded files.</td></tr>
</table>

<h3>Display</h3>
<table>
  <tr><th>Setting</th><th>Default</th><th>Effect</th></tr>
  <tr><td>Theme</td><td>Dark</td><td>Dark or light application colour scheme</td></tr>
  <tr><td>C1 cursor colour</td><td>Gold (#FFD700)</td><td>Colour of the C1 vertical line</td></tr>
  <tr><td>C2 cursor colour</td><td>Cyan (#00E5FF)</td><td>Colour of the C2 vertical line</td></tr>
  <tr><td>Panel text size</td><td>9 pt</td>
      <td>Font size in the Measurements dock (range 7–14 pt). Takes effect on Apply.</td></tr>
</table>

<h3>PMU</h3>
<table>
  <tr><th>Setting</th><th>Default</th><th>Effect</th></tr>
  <tr><td>Default timezone</td><td>SGT (UTC+8)</td>
      <td>Default timezone pre-filled in the PMU Import dialog for broken-timestamp files</td></tr>
</table>

<h3>Buttons</h3>
<ul>
  <li><b>OK</b> — save changes and close</li>
  <li><b>Apply</b> — save changes without closing</li>
  <li><b>Cancel</b> — discard changes</li>
  <li><b>Restore Defaults</b> — reset all fields to factory values (does not save
      until OK or Apply)</li>
</ul>
""")


_SHORTCUTS = _page("Keyboard Shortcuts & Mouse Controls", """
<h3>Global shortcuts</h3>
<table>
  <tr><th>Key</th><th>Action</th></tr>
  <tr><td><code>Ctrl+O</code></td><td>Open file</td></tr>
  <tr><td><code>Ctrl+R</code></td><td>Toggle RMS Converter dock</td></tr>
  <tr><td><code>Ctrl+,</code></td><td>Open Preferences</td></tr>
  <tr><td><code>F1</code></td><td>Open User Guide (this dialog)</td></tr>
  <tr><td><code>Ctrl+Q</code></td><td>Exit application</td></tr>
</table>

<h3>Canvas — mouse</h3>
<table>
  <tr><th>Action</th><th>Effect</th></tr>
  <tr><td>Scroll wheel</td><td>Zoom in / out on time axis</td></tr>
  <tr><td>Left-click drag on plot</td><td>Move C1 cursor to click position</td></tr>
  <tr><td>Left-click drag on cursor line</td><td>Drag cursor</td></tr>
  <tr><td>Right-click drag on plot</td><td>Pan time axis</td></tr>
  <tr><td>Right-click on plot (no cursor)</td><td>Context menu: zoom, cursor enable</td></tr>
  <tr><td>Drag readout overlay</td><td>Move the cursor value box</td></tr>
</table>

<h3>Canvas — right-click menu (plot area)</h3>
<table>
  <tr><th>Item</th><th>Description</th></tr>
  <tr><td>Enable C1 / C2 Cursor</td><td>Show or hide a measurement cursor</td></tr>
  <tr><td>Zoom to Fit (this group)</td><td>Reset X range to show all data in this group</td></tr>
  <tr><td>Zoom to Fit (all groups)</td><td>Reset X range in every canvas group</td></tr>
</table>

<h3>File panel — right-click menu (file row)</h3>
<table>
  <tr><th>Item</th><th>Description</th></tr>
  <tr><td>Set Start Time…</td><td>Manually enter the file's UTC start time</td></tr>
  <tr><td>Auto-align from Frequency</td><td>Cross-correlate frequency channels
      to compute alignment offset</td></tr>
  <tr><td>Voltage Convention</td><td>Set line-to-line or line-to-earth for PU divisor</td></tr>
  <tr><td>Move to Group…</td><td>Move this file to a different canvas group</td></tr>
  <tr><td>Remove file</td><td>Unload the file from the canvas</td></tr>
</table>

<h3>File panel — right-click menu (channel row)</h3>
<table>
  <tr><th>Item</th><th>Description</th></tr>
  <tr><td>Raw / RMS / Value (locked)</td><td>Switch display mode for this channel</td></tr>
  <tr><td>Display as Scatter</td><td>Toggle scatter vs line rendering</td></tr>
</table>
""")


# ── Topic registry ────────────────────────────────────────────────────────────

_TOPICS: list[tuple[str, str]] = [
    ("Getting Started",            _GETTING_STARTED),
    ("Window Layout",              _WINDOW_LAYOUT),
    ("Canvas Groups",              _CANVAS_GROUPS),
    ("Loading Files",              _LOADING_FILES),
    ("Channels & Display Modes",   _CHANNELS_MODES),
    ("Time Alignment & Offset",    _TIME_OFFSET),
    ("Cursors & Measurements",     _CURSORS),
    ("Phasor Display",             _PHASOR),
    ("Voltage PU Mode",            _PU_MODE),
    ("RMS Converter",              _RMS_CONVERTER),
    ("Preferences",                _PREFERENCES),
    ("Keyboard Shortcuts",         _SHORTCUTS),
]


# ── Dialog ────────────────────────────────────────────────────────────────────

class HelpDialog(QDialog):
    """User Guide dialog — left nav list, right QTextBrowser.

    Opened from Help > User Guide (F1).
    """

    def __init__(self, parent=None, initial_topic: int = 0) -> None:
        """Initialise the help dialog.

        Args:
            parent:        Parent widget (typically MainWindow).
            initial_topic: Index into _TOPICS for the page shown on open.
        """
        super().__init__(parent)
        self.setWindowTitle("PowerWave Analyst — User Guide")
        self.resize(980, 680)
        self.setMinimumSize(700, 480)

        self._topics = _TOPICS
        self._build_ui()
        self._nav.setCurrentRow(initial_topic)

    # ── UI construction ───────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(4)

        # ── Left nav ──────────────────────────────────────────────────────
        self._nav = QListWidget()
        self._nav.setFixedWidth(190)
        self._nav.setStyleSheet(
            "QListWidget { background: #1A1A1A; color: #CCCCCC; "
            "border: 1px solid #333; font-size: 9pt; }"
            "QListWidget::item:selected { background: #2A4A7A; color: #FFFFFF; }"
            "QListWidget::item:hover { background: #2A2A2A; }"
        )
        for title, _ in self._topics:
            item = QListWidgetItem(title)
            self._nav.addItem(item)
        self._nav.currentRowChanged.connect(self._on_topic_changed)
        splitter.addWidget(self._nav)

        # ── Right browser ─────────────────────────────────────────────────
        self._browser = QTextBrowser()
        self._browser.setReadOnly(True)
        self._browser.setOpenLinks(False)
        self._browser.setStyleSheet(
            "QTextBrowser { background: #1E1E1E; color: #DDDDDD; "
            "border: 1px solid #333; }"
        )
        splitter.addWidget(self._browser)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter, stretch=1)

        # ── Close button ──────────────────────────────────────────────────
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.accept)
        root.addWidget(buttons)

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _on_topic_changed(self, row: int) -> None:
        if 0 <= row < len(self._topics):
            _title, html = self._topics[row]
            self._browser.setHtml(html)
            self._browser.verticalScrollBar().setValue(0)
