"""
src/ui/pmu_import_dialog.py

Two dialogs for PMU CSV timestamp management:

  PmuImportDialog    — shown at import time when broken/ambiguous timestamps
                       are detected.  Collects the actual start time anchor
                       from the user so the file is correctly placed on the
                       shared time axis.

  SetStartTimeDialog — lightweight post-load dialog accessible via right-click
                       → "Set Start Time…".  Lets the user correct or override
                       a file's epoch without re-importing.

Architecture: Presentation layer (ui/) — imports parsers/ only (LAW 1).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from PyQt6.QtCore import QDate, QTime, Qt
from PyQt6.QtWidgets import (
    QComboBox,
    QDateEdit,
    QDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QCheckBox,
    QTimeEdit,
    QVBoxLayout,
    QWidget,
)

from parsers.pmu_import_validator import IssueKind, IssueSeverity, ParseInspectionReport

# SGT and MYT are both UTC+8 (Malaysia/Singapore standard time)
_TZ_OFFSETS: dict[str, int] = {
    'SGT (UTC+8)': 8,
    'MYT (UTC+8)': 8,
    'UTC':         0,
}

_DARK_STYLE = """
    QDialog       { background: #2A2A2A; }
    QLabel        { color: #DDDDDD; font-size: 8pt; }
    QGroupBox     { color: #AAAAAA; font-size: 8pt;
                    border: 1px solid #444; border-radius: 4px;
                    margin-top: 8px; padding-top: 10px; }
    QGroupBox::title { subcontrol-origin: margin; left: 8px; }
    QDateEdit, QTimeEdit, QComboBox {
                    background: #333; color: #DDD;
                    border: 1px solid #555; padding: 2px 4px; }
    QPushButton   { background: #3A3A3A; color: #DDD;
                    border: 1px solid #555; padding: 4px 14px;
                    border-radius: 3px; font-size: 8pt; }
    QPushButton:hover  { background: #4A4A4A; }
    QPushButton:default { border: 1px solid #6688CC; }
    QCheckBox     { color: #BBBBBB; font-size: 8pt; }
"""


# ── PmuImportDialog ───────────────────────────────────────────────────────────

class PmuImportDialog(QDialog):
    """Import-time verification dialog for PMU CSV files with data anomalies.

    Shows BLOCKER issues (requiring user input) and INFO/auto-resolved items.
    The user supplies the actual recording start time when the file's timestamp
    is broken or missing.

    Usage::

        hints = [('275BAHS_KAWA1', '16:12:00 SGT')]
        dlg = PmuImportDialog(report, hints, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            anchor_utc = dlg.anchor_utc   # UTC datetime for t[0]
    """

    def __init__(
        self,
        report:           ParseInspectionReport,
        other_file_hints: list[tuple[str, str]],
        parent:           Optional[QWidget] = None,
    ) -> None:
        """Build the dialog.

        Args:
            report:           ParseInspectionReport from pmu_import_validator.
            other_file_hints: List of (filename, local_time_str) for files
                              already loaded on the same date — shown as hints.
            parent:           Optional parent widget.
        """
        super().__init__(parent)
        self.setWindowTitle('PMU Import — Verification Required')
        self.setMinimumWidth(520)
        self.setModal(True)

        self._anchor_utc: Optional[datetime] = None
        self._date_edit:  Optional[QDateEdit] = None
        self._time_edit:  Optional[QTimeEdit] = None
        self._tz_combo:   Optional[QComboBox] = None

        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(14, 12, 14, 12)

        # ── Header ────────────────────────────────────────────────────────────
        hdr = QLabel(
            f'<b>File:</b>&nbsp;{report.filepath.name}&nbsp;&nbsp;&nbsp;'
            f'<b>Station:</b>&nbsp;{report.station_name}'
        )
        hdr.setStyleSheet('font-size: 9pt; color: #DDDDDD;')
        root.addWidget(hdr)
        root.addWidget(_hline())

        # ── Blocker issues ────────────────────────────────────────────────────
        blockers = [i for i in report.issues if i.severity == IssueSeverity.BLOCKER]
        if blockers:
            warn = QLabel(
                f'⚠&nbsp;&nbsp;<b>{len(blockers)} issue(s) require your input</b> '
                f'before this file can be placed correctly on the shared time axis.'
            )
            warn.setStyleSheet('color: #FFAA44; font-size: 9pt;')
            warn.setWordWrap(True)
            root.addWidget(warn)

            for issue in blockers:
                root.addWidget(
                    self._build_blocker_box(issue, report, other_file_hints)
                )

        # ── Auto-resolved items ───────────────────────────────────────────────
        if report.auto_resolved:
            ar_box = QGroupBox('ℹ  Auto-resolved  (no input needed)')
            ar_box.setStyleSheet(
                'QGroupBox { color: #88CC88; font-size: 8pt; '
                'border: 1px solid #336633; border-radius: 4px; '
                'margin-top: 8px; padding-top: 10px; }'
                'QGroupBox::title { subcontrol-origin: margin; left: 8px; }'
            )
            ar_layout = QVBoxLayout(ar_box)
            ar_layout.setSpacing(3)
            for item in report.auto_resolved:
                lbl = QLabel(f'• {item.description}')
                lbl.setStyleSheet('color: #AAAAAA; font-size: 8pt;')
                ar_layout.addWidget(lbl)
            root.addWidget(ar_box)

        root.addWidget(_hline())

        # ── Remember checkbox ─────────────────────────────────────────────────
        self._remember_cb = QCheckBox(
            f'Remember these corrections for station "{report.station_name}"'
        )
        self._remember_cb.setChecked(True)
        root.addWidget(self._remember_cb)

        # ── Buttons ───────────────────────────────────────────────────────────
        btn_row = QHBoxLayout()
        btn_row.addStretch()

        skip_btn = QPushButton('Import without time fix')
        skip_btn.setToolTip(
            'Import using synthetic t₀ = 0.\n'
            'The file will not align with other files until a start time\n'
            'is set later via right-click → "Set Start Time…".'
        )
        skip_btn.clicked.connect(self._on_skip)
        btn_row.addWidget(skip_btn)

        import_btn = QPushButton('Import')
        import_btn.setDefault(True)
        import_btn.setStyleSheet(
            'QPushButton { background: #2244AA; color: #EEE; '
            'border: 1px solid #4466CC; padding: 4px 18px; '
            'border-radius: 3px; font-weight: bold; font-size: 8pt; }'
            'QPushButton:hover { background: #3355BB; }'
        )
        import_btn.clicked.connect(self._on_import)
        btn_row.addWidget(import_btn)

        root.addLayout(btn_row)

        self.setStyleSheet(_DARK_STYLE)

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def anchor_utc(self) -> Optional[datetime]:
        """UTC anchor datetime for t[0]; None when the user skipped."""
        return self._anchor_utc

    @property
    def remember(self) -> bool:
        """True if the user wants the correction saved as a profile."""
        return self._remember_cb.isChecked()

    # ── Issue box builders ────────────────────────────────────────────────────

    def _build_blocker_box(
        self,
        issue:   object,
        report:  ParseInspectionReport,
        hints:   list[tuple[str, str]],
    ) -> QGroupBox:
        """Build a QGroupBox for one BLOCKER issue."""
        title = issue.kind.value.replace('_', ' ').title()  # type: ignore[union-attr]
        box = QGroupBox(f'⚠  {title}')
        box.setStyleSheet(
            'QGroupBox { color: #FFAA44; font-size: 8pt; '
            'border: 1px solid #664400; border-radius: 4px; '
            'margin-top: 8px; padding-top: 10px; }'
            'QGroupBox::title { subcontrol-origin: margin; left: 8px; }'
        )
        layout = QVBoxLayout(box)
        layout.setSpacing(6)

        desc_lbl = QLabel(issue.description)  # type: ignore[union-attr]
        desc_lbl.setWordWrap(True)
        desc_lbl.setStyleSheet('color: #CCCCCC; font-size: 8pt;')
        layout.addWidget(desc_lbl)

        found_row = QHBoxLayout()
        found_row.addWidget(QLabel('Found in file:'))
        found_val = QLabel(
            f'<code>{report.first_date_str}&nbsp;&nbsp;{report.first_time_str}</code>'
        )
        found_val.setStyleSheet('color: #FF8888; font-size: 8pt;')
        found_row.addWidget(found_val)
        found_row.addStretch()
        layout.addLayout(found_row)

        if issue.kind == IssueKind.TIMESTAMP_BROKEN:  # type: ignore[union-attr]
            layout.addWidget(
                self._build_time_input(report.first_date_str, hints)
            )

        return box

    def _build_time_input(
        self,
        date_str: str,
        hints:    list[tuple[str, str]],
    ) -> QWidget:
        """Build the Date + Time + Timezone input widget."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(5)

        prompt = QLabel('Enter the actual start time for row 1 of this file:')
        prompt.setStyleSheet('color: #AAAAAA; font-size: 8pt;')
        layout.addWidget(prompt)

        row = QHBoxLayout()
        row.setSpacing(6)

        row.addWidget(QLabel('Date:'))
        self._date_edit = QDateEdit()
        self._date_edit.setDisplayFormat('MM/dd/yyyy')
        self._date_edit.setCalendarPopup(True)
        self._date_edit.setFixedWidth(115)
        qd = _parse_qdate(date_str)
        if qd.isValid():
            self._date_edit.setDate(qd)
        row.addWidget(self._date_edit)

        row.addSpacing(4)
        row.addWidget(QLabel('Time:'))
        self._time_edit = QTimeEdit()
        self._time_edit.setDisplayFormat('HH:mm:ss')
        self._time_edit.setFixedWidth(85)
        row.addWidget(self._time_edit)

        row.addSpacing(4)
        row.addWidget(QLabel('Timezone:'))
        self._tz_combo = QComboBox()
        self._tz_combo.addItems(list(_TZ_OFFSETS.keys()))
        self._tz_combo.setFixedWidth(115)
        row.addWidget(self._tz_combo)
        row.addStretch()
        layout.addLayout(row)

        # Hints from other loaded files
        if hints:
            hint_hdr = QLabel(
                f'<i>Hint — other loaded files recorded on {date_str}:</i>'
            )
            hint_hdr.setStyleSheet('color: #777777; font-size: 7pt;')
            layout.addWidget(hint_hdr)
            for fname, t_str in hints:
                h = QLabel(f'&nbsp;&nbsp;&nbsp;• {fname}  →  {t_str}')
                h.setStyleSheet('color: #44AAFF; font-size: 7pt;')
                layout.addWidget(h)

        return container

    # ── Slots ─────────────────────────────────────────────────────────────────

    def _on_import(self) -> None:
        """Read date/time fields, compute UTC anchor, accept dialog."""
        if self._date_edit and self._time_edit and self._tz_combo:
            qd = self._date_edit.date()
            qt = self._time_edit.time()
            local_dt = datetime(
                qd.year(), qd.month(), qd.day(),
                qt.hour(), qt.minute(), qt.second(),
            )
            offset_h = _TZ_OFFSETS.get(self._tz_combo.currentText(), 8)
            self._anchor_utc = local_dt - timedelta(hours=offset_h)
        self.accept()

    def _on_skip(self) -> None:
        """Import with synthetic t₀ = 0; leave anchor as None."""
        self._anchor_utc = None
        self.reject()


# ── SetStartTimeDialog ────────────────────────────────────────────────────────

class SetStartTimeDialog(QDialog):
    """Lightweight dialog for post-load start-time correction.

    Accessible via right-click → "Set Start Time…" on a file item in the
    Unified Canvas tree.  Lets the user fix the epoch without re-importing.

    Usage::

        dlg = SetStartTimeDialog(loaded.start_epoch, loaded.path.stem, self)
        if dlg.exec() == QDialog.DialogCode.Accepted and dlg.anchor_utc:
            loaded.start_epoch = epoch_from_datetime(dlg.anchor_utc)
    """

    def __init__(
        self,
        current_epoch: float,
        filename:      str,
        parent:        Optional[QWidget] = None,
    ) -> None:
        """Build the dialog.

        Args:
            current_epoch: Current POSIX epoch (float) of the file's t[0].
            filename:      Short filename shown in the header label.
            parent:        Optional parent widget.
        """
        super().__init__(parent)
        self.setWindowTitle('Set Start Time')
        self.setMinimumWidth(420)
        self.setModal(True)

        self._anchor_utc: Optional[datetime] = None

        root = QVBoxLayout(self)
        root.setSpacing(10)
        root.setContentsMargins(14, 12, 14, 12)

        hdr = QLabel(f'<b>File:</b>&nbsp;{filename}')
        hdr.setStyleSheet('font-size: 9pt; color: #DDDDDD;')
        root.addWidget(hdr)

        # Show the current epoch if it's valid (not the 1970 fallback)
        if current_epoch > 86400:
            current_utc = datetime.utcfromtimestamp(current_epoch)
            cur_lbl = QLabel(
                f'Current UTC start time:&nbsp;&nbsp;'
                f'<code>{current_utc.strftime("%Y-%m-%d  %H:%M:%S")}</code>'
            )
            cur_lbl.setStyleSheet('color: #AAAAAA; font-size: 8pt;')
            root.addWidget(cur_lbl)
        else:
            cur_lbl = QLabel(
                'Current start time: <span style="color:#FF8888">not set '
                '(synthetic t₀ = 0)</span>'
            )
            cur_lbl.setStyleSheet('font-size: 8pt;')
            root.addWidget(cur_lbl)

        root.addWidget(_hline())

        prompt = QLabel('Enter the correct recording start time:')
        prompt.setStyleSheet('color: #CCCCCC; font-size: 8pt;')
        root.addWidget(prompt)

        row = QHBoxLayout()
        row.setSpacing(6)

        row.addWidget(QLabel('Date:'))
        self._date_edit = QDateEdit()
        self._date_edit.setDisplayFormat('MM/dd/yyyy')
        self._date_edit.setCalendarPopup(True)
        self._date_edit.setFixedWidth(115)
        if current_epoch > 86400:
            dt = datetime.utcfromtimestamp(current_epoch)
            self._date_edit.setDate(QDate(dt.year, dt.month, dt.day))
        row.addWidget(self._date_edit)

        row.addSpacing(4)
        row.addWidget(QLabel('Time:'))
        self._time_edit = QTimeEdit()
        self._time_edit.setDisplayFormat('HH:mm:ss')
        self._time_edit.setFixedWidth(85)
        if current_epoch > 86400:
            dt = datetime.utcfromtimestamp(current_epoch)
            self._time_edit.setTime(QTime(dt.hour, dt.minute, dt.second))
        row.addWidget(self._time_edit)

        row.addSpacing(4)
        row.addWidget(QLabel('Timezone:'))
        self._tz_combo = QComboBox()
        self._tz_combo.addItems(list(_TZ_OFFSETS.keys()))
        self._tz_combo.setFixedWidth(115)
        row.addWidget(self._tz_combo)
        row.addStretch()
        root.addLayout(row)

        root.addWidget(_hline())

        btn_row = QHBoxLayout()
        btn_row.addStretch()

        cancel_btn = QPushButton('Cancel')
        cancel_btn.clicked.connect(self.reject)
        btn_row.addWidget(cancel_btn)

        ok_btn = QPushButton('Apply')
        ok_btn.setDefault(True)
        ok_btn.setStyleSheet(
            'QPushButton { background: #2244AA; color: #EEE; '
            'border: 1px solid #4466CC; padding: 4px 18px; '
            'border-radius: 3px; font-weight: bold; font-size: 8pt; }'
            'QPushButton:hover { background: #3355BB; }'
        )
        ok_btn.clicked.connect(self._on_apply)
        btn_row.addWidget(ok_btn)
        root.addLayout(btn_row)

        self.setStyleSheet(_DARK_STYLE)

    @property
    def anchor_utc(self) -> Optional[datetime]:
        """UTC anchor datetime entered by the user; None if cancelled."""
        return self._anchor_utc

    def _on_apply(self) -> None:
        qd = self._date_edit.date()
        qt = self._time_edit.time()
        local_dt = datetime(
            qd.year(), qd.month(), qd.day(),
            qt.hour(), qt.minute(), qt.second(),
        )
        offset_h = _TZ_OFFSETS.get(self._tz_combo.currentText(), 8)
        self._anchor_utc = local_dt - timedelta(hours=offset_h)
        self.accept()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _hline() -> QFrame:
    """Return a thin horizontal separator."""
    line = QFrame()
    line.setFrameShape(QFrame.Shape.HLine)
    line.setStyleSheet('color: #444444;')
    return line


def _parse_qdate(date_str: str) -> QDate:
    """Try common date formats and return a QDate; invalid QDate on failure."""
    for fmt in ('%m/%d/%y', '%m/%d/%Y', '%d/%m/%y', '%d/%m/%Y', '%Y-%m-%d'):
        try:
            d = datetime.strptime(date_str.strip(), fmt)
            return QDate(d.year, d.month, d.day)
        except ValueError:
            continue
    return QDate()
