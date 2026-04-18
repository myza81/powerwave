# Dynamic Y-Axis Expansion for Analogue Channels

---

## OBJECTIVE
Implement a **dynamic Y-axis auto-fitting feature** for analogue channels in PowerWave Analyst. When a user clicks/selects a specific analogue channel, the Y-axis should automatically expand/contract to tightly fit the visible data range with 5–10% padding, snapping instantly to the new range. A global "Auto-fit All" button will reset all analogue channels to their optimal Y-axis range.

---

## KEY REQUIREMENTS

### 1. **Channel Selection & Y-Axis Expansion**
- **Trigger**: User clicks on an analogue channel (in the channel list, waveform area, or label panel)
- **Action**: Calculate the min/max of the visible (time-zoomed) analogue data for that channel
- **Padding**: Add 5–10% padding above and below the data range:
  ```
  data_range = max_value - min_value
  padding = data_range * 0.075  # 7.5% padding (adjust as needed)
  new_min = min_value - padding
  new_max = max_value + padding
  ```
- **Animation**: Snap instantly (no smooth animation)
- **Apply**: Update the channel's Y-axis bounds immediately in the OpenGL rendering

### 2. **Digital Channels: NO Y-Axis Expansion**
- Digital channels (state/binary channels) should **NOT** participate in auto-fit logic
- They retain fixed Y-axis ranges (e.g., 0–1.2 or as configured)
- Clicking a digital channel should NOT trigger Y-axis expansion
- You may optionally highlight/select the digital channel visually, but do not change its scale

### 3. **Auto-Fit All Button**
- **Location**: Add a button in the toolbar or channel control panel labeled **"Auto-fit All"**
- **Action**: When clicked, recalculate and apply optimal Y-axis ranges to **ALL analogue channels**
- **Respects time-zoom**: The auto-fit calculation uses the current visible time window (respects horizontal zoom)
- **Visual feedback**: Briefly highlight or provide a status message (e.g., "Y-axes adjusted")

### 4. **Interaction Logic**
- Clicking an already-selected analogue channel should toggle or refresh its auto-fit
- Digital channels should remain unaffected
- Multi-select mode (if applicable) should apply auto-fit to all selected analogue channels only

### 5. **Integration with Existing Code**
- **File locations** (based on project structure):
  - `src/ui/waveform_canvas.py` — OpenGL rendering, Y-axis bounds storage
  - `src/ui/channel_panel.py` — Channel selection logic and event handling
  - `src/utils/signal_processing.py` — Min/max calculation utilities (use vectorised NumPy)
  - `src/models/disturbance_record.py` — Channel type detection (analogue vs. digital)

- **Existing methods to leverage**:
  - `AnalogueChannel.get_visible_range(start_idx, end_idx)` or equivalent
  - `DisturbanceRecord.get_channel_by_name()` for channel lookup
  - OpenGL uniform/shader state for Y-axis bounds (e.g., `u_ymin`, `u_ymax` per channel)

- **No breaking changes**: Preserve existing zoom, pan, and rendering logic

---

## IMPLEMENTATION STEPS

### Step 1: Add Y-Axis Expansion Logic
**File**: `src/utils/signal_processing.py`

Create a new function:
```python
def calculate_optimal_yaxis_range(data, padding_percent=7.5):
    """
    Calculate optimal Y-axis range for analogue data with padding.
    
    Args:
        data (np.ndarray): Channel data (1D array of float values)
        padding_percent (float): Padding percentage (default 7.5%)
    
    Returns:
        tuple: (y_min, y_max) with padding applied
    """
    data_min = np.nanmin(data)
    data_max = np.nanmax(data)
    data_range = data_max - data_min
    
    if data_range == 0:
        # Handle flat data (all same value)
        data_range = 1.0
    
    padding = data_range * (padding_percent / 100.0)
    y_min = data_min - padding
    y_max = data_max + padding
    
    return y_min, y_max
```

### Step 2: Extend Channel Data Model
**File**: `src/models/disturbance_record.py` (or `src/models/analogue_channel.py`)

Add Y-axis state to analogue channels:
```python
class AnalogueChannel:
    def __init__(self, ...):
        # ... existing code ...
        self.y_min_auto = None  # Auto-fitted Y-min
        self.y_max_auto = None  # Auto-fitted Y-max
        self.is_selected = False
    
    def set_auto_yaxis(self, y_min, y_max):
        """Set auto-fitted Y-axis bounds."""
        self.y_min_auto = y_min
        self.y_max_auto = y_max
    
    def get_yaxis_bounds(self):
        """Return current Y-axis bounds (auto or default)."""
        if self.y_min_auto is not None and self.y_max_auto is not None:
            return self.y_min_auto, self.y_max_auto
        # Fall back to default or user-set bounds
        return self.y_min_default, self.y_max_default
```

### Step 3: Implement Channel Selection & Click Handler
**File**: `src/ui/channel_panel.py` (or `src/ui/waveform_canvas.py`)

Add click event handler for channel selection:
```python
def on_channel_clicked(self, channel_name):
    """
    Handle channel selection click.
    Trigger Y-axis auto-fit for analogue channels only.
    """
    channel = self.disturbance_record.get_channel_by_name(channel_name)
    
    if channel is None:
        return
    
    # Skip digital channels
    if isinstance(channel, DigitalChannel):
        channel.is_selected = True
        self.update_canvas()
        return
    
    # Process analogue channels
    if isinstance(channel, AnalogueChannel):
        channel.is_selected = True
        
        # Get visible data range (respect current time zoom)
        visible_data = self.get_visible_data_for_channel(channel)
        
        # Calculate optimal Y-axis range
        y_min, y_max = calculate_optimal_yaxis_range(
            visible_data, padding_percent=7.5
        )
        
        # Apply to channel
        channel.set_auto_yaxis(y_min, y_max)
        
        # Update rendering
        self.update_canvas()

def get_visible_data_for_channel(self, channel):
    """Extract visible portion of channel data based on current time zoom."""
    start_idx = self.viewport_start_index
    end_idx = self.viewport_end_index
    return channel.samples[start_idx:end_idx]
```

### Step 4: Add Auto-Fit All Button
**File**: `src/ui/toolbar.py` or `src/ui/main_window.py`

Add button to toolbar:
```python
self.auto_fit_button = QPushButton("Auto-fit All")
self.auto_fit_button.clicked.connect(self.on_auto_fit_all_clicked)
self.toolbar.addWidget(self.auto_fit_button)

def on_auto_fit_all_clicked(self):
    """Auto-fit all analogue channels to visible data range."""
    for channel in self.disturbance_record.analogue_channels:
        visible_data = self.get_visible_data_for_channel(channel)
        y_min, y_max = calculate_optimal_yaxis_range(visible_data, padding_percent=7.5)
        channel.set_auto_yaxis(y_min, y_max)
    
    # Optional: visual feedback
    self.status_bar.showMessage("Y-axes auto-fitted for all analogue channels", 2000)
    self.update_canvas()
```

### Step 5: Update OpenGL Rendering
**File**: `src/ui/waveform_canvas.py`

Modify Y-axis bounds lookup in shader:
```python
def get_channel_yaxis_bounds(self, channel):
    """Get Y-axis bounds for rendering (respects auto-fit)."""
    if isinstance(channel, AnalogueChannel):
        y_min, y_max = channel.get_yaxis_bounds()
        return y_min, y_max
    # Digital channels use fixed bounds
    return 0.0, 1.2

def render_analogue_channel(self, channel):
    """Render analogue waveform with current Y-axis bounds."""
    y_min, y_max = self.get_channel_yaxis_bounds(channel)
    
    # Pass to shader via uniform
    self.shader_program.setUniformValue(
        self.shader_program.uniformLocation("u_ymin"),
        float(y_min)
    )
    self.shader_program.setUniformValue(
        self.shader_program.uniformLocation("u_ymax"),
        float(y_max)
    )
    
    # Render as normal
    self.gl.glDrawArrays(...)
```

---

## DATA FLOW DIAGRAM

```
User clicks channel
    ↓
on_channel_clicked(channel_name)
    ↓
Is it analogue?
    ├─ YES → get_visible_data_for_channel()
    │        ↓
    │        calculate_optimal_yaxis_range()
    │        ↓
    │        channel.set_auto_yaxis(y_min, y_max)
    │        ↓
    │        update_canvas() [OpenGL re-renders with new bounds]
    │
    └─ NO (digital) → Mark as selected, no Y-axis change
```

---

## TESTING CHECKLIST

- [ ] Click an analogue channel → Y-axis tightens to fit data with 5–10% padding
- [ ] Click a digital channel → No Y-axis expansion (visual feedback only)
- [ ] Zoom horizontally (time-axis) → Auto-fit recalculates for visible window only
- [ ] Click "Auto-fit All" → All analogue channels auto-fit simultaneously
- [ ] Padding is visually consistent (no data clipping, clean space above/below)
- [ ] Performance: No lag when selecting channels or clicking auto-fit
- [ ] Multiple clicks on same channel → Y-axis updates correctly
- [ ] Mix of analogue + digital channels → Only analogue channels scale

---

## EDGE CASES TO HANDLE

1. **Flat data** (all samples = same value):
   - Data range = 0 → Default padding logic fails
   - **Solution**: Set `data_range = 1.0` if `data_range == 0`

2. **NaN or inf values** in data:
   - Use `np.nanmin()`, `np.nanmax()` to ignore NaN
   - Sanitize inf values before calculation

3. **Very small data range** (e.g., 0.00001):
   - Padding calculation should scale proportionally
   - Consider minimum visible range (e.g., 0.001) to avoid underflow

4. **Empty visible window** (zoom too far in):
   - Fall back to full channel range with padding

---

## OPTIONAL ENHANCEMENTS (Future)

- **Manual Y-axis override**: Let user lock/unlock auto-fit for specific channels
- **Padding slider**: Allow user to adjust padding % (5–20%) via UI control
- **Double-click to reset**: Double-click channel to reset to default bounds
- **Keyboard shortcut**: `Ctrl+A` or `Cmd+A` for auto-fit all (if not conflicting)
- **Per-channel persistence**: Save auto-fit state when switching files

---

## SUCCESS CRITERIA

✅ Analogue channels expand Y-axis on selection with 5–10% padding
✅ Digital channels are unaffected
✅ "Auto-fit All" button works globally
✅ Instant snap (no animation delay)
✅ Respects current time-zoom window
✅ No performance degradation
✅ Code is clean, vectorised (NumPy), and well-documented

---

## QUESTIONS FOR CLARIFICATION (if needed)

If anything is unclear, ask:
1. Should auto-fit preserve user's manual Y-axis adjustments, or always override?
2. Is there a min/max padding value, or is 7.5% fixed?
3. Should clicking a channel deselect others (single-select) or allow multi-select?
4. Do you want visual highlight (e.g., border, color change) when a channel is selected?