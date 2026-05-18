from __future__ import annotations

from dataclasses import dataclass
from typing import Any


CLIP_TO_VIEW = True
AUTO_DOWNSAMPLE = True
DOWNSAMPLE_METHOD = "peak"
ANTIALIAS = False
Z_VALUE = 20


@dataclass(frozen=True, slots=True)
class OverlayRenderPolicy:
    """Default PyQtGraph policy for lightweight overlay curves."""

    clip_to_view: bool = CLIP_TO_VIEW
    auto_downsample: bool = AUTO_DOWNSAMPLE
    downsample_method: str = DOWNSAMPLE_METHOD
    antialias: bool = ANTIALIAS
    z_value: int = Z_VALUE


DEFAULT_OVERLAY_RENDER_POLICY = OverlayRenderPolicy()


def apply_curve_policy(
    curve: Any,
    policy: OverlayRenderPolicy = DEFAULT_OVERLAY_RENDER_POLICY,
) -> Any:
    """Apply the standard overlay rendering policy to a PyQtGraph curve."""
    if hasattr(curve, "setClipToView"):
        curve.setClipToView(policy.clip_to_view)
    if hasattr(curve, "setDownsampling"):
        curve.setDownsampling(
            auto=policy.auto_downsample,
            method=policy.downsample_method,
        )
    if hasattr(curve, "setAlpha") and policy.antialias:
        pass
    opts = getattr(curve, "opts", None)
    if isinstance(opts, dict):
        opts["antialias"] = policy.antialias
    if hasattr(curve, "setZValue"):
        curve.setZValue(policy.z_value)
    return curve
