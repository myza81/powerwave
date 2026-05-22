"""Contextual analytics suggestion engine for PowerWave."""
from app.analytics.suggestions.suggestion_engine import (
    Suggestion,
    SuggestionContext,
    generate_suggestions,
)

__all__ = ["Suggestion", "SuggestionContext", "generate_suggestions"]
