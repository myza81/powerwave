# Legacy source tree

This directory is **not** the canonical package root. It is retained for reference only.

The active application stack lives under `app/`. Packaging (`pyproject.toml`) now
points at the repo root with `include = ["app*"]`.

Do not import from `src.*` in new code. Modules here are candidates for deletion
once any remaining tests that still exercise them are migrated or removed.
