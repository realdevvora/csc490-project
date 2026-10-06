# Local data

This directory documents local dataset storage; dataset contents are not committed to Git.

Expected local subdirectories are:

- `raw/` — immutable, revision-pinned upstream Design2Code downloads;
- `interim/` — browser-ready pages and temporary render workspaces;
- `processed/` — accepted screenshot pairs, metadata, manifests, and frozen splits;
- `external/` — the local DiffSpot copy used only for final external evaluation.

Record the upstream dataset name, exact revision, acquisition date, imported count, and checksum or manifest before using data in an experiment. Generated data should be reproducible from a source page, mutation record, and render configuration.

Do not place credentials in this directory. Do not commit datasets, screenshots, archives, or checkpoints.
