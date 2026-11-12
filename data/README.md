# Local data

This directory documents local dataset storage; dataset contents are not committed to Git.

Expected local subdirectories are:

- `raw/` — immutable, revision-pinned upstream Design2Code downloads;
- `interim/` — browser-ready pages and temporary render workspaces;
- `processed/` — accepted screenshot pairs, metadata, and manifests;
- `splits/` — frozen page-level splits. These are small and **committed to Git**, unlike the other folders. `splits_85_15.json` is the team split (390 train / 69 val pages). It leaves out 25 pages whose `no_change` pair is not actually identical, together with all their pairs, and is regenerated with `scripts/make_team_split.py`;
- `external/` — the local DiffSpot copy used only for final external evaluation.

Record the upstream dataset name, exact revision, acquisition date, imported count, and checksum or manifest before using data in an experiment. Generated data should be reproducible from a source page, mutation record, and render configuration.

Do not place credentials in this directory. Do not commit datasets, screenshots, archives, or checkpoints.
