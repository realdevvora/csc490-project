#!/usr/bin/env python3
"""
Cross-browser rendering generator for the CSC490 regression-detection dataset.

Takes the same parquet dataset used by the mutation notebook, renders each
page's *unmodified* HTML in several browsers (Chromium, Google Chrome,
Firefox, WebKit, Microsoft Edge), and writes output in the same layout as the
mutation notebook:

    <output-dir>/
        screenshots/<mutation_id>.png
        metadata/<mutation_id>.json
        master_dataset.json
        (optional) <output-dir>.zip

Every sample is labelled "benign": the HTML is identical, so any pixel
differences come purely from browser rendering engines and should NOT be
flagged as a meaningful regression.

Setup (one time):
    pip install playwright beautifulsoup4 pandas pyarrow
    playwright install chromium firefox webkit chrome msedge
    # On Linux you may also need:  playwright install-deps

Usage:
    python cross_browser_render.py --input train-00000-of-00001.parquet
    python cross_browser_render.py --input data.parquet --limit 10 \
        --browsers chromium firefox
"""

import argparse
import asyncio
import json
import os
import shutil
import sys

import pandas as pd
from playwright.async_api import async_playwright

# ---------------------------------------------------------------------------
# Browser definitions
# ---------------------------------------------------------------------------
# name -> (playwright engine attribute, optional branded-channel)
#   - "chromium": Playwright's bundled open-source Chromium build
#   - "chrome" / "msedge": real branded browsers, selected via `channel`
#   - "webkit": Playwright's WebKit build (stands in for Safari's engine)
BROWSERS = {
    "chromium": ("chromium", None),
    "chrome": ("chromium", "chrome"),
    "firefox": ("firefox", None),
    "webkit": ("webkit", None),
    "edge": ("chromium", "msedge"),
}

MUTATION_TYPE = "browser_rendering"
LABEL = "benign"
VIEWPORT = {"width": 1280, "height": 800}  # same as the mutation notebook


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
async def render_with_browser(p, browser_name, rows, images_dir, metadata_dir):
    """Render every row's HTML in one browser. Returns a list of records."""
    engine_attr, channel = BROWSERS[browser_name]
    engine = getattr(p, engine_attr)

    launch_kwargs = {"headless": True}
    if channel:
        launch_kwargs["channel"] = channel

    try:
        browser = await engine.launch(**launch_kwargs)
    except Exception as e:
        print(f"[{browser_name}] Could not launch, skipping this browser: {e}")
        print(f"[{browser_name}] Hint: run `playwright install "
              f"{channel or engine_attr}`")
        return []

    print(f"[{browser_name}] Launched (version {browser.version}).")
    records = []
    page = await browser.new_page(viewport=VIEWPORT)

    for index, raw_html in rows:
        parent_id = f"d2c_{index:03d}"
        mutation_id = f"{parent_id}_{MUTATION_TYPE}_{browser_name}"

        try:
            await page.set_content(raw_html, wait_until="load")
            await page.wait_for_timeout(100)

            screenshot_path = os.path.join(images_dir, f"{mutation_id}.png")
            await page.screenshot(path=screenshot_path, full_page=True)

            record = {
                "mutation_id": mutation_id,
                "parent_id": parent_id,
                "mutation_type": MUTATION_TYPE,
                "label": LABEL,
                "screenshot_path": screenshot_path,
                "mutated_html": raw_html,  # unchanged; kept for schema parity
                # extra fields (not in the mutation notebook's records)
                "browser": browser_name,
                "browser_version": browser.version,
            }
            records.append(record)

            with open(os.path.join(metadata_dir, f"{mutation_id}.json"), "w") as f:
                json.dump(record, f, indent=4)

        except Exception as e:
            print(f"[{browser_name}] Warning: render bypass on {mutation_id}: {e}")
            continue

    await browser.close()
    print(f"[{browser_name}] Done: {len(records)} screenshots.")
    return records


async def run_pipeline(df, output_dir, browser_names, make_zip):
    images_dir = os.path.join(output_dir, "screenshots")
    metadata_dir = os.path.join(output_dir, "metadata")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(metadata_dir, exist_ok=True)

    # Collect (index, html) pairs once, skipping empty rows (as in the notebook)
    rows = []
    for index, row in df.iterrows():
        raw_html = row.get("text")
        if raw_html:
            rows.append((index, raw_html))

    print(f"Rendering {len(rows)} pages in {len(browser_names)} browser(s): "
          f"{', '.join(browser_names)}")

    metadata_records = []
    async with async_playwright() as p:
        # One browser at a time: keeps memory low and lets us skip any that
        # aren't installed without affecting the rest.
        for name in browser_names:
            metadata_records.extend(
                await render_with_browser(p, name, rows, images_dir, metadata_dir)
            )

    master_json_path = os.path.join(output_dir, "master_dataset.json")
    with open(master_json_path, "w") as f:
        json.dump(metadata_records, f, indent=4)

    print(f"\nFinished! Total items generated: {len(metadata_records)}")

    if not metadata_records:
        print("Error: no renders generated. Check the dataframe and that "
              "browsers are installed.")
        return 1

    if make_zip:
        zip_base = output_dir.rstrip("/\\")
        print("Saving output folder as a zip file...")
        shutil.make_archive(base_name=zip_base, format="zip", root_dir=output_dir)
        print(f"Success! Archive written to {zip_base}.zip")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Render dataset HTML in multiple browsers and label the "
                    "results as benign samples."
    )
    parser.add_argument("--input", required=True,
                        help="Path to the parquet dataset (needs a 'text' "
                             "column containing HTML).")
    parser.add_argument("--output-dir", default="design2code_browser_variants",
                        help="Output directory (default: %(default)s).")
    parser.add_argument("--browsers", nargs="+", choices=list(BROWSERS),
                        default=list(BROWSERS),
                        help="Browsers to render with (default: all).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only process the first N rows (for testing).")
    parser.add_argument("--no-zip", action="store_true",
                        help="Skip creating the zip archive.")
    return parser.parse_args()


def main():
    args = parse_args()

    df = pd.read_parquet(args.input)
    print("Loaded columns:", df.columns.tolist())
    print(f"Total rows found: {len(df)}")
    if "text" not in df.columns:
        sys.exit("Error: dataset has no 'text' column with HTML content.")
    if args.limit:
        df = df.head(args.limit)

    exit_code = asyncio.run(
        run_pipeline(df, args.output_dir, args.browsers, not args.no_zip)
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
