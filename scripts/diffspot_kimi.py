"""Ask Kimi K2.5 to verify each detector-proposed box on DiffSpot (Test 2 inference).

Moonshot retired kimi-k2.5 from its own API on 2026-08-31, so calls go through
OpenRouter, pinned to one host (Amazon Bedrock) with no fallback to others.

Reads the prepared ``inference.jsonl`` (optionally only a saved sample's IDs) and
one base prediction file. Pairs without a box are not sent to Kimi. Needs
``OPENROUTER_API_KEY`` in the environment or in the repository's ``.env``.

    python3 scripts/diffspot_kimi.py --dataset data/external/diffspot-full \\
        --predictions outputs/diffspot/fc-siam-base.jsonl --out outputs/diffspot/fc-siam-kimi \\
        [--sample outputs/diffspot-smoke/sample44] [--thinking disabled]

Writes ``responses.jsonl`` (raw reply, decision, description, token usage per pair),
``verification.jsonl`` (the decision sidecar) and ``run.json`` (settings, tokens, cost).
Already-answered pairs are skipped, so an interrupted run can be resumed.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PROMPT_FILE = ROOT / "configs/prompts/kimi_diffspot_verification_v1.txt"
NO_CHANGE = "No differences identified."
# Frozen Test 2 settings from scripts/diffspot.py freeze
MODEL, TEMPERATURE, MAX_TOKENS, PADDING_PX, OVERVIEW_WIDTH_PX = "kimi-k2.5", 0, 16384, 32, 1280
OPENROUTER_MODEL, PROVIDER = "moonshotai/kimi-k2.5", "amazon-bedrock"


def load_key() -> str:
    # The environment wins; otherwise read KEY=value lines from the repository's .env
    env = ROOT / ".env"
    if "OPENROUTER_API_KEY" not in os.environ and env.exists():
        for line in env.read_text().splitlines():
            key, _, value = line.partition("=")
            if key.strip() == "OPENROUTER_API_KEY":
                os.environ["OPENROUTER_API_KEY"] = value.strip().strip("\"'")
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit(f"OPENROUTER_API_KEY is not set; add it to {env}")
    return os.environ["OPENROUTER_API_KEY"]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def data_url(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def overview(image: Image.Image) -> Image.Image:
    if image.width == OVERVIEW_WIDTH_PX:
        return image
    return image.resize((OVERVIEW_WIDTH_PX, round(image.height * OVERVIEW_WIDTH_PX / image.width)), Image.LANCZOS)


def crop_region(box: list[float], size: tuple[int, int]) -> tuple[int, int, int, int]:
    # Same padded region in both screenshots, clipped to the larger canvas
    x, y, w, h = box
    return (max(0, int(x) - PADDING_PX), max(0, int(y) - PADDING_PX),
            min(size[0], int(x + w + 0.999) + PADDING_PX), min(size[1], int(y + h + 0.999) + PADDING_PX))


def messages(prompt: str, before: Image.Image, after: Image.Image, box: list[float]) -> list[dict]:
    size = (max(before.width, after.width), max(before.height, after.height))
    region = crop_region(box, size)
    crops = [_padded(image, size).crop(region) for image in (before, after)]
    # Page overviews come before the crops so the shared start of the request can be cached
    parts = [("Image A overview (expected):", overview(before)), ("Image B overview (actual):", overview(after)),
             ("Image A crop (expected), proposed region:", crops[0]), ("Image B crop (actual), proposed region:", crops[1])]
    content = []
    for label, image in parts:
        content += [{"type": "text", "text": label}, {"type": "image_url", "image_url": {"url": data_url(image)}}]
    return [{"role": "system", "content": prompt}, {"role": "user", "content": content}]


def _padded(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    canvas = Image.new("RGB", size, "white")
    canvas.paste(image, (0, 0))
    return canvas


def call(key: str, body: dict) -> dict:
    request = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", json.dumps(body).encode(),
                                     {"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            detail = error.read().decode(errors="replace")
            if error.code not in (429, 500, 502, 503) or attempt == 4:
                raise SystemExit(f"Kimi API error {error.code}: {detail}")
        time.sleep(2 ** attempt * 5)


def parse(reply: str) -> tuple[str, str]:
    # Take the outermost JSON object; anything unreadable or out of contract counts as unsure
    try:
        found = json.loads(reply[reply.index("{"): reply.rindex("}") + 1])
        decision, description = found["decision"], found["description"]
    except (ValueError, KeyError, TypeError):
        return "unsure", NO_CHANGE
    if decision not in ("accept", "reject", "unsure") or not isinstance(description, str) or not description.strip():
        return "unsure", NO_CHANGE
    return decision, description.strip() if decision == "accept" else NO_CHANGE


def cost(usage: dict) -> float:
    # OpenRouter reports the amount actually charged for each call
    return usage.get("cost", 0) or 0


def cached(usage: dict) -> int:
    return (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="prepared DiffSpot folder")
    parser.add_argument("--predictions", required=True, help="base prediction JSONL in the shared format")
    parser.add_argument("--out", required=True, help="output folder")
    parser.add_argument("--sample", help="saved smoke sample folder; only its pairs are run")
    parser.add_argument("--thinking", choices=("enabled", "disabled"), default="disabled")
    parser.add_argument("--limit", type=int, help="run only the first N pairs")
    args = parser.parse_args()

    dataset, out = Path(args.dataset), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pairs = read_jsonl(Path(args.sample or args.dataset) / "inference.jsonl")[: args.limit]
    predictions = {row["sample_id"]: row for row in read_jsonl(Path(args.predictions))}
    missing = [row["sample_id"] for row in pairs if row["sample_id"] not in predictions]
    if missing:
        raise SystemExit(f"{len(missing)} pairs have no base prediction, e.g. {missing[0]}")
    prompt = PROMPT_FILE.read_text()
    key = load_key()
    source_sha = hashlib.sha256(Path(args.predictions).read_bytes()).hexdigest()
    settings = dict(model=MODEL, openrouter_model=OPENROUTER_MODEL, provider=PROVIDER, temperature=TEMPERATURE, max_tokens=MAX_TOKENS, thinking=args.thinking,
                    crop_padding_px=PADDING_PX, overview_width_px=OVERVIEW_WIDTH_PX,
                    prompt_file=str(PROMPT_FILE.relative_to(ROOT)),
                    prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(), source_prediction_sha256=source_sha)

    responses_path = out / "responses.jsonl"
    done = {row["sample_id"] for row in read_jsonl(responses_path)} if responses_path.exists() else set()
    with responses_path.open("a") as responses:
        for index, pair in enumerate(pairs, 1):
            sample_id, box = pair["sample_id"], predictions[pair["sample_id"]].get("box")
            if sample_id in done:
                continue
            row = dict(sample_id=sample_id, box=box, called=box is not None, usage={})
            if box is None:
                row.update(raw_response="No detector proposal; Kimi was not called.", decision="reject", description=NO_CHANGE)
            else:
                before = Image.open(dataset / pair["reference_path"]).convert("RGB")
                after = Image.open(dataset / pair["candidate_path"]).convert("RGB")
                body = dict(model=OPENROUTER_MODEL, messages=messages(prompt, before, after, box),
                            temperature=TEMPERATURE, max_tokens=MAX_TOKENS,
                            reasoning={"enabled": args.thinking == "enabled"}, usage={"include": True},
                            provider={"order": [PROVIDER], "allow_fallbacks": False})
                reply = call(key, body)
                text = reply["choices"][0]["message"].get("content") or ""
                decision, description = parse(text)
                row.update(raw_response=text, decision=decision, description=description, usage=reply.get("usage", {}),
                           finish_reason=reply["choices"][0].get("finish_reason"), provider=reply.get("provider"))
            responses.write(json.dumps(row, ensure_ascii=False) + "\n")
            responses.flush()
            print(f"{index}/{len(pairs)} {row['decision']:7s} ${cost(row['usage']):.4f}")

    rows = read_jsonl(responses_path)
    with (out / "verification.jsonl").open("w") as sidecar:
        for row in rows:
            sidecar.write(json.dumps(dict(sample_id=row["sample_id"], decision=row["decision"],
                                          box=row["box"] if row["decision"] == "accept" else None,
                                          description=row["description"], source_prediction_sha256=source_sha)) + "\n")
    called = [row for row in rows if row["called"]]
    totals = {name: sum(row["usage"].get(name, 0) or 0 for row in called) for name in ("prompt_tokens", "completion_tokens")}
    totals["cached_tokens"] = sum(cached(row["usage"]) for row in called)
    totals["reasoning_tokens"] = sum((row["usage"].get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0
                                     for row in called)
    spent = sum(cost(row["usage"]) for row in called)
    summary = dict(settings=settings, pairs=len(rows), called=len(called), tokens=totals, cost_usd=round(spent, 4),
                   cost_per_called_pair_usd=round(spent / max(1, len(called)), 5),
                   providers=sorted({row.get("provider") or "" for row in called}))
    (out / "run.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("called", "tokens", "cost_usd", "cost_per_called_pair_usd")}, indent=2))


if __name__ == "__main__":
    main()
