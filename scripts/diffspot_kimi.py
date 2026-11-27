"""Ask Kimi K2.5 to verify each detector-proposed box on DiffSpot (Test 2 inference).

Moonshot retired kimi-k2.5 from its own API on 2026-08-31, so calls go through
OpenRouter, pinned to one host (Amazon Bedrock) with no fallback to others.
Pairs without a box are not sent to Kimi. Needs ``OPENROUTER_API_KEY``.

    python3 scripts/diffspot_kimi.py --dataset data/external/diffspot-full \\
        --predictions outputs/diffspot/fc-siam-base.jsonl --out outputs/diffspot/fc-siam-kimi \\
        [--sample outputs/diffspot-smoke/sample44] [--max-cost 8.5] [--workers 4]

Writes ``responses.jsonl`` (one row per pair, appended as each finishes),
``verification.jsonl`` and ``run.json`` once every pair is answered. Rerunning
continues where it stopped. Exit code 3 means the spending cap or the account
credit stopped the run; raise the cap or top up, then rerun.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import sys

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from openrouter import BUDGET_EXIT, FATAL_EXIT, ROOT, BudgetStop, call, cost, load_key, resume, run_all

PROMPT_FILE = ROOT / "configs/prompts/kimi_diffspot_verification_v1.txt"
NO_CHANGE = "No differences identified."
NOT_CALLED = "No detector proposal; Kimi was not called."
# Frozen Test 2 settings (scripts/diffspot.py freeze, configs/diffspot.fc-siam.json)
MODEL, TEMPERATURE, MAX_TOKENS, PADDING_PX, OVERVIEW_WIDTH_PX = "kimi-k2.5", 0, 16384, 32, 1280
OPENROUTER_MODEL, PROVIDER = "moonshotai/kimi-k2.5", "amazon-bedrock"
LABELS = ("Image A overview (expected):", "Image B overview (actual):",
          "Image A crop (expected), proposed region:", "Image B crop (actual), proposed region:")


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


def padded(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    canvas = Image.new("RGB", size, "white")
    canvas.paste(image, (0, 0))
    return canvas


def crop_region(box: list[float], size: tuple[int, int]) -> tuple[int, int, int, int]:
    # Same padded region in both screenshots, clipped to the larger canvas
    x, y, w, h = box
    return (max(0, int(x) - PADDING_PX), max(0, int(y) - PADDING_PX),
            min(size[0], int(x + w + 0.999) + PADDING_PX), min(size[1], int(y + h + 0.999) + PADDING_PX))


def messages(prompt: str, before: Image.Image, after: Image.Image, box: list[float]) -> list[dict]:
    size = (max(before.width, after.width), max(before.height, after.height))
    region = crop_region(box, size)
    crops = [padded(image, size).crop(region) for image in (before, after)]
    # Page overviews come before the crops so the shared start of the request can be cached
    images = (overview(before), overview(after), *crops)
    content = []
    for label, image in zip(LABELS, images):
        content += [{"type": "text", "text": label}, {"type": "image_url", "image_url": {"url": data_url(image)}}]
    return [{"role": "system", "content": prompt}, {"role": "user", "content": content}]


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, help="prepared DiffSpot folder")
    parser.add_argument("--predictions", required=True, help="base prediction JSONL in the shared format")
    parser.add_argument("--out", required=True, help="output folder")
    parser.add_argument("--sample", help="saved smoke sample folder; only its pairs are run")
    parser.add_argument("--max-cost", type=float, default=8.5, help="stop starting calls once this many USD are spent")
    parser.add_argument("--workers", type=int, default=4, help="calls in flight at once")
    args = parser.parse_args()

    dataset, out = Path(args.dataset), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pairs = read_jsonl(Path(args.sample or args.dataset) / "inference.jsonl")
    predictions = {row["sample_id"]: row for row in read_jsonl(Path(args.predictions))}
    missing = [row["sample_id"] for row in pairs if row["sample_id"] not in predictions]
    if missing:
        raise SystemExit(f"{len(missing)} pairs have no base prediction, e.g. {missing[0]}")
    prompt = PROMPT_FILE.read_text()
    source_sha = hashlib.sha256(Path(args.predictions).read_bytes()).hexdigest()
    settings = dict(model=MODEL, openrouter_model=OPENROUTER_MODEL, provider=PROVIDER, temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS, thinking="disabled", crop_padding_px=PADDING_PX,
                    overview_width_px=OVERVIEW_WIDTH_PX, image_labels=LABELS,
                    prompt_file=str(PROMPT_FILE.relative_to(ROOT)),
                    prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(), source_prediction_sha256=source_sha)

    # A rerun must use the same base predictions, or old answers would be mixed with new boxes
    settings_path = out / "settings.json"
    if settings_path.exists() and json.loads(settings_path.read_text())["source_prediction_sha256"] != source_sha:
        raise SystemExit(f"{out} was started with different base predictions; use a new --out")
    settings_path.write_text(json.dumps(settings, indent=2) + "\n")

    responses_path = out / "responses.jsonl"
    done = resume(responses_path)
    answered = {row["sample_id"] for row in done}
    spent = sum(cost(row.get("usage")) for row in done)
    todo = [pair for pair in pairs if pair["sample_id"] not in answered]
    print(f"{len(answered)} of {len(pairs)} pairs already answered (${spent:.4f}); {len(todo)} to go", flush=True)
    key = load_key() if any(predictions[p["sample_id"]].get("box") for p in todo) else ""

    def work(pair: dict) -> tuple[dict, float]:
        box = predictions[pair["sample_id"]].get("box")
        row = dict(sample_id=pair["sample_id"], box=box, called=box is not None, usage={})
        if box is None:
            row.update(raw_response=NOT_CALLED, decision="reject", description=NO_CHANGE)
            return row, 0.0
        before = Image.open(dataset / pair["reference_path"]).convert("RGB")
        after = Image.open(dataset / pair["candidate_path"]).convert("RGB")
        reply = call(key, dict(model=OPENROUTER_MODEL, messages=messages(prompt, before, after, box),
                               temperature=TEMPERATURE, max_tokens=MAX_TOKENS, reasoning={"enabled": False},
                               usage={"include": True}, provider={"order": [PROVIDER], "allow_fallbacks": False}))
        # Thinking is frozen off; stop rather than pay for thinking tokens if a host ever ignores that
        thinking = (reply.get("usage", {}).get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
        if thinking:
            print(f"Kimi replied with {thinking} thinking tokens although thinking is off; stopping", flush=True)
            raise SystemExit(FATAL_EXIT)
        choice = reply["choices"][0]
        text = choice["message"].get("content") or ""
        decision, description = parse(text)
        row.update(raw_response=text or "(empty reply)", decision=decision, description=description,
                   usage=reply.get("usage", {}), finish_reason=choice.get("finish_reason"), provider=reply.get("provider"))
        print(f"  {decision:7s} ${cost(row['usage']):.4f}", flush=True)
        return row, cost(row["usage"])

    try:
        spent = run_all(todo, work, responses_path, spent, args.max_cost, args.workers)
    except BudgetStop as stop:
        print(stop)
        raise SystemExit(BUDGET_EXIT)

    rows = resume(responses_path)
    with (out / "verification.jsonl").open("w") as sidecar:
        for row in rows:
            sidecar.write(json.dumps(dict(sample_id=row["sample_id"], decision=row["decision"],
                                          box=row["box"] if row["decision"] == "accept" else None,
                                          description=row["description"], source_prediction_sha256=source_sha)) + "\n")
    called = [row for row in rows if row["called"]]
    usage = [row["usage"] for row in called]
    summary = dict(settings=settings, pairs=len(rows), called=len(called),
                   decisions={d: sum(row["decision"] == d for row in rows) for d in ("accept", "reject", "unsure")},
                   tokens=dict(prompt=sum(u.get("prompt_tokens", 0) for u in usage),
                               completion=sum(u.get("completion_tokens", 0) for u in usage)),
                   cost_usd=round(spent, 4), providers=sorted({row.get("provider") or "" for row in called}))
    (out / "run.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("pairs", "called", "decisions", "cost_usd", "providers")}))


if __name__ == "__main__":
    main()
