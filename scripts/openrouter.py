"""Shared OpenRouter plumbing for the DiffSpot Kimi and judge runners.

- the API key from the environment or the repository's ``.env``;
- calls that retry through rate limits, server errors and network drops;
- a spending cap that stops a run cleanly;
- resumable JSONL output that survives being killed mid-write;
- a few calls in flight at once.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Callable, Iterable

ROOT = Path(__file__).resolve().parents[1]
API = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")  # overridable for offline tests
URL = API + "/chat/completions"
BUDGET_EXIT = 3  # exit code when the spending cap or the account credit stops a run
FATAL_EXIT = 2  # exit code for a request that retrying cannot fix


class BudgetStop(Exception):
    """The spending cap was reached or OpenRouter reports no credit left."""


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


def call(key: str, body: dict, attempts: int = 8) -> dict:
    # Retry what can recover (rate limits, server errors, dropped connections); stop on anything else
    request = urllib.request.Request(URL, json.dumps(body).encode(),
                                     {"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                reply = json.load(response)
            if reply.get("choices"):
                return reply
            code, detail = (reply.get("error") or {}).get("code", 0), reply
        except urllib.error.HTTPError as error:
            code, detail = error.code, error.read().decode(errors="replace")
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, json.JSONDecodeError) as error:
            code, detail = None, repr(error)
        if code == 402:
            raise BudgetStop(f"OpenRouter has no credit left: {detail}")
        if code in (400, 401, 403, 404):
            print(f"OpenRouter refused the request ({code}): {detail}", flush=True)
            raise SystemExit(FATAL_EXIT)
        print(f"  retry {attempt + 1}/{attempts} after {code or 'network error'}: {str(detail)[:200]}", flush=True)
        time.sleep(min(300, 5 * 2 ** attempt))
    raise RuntimeError(f"OpenRouter call failed after {attempts} attempts: {str(detail)[:500]}")


def remaining_credit(key: str) -> float:
    # Account credit left, from OpenRouter's credits endpoint
    request = urllib.request.Request(API + "/credits", headers={"Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(request, timeout=60) as response:
        data = json.load(response)["data"]
    return data["total_credits"] - data["total_usage"]


def cost(usage: dict) -> float:
    # OpenRouter reports the amount actually charged for each call
    return (usage or {}).get("cost", 0) or 0


def resume(path: Path) -> list[dict]:
    # Read finished rows; a half-written last line from a killed run is dropped
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    rows = []
    for number, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if number != len(lines) - 1:
                raise SystemExit(f"{path} line {number + 1} is corrupt")
            path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
            print(f"dropped a half-written last line in {path}")
    return rows


def run_all(items: Iterable, work: Callable[[object], tuple[dict, float]], out: Path,
            spent: float, max_cost: float, workers: int) -> float:
    """Run ``work`` on each item, a few at a time, appending each finished row to ``out``.

    ``work`` returns the row and what it cost. No new call starts once ``spent``
    reaches ``max_cost``; calls already in flight are finished and saved.
    """
    lock, pending, queue = threading.Lock(), set(), list(items)
    with out.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(workers) as pool:
        def save(future):
            nonlocal spent
            row, charged = future.result()
            with lock:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
                spent += charged
        try:
            while True:
                while queue and len(pending) < workers and spent < max_cost:
                    pending.add(pool.submit(work, queue.pop(0)))
                if not pending:
                    break
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in done:
                    save(future)
        finally:
            for future in wait(pending).done:
                if not future.exception():
                    save(future)
    if queue:
        raise BudgetStop(f"spending cap ${max_cost:.2f} reached (spent ${spent:.4f}); rerun with a higher cap to continue")
    return spent
