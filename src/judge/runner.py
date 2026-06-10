"""Judge runner (export seam, file 2 of 3).

Scores the 140 standardized items (items.py) against the rubric (rubric.yaml)
with three pinned OpenRouter judges, one pass each, temperature 0.

OpenRouter is OpenAI-compatible, so this is a single provider-neutral client
with swappable model slugs -- NOT the Anthropic SDK. Slugs are pinned (no
auto-latest) and recorded in every output row for reproducibility.

Output: a tidy long table at results/judge/scores.jsonl, one row per
(item_id, judge, criterion, target). The agreement layer consumes only this
table. The run is resumable: rows already present (same id|judge|criterion|
target key) are skipped, so an interrupted run just continues.

Run from src/:  OPENROUTER_API_KEY=... uv run python -m judge.runner
Dry run (build prompts, no API calls):  uv run python -m judge.runner --dry-run
"""

import argparse
import json
import os
import re
import signal
import time
from pathlib import Path

import yaml

from config import RESULTS_DIR
from judge.items import standardized_items

# Pinned judges (Ali, 2026-06-10). Three families = less-correlated raters.
JUDGES = [
    "x-ai/grok-4.3",
    "openai/gpt-5.5",
    "anthropic/claude-opus-4.8",
]

RUBRIC_PATH = Path(__file__).parent / "rubric.yaml"
SCORES_PATH = RESULTS_DIR / "judge" / "scores.jsonl"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
MAX_RETRIES = 3          # for transient errors only (network/rate-limit); doubles backoff
BACKOFF_BASE = 2         # seconds; doubles each attempt (2, 4, 8)
CALL_TIMEOUT = 45        # HARD per-call wall-clock cap (SIGALRM). Fires even during
                         # OpenRouter keepalive floods, where a read-timeout never would.


class CallTimeout(Exception):
    """Raised by SIGALRM when a single judge call exceeds CALL_TIMEOUT."""


def _alarm_handler(signum, frame):
    raise CallTimeout()
ENV_PATH = Path(__file__).parent.parent / ".env"   # src/.env (gitignored)


def load_env(path=ENV_PATH):
    """Minimal .env loader (no python-dotenv dep). KEY=VALUE lines, '#' comments.
    Does not overwrite vars already set in the real environment."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def load_rubric(path=RUBRIC_PATH):
    return yaml.safe_load(path.read_text())


def _judgments(item, rubric):
    """Expand one item into its (criterion, target, filled_anchors) judgments.

    style_match -> 1 (target = the single artist)
    presence    -> 1 per blended artist (target = that artist; the other is context)
    coherence   -> 1 (no target)
    """
    meta = item["metadata"]
    targets = meta["targets"]
    for criterion in meta["criteria"]:
        spec = rubric["criteria"][criterion]
        if criterion == "presence":
            for i, x in enumerate(targets):
                y = targets[1 - i] if len(targets) == 2 else ""
                fill = {"artist_x": x, "artist_y": y}
                yield criterion, x, _render(spec, fill)
        elif criterion == "style_match":
            yield criterion, targets[0], _render(spec, {"artist": targets[0]})
        else:  # coherence (artist-agnostic)
            yield criterion, None, _render(spec, {})


def _render(spec, fill):
    """Resolve placeholders in a criterion's definition + anchors."""
    return {
        "definition": spec["definition"].format(**fill).strip(),
        "anchor_1": spec["anchors"]["1"].format(**fill).strip(),
        "anchor_5": spec["anchors"]["5"].format(**fill).strip(),
    }


def build_prompt(item, criterion, rendered, rubric):
    lo, hi = rubric["scale"]["min"], rubric["scale"]["max"]
    system = (
        "You are an expert evaluator of song lyrics. Score the lyrics below on a "
        f"single criterion using an integer {lo}-{hi} Likert scale. Judge ONLY this "
        "criterion; ignore all others.\n\n"
        f"CRITERION: {criterion}\n"
        f"DEFINITION: {rendered['definition']}\n\n"
        f"ANCHOR {lo} (lowest): {rendered['anchor_1']}\n"
        f"ANCHOR {hi} (highest): {rendered['anchor_5']}\n"
        f"Scores {lo+1}-{hi-1} interpolate between the anchors.\n\n"
        'Respond with ONLY a JSON object: {"score": <integer>, "reasoning": "<one sentence>"}.'
    )
    user = f"LYRICS:\n\n{item['text']}"
    return system, user


_INT_RE = re.compile(r"-?\d+")


def parse_score(content, lo, hi):
    """Pull an integer score from the model reply; clamp to [lo, hi]."""
    reasoning = ""
    score = None
    try:
        obj = json.loads(content)
        score = int(obj["score"])
        reasoning = str(obj.get("reasoning", ""))
    except Exception:
        m = _INT_RE.search(content or "")
        if m:
            score = int(m.group())
    if score is None:
        return None, content
    return max(lo, min(hi, score)), reasoning


def _load_done(path):
    """Resume support: keys already scored."""
    done = set()
    if path.exists():
        for line in path.open():
            r = json.loads(line)
            done.add((r["item_id"], r["judge"], r["criterion"], r["target"]))
    return done


def run(dry_run=False, out=SCORES_PATH, judges=JUDGES):
    rubric = load_rubric()
    lo, hi = rubric["scale"]["min"], rubric["scale"]["max"]
    items = list(standardized_items())

    # Flatten to the full work list of (item, criterion, target, prompt) x judge.
    work = []
    for item in items:
        for criterion, target, rendered in _judgments(item, rubric):
            system, user = build_prompt(item, criterion, rendered, rubric)
            for judge in judges:
                work.append((item, criterion, target, judge, system, user))

    if dry_run:
        print(f"items={len(items)} judges={len(judges)} judgments={len(work)}")
        item, criterion, target, judge, system, user = work[0]
        print(f"\n--- sample prompt: {item['id']} / {criterion} / target={target} / {judge} ---")
        print(system)
        print(user[:200], "...")
        return

    from openai import OpenAI
    from tqdm import tqdm

    load_env()
    print(f">> runner v4 (hard-cap skip): timeout={CALL_TIMEOUT}s, retries={MAX_RETRIES}, judges={len(judges)}")
    client = OpenAI(
        base_url=OPENROUTER_BASE_URL,
        api_key=os.environ["OPENROUTER_API_KEY"],
        timeout=CALL_TIMEOUT,   # per-call; a stalled connection raises instead of hanging forever
        max_retries=0,          # we do our own retry/backoff below
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    done = _load_done(out)
    todo = [w for w in work if (w[0]["id"], w[3], w[1], w[2]) not in done]
    if not todo:
        print(f"nothing to do -- all {len(work)} judgments already in {out}")
        return

    def _call(judge, system, user):
        """One streamed judge call under a HARD SIGALRM wall-clock cap.
        Streaming so OpenRouter ': OPENROUTER PROCESSING' keepalive comments
        (slow/reasoning models like grok) don't break JSON parsing -- SSE
        comments yield no chunk. The alarm fires even while blocked in the
        socket read, which a read-timeout cannot do during a keepalive flood."""
        signal.alarm(CALL_TIMEOUT)
        try:
            stream = client.chat.completions.create(
                model=judge, temperature=0, stream=True,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}],
            )
            parts = []
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    parts.append(chunk.choices[0].delta.content)
            return "".join(parts)
        finally:
            signal.alarm(0)

    signal.signal(signal.SIGALRM, _alarm_handler)
    written = 0
    skipped = []          # (item_id, judge, criterion, reason)
    last_item = None
    pbar = tqdm(todo, unit="judgment", desc="judging")
    with out.open("a") as f:
        for item, criterion, target, judge, system, user in pbar:
            # Print the lyric once per new item, above the bar.
            if item["id"] != last_item:
                last_item = item["id"]
                head = " / ".join(item["metadata"]["targets"])
                snippet = " / ".join(item["text"].split("\n")[:2])[:90]
                tqdm.write(f"\n[{item['id']}] {head}\n  “{snippet}…”")

            # A reasoning-stall is deterministic, so a timeout SKIPS immediately
            # (retrying just burns another CALL_TIMEOUT). Transient errors
            # (network/rate-limit) retry with backoff, then skip. Skips are never
            # written, so a resume re-attempts them.
            content = None
            for attempt in range(MAX_RETRIES):
                try:
                    content = _call(judge, system, user)
                    break
                except CallTimeout:
                    skipped.append((item["id"], judge, criterion, f"timeout>{CALL_TIMEOUT}s"))
                    tqdm.write(f"  SKIP {item['id']} {judge.split('/')[-1]} {criterion} -- stalled >{CALL_TIMEOUT}s")
                    break
                except Exception as e:
                    wait = BACKOFF_BASE * (2 ** attempt)
                    tqdm.write(f"  retry {attempt+1}/{MAX_RETRIES} ({judge.split('/')[-1]}) "
                               f"after {type(e).__name__}: {str(e)[:120]} -- sleeping {wait}s")
                    time.sleep(wait)
            if content is None:
                if not skipped or skipped[-1][:2] != (item["id"], judge):
                    skipped.append((item["id"], judge, criterion, f"failed {MAX_RETRIES}x"))
                    tqdm.write(f"  SKIP {item['id']} {judge.split('/')[-1]} {criterion} -- failed, will retry on resume")
                continue
            score, reasoning = parse_score(content, lo, hi)
            f.write(json.dumps({
                "item_id": item["id"],
                "config_id": item["metadata"]["config_id"],
                "item_type": item["metadata"]["item_type"],
                "judge": judge,
                "criterion": criterion,
                "target": target,
                "score": score,
                "reasoning": reasoning,
            }, ensure_ascii=False) + "\n")
            f.flush()
            written += 1

            tag = criterion if target is None else f"{criterion}:{target}"
            short_judge = judge.split("/")[-1]
            shown = "??" if score is None else score
            tqdm.write(f"  {shown}  {tag:<22} {short_judge}")
            if score is None:
                tqdm.write(f"  WARN unparseable -> {content!r:.80}")
            pbar.set_postfix_str(f"{written} written, {len(skipped)} skipped")
    print(f"\nwrote {written} new rows -> {out} (total work {len(work)})")
    if skipped:
        from collections import Counter
        by_judge = Counter(j for _, j, _, _ in skipped)
        print(f"skipped {len(skipped)} judgments (re-run to retry them):")
        for j, n in by_judge.items():
            print(f"  {n:>3}  {j}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="build prompts, no API calls")
    args = ap.parse_args()
    run(dry_run=args.dry_run)
