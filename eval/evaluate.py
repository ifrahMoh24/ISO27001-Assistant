"""Measure retrieval accuracy on eval/questions.json.

Run from the project root:
    python -m eval.evaluate

Reports, for in-scope questions:
  - hit@1: the top result is one of the expected controls
  - hit@k: an expected control is anywhere in the top k results
  - MRR:   mean reciprocal rank of the first expected control
and, for off-topic questions, how many are correctly rejected (nothing
passes the distance cutoff). A sweep over cutoffs shows the trade-off
that MAX_DISTANCE is chosen from.
"""

import argparse
import json
from pathlib import Path

from backend.retrieval import MAX_DISTANCE, search

QUESTIONS_PATH = Path(__file__).resolve().parent / "questions.json"


def matches(source, expected) -> bool:
    return source.control in expected or source.title in expected


def first_hit_rank(sources, expected):
    for rank, s in enumerate(sources, start=1):
        if matches(s, expected):
            return rank
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-k", type=int, default=4)
    args = parser.parse_args()
    k = args.top_k

    cases = json.loads(QUESTIONS_PATH.read_text())["questions"]
    in_scope = [c for c in cases if c["expected"]]
    off_topic = [c for c in cases if not c["expected"]]

    # Retrieve once without a cutoff, then apply cutoffs afterwards.
    results = {c["question"]: search(c["question"], k, max_distance=2.0) for c in cases}

    def evaluate(cutoff):
        hit1 = hitk = rr = 0.0
        misses = []
        for c in in_scope:
            kept = [s for s in results[c["question"]] if s.distance <= cutoff]
            rank = first_hit_rank(kept, c["expected"])
            hit1 += rank == 1
            hitk += rank is not None
            rr += 1 / rank if rank else 0
            if rank != 1:
                top = kept[0].control or kept[0].title if kept else "nothing"
                misses.append((c["question"], c["expected"], top, rank))
        rejected = sum(
            all(s.distance > cutoff for s in results[c["question"]]) for c in off_topic
        )
        n = len(in_scope)
        return hit1 / n, hitk / n, rr / n, rejected, misses

    hit1, hitk, mrr, rejected, misses = evaluate(MAX_DISTANCE)

    print(f"\nRetrieval evaluation  ({len(in_scope)} in-scope, {len(off_topic)} off-topic questions, "
          f"top_k={k}, cutoff={MAX_DISTANCE})")
    print("-" * 70)
    print(f"hit@1               {hit1:6.1%}")
    print(f"hit@{k}               {hitk:6.1%}")
    print(f"MRR                 {mrr:6.3f}")
    print(f"off-topic rejected  {rejected}/{len(off_topic)}")

    if misses:
        print("\nQuestions where the top result was not an expected control:")
        for q, expected, top, rank in misses:
            where = f"found at rank {rank}" if rank else f"not in top {k}"
            print(f"  - {q}\n      expected {expected}, got {top} ({where})")

    print("\nCutoff sweep (in-scope hit@k vs off-topic rejected):")
    for cutoff in (0.70, 0.75, 0.80, 0.85, 0.90):
        _, hk, _, rej, _ = evaluate(cutoff)
        marker = "  <- current" if abs(cutoff - MAX_DISTANCE) < 1e-9 else ""
        print(f"  {cutoff:.2f}   hit@{k} {hk:6.1%}   rejected {rej}/{len(off_topic)}{marker}")


if __name__ == "__main__":
    main()
