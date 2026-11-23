"""Combine the shard files of one evaluation into a single report.

eval_policy.py --shard I/N evaluates every Nth pose of the fixed test set, so N
jobs can share it. This reads their --out files, checks they cover the same
checkpoint and settings with no pose missing or repeated, and prints the report
for the whole set (success rate with its confidence interval, outcomes, splits
by region and by distance to the training data).

Usage:
    python merge_eval.py eval/act_040000_*.json --out eval/act_040000_100.json
"""
import argparse
import json
import pathlib

from eval_policy import summarize


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("shards", nargs="+", help="--out files from eval_policy.py")
    ap.add_argument("--out", default=None, help="write the merged result as JSON")
    ap.add_argument("--expect", type=int, default=None,
                    help="number of poses the full test set has, to catch a lost shard")
    args = ap.parse_args()

    docs = [json.loads(pathlib.Path(p).read_text()) for p in args.shards]
    for d in docs:      # files from before the fixed test set record only a seed
        d.setdefault("poses", f"random seed {d.get('seed')}")
    for key in ("checkpoint", "poses", "n_action_steps", "temporal_ensemble_coeff",
                "max_steps"):
        values = {json.dumps(d.get(key)) for d in docs}
        if len(values) > 1:
            raise SystemExit(f"shards disagree on {key}: {sorted(values)}")

    results = sorted((r for d in docs for r in d["results"]),
                     key=lambda r: r["pose_id"])
    ids = [r["pose_id"] for r in results]
    if len(set(ids)) != len(ids):
        raise SystemExit("a pose appears in more than one shard")
    expect = args.expect or max(ids) + 1
    missing = sorted(set(range(expect)) - set(ids))
    if missing:
        raise SystemExit(f"{len(missing)} poses missing (first: {missing[:8]}); "
                         f"a shard is incomplete or absent")

    print(f"{docs[0]['checkpoint']}\n{len(docs)} shards, {len(results)} poses "
          f"from {docs[0]['poses']}")
    k, n, lo, hi = summarize(results)

    if args.out:
        out = pathlib.Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        merged = {key: docs[0].get(key) for key in (
            "checkpoint", "poses", "max_steps", "n_action_steps",
            "temporal_ensemble_coeff")}
        merged.update({"shard": None, "episodes": n, "successes": k,
                       "success_rate": k / n, "ci95": [lo, hi], "results": results})
        out.write_text(json.dumps(merged, indent=2))
        print(f"results -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
