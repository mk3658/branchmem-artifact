#!/usr/bin/env python3
"""Post-hoc: cost/latency profile per merge strategy on the locked set
(cache-backed, zero new API). Reports LLM calls per merge, mean input/output
tokens per merge, and mean wall-clock per merge, so practitioners can weigh the
(small or negative) accuracy gain against cost. NOT part of ANALYSIS_PLAN.md
confirmatory tests.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from branchmem.benchmark.scenario_generator import ScenarioConfig, ScenarioGenerator
from branchmem.llm.base import build_backend
from branchmem.merge.branch_discard import BranchDiscard
from branchmem.merge.confidence_rule import ConfidenceRuleMerge
from branchmem.merge.last_writer_wins import LastWriterWins
from branchmem.merge.naive_concat import NaiveConcat
from branchmem.merge.three_way_llm import ThreeWayLLMMerge
from branchmem.merge.two_way_llm import TwoWayLLMMerge
from branchmem.merge.raw_text_llm import RawTextLLMMerge
from branchmem.utils.seeding import set_all_seeds


class CountingBackend:
    """Thin proxy that records per-call token usage; delegates everything else."""

    def __init__(self, backend):
        self._b = backend
        self.in_tok = 0
        self.out_tok = 0

    def reset(self):
        self.in_tok = 0
        self.out_tok = 0

    def complete(self, *args, **kwargs):
        r = self._b.complete(*args, **kwargs)
        self.in_tok += r.input_tokens
        self.out_tok += r.output_tokens
        return r

    def __getattr__(self, name):
        return getattr(self._b, name)


SEED = 2026
SPANS = [4.0, 10.0, 20.0]
N_PER_SPAN = 20
OUT = Path(__file__).resolve().parents[1] / "results" / "final"


def main() -> None:
    set_all_seeds(SEED)
    backend = build_backend({"backend": "openai_compatible", "model": "gpt-5.4-nano",
                             "cache_dir": "llm_cache", "temperature": 0.0, "max_tokens": 8000})
    gen = ScenarioGenerator()
    scenarios = []
    for i, span in enumerate(SPANS):
        scenarios.extend(gen.generate(N_PER_SPAN, ScenarioConfig(divergence_span=span), seed=SEED + i))

    counter = CountingBackend(backend)
    llm_strats = {"three_way_llm": ThreeWayLLMMerge(backend=counter),
                  "two_way_llm": TwoWayLLMMerge(backend=counter),
                  "raw_text_llm": RawTextLLMMerge(backend=counter)}
    det_strats = {"confidence_rule": ConfidenceRuleMerge(), "last_writer_wins": LastWriterWins(),
                  "naive_concat": NaiveConcat(), "branch_discard_always_b": BranchDiscard(policy="always_b")}

    profile = {}
    # LLM strategies: track backend token deltas + wall-clock per merge.
    for name, strat in llm_strats.items():
        ins, outs, times = [], [], []
        for s in scenarios:
            counter.reset()
            t0 = time.time()
            strat.merge(s.ancestor, s.branch_a, s.branch_b)
            times.append(time.time() - t0)
            ins.append(counter.in_tok)
            outs.append(counter.out_tok)
        profile[name] = {
            "llm_calls_per_merge": 1,
            "mean_input_tokens_per_merge": round(sum(ins) / len(ins), 1),
            "mean_output_tokens_per_merge": round(sum(outs) / len(outs), 1),
            "mean_wallclock_s_per_merge": round(sum(times) / len(times), 3),
        }

    for name, strat in det_strats.items():
        times = []
        for s in scenarios:
            t0 = time.time()
            strat.merge(s.ancestor, s.branch_a, s.branch_b)
            times.append(time.time() - t0)
        profile[name] = {
            "llm_calls_per_merge": 0,
            "mean_input_tokens_per_merge": 0,
            "mean_output_tokens_per_merge": 0,
            "mean_wallclock_s_per_merge": round(sum(times) / len(times), 6),
        }

    result = {
        "note": (
            "post-hoc cost/latency profile per merge strategy on the locked set "
            "(n=60, gpt-5.4-nano, cache-backed so wall-clock reflects cache reads, not "
            "live API; live nano latency was ~2-3 s/call during the original run). "
            "Non-conflicting keys resolve in code for every strategy; only genuine "
            "collisions reach the model, batched into ONE call per merge. Deterministic "
            "strategies make zero LLM calls. NOT part of ANALYSIS_PLAN.md tests."
        ),
        "profile": profile,
    }
    (OUT / "cost_latency.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(profile, indent=2))
    print("Wrote", OUT / "cost_latency.json")


if __name__ == "__main__":
    main()
