# Agent TODO

The current agent work is evaluation of simple controller rules. Verifier training and model comparisons remain in [`verifier/TODO.md`](../verifier/TODO.md).

- [x] Keep the fixed one-search, one-verify FEVER loop runnable with `python -m agent.agent_loop`; archived traces are in [`data/traces/`](data/traces/) and the schema is in [`docs/trace-format.md`](docs/trace-format.md).
- [x] Isolate the 12-claim synthetic controller scaffold under [`experiments/toy_scaffold/`](experiments/toy_scaffold/). Its results are interface checks, not FEVER evidence.
- [ ] Run [`evaluation/eval.sbatch`](evaluation/eval.sbatch) on the frozen 900 FEVER dev claims. Compare the first stochastic verifier draw with the simulated NEI/confidence retry policy; inspect FEVER score, verdict and evidence disagreement, early wrong stops, call cost, and gold-evidence coverage.
- [ ] Test a learned controller only if retry improves the matched FEVER accuracy–cost tradeoff. Build controller rollouts from FEVER **train** claims; keep the frozen 900 dev claims for evaluation.
