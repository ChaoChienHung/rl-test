# Agent TODO

The current agent work is evaluation of simple controller rules. Verifier training and model comparisons remain in [`verifier/TODO.md`](../verifier/TODO.md).

- [x] Keep the fixed one-search, one-verify FEVER loop runnable with `python -m agent.agent_loop`; archived traces are in [`data/traces/`](data/traces/) and the schema is in [`docs/trace-format.md`](docs/trace-format.md).
- [x] Isolate the 12-claim synthetic controller scaffold under [`experiments/toy_scaffold/`](experiments/toy_scaffold/). Its results are interface checks, not FEVER evidence.
- [ ] Run [`eval.sbatch`](eval.sbatch) on the frozen 900 FEVER dev claims for the sampling-retry evaluation. Compare the first stochastic verifier draw with the simulated NEI/confidence retry policy; inspect FEVER score, verdict and evidence disagreement, early wrong stops, call cost, and gold-evidence coverage.
- [ ] Run exhaustive 10-sentence evidence batches on the frozen 900 dev claims with `EVIDENCE_BATCHES_ONLY=1`. Save the shared batch outputs under Git-ignored `data/evidence_batches/`; inspect empty unions, split gold evidence and token cost. Code and usage: [`evaluation/results/evidence_batches/`](evaluation/results/evidence_batches/).
- [ ] On the completed Qwen 900-claim batch data, compare RSI-Jev v2.1 2B against the predeclared heuristic controller choosing the final Qwen evidence route: cited union or full context. Give both the same claim and batch-result summaries; **after both choices**, run fresh final Qwen verification. Compare paired accuracy, FEVER, evidence F1, per-label recall, route disagreement and call cost. Optionally compare OpenJev-1.7B under the same state and actions. Code and usage: [`evaluation/results/jev_controller/`](evaluation/results/jev_controller/).
- [ ] Test a learned controller only if retry improves the matched FEVER accuracy–cost tradeoff. Build controller rollouts from FEVER **train** claims; keep the frozen 900 dev claims for evaluation.
