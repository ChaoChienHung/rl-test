"""Compatibility entry point for Slurm jobs submitted before the code move.

New submissions use agent.evaluation.results.sampling_retry.eval_sampling_retry.
"""

from agent.evaluation.results.sampling_retry.eval_sampling_retry import main


if __name__ == "__main__":
    main()
