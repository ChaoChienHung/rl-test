"""Train Fast-Slow Native-Thinking GRPO with Cognitive Self-Correction Reward on 1,000 claims."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from verifier.experiments._shared.grpo_fastslow import main


if __name__ == "__main__":
    main(Path(__file__).parent.name, n_train=1000, default_steps=0)
