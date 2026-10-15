"""Train native-thinking GRPO from the disjoint 4,695-row SFT adapter."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from verifier.experiments._shared.grpo_cot import main


if __name__ == "__main__":
    main(Path(__file__).parent.name, n_train=1200, default_steps=0)
