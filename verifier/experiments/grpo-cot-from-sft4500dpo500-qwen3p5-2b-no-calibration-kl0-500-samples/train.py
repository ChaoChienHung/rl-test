"""Native-thinking GRPO on the GRPO500 split after SFT4500 then DPO500."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from verifier.experiments._shared.grpo_cot import main


if __name__ == "__main__":
    main(Path(__file__).parent.name, n_train=500, default_steps=0, require_dpo=True)
