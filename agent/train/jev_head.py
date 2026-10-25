"""Differentiable RSI-Jev choice head for train-only policy-gradient pilots.

The released tower is frozen. Only its option scorer is updated. The saved head
is an overlay on the exact release, not a complete RSI-Jev checkpoint.
"""

import json
from pathlib import Path


def choice_question(key, instructions, criteria):
    from rsijev.contract import Question
    return Question(key=key, mode="choice", instructions=instructions,
                    options=tuple(criteria), criteria=criteria)


class JevHeadPolicy:
    def __init__(self, release, head_path=None, device="cuda"):
        import torch
        from serve.release import load_release, resolve_ckpt
        self.torch = torch
        self.device = device
        self.release = release
        self.model, self.tokenizer, self.encoder, self.meta = load_release(
            resolve_ckpt(release), device, infer_dtype=torch.bfloat16 if device == "cuda" else None)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        if head_path is not None:
            block = torch.load(head_path, map_location="cpu", weights_only=True)
            if block["release"] != release:
                raise ValueError("Jev head was trained against another release")
            self.model.scorer.load_state_dict(block["scorer"])
        for parameter in self.model.scorer.parameters():
            parameter.requires_grad_(True)
        self.model.eval()

    def logits(self, state, question, scorer=None):
        import torch
        from rsijev.encode import collate, encode_question, unpermute_logits
        encoded = encode_question(self.tokenizer, state, question, self.encoder)
        batch = collate(self.tokenizer, [encoded], max_options=len(question.options),
                        device=self.device)
        original = self.model.scorer
        if scorer is not None:
            self.model.scorer = scorer
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=self.device == "cuda"):
                output = self.model(**batch)
            output = unpermute_logits(output.float(), batch["option_perm"], batch["option_mask"])
            return output[0, :len(question.options)]
        finally:
            if scorer is not None:
                self.model.scorer = original

    def predict(self, *, state, questions):
        answers = {}
        for key, spec in questions.items():
            q = choice_question(key, spec["instructions"], spec["criteria"])
            with self.torch.no_grad():
                probs = self.logits(state, q).softmax(-1).cpu().tolist()
            choice = q.options[max(range(len(probs)), key=probs.__getitem__)]
            answers[key] = {"choice": choice,
                            "probabilities": dict(zip(q.options, probs)),
                            "confidence": max(probs)}
        return {"answers": answers}

    def save(self, path, extra):
        import torch
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"release": self.release,
                    "scorer": {k: v.detach().cpu() for k, v in self.model.scorer.state_dict().items()}},
                   destination)
        destination.with_suffix(".json").write_text(json.dumps(extra, indent=2) + "\n")
