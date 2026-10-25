"""Head-only policy-gradient pilot for Jev STOP/RETRY on FEVER train claims.

Uses precomputed three-draw Qwen rollouts for the exact committed 5,500 SFT
train IDs. A sampled policy trajectory includes final pooled-evidence retry
decisions. Reward uses pooled-output quality and extra-call cost; chunk
actions receive only downstream reward. No action labels are required.
"""

import argparse
import copy
import importlib.metadata
import json
import random
import subprocess
from pathlib import Path

from agent.data.build_evidence_batches import scored, training_claims
from agent.evaluation.pooled_retry import FINAL_QUESTION, selected_draw_index, state_for_pool
from agent.evaluation.results.rsi_jev_evidence_selection_baseline.eval_rsi_jev_evidence_selection_baseline import (
    ACTION, DROP, KEEP, cited_candidates, questions_for, selection_state_for,
)
from agent.evaluation.results.rsi_jev_retry_baseline.eval_rsi_jev_retry_baseline import (
    QUESTION, evidence_for, state_for,
)
from agent.evaluation.results.sampling_retry.eval_sampling_retry import PROMPTS
from agent.train.jev_head import JevHeadPolicy, choice_question
from agent.train.replay_env import (MAX_DRAWS, RETRY, STOP,
                                    best_gold_set_evidence_credit,
                                    verdict_retry_feedback)
from verifier.model import LocalModel, gold_sets, render


def validate_train_source(source):
    manifest_path = source / "manifest.json"
    metrics_path = source / "metrics.json"
    rows_path = source / "predictions.jsonl"
    if any(not path.is_file() for path in (manifest_path, metrics_path, rows_path)):
        raise FileNotFoundError("incomplete train-only three-draw source")
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("split") != "train" or manifest.get("batches_only") is not True
            or manifest.get("draws_per_chunk") != MAX_DRAWS
            or manifest.get("do_sample") is not True
            or manifest.get("batch_size") != 10
            or manifest.get("n_claims") != 5500):
        raise ValueError("RL source must contain 5,500 train-only three-draw 10-sentence claims")
    ids_file = manifest.get("claim_ids_file")
    if not ids_file or Path(ids_file).resolve() != Path(
            "verifier/data/train/sft/fever-oracle-direct-5500/labeled_prompts.jsonl").resolve():
        raise ValueError("train source must use the committed direct-SFT 5,500 IDs")
    claims = {c["id"]: c for c in training_claims(ids_file)}
    with rows_path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    if len(rows) != 5500 or {r["id"] for r in rows} != set(claims):
        raise ValueError("train rollout IDs differ from committed 5,500 train IDs")
    if json.loads(metrics_path.read_text()).get("n_claims") != 5500:
        raise ValueError("train rollout metrics are incomplete")
    for row in rows:
        if row["claim"] != claims[row["id"]]["claim"]:
            raise ValueError("train claim text mismatch")
        if [item for chunk in row["chunks"] for item in chunk["evidence"]] != row["shown"]:
            raise ValueError("train chunk order mismatch")
        if any(len(chunk["draws"]) != MAX_DRAWS for chunk in row["chunks"]):
            raise ValueError("train chunk lacks three draws")
    return manifest, rows, claims


def pooled_decision_reward(pooled, claim, chunk_extra_draws, extra_draw_cost=0.05,
                           evidence_weight=0.5):
    """Verdict-only retry feedback plus partial best-gold-set evidence credit."""
    final = pooled["selected"]
    feedback = [verdict_retry_feedback(decision["action"],
                                       pooled["draws"][decision["draw_index"]], final)
                for decision in pooled["decisions"] if decision["draw_index"] < MAX_DRAWS - 1]
    action_reward = sum(feedback) / len(feedback) if feedback else 0.0
    verdict_reward = 1.0 if final["correct"] else -1.0
    gold_groups = [{f"{page}:{line}" for page, line in group}
                   for group in gold_sets(claim)]
    evidence_credit, evidence_detail = best_gold_set_evidence_credit(
        final["gold"], final.get("evidence", []), gold_groups)
    extra_draws = chunk_extra_draws + pooled["extra_draws"]
    call_penalty = extra_draw_cost * extra_draws
    return verdict_reward + evidence_weight * evidence_credit + action_reward - call_penalty, {
        "final_verdict_correct": bool(final["correct"]), "verdict_reward": verdict_reward,
        "evidence_credit": evidence_credit, "evidence_weight": evidence_weight,
        "evidence_detail": evidence_detail,
        "pooled_action_feedback": feedback, "pooled_action_reward": action_reward,
        "chunk_extra_draws": chunk_extra_draws,
        "pooled_extra_draws": pooled["extra_draws"], "call_penalty": call_penalty,
        "action_rule": "STOP correct +1 / wrong -1; RETRY correct -1.5; RETRY wrong +0.25 only if final verdict correct, else -1"}


def sample_question(policy, reference, state, key, specification, kl_coef):
    """One sampled typed choice with differentiable option log probability."""
    torch = policy.torch
    question = choice_question(key, specification["instructions"], specification["criteria"])
    logits = policy.logits(state, question)
    distribution = torch.distributions.Categorical(logits=logits)
    index = distribution.sample()
    kl = torch.zeros((), device=logits.device)
    if kl_coef:
        with torch.no_grad():
            ref_logits = policy.logits(state, question, scorer=reference)
            ref_probs = ref_logits.softmax(-1)
        kl = torch.nn.functional.kl_div(logits.log_softmax(-1), ref_probs, reduction="sum")
    return (question.options[int(index)], distribution.log_prob(index),
            distribution.entropy(), kl)


def sample_episode(row, policy, reference, variant, kl_coef=0.0):
    saved = set()
    trace = []
    log_probs, entropies, kls = [], [], []
    extra_draws = 0
    numbers = {item["id"]: f"E{i + 1}" for i, item in enumerate(row["shown"])}
    for ci, chunk in enumerate(row["chunks"]):
        chunk_saved = set()
        for di, draw in enumerate(chunk["draws"]):
            state = (selection_state_for(row, ci, di) if variant == "selection"
                     else state_for(row, ci, di))
            if di < MAX_DRAWS - 1:
                spec = QUESTION["action"] if variant == "citation" else ACTION
                action, logp, entropy, kl = sample_question(
                    policy, reference, state, "action", spec, kl_coef)
                log_probs.append(logp)
                entropies.append(entropy)
                kls.append(kl)
            else:
                action = STOP
            kept = set()
            if variant == "selection":
                for key, spec in questions_for(chunk, draw, numbers, allow_retry=False).items():
                    choice, logp, entropy, kl = sample_question(
                        policy, reference, state, key, spec, kl_coef)
                    log_probs.append(logp)
                    entropies.append(entropy)
                    kls.append(kl)
                    if choice == KEEP:
                        candidate_index = int(key.split("_")[1])
                        kept.add(cited_candidates(chunk, draw)[candidate_index]["id"])
                chunk_saved.update(kept)
            elif action == STOP and draw.get("format_ok"):
                kept = set(draw.get("evidence", []))
                allowed = {item["id"] for item in chunk["evidence"]}
                if not kept <= allowed:
                    raise ValueError("verifier citation outside current train chunk")
                chunk_saved.update(kept)
            trace.append({"chunk_index": ci, "draw_index": di, "action": action,
                          "kept_ids": sorted(kept)})
            if action == STOP:
                break
            extra_draws += 1
        saved.update(chunk_saved)
    selected = [item for item in row["shown"] if item["id"] in saved]
    return {"trace": trace, "selected_evidence": selected, "extra_draws": extra_draws,
            "log_probs": log_probs, "entropies": entropies, "kls": kls}


def sample_pooled_episode(row, claim, episode, verifier, policy, reference,
                          kl_coef, seed):
    """Train the same pooled STOP/RETRY decision used at validation time."""
    evidence = evidence_for(episode["selected_evidence"])
    prompt = render(row["claim"], evidence)
    with policy.torch.no_grad():
        generations = verifier.run([prompt] * MAX_DRAWS, seed=seed)
        if len(generations) != MAX_DRAWS:
            raise ValueError("pooled verifier did not return three draws")
        draws = [scored(claim, evidence, generation) for generation in generations]
    decisions = []
    log_probs, entropies, kls = [], [], []
    for di, draw in enumerate(draws):
        if di < MAX_DRAWS - 1:
            action, logp, entropy, kl = sample_question(
                policy, reference,
                state_for_pool(row["claim"], episode["selected_evidence"], draw, di),
                "action", FINAL_QUESTION["action"], kl_coef)
            log_probs.append(logp)
            entropies.append(entropy)
            kls.append(kl)
        else:
            action = STOP
        decisions.append({"draw_index": di, "action": action})
        if action == STOP:
            break
    selected = selected_draw_index(draws, len(decisions))
    return {"selected": draws[selected], "selected_draw_index": selected,
            "stopped_at_draw_index": len(decisions) - 1,
            "decisions": decisions, "draws": draws,
            "extra_draws": len(decisions) - 1,
            "log_probs": log_probs, "entropies": entropies, "kls": kls}


def run(args):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Jev RL pilot requires a CUDA GPU allocation")
    source = Path(args.source_run)
    source_manifest, rows, claims = validate_train_source(source)
    output = Path(args.output_dir)
    if output.exists():
        raise FileExistsError(f"training output already exists: {output}")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    policy = JevHeadPolicy(args.jev_model, device="cuda")
    reference = copy.deepcopy(policy.model.scorer).eval() if args.kl_coef else None
    if reference is not None:
        for parameter in reference.parameters():
            parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(policy.model.scorer.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    system, thinking, _ = PROMPTS[source_manifest["prompt_style"]]
    verifier = LocalModel(source_manifest["model"],
                          batch_size=source_manifest["model_batch_size"],
                          adapter=source_manifest["adapter"], system_prompt=system,
                          max_new_tokens=source_manifest["max_new_tokens"],
                          enable_thinking=thinking, do_sample=True,
                          temperature=source_manifest.get("temperature", 0.7),
                          top_p=source_manifest.get("top_p", 0.9))
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=False).stdout.strip() or None
    record = {"training_code": __name__, "split": "FEVER train",
              "train_ids_file": source_manifest["claim_ids_file"], "n_claims": len(rows),
              "source_run": str(source), "source_commit": source_manifest.get("commit"),
              "commit": revision, "jev_model": args.jev_model,
              "jev_package_version": importlib.metadata.version("rsi-jev"),
              "variant": args.variant, "seed": args.seed, "epochs": args.epochs,
              "learning_rate": args.lr, "weight_decay": args.weight_decay,
              "kl_coef": args.kl_coef, "entropy_coef": args.entropy_coef,
              "baseline_ema": args.baseline_ema,
              "reward_rule": "selected verdict +/-1 + 0.5 * best-gold-set partial evidence credit + mean asymmetric verdict-only pooled-action feedback - 0.05 per extra draw",
              "pooled_action_rule": "STOP correct +1 / wrong -1; RETRY correct -1.5; RETRY wrong +0.25 only if selected verdict becomes correct, otherwise -1",
              "gold_evidence_rule": "best alternative set: overlap/size + 0.25 complete-set bonus - 0.25 off-gold citation fraction; NEI empty +0.25 else -0.25",
              "chunk_gold_action_reward": False,
              "pooled_draws": MAX_DRAWS,
              "pooled_seed_offset": 100000,
              "pooled_do_sample": True,
              "pooled_temperature": source_manifest.get("temperature", 0.7),
              "pooled_top_p": source_manifest.get("top_p", 0.9),
              "pooled_policy": "early STOP uses current draw; forced third draw uses verdict majority then token confidence",
              "updated_parameters": "RSI-Jev option scorer only; tower frozen",
              "validation_claims_used": False}
    output.mkdir(parents=True)
    (output / "manifest.json").write_text(json.dumps(record, indent=2) + "\n")
    baseline = 0.0
    index = 0
    with (output / "episodes.jsonl").open("w", encoding="utf-8") as stream:
        for epoch in range(args.epochs):
            random.Random(args.seed + epoch).shuffle(rows)
            for row in rows:
                index += 1
                claim = claims[row["id"]]
                episode = sample_episode(row, policy, reference, args.variant, args.kl_coef)
                pooled = sample_pooled_episode(
                    row, claim, episode, verifier, policy, reference, args.kl_coef,
                    seed=source_manifest["generation_seed"] + 100000 + int(row["id"])
                         + epoch * 1000000)
                final = pooled["selected"]
                saved = [item["id"] for item in episode["selected_evidence"]]
                extra_draws = episode["extra_draws"] + pooled["extra_draws"]
                reward, reward_detail = pooled_decision_reward(
                    pooled, claim, episode["extra_draws"])
                advantage = reward - baseline
                baseline += args.baseline_ema * advantage
                log_probs = episode["log_probs"] + pooled["log_probs"]
                entropies = episode["entropies"] + pooled["entropies"]
                kls = episode["kls"] + pooled["kls"]
                if log_probs:
                    loss = -advantage * torch.stack(log_probs).sum()
                    loss = loss / len(log_probs)
                    if args.entropy_coef:
                        loss = loss - args.entropy_coef * torch.stack(entropies).mean()
                    if args.kl_coef:
                        loss = loss + args.kl_coef * torch.stack(kls).mean()
                    if not torch.isfinite(loss):
                        raise FloatingPointError("non-finite Jev policy loss")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(policy.model.scorer.parameters(), 1.0)
                    optimizer.step()
                    loss_value = float(loss.detach())
                else:
                    loss_value = None
                event = {"epoch": epoch, "index": index, "claim_id": row["id"],
                         "reward": reward, "baseline": baseline, "loss": loss_value,
                         "final_correct": final["correct"], "final_fever": final["fever_score"],
                         "extra_draws": extra_draws,
                         "chunk_extra_draws": episode["extra_draws"],
                         "pooled_extra_draws": pooled["extra_draws"],
                         "saved_evidence": saved, "reward_detail": reward_detail,
                         "trace": episode["trace"],
                         "pooled_trace": pooled["decisions"],
                         "pooled_selected_draw_index": pooled["selected_draw_index"],
                         "pooled_draw_summaries": [
                             {key: draw.get(key) for key in ("pred", "token_conf", "format_ok",
                                                               "evidence", "correct", "fever_score")}
                             for draw in pooled["draws"]]}
                stream.write(json.dumps(event, ensure_ascii=False) + "\n")
                stream.flush()
                if index % args.save_every == 0:
                    policy.save(output / "artifacts" / "checkpoints" / "scorer.pt",
                                {"episode": index, "release": args.jev_model,
                                 "variant": args.variant, "baseline": baseline})
                    print(f"{index}/{len(rows) * args.epochs} reward={reward:.3f} baseline={baseline:.3f}", flush=True)
    policy.save(output / "artifacts" / "checkpoints" / "scorer.pt",
                {"episode": index, "release": args.jev_model,
                 "variant": args.variant, "baseline": baseline})
    print(f"Training completed: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--jev-model", default="shgao/rsi-jev-v2.1-qwen3.5-2b")
    parser.add_argument("--variant", choices=("citation", "selection"), required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--kl-coef", type=float, default=0.01)
    parser.add_argument("--entropy-coef", type=float, default=0.001)
    parser.add_argument("--baseline-ema", type=float, default=0.02)
    parser.add_argument("--save-every", type=int, default=100)
    args = parser.parse_args()
    if args.epochs < 1 or args.save_every < 1 or args.lr <= 0 or not 0 < args.baseline_ema <= 1:
        parser.error("epochs/save-every/lr must be positive and baseline-ema in (0, 1]")
    run(args)


if __name__ == "__main__":
    main()
