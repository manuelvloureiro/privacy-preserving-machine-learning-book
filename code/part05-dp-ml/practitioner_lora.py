"""Fine-tune a small language model with LoRA, without and with differential privacy, and measure memorization.

    uv run python code/part05-dp-ml/practitioner_lora.py
    uv run python code/part05-dp-ml/practitioner_lora.py --epsilon 2 --rank 4 --save-adapter adapter_eps2

Options: --seed (default 0), --epsilon (target ε of the DP run, default 8), --rank (LoRA rank r, with
lora_alpha = 2r, default 8) and --save-adapter PATH (off by default). With --save-adapter, the DP adapter is saved
with peft's save_pretrained (adapter_config.json and adapter_model.safetensors, about 400 KB at r = 8), loaded back
onto the same pinned Pythia-14m checkpoint with PeftModel.from_pretrained, and its held-out loss is printed
again as a check. Every printed label uses the actual ε.

Serves two practitioner pages: Part 5's workflow "Fine-tune a language model privately with LoRA" and step 4
(canaries and extraction) of Part 2's risk-assessment workflow. Every number printed with the default seed 0 is
checked by tests/test_practitioner_lora.py, at the precision the pages quote.

Model: Pythia-14m (EleutherAI, Hugging Face EleutherAI/pythia-14m, revision pinned in practitioner_datasets,
licence Apache-2.0), a 14-million-parameter causal language model and its tokenizer, loaded in float32.
Text: Tiny Shakespeare (Karpathy's char-rnn data, public-domain text, 1,115,394 characters), via
practitioner_datasets.shakespeare().

Runtime: it runs two fine-tunings of 96 steps each and audits 15 canaries in each model. The run is reduced
(n, sequence length, epochs) so that it fits a test, and its wall time depends heavily on how busy the CPU is.
Measured on a 12-core CPU shared with other jobs:
- about 2.5 minutes at load average 6-14 (141-185 s);
- 5.8 minutes at load average about 19 (348 s);
- 6.5-7 minutes in a walkthrough on a busy machine.
- 20 minutes at load average 26-33, with several copies of the script running at once.
Expect about 2 minutes on an idle machine. That figure is an estimate: this machine was never idle while the
script was timed, and an earlier, slightly lighter configuration took 122-128 s at load average 2.5-6.5.

What it does:

1. Tokenizes the text and cuts it into sequences of 32 tokens (one or two lines of verse). The first 1,024
   sequences (32,768 tokens, about a tenth of the text) are the private training set; the last 128 sequences are
   held out. The privacy unit is one training sequence, so n = 1,024 and δ = 10^floor(log10(n^-1.1)) = 1e-4.
2. Inserts canaries: secrets in the format "the pin of <name> is d d d d d", digits separated by spaces so that
   every one of the 10^5 candidate secrets is exactly five tokens (checked). Three canaries each are inserted 1,
   4, 16 and 64 times (each copy over the start of a different training sequence, 255 sequences in all, a quarter
   of the training set, so that a small model memorizes within a short run); three more are controls, never
   inserted. A canary with 16 copies is a group of 16 privacy units, so DP's guarantee for it is that of group
   privacy, much weaker than for one sequence.
3. Adds LoRA adapters (r = 8, alpha = 16, on every linear layer: attention query_key_value and dense, MLP
   dense_h_to_4h and dense_4h_to_h) and fine-tunes twice from the same adapters, both with AdamW at learning rate
   1e-3 for 3 epochs: once non-privately (shuffled batches of 32), once with Opacus DP-SGD at ε = 8 (Poisson
   sampling with expected batch 32, so 96 steps; clipping norm 1; noise
   multiplier from make_private_with_epsilon with the PRV accountant), cross-checked with dp_accounting's PLD
   accountant.
4. Reports held-out loss and perplexity for the pretrained model and both fine-tuned models, and the trainable
   parameter counts of LoRA and of full fine-tuning.
5. For each canary, in the non-private and the DP model: the exposure, from the summed log-probability of its five
   digit tokens after its prefix (never the mean loss), ranked against 2,000 random candidate secrets, which caps
   a measurable exposure at log2(2,001) = 10.97 bits (the full space would allow log2(10^5) = 16.61); an exposure
   extrapolated from a skew-normal fit to the candidates' scores (capped at 16.61); and whether greedy decoding
   from the prefix writes the secret. Uninserted controls should average 1/ln 2 = 1.44 bits. The two exposures,
   which no library installable here computes, come from code/practitioner_privacy_tests.py (after
   `compute_exposure_interpolation` and `compute_exposure_extrapolation` in TensorFlow Privacy's
   `privacy_tests/secret_sharer/exposures.py`).

Prints every quoted number on a labelled line. Determinism: torch runs on 4 threads with seed 0; held-out losses
are printed to 3 decimals and exposures to 1 decimal, which reruns reproduce.
"""

import argparse
import copy
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import dp_accounting
import numpy as np
import torch
from numpy.typing import NDArray
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator
from peft import LoraConfig, PeftModel, get_peft_model
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import practitioner_datasets
from practitioner_privacy_tests import exposure, extrapolated_exposure

SEQ_LEN = 32
N_TRAIN, N_HELDOUT = 1024, 128
BATCH, EPOCHS = 32, 3
LR = 1e-3
EPSILON, RANK, MAX_GRAD_NORM = 8.0, 8, 1.0  # defaults of --epsilon and --rank
COPIES: tuple[int, ...] = (1, 4, 16, 64)
PER_GROUP = 3  # canaries per number of copies, and the number of controls
N_CANDIDATES = 2000
SECRET_SPACE = 10**5
NAMES = [
    "Alice", "Brian", "Chloe", "Daniel", "Emily", "Felix", "Grace", "Hugo",
    "Irene", "Jacob", "Karen", "Liam", "Megan", "Nathan", "Olivia",
]  # fmt: skip

Scores = NDArray[np.float64]


def delta_for(n: int) -> float:
    """The largest power of ten at most n^-1.1."""
    return 10.0 ** math.floor(math.log10(n**-1.1))


# --- Data ---------------------------------------------------------------------------------------------


def token_sequences(tokenizer: PreTrainedTokenizerBase, text: str) -> torch.Tensor:
    """The text's tokens cut into consecutive sequences of SEQ_LEN tokens (the remainder dropped)."""
    ids = torch.tensor(tokenizer(text)["input_ids"], dtype=torch.long)
    usable = len(ids) // SEQ_LEN * SEQ_LEN
    return ids[:usable].view(-1, SEQ_LEN)


@dataclass(frozen=True)
class Canary:
    name: str
    secret: str  # five digits separated by spaces, "3 1 4 1 5"
    copies: int  # 0 for a control

    @property
    def prefix(self) -> str:
        return f"the pin of {self.name} is"

    @property
    def text(self) -> str:
        return f"{self.prefix} {self.secret}\n"


def random_secret(rng: np.random.Generator) -> str:
    return " ".join(str(d) for d in rng.integers(0, 10, size=5))


def make_canaries(rng: np.random.Generator) -> list[Canary]:
    counts = [c for c in COPIES for _ in range(PER_GROUP)] + [0] * PER_GROUP
    return [Canary(name, random_secret(rng), c) for name, c in zip(NAMES, counts, strict=True)]


# excerpt: lora-canary-insert
def insert_canaries(
    train: torch.Tensor, canaries: list[Canary], tokenizer: PreTrainedTokenizerBase, rng: np.random.Generator
) -> torch.Tensor:
    """Write each copy of each canary over the start of a different training sequence."""
    train = train.clone()
    total = sum(c.copies for c in canaries)
    rows = iter(rng.choice(len(train), size=total, replace=False).tolist())
    for canary in canaries:
        tokens = torch.tensor(tokenizer(canary.text)["input_ids"])
        for _ in range(canary.copies):
            train[next(rows), : len(tokens)] = tokens
    return train


# end excerpt


# --- Model --------------------------------------------------------------------------------------------


def load_tokenizer(folder: Path) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(folder)  # pyright: ignore[reportUnknownVariableType]
    assert isinstance(tokenizer, PreTrainedTokenizerBase)  # transformers' return type is only partly declared
    return tokenizer


def load_model(folder: Path) -> PreTrainedModel:
    return AutoModelForCausalLM.from_pretrained(folder, dtype=torch.float32)


def trainable(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def add_lora(model: PreTrainedModel, rank: int) -> PeftModel:
    # excerpt: lora-config
    attention_and_mlp = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]  # every linear layer
    config = LoraConfig(r=rank, lora_alpha=2 * rank, target_modules=attention_and_mlp, lora_dropout=0.0)
    lora_model = get_peft_model(model, config)  # freezes the pretrained weights
    trainable_count = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    total_count = sum(p.numel() for p in lora_model.parameters())
    print(f"trainable parameters: LoRA {trainable_count:,}  full fine-tuning {total_count - trainable_count:,}")
    # end excerpt
    return lora_model


@torch.no_grad()
def heldout_loss(model: torch.nn.Module, heldout: torch.Tensor) -> float:
    """Mean next-token cross-entropy (nats per token) on the held-out sequences."""
    model.eval()
    losses = [float(model(input_ids=batch, labels=batch).loss) for batch in torch.split(heldout, BATCH)]
    return float(np.mean(losses))


# --- Training -----------------------------------------------------------------------------------------


def train_nonprivate(model: PeftModel, train: torch.Tensor) -> None:
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    loader = DataLoader(TensorDataset(train), batch_size=BATCH, shuffle=True)
    model.train()
    for _ in range(EPOCHS):
        for (batch,) in loader:
            optimizer.zero_grad()
            model(input_ids=batch, labels=batch).loss.backward()
            optimizer.step()


@dataclass(frozen=True)
class PrivateRun:
    noise_multiplier: float
    sample_rate: float
    steps: int
    epsilon: float  # from Opacus's accountant after training


def train_private(model: PeftModel, train: torch.Tensor, epsilon: float, delta: float) -> PrivateRun:
    print(f"validator errors: {len(ModuleValidator.validate(model, strict=False))}")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    loader = DataLoader(TensorDataset(train), batch_size=BATCH)
    # excerpt: lora-private
    privacy_engine = PrivacyEngine(accountant="prv")
    private_model, optimizer, loader = privacy_engine.make_private_with_epsilon(
        module=model,
        optimizer=optimizer,
        data_loader=loader,  # replaced by Poisson sampling with rate BATCH / n
        target_epsilon=epsilon,
        target_delta=delta,
        epochs=EPOCHS,
        max_grad_norm=MAX_GRAD_NORM,
        grad_sample_mode="hooks",
    )
    print(f"noise multiplier: {optimizer.noise_multiplier:.3f}")
    # end excerpt
    private_model.train()
    steps = 0
    for _ in range(EPOCHS):
        # excerpt: lora-step
        for (batch,) in loader:  # each batch holds a random number of sequences
            optimizer.zero_grad()
            loss = private_model(input_ids=batch, labels=batch).loss
            loss.backward()  # Opacus's hooks keep one gradient per sequence
            optimizer.step()  # clips each one to MAX_GRAD_NORM, sums, adds noise
            steps += 1
        # end excerpt
    private_model.remove_hooks()  # the per-sequence gradient hooks would slow down evaluation
    return PrivateRun(optimizer.noise_multiplier, BATCH / len(train), steps, privacy_engine.get_epsilon(delta))


def pld_epsilon(noise_multiplier: float, sample_rate: float, steps: int, delta: float) -> float:
    step = dp_accounting.PoissonSampledDpEvent(sample_rate, dp_accounting.GaussianDpEvent(noise_multiplier))
    accountant = dp_accounting.pld.PLDAccountant()
    accountant.compose(step, steps)
    return accountant.get_epsilon(delta)


# --- Memorization -------------------------------------------------------------------------------------


# excerpt: lora-exposure
@torch.no_grad()
def secret_scores(model: PeftModel, tokenizer: PreTrainedTokenizerBase, prefix: str, secrets: list[str]) -> Scores:
    """Summed log-probability of each secret's five digit tokens after the prefix."""
    prefix_ids = tokenizer(prefix)["input_ids"]
    rows = [prefix_ids + tokenizer(" " + s)["input_ids"] for s in secrets]
    ids = torch.tensor(rows)  # every secret is five tokens, so the rows have equal length
    logits = model(input_ids=ids[:, :-1], logits_to_keep=5).logits  # the predictions of the five digits
    logprobs = logits.log_softmax(dim=-1).gather(-1, ids[:, -5:, None])
    return logprobs.sum(dim=(1, 2)).double().numpy()


def canary_exposure(
    model: PeftModel, tokenizer: PreTrainedTokenizerBase, canary: Canary, candidates: list[str]
) -> tuple[float, float]:
    """The exposure of the canary's secret among the candidates, measured and extrapolated past the cap."""
    candidate_scores = scores_batched(model, tokenizer, canary.prefix, candidates)
    score = float(scores_batched(model, tokenizer, canary.prefix, [canary.secret])[0])
    return exposure(score, candidate_scores), extrapolated_exposure(score, candidate_scores, SECRET_SPACE)


# end excerpt


def scores_batched(model: PeftModel, tokenizer: PreTrainedTokenizerBase, prefix: str, secrets: list[str]) -> Scores:
    parts = [secret_scores(model, tokenizer, prefix, secrets[i : i + 250]) for i in range(0, len(secrets), 250)]
    return np.concatenate(parts)


# excerpt: lora-extract
@torch.no_grad()
def greedy_completion(model: PeftModel, tokenizer: PreTrainedTokenizerBase, prefix: str) -> str:
    """The five tokens the model writes after the prefix when it always picks the likeliest one."""
    ids = tokenizer(prefix, return_tensors="pt")["input_ids"]
    for _ in range(5):
        next_token = model(input_ids=ids).logits[:, -1].argmax(dim=-1, keepdim=True)
        ids = torch.cat([ids, next_token], dim=1)
    return tokenizer.batch_decode(ids[:, -5:])[0].strip()


# end excerpt


@dataclass(frozen=True)
class CanaryResult:
    canary: Canary
    exposure: float
    extrapolated: float
    extracted: bool


def audit(
    model: PeftModel, tokenizer: PreTrainedTokenizerBase, canaries: list[Canary], candidates: list[str]
) -> list[CanaryResult]:
    model.eval()
    results: list[CanaryResult] = []
    for canary in canaries:
        measured, extrapolated = canary_exposure(model, tokenizer, canary, candidates)
        completion = greedy_completion(model, tokenizer, canary.prefix)
        results.append(CanaryResult(canary, measured, extrapolated, completion == canary.secret))
    return results


def report(label: str, results: list[CanaryResult]) -> None:
    for r in results:
        kind = f"{r.canary.copies:2d} copies" if r.canary.copies else "  control"
        print(
            f"{label} {kind} {r.canary.name:<7} {r.canary.secret}: exposure {r.exposure:4.1f} bits"
            f"  extrapolated {r.extrapolated:4.1f}  greedy {'extracted' if r.extracted else 'no'}"
        )
    for copies in (*COPIES, 0):
        group = [r for r in results if r.canary.copies == copies]
        kind = f"{copies} copies" if copies else "controls"
        mean = np.mean([r.exposure for r in group])
        mean_x = np.mean([r.extrapolated for r in group])
        extracted = sum(r.extracted for r in group)
        print(
            f"{label} {kind}: mean exposure {mean:.1f} bits  extrapolated {mean_x:.1f}  extracted {extracted}/{len(group)}"
        )


def save_and_reload(private: PeftModel, path: str, folder: Path, heldout: torch.Tensor) -> None:
    """Save the DP adapter (a few hundred KB), load it back onto the pinned base model, and check it."""
    # excerpt: lora-save
    private.save_pretrained(path)  # only the adapter weights and config, after Opacus's hooks are removed
    base_model = AutoModelForCausalLM.from_pretrained(folder, dtype=torch.float32)  # the same pinned checkpoint
    reloaded = PeftModel.from_pretrained(base_model, path)
    # end excerpt
    print(f"DP adapter saved to {path}; reloaded held-out loss {heldout_loss(reloaded, heldout):.3f} nats/token")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fine-tune Pythia-14m with LoRA, with and without DP, and audit canaries."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epsilon", type=float, default=EPSILON, help="target epsilon of the DP run")
    parser.add_argument("--rank", type=int, default=RANK, help="LoRA rank r (lora_alpha = 2r)")
    parser.add_argument("--save-adapter", type=str, default=None, help="folder to save the DP adapter in")
    args = parser.parse_args()
    torch.set_num_threads(4)
    rng = np.random.default_rng(args.seed)

    folder = practitioner_datasets.pythia()
    tokenizer = load_tokenizer(folder)
    sequences = token_sequences(tokenizer, practitioner_datasets.shakespeare())
    heldout = sequences[-N_HELDOUT:]
    canaries = make_canaries(rng)
    train = insert_canaries(sequences[:N_TRAIN], canaries, tokenizer, rng)
    delta = delta_for(len(train))
    print(f"sequences of {SEQ_LEN} tokens: {len(sequences):,}  training n={len(train):,}  held out {len(heldout)}")
    print(f"delta={delta:.0e} (privacy unit: one training sequence)")
    print(f"canary sequences: {sum(c.copies for c in canaries)} ({PER_GROUP} canaries each at {COPIES} copies)")
    lengths = {len(tokenizer(" " + random_secret(rng))["input_ids"]) for _ in range(1000)}
    print(f"tokens per secret: {sorted(lengths)}")
    candidates = [random_secret(rng) for _ in range(N_CANDIDATES)]
    print(f"candidates: {N_CANDIDATES:,}  exposure cap {math.log2(N_CANDIDATES + 1):.2f} bits"
          f"  (full space {math.log2(SECRET_SPACE):.2f})")  # fmt: skip

    base = load_model(folder)
    full_count = trainable(base)
    base_loss = heldout_loss(base, heldout)
    print(f"full model parameters: {full_count:,}")

    torch.manual_seed(args.seed)
    nonprivate = add_lora(load_model(folder), args.rank)
    private = copy.deepcopy(nonprivate)  # both runs start from the same adapters
    train_nonprivate(nonprivate, train)
    nonprivate_loss = heldout_loss(nonprivate, heldout)

    run = train_private(private, train, args.epsilon, delta)
    private_loss = heldout_loss(private, heldout)
    print(f"sample rate q={run.sample_rate:.5f}  steps={run.steps}  noise multiplier={run.noise_multiplier:.3f}")
    print(f"epsilon (Opacus PRV)={run.epsilon:.2f}  epsilon (dp_accounting PLD)="
          f"{pld_epsilon(run.noise_multiplier, run.sample_rate, run.steps, delta):.2f}  at delta={delta:.0e}")  # fmt: skip

    for label, loss in (
        ("pretrained", base_loss),
        ("LoRA non-private", nonprivate_loss),
        (f"LoRA DP eps={args.epsilon:g}", private_loss),
    ):
        print(f"held-out loss {label}: {loss:.3f} nats/token  perplexity {math.exp(loss):.1f}")

    report("non-private", audit(nonprivate, tokenizer, canaries, candidates))
    report("DP", audit(private, tokenizer, canaries, candidates))
    if args.save_adapter is not None:
        save_and_reload(private, args.save_adapter, folder, heldout)


if __name__ == "__main__":
    main()
