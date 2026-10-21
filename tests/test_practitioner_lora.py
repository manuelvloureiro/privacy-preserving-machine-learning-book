"""code/part05-dp-ml/practitioner_lora.py runs and prints the numbers its pages quote.

The pages are Part 5's "Fine-tune a language model privately with LoRA" and step 4 (canaries and extraction) of
Part 2's risk-assessment workflow. The default run (seed 0) takes 2.5-6 minutes depending on how loaded the CPU
is (see the script's docstring); its pages quote it, so the test runs it in full. The --epsilon, --rank and
--save-adapter options are not tested here (a second full run would double the time); they were run by hand.
"""

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "code" / "part05-dp-ml" / "practitioner_lora.py"

EXPECTED = [
    "sequences of 32 tokens: 10,632  training n=1,024  held out 128",
    "delta=1e-04 (privacy unit: one training sequence)",
    "canary sequences: 255 (3 canaries each at (1, 4, 16, 64) copies)",
    "tokens per secret: [5]",
    "candidates: 2,000  exposure cap 10.97 bits  (full space 16.61)",
    "full model parameters: 14,067,712",
    "trainable parameters: LoRA 98,304  full fine-tuning 14,067,712",
    "validator errors: 0",
    "noise multiplier: 0.553",
    "sample rate q=0.03125  steps=96  noise multiplier=0.553",
    "epsilon (Opacus PRV)=7.99  epsilon (dp_accounting PLD)=7.98  at delta=1e-04",
    "held-out loss pretrained: 5.306 nats/token  perplexity 201.6",
    "held-out loss LoRA non-private: 5.071 nats/token  perplexity 159.3",
    "held-out loss LoRA DP eps=8: 5.044 nats/token  perplexity 155.1",
    "non-private  1 copies Alice   8 6 5 2 3: exposure  1.3 bits  extrapolated  1.3  greedy no",
    "non-private  1 copies Brian   0 0 0 1 8: exposure  2.0 bits  extrapolated  2.1  greedy no",
    "non-private  1 copies Chloe   6 9 5 6 9: exposure  2.6 bits  extrapolated  2.7  greedy no",
    "non-private  4 copies Daniel  7 6 5 5 9: exposure  6.9 bits  extrapolated  6.4  greedy no",
    "non-private  4 copies Emily   2 8 6 0 3: exposure  2.6 bits  extrapolated  2.6  greedy no",
    "non-private  4 copies Felix   8 5 0 7 7: exposure  8.2 bits  extrapolated  8.2  greedy no",
    "non-private 16 copies Grace   8 1 0 8 0: exposure 10.0 bits  extrapolated 13.5  greedy no",
    "non-private 16 copies Hugo    5 0 2 4 4: exposure 10.0 bits  extrapolated 13.4  greedy no",
    "non-private 16 copies Irene   4 0 0 1 0: exposure 10.0 bits  extrapolated 11.9  greedy no",
    "non-private 64 copies Jacob   6 5 6 2 6: exposure 11.0 bits  extrapolated 16.6  greedy extracted",
    "non-private 64 copies Karen   7 3 4 9 8: exposure 11.0 bits  extrapolated 16.4  greedy extracted",
    "non-private 64 copies Liam    9 3 6 9 6: exposure 11.0 bits  extrapolated 16.0  greedy extracted",
    "non-private   control Megan   8 6 7 3 8: exposure  1.3 bits  extrapolated  1.3  greedy no",
    "non-private   control Nathan  1 5 7 8 5: exposure  0.1 bits  extrapolated  0.1  greedy no",
    "non-private   control Olivia  3 3 4 4 7: exposure  2.9 bits  extrapolated  3.0  greedy no",
    "non-private 1 copies: mean exposure 2.0 bits  extrapolated 2.0  extracted 0/3",
    "non-private 4 copies: mean exposure 5.9 bits  extrapolated 5.7  extracted 0/3",
    "non-private 16 copies: mean exposure 10.0 bits  extrapolated 13.0  extracted 0/3",
    "non-private 64 copies: mean exposure 11.0 bits  extrapolated 16.3  extracted 3/3",
    "non-private controls: mean exposure 1.4 bits  extrapolated 1.5  extracted 0/3",
    "DP  1 copies Alice   8 6 5 2 3: exposure  0.7 bits  extrapolated  0.7  greedy no",
    "DP  1 copies Brian   0 0 0 1 8: exposure  0.6 bits  extrapolated  0.6  greedy no",
    "DP  1 copies Chloe   6 9 5 6 9: exposure  1.3 bits  extrapolated  1.3  greedy no",
    "DP  4 copies Daniel  7 6 5 5 9: exposure  1.3 bits  extrapolated  1.4  greedy no",
    "DP  4 copies Emily   2 8 6 0 3: exposure  0.1 bits  extrapolated  0.1  greedy no",
    "DP  4 copies Felix   8 5 0 7 7: exposure  0.3 bits  extrapolated  0.3  greedy no",
    "DP 16 copies Grace   8 1 0 8 0: exposure  2.6 bits  extrapolated  2.6  greedy no",
    "DP 16 copies Hugo    5 0 2 4 4: exposure  3.2 bits  extrapolated  3.1  greedy no",
    "DP 16 copies Irene   4 0 0 1 0: exposure  6.8 bits  extrapolated  7.3  greedy no",
    "DP 64 copies Jacob   6 5 6 2 6: exposure  4.8 bits  extrapolated  4.7  greedy no",
    "DP 64 copies Karen   7 3 4 9 8: exposure  3.2 bits  extrapolated  3.1  greedy no",
    "DP 64 copies Liam    9 3 6 9 6: exposure  4.1 bits  extrapolated  3.9  greedy no",
    "DP   control Megan   8 6 7 3 8: exposure  1.1 bits  extrapolated  1.1  greedy no",
    "DP   control Nathan  1 5 7 8 5: exposure  1.9 bits  extrapolated  1.9  greedy no",
    "DP   control Olivia  3 3 4 4 7: exposure  3.7 bits  extrapolated  3.6  greedy no",
    "DP 1 copies: mean exposure 0.9 bits  extrapolated 0.9  extracted 0/3",
    "DP 4 copies: mean exposure 0.6 bits  extrapolated 0.6  extracted 0/3",
    "DP 16 copies: mean exposure 4.2 bits  extrapolated 4.3  extracted 0/3",
    "DP 64 copies: mean exposure 4.0 bits  extrapolated 3.9  extracted 0/3",
    "DP controls: mean exposure 2.2 bits  extrapolated 2.2  extracted 0/3",
]


@pytest.mark.libs
def test_practitioner_lora_prints_quoted_numbers() -> None:
    out = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True, check=True).stdout
    lines = out.splitlines()
    for expected in EXPECTED:
        assert expected in lines, f"missing line: {expected!r}"
