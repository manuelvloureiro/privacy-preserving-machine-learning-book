"""code/part02-attacks/membership_inference.py runs, writes its figures, and reproduces the numbers in book-full/part02-attacks/02-membership-inference.md."""

from pathlib import Path

from helpers import run

CODE = Path(__file__).resolve().parent.parent / "code" / "part02-attacks"


def test_membership_inference_matches_section(tmp_path: Path) -> None:
    out = run(CODE / "membership_inference.py", tmp_path)
    for expected in (
        # Setup
        (
            "pool=1000  d=100  outliers=50  lambda=0.001  target models=128  shadow models=64  "
            "member pairs=64000  non-member pairs=64000"
        ),
        # Loss attack and the gap
        "train error=0.1495  test error=0.2663  gap=0.1168",
        "'correct means member': TPR=0.8505  FPR=0.7337  advantage=0.1168",
        "loss clipped at B=2: train=0.4158  test=0.5433  gap/B=0.0638  simulated randomized attack advantage=0.0639",
        "Yeom threshold tau=mean train loss=0.4159: TPR=0.5936  FPR=0.4834  advantage=0.1102",
        # Attacks
        "loss threshold: AUC=0.577  best advantage=0.122  TPR@0.1%=0.0009  TPR@1%=0.0103",
        "shadow attack model: AUC=0.577  best advantage=0.121  TPR@0.1%=0.0008  TPR@1%=0.0104",
        "LiRA offline: AUC=0.632  best advantage=0.189  TPR@0.1%=0.0030  TPR@1%=0.0273",
        "LiRA online: AUC=0.629  best advantage=0.175  TPR@0.1%=0.0149  TPR@1%=0.0386",
        "LiRA online, global variance: AUC=0.623  best advantage=0.169  TPR@0.1%=0.0177  TPR@1%=0.0396",
        "952 true positives of 64000, 95% CI [0.0140, 0.0158]; 64 false positives of 64000, 95% CI [0.00077, 0.00128]",
        "loss attack TPR at FPR 0.001: 17.0x",
        "on targets TPR=0.0098  FPR=0.0095",
        "class -1 b=0.238  class +1 b=0.158",
        # Two kinds of record
        "global threshold on phi: t=6.075  TPR=0.0048",
        "same FPR on each type: TPR=0.0455",
        "log-threshold=1.497  FPR typical=5.90e-04  FPR atypical=0.0088  TPR=0.0503",
        "type weight 0.95: TPR at FPR 0.001 alone=0.0048",
        "type weight 0.05: TPR at FPR 0.001 alone=0.8185",
        # Same AUC
        "A (5% exposed): AUC=0.5250  best advantage=0.0500  TPR@0.1%=0.05095",
        "B (weak on all): mu=0.0887  AUC=0.5250  best advantage=0.0354  TPR@0.1%=0.00134",
        # Outliers
        "separation: inliers=0.461  outliers=1.539",
        "loss self-influence: inliers=0.0897  outliers=2.0999",
        "mean loss self-influence=0.1902  outliers' share of it=0.552",
        "TPR inliers=0.0020  outliers=0.2597; FPR inliers=0.00086  outliers=0.00375",
        "share of true positives that are outliers=0.873",
        "Gaussian-predicted TPR at FPR 0.001: inliers=0.0074  outliers=0.2145",
        # Number of shadow models
        "online: 4=0.0015  8=0.0009  16=0.0023  32=0.0086  64=0.0149",
        "online_global: 4=0.0092  8=0.0127  16=0.0151  32=0.0171  64=0.0177",
        "offline: 4=0.0018  8=0.0023  16=0.0029  32=0.0035  64=0.0030",
        "offline_global: 4=0.0098  8=0.0111  16=0.0112  32=0.0114  64=0.0118",
    ):
        assert expected in out
    assert {f.name for f in tmp_path.iterdir()} == {
        "mia_loss_distributions.svg",
        "mia_lira_example.svg",
        "mia_same_auc.svg",
        "mia_roc.svg",
        "mia_outliers.svg",
        "mia_shadow_count.svg",
    }
