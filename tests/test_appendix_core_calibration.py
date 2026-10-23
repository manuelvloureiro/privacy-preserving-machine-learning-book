"""code/appendix-derivations/apx_core_calibration.py runs, writes its figures, and reproduces the numbers in book-full/appendix-derivations/01-calibration.md."""

import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent / "code" / "appendix-derivations"


def run(script: Path, out: Path) -> str:
    return subprocess.run(
        [sys.executable, str(script), "--out", str(out)], capture_output=True, text=True, check=True
    ).stdout


def test_appendix_core_calibration_matches_page(tmp_path: Path) -> None:
    out = run(CODE / "apx_core_calibration.py", tmp_path)
    for expected in (
        # 1. Three outcomes.
        "ratios=[2.5, 1.0, 0.4]  largest=2.5000  pure eps=ln 2.5=0.9163  largest event ratio=2.5000",
        "excess sum=0.1000  brute force over 8 events=0.1000  Pr_P[L > eps]=0.5000",
        # 2. Sensitivities.
        "[2 count] add/remove l1=1.0000 l2=1.0000  replace l1=0.0000 l2=0.0000",
        "[2 sum] add/remove l1=3.0000 l2=3.0000  replace l1=3.0000 l2=3.0000",
        "[2 mean] add/remove l1=1.5000 l2=1.5000  replace l1=3.0000 l2=3.0000",
        "[2 histogram] add/remove l1=1.0000 l2=1.0000  replace l1=2.0000 l2=1.4142",
        "[2 yes counts] add/remove l1=3.0000 l2=1.7321  replace l1=3.0000 l2=1.7321",
        "[2 mean, n=3 records] replace l1=1.0000 (B/n=1.0000)",
        # 3. Laplace.
        "[3 Laplace y=1230] p_D=0.0338 p_D'=0.0205 ratio=1.6487 L=0.5000",
        "[3 Laplace y=1234.5] p_D=0.1947 p_D'=0.1947 ratio=1.0000 L=0.0000",
        "[3 Laplace y=1240] p_D=0.0124 p_D'=0.0205 ratio=0.6065 L=-0.5000",
        "loss ranges over [-0.5000, 0.5000]",
        "Pr_D=0.5000 Pr_D'=0.3033 ratio=1.6487",
        "b=6  largest loss over 100000 outputs=0.5000",
        "mean |Z|=2.0026 (exact 2)  95% bound b ln 20=5.9915  simulated 95th percentile=5.9938  Pr[|Z| > 5.9915]=0.0501",
        "eps=0.1: 29.9573",
        "eps=1: 2.9957",
        # 4. Randomized response.
        "t=0.6225  1-t=0.3775  ratio=1.6487  2t-1=0.2449",
        "E[ybar]=0.4510  estimate=0.3000  sd formula=0.0626  simulated mean=0.2996 sd=0.0629",
        # 5. Exponential mechanism.
        "weights D=[4.4817, 2.7183, 1.6487, 1.0]  Z(D)=9.8487  Z(D')=11.6121",
        "P_D=[0.4551, 0.276, 0.1674, 0.1015]  P_D'=[0.3859, 0.3859, 0.142, 0.0861]",
        "normalizer ratio=1.1790  bound e^(eps/2)=1.6487  largest |loss|=0.3353",
        "[5 EM vote gap>=2] exact=0.2689  bound k e^(-eps c/2)=1.4715",
        "gap=13.8240  bound e^-3=0.0498  exact=9.12e-04  Pr[best]=0.3935",
        # 6. Classic Gaussian.
        "c=4.8448 sigma=9.6896 t=4.7932  exact=1.61e-08  Pr[L>eps]=8.21e-07  Mills=8.54e-07  final=8.55e-07",
        "target/exact=622  target/final=11.7  tail/exact=51.0  1.96 sigma=18.9916",
        "[6 classic tail/exact eps=0.05] 505.9",
        "[6 classic tail/exact eps=1] 25.8",
        "needed t=0.5262  root c=1.0176  holds for delta<=0.7448  ln(1.25/1e-5)=11.7361",
        # 7. Exact Gaussian.
        "P(B)=0.3085  e^eps Q(B)=0.1816  delta=0.1269  integrated=0.1269  Monte Carlo=0.1270",
        "22 pairs, largest relative difference=3.1e-06",
        "largest difference below 1e-9: True",
        "exact sigma=7.0318  delta there=1.000e-05  classic sigma=9.6896  ratio=1.378  true delta at classic=1.61e-08",
        "[7 table eps=0.1] classic=48.4481  exact=30.7496  ratio=1.576",
        "[7 table eps=1] classic=4.8448  exact=3.7306  ratio=1.299  true delta at classic=4.11e-08",
        "[7 table eps=8] classic=0.6056  exact=0.6002  ratio=1.009  true delta at classic=7.97e-06",
        "[7 ratio eps=0.05] classic/exact=1.677",
        "true delta exceeds 1e-5 for eps > 8.42",
        "1.96 sigma=13.7824",
        # 8. Trade-offs.
        "mu=0.2500  alpha=-3.8750  beta=-4.1250  Phi(alpha)=5.3312e-05  Phi(beta)=1.8537e-05  e=2.7183",
        "e Phi(beta)=5.0388e-05  delta=2.9243e-06  first/delta=18.2",
        "bad-set integral=2.9243e-06  D2=2 sigma=8: delta=2.9243e-06  D2=0.5 sigma=2: delta=2.9243e-06",
        "[8 tail x=8] 1 - Phi(x)=6.6613e-16  erf form Phi(-x)=6.1062e-16  erfc form=6.2210e-16",
        "[8 tail x=9] 1 - Phi(x)=0.0000e+00  erf form Phi(-x)=0.0000e+00  erfc form=1.1286e-19",
        "delta=1e-09] sigma=5.4953  erfc=1.0000e-09  erf=1.0000e-09",
        "delta=1e-12] sigma=6.5578  erfc=1.0000e-12  erf=9.9997e-13",
        "delta=1e-15] sigma=7.4870  erfc=1.0000e-15  erf=9.6226e-16",
        "delta=1e-18] sigma=8.3220  erfc=1.0000e-18  erf=5.5511e-17",
        "first/delta=32.5",
        "first/delta=45.4",
        "first/delta=58.5",
        "first/delta=71.7",
        "[8 curve mu=0.25] sigma/D2=4  eps at 1e-5=0.9263  delta(0)=0.0995  delta(1)=2.92e-06  delta(2)=5.09e-17",
        "[8 curve mu=0.5] sigma/D2=2  eps at 1e-5=1.9931",
        "[8 curve mu=1] sigma/D2=1  eps at 1e-5=4.3772",
        "[8 curve mu=2] sigma/D2=0.5  eps at 1e-5=9.9973",
        "matches -e^eps Phi(beta) to below 1e-9: True",
        "[8 grid eps=0.1] 1e-03: 17.4044  1e-05: 30.7496  1e-07: 41.3295  1e-09: 50.2098",
        "[8 grid eps=0.5] 1e-03: 4.6101  1e-05: 7.0318  1e-07: 8.9957  1e-09: 10.6739",
        "[8 grid eps=1] 1e-03: 2.5747  1e-05: 3.7306  1e-07: 4.6787  1e-09: 5.4953",
        "[8 grid eps=2] 1e-03: 1.4452  1e-05: 1.9938  1e-07: 2.4491  1e-09: 2.8445",
        "[8 grid eps=4] 1e-03: 0.8231  1e-05: 1.0812  1e-07: 1.2978  1e-09: 1.4878",
        "z=4.2649  bend at z^2/2=9.09",
        "[8 sigma vs eps=0.1] exact=30.7496  eps*sigma=3.0750  tail rule=42.7658  tail/exact=1.391  z/eps=42.6489",
        "1/sqrt(2 eps)=2.2361  first/delta=111.2",
        "[8 sigma vs eps=0.5] exact=7.0318  eps*sigma=3.5159  tail rule=8.6454  tail/exact=1.229  z/eps=8.5298",
        "1/sqrt(2 eps)=1.0000",
        "[8 sigma vs eps=1] exact=3.7306  eps*sigma=3.7306  tail rule=4.3791  tail/exact=1.174  z/eps=4.2649",
        "1/sqrt(2 eps)=0.7071",
        "[8 sigma vs eps=2] exact=1.9938  eps*sigma=3.9876  tail rule=2.2439  tail/exact=1.125  z/eps=2.1324",
        "1/sqrt(2 eps)=0.5000",
        "[8 sigma vs eps=4] exact=1.0812  eps*sigma=4.3246  tail rule=1.1728  tail/exact=1.085  z/eps=1.0662",
        "1/sqrt(2 eps)=0.3536",
        "[8 sigma vs eps=8] exact=0.6002  eps*sigma=4.8018  tail rule=0.6320  tail/exact=1.053  z/eps=0.5331",
        "1/sqrt(2 eps)=0.2500  first/delta=3.6",
        "[8 sigma vs eps=32] exact=0.2049  eps*sigma=6.5579  tail rule=0.2083  tail/exact=1.016  z/eps=0.1333",
        "1/sqrt(2 eps)=0.1250",
        "[8 per decade eps=0.5] sigma 1e-5=7.0318  1e-6=8.0576  ratio=1.146  added=1.0258",
        "[8 per decade eps=1] sigma 1e-5=3.7306  1e-6=4.2247  ratio=1.132",
        "z(1e-6)=4.7534  z ratio=1.115",
        "[8 1e-5 to 1e-9 eps=0.5] ratio=1.518  halving eps 0.5 to 0.25: ratio=1.889",
        "[8 eps vs sigma=0.5] exact=9.9973  tail rule=10.5298",
        "[8 eps vs sigma=1] exact=4.3772  tail rule=4.7649",
        "[8 eps vs sigma=2] exact=1.9931  tail rule=2.2574",
        "[8 eps vs sigma=5] exact=0.7255  tail rule=0.8730",
        "[8 eps vs sigma=10] exact=0.3407  tail rule=0.4315",
        "[8 eps vs sigma=20] exact=0.1600  tail rule=0.2145",
        "sqrt(k) sigma=11.7973  split (eps/k, delta/k)=36.3047  split/composed=3.08  check delta(1; mu sqrt k)=1.000e-05",
        "sqrt(k) sigma=37.3063  split (eps/k, delta/k)=362.0183  split/composed=9.70",
        "ratio=1.367  exposure n=1e6 delta=1e-5=0.99995  eps*sigma at 1 over 0.1=1.213",
        "[8 n=1000] sigma at eps=1: 1e-5=3.7306  1/n=2.5747  1/n^1.1=2.7660  1/n^2=4.2247",
        "exposure: 1e-5=0.00995  1/n=0.632  1/n^1.1=0.394",
        "[8 n=10000] sigma at eps=1: 1e-5=3.7306  1/n=3.1857  1/n^1.1=3.4106  1/n^2=5.1003",
        "exposure: 1e-5=0.0952  1/n=0.632  1/n^1.1=0.328",
        "[8 n=100000] sigma at eps=1: 1e-5=3.7306  1/n=3.7306  1/n^1.1=3.9833  1/n^2=5.8678",
        "exposure: 1e-5=0.632  1/n=0.632  1/n^1.1=0.271",
        "[8 n=1000000] sigma at eps=1: 1e-5=3.7306  1/n=4.2247  1/n^1.1=4.5013  1/n^2=6.5578",
        "exposure: 1e-5=1  1/n=0.632  1/n^1.1=0.222",
    ):
        assert expected in out, expected
    assert {f.name for f in tmp_path.iterdir()} == {
        "cal_excess.svg",
        "cal_laplace_ratio.svg",
        "cal_privacy_loss.svg",
        "cal_exponential.svg",
        "cal_classic_chain.svg",
        "cal_bad_region.svg",
        "cal_sigma_vs_eps.svg",
        "cal_delta_curves.svg",
        "cal_tradeoff_map.svg",
    }
