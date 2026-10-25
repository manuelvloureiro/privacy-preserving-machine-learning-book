"""code/part02-attacks/linkage.py runs, writes its figures, and reproduces the numbers in book-full/part02-attacks/07-linkage-reidentification.md."""

from pathlib import Path

from helpers import run

CODE = Path(__file__).resolve().parent.parent / "code" / "part02-attacks"


def test_linkage_matches_section(tmp_path: Path) -> None:
    out = run(CODE / "linkage.py", tmp_path)
    for expected in (
        # Toolkit and bits
        "(1-p)^k=0.36790  lower e^(-pk/(1-p))=0.36788  upper e^(-pk)=0.36792",
        "log2 of world population 8e9 = 32.9 bits;  US 1990 = 27.9 bits",
        "ZIP=15.3  birth date=14.8  sex=1.0  total=31.1 bits",
        "cells=2.34e+09  n/M=0.106  uniform unique fraction=0.899",
        "world, 33 bits: n/M=0.931  unique fraction=0.394",
        "bits for 0.5 unique: world=33.4  above log2 n by 0.53",
        "bits for 0.9 unique: world=36.1  above log2 n by 3.25",
        "bits for 0.99 unique: world=39.5  above log2 n by 6.64",
        "class sizes [1, 1, 1, 2, 3]: success=0.625",
        # Hashed identifiers
        "md5('0')=cfcd208495d565ef66e7dff9f98764da",
        "six-digit identifiers: space=1000000 (19.9 bits)  hashes computed=1000000  inverted 5 of 5",
        "taxi: trips=1.73e+08 (27.4 bits)  licences 20.9 bits  medallions 24.4 bits",
        "ten-digit phone numbers: 33.2 bits",
        "one trip: pickup minute=525600 (19.0 bits)  pickup square=78000 (16.3 bits)  total=35.3 bits",
        # Synthetic census
        "cum bits=9.31 cells=1280  exact=0.0158  simulated=0.0160 (sd 0.0011)  collision bound=0.0000  "
        "entropy rule=0.0000  uniform cells=0.0004",
        "cum bits=12.90 cells=15360  exact=0.3328  simulated=0.3332 (sd 0.0049)  collision bound=0.0834  "
        "entropy rule=0.2692  uniform cells=0.5215",
        "cum bits=14.60 cells=61440  exact=0.5789  simulated=0.5797 (sd 0.0053)  collision bound=0.4237  "
        "entropy rule=0.6690  uniform cells=0.8498",
        "cum bits=16.48 cells=368640  exact=0.7939  simulated=0.7918 (sd 0.0048)  collision bound=0.7443  "
        "entropy rule=0.8967  uniform cells=0.9732",
        "cum bits=20.39 cells=7372800  exact=0.9747  simulated=0.9737 (sd 0.0027)  collision bound=0.9736  "
        "entropy rule=0.9928  uniform cells=0.9986",
        "cum bits=22.57 cells=58982400  exact=0.9925  simulated=0.9926 (sd 0.0012)  collision bound=0.9924  "
        "entropy rule=0.9984  uniform cells=0.9998",
        "log2 n = 13.29",
        "McDiarmid 95% half-width for the fraction unique: 0.0272",
        # Sparse ratings
        "ratings=203029  per user: median=30 mean=40.6  density=0.0102",
        "log2 n = 12.29",
        "bits in one aux item: median=5.83",
        "Netflix: log2 480189 = 18.9; a movie rated by 1% carries 6.64 bits",
        "m=2 wrong=0: uniform=0.428  ns=0.440  lr=0.402",
        "m=4 wrong=0: uniform=0.936  ns=0.941  lr=0.924",
        "m=8 wrong=0: uniform=1.000  ns=1.000  lr=1.000",
        "m=8 wrong=2: uniform=0.943  ns=0.989  lr=0.978",
        "phi=1.5: correct accepts=0.948  outsider accepts=0.263",
        "median eccentricity: insiders=7.32  outsiders=0.72",
        "phi=4.83  correct accepts=0.730  outsider accepts=0.010",
        # Sampling
        "population unique fraction: exact=0.8279  this population=0.8281",
        "pi=0.01: P(pop unique | sample unique) exact=0.8301  simulated=0.8334 (19951 sample uniques)",
        "pi=0.1: P(pop unique | sample unique) exact=0.8493",
        "pi=0.5: P(pop unique | sample unique) exact=0.9235",
        "lambda=0.1: kappa at pi=0.01: 0.910  pi=0.5: 0.952   xi at pi=0.01: 0.906  pi=0.5: 0.951",
        "lambda=0.5: kappa at pi=0.01: 0.669  pi=0.5: 0.800   xi at pi=0.01: 0.610  pi=0.5: 0.779",
        "lambda=2: kappa at pi=0.01: 0.336  pi=0.5: 0.500   xi at pi=0.01: 0.138  pi=0.5: 0.368",
        "993 sample uniques, mean predicted=0.817  actual=0.824",
        "bin: predicted=0.284  actual=0.303  count=119",
        "bin: predicted=0.980  actual=0.993  count=415",
    ):
        assert expected in out, expected
    assert {f.name for f in tmp_path.iterdir()} == {
        "link_bits.svg",
        "link_uniqueness.svg",
        "link_netflix.svg",
        "link_eccentricity_roc.svg",
        "link_sampling.svg",
    }
