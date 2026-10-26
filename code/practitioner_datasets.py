"""Load the public datasets that the practitioner edition's workflow scripts run on.

    uv run python code/practitioner_datasets.py fetch    # fill the cache in data/ (what `make datasets` runs)

Everything is downloaded once into data/ at the repository's root, which git ignores, and read from there
afterwards, so the scripts run offline once the cache is filled. Nothing here is committed.

Datasets:

- Adult (UCI Machine Learning Repository, dataset 2, "Census Income"; Becker & Kohavi, 1996; licence CC BY 4.0).
  https://archive.ics.uci.edu/static/public/2/adult.zip. The files adult.data (32,561 rows) and adult.test (16,281
  rows) together are the 48,842 rows that OpenML publishes as data_id 1590. Each row is one person from the 1994 US
  Current Population Survey; the label is whether their income exceeds $50,000.
- Online Retail (UCI Machine Learning Repository, dataset 352; Chen, 2015; licence CC BY 4.0).
  https://archive.ics.uci.edu/static/public/352/online+retail.zip. 541,909 invoice lines of a UK online shop,
  December 2010 to December 2011. Lines with a customer id belong to 4,372 customers, most of whom have many lines,
  which is what contribution bounding needs. Converted once from .xlsx to Parquet.
- Tiny Shakespeare (Karpathy's char-rnn data, public-domain text, about 1.1 MB).
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt.
- Pythia-14m (EleutherAI, Hugging Face model EleutherAI/pythia-14m, licence Apache-2.0), a 14-million-parameter
  causal language model, small enough to fine-tune on a CPU.
- France LFS (Eurostat, EU Labour Force Survey public microdata, France, 2005-2013, as preprocessed by the OpenDP
  project for its tutorials: https://github.com/opendp/dp-test-datasets/blob/main/data/eurostat/README.ipynb).
  https://raw.githubusercontent.com/opendp/dp-test-datasets/refs/heads/main/data/eurostat/V2_FR_LFS.csv.zip. The
  3,811,932 quarterly responses of 1,682,306 people, with a synthetic person identifier PIDENT and the columns YEAR,
  QUARTER, SEX, AGE, HWUSUAL, HWACTUAL and ILOSTAT. The same file that `dp.examples.get_france_lfs_path()` downloads,
  kept here so that every download lands in data/. Converted once from CSV to Parquet, which carries its schema.
  Eurostat's public microdata are free to reuse for non-commercial research and teaching (Eurostat's licence for the
  public-use files). Used by the OpenDP book.

The datasets bundled with scikit-learn (breast cancer, digits, diabetes) need no download and are loaded directly
with sklearn.datasets.
"""

import sys
import urllib.request
import zipfile
from pathlib import Path

import polars as pl
from huggingface_hub import snapshot_download  # pyright: ignore[reportUnknownVariableType]

DATA = Path(__file__).resolve().parent.parent / "data"

ADULT_URL = "https://archive.ics.uci.edu/static/public/2/adult.zip"
RETAIL_URL = "https://archive.ics.uci.edu/static/public/352/online+retail.zip"
SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
FRANCE_LFS_URL = (
    "https://raw.githubusercontent.com/opendp/dp-test-datasets/refs/heads/main/data/eurostat/V2_FR_LFS.csv.zip"
)
PYTHIA = "EleutherAI/pythia-14m"
PYTHIA_REVISION = "cf967c0a9a04383db6f7b1108d86b2962634b4ac"

ADULT_COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]
ADULT_NUMERIC = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]


def _download(url: str, target: Path) -> Path:
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(target.suffix + ".part")
        with urllib.request.urlopen(url, timeout=120) as response, tmp.open("wb") as out:
            out.write(response.read())
        tmp.rename(target)
    return target


def adult() -> pl.DataFrame:
    """The 48,842 rows of Adult, training file first. Missing values ('?') are null; income is '<=50K' or '>50K'."""
    archive = _download(ADULT_URL, DATA / "uci" / "adult.zip")
    frames: list[pl.DataFrame] = []
    with zipfile.ZipFile(archive) as z:
        for name, skip in (("adult.data", 0), ("adult.test", 1)):  # adult.test starts with a comment line
            frame = pl.read_csv(
                z.read(name),
                has_header=False,
                new_columns=ADULT_COLUMNS,
                skip_rows=skip,
                null_values="?",
                schema_overrides={c: pl.Int64 for c in ADULT_NUMERIC},
            )
            frames.append(frame.filter(pl.col("age").is_not_null()))
    df = pl.concat(frames)
    strings = [c for c in ADULT_COLUMNS if c not in ADULT_NUMERIC]
    df = df.with_columns(pl.col(strings).str.strip_chars())  # the source puts a space after each comma
    df = df.with_columns(pl.col(strings).replace("?", None))
    return df.with_columns(pl.col("income").str.strip_chars_end("."))  # adult.test writes '>50K.'


def online_retail() -> pl.DataFrame:
    """The 541,909 invoice lines of Online Retail, with columns as in the source and CustomerID as an integer or null."""
    parquet = DATA / "uci" / "online_retail.parquet"
    if not parquet.exists():
        archive = _download(RETAIL_URL, DATA / "uci" / "online+retail.zip")
        with zipfile.ZipFile(archive) as z:
            xlsx = z.read("Online Retail.xlsx")
        frame = pl.read_excel(xlsx, engine="calamine")
        frame = frame.with_columns(pl.col("CustomerID").cast(pl.Int64), pl.col("InvoiceNo").cast(pl.String))
        frame.write_parquet(parquet)
    return pl.read_parquet(parquet)


def france_lfs() -> Path:
    """The path of the France LFS sample as Parquet (3,811,932 rows), for `pl.scan_parquet`.

    The OpenDP book scans it lazily, as the OpenDP documentation scans its CSV copy, so this returns the path
    rather than a frame. Integers stay integers; AGE, HWUSUAL and HWACTUAL are floats with nulls, as in the source.
    """
    parquet = DATA / "eurostat" / "france_lfs.parquet"
    if not parquet.exists():
        archive = _download(FRANCE_LFS_URL, DATA / "eurostat" / "V2_FR_LFS.csv.zip")
        with zipfile.ZipFile(archive) as z:
            csv = z.read("V2_FR_LFS.csv")
        pl.read_csv(csv, infer_schema_length=10_000).write_parquet(parquet)
    return parquet


def shakespeare() -> str:
    """The Tiny Shakespeare text, about 1.1 million characters."""
    return _download(SHAKESPEARE_URL, DATA / "shakespeare.txt").read_text()


def pythia() -> Path:
    """The local folder holding the Pythia-14m checkpoint and tokenizer, pinned to one revision."""
    folder = snapshot_download(
        PYTHIA,
        revision=PYTHIA_REVISION,
        cache_dir=str(DATA / "hf"),
        allow_patterns=["*.json", "*.safetensors", "*.txt"],
    )
    return Path(folder)


def fetch() -> None:
    """Fill the cache with every dataset the practitioner scripts use."""
    print(f"Adult: {len(adult()):,} rows")
    print(f"Online Retail: {online_retail().height:,} rows")
    print(f"Tiny Shakespeare: {len(shakespeare()):,} characters")
    print(f"France LFS: {pl.scan_parquet(france_lfs()).select(pl.len()).collect().item():,} rows")
    print(f"Pythia-14m: {pythia()}")


if __name__ == "__main__":
    if sys.argv[1:] != ["fetch"]:
        sys.exit(__doc__)
    fetch()
