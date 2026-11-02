"""Keep rows where all three acuity values are valid Snellen (20/xx) readings,
and add decimal and LogMAR columns for each eye.

Valid reading: 20/NN with an optional modifier (20/15, 20/15-, 20/15 -1,
20/25 +2, 20/20+, ...). The original text is kept in the acuity columns.

Conversions:
  LogMAR  = log10(NN / 20) + adjustment
            each letter = 0.02 LogMAR (5 letters per 0.1 line)
            '+n' (n letters better) -> -0.02 * n
            '-n' (n letters worse)  -> +0.02 * n
            a bare sign with no digit counts as 1 letter
  Decimal = 10 ** LogMAR   (20/20 = 1.00, worse = larger)
"""

import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
SRC = HERE / "binoacuitydata.xlsx"
DST = HERE / "binoacuitydata_valid.xlsx"
NONSTANDARD_STEREO = HERE / "nonstandard_stereo.xlsx"
NONSTANDARD_MONO = HERE / "nonstandard_mono.xlsx"
NONINT_AGE = HERE / "nonint_age.xlsx"
ACUITY_COLS = ["Last VA CC - Both", "Last VA CC - Left", "Last VA CC - Right"]
VALID = re.compile(r"^20/(\d+)\s*(?:([+-])\s*(\d*))?$")
LETTER_STEP = 0.02


def parse_logmar(value) -> float:
    """Return LogMAR for a valid 20/xx string, or NaN if it is not valid."""
    if not isinstance(value, str):
        return math.nan
    m = VALID.match(value.strip())
    if not m:
        return math.nan
    denom, sign, letters = m.groups()
    if int(denom) == 0:  # 20/0 and 20/00 are not valid readings
        return math.nan
    logmar = math.log10(int(denom) / 20)
    if sign:
        n = int(letters) if letters else 1
        logmar += -LETTER_STEP * n if sign == "+" else LETTER_STEP * n
    return logmar


def main(src: Path = SRC, dst: Path = DST) -> None:
    df = pd.read_excel(src)
    logmar = df[ACUITY_COLS].applymap(parse_logmar)
    keep = logmar.notna().all(axis=1)

    # Valid file also requires Age to be a whole number (e.g. "9-", "6." are dropped).
    age_ok = df["Age"].astype(str).str.strip().str.fullmatch(r"\d+")
    print(f"Dropped {(keep & ~age_ok).sum()} valid-acuity rows with non-integer Age")
    df[keep & ~age_ok].to_excel(NONINT_AGE, index=False)
    out = df[keep & age_ok].copy()
    out["Age"] = out["Age"].astype(str).str.strip().astype(int)
    for col in ACUITY_COLS:
        eye = col.replace("Last VA CC - ", "")
        out[f"{eye} LogMAR"] = logmar.loc[keep, col]
        out[f"{eye} Decimal"] = 10 ** logmar.loc[keep, col]

    # Better eye = lower LogMAR of Left/Right; "Both" is binocular, so excluded.
    out["logmar_better"] = out[["Left LogMAR", "Right LogMAR"]].min(axis=1)
    out["logmar_worse"] = out[["Left LogMAR", "Right LogMAR"]].max(axis=1)
    out["idiff"] = out["logmar_worse"] - out["logmar_better"]
    # Binocular advantage: better-eye logMAR minus binocular logMAR.
    # Positive = binocular is better (lower logMAR) than the better eye.
    out["advantage"] = out["logmar_better"] - out["Both LogMAR"]
    # "L" or "R" for the better eye; "=" when both eyes are equal.
    out["better_eye"] = np.select(
        [out["Left LogMAR"] < out["Right LogMAR"], out["Right LogMAR"] < out["Left LogMAR"]],
        ["L", "R"],
        default="=",
    )

    out.columns = [c.replace(" ", "_") for c in out.columns]
    out.to_excel(dst, index=False)
    print(f"Kept {len(out)} of {len(df)} rows -> {dst}")

    # Non-standard rows, original columns unchanged. "stereo" = both eyes
    # populated but uninterpretable; "mono" = at least one eye blank.
    bad = df[~keep]
    both_eyes = df.loc[~keep, ["Last VA CC - Left", "Last VA CC - Right"]].notna().all(axis=1)
    bad[both_eyes].to_excel(NONSTANDARD_STEREO, index=False)
    bad[~both_eyes].to_excel(NONSTANDARD_MONO, index=False)
    print(f"Wrote {both_eyes.sum()} stereo rows -> {NONSTANDARD_STEREO}")
    print(f"Wrote {(~both_eyes).sum()} mono rows -> {NONSTANDARD_MONO}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else SRC,
         Path(sys.argv[2]) if len(sys.argv) > 2 else DST)
