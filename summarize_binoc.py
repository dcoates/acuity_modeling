"""Summarize the first rows of binoacuitydata.xlsx."""

import sys
from pathlib import Path

import pandas as pd

DATA_FILE = Path(__file__).with_name("binoacuitydata.xlsx")
N_ROWS = 5


def main(path: Path = DATA_FILE, n_rows: int = N_ROWS) -> None:
    sheets = pd.read_excel(path, sheet_name=None)
    for name, df in sheets.items():
        print(f"=== Sheet: {name} ({df.shape[0]} rows x {df.shape[1]} columns) ===")
        print("Columns:", list(df.columns))
        print()
        print(df.head(n_rows).to_string())
        print()


if __name__ == "__main__":
    file_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA_FILE
    main(file_arg)
