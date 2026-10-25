
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

files = [
    BASE_DIR / "mta_1706.csv",
    BASE_DIR / "mta_1708.csv",
    BASE_DIR / "mta_1710.csv",
    BASE_DIR / "mta_1712.csv"
]

for file in files:
    print("\n" + "=" * 80)
    print("FILE:", file.name)
    print("=" * 80)

    if not file.exists():
        print("File not found:", file)
        continue

    try:
        df = pd.read_csv(file, nrows=5)

        print("\nCOLUMNS:")
        print(df.columns.tolist())

        print("\nDATA TYPES:")
        print(df.dtypes)

        print("\nFIRST 5 ROWS:")
        print(df.to_string(index=False))

    except Exception as error:
        print("Error:", error)
