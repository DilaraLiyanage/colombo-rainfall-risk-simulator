
"""Prepare the 2001–2020 rainfall calibration dataset.

Uses the previously validated daily CSV.
Does not use 2021–2025 rainfall observations for calibration.
Never overwrites an existing output with different contents.
"""

import argparse
import csv
import hashlib
import io
import math
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SOURCE = ROOT / "data/processed/colombo_daily_rainfall_2001_2025.csv"
OUTPUT = ROOT / "data/processed/colombo_calibration_2001_2020.csv"

EXPECTED_SOURCE_SHA256 = (
    "c01c4646a8e083dc36607db954f4d804bb26c878e7ac00f63af298014f78e53e"
)

EXPECTED_CALIBRATION_SHA256 = (
    "0fcaa7daca76d1cfbd7543e2b3a3a6cb3aa4509d2c8aab532bf5ddb0b65c89a0"
)

START = date(2001, 1, 1)
END = date(2020, 12, 31)
EXPECTED_DAYS = 7305


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def prepare(source: Path = SOURCE, output: Path = OUTPUT) -> None:
    # Check the source file before extracting calibration observations.
    if sha256(source.read_bytes()) != EXPECTED_SOURCE_SHA256:
        raise ValueError("Source CSV checksum mismatch")

    rows = []

    with source.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)

        if reader.fieldnames != ["date", "rainfall_mm"]:
            raise ValueError("Unexpected source CSV schema")

        for row in reader:
            day = date.fromisoformat(row["date"])

            # Stop before accessing holdout rainfall values.
            if day > END:
                break

            expected_date = START + timedelta(days=len(rows))

            if day != expected_date:
                raise ValueError("Calibration calendar mismatch")

            value = float(row["rainfall_mm"])

            if not math.isfinite(value) or value < 0:
                raise ValueError("Invalid calibration rainfall")

            rows.append(row)

    if len(rows) != EXPECTED_DAYS:
        raise ValueError("Incorrect number of calibration days")

    if date.fromisoformat(rows[-1]["date"]) != END:
        raise ValueError("Incorrect calibration end date")

    # Generate deterministic LF-terminated CSV bytes.
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=["date", "rainfall_mm"],
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)

    candidate = buffer.getvalue().encode("utf-8")
    checksum = sha256(candidate)

    if checksum != EXPECTED_CALIBRATION_SHA256:
        raise ValueError("Calibration output checksum mismatch")

    if output.exists():
        if output.read_bytes() != candidate:
            raise ValueError(
                "Existing calibration CSV differs; refusing to overwrite"
            )
        print("PASS: existing calibration dataset verified")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as file:
            file.write(candidate)
        print("PASS: calibration dataset created")

    print("Records:", len(rows))
    print("Period:", START, "to", END)
    print("SHA-256:", checksum)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    prepare(args.source, args.output)