
"""Tests for reproducible calibration dataset preparation."""

import hashlib
import tempfile
import unittest
from pathlib import Path

from src.prepare_calibration import (
    SOURCE,
    EXPECTED_CALIBRATION_SHA256,
    prepare,
)


class CalibrationTests(unittest.TestCase):

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name) / "calibration.csv"

    def test_calibration_generation(self):
        """Generated calibration data matches the validated checksum."""
        prepare(SOURCE, self.output)

        self.assertTrue(self.output.is_file())
        actual_hash = hashlib.sha256(
            self.output.read_bytes()
        ).hexdigest()

        self.assertEqual(
            actual_hash,
            EXPECTED_CALIBRATION_SHA256,
        )

    def test_existing_identical_file(self):
        """An identical existing output is accepted without changes."""
        prepare(SOURCE, self.output)
        before = self.output.read_bytes()

        prepare(SOURCE, self.output)

        self.assertEqual(self.output.read_bytes(), before)

    def test_modified_output_rejected(self):
        """An unexpected existing output must never be overwritten."""
        prepare(SOURCE, self.output)

        self.output.write_bytes(b"unexpected content")

        with self.assertRaises(ValueError):
            prepare(SOURCE, self.output)

        self.assertEqual(
            self.output.read_bytes(),
            b"unexpected content",
        )

    def test_modified_source_rejected(self):
        """Source data with an incorrect checksum must be rejected."""
        invalid_source = Path(self.temp.name) / "invalid.csv"
        invalid_source.write_text(
            "date,rainfall_mm\n2001-01-01,1.0\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(
            ValueError, "Source CSV checksum mismatch"
        ):
            prepare(invalid_source, self.output)

        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()