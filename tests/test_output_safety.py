"""Failure-injection tests. Run: RAINFALL_RAW_CSV=/path/to/raw.csv python -m unittest discover -s tests -v"""
import os
import contextlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
RAW = Path(os.environ['RAINFALL_RAW_CSV']).resolve() if os.environ.get('RAINFALL_RAW_CSV') else None
SPEC = importlib.util.spec_from_file_location('preprocess', ROOT / 'src' / 'preprocess_rainfall.py')
p = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(p)

@unittest.skipUnless(RAW is not None, "Set RAINFALL_RAW_CSV to the verified external raw CSV")
class SafetyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)
        # Actual valid outputs from the successful corrected run.
        for name in (p.OUT_DAILY, p.OUT_REPORT):
            (self.out/name).write_bytes((ROOT / ('data/processed' if name == p.OUT_DAILY else 'docs') / name).read_bytes())
        self.before = {x.name:x.read_bytes() for x in self.out.iterdir()}

    def assert_preserved(self):
        self.assertEqual({x.name:x.read_bytes() for x in self.out.iterdir()}, self.before)

    def run_quiet(self):
        with contextlib.redirect_stdout(io.StringIO()):
            p.run(RAW, self.out)

    def test_early_hash_failure(self):
        with patch.object(p, 'EXPECTED_SHA256', '0'*64):
            with self.assertRaises(p.ValidationError): self.run_quiet()
        self.assert_preserved()

    def test_late_validation_failure(self):
        real = p.Checks.check
        def injected(checks, name, ok, detail=''):
            if name.startswith('Input file unchanged'): ok=False
            return real(checks,name,ok,detail)
        with patch.object(p.Checks, 'check', injected):
            with self.assertRaises(p.ValidationError): self.run_quiet()
        self.assert_preserved()

    def test_partial_report_write_failure(self):
        def fail(path,*args):
            path.write_text('incomplete candidate')
            raise OSError('injected report write failure')
        with patch.object(p, 'write_report', fail):
            with self.assertRaises(OSError): self.run_quiet()
        self.assert_preserved()

    def test_invalid_report_readback(self):
        with patch.object(p, 'write_report', lambda path,*args:path.write_text('invalid report')):
            with self.assertRaises(p.ValidationError): self.run_quiet()
        self.assert_preserved()

    def test_csv_readback_failure(self):
        real=p.pd.read_csv
        def fail(path,*args,**kwargs):
            if '.rainfall-stage-' in str(path): raise ValueError('injected candidate read failure')
            return real(path,*args,**kwargs)
        with patch.object(p.pd,'read_csv',fail):
            with self.assertRaises(ValueError): self.run_quiet()
        self.assert_preserved()

    def replace_failure(self, existing):
        if not existing:
            for x in self.out.iterdir(): x.unlink()
            self.before={}
        real=p.os.replace
        def fail(src,dst):
            if Path(src).name==p.OUT_REPORT: raise OSError('injected second replacement failure')
            return real(src,dst)
        with patch.object(p.os,'replace',fail):
            with self.assertRaises(OSError): self.run_quiet()
        self.assert_preserved()

    def test_second_replace_rollback_existing(self): self.replace_failure(True)
    def test_second_replace_rollback_new(self): self.replace_failure(False)

if __name__=='__main__': unittest.main(verbosity=2)
