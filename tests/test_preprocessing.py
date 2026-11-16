"""Independent decimal verifier and unittest integration; no preprocessing imports."""
import csv, hashlib, json, sys, statistics
from pathlib import Path
from datetime import datetime, timedelta, date
from decimal import Decimal, getcontext
from collections import Counter, defaultdict
import os
import subprocess
import tempfile
import unittest


def main():
    getcontext().prec=50
    import argparse
    ap=argparse.ArgumentParser(description="Independent CSV/decimal rainfall verification; no preprocessing imports")
    ap.add_argument('--raw',type=Path,required=True)
    ap.add_argument('--cleaned',type=Path,required=True)
    ap.add_argument('--reference',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    raw=args.raw; clean=args.cleaned
    files={'raw':raw,'cleaned':clean,'reference':args.reference}
    before={n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in files.items()}
    assert before['raw']=='5cca3bfc3ab1dc36a86a9232752b5ac77afa45cd4838beb7d6d706d45fa14d48'
    assert before['cleaned']==before['reference']=='c01c4646a8e083dc36607db954f4d804bb26c878e7ac00f63af298014f78e53e'
    D=Decimal; z=D(0)
    buckets=defaultdict(list); signed=defaultdict(list); counts=Counter(); coords=set(); seen=set(); neg=[]; times=[]; missing=0; badsteps=0; offhour=0; hourlymax=(z,None); neg_hours=Counter(); smallpos=0
    with raw.open(newline='') as f:
     r=csv.DictReader(f); schema=r.fieldnames
     for row in r:
      missing+=sum(v is None or v=='' for v in row.values())
      t=datetime.strptime(row['valid_time'],'%Y-%m-%d %H:%M:%S')
      if times and t-times[-1]!=timedelta(hours=1): badsteps+=1
      times.append(t); seen.add(t); offhour+=bool(t.minute or t.second)
      p=D(row['tp']); lat=D(row['latitude']); lon=D(row['longitude'])
      assert all(v.is_finite() for v in (p,lat,lon))
      coords.add((str(lat),str(lon)))
      # Explicit interval membership: only a midnight endpoint goes to preceding date.
      day=t.date() if t.hour else t.date()-timedelta(days=1)
      signed[day].append(p*1000); buckets[day].append(max(z,p)*1000); counts[day]+=1
      if p<0: neg.append((t,p,day)); neg_hours[t.hour]+=1
      if 0<p<=D('1e-7'): smallpos+=1
      if date(2001,1,1)<=day<=date(2025,12,31) and p*1000>hourlymax[0]: hourlymax=(p*1000,str(t))
    days=sorted(d for d in buckets if counts[d]==24)
    expected=[date(2001,1,1)+timedelta(days=i) for i in range(9131)]
    assert days==expected
    series={d:sum(buckets[d],z) for d in days}; total=sum(series.values(),z)
    with clean.open(newline='') as f:
     r=csv.DictReader(f); clean_schema=r.fieldnames; cr=list(r)
    cd=[date.fromisoformat(r['date']) for r in cr]; assert cd==days
    cv={date.fromisoformat(r['date']):D(r['rainfall_mm']) for r in cr}
    diffs={d:abs(series[d]-cv[d]) for d in days}
    annual={str(y):sum((series[d] for d in days if d.year==y),z) for y in range(2001,2026)}
    clipday={d:series[d]-sum(signed[d],z) for d in days}
    out=dict(python=sys.version,hashes=before,schema=schema,clean_schema=clean_schema,rows=len(times),first=str(times[0]),last=str(times[-1]),duplicates=len(times)-len(seen),missing_cells=missing,bad_steps=badsteps,offhour=offhour,coordinates=sorted(coords),partial_days={str(d):counts[d] for d in counts if counts[d]!=24},used_hours=sum(counts[d] for d in days),days=len(days),year_counts=dict(Counter(d.year for d in days)),leap_days=[str(d) for d in days if d.month==2 and d.day==29],neg_count=len(neg),neg_min=min(p for t,p,d in neg),neg_max=max(p for t,p,d in neg),neg_beyond_tolerance=sum(p < -D('1e-7') for t,p,d in neg),negative_by_hour=dict(sorted(neg_hours.items())),negative_used=sum(d in series for t,p,d in neg),clip_total_mm=sum((-p*1000 for t,p,d in neg),z),negative_affected_days=sum(v>0 for v in clipday.values()),max_daily_clip_mm=max(clipday.values()),small_positive_count=smallpos,total_mm=total,signed_used_mm=sum((sum(signed[d],z) for d in days),z),signed_all_mm=sum((sum(v,z) for v in signed.values()),z),excluded_signed_mm=sum((sum(v,z) for d,v in signed.items() if d not in series),z),annual=annual,annual_mean_mm=total/25,previous_duration_normalized_mean_mm=total/(D(9131)/D('365.2425')),maximum_date=str(max(series,key=series.get)),maximum_mm=max(series.values()),minimum_date=str(min(series,key=series.get)),minimum_mm=min(series.values()),mean_mm=total/9131,median_mm=statistics.median(series.values()),std_mm=statistics.stdev(series.values()),max_abs_difference_mm=max(diffs.values()),max_difference_dates=[str(d) for d,v in diffs.items() if v==max(diffs.values())],mismatches_gt_1e9=[str(d) for d,v in diffs.items() if v>D('1e-9')],decimal_unequal_count=sum(v!=0 for v in diffs.values()),clean_total_mm=sum(cv.values(),z),hourly_max=hourlymax,flags={str(d):v for d,v in series.items() if v>150},first_daily=series[days[0]],last_daily=series[days[-1]])
    assert before=={n:hashlib.sha256(p.read_bytes()).hexdigest() for n,p in files.items()}
    out['byte_identical_to_reference']=clean.read_bytes()==args.reference.read_bytes()
    with args.reference.open(newline='') as f:
        ref=list(csv.DictReader(f))
    assert [r['date'] for r in ref]==[r['date'] for r in cr]
    out['max_cleaned_reference_difference_mm']=str(max(abs(D(a['rainfall_mm'])-D(b['rainfall_mm'])) for a,b in zip(cr,ref)))
    assert not out['mismatches_gt_1e9']
    assert out['byte_identical_to_reference']
    assert out['neg_beyond_tolerance']==0
    assert out['duplicates']==out['missing_cells']==out['bad_steps']==out['offhour']==0
    assert len(coords)==1
    assert out['negative_used']==1363
    args.output.write_text(json.dumps(out,default=str,indent=2)+'\n')
    print('PASS: 9131 daily records; max independent difference',out['max_abs_difference_mm'],'mm')
    print('PASS: cleaned CSV byte-identical to reference:',out['byte_identical_to_reference'])
    print('Mean annual rainfall:',out['annual_mean_mm'],'mm/year')
    print('SHA-256:',before)


@unittest.skipUnless(os.environ.get('RAINFALL_RAW_CSV'), 'Set RAINFALL_RAW_CSV to the verified external raw CSV')
class IndependentPreprocessingTests(unittest.TestCase):
    def test_independent_decimal_verification(self):
        root = Path(__file__).resolve().parents[1]
        cleaned = root / 'data/processed/colombo_daily_rainfall_2001_2025.csv'
        with tempfile.TemporaryDirectory() as temp:
            result = subprocess.run([
                sys.executable, str(Path(__file__).resolve()),
                '--raw', os.environ['RAINFALL_RAW_CSV'],
                '--cleaned', str(cleaned), '--reference', str(cleaned),
                '--output', str(Path(temp) / 'verification.json'),
            ], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            data = json.loads((Path(temp) / 'verification.json').read_text())
            self.assertEqual(data['days'], 9131)
            self.assertLessEqual(Decimal(data['max_abs_difference_mm']), Decimal('1e-9'))
            self.assertLess(abs(Decimal(data['annual_mean_mm']) - Decimal('2886.3380588')), Decimal('1e-7'))


if __name__ == '__main__':
    main()
