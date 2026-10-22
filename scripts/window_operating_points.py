"""Compare full-recording candidates at matched recall, including half-volume diagnostics."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def build(output):
 """Choose thresholds on clean validation only and reuse them for paired gain tests."""
 from treevibes.evaluate import evaluate_probabilities
 rows=json.loads((output/'comparison.json').read_text());points=[]
 for row in rows:
  if row['id'].endswith('_half'):continue
  d=np.load(output/'predictions'/f"{row['id']}.npz");y=d['labels'];p=d['probabilities']
  for target in [.98,1.]:
   threshold=float(np.sort(p[y==1])[::-1][int(np.ceil(target*sum(y==1)))-1])
   for suffix in ['', '_half']:
    path=output/'predictions'/f"{row['id']}{suffix}.npz"
    if not path.exists():continue
    value=np.load(path);assert np.array_equal(value['ids'],d['ids']) and np.array_equal(value['labels'],y)
    m=evaluate_probabilities(y,value['probabilities'],threshold)
    points.append(dict(id=row['id']+suffix,target_clean_recall=target,threshold=threshold,recall=m['recall'],f1=m['f1'],accuracy=m['accuracy'],false_positives=m['confusion'][0][1],false_negatives=m['confusion'][1][0],test_set_evaluated=False))
 (output/'operating_points.json').write_text(json.dumps(points,indent=2),encoding='utf-8')
 with (output/'operating_points.csv').open('w',newline='',encoding='utf-8') as f:
  writer=csv.DictWriter(f,fieldnames=list(points[0]));writer.writeheader();writer.writerows(points)
 return points


if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path);args=parser.parse_args();print(json.dumps(build(args.output)))
