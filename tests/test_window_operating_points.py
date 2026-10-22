"""Verify gain checks do not silently retune their decision threshold."""
import json
import numpy as np
from scripts.window_operating_points import build


def test_half_volume_uses_clean_threshold_and_ordered_ids(tmp_path):
 """A changed score distribution must remain visible at the original operating point."""
 (tmp_path/'predictions').mkdir()
 (tmp_path/'comparison.json').write_text(json.dumps([{'id':'gain_float'},{'id':'gain_float_half'}]))
 y=np.array([0,0,1,1]);ids=np.array(['a','b','c','d'])
 np.savez(tmp_path/'predictions/gain_float.npz',labels=y,ids=ids,probabilities=[.1,.2,.9,.8])
 np.savez(tmp_path/'predictions/gain_float_half.npz',labels=y,ids=ids,probabilities=[.1,.2,.7,.6])
 points=build(tmp_path)
 for target in [.98,1.]:
  clean=next(r for r in points if r['id']=='gain_float' and r['target_clean_recall']==target)
  half=next(r for r in points if r['id']=='gain_float_half' and r['target_clean_recall']==target)
  assert clean['threshold']==half['threshold']==.8
  assert clean['recall']==1 and half['recall']==0
  assert half['false_negatives']==2
