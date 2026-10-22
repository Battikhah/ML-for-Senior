"""Guard full recording coverage, gain invariance, and recording-level data separation."""
import unittest
import numpy as np
try:
 from scripts import window_campaign as campaign
except ImportError:
 campaign=None

class WindowCampaignTests(unittest.TestCase):
 """Catch dropped tails and leakage before running the Windows GPU campaign."""
 def test_tail_is_covered_without_duplicate_windows(self):
  """A recording with a fractional final window must retain that final audio."""
  self.assertIsNotNone(campaign,'The full-recording campaign is not implemented')
  self.assertEqual(campaign.window_starts(163680),[0,40000,80000,83680])
  self.assertEqual(campaign.window_starts(80000),[0])
  self.assertEqual(campaign.window_starts(100),[0])
  self.assertEqual(campaign.window_starts(160000),[0,40000,80000])
 def test_gain_normalization_removes_volume_scaling_without_nan(self):
  """Changing microphone gain should not change normalized waveform shape."""
  self.assertIsNotNone(campaign,'The full-recording campaign is not implemented')
  x=np.sin(np.arange(80000)*.19).astype('float32')*.01
  np.testing.assert_allclose(campaign.gain_normalize(x),campaign.gain_normalize(x*.5),atol=1e-7)
  self.assertTrue(np.isfinite(campaign.gain_normalize(np.zeros(80000))).all())
 def test_aggregation_excludes_padding_and_respects_recording_ids(self):
  """Padding must never turn an all-negative recording into a positive one."""
  self.assertIsNotNone(campaign,'The full-recording campaign is not implemented')
  np.testing.assert_allclose(campaign.aggregate(np.array([-3.,-1.,2.]),np.array([0,0,1]),2),[-1.,2.])
  with self.assertRaises(ValueError):campaign.aggregate(np.array([1.]),np.array([0]),2)
 def test_disjoint_recordings_are_required(self):
  """All crops of a recording must remain in its original dataset partition."""
  self.assertIsNotNone(campaign,'The full-recording campaign is not implemented')
  campaign.check_cohort(['a','b'],['c'])
  with self.assertRaises(ValueError):campaign.check_cohort(['a'],['a'])

if __name__=='__main__':unittest.main()
