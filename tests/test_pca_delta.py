"""Exercise exact reconstruction, byte accounting, and bounded fallback before real data."""
import unittest
import numpy as np
try:
 from scripts import pca_delta as codec
except ImportError:
 codec=None

class DeltaCodecTests(unittest.TestCase):
 """Catch overflow at the byte extremes and differences hidden by aggregate statistics."""
 def test_extreme_and_random_values_roundtrip(self):
  """Deltas between 0 and 255 must never wrap before encoding."""
  self.assertIsNotNone(codec,'Delta codec is not implemented')
  rng=np.random.default_rng(45)
  for x in [np.tile(np.array([0,255,0,255,0],dtype='uint8')[:,None],(1,16)),rng.integers(0,256,(5,16),dtype='uint8')]:
   for mode in [0,4,5,6,7,8,*range(16,24)]:
    packet=codec.encode(x,mode)
    np.testing.assert_array_equal(codec.decode(packet),x)
 def test_constant_five_windows_have_no_exception_cost(self):
  """Five identical windows need one 16-byte base plus the packed zero deltas."""
  self.assertIsNotNone(codec,'Delta codec is not implemented')
  x=np.full((5,16),128,dtype='uint8')
  self.assertEqual(len(codec.encode(x,4)),50)
  self.assertEqual(len(codec.encode(x,5)),58)
  self.assertEqual(len(codec.encode(x,6)),66)
  self.assertEqual(len(codec.encode(x,16)),26)
 def test_adaptive_raw_fallback_is_bounded_and_packet_budget_is_real(self):
  """Random data can fall back to raw values, while budget fallback keeps valid windows."""
  self.assertIsNotNone(codec,'Delta codec is not implemented')
  x=np.random.default_rng(41).integers(0,256,(5,16),dtype='uint8')
  packet=codec.best_packet(x)
  self.assertLessEqual(len(packet),82)
  packet,indices=codec.fit_budget(x,64)
  self.assertLessEqual(len(packet),64)
  np.testing.assert_array_equal(codec.decode(packet),x[indices])
 def test_malformed_packets_and_out_of_range_input_are_rejected(self):
  """A corrupt stream must not silently create different PCA byte values."""
  self.assertIsNotNone(codec,'Delta codec is not implemented')
  with self.assertRaises(ValueError):codec.decode(bytes([4,5]))
  with self.assertRaises(ValueError):codec.encode(np.full((5,16),256),4)
  with self.assertRaises(ValueError):codec.decode(bytes([99,1])+bytes(16))

if __name__=='__main__':unittest.main()
