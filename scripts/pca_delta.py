"""Losslessly pack PCA byte vectors with cross-window deltas and explicit escape values.

Packets contain a mode byte, window-count byte, first 16-byte vector, and a
bit-packed delta stream. Mode 0 stores raw bytes; 4–8 use escape-coded zigzag
residuals; 16–23 use Rice parameters 0–7. No PCA precision is discarded.
"""
import numpy as np


class BitWriter:
 """Pack exact fields into bytes without charging one byte per small residual."""
 def __init__(self):
  """Keep only the pending partial byte and completed output bytes."""
  self.data=bytearray();self.pending=0;self.count=0
 def put(self,value,width):
  """Append an unsigned field, permitting zero-width Rice remainders."""
  if value<0 or value.bit_length()>width:raise ValueError('Bit field overflow')
  self.pending=(self.pending<<width)|int(value);self.count+=width
  while self.count>=8:
   self.count-=8;self.data.append((self.pending>>self.count)&255)
   self.pending&=(1<<self.count)-1
 def finish(self):
  """Charge the final partial byte, padding its unused low bits with zero."""
  if self.count:self.data.append(self.pending<<(8-self.count))
  return bytes(self.data)


class BitReader:
 """Read bounded fields and reject truncated escape/unary sequences."""
 def __init__(self,data):
  """Track bit position independently of the packet header and base vector."""
  self.data=data;self.position=0
 def get(self,width):
  """Return the next unsigned field or fail before reading outside the packet."""
  if self.position+width>len(self.data)*8:raise ValueError('Truncated bit stream')
  value=0
  for _ in range(width):
   value=(value<<1)|((self.data[self.position//8]>>(7-self.position%8))&1);self.position+=1
  return value


def encode(values,mode):
 """Preserve 8-bit PCA codes exactly, widening subtraction before predicting deltas."""
 x=np.asarray(values)
 if x.ndim!=2 or x.shape[1]!=16 or not 1<=len(x)<=5 or not np.issubdtype(x.dtype,np.integer) or np.any(x<0) or np.any(x>255):raise ValueError('Expected 1–5 by 16 integer byte codes')
 if mode not in [0,4,5,6,7,8,*range(16,24)]:raise ValueError('Unknown codec mode')
 x=x.astype('uint8');header=bytes([mode,len(x)])
 if mode==0:return header+x.tobytes()
 writer=BitWriter()
 for t in range(1,len(x)):
  for c in range(16):
   delta=int(x[t,c])-int(x[t-1,c]);zigzag=2*delta if delta>=0 else -2*delta-1
   if mode<16:
    escape=(1<<mode)-1
    writer.put(min(zigzag,escape),mode)
    if zigzag>=escape:writer.put(int(x[t,c]),8)
   else:
    k=mode-16;q=zigzag>>k
    writer.put((1<<q)-1,q);writer.put(0,1);writer.put(zigzag&((1<<k)-1),k)
 return header+x[0].tobytes()+writer.finish()


def decode(packet):
 """Reconstruct PCA codes exactly and reject malformed lengths or byte overflow."""
 if len(packet)<2:raise ValueError('Missing packet header')
 mode,count=packet[:2]
 if mode not in [0,4,5,6,7,8,*range(16,24)] or not 1<=count<=5:raise ValueError('Invalid codec header')
 if mode==0:
  if len(packet)!=2+16*count:raise ValueError('Wrong raw length')
  return np.frombuffer(packet[2:],dtype='uint8').reshape(count,16).copy()
 if len(packet)<18:raise ValueError('Missing base vector')
 result=np.empty((count,16),dtype='uint8');result[0]=np.frombuffer(packet[2:18],dtype='uint8');reader=BitReader(packet[18:])
 for t in range(1,count):
  for c in range(16):
   if mode<16:
    z=reader.get(mode)
    if z==(1<<mode)-1:result[t,c]=reader.get(8);continue
   else:
    k=mode-16;q=0
    while reader.get(1):
     q+=1
     if q>510:raise ValueError('Rice quotient outside byte delta range')
    z=(q<<k)|reader.get(k)
   delta=z//2 if z%2==0 else -(z//2)-1;value=int(result[t-1,c])+delta
   if not 0<=value<=255:raise ValueError('Decoded byte outside valid range')
   result[t,c]=value
 expected=(reader.position+7)//8
 if expected!=len(packet)-18:raise ValueError('Trailing bytes')
 if reader.position%8 and packet[-1]&((1<<(8-reader.position%8))-1):raise ValueError('Nonzero padding')
 return result


def best_packet(values):
 """Bound expansion with a raw fallback while testing inexpensive lossless codecs."""
 return min((encode(values,mode) for mode in [0,4,5,6,7,8,*range(16,24)]),key=lambda p:(len(p),p[0]))


def fit_budget(values,budget=64):
 """Reduce window count only when all lossless encodings exceed the packet budget."""
 for count in range(len(values),0,-1):
  indices=np.rint(np.linspace(0,len(values)-1,count)).astype(int)
  packet=best_packet(values[indices])
  if len(packet)<=budget:return packet,indices
 raise ValueError('Budget cannot hold even one PCA vector and codec header')
