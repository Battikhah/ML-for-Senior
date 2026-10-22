"""Measure lossless five-window PCA16 packet sizes across original train/validation WAVs."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
from pca_delta import encode,decode,best_packet,fit_budget
import window_campaign as frontend


def run(review,output):
 """Reuse the measured raw PCA16 profile; never fit anything on validation or open test data."""
 import librosa
 from scipy.signal import butter,filtfilt
 output.mkdir(parents=True,exist_ok=True);previous=Path(frontend.read(review/'config.json')['previous'])
 source=Path(frontend.read(previous/'audit/audio_source.json')['root']);settings=frontend.read(previous/'settings.json')
 contract=frontend.read(review/'cache/raw/contract.json');profile=np.load(review/'runs/server_raw/pca16_codec.npz')
 mean=np.array(contract['mean']);scale=np.array(contract['scale'])
 manifest={}
 for p in (Path(settings['original'])/'cache').glob('*/contract.json'):
  if frontend.read(p)['dataset_id']==frontend.read(previous/'audit/selection.json')['primary_dataset_id']:
   manifest={r['id']:r['sha256'] for r in frontend.read(p.parent/'manifest.json')};break
 if not manifest:raise ValueError('Cannot verify original WAV identities')
 ids={p:np.load(previous/f'cache/mfcc13/ids_{p}.npy') for p in ['train','val']};frontend.check_cohort(list(ids['train']),list(ids['val']))
 b,a=butter(4,[100,2500],btype='bandpass',fs=8000)
 methods={'raw':0,**{f'escape_{bits}bit':bits for bits in range(4,9)},**{f'rice_k{k}':16+k for k in range(8)}}
 rows=[];codes_all=[];mode_counts={};deltas=[];clock=time.monotonic();roundtrips=0
 for partition in ['train','val']:
  for identifier in ids[partition]:
   identifier=str(identifier);path=next((p/identifier for p in [source,source/'field/field/train'] if (p/identifier).is_file()),None)
   if path is None or hashlib.sha256(path.read_bytes()).hexdigest()!=manifest[identifier]:raise ValueError(f'WAV mismatch: {identifier}')
   signal,_=librosa.load(path,sr=8000,mono=True);last=max(0,len(signal)-80000)
   starts=np.unique(np.rint(np.linspace(0,last,5)).astype(int));features=[]
   for start in starts:
    clip=signal[start:start+80000];clip=np.pad(clip,(0,max(0,80000-len(clip))))
    features.append(frontend.mfcc(filtfilt(b,a,clip)))
   x=((np.stack(features)-mean)/scale).astype('float32').reshape(len(starts),-1)
   projected=(x-profile['mean'])@profile['components'].T
   codes=np.clip(np.rint((projected-profile['lo'])/profile['scale']),0,255).astype('uint8');codes_all.append(codes)
   delta=np.diff(codes.astype('int16'),axis=0).ravel();deltas.extend(delta.tolist())
   zig=np.where(delta>=0,2*delta,-2*delta-1)
   row=dict(id=identifier,partition=partition,duration_seconds=len(signal)/8000,window_count=len(codes),starts_samples=json.dumps(starts.tolist()),max_absolute_delta=int(max(abs(delta),default=0)),fixed_zigzag_bits_required=max(1,int(max(zig,default=0)).bit_length()))
   for name,mode in methods.items():
    packet=encode(codes,mode);np.testing.assert_array_equal(decode(packet),codes);roundtrips+=1;row[name+'_bytes']=len(packet)
   packet=best_packet(codes);np.testing.assert_array_equal(decode(packet),codes);roundtrips+=1
   fallback,indices=fit_budget(codes,64);np.testing.assert_array_equal(decode(fallback),codes[indices]);roundtrips+=1
   row.update(adaptive_bytes=len(packet),adaptive_mode=int(packet[0]),fits_64=len(packet)<=64,budget_bytes=len(fallback),budget_window_count=len(indices),budget_indices=json.dumps(indices.tolist()))
   mode_counts[str(packet[0])]=mode_counts.get(str(packet[0]),0)+1;rows.append(row)
   if len(rows)%200==0:print(f'DELTA_SAMPLES {len(rows)}/2070 {time.monotonic()-clock:.1f}s',flush=True)
 eligible=[r for r in rows if r['window_count']==5]
 def summarize(subset):
  """Exclude short single-window files from the five-window fit-rate denominator."""
  stats={}
  for method in [*methods,'adaptive']:
   sizes=np.array([r[method+'_bytes'] for r in subset])
   stats[method]=dict(min_bytes=int(sizes.min()),median_bytes=float(np.median(sizes)),p95_bytes=float(np.percentile(sizes,95)),max_bytes=int(sizes.max()),fit_count=int(sum(sizes<=64)),count=len(sizes),fit_fraction=float(np.mean(sizes<=64)))
  return stats
 all_delta=np.array(deltas);eligible_delta=np.concatenate([np.diff(c.astype('int16'),axis=0).ravel() for c in codes_all if len(c)==5])
 worst=max(eligible,key=lambda r:r['adaptive_bytes']);packing_bits={str(bits):int(sum(r['fixed_zigzag_bits_required']==bits for r in eligible)) for bits in range(1,10)}
 report=dict(recordings=len(rows),partitions={p:sum(r['partition']==p for r in rows) for p in ['train','val']},five_window_recordings=len(eligible),short_recordings=len(rows)-len(eligible),header_bytes=2,pca_values_per_window=16,pca_precision_bits=8,budget_bytes=64,test_set_evaluated=False,wav_hashes_verified=True,exact_roundtrip_checks=roundtrips,elapsed_seconds=time.monotonic()-clock,five_window_statistics=summarize(eligible),by_partition={p:summarize([r for r in eligible if r['partition']==p]) for p in ['train','val']},fixed_zigzag_bits_per_recording=packing_bits,delta_range=[int(all_delta.min()),int(all_delta.max())],delta_coverage={str(bits):float(np.mean((eligible_delta>=-(1<<(bits-1)))&(eligible_delta<1<<(bits-1)))) for bits in range(4,10)},optimal_mode_counts=mode_counts,budget_window_counts={str(n):sum(r['budget_window_count']==n for r in rows) for n in range(1,6)},worst_observed=worst,arbitrary_data_worst_case=dict(five_windows_raw_bytes=82,guaranteed_windows_with_two_byte_codec_header_at_64=3),classification_retested=False)
 frontend.save(output/'summary.json',report)
 with (output/'per_recording.csv').open('w',newline='',encoding='utf-8') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
 offsets=np.cumsum([0]+[len(c) for c in codes_all]);np.savez_compressed(output/'pca_codes.npz',codes=np.concatenate(codes_all),offsets=offsets,ids=np.array([r['id'] for r in rows]))
 lines=['# Lossless PCA16 delta encoding: measured payload sizes','',f"Tested {len(rows)} original recordings ({report['partitions']['train']} train, {report['partitions']['val']} validation). No final-test data. Five distinct windows are available in {len(eligible)} recordings; short recordings are reported separately.",'', 'All PCA coefficients retain their original 8-bit values. These are delta code widths, not reduced PCA precision. Every packet round-trip reconstructs the original byte vectors exactly.','', '| Method | Median bytes | 95th percentile | Worst observed bytes | Five-window packets ≤64 bytes |','|---|---:|---:|---:|---:|']
 for name in [*methods,'adaptive']:
  s=report['five_window_statistics'][name];lines.append(f"| {name} | {s['median_bytes']:g} | {s['p95_bytes']:g} | {s['max_bytes']} | {s['fit_count']}/{s['count']} ({s['fit_fraction']:.1%}) |")
 lines+=['',f"Observed signed delta range: {report['delta_range']}. Required fixed zigzag residual widths per five-window recording: {packing_bits}.",'',f"Budget fallback window counts across all recordings: {report['budget_window_counts']}.",'', 'Each byte count includes 2 codec-header bytes, the complete first vector, all escapes or unary codes, and final-byte padding. Node IDs, timestamp, checksums/security, and radio framing are additional.','', 'A six-bit residual stream with no escapes already takes 66 bytes for five windows once the two codec-header bytes are counted. Four raw windows require 66 bytes including that header. Therefore four windows are not universally guaranteed under an all-in 64-byte codec budget; three raw windows (50 bytes) are guaranteed.','', 'The adaptive codec chooses the shortest actual raw, escape-coded, or Rice-coded packet. Reducing the number of windows is a separate fallback and changes coverage; classifier performance for this five-window schedule/fallback has not been retested.','', 'Windows are ten seconds long and evenly spaced across each complete file. A short file has one padded window, not five duplicate copies that would artificially inflate the compression rate. The PCA16 profile and normalization come from the completed raw full-recording server campaign, fitted on training windows only.']
 (output/'results.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
 print('SUMMARY',json.dumps(report),flush=True)


if __name__=='__main__':
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--review',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();run(args.review,args.output)
