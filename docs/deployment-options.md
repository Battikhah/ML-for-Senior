# ESP32 and server classification: measured feasibility comparison

The evidence is from the original public experiment ZIP, not the final Windows campaign. Forty completed runs passed artifact checks. No model was retrained and no final-test labels were loaded or scored. Models and thresholds were selected using validation; these are development results. The snapshot uses file/duplicate grouping, not tree/session holdout. Field generalization remains unverified.

## What was measured

The two best saved remote Keras models were evaluated at their original fixed thresholds after eight feature representations: float32, coefficient-wise 8/4/2-bit coding, and training-only PCA at 16/24/32/64 components followed by 8-bit coding. Packed coefficient codes were serialized and decoded before inference. PCA profiles require a versioned, preinstalled calibration contract on both sides.

Three local model families were converted to full INT8 using training-only calibration, evaluated on validation, and bundled with the existing ESP32 fixture benchmark. All three exports succeeded. Their model sizes are not whole-firmware sizes or RAM estimates.

| Design | Feature bytes per measurement | Validation F1 | Infested recall | False-positive rate |
|---|---:|---:|---:|---:|
| Server temporal_w64, float32 | 1,560 | 0.9304 | 0.9817 | 0.0556 |
| Same server model, 8-bit features | 390 | 0.9304 | 0.9817 | 0.0556 |
| Same server model, 4-bit features | 195 | 0.8525 | 0.9541 | 0.1230 |
| Same server model, PCA16 + 8-bit | 16 | 0.9035 | 0.9450 | 0.0635 |
| Local tiny_cnn_w12, INT8 model | Local results can be summarized daily | 0.8987 | 0.9358 | See export metadata |

Local exported model sizes: tiny_mlp_w12 12,304 bytes; tiny_cnn_w12 19,072 bytes; tiny_depthwise_w12 10,232 bytes. These are candidate model files, not proof of ESP32 deployment. Inference arena, firmware flash, sampling buffers, input-feature parity, latency and energy require board measurements.

## What a daily report means

Assume 12 measurements per day solely for this comparison. The actual schedule is configurable.

- Local classification: run inference for every measurement, preserve counts/quality/max score locally, and send a proposed 24-byte summary daily. The summary packet schema still needs firmware implementation and delivery testing.
- Server classification with full MFCC: retain all twelve 390-byte int8 matrices, totaling 4,680 feature bytes/day, then send them in a daily batch or individually.
- Server classification with PCA16: the node performs feature extraction and a linear encoder, rather than classification. Twelve 16-byte vectors total 192 feature bytes/day. The server reconstructs the normalized matrix and runs its classifier. This is real measured compression, but its recall loss must be acceptable.

A daily report may consist of multiple radio packets. Under an illustrative 200-byte raw-radio limit and 16-byte metadata per fragment, the daily batches require 5,096 transmitted payload bytes/26 packets for full int8 MFCC, versus 224 bytes/2 packets for PCA16. The estimated SF7 airtimes are 8.1065 seconds and 0.3794 seconds respectively; a 24-byte local summary is 0.0617 seconds. Calculations use BW125, CR4/5, explicit header, CRC and eight preamble symbols. They exclude authentication, ACKs, retries, synchronization and LoRaWAN overhead. They do not establish regulatory or fleet capacity. The 16-byte fragment metadata is a budgeting assumption, not a complete implemented secure transport protocol.

## C encoder evidence

The PCA16 profile was exported as portable C. Eight held-out feature fixtures matched the Python codes exactly on the host compiler (maximum code difference zero). The profile constants occupy 26,752 bytes and the projection requires 6,240 multiplications per measurement, excluding MFCC extraction and normalization. This supports feasibility investigation, not an ESP32 latency/RAM/energy claim. The frozen Python profile is needed for server reconstruction.

## Recommended next decision

Keep both options until board results exist. Local classification has the smallest routine radio load; PCA16 remote classification provides richer evidence with much lower payload than full MFCC and similar validation F1 to these local candidates. Full int8 MFCC remote classification preserves the highest measured snapshot performance. Four-bit MFCC caused a substantial false-positive increase and should not be selected solely because it fits a packet.

Run `TreeVibes_ESP32_Server_Comparison.ipynb` after the current Windows campaign finishes to evaluate its newer distilled students. It uses a separate output folder and does not edit or restart the campaign. Then benchmark full INT8 candidates and the encoder on the ESP32 using matching firmware/extractor contracts. A classifier's recall is only one factor in full-system recall; acquisition, gating, reporting and delivery must also be measured.

## Deliverables

- `artifacts/deployment_comparison_verified/deployment_comparison.xlsx`: model and transport tables.
- `artifacts/deployment_comparison_verified/comparison.json`: detailed metrics and limitations.
- `artifacts/deployment_comparison_verified/local_exports`: full INT8 models, resolver/normalization fixtures and generated firmware files.
- `artifacts/deployment_comparison_verified/pca16_c_encoder`: C encoder, eight fixtures and host verification.
- `scripts/compare_deployment.py`: reproducible standalone analysis.

Airtime formula: https://www.hoperf.com/uploads/RFM95W-V2.0_1695351437.pdf, section 4.1.1.7.
