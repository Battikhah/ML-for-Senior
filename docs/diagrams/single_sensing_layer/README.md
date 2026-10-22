# One sensing layer: local and server classification flows

These are proposed operational diagrams for **one piezo sensor, one ESP32-WROOM32, and one RFM95W link**. The three columns are stages inside the same sensing layer, not three sensing layers. The gateway/server box is an external destination only; no cluster-head, farm, or multi-node scheduling flow is drawn. Matching letter circles connect stages: A starts per-window processing; C enters reporting.

The visual reference is Figure 4.5, “Sensing Layer operational flowchart,” printed page 43 / PDF page 54 of the supplied `main.pdf`. The reference report remains unchanged. Its legacy 15×13 feature shape, fixed scheduling/airtime numbers, and energy-gate-as-insect-detection interpretation are not carried into these diagrams.

## Files

- `local_classification_single_layer.svg`: editable vector figure for local INT8 classification.
- `server_classification_single_layer.svg`: editable vector figure for PCA16 server classification.
- `single_sensing_layer_flows.drawio`: two native diagrams.net pages with editable boxes, diamonds, arrows and text.
- `../../../output/pdf/single_sensing_layer/local_classification_single_layer.pdf`: local route, one vector page.
- `../../../output/pdf/single_sensing_layer/server_classification_single_layer.pdf`: server route, one vector page.
- `../../../output/pdf/single_sensing_layer/single_sensing_layer_flows.pdf`: both routes, two vector pages.
- `flow_data.json`: source node/edge coordinates and labels.
- `validation.json`: structural checks and output hashes.

## What the local route does

The node captures a complete recording, processes all scheduled windows, normalizes MFCCs using training-derived constants, runs the INT8 residual classifier and takes the maximum window RPW score. It applies the saved operating threshold, persists daily counts and quality statistics, and prepares the proposed 24-byte summary. The 55,864-byte figure describes the model file, not the full firmware or runtime RAM.

The gateway relays results. The server stores and displays decisions that the sensing node has already made. A server acoustic classifier is not required for this route.

## What the server route does

The sensing node extracts the same 30×13 MFCCs, applies the frozen normalization and PCA mean/projection, and quantizes the sixteen PCA coordinates to sixteen bytes per window. It does not generate RPW/healthy decisions. Delta/Rice packing preserves those byte values exactly; a raw fallback prevents uncontrolled compression expansion.

The server checks the profile/version, reassembles fragments, decodes the lossless stream, dequantizes, reverses PCA, and classifies the reconstructed **already normalized** MFCCs. It must not apply coefficient normalization a second time. It combines window scores using the matched maximum-score policy and saved threshold.

The locked evaluated classifier uses 10-second windows with 5-second stride and a final tail window. A 20.5-second example has starts at 0, 5, 10, and 10.5 seconds. Window counts depend on recording length. Five-window adaptive packing has been measured separately; its classifier/selection policy has not been revalidated. The diagram preserves the evaluated windows and fragments oversized records rather than silently dropping a window to force a fit.

## Reporting and transport assumptions

- Every node schedules at least one daily report, including a status/heartbeat when no capture is valid. Sample cadence is configurable; the diagram does not require a specific number of measurements per day.
- A server-side daily report may need multiple packets to carry all retained feature records. A 64-byte feature-block target is not a full on-air packet size. Node/time/version identifiers, fragment headers, integrity/security fields, ACKs/retries and radio framing consume additional bytes.
- The local 24-byte daily body remains a proposed schema. It is not an implemented complete secure transport frame.
- Sequence-based acknowledgement, bounded retries, persistent queues and deduplication are proposed protocol behavior. ACK should mean the gateway has durably accepted responsibility for the packet, not merely that the node called a transmit function. Gateway backhaul buffering remains external to this single-layer flow.
- Clear only acknowledged data. If queued packets remain, schedule an immediate transmission wake to drain the batch. A failed link can prevent delivery; persist/flag an overdue report rather than claiming daily delivery succeeded.
- Radio frequency, power, SF/BW, transmit scheduling and backoff are configurable. The diagrams do not make an airtime, capacity or deployment-compliance claim.

## Quality and firmware boundaries

Acquisition faults and clipping are quality outcomes, not a healthy/RPW diagnosis. Invalid captures increment status counters and still lead to the daily-report check. The report's maximum/minimum energy variation statistic is not a measured noise-reference SNR or proof of RPW activity. Any additional rejection rule needs its own coverage evaluation.

The illustrated path is based on the existing reference features and current model/PCA contracts, with no per-window gain-normalization change. The host reference uses a zero-phase bandpass. A causal firmware substitute changes features and therefore needs parity checks or a matched training pipeline. The diagrams do not certify ESP32 buffer allocation, model arena RAM, timing, power, or deployed firmware behavior. No firmware, model or training results were modified to create these figures.

## Rebuild

In a Python environment containing ReportLab:

```bash
python scripts/build_single_layer_flows.py
```

The generator validates label widths, diamond fit, unique nodes, in-bounds geometry and decision branches. It exports SVG, diagrams.net XML, individual PDFs and the combined PDF from the same node/edge definitions. 