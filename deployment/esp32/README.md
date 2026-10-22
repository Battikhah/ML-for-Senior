# ESP32-WROOM-32 inference benchmark

This harness tests an exported model on a fixed host-generated MFCC fixture. It does not implement audio acquisition or claim parity between librosa and an embedded MFCC extractor. Use the supplied whole-device ceilings of 2 MB flash and approximately 0.5 MB RAM, with no PSRAM. Confirm actual flash capacity and partitions on your board.

First generate a validated full INT8 export, then its firmware bundle:

```bash
python -m treevibes.cli bundle --export artifacts/exports/student_ptq --output deployment/esp32/main/generated
cd deployment/esp32
idf.py set-target esp32
idf.py build
idf.py size
idf.py size-components
idf.py -p YOUR_SERIAL_PORT flash monitor
```

Install ESP-IDF 5.3–6.0 first. The component manifest requests `esp-tflite-micro` 1.3.5, which integrates ESP-NN. The dependency manager's generated lockfile records the resolved component graph; retain that lockfile with your build evidence. If that version or an operation is incompatible with your ESP-IDF installation, record the failed build; changing the runtime version creates a new build configuration requiring re-verification.

The generated resolver lists only actual exported operators. Operator names alone do not establish that the micro runtime supports the emitted versions; a successful target build and fixture invocation supply that evidence. `model_data.cc` is generated and must not be hand-edited.

The default tensor arena reservation is 80 KiB. Increase or decrease `TREEVIBES_ARENA_BYTES` in `main.cc` based on allocation results and the remaining application RAM. An allocation failure is evidence that this reservation or model cannot run in the current configuration, not evidence that all of the nominal SRAM is available.

The benchmark reports mean latency for 20 warm invocations, arena usage/reservation, free/minimum/largest internal heap, stack headroom, and the difference from the host fixture output. A difference of at most one quantization unit is the stated fixture tolerance. Firmware normalization and input encoding are included; waveform-to-MFCC processing is pending.

Save the JSON serial line to a file. Add actual `firmware_bytes` and `app_partition_bytes` from the build/partition evidence, then import it:

```bash
python -m treevibes.cli board --measurement board-measurement.json --output artifacts/board
python -m treevibes.cli report --output artifacts --workbook artifacts/leaderboard.xlsx
```

Full deployment additionally needs capture buffers, FFT/filter/MFCC workspace, application stacks, persistent data and communications. Measure those in the complete application, including Wi-Fi if enabled in service. This benchmark disables Wi-Fi/Bluetooth and cannot certify the connected application. Do not call a model device-compatible until the actual embedded MFCC implementation passes numeric parity fixtures against its training feature contract.

No physical board or ESP-IDF toolchain was available during initial authoring; build and board status must remain pending until the commands above run.
