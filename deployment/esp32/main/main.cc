/**
 * Benchmark generated INT8 models on the original ESP32 without claiming that
 * the inference-only memory footprint covers audio capture or communications.
 */
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include "esp_heap_caps.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "model_data.h"
#include "model_config.h"

#ifndef TREEVIBES_ARENA_BYTES
#define TREEVIBES_ARENA_BYTES (80 * 1024)
#endif

/** Run the reference MFCC fixture through the actual micro interpreter. */
extern "C" void app_main() {
    constexpr char tag[] = "treevibes";
    const auto* model = tflite::GetModel(g_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(tag, "Model schema mismatch");
        return;
    }
    auto* arena = static_cast<uint8_t*>(heap_caps_aligned_alloc(
        16, TREEVIBES_ARENA_BYTES, MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT));
    if (!arena) {
        ESP_LOGE(tag, "Arena allocation failed; largest internal block=%u",
                 static_cast<unsigned>(heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)));
        return;
    }
    // Allocate interpreter objects on the heap to retain stack headroom.
    auto* resolver = new tflite::MicroMutableOpResolver<32>();
    if (RegisterModelOps(*resolver) != kTfLiteOk) {
        ESP_LOGE(tag, "Resolver registration failed");
        delete resolver; heap_caps_free(arena); return;
    }
    auto* interpreter = new tflite::MicroInterpreter(model, *resolver, arena, TREEVIBES_ARENA_BYTES);
    if (interpreter->AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(tag, "Tensor allocation failed; increase arena only if application RAM permits");
        delete interpreter; delete resolver; heap_caps_free(arena); return;
    }
    TfLiteTensor* input = interpreter->input(0);
    TfLiteTensor* output = interpreter->output(0);
    if (input->type != kTfLiteInt8 || output->type != kTfLiteInt8 ||
        input->bytes != kInputCount || output->bytes != 1) {
        ESP_LOGE(tag, "Unexpected model input/output contract");
        delete interpreter; delete resolver; heap_caps_free(arena); return;
    }
    // Reproduce Python's training normalization and saturating input quantization.
    for (int i = 0; i < kInputCount; ++i) {
        const int coefficient = i % kCoefficients;
        const float normalized = (kRawFixture[i] - kMeans[coefficient]) / kScales[coefficient];
        const long quantized = lrintf(normalized / input->params.scale + input->params.zero_point);
        input->data.int8[i] = static_cast<int8_t>(std::max(-128L, std::min(127L, quantized)));
    }
    if (interpreter->Invoke() != kTfLiteOk) {
        ESP_LOGE(tag, "Warm-up inference failed");
        delete interpreter; delete resolver; heap_caps_free(arena); return;
    }
    constexpr int repeats = 20;
    const int64_t start = esp_timer_get_time();
    for (int i = 0; i < repeats; ++i) {
        if (interpreter->Invoke() != kTfLiteOk) {
            ESP_LOGE(tag, "Inference failed");
            delete interpreter; delete resolver; heap_caps_free(arena); return;
        }
    }
    const int64_t latency = (esp_timer_get_time() - start) / repeats;
    const int actual = output->data.int8[0];
    // One quantized unit tolerates target-kernel rounding; report the exact delta.
    const int difference = std::abs(actual - kExpectedOutput);
    const float logit = (actual - output->params.zero_point) * output->params.scale;
    const float probability = 1.f / (1.f + std::exp(-logit));
    ESP_LOGI(tag, "probability=%.6f infested=%d fixture_delta=%d", probability,
             probability >= kThreshold, difference);
    printf("{\"model_sha256\":\"%s\",\"scope\":\"inference_harness\","
           "\"arena_used_bytes\":%u,\"arena_reserved_bytes\":%u,\"inference_us\":%lld,"
           "\"fixture_pass\":%s,\"fixture_delta\":%d,\"free_internal_heap\":%u,"
           "\"minimum_free_heap\":%u,\"largest_free_block\":%u,\"stack_high_water_bytes\":%u}\n",
           kModelHash, static_cast<unsigned>(interpreter->arena_used_bytes()), TREEVIBES_ARENA_BYTES,
           static_cast<long long>(latency), difference <= 1 ? "true" : "false", difference,
           static_cast<unsigned>(heap_caps_get_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)),
           static_cast<unsigned>(heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)),
           static_cast<unsigned>(heap_caps_get_largest_free_block(MALLOC_CAP_INTERNAL | MALLOC_CAP_8BIT)),
           static_cast<unsigned>(uxTaskGetStackHighWaterMark(nullptr)));
    delete interpreter; delete resolver; heap_caps_free(arena);
}
