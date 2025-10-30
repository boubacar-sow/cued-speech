#ifndef TENSORFLOW_LITE_PROFILING_TELEMETRY_PROFILER_H_
#define TENSORFLOW_LITE_PROFILING_TELEMETRY_PROFILER_H_

namespace tflite {
namespace telemetry {

class TelemetryProfiler {
 public:
  virtual ~TelemetryProfiler() = default;
};

}  // namespace telemetry
}  // namespace tflite

#endif  // TENSORFLOW_LITE_PROFILING_TELEMETRY_PROFILER_H_

