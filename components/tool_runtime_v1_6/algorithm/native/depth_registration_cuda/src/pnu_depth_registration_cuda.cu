#include "pnu_depth_registration_cuda.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>

namespace {

constexpr int kThreads = 256;
constexpr int kCounterCount = 3;

struct DeviceParameters {
  int depth_width;
  int depth_height;
  int color_width;
  int color_height;
  float rotation[9];
  float translation[3];
  float color_k[9];
  float distortion[12];
};

struct Context {
  int device_id = 0;
  size_t input_elements = 0;
  size_t output_elements = 0;
  float2 *depth_ray_xy = nullptr;
  void *native_depth = nullptr;
  float *aligned_depth = nullptr;
  unsigned long long *counters = nullptr;
  DeviceParameters *parameters = nullptr;
  cudaStream_t stream = nullptr;
  cudaEvent_t started = nullptr;
  cudaEvent_t completed = nullptr;
  bool poisoned = false;
  std::mutex mutex;
};

void clear_error(char *output, size_t capacity) {
  if (output != nullptr && capacity > 0) {
    output[0] = '\0';
  }
}

void write_error(char *output, size_t capacity, const char *message) {
  if (output == nullptr || capacity == 0) {
    return;
  }
  std::snprintf(output, capacity, "%s", message == nullptr ? "unknown error" : message);
}

void write_cuda_error(
    char *output, size_t capacity, const char *operation, cudaError_t status) {
  if (output == nullptr || capacity == 0) {
    return;
  }
  std::snprintf(
      output,
      capacity,
      "%s: %s (%d)",
      operation,
      cudaGetErrorString(status),
      static_cast<int>(status));
}

void destroy_context(Context *context) noexcept {
  if (context == nullptr) {
    return;
  }
  cudaSetDevice(context->device_id);
  if (context->stream != nullptr) {
    cudaStreamSynchronize(context->stream);
  }
  if (context->completed != nullptr) {
    cudaEventDestroy(context->completed);
  }
  if (context->started != nullptr) {
    cudaEventDestroy(context->started);
  }
  if (context->parameters != nullptr) {
    cudaFree(context->parameters);
  }
  if (context->counters != nullptr) {
    cudaFree(context->counters);
  }
  if (context->aligned_depth != nullptr) {
    cudaFree(context->aligned_depth);
  }
  if (context->native_depth != nullptr) {
    cudaFree(context->native_depth);
  }
  if (context->depth_ray_xy != nullptr) {
    cudaFree(context->depth_ray_xy);
  }
  if (context->stream != nullptr) {
    cudaStreamDestroy(context->stream);
  }
  delete context;
}

__global__ void initialize_output_kernel(float *output, size_t elements) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < elements) {
    output[index] = __int_as_float(0x7f800000);
  }
}

template <typename InputType>
__global__ void project_kernel(
    const InputType *input,
    size_t elements,
    const float2 *ray_xy,
    const DeviceParameters *parameters,
    float scale,
    float minimum_depth,
    float maximum_depth,
    float *aligned_depth,
    unsigned long long *counters) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned active_mask = __activemask();
  const int lane = threadIdx.x & 31;

  bool source_valid = false;
  bool projected = false;
  float z_color = 0.0f;
  int pixel_x = -1;
  int pixel_y = -1;

  if (index < elements) {
    const float raw_depth = static_cast<float>(input[index]);
    const float depth = __fmul_rn(raw_depth, scale);
    source_valid = isfinite(depth) && depth >= minimum_depth && depth <= maximum_depth;

    if (source_valid) {
      const float2 ray = ray_xy[index];
      const float x_depth = __fmul_rn(ray.x, depth);
      const float y_depth = __fmul_rn(ray.y, depth);

      const float *rotation = parameters->rotation;
      const float *translation = parameters->translation;
      const float x_color = __fadd_rn(
          __fadd_rn(
              __fadd_rn(
                  __fmul_rn(rotation[0], x_depth),
                  __fmul_rn(rotation[1], y_depth)),
              __fmul_rn(rotation[2], depth)),
          translation[0]);
      const float y_color = __fadd_rn(
          __fadd_rn(
              __fadd_rn(
                  __fmul_rn(rotation[3], x_depth),
                  __fmul_rn(rotation[4], y_depth)),
              __fmul_rn(rotation[5], depth)),
          translation[1]);
      z_color = __fadd_rn(
          __fadd_rn(
              __fadd_rn(
                  __fmul_rn(rotation[6], x_depth),
                  __fmul_rn(rotation[7], y_depth)),
              __fmul_rn(rotation[8], depth)),
          translation[2]);

      if (isfinite(x_color) && isfinite(y_color) && isfinite(z_color) && z_color > 0.0f) {
        const float normalized_x = x_color / z_color;
        const float normalized_y = y_color / z_color;
        const float squared_x = __fmul_rn(normalized_x, normalized_x);
        const float squared_y = __fmul_rn(normalized_y, normalized_y);
        const float radius2 = __fadd_rn(squared_x, squared_y);
        const float radius4 = __fmul_rn(radius2, radius2);
        const float radius6 = __fmul_rn(radius4, radius2);
        const float *d = parameters->distortion;

        const float numerator = __fadd_rn(
            __fadd_rn(
                __fadd_rn(1.0f, __fmul_rn(d[0], radius2)),
                __fmul_rn(d[1], radius4)),
            __fmul_rn(d[4], radius6));
        const float denominator = __fadd_rn(
            __fadd_rn(
                __fadd_rn(1.0f, __fmul_rn(d[5], radius2)),
                __fmul_rn(d[6], radius4)),
            __fmul_rn(d[7], radius6));
        const float radial = numerator / denominator;
        const float xy = __fmul_rn(normalized_x, normalized_y);

        const float distorted_x = __fadd_rn(
            __fadd_rn(
                __fadd_rn(
                    __fadd_rn(
                        __fmul_rn(normalized_x, radial),
                        __fmul_rn(2.0f * d[2], xy)),
                    __fmul_rn(d[3], __fadd_rn(radius2, 2.0f * squared_x))),
                __fmul_rn(d[8], radius2)),
            __fmul_rn(d[9], radius4));
        const float distorted_y = __fadd_rn(
            __fadd_rn(
                __fadd_rn(
                    __fadd_rn(
                        __fmul_rn(normalized_y, radial),
                        __fmul_rn(d[2], __fadd_rn(radius2, 2.0f * squared_y))),
                    __fmul_rn(2.0f * d[3], xy)),
                __fmul_rn(d[10], radius2)),
            __fmul_rn(d[11], radius4));

        const float *k = parameters->color_k;
        const float pixel_x_float = __fadd_rn(
            __fadd_rn(__fmul_rn(k[0], distorted_x), __fmul_rn(k[1], distorted_y)),
            k[2]);
        const float pixel_y_float = __fadd_rn(
            __fadd_rn(__fmul_rn(k[3], distorted_x), __fmul_rn(k[4], distorted_y)),
            k[5]);
        if (isfinite(pixel_x_float) && isfinite(pixel_y_float)) {
          pixel_x = __float2int_rn(pixel_x_float);
          pixel_y = __float2int_rn(pixel_y_float);
          projected =
              pixel_x >= 0 && pixel_x < parameters->color_width &&
              pixel_y >= 0 && pixel_y < parameters->color_height;
        }
      }
    }
  }

  const unsigned source_votes = __ballot_sync(active_mask, source_valid);
  if (lane == 0) {
    atomicAdd(&counters[0], static_cast<unsigned long long>(__popc(source_votes)));
  }
  const unsigned projected_votes = __ballot_sync(active_mask, projected);
  if (lane == 0) {
    atomicAdd(&counters[1], static_cast<unsigned long long>(__popc(projected_votes)));
  }
  if (projected) {
    const size_t output_index =
        static_cast<size_t>(pixel_y) * parameters->color_width + pixel_x;
    atomicMin(
        reinterpret_cast<unsigned int *>(&aligned_depth[output_index]),
        __float_as_uint(z_color));
  }
}

__global__ void finalize_output_kernel(
    float *output, size_t elements, unsigned long long *counters) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned active_mask = __activemask();
  const int lane = threadIdx.x & 31;
  bool valid = false;
  if (index < elements) {
    valid = isfinite(output[index]);
    if (!valid) {
      output[index] = nanf("");
    }
  }
  const unsigned votes = __ballot_sync(active_mask, valid);
  if (lane == 0) {
    atomicAdd(&counters[2], static_cast<unsigned long long>(__popc(votes)));
  }
}

template <typename InputType>
int register_typed(
    Context *context,
    const void *native_depth,
    float scale,
    float minimum,
    float maximum,
    float *output,
    pnu_dcr_diagnostics_v1 *diagnostics,
    char *error,
    size_t error_capacity) {
  cudaError_t status = cudaEventRecord(context->started, context->stream);
  if (status != cudaSuccess) {
    write_cuda_error(error, error_capacity, "cudaEventRecord(start)", status);
    return PNU_DCR_CUDA_ERROR;
  }
  status = cudaMemcpyAsync(
      context->native_depth,
      native_depth,
      context->input_elements * sizeof(InputType),
      cudaMemcpyHostToDevice,
      context->stream);
  if (status != cudaSuccess) {
    write_cuda_error(error, error_capacity, "native depth H2D", status);
    return PNU_DCR_CUDA_ERROR;
  }
  status = cudaMemsetAsync(
      context->counters,
      0,
      kCounterCount * sizeof(unsigned long long),
      context->stream);
  if (status != cudaSuccess) {
    write_cuda_error(error, error_capacity, "counter reset", status);
    return PNU_DCR_CUDA_ERROR;
  }

  const int input_blocks = static_cast<int>(
      (context->input_elements + kThreads - 1) / kThreads);
  const int output_blocks = static_cast<int>(
      (context->output_elements + kThreads - 1) / kThreads);
  initialize_output_kernel<<<output_blocks, kThreads, 0, context->stream>>>(
      context->aligned_depth, context->output_elements);
  project_kernel<InputType><<<input_blocks, kThreads, 0, context->stream>>>(
      static_cast<const InputType *>(context->native_depth),
      context->input_elements,
      context->depth_ray_xy,
      context->parameters,
      scale,
      minimum,
      maximum,
      context->aligned_depth,
      context->counters);
  finalize_output_kernel<<<output_blocks, kThreads, 0, context->stream>>>(
      context->aligned_depth, context->output_elements, context->counters);
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    write_cuda_error(error, error_capacity, "depth registration kernel launch", status);
    return PNU_DCR_CUDA_ERROR;
  }

  unsigned long long host_counters[kCounterCount] = {};
  status = cudaMemcpyAsync(
      output,
      context->aligned_depth,
      context->output_elements * sizeof(float),
      cudaMemcpyDeviceToHost,
      context->stream);
  if (status != cudaSuccess) {
    write_cuda_error(error, error_capacity, "aligned depth D2H", status);
    return PNU_DCR_CUDA_ERROR;
  }
  status = cudaMemcpyAsync(
      host_counters,
      context->counters,
      sizeof(host_counters),
      cudaMemcpyDeviceToHost,
      context->stream);
  if (status != cudaSuccess) {
    write_cuda_error(error, error_capacity, "diagnostics D2H", status);
    return PNU_DCR_CUDA_ERROR;
  }
  status = cudaEventRecord(context->completed, context->stream);
  if (status != cudaSuccess) {
    write_cuda_error(error, error_capacity, "cudaEventRecord(stop)", status);
    return PNU_DCR_CUDA_ERROR;
  }
  status = cudaEventSynchronize(context->completed);
  if (status != cudaSuccess) {
    write_cuda_error(error, error_capacity, "CUDA stream completion", status);
    return PNU_DCR_CUDA_ERROR;
  }
  float elapsed_ms = 0.0f;
  status = cudaEventElapsedTime(&elapsed_ms, context->started, context->completed);
  if (status != cudaSuccess) {
    write_cuda_error(error, error_capacity, "CUDA event elapsed time", status);
    return PNU_DCR_CUDA_ERROR;
  }

  diagnostics->source_valid_pixels = host_counters[0];
  diagnostics->projected_points = host_counters[1];
  diagnostics->aligned_valid_pixels = host_counters[2];
  diagnostics->gpu_elapsed_ms = elapsed_ms;
  return PNU_DCR_OK;
}

}  // namespace

extern "C" uint32_t pnu_dcr_abi_version(void) {
  return PNU_DCR_ABI_VERSION;
}

extern "C" const char *pnu_dcr_library_version(void) {
  return "0.1.0-cuda-cabi-v1";
}

extern "C" int pnu_dcr_create_v1(
    const pnu_dcr_config_v1 *config,
    const float *depth_ray_xy,
    size_t ray_count,
    void **handle_out,
    char *error_out,
    size_t error_capacity) {
  clear_error(error_out, error_capacity);
  if (handle_out != nullptr) {
    *handle_out = nullptr;
  }
  try {
    if (config == nullptr || depth_ray_xy == nullptr || handle_out == nullptr) {
      write_error(error_out, error_capacity, "config, ray buffer and handle_out are required");
      return PNU_DCR_INVALID_ARGUMENT;
    }
    if (config->struct_size != sizeof(pnu_dcr_config_v1) ||
        config->abi_version != PNU_DCR_ABI_VERSION) {
      write_error(error_out, error_capacity, "depth registration config ABI mismatch");
      return PNU_DCR_ABI_MISMATCH;
    }
    if (config->depth_width <= 0 || config->depth_height <= 0 ||
        config->color_width <= 0 || config->color_height <= 0 ||
        config->distortion_count < 0 || config->distortion_count > 12) {
      write_error(error_out, error_capacity, "invalid dimensions or distortion count");
      return PNU_DCR_INVALID_ARGUMENT;
    }
    const size_t input_elements =
        static_cast<size_t>(config->depth_width) * config->depth_height;
    const size_t output_elements =
        static_cast<size_t>(config->color_width) * config->color_height;
    if (ray_count != input_elements) {
      write_error(error_out, error_capacity, "depth ray count does not match depth dimensions");
      return PNU_DCR_INVALID_ARGUMENT;
    }

    auto *context = new (std::nothrow) Context();
    if (context == nullptr) {
      write_error(error_out, error_capacity, "could not allocate depth registration context");
      return PNU_DCR_INTERNAL_ERROR;
    }
    context->device_id = config->device_id;
    context->input_elements = input_elements;
    context->output_elements = output_elements;

    cudaError_t status = cudaSetDevice(context->device_id);
    if (status != cudaSuccess) {
      write_cuda_error(error_out, error_capacity, "cudaSetDevice", status);
      destroy_context(context);
      return PNU_DCR_CUDA_ERROR;
    }
    status = cudaStreamCreateWithFlags(&context->stream, cudaStreamNonBlocking);
    if (status == cudaSuccess) {
      status = cudaEventCreate(&context->started);
    }
    if (status == cudaSuccess) {
      status = cudaEventCreate(&context->completed);
    }
    if (status == cudaSuccess) {
      status = cudaMalloc(
          reinterpret_cast<void **>(&context->depth_ray_xy),
          input_elements * sizeof(float2));
    }
    if (status == cudaSuccess) {
      status = cudaMalloc(
          &context->native_depth,
          input_elements * sizeof(float));
    }
    if (status == cudaSuccess) {
      status = cudaMalloc(
          reinterpret_cast<void **>(&context->aligned_depth),
          output_elements * sizeof(float));
    }
    if (status == cudaSuccess) {
      status = cudaMalloc(
          reinterpret_cast<void **>(&context->counters),
          kCounterCount * sizeof(unsigned long long));
    }
    if (status == cudaSuccess) {
      status = cudaMalloc(
          reinterpret_cast<void **>(&context->parameters),
          sizeof(DeviceParameters));
    }
    if (status != cudaSuccess) {
      write_cuda_error(error_out, error_capacity, "CUDA context allocation", status);
      destroy_context(context);
      return PNU_DCR_CUDA_ERROR;
    }

    DeviceParameters parameters{};
    parameters.depth_width = config->depth_width;
    parameters.depth_height = config->depth_height;
    parameters.color_width = config->color_width;
    parameters.color_height = config->color_height;
    std::memcpy(parameters.rotation, config->rotation_row_major, sizeof(parameters.rotation));
    std::memcpy(parameters.translation, config->translation_m, sizeof(parameters.translation));
    std::memcpy(parameters.color_k, config->color_k_row_major, sizeof(parameters.color_k));
    std::memcpy(parameters.distortion, config->distortion, sizeof(parameters.distortion));

    status = cudaMemcpyAsync(
        context->depth_ray_xy,
        depth_ray_xy,
        input_elements * sizeof(float2),
        cudaMemcpyHostToDevice,
        context->stream);
    if (status == cudaSuccess) {
      status = cudaMemcpyAsync(
          context->parameters,
          &parameters,
          sizeof(parameters),
          cudaMemcpyHostToDevice,
          context->stream);
    }
    if (status == cudaSuccess) {
      status = cudaStreamSynchronize(context->stream);
    }
    if (status != cudaSuccess) {
      write_cuda_error(error_out, error_capacity, "CUDA context initialization", status);
      destroy_context(context);
      return PNU_DCR_CUDA_ERROR;
    }
    *handle_out = context;
    return PNU_DCR_OK;
  } catch (...) {
    write_error(error_out, error_capacity, "unexpected exception while creating CUDA context");
    return PNU_DCR_INTERNAL_ERROR;
  }
}

extern "C" int pnu_dcr_register_v1(
    void *opaque_handle,
    const void *native_depth,
    size_t input_elements,
    uint32_t input_type,
    float depth_scale_m_per_unit,
    float minimum_depth_m,
    float maximum_depth_m,
    float *aligned_depth_out,
    size_t output_elements,
    pnu_dcr_diagnostics_v1 *diagnostics_out,
    char *error_out,
    size_t error_capacity) {
  clear_error(error_out, error_capacity);
  try {
    auto *context = static_cast<Context *>(opaque_handle);
    if (context == nullptr || native_depth == nullptr || aligned_depth_out == nullptr ||
        diagnostics_out == nullptr) {
      write_error(error_out, error_capacity, "handle, input, output and diagnostics are required");
      return PNU_DCR_INVALID_ARGUMENT;
    }
    if (diagnostics_out->struct_size != sizeof(pnu_dcr_diagnostics_v1) ||
        diagnostics_out->abi_version != PNU_DCR_ABI_VERSION) {
      write_error(error_out, error_capacity, "depth registration diagnostics ABI mismatch");
      return PNU_DCR_ABI_MISMATCH;
    }
    if (input_elements != context->input_elements ||
        output_elements != context->output_elements ||
        !std::isfinite(depth_scale_m_per_unit) || depth_scale_m_per_unit <= 0.0f ||
        !std::isfinite(minimum_depth_m) || !std::isfinite(maximum_depth_m) ||
        minimum_depth_m < 0.0f || minimum_depth_m >= maximum_depth_m) {
      write_error(error_out, error_capacity, "invalid registration buffers, scale or depth limits");
      return PNU_DCR_INVALID_ARGUMENT;
    }

    std::lock_guard<std::mutex> lock(context->mutex);
    if (context->poisoned) {
      write_error(error_out, error_capacity, "CUDA depth registration context is poisoned");
      return PNU_DCR_CUDA_ERROR;
    }
    cudaError_t status = cudaSetDevice(context->device_id);
    if (status != cudaSuccess) {
      context->poisoned = true;
      write_cuda_error(error_out, error_capacity, "cudaSetDevice", status);
      return PNU_DCR_CUDA_ERROR;
    }

    int result = PNU_DCR_INVALID_ARGUMENT;
    if (input_type == PNU_DCR_INPUT_U16) {
      result = register_typed<uint16_t>(
          context,
          native_depth,
          depth_scale_m_per_unit,
          minimum_depth_m,
          maximum_depth_m,
          aligned_depth_out,
          diagnostics_out,
          error_out,
          error_capacity);
    } else if (input_type == PNU_DCR_INPUT_F32) {
      result = register_typed<float>(
          context,
          native_depth,
          depth_scale_m_per_unit,
          minimum_depth_m,
          maximum_depth_m,
          aligned_depth_out,
          diagnostics_out,
          error_out,
          error_capacity);
    } else {
      write_error(error_out, error_capacity, "unsupported native depth input type");
      return PNU_DCR_INVALID_ARGUMENT;
    }
    if (result == PNU_DCR_CUDA_ERROR || result == PNU_DCR_INTERNAL_ERROR) {
      context->poisoned = true;
    }
    return result;
  } catch (...) {
    write_error(error_out, error_capacity, "unexpected exception during CUDA registration");
    return PNU_DCR_INTERNAL_ERROR;
  }
}

extern "C" void pnu_dcr_destroy_v1(void *opaque_handle) {
  try {
    destroy_context(static_cast<Context *>(opaque_handle));
  } catch (...) {
  }
}
