#ifndef PNU_DEPTH_REGISTRATION_CUDA_H
#define PNU_DEPTH_REGISTRATION_CUDA_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define PNU_DCR_EXPORT __declspec(dllexport)
#else
#define PNU_DCR_EXPORT __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define PNU_DCR_ABI_VERSION 1u

enum pnu_dcr_status {
  PNU_DCR_OK = 0,
  PNU_DCR_INVALID_ARGUMENT = 1,
  PNU_DCR_ABI_MISMATCH = 2,
  PNU_DCR_CUDA_ERROR = 3,
  PNU_DCR_INTERNAL_ERROR = 4,
};

enum pnu_dcr_input_type {
  PNU_DCR_INPUT_U16 = 1,
  PNU_DCR_INPUT_F32 = 2,
};

typedef struct pnu_dcr_config_v1 {
  uint32_t struct_size;
  uint32_t abi_version;
  int32_t device_id;
  int32_t depth_width;
  int32_t depth_height;
  int32_t color_width;
  int32_t color_height;
  int32_t distortion_count;
  float rotation_row_major[9];
  float translation_m[3];
  float color_k_row_major[9];
  float distortion[12];
} pnu_dcr_config_v1;

typedef struct pnu_dcr_diagnostics_v1 {
  uint32_t struct_size;
  uint32_t abi_version;
  uint64_t source_valid_pixels;
  uint64_t projected_points;
  uint64_t aligned_valid_pixels;
  float gpu_elapsed_ms;
} pnu_dcr_diagnostics_v1;

PNU_DCR_EXPORT uint32_t pnu_dcr_abi_version(void);
PNU_DCR_EXPORT const char *pnu_dcr_library_version(void);

PNU_DCR_EXPORT int pnu_dcr_create_v1(
    const pnu_dcr_config_v1 *config,
    const float *depth_ray_xy,
    size_t ray_count,
    void **handle_out,
    char *error_out,
    size_t error_capacity);

PNU_DCR_EXPORT int pnu_dcr_register_v1(
    void *handle,
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
    size_t error_capacity);

PNU_DCR_EXPORT void pnu_dcr_destroy_v1(void *handle);

#ifdef __cplusplus
}
#endif

#endif
