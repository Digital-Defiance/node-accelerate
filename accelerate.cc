// Apple Accelerate Framework Native Addon for Node.js
// Provides high-performance BLAS/vDSP operations for M4 Max
//
// This addon exposes Apple's Accelerate framework to JavaScript,
// giving you hardware-optimized:
// - Matrix multiplication (BLAS)
// - Vector operations (vDSP)
// - FFT (vDSP)
// - Convolution (vDSP)

#include <node_api.h>
#include <Accelerate/Accelerate.h>
#include <cstring>
#include <vector>

// Helper to throw JS errors
#define NAPI_THROW(env, msg) \
  napi_throw_error(env, nullptr, msg); \
  return nullptr;

#define NAPI_CHECK(call) \
  if ((call) != napi_ok) { \
    napi_throw_error(env, nullptr, "NAPI call failed"); \
    return nullptr; \
  }

// Get Float64Array data
static double* GetFloat64ArrayData(napi_env env, napi_value value, size_t* length) {
  bool is_typedarray;
  napi_is_typedarray(env, value, &is_typedarray);
  if (!is_typedarray) return nullptr;
  
  napi_typedarray_type type;
  size_t byte_length;
  void* data;
  napi_value arraybuffer;
  size_t byte_offset;
  
  napi_get_typedarray_info(env, value, &type, length, &data, &arraybuffer, &byte_offset);
  
  if (type != napi_float64_array) return nullptr;
  return static_cast<double*>(data);
}

// Get Float32Array data
static float* GetFloat32ArrayData(napi_env env, napi_value value, size_t* length) {
  bool is_typedarray;
  napi_is_typedarray(env, value, &is_typedarray);
  if (!is_typedarray) return nullptr;
  
  napi_typedarray_type type;
  size_t byte_length;
  void* data;
  napi_value arraybuffer;
  size_t byte_offset;
  
  napi_get_typedarray_info(env, value, &type, length, &data, &arraybuffer, &byte_offset);
  
  if (type != napi_float32_array) return nullptr;
  return static_cast<float*>(data);
}

// Matrix multiplication using BLAS dgemm (double precision)
// C = alpha * A * B + beta * C
// A is MxK, B is KxN, C is MxN
static napi_value MatMulDouble(napi_env env, napi_callback_info info) {
  size_t argc = 6;
  napi_value args[6];
  NAPI_CHECK(napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
  
  if (argc < 6) {
    NAPI_THROW(env, "matmul requires 6 arguments: A, B, C, M, K, N");
  }
  
  size_t len_a, len_b, len_c;
  double* A = GetFloat64ArrayData(env, args[0], &len_a);
  double* B = GetFloat64ArrayData(env, args[1], &len_b);
  double* C = GetFloat64ArrayData(env, args[2], &len_c);
  
  if (!A || !B || !C) {
    NAPI_THROW(env, "Arguments must be Float64Arrays");
  }
  
  int32_t M, K, N;
  napi_get_value_int32(env, args[3], &M);
  napi_get_value_int32(env, args[4], &K);
  napi_get_value_int32(env, args[5], &N);
  
  // BLAS dgemm: C = 1.0 * A * B + 0.0 * C
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              M, N, K,
              1.0, A, K,
              B, N,
              0.0, C, N);
  
  return args[2]; // Return C
}

// Matrix multiplication using BLAS sgemm (single precision)
static napi_value MatMulFloat(napi_env env, napi_callback_info info) {
  size_t argc = 6;
  napi_value args[6];
  NAPI_CHECK(napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
  
  if (argc < 6) {
    NAPI_THROW(env, "matmulFloat requires 6 arguments: A, B, C, M, K, N");
  }
  
  size_t len_a, len_b, len_c;
  float* A = GetFloat32ArrayData(env, args[0], &len_a);
  float* B = GetFloat32ArrayData(env, args[1], &len_b);
  float* C = GetFloat32ArrayData(env, args[2], &len_c);
  
  if (!A || !B || !C) {
    NAPI_THROW(env, "Arguments must be Float32Arrays");
  }
  
  int32_t M, K, N;
  napi_get_value_int32(env, args[3], &M);
  napi_get_value_int32(env, args[4], &K);
  napi_get_value_int32(env, args[5], &N);
  
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              M, N, K,
              1.0f, A, K,
              B, N,
              0.0f, C, N);
  
  return args[2];
}

// Vector dot product using vDSP
static napi_value DotProduct(napi_env env, napi_callback_info info) {
  size_t argc = 2;
  napi_value args[2];
  NAPI_CHECK(napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
  
  size_t len_a, len_b;
  double* A = GetFloat64ArrayData(env, args[0], &len_a);
  double* B = GetFloat64ArrayData(env, args[1], &len_b);
  
  if (!A || !B) {
    NAPI_THROW(env, "Arguments must be Float64Arrays");
  }
  
  size_t len = len_a < len_b ? len_a : len_b;
  double result;
  vDSP_dotprD(A, 1, B, 1, &result, len);
  
  napi_value js_result;
  napi_create_double(env, result, &js_result);
  return js_result;
}

// Vector add: C = A + B
static napi_value VectorAdd(napi_env env, napi_callback_info info) {
  size_t argc = 3;
  napi_value args[3];
  NAPI_CHECK(napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
  
  size_t len_a, len_b, len_c;
  double* A = GetFloat64ArrayData(env, args[0], &len_a);
  double* B = GetFloat64ArrayData(env, args[1], &len_b);
  double* C = GetFloat64ArrayData(env, args[2], &len_c);
  
  if (!A || !B || !C) {
    NAPI_THROW(env, "Arguments must be Float64Arrays");
  }
  
  size_t len = len_a < len_b ? len_a : len_b;
  len = len < len_c ? len : len_c;
  
  vDSP_vaddD(A, 1, B, 1, C, 1, len);
  
  return args[2];
}

// Vector subtract: C = A - B
static napi_value VectorSub(napi_env env, napi_callback_info info) {
  size_t argc = 3;
  napi_value args[3];
  NAPI_CHECK(napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
  
  size_t len_a, len_b, len_c;
  double* A = GetFloat64ArrayData(env, args[0], &len_a);
  double* B = GetFloat64ArrayData(env, args[1], &len_b);
  double* C = GetFloat64ArrayData(env, args[2], &len_c);
  
  if (!A || !B || !C) {
    NAPI_THROW(env, "Arguments must be Float64Arrays");
  }
  
  size_t len = len_a < len_b ? len_a : len_b;
  len = len < len_c ? len : len_c;
  
  vDSP_vsubD(B, 1, A, 1, C, 1, len);  // Note: vDSP_vsubD computes C = A - B as C[i] = B[i] - A[i]
  
  return args[2];
}

// Vector multiply: C = A * B (element-wise)
static napi_value VectorMul(napi_env env, napi_callback_info info) {
  size_t argc = 3;
  napi_value args[3];
  NAPI_CHECK(napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
  
  size_t len_a, len_b, len_c;
  double* A = GetFloat64ArrayData(env, args[0], &len_a);
  double* B = GetFloat64ArrayData(env, args[1], &len_b);
  double* C = GetFloat64ArrayData(env, args[2], &len_c);
  
  if (!A || !B || !C) {
    NAPI_THROW(env, "Arguments must be Float64Arrays");
  }
  
  size_t len = len_a < len_b ? len_a : len_b;
  len = len < len_c ? len : len_c;
  
  vDSP_vmulD(A, 1, B, 1, C, 1, len);
  
  return args[2];
}

// Vector divide: C = A / B (element-wise)
static napi_value VectorDiv(napi_env env, napi_callback_info info) {
  size_t argc = 3;
  napi_value args[3];
  NAPI_CHECK(napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
  
  size_t len_a, len_b, len_c;
  double* A = GetFloat64ArrayData(env, args[0], &len_a);
  double* B = GetFloat64ArrayData(env, args[1], &len_b);
  double* C = GetFloat64ArrayData(env, args[2], &len_c);
  
  if (!A || !B || !C) {
    NAPI_THROW(env, "Arguments must be Float64Arrays");
  }
  
  size_t len = len_a < len_b ? len_a : len_b;
  len = len < len_c ? len : len_c;
  
  vDSP_vdivD(B, 1, A, 1, C, 1, len);  // Note: vDSP_vdivD computes C = A / B as C[i] = B[i] / A[i]
  
  return args[2];
}

// Vector scale: B = A * scalar
static napi_value VectorScale(napi_env env, napi_callback_info info) {
  size_t argc = 3;
  napi_value args[3];
  NAPI_CHECK(napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
  
  size_t len_a, len_b;
  double* A = GetFloat64ArrayData(env, args[0], &len_a);
  double scalar;
  napi_get_value_double(env, args[1], &scalar);
  double* B = GetFloat64ArrayData(env, args[2], &len_b);
  
  if (!A || !B) {
    NAPI_THROW(env, "First and third arguments must be Float64Arrays");
  }
  
  size_t len = len_a < len_b ? len_a : len_b;
  vDSP_vsmulD(A, 1, &scalar, B, 1, len);
  
  return args[2];
}

// Sum of vector elements
static napi_value VectorSum(napi_env env, napi_callback_info info) {
  size_t argc = 1;
  napi_value args[1];
  NAPI_CHECK(napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
  
  size_t len;
  double* A = GetFloat64ArrayData(env, args[0], &len);
  
  if (!A) {
    NAPI_THROW(env, "Argument must be Float64Array");
  }
  
  double result;
  vDSP_sveD(A, 1, &result, len);
  
  napi_value js_result;
  napi_create_double(env, result, &js_result);
  return js_result;
}

// Mean of vector elements
static napi_value VectorMean(napi_env env, napi_callback_info info) {
  size_t argc = 1;
  napi_value args[1];
  NAPI_CHECK(napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
  
  size_t len;
  double* A = GetFloat64ArrayData(env, args[0], &len);
  
  if (!A) {
    NAPI_THROW(env, "Argument must be Float64Array");
  }
  
  double result;
  vDSP_meanvD(A, 1, &result, len);
  
  napi_value js_result;
  napi_create_double(env, result, &js_result);
  return js_result;
}

// Max of vector elements
static napi_value VectorMax(napi_env env, napi_callback_info info) {
  size_t argc = 1;
  napi_value args[1];
  NAPI_CHECK(napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
  
  size_t len;
  double* A = GetFloat64ArrayData(env, args[0], &len);
  
  if (!A) {
    NAPI_THROW(env, "Argument must be Float64Array");
  }
  
  double result;
  vDSP_maxvD(A, 1, &result, len);
  
  napi_value js_result;
  napi_create_double(env, result, &js_result);
  return js_result;
}

// Min of vector elements
static napi_value VectorMin(napi_env env, napi_callback_info info) {
  size_t argc = 1;
  napi_value args[1];
  NAPI_CHECK(napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
  
  size_t len;
  double* A = GetFloat64ArrayData(env, args[0], &len);
  
  if (!A) {
    NAPI_THROW(env, "Argument must be Float64Array");
  }
  
  double result;
  vDSP_minvD(A, 1, &result, len);
  
  napi_value js_result;
  napi_create_double(env, result, &js_result);
  return js_result;
}

// FFT (real to complex)
static napi_value FFT(napi_env env, napi_callback_info info) {
  size_t argc = 2;
  napi_value args[2];
  NAPI_CHECK(napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
  
  size_t len;
  double* input = GetFloat64ArrayData(env, args[0], &len);
  
  if (!input) {
    NAPI_THROW(env, "First argument must be Float64Array");
  }
  
  // Find log2 of length
  vDSP_Length log2n = 0;
  vDSP_Length n = len;
  while (n > 1) {
    n >>= 1;
    log2n++;
  }
  
  if ((1UL << log2n) != len) {
    NAPI_THROW(env, "Input length must be a power of 2");
  }
  
  // Setup FFT
  FFTSetupD setup = vDSP_create_fftsetupD(log2n, FFT_RADIX2);
  if (!setup) {
    NAPI_THROW(env, "Failed to create FFT setup");
  }
  
  // Allocate split complex arrays
  size_t half = len / 2;
  std::vector<double> real(half);
  std::vector<double> imag(half);
  
  DSPDoubleSplitComplex split;
  split.realp = real.data();
  split.imagp = imag.data();
  
  // Convert to split complex
  vDSP_ctozD((DSPDoubleComplex*)input, 2, &split, 1, half);
  
  // Perform FFT
  vDSP_fft_zripD(setup, &split, 1, log2n, FFT_FORWARD);
  
  // Scale
  double scale = 0.5;
  vDSP_vsmulD(split.realp, 1, &scale, split.realp, 1, half);
  vDSP_vsmulD(split.imagp, 1, &scale, split.imagp, 1, half);
  
  // Create output array (interleaved real/imag)
  napi_value arraybuffer;
  void* data;
  napi_create_arraybuffer(env, len * sizeof(double), &data, &arraybuffer);
  
  double* output = static_cast<double*>(data);
  for (size_t i = 0; i < half; i++) {
    output[i * 2] = real[i];
    output[i * 2 + 1] = imag[i];
  }
  
  napi_value result;
  napi_create_typedarray(env, napi_float64_array, len, arraybuffer, 0, &result);
  
  vDSP_destroy_fftsetupD(setup);
  
  return result;
}

// Matrix-vector multiply: y = A * x
static napi_value MatVecMul(napi_env env, napi_callback_info info) {
  size_t argc = 5;
  napi_value args[5];
  NAPI_CHECK(napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
  
  size_t len_a, len_x, len_y;
  double* A = GetFloat64ArrayData(env, args[0], &len_a);
  double* x = GetFloat64ArrayData(env, args[1], &len_x);
  double* y = GetFloat64ArrayData(env, args[2], &len_y);
  
  if (!A || !x || !y) {
    NAPI_THROW(env, "Arguments must be Float64Arrays");
  }
  
  int32_t M, N;
  NAPI_CHECK(napi_get_value_int32(env, args[3], &M));
  NAPI_CHECK(napi_get_value_int32(env, args[4], &N));
  
  // y = A * x using GEMV
  cblas_dgemv(CblasRowMajor, CblasNoTrans, M, N, 1.0, A, N, x, 1, 0.0, y, 1);
  
  return args[2];
}

// AXPY: y = a*x + y
static napi_value AXPY(napi_env env, napi_callback_info info) {
  size_t argc = 3;
  napi_value args[3];
  NAPI_CHECK(napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
  
  double alpha;
  NAPI_CHECK(napi_get_value_double(env, args[0], &alpha));
  
  size_t len_x, len_y;
  double* x = GetFloat64ArrayData(env, args[1], &len_x);
  double* y = GetFloat64ArrayData(env, args[2], &len_y);
  
  if (!x || !y) {
    NAPI_THROW(env, "Arguments must be Float64Arrays");
  }
  
  size_t len = len_x < len_y ? len_x : len_y;
  cblas_daxpy(len, alpha, x, 1, y, 1);
  
  return args[2];
}

// Vector absolute value
static napi_value VectorAbs(napi_env env, napi_callback_info info) {
  size_t argc = 2;
  napi_value args[2];
  NAPI_CHECK(napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
  
  size_t len_a, len_b;
  double* A = GetFloat64ArrayData(env, args[0], &len_a);
  double* B = GetFloat64ArrayData(env, args[1], &len_b);
  
  if (!A || !B) {
    NAPI_THROW(env, "Arguments must be Float64Arrays");
  }
  
  size_t len = len_a < len_b ? len_a : len_b;
  vDSP_vabsD(A, 1, B, 1, len);
  
  return args[1];
}

// Vector square: b = a^2
static napi_value VectorSquare(napi_env env, napi_callback_info info) {
  size_t argc = 2;
  napi_value args[2];
  NAPI_CHECK(napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
  
  size_t len_a, len_b;
  double* A = GetFloat64ArrayData(env, args[0], &len_a);
  double* B = GetFloat64ArrayData(env, args[1], &len_b);
  
  if (!A || !B) {
    NAPI_THROW(env, "Arguments must be Float64Arrays");
  }
  
  size_t len = len_a < len_b ? len_a : len_b;
  vDSP_vsqD(A, 1, B, 1, len);
  
  return args[1];
}

// Vector square root
static napi_value VectorSqrt(napi_env env, napi_callback_info info) {
  size_t argc = 2;
  napi_value args[2];
  NAPI_CHECK(napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
  
  size_t len_a, len_b;
  double* A = GetFloat64ArrayData(env, args[0], &len_a);
  double* B = GetFloat64ArrayData(env, args[1], &len_b);
  
  if (!A || !B) {
    NAPI_THROW(env, "Arguments must be Float64Arrays");
  }
  
  size_t len = len_a < len_b ? len_a : len_b;
  
  // vDSP doesn't have sqrt, use vForce
  int n = (int)len;
  vvsqrt(B, A, &n);
  
  return args[1];
}

// Vector normalize (unit vector)
static napi_value VectorNormalize(napi_env env, napi_callback_info info) {
  size_t argc = 2;
  napi_value args[2];
  NAPI_CHECK(napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
  
  size_t len_a, len_b;
  double* A = GetFloat64ArrayData(env, args[0], &len_a);
  double* B = GetFloat64ArrayData(env, args[1], &len_b);
  
  if (!A || !B) {
    NAPI_THROW(env, "Arguments must be Float64Arrays");
  }
  
  size_t len = len_a < len_b ? len_a : len_b;
  
  // Compute magnitude
  double magnitude;
  vDSP_dotprD(A, 1, A, 1, &magnitude, len);
  magnitude = sqrt(magnitude);
  
  if (magnitude > 0) {
    double inv_mag = 1.0 / magnitude;
    vDSP_vsmulD(A, 1, &inv_mag, B, 1, len);
  } else {
    // Zero vector, just copy
    memcpy(B, A, len * sizeof(double));
  }
  
  return args[1];
}

// Euclidean distance between two vectors
static napi_value EuclideanDistance(napi_env env, napi_callback_info info) {
  size_t argc = 2;
  napi_value args[2];
  NAPI_CHECK(napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
  
  size_t len_a, len_b;
  double* A = GetFloat64ArrayData(env, args[0], &len_a);
  double* B = GetFloat64ArrayData(env, args[1], &len_b);
  
  if (!A || !B) {
    NAPI_THROW(env, "Arguments must be Float64Arrays");
  }
  
  size_t len = len_a < len_b ? len_a : len_b;
  
  // Compute distance using vDSP
  double result;
  vDSP_distancesqD(A, 1, B, 1, &result, len);
  result = sqrt(result);
  
  napi_value js_result;
  napi_create_double(env, result, &js_result);
  return js_result;
}

// RMS (Root Mean Square)
static napi_value VectorRMS(napi_env env, napi_callback_info info) {
  size_t argc = 1;
  napi_value args[1];
  NAPI_CHECK(napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
  
  size_t len;
  double* A = GetFloat64ArrayData(env, args[0], &len);
  
  if (!A) {
    NAPI_THROW(env, "Argument must be Float64Array");
  }
  
  double result;
  vDSP_rmsqvD(A, 1, &result, len);
  
  napi_value js_result;
  napi_create_double(env, result, &js_result);
  return js_result;
}

// ============================================================================
// ADDITIONAL ACCELERATE FUNCTIONS
// ============================================================================

// Variance
static napi_value VectorVariance(napi_env env, napi_callback_info info) {
  size_t argc = 1;
  napi_value args[1];
  NAPI_CHECK(napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
  
  size_t len;
  double* A = GetFloat64ArrayData(env, args[0], &len);
  
  if (!A) {
    NAPI_THROW(env, "Argument must be Float64Array");
  }
  
  double mean_val, variance;
  vDSP_meanvD(A, 1, &mean_val, len);
  
  // Compute variance manually
  std::vector<double> diff(len);
  double neg_mean = -mean_val;
  vDSP_vsaddD(A, 1, &neg_mean, diff.data(), 1, len);
  vDSP_vsqD(diff.data(), 1, diff.data(), 1, len);
  vDSP_sveD(diff.data(), 1, &variance, len);
  variance /= len;
  
  napi_value js_result;
  napi_create_double(env, variance, &js_result);
  return js_result;
}

// Standard Deviation
static napi_value VectorStdDev(napi_env env, napi_callback_info info) {
  size_t argc = 1;
  napi_value args[1];
  NAPI_CHECK(napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
  
  size_t len;
  double* A = GetFloat64ArrayData(env, args[0], &len);
  
  if (!A) {
    NAPI_THROW(env, "Argument must be Float64Array");
  }
  
  double mean_val, variance;
  vDSP_meanvD(A, 1, &mean_val, len);
  
  std::vector<double> diff(len);
  double neg_mean = -mean_val;
  vDSP_vsaddD(A, 1, &neg_mean, diff.data(), 1, len);
  vDSP_vsqD(diff.data(), 1, diff.data(), 1, len);
  vDSP_sveD(diff.data(), 1, &variance, len);
  variance /= len;
  
  double stddev = sqrt(variance);
  
  napi_value js_result;
  napi_create_double(env, stddev, &js_result);
  return js_result;
}

// Min and Max together
static napi_value VectorMinMax(napi_env env, napi_callback_info info) {
  size_t argc = 1;
  napi_value args[1];
  NAPI_CHECK(napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
  
  size_t len;
  double* A = GetFloat64ArrayData(env, args[0], &len);
  
  if (!A) {
    NAPI_THROW(env, "Argument must be Float64Array");
  }
  
  double min_val, max_val;
  vDSP_minvD(A, 1, &min_val, len);
  vDSP_maxvD(A, 1, &max_val, len);
  
  napi_value result, min_js, max_js;
  napi_create_object(env, &result);
  napi_create_double(env, min_val, &min_js);
  napi_create_double(env, max_val, &max_js);
  napi_set_named_property(env, result, "min", min_js);
  napi_set_named_property(env, result, "max", max_js);
  
  return result;
}

// Trigonometric functions
static napi_value VectorSin(napi_env env, napi_callback_info info) {
  size_t argc = 2;
  napi_value args[2];
  NAPI_CHECK(napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
  
  size_t len_a, len_b;
  double* A = GetFloat64ArrayData(env, args[0], &len_a);
  double* B = GetFloat64ArrayData(env, args[1], &len_b);
  
  if (!A || !B) {
    NAPI_THROW(env, "Arguments must be Float64Arrays");
  }
  
  size_t len = len_a < len_b ? len_a : len_b;
  int n = (int)len;
  vvsin(B, A, &n);
  
  return args[1];
}

static napi_value VectorCos(napi_env env, napi_callback_info info) {
  size_t argc = 2;
  napi_value args[2];
  NAPI_CHECK(napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
  
  size_t len_a, len_b;
  double* A = GetFloat64ArrayData(env, args[0], &len_a);
  double* B = GetFloat64ArrayData(env, args[1], &len_b);
  
  if (!A || !B) {
    NAPI_THROW(env, "Arguments must be Float64Arrays");
  }
  
  size_t len = len_a < len_b ? len_a : len_b;
  int n = (int)len;
  vvcos(B, A, &n);
  
  return args[1];
}

static napi_value VectorTan(napi_env env, napi_callback_info info) {
  size_t argc = 2;
  napi_value args[2];
  NAPI_CHECK(napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
  
  size_t len_a, len_b;
  double* A = GetFloat64ArrayData(env, args[0], &len_a);
  double* B = GetFloat64ArrayData(env, args[1], &len_b);
  
  if (!A || !B) {
    NAPI_THROW(env, "Arguments must be Float64Arrays");
  }
  
  size_t len = len_a < len_b ? len_a : len_b;
  int n = (int)len;
  vvtan(B, A, &n);
  
  return args[1];
}

// Exponential and logarithm
static napi_value VectorExp(napi_env env, napi_callback_info info) {
  size_t argc = 2;
  napi_value args[2];
  NAPI_CHECK(napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
  
  size_t len_a, len_b;
  double* A = GetFloat64ArrayData(env, args[0], &len_a);
  double* B = GetFloat64ArrayData(env, args[1], &len_b);
  
  if (!A || !B) {
    NAPI_THROW(env, "Arguments must be Float64Arrays");
  }
  
  size_t len = len_a < len_b ? len_a : len_b;
  int n = (int)len;
  vvexp(B, A, &n);
  
  return args[1];
}

static napi_value VectorLog(napi_env env, napi_callback_info info) {
  size_t argc = 2;
  napi_value args[2];
  NAPI_CHECK(napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
  
  size_t len_a, len_b;
  double* A = GetFloat64ArrayData(env, args[0], &len_a);
  double* B = GetFloat64ArrayData(env, args[1], &len_b);
  
  if (!A || !B) {
    NAPI_THROW(env, "Arguments must be Float64Arrays");
  }
  
  size_t len = len_a < len_b ? len_a : len_b;
  int n = (int)len;
  vvlog(B, A, &n);
  
  return args[1];
}

static napi_value VectorLog10(napi_env env, napi_callback_info info) {
  size_t argc = 2;
  napi_value args[2];
  NAPI_CHECK(napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
  
  size_t len_a, len_b;
  double* A = GetFloat64ArrayData(env, args[0], &len_a);
  double* B = GetFloat64ArrayData(env, args[1], &len_b);
  
  if (!A || !B) {
    NAPI_THROW(env, "Arguments must be Float64Arrays");
  }
  
  size_t len = len_a < len_b ? len_a : len_b;
  int n = (int)len;
  vvlog10(B, A, &n);
  
  return args[1];
}

// Power functions
static napi_value VectorPow(napi_env env, napi_callback_info info) {
  size_t argc = 3;
  napi_value args[3];
  NAPI_CHECK(napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
  
  size_t len_a, len_b, len_c;
  double* A = GetFloat64ArrayData(env, args[0], &len_a);
  double* B = GetFloat64ArrayData(env, args[1], &len_b);
  double* C = GetFloat64ArrayData(env, args[2], &len_c);
  
  if (!A || !B || !C) {
    NAPI_THROW(env, "Arguments must be Float64Arrays");
  }
  
  size_t len = len_a < len_b ? len_a : len_b;
  len = len < len_c ? len : len_c;
  int n = (int)len;
  vvpow(C, B, A, &n);
  
  return args[2];
}

// Clipping
static napi_value VectorClip(napi_env env, napi_callback_info info) {
  size_t argc = 4;
  napi_value args[4];
  NAPI_CHECK(napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
  
  size_t len_a, len_b;
  double* A = GetFloat64ArrayData(env, args[0], &len_a);
  double* B = GetFloat64ArrayData(env, args[1], &len_b);
  
  if (!A || !B) {
    NAPI_THROW(env, "First two arguments must be Float64Arrays");
  }
  
  double min_val, max_val;
  NAPI_CHECK(napi_get_value_double(env, args[2], &min_val));
  NAPI_CHECK(napi_get_value_double(env, args[3], &max_val));
  
  size_t len = len_a < len_b ? len_a : len_b;
  vDSP_vclipD(A, 1, &min_val, &max_val, B, 1, len);
  
  return args[1];
}

// Threshold
static napi_value VectorThreshold(napi_env env, napi_callback_info info) {
  size_t argc = 3;
  napi_value args[3];
  NAPI_CHECK(napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
  
  size_t len_a, len_b;
  double* A = GetFloat64ArrayData(env, args[0], &len_a);
  double* B = GetFloat64ArrayData(env, args[1], &len_b);
  
  if (!A || !B) {
    NAPI_THROW(env, "First two arguments must be Float64Arrays");
  }
  
  double threshold;
  NAPI_CHECK(napi_get_value_double(env, args[2], &threshold));
  
  size_t len = len_a < len_b ? len_a : len_b;
  vDSP_vthrD(A, 1, &threshold, B, 1, len);
  
  return args[1];
}

// Convolution
static napi_value Convolve(napi_env env, napi_callback_info info) {
  size_t argc = 3;
  napi_value args[3];
  NAPI_CHECK(napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
  
  size_t signal_len, kernel_len, result_len;
  double* signal = GetFloat64ArrayData(env, args[0], &signal_len);
  double* kernel = GetFloat64ArrayData(env, args[1], &kernel_len);
  double* result = GetFloat64ArrayData(env, args[2], &result_len);
  
  if (!signal || !kernel || !result) {
    NAPI_THROW(env, "Arguments must be Float64Arrays");
  }
  
  vDSP_convD(signal, 1, kernel, 1, result, 1, result_len, kernel_len);
  
  return args[2];
}

// Cross-correlation
static napi_value CrossCorrelation(napi_env env, napi_callback_info info) {
  size_t argc = 3;
  napi_value args[3];
  NAPI_CHECK(napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
  
  size_t len_a, len_b, len_c;
  double* A = GetFloat64ArrayData(env, args[0], &len_a);
  double* B = GetFloat64ArrayData(env, args[1], &len_b);
  double* C = GetFloat64ArrayData(env, args[2], &len_c);
  
  if (!A || !B || !C) {
    NAPI_THROW(env, "Arguments must be Float64Arrays");
  }
  
  size_t result_len = len_a + len_b - 1;
  if (len_c < result_len) {
    NAPI_THROW(env, "Result array too small");
  }
  
  // Use convolution with reversed kernel for correlation
  std::vector<double> b_reversed(len_b);
  for (size_t i = 0; i < len_b; i++) {
    b_reversed[i] = B[len_b - 1 - i];
  }
  
  vDSP_convD(A, 1, b_reversed.data(), 1, C, 1, len_a, len_b);
  
  return args[2];
}

// Window functions
static napi_value HammingWindow(napi_env env, napi_callback_info info) {
  size_t argc = 1;
  napi_value args[1];
  NAPI_CHECK(napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
  
  int32_t length;
  NAPI_CHECK(napi_get_value_int32(env, args[0], &length));
  
  napi_value arraybuffer;
  void* data;
  napi_create_arraybuffer(env, length * sizeof(double), &data, &arraybuffer);
  double* window = static_cast<double*>(data);
  
  vDSP_hamm_windowD(window, length, 0);
  
  napi_value result;
  napi_create_typedarray(env, napi_float64_array, length, arraybuffer, 0, &result);
  
  return result;
}

static napi_value HanningWindow(napi_env env, napi_callback_info info) {
  size_t argc = 1;
  napi_value args[1];
  NAPI_CHECK(napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
  
  int32_t length;
  NAPI_CHECK(napi_get_value_int32(env, args[0], &length));
  
  napi_value arraybuffer;
  void* data;
  napi_create_arraybuffer(env, length * sizeof(double), &data, &arraybuffer);
  double* window = static_cast<double*>(data);
  
  vDSP_hann_windowD(window, length, 0);
  
  napi_value result;
  napi_create_typedarray(env, napi_float64_array, length, arraybuffer, 0, &result);
  
  return result;
}

static napi_value BlackmanWindow(napi_env env, napi_callback_info info) {
  size_t argc = 1;
  napi_value args[1];
  NAPI_CHECK(napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
  
  int32_t length;
  NAPI_CHECK(napi_get_value_int32(env, args[0], &length));
  
  napi_value arraybuffer;
  void* data;
  napi_create_arraybuffer(env, length * sizeof(double), &data, &arraybuffer);
  double* window = static_cast<double*>(data);
  
  vDSP_blkman_windowD(window, length, 0);
  
  napi_value result;
  napi_create_typedarray(env, napi_float64_array, length, arraybuffer, 0, &result);
  
  return result;
}

// Matrix transpose
static napi_value MatrixTranspose(napi_env env, napi_callback_info info) {
  size_t argc = 4;
  napi_value args[4];
  NAPI_CHECK(napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
  
  size_t len_a, len_b;
  double* A = GetFloat64ArrayData(env, args[0], &len_a);
  double* B = GetFloat64ArrayData(env, args[1], &len_b);
  
  if (!A || !B) {
    NAPI_THROW(env, "First two arguments must be Float64Arrays");
  }
  
  int32_t rows, cols;
  NAPI_CHECK(napi_get_value_int32(env, args[2], &rows));
  NAPI_CHECK(napi_get_value_int32(env, args[3], &cols));
  
  vDSP_mtransD(A, 1, B, 1, cols, rows);
  
  return args[1];
}

// Inverse FFT
static napi_value IFFT(napi_env env, napi_callback_info info) {
  size_t argc = 2;
  napi_value args[2];
  NAPI_CHECK(napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
  
  size_t real_len, imag_len;
  double* real_in = GetFloat64ArrayData(env, args[0], &real_len);
  double* imag_in = GetFloat64ArrayData(env, args[1], &imag_len);
  
  if (!real_in || !imag_in) {
    NAPI_THROW(env, "Arguments must be Float64Arrays");
  }
  
  if (real_len != imag_len) {
    NAPI_THROW(env, "Real and imaginary arrays must have same length");
  }
  
  size_t half = real_len;
  size_t len = half * 2;
  
  vDSP_Length log2n = 0;
  vDSP_Length n = len;
  while (n > 1) {
    n >>= 1;
    log2n++;
  }
  
  if ((1UL << log2n) != len) {
    NAPI_THROW(env, "Length must be power of 2");
  }
  
  FFTSetupD setup = vDSP_create_fftsetupD(log2n, FFT_RADIX2);
  if (!setup) {
    NAPI_THROW(env, "Failed to create FFT setup");
  }
  
  std::vector<double> real(half);
  std::vector<double> imag(half);
  memcpy(real.data(), real_in, half * sizeof(double));
  memcpy(imag.data(), imag_in, half * sizeof(double));
  
  DSPDoubleSplitComplex split;
  split.realp = real.data();
  split.imagp = imag.data();
  
  vDSP_fft_zripD(setup, &split, 1, log2n, FFT_INVERSE);
  
  double scale = 1.0 / len;
  vDSP_vsmulD(split.realp, 1, &scale, split.realp, 1, half);
  vDSP_vsmulD(split.imagp, 1, &scale, split.imagp, 1, half);
  
  napi_value arraybuffer;
  void* data;
  napi_create_arraybuffer(env, len * sizeof(double), &data, &arraybuffer);
  double* output = static_cast<double*>(data);
  
  vDSP_ztocD(&split, 1, (DSPDoubleComplex*)output, 2, half);
  
  napi_value result;
  napi_create_typedarray(env, napi_float64_array, len, arraybuffer, 0, &result);
  
  vDSP_destroy_fftsetupD(setup);
  
  return result;
}

// Linear interpolation
static napi_value Interp1D(napi_env env, napi_callback_info info) {
  size_t argc = 4;
  napi_value args[4];
  NAPI_CHECK(napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
  
  size_t x_len, y_len, xi_len, yi_len;
  double* x = GetFloat64ArrayData(env, args[0], &x_len);
  double* y = GetFloat64ArrayData(env, args[1], &y_len);
  double* xi = GetFloat64ArrayData(env, args[2], &xi_len);
  double* yi = GetFloat64ArrayData(env, args[3], &yi_len);
  
  if (!x || !y || !xi || !yi) {
    NAPI_THROW(env, "Arguments must be Float64Arrays");
  }
  
  // Simple linear interpolation implementation
  for (size_t i = 0; i < xi_len; i++) {
    double xi_val = xi[i];
    
    // Find the interval
    size_t j = 0;
    while (j < x_len - 1 && x[j + 1] < xi_val) {
      j++;
    }
    
    if (j >= x_len - 1) {
      yi[i] = y[x_len - 1];
    } else {
      // Linear interpolation
      double t = (xi_val - x[j]) / (x[j + 1] - x[j]);
      yi[i] = y[j] + t * (y[j + 1] - y[j]);
    }
  }
  
  return args[3];
}

// Reverse vector
static napi_value VectorReverse(napi_env env, napi_callback_info info) {
  size_t argc = 2;
  napi_value args[2];
  NAPI_CHECK(napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
  
  size_t len_a, len_b;
  double* A = GetFloat64ArrayData(env, args[0], &len_a);
  double* B = GetFloat64ArrayData(env, args[1], &len_b);
  
  if (!A || !B) {
    NAPI_THROW(env, "Arguments must be Float64Arrays");
  }
  
  size_t len = len_a < len_b ? len_a : len_b;
  
  // vDSP_vrvrsD reverses in-place, so copy first then reverse
  memcpy(B, A, len * sizeof(double));
  vDSP_vrvrsD(B, 1, len);
  
  return args[1];
}

// Vector negate
static napi_value VectorNegate(napi_env env, napi_callback_info info) {
  size_t argc = 2;
  napi_value args[2];
  NAPI_CHECK(napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
  
  size_t len_a, len_b;
  double* A = GetFloat64ArrayData(env, args[0], &len_a);
  double* B = GetFloat64ArrayData(env, args[1], &len_b);
  
  if (!A || !B) {
    NAPI_THROW(env, "Arguments must be Float64Arrays");
  }
  
  size_t len = len_a < len_b ? len_a : len_b;
  vDSP_vnegD(A, 1, B, 1, len);
  
  return args[1];
}

// Sum of squares
static napi_value SumOfSquares(napi_env env, napi_callback_info info) {
  size_t argc = 1;
  napi_value args[1];
  NAPI_CHECK(napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
  
  size_t len;
  double* A = GetFloat64ArrayData(env, args[0], &len);
  
  if (!A) {
    NAPI_THROW(env, "Argument must be Float64Array");
  }
  
  double result;
  vDSP_svesqD(A, 1, &result, len);
  
  napi_value js_result;
  napi_create_double(env, result, &js_result);
  return js_result;
}

// Mean magnitude
static napi_value MeanMagnitude(napi_env env, napi_callback_info info) {
  size_t argc = 1;
  napi_value args[1];
  NAPI_CHECK(napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
  
  size_t len;
  double* A = GetFloat64ArrayData(env, args[0], &len);
  
  if (!A) {
    NAPI_THROW(env, "Argument must be Float64Array");
  }
  
  double result;
  vDSP_meamgvD(A, 1, &result, len);
  
  napi_value js_result;
  napi_create_double(env, result, &js_result);
  return js_result;
}

// Mean square
static napi_value MeanSquare(napi_env env, napi_callback_info info) {
  size_t argc = 1;
  napi_value args[1];
  NAPI_CHECK(napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
  
  size_t len;
  double* A = GetFloat64ArrayData(env, args[0], &len);
  
  if (!A) {
    NAPI_THROW(env, "Argument must be Float64Array");
  }
  
  double result;
  vDSP_measqvD(A, 1, &result, len);
  
  napi_value js_result;
  napi_create_double(env, result, &js_result);
  return js_result;
}

// ============================================================================
// ADDITIONAL BLAS OPERATIONS
// ============================================================================

// Vector copy: y = x
static napi_value VectorCopy(napi_env env, napi_callback_info info) {
  size_t argc = 2;
  napi_value args[2];
  NAPI_CHECK(napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
  
  size_t len_x, len_y;
  double* x = GetFloat64ArrayData(env, args[0], &len_x);
  double* y = GetFloat64ArrayData(env, args[1], &len_y);
  
  if (!x || !y) {
    NAPI_THROW(env, "Arguments must be Float64Arrays");
  }
  
  size_t len = len_x < len_y ? len_x : len_y;
  cblas_dcopy(len, x, 1, y, 1);
  
  return args[1];
}

// Vector swap: swap x and y
static napi_value VectorSwap(napi_env env, napi_callback_info info) {
  size_t argc = 2;
  napi_value args[2];
  NAPI_CHECK(napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
  
  size_t len_x, len_y;
  double* x = GetFloat64ArrayData(env, args[0], &len_x);
  double* y = GetFloat64ArrayData(env, args[1], &len_y);
  
  if (!x || !y) {
    NAPI_THROW(env, "Arguments must be Float64Arrays");
  }
  
  size_t len = len_x < len_y ? len_x : len_y;
  cblas_dswap(len, x, 1, y, 1);
  
  return args[0];
}

// Vector norm (L2 norm / Euclidean length)
static napi_value VectorNorm(napi_env env, napi_callback_info info) {
  size_t argc = 1;
  napi_value args[1];
  NAPI_CHECK(napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
  
  size_t len;
  double* x = GetFloat64ArrayData(env, args[0], &len);
  
  if (!x) {
    NAPI_THROW(env, "Argument must be Float64Array");
  }
  
  double result = cblas_dnrm2(len, x, 1);
  
  napi_value js_result;
  napi_create_double(env, result, &js_result);
  return js_result;
}

// Vector sum of absolute values
static napi_value VectorAbsSum(napi_env env, napi_callback_info info) {
  size_t argc = 1;
  napi_value args[1];
  NAPI_CHECK(napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
  
  size_t len;
  double* x = GetFloat64ArrayData(env, args[0], &len);
  
  if (!x) {
    NAPI_THROW(env, "Argument must be Float64Array");
  }
  
  double result = cblas_dasum(len, x, 1);
  
  napi_value js_result;
  napi_create_double(env, result, &js_result);
  return js_result;
}

// Index of maximum absolute value
static napi_value VectorMaxAbsIndex(napi_env env, napi_callback_info info) {
  size_t argc = 1;
  napi_value args[1];
  NAPI_CHECK(napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
  
  size_t len;
  double* x = GetFloat64ArrayData(env, args[0], &len);
  
  if (!x) {
    NAPI_THROW(env, "Argument must be Float64Array");
  }
  
  size_t result = cblas_idamax(len, x, 1);
  
  napi_value js_result;
  napi_create_uint32(env, result, &js_result);
  return js_result;
}

// Rotation: apply Givens rotation
static napi_value VectorRotation(napi_env env, napi_callback_info info) {
  size_t argc = 4;
  napi_value args[4];
  NAPI_CHECK(napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
  
  size_t len_x, len_y;
  double* x = GetFloat64ArrayData(env, args[0], &len_x);
  double* y = GetFloat64ArrayData(env, args[1], &len_y);
  
  if (!x || !y) {
    NAPI_THROW(env, "First two arguments must be Float64Arrays");
  }
  
  double c, s;
  NAPI_CHECK(napi_get_value_double(env, args[2], &c));
  NAPI_CHECK(napi_get_value_double(env, args[3], &s));
  
  size_t len = len_x < len_y ? len_x : len_y;
  cblas_drot(len, x, 1, y, 1, c, s);
  
  return args[0];
}

// ============================================================================
// ADDITIONAL vDSP OPERATIONS
// ============================================================================

// Vector fill with scalar
static napi_value VectorFill(napi_env env, napi_callback_info info) {
  size_t argc = 2;
  napi_value args[2];
  NAPI_CHECK(napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
  
  double scalar;
  NAPI_CHECK(napi_get_value_double(env, args[0], &scalar));
  
  size_t len;
  double* vec = GetFloat64ArrayData(env, args[1], &len);
  
  if (!vec) {
    NAPI_THROW(env, "Second argument must be Float64Array");
  }
  
  vDSP_vfillD(&scalar, vec, 1, len);
  
  return args[1];
}

// Vector ramp: generate linear sequence
static napi_value VectorRamp(napi_env env, napi_callback_info info) {
  size_t argc = 3;
  napi_value args[3];
  NAPI_CHECK(napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
  
  double start, step;
  NAPI_CHECK(napi_get_value_double(env, args[0], &start));
  NAPI_CHECK(napi_get_value_double(env, args[1], &step));
  
  size_t len;
  double* vec = GetFloat64ArrayData(env, args[2], &len);
  
  if (!vec) {
    NAPI_THROW(env, "Third argument must be Float64Array");
  }
  
  vDSP_vrampD(&start, &step, vec, 1, len);
  
  return args[2];
}

// Vector add with scalar: c = a + b (scalar)
static napi_value VectorAddScalar(napi_env env, napi_callback_info info) {
  size_t argc = 3;
  napi_value args[3];
  NAPI_CHECK(napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
  
  size_t len_a, len_c;
  double* a = GetFloat64ArrayData(env, args[0], &len_a);
  
  double scalar;
  NAPI_CHECK(napi_get_value_double(env, args[1], &scalar));
  
  double* c = GetFloat64ArrayData(env, args[2], &len_c);
  
  if (!a || !c) {
    NAPI_THROW(env, "First and third arguments must be Float64Arrays");
  }
  
  size_t len = len_a < len_c ? len_a : len_c;
  vDSP_vsaddD(a, 1, &scalar, c, 1, len);
  
  return args[2];
}

// Vector multiply and add: d = (a * b) + c
static napi_value VectorMultiplyAdd(napi_env env, napi_callback_info info) {
  size_t argc = 4;
  napi_value args[4];
  NAPI_CHECK(napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
  
  size_t len_a, len_b, len_c, len_d;
  double* a = GetFloat64ArrayData(env, args[0], &len_a);
  double* b = GetFloat64ArrayData(env, args[1], &len_b);
  double* c = GetFloat64ArrayData(env, args[2], &len_c);
  double* d = GetFloat64ArrayData(env, args[3], &len_d);
  
  if (!a || !b || !c || !d) {
    NAPI_THROW(env, "Arguments must be Float64Arrays");
  }
  
  size_t len = len_a;
  if (len_b < len) len = len_b;
  if (len_c < len) len = len_c;
  if (len_d < len) len = len_d;
  
  vDSP_vmaD(a, 1, b, 1, c, 1, d, 1, len);
  
  return args[3];
}

// Vector multiply and scalar add: d = (a * b) + c (scalar)
static napi_value VectorMultiplyScalarAdd(napi_env env, napi_callback_info info) {
  size_t argc = 4;
  napi_value args[4];
  NAPI_CHECK(napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
  
  size_t len_a, len_b, len_d;
  double* a = GetFloat64ArrayData(env, args[0], &len_a);
  double* b = GetFloat64ArrayData(env, args[1], &len_b);
  
  double c;
  NAPI_CHECK(napi_get_value_double(env, args[2], &c));
  
  double* d = GetFloat64ArrayData(env, args[3], &len_d);
  
  if (!a || !b || !d) {
    NAPI_THROW(env, "First, second, and fourth arguments must be Float64Arrays");
  }
  
  size_t len = len_a;
  if (len_b < len) len = len_b;
  if (len_d < len) len = len_d;
  
  vDSP_vmsaD(a, 1, b, 1, &c, d, 1, len);
  
  return args[3];
}

// Linear interpolation between two vectors
static napi_value VectorLinearInterpolate(napi_env env, napi_callback_info info) {
  size_t argc = 4;
  napi_value args[4];
  NAPI_CHECK(napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
  
  size_t len_a, len_b, len_c;
  double* a = GetFloat64ArrayData(env, args[0], &len_a);
  double* b = GetFloat64ArrayData(env, args[1], &len_b);
  
  double t;
  NAPI_CHECK(napi_get_value_double(env, args[2], &t));
  
  double* c = GetFloat64ArrayData(env, args[3], &len_c);
  
  if (!a || !b || !c) {
    NAPI_THROW(env, "First, second, and fourth arguments must be Float64Arrays");
  }
  
  size_t len = len_a;
  if (len_b < len) len = len_b;
  if (len_c < len) len = len_c;
  
  vDSP_vintbD(a, 1, b, 1, &t, c, 1, len);
  
  return args[3];
}

// Vector clear (set to zero)
static napi_value VectorClear(napi_env env, napi_callback_info info) {
  size_t argc = 1;
  napi_value args[1];
  NAPI_CHECK(napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
  
  size_t len;
  double* vec = GetFloat64ArrayData(env, args[0], &len);
  
  if (!vec) {
    NAPI_THROW(env, "Argument must be Float64Array");
  }
  
  vDSP_vclrD(vec, 1, len);
  
  return args[0];
}

// Vector limit (saturate)
static napi_value VectorLimit(napi_env env, napi_callback_info info) {
  size_t argc = 4;
  napi_value args[4];
  NAPI_CHECK(napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
  
  size_t len_a, len_c;
  double* a = GetFloat64ArrayData(env, args[0], &len_a);
  
  double low, high;
  NAPI_CHECK(napi_get_value_double(env, args[1], &low));
  NAPI_CHECK(napi_get_value_double(env, args[2], &high));
  
  double* c = GetFloat64ArrayData(env, args[3], &len_c);
  
  if (!a || !c) {
    NAPI_THROW(env, "First and fourth arguments must be Float64Arrays");
  }
  
  size_t len = len_a < len_c ? len_a : len_c;
  vDSP_vlimD(a, 1, &low, &high, c, 1, len);
  
  return args[3];
}

// Vector maximum magnitude
static napi_value VectorMaxMagnitude(napi_env env, napi_callback_info info) {
  size_t argc = 1;
  napi_value args[1];
  NAPI_CHECK(napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
  
  size_t len;
  double* a = GetFloat64ArrayData(env, args[0], &len);
  
  if (!a) {
    NAPI_THROW(env, "Argument must be Float64Array");
  }
  
  double result;
  vDSP_maxmgvD(a, 1, &result, len);
  
  napi_value js_result;
  napi_create_double(env, result, &js_result);
  return js_result;
}

// Vector minimum magnitude
static napi_value VectorMinMagnitude(napi_env env, napi_callback_info info) {
  size_t argc = 1;
  napi_value args[1];
  NAPI_CHECK(napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
  
  size_t len;
  double* a = GetFloat64ArrayData(env, args[0], &len);
  
  if (!a) {
    NAPI_THROW(env, "Argument must be Float64Array");
  }
  
  double result;
  vDSP_minmgvD(a, 1, &result, len);
  
  napi_value js_result;
  napi_create_double(env, result, &js_result);
  return js_result;
}

// ============================================================================
// MORE MATH FUNCTIONS (vForce)
// ============================================================================

// Inverse (reciprocal): b = 1/a
static napi_value VectorReciprocal(napi_env env, napi_callback_info info) {
  size_t argc = 2;
  napi_value args[2];
  NAPI_CHECK(napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
  
  size_t len_a, len_b;
  double* a = GetFloat64ArrayData(env, args[0], &len_a);
  double* b = GetFloat64ArrayData(env, args[1], &len_b);
  
  if (!a || !b) {
    NAPI_THROW(env, "Arguments must be Float64Arrays");
  }
  
  size_t len = len_a < len_b ? len_a : len_b;
  int n = (int)len;
  vvrec(b, a, &n);
  
  return args[1];
}

// Inverse square root: b = 1/sqrt(a)
static napi_value VectorInverseSqrt(napi_env env, napi_callback_info info) {
  size_t argc = 2;
  napi_value args[2];
  NAPI_CHECK(napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
  
  size_t len_a, len_b;
  double* a = GetFloat64ArrayData(env, args[0], &len_a);
  double* b = GetFloat64ArrayData(env, args[1], &len_b);
  
  if (!a || !b) {
    NAPI_THROW(env, "Arguments must be Float64Arrays");
  }
  
  size_t len = len_a < len_b ? len_a : len_b;
  int n = (int)len;
  vvrsqrt(b, a, &n);
  
  return args[1];
}

// Hyperbolic sine
static napi_value VectorSinh(napi_env env, napi_callback_info info) {
  size_t argc = 2;
  napi_value args[2];
  NAPI_CHECK(napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
  
  size_t len_a, len_b;
  double* a = GetFloat64ArrayData(env, args[0], &len_a);
  double* b = GetFloat64ArrayData(env, args[1], &len_b);
  
  if (!a || !b) {
    NAPI_THROW(env, "Arguments must be Float64Arrays");
  }
  
  size_t len = len_a < len_b ? len_a : len_b;
  int n = (int)len;
  vvsinh(b, a, &n);
  
  return args[1];
}

// Hyperbolic cosine
static napi_value VectorCosh(napi_env env, napi_callback_info info) {
  size_t argc = 2;
  napi_value args[2];
  NAPI_CHECK(napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
  
  size_t len_a, len_b;
  double* a = GetFloat64ArrayData(env, args[0], &len_a);
  double* b = GetFloat64ArrayData(env, args[1], &len_b);
  
  if (!a || !b) {
    NAPI_THROW(env, "Arguments must be Float64Arrays");
  }
  
  size_t len = len_a < len_b ? len_a : len_b;
  int n = (int)len;
  vvcosh(b, a, &n);
  
  return args[1];
}

// Hyperbolic tangent
static napi_value VectorTanh(napi_env env, napi_callback_info info) {
  size_t argc = 2;
  napi_value args[2];
  NAPI_CHECK(napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
  
  size_t len_a, len_b;
  double* a = GetFloat64ArrayData(env, args[0], &len_a);
  double* b = GetFloat64ArrayData(env, args[1], &len_b);
  
  if (!a || !b) {
    NAPI_THROW(env, "Arguments must be Float64Arrays");
  }
  
  size_t len = len_a < len_b ? len_a : len_b;
  int n = (int)len;
  vvtanh(b, a, &n);
  
  return args[1];
}

// Inverse trig functions
static napi_value VectorAsin(napi_env env, napi_callback_info info) {
  size_t argc = 2;
  napi_value args[2];
  NAPI_CHECK(napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
  
  size_t len_a, len_b;
  double* a = GetFloat64ArrayData(env, args[0], &len_a);
  double* b = GetFloat64ArrayData(env, args[1], &len_b);
  
  if (!a || !b) {
    NAPI_THROW(env, "Arguments must be Float64Arrays");
  }
  
  size_t len = len_a < len_b ? len_a : len_b;
  int n = (int)len;
  vvasin(b, a, &n);
  
  return args[1];
}

static napi_value VectorAcos(napi_env env, napi_callback_info info) {
  size_t argc = 2;
  napi_value args[2];
  NAPI_CHECK(napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
  
  size_t len_a, len_b;
  double* a = GetFloat64ArrayData(env, args[0], &len_a);
  double* b = GetFloat64ArrayData(env, args[1], &len_b);
  
  if (!a || !b) {
    NAPI_THROW(env, "Arguments must be Float64Arrays");
  }
  
  size_t len = len_a < len_b ? len_a : len_b;
  int n = (int)len;
  vvacos(b, a, &n);
  
  return args[1];
}

static napi_value VectorAtan(napi_env env, napi_callback_info info) {
  size_t argc = 2;
  napi_value args[2];
  NAPI_CHECK(napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
  
  size_t len_a, len_b;
  double* a = GetFloat64ArrayData(env, args[0], &len_a);
  double* b = GetFloat64ArrayData(env, args[1], &len_b);
  
  if (!a || !b) {
    NAPI_THROW(env, "Arguments must be Float64Arrays");
  }
  
  size_t len = len_a < len_b ? len_a : len_b;
  int n = (int)len;
  vvatan(b, a, &n);
  
  return args[1];
}

// Atan2: c = atan2(a, b)
static napi_value VectorAtan2(napi_env env, napi_callback_info info) {
  size_t argc = 3;
  napi_value args[3];
  NAPI_CHECK(napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
  
  size_t len_a, len_b, len_c;
  double* a = GetFloat64ArrayData(env, args[0], &len_a);
  double* b = GetFloat64ArrayData(env, args[1], &len_b);
  double* c = GetFloat64ArrayData(env, args[2], &len_c);
  
  if (!a || !b || !c) {
    NAPI_THROW(env, "Arguments must be Float64Arrays");
  }
  
  size_t len = len_a;
  if (len_b < len) len = len_b;
  if (len_c < len) len = len_c;
  int n = (int)len;
  vvatan2(c, a, b, &n);
  
  return args[2];
}

// Ceiling
static napi_value VectorCeil(napi_env env, napi_callback_info info) {
  size_t argc = 2;
  napi_value args[2];
  NAPI_CHECK(napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
  
  size_t len_a, len_b;
  double* a = GetFloat64ArrayData(env, args[0], &len_a);
  double* b = GetFloat64ArrayData(env, args[1], &len_b);
  
  if (!a || !b) {
    NAPI_THROW(env, "Arguments must be Float64Arrays");
  }
  
  size_t len = len_a < len_b ? len_a : len_b;
  int n = (int)len;
  vvceil(b, a, &n);
  
  return args[1];
}

// Floor
static napi_value VectorFloor(napi_env env, napi_callback_info info) {
  size_t argc = 2;
  napi_value args[2];
  NAPI_CHECK(napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
  
  size_t len_a, len_b;
  double* a = GetFloat64ArrayData(env, args[0], &len_a);
  double* b = GetFloat64ArrayData(env, args[1], &len_b);
  
  if (!a || !b) {
    NAPI_THROW(env, "Arguments must be Float64Arrays");
  }
  
  size_t len = len_a < len_b ? len_a : len_b;
  int n = (int)len;
  vvfloor(b, a, &n);
  
  return args[1];
}

// Truncate (round toward zero)
static napi_value VectorTrunc(napi_env env, napi_callback_info info) {
  size_t argc = 2;
  napi_value args[2];
  NAPI_CHECK(napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
  
  size_t len_a, len_b;
  double* a = GetFloat64ArrayData(env, args[0], &len_a);
  double* b = GetFloat64ArrayData(env, args[1], &len_b);
  
  if (!a || !b) {
    NAPI_THROW(env, "Arguments must be Float64Arrays");
  }
  
  size_t len = len_a < len_b ? len_a : len_b;
  int n = (int)len;
  vvint(b, a, &n);
  
  return args[1];
}

// Copysign: c = copysign(a, b) - magnitude of a with sign of b
static napi_value VectorCopysign(napi_env env, napi_callback_info info) {
  size_t argc = 3;
  napi_value args[3];
  NAPI_CHECK(napi_get_cb_info(env, info, &argc, args, nullptr, nullptr));
  
  size_t len_a, len_b, len_c;
  double* a = GetFloat64ArrayData(env, args[0], &len_a);
  double* b = GetFloat64ArrayData(env, args[1], &len_b);
  double* c = GetFloat64ArrayData(env, args[2], &len_c);
  
  if (!a || !b || !c) {
    NAPI_THROW(env, "Arguments must be Float64Arrays");
  }
  
  size_t len = len_a;
  if (len_b < len) len = len_b;
  if (len_c < len) len = len_c;
  int n = (int)len;
  vvcopysign(c, b, a, &n);
  
  return args[2];
}

// Module initialization
static napi_value Init(napi_env env, napi_value exports) {
  napi_property_descriptor props[] = {
    // Matrix operations
    {"matmul", nullptr, MatMulDouble, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"matmulFloat", nullptr, MatMulFloat, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"matvec", nullptr, MatVecMul, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"transpose", nullptr, MatrixTranspose, nullptr, nullptr, nullptr, napi_default, nullptr},
    
    // BLAS operations
    {"axpy", nullptr, AXPY, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"copy", nullptr, VectorCopy, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"swap", nullptr, VectorSwap, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"norm", nullptr, VectorNorm, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"abssum", nullptr, VectorAbsSum, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"maxAbsIndex", nullptr, VectorMaxAbsIndex, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"rot", nullptr, VectorRotation, nullptr, nullptr, nullptr, napi_default, nullptr},
    
    // Vector arithmetic
    {"dot", nullptr, DotProduct, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"vadd", nullptr, VectorAdd, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"vsub", nullptr, VectorSub, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"vmul", nullptr, VectorMul, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"vdiv", nullptr, VectorDiv, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"vscale", nullptr, VectorScale, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"vneg", nullptr, VectorNegate, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"vaddScalar", nullptr, VectorAddScalar, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"vma", nullptr, VectorMultiplyAdd, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"vmsa", nullptr, VectorMultiplyScalarAdd, nullptr, nullptr, nullptr, napi_default, nullptr},
    
    // Vector functions
    {"vabs", nullptr, VectorAbs, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"vsquare", nullptr, VectorSquare, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"vsqrt", nullptr, VectorSqrt, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"normalize", nullptr, VectorNormalize, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"vreverse", nullptr, VectorReverse, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"vfill", nullptr, VectorFill, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"vramp", nullptr, VectorRamp, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"vlerp", nullptr, VectorLinearInterpolate, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"vclear", nullptr, VectorClear, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"vlimit", nullptr, VectorLimit, nullptr, nullptr, nullptr, napi_default, nullptr},
    
    // Trigonometric
    {"vsin", nullptr, VectorSin, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"vcos", nullptr, VectorCos, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"vtan", nullptr, VectorTan, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"vasin", nullptr, VectorAsin, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"vacos", nullptr, VectorAcos, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"vatan", nullptr, VectorAtan, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"vatan2", nullptr, VectorAtan2, nullptr, nullptr, nullptr, napi_default, nullptr},
    
    // Hyperbolic
    {"vsinh", nullptr, VectorSinh, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"vcosh", nullptr, VectorCosh, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"vtanh", nullptr, VectorTanh, nullptr, nullptr, nullptr, napi_default, nullptr},
    
    // Exponential/Logarithmic
    {"vexp", nullptr, VectorExp, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"vlog", nullptr, VectorLog, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"vlog10", nullptr, VectorLog10, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"vpow", nullptr, VectorPow, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"vreciprocal", nullptr, VectorReciprocal, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"vrsqrt", nullptr, VectorInverseSqrt, nullptr, nullptr, nullptr, napi_default, nullptr},
    
    // Rounding
    {"vceil", nullptr, VectorCeil, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"vfloor", nullptr, VectorFloor, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"vtrunc", nullptr, VectorTrunc, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"vcopysign", nullptr, VectorCopysign, nullptr, nullptr, nullptr, napi_default, nullptr},
    
    // Clipping/Thresholding
    {"vclip", nullptr, VectorClip, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"vthreshold", nullptr, VectorThreshold, nullptr, nullptr, nullptr, napi_default, nullptr},
    
    // Reductions
    {"sum", nullptr, VectorSum, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"mean", nullptr, VectorMean, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"max", nullptr, VectorMax, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"min", nullptr, VectorMin, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"minmax", nullptr, VectorMinMax, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"rms", nullptr, VectorRMS, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"variance", nullptr, VectorVariance, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"stddev", nullptr, VectorStdDev, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"sumOfSquares", nullptr, SumOfSquares, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"meanMagnitude", nullptr, MeanMagnitude, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"meanSquare", nullptr, MeanSquare, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"maxMagnitude", nullptr, VectorMaxMagnitude, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"minMagnitude", nullptr, VectorMinMagnitude, nullptr, nullptr, nullptr, napi_default, nullptr},
    
    // Distance metrics
    {"euclidean", nullptr, EuclideanDistance, nullptr, nullptr, nullptr, napi_default, nullptr},
    
    // Signal processing
    {"fft", nullptr, FFT, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"ifft", nullptr, IFFT, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"conv", nullptr, Convolve, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"xcorr", nullptr, CrossCorrelation, nullptr, nullptr, nullptr, napi_default, nullptr},
    
    // Window functions
    {"hamming", nullptr, HammingWindow, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"hanning", nullptr, HanningWindow, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"blackman", nullptr, BlackmanWindow, nullptr, nullptr, nullptr, napi_default, nullptr},
    
    // Interpolation
    {"interp1d", nullptr, Interp1D, nullptr, nullptr, nullptr, napi_default, nullptr},
  };
  
  napi_define_properties(env, exports, sizeof(props) / sizeof(props[0]), props);
  
  return exports;
}

NAPI_MODULE(NODE_GYP_MODULE_NAME, Init)
