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

// Module initialization
static napi_value Init(napi_env env, napi_value exports) {
  napi_property_descriptor props[] = {
    // Matrix operations
    {"matmul", nullptr, MatMulDouble, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"matmulFloat", nullptr, MatMulFloat, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"matvec", nullptr, MatVecMul, nullptr, nullptr, nullptr, napi_default, nullptr},
    
    // BLAS operations
    {"axpy", nullptr, AXPY, nullptr, nullptr, nullptr, napi_default, nullptr},
    
    // Vector arithmetic
    {"dot", nullptr, DotProduct, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"vadd", nullptr, VectorAdd, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"vsub", nullptr, VectorSub, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"vmul", nullptr, VectorMul, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"vdiv", nullptr, VectorDiv, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"vscale", nullptr, VectorScale, nullptr, nullptr, nullptr, napi_default, nullptr},
    
    // Vector functions
    {"vabs", nullptr, VectorAbs, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"vsquare", nullptr, VectorSquare, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"vsqrt", nullptr, VectorSqrt, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"normalize", nullptr, VectorNormalize, nullptr, nullptr, nullptr, napi_default, nullptr},
    
    // Reductions
    {"sum", nullptr, VectorSum, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"mean", nullptr, VectorMean, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"max", nullptr, VectorMax, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"min", nullptr, VectorMin, nullptr, nullptr, nullptr, napi_default, nullptr},
    {"rms", nullptr, VectorRMS, nullptr, nullptr, nullptr, napi_default, nullptr},
    
    // Distance metrics
    {"euclidean", nullptr, EuclideanDistance, nullptr, nullptr, nullptr, napi_default, nullptr},
    
    // Signal processing
    {"fft", nullptr, FFT, nullptr, nullptr, nullptr, napi_default, nullptr},
  };
  
  napi_define_properties(env, exports, sizeof(props) / sizeof(props[0]), props);
  
  return exports;
}

NAPI_MODULE(NODE_GYP_MODULE_NAME, Init)
