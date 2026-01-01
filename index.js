/**
 * Apple Accelerate Framework for Node.js
 * 
 * High-performance BLAS/vDSP operations optimized for M4 Max
 * 
 * Usage:
 *   const accelerate = require('accelerate-m4');
 *   
 *   // Matrix multiplication (100x faster than JS for large matrices)
 *   const A = new Float64Array(M * K);
 *   const B = new Float64Array(K * N);
 *   const C = new Float64Array(M * N);
 *   accelerate.matmul(A, B, C, M, K, N);
 *   
 *   // Vector operations
 *   const result = accelerate.dot(vec1, vec2);
 *   accelerate.vadd(a, b, c);  // c = a + b
 *   accelerate.vmul(a, b, c);  // c = a * b (element-wise)
 *   accelerate.vscale(a, 2.0, b);  // b = a * 2.0
 *   
 *   // Reductions
 *   const sum = accelerate.sum(vec);
 *   const mean = accelerate.mean(vec);
 *   const max = accelerate.max(vec);
 *   const min = accelerate.min(vec);
 *   
 *   // FFT (input length must be power of 2)
 *   const spectrum = accelerate.fft(signal);
 */

const path = require('path');
const os = require('os');

// Platform validation
function validatePlatform() {
  const platform = process.platform;
  const arch = process.arch;
  
  if (platform !== 'darwin') {
    const message = platform === 'linux' && arch === 'arm64'
      ? `node-accelerate requires macOS, but detected Linux ARM64.\n` +
        `This package uses Apple's Accelerate framework which is only available on macOS.\n` +
        `Linux ARM64 systems (like Raspberry Pi, AWS Graviton) are not supported.\n` +
        `\n` +
        `For Linux ARM64, consider using:\n` +
        `  - OpenBLAS: https://www.openblas.net/\n` +
        `  - Eigen: https://eigen.tuxfamily.org/\n` +
        `  - BLIS: https://github.com/flame/blis`
      : `node-accelerate requires macOS (darwin), but detected ${platform}.\n` +
        `This package uses Apple's Accelerate framework which is only available on macOS.\n` +
        `Supported platforms: macOS only (Apple Silicon or Intel)`;
    
    throw new Error(message);
  }
  
  if (arch !== 'arm64' && arch !== 'x64') {
    throw new Error(
      `node-accelerate requires ARM64 or x64 architecture, but detected ${arch}.\n` +
      `Supported architectures:\n` +
      `  - arm64 (Apple Silicon: M1/M2/M3/M4)\n` +
      `  - x64 (Intel Macs)\n` +
      `\n` +
      `Your architecture (${arch}) is not supported.`
    );
  }
  
  // Check for Accelerate framework (should always be present on macOS)
  const fs = require('fs');
  const acceleratePath = '/System/Library/Frameworks/Accelerate.framework';
  if (!fs.existsSync(acceleratePath)) {
    throw new Error(
      `Apple Accelerate framework not found at ${acceleratePath}.\n` +
      `This is unusual for macOS. Please ensure you're running on a standard macOS system.`
    );
  }
}

// Validate platform before loading native module
validatePlatform();

let accelerate;
try {
  accelerate = require('./build/Release/accelerate.node');
} catch (e) {
  // Try debug build
  try {
    accelerate = require('./build/Debug/accelerate.node');
  } catch (e2) {
    throw new Error(
      'Failed to load node-accelerate native module.\n' +
      'This usually means the module needs to be built.\n\n' +
      'To fix this, run: npm rebuild node-accelerate\n\n' +
      'Requirements:\n' +
      '  - macOS (Apple Silicon or Intel)\n' +
      '  - Xcode Command Line Tools (run: xcode-select --install)\n' +
      '  - Node.js >= 18.0.0\n\n' +
      'Original error: ' + e.message
    );
  }
}

/**
 * Matrix multiplication: C = A * B
 * @param {Float64Array} A - Matrix A (M x K, row-major)
 * @param {Float64Array} B - Matrix B (K x N, row-major)
 * @param {Float64Array} C - Output matrix C (M x N, row-major)
 * @param {number} M - Rows in A
 * @param {number} K - Columns in A / Rows in B
 * @param {number} N - Columns in B
 * @returns {Float64Array} C
 */
function matmul(A, B, C, M, K, N) {
  if (!(A instanceof Float64Array) || !(B instanceof Float64Array) || !(C instanceof Float64Array)) {
    throw new TypeError('Arguments must be Float64Arrays');
  }
  if (A.length !== M * K) throw new RangeError(`A must have ${M * K} elements`);
  if (B.length !== K * N) throw new RangeError(`B must have ${K * N} elements`);
  if (C.length !== M * N) throw new RangeError(`C must have ${M * N} elements`);
  
  return accelerate.matmul(A, B, C, M, K, N);
}

/**
 * Matrix multiplication (single precision): C = A * B
 * @param {Float32Array} A - Matrix A (M x K, row-major)
 * @param {Float32Array} B - Matrix B (K x N, row-major)
 * @param {Float32Array} C - Output matrix C (M x N, row-major)
 * @param {number} M - Rows in A
 * @param {number} K - Columns in A / Rows in B
 * @param {number} N - Columns in B
 * @returns {Float32Array} C
 */
function matmulFloat(A, B, C, M, K, N) {
  if (!(A instanceof Float32Array) || !(B instanceof Float32Array) || !(C instanceof Float32Array)) {
    throw new TypeError('Arguments must be Float32Arrays');
  }
  return accelerate.matmulFloat(A, B, C, M, K, N);
}

/**
 * Dot product of two vectors
 * @param {Float64Array} a 
 * @param {Float64Array} b 
 * @returns {number}
 */
function dot(a, b) {
  if (!(a instanceof Float64Array) || !(b instanceof Float64Array)) {
    throw new TypeError('Arguments must be Float64Arrays');
  }
  return accelerate.dot(a, b);
}

/**
 * Vector addition: c = a + b
 * @param {Float64Array} a 
 * @param {Float64Array} b 
 * @param {Float64Array} c - Output
 * @returns {Float64Array} c
 */
function vadd(a, b, c) {
  return accelerate.vadd(a, b, c);
}

/**
 * Vector subtraction: c = a - b
 * @param {Float64Array} a 
 * @param {Float64Array} b 
 * @param {Float64Array} c - Output
 * @returns {Float64Array} c
 */
function vsub(a, b, c) {
  return accelerate.vsub(a, b, c);
}

/**
 * Element-wise vector multiplication: c = a * b
 * @param {Float64Array} a 
 * @param {Float64Array} b 
 * @param {Float64Array} c - Output
 * @returns {Float64Array} c
 */
function vmul(a, b, c) {
  return accelerate.vmul(a, b, c);
}

/**
 * Element-wise vector division: c = a / b
 * @param {Float64Array} a 
 * @param {Float64Array} b 
 * @param {Float64Array} c - Output
 * @returns {Float64Array} c
 */
function vdiv(a, b, c) {
  return accelerate.vdiv(a, b, c);
}

/**
 * Vector scaling: b = a * scalar
 * @param {Float64Array} a 
 * @param {number} scalar 
 * @param {Float64Array} b - Output
 * @returns {Float64Array} b
 */
function vscale(a, scalar, b) {
  return accelerate.vscale(a, scalar, b);
}

/**
 * Sum of vector elements
 * @param {Float64Array} a 
 * @returns {number}
 */
function sum(a) {
  return accelerate.sum(a);
}

/**
 * Mean of vector elements
 * @param {Float64Array} a 
 * @returns {number}
 */
function mean(a) {
  return accelerate.mean(a);
}

/**
 * Maximum element in vector
 * @param {Float64Array} a 
 * @returns {number}
 */
function max(a) {
  return accelerate.max(a);
}

/**
 * Minimum element in vector
 * @param {Float64Array} a 
 * @returns {number}
 */
function min(a) {
  return accelerate.min(a);
}

/**
 * Fast Fourier Transform
 * @param {Float64Array} input - Real input (length must be power of 2)
 * @returns {{real: Float64Array, imag: Float64Array}} Complex output
 */
function fft(input) {
  if (!(input instanceof Float64Array)) {
    throw new TypeError('Input must be Float64Array');
  }
  const len = input.length;
  if ((len & (len - 1)) !== 0) {
    throw new RangeError('Input length must be a power of 2');
  }
  
  const interleaved = accelerate.fft(input);
  const half = len / 2;
  const real = new Float64Array(half);
  const imag = new Float64Array(half);
  
  for (let i = 0; i < half; i++) {
    real[i] = interleaved[i * 2];
    imag[i] = interleaved[i * 2 + 1];
  }
  
  return { real, imag };
}

/**
 * Matrix-vector multiplication: y = A * x
 * @param {Float64Array} A - Matrix (M × N, row-major)
 * @param {Float64Array} x - Vector (N elements)
 * @param {Float64Array} y - Output vector (M elements)
 * @param {number} M - Rows in A
 * @param {number} N - Columns in A
 * @returns {Float64Array} y
 */
function matvec(A, x, y, M, N) {
  return accelerate.matvec(A, x, y, M, N);
}

/**
 * AXPY operation: y = alpha*x + y
 * @param {number} alpha - Scalar multiplier
 * @param {Float64Array} x - Input vector
 * @param {Float64Array} y - Input/output vector
 * @returns {Float64Array} y
 */
function axpy(alpha, x, y) {
  return accelerate.axpy(alpha, x, y);
}

/**
 * Vector absolute value: b = |a|
 * @param {Float64Array} a - Input vector
 * @param {Float64Array} b - Output vector
 * @returns {Float64Array} b
 */
function vabs(a, b) {
  return accelerate.vabs(a, b);
}

/**
 * Vector square: b = a^2
 * @param {Float64Array} a - Input vector
 * @param {Float64Array} b - Output vector
 * @returns {Float64Array} b
 */
function vsquare(a, b) {
  return accelerate.vsquare(a, b);
}

/**
 * Vector square root: b = sqrt(a)
 * @param {Float64Array} a - Input vector
 * @param {Float64Array} b - Output vector
 * @returns {Float64Array} b
 */
function vsqrt(a, b) {
  return accelerate.vsqrt(a, b);
}

/**
 * Normalize vector to unit length
 * @param {Float64Array} a - Input vector
 * @param {Float64Array} b - Output vector (unit vector)
 * @returns {Float64Array} b
 */
function normalize(a, b) {
  return accelerate.normalize(a, b);
}

/**
 * Euclidean distance between two vectors
 * @param {Float64Array} a - First vector
 * @param {Float64Array} b - Second vector
 * @returns {number} Distance
 */
function euclidean(a, b) {
  return accelerate.euclidean(a, b);
}

/**
 * Root Mean Square of vector
 * @param {Float64Array} a - Input vector
 * @returns {number} RMS value
 */
function rms(a) {
  return accelerate.rms(a);
}

/**
 * Variance of vector elements
 * @param {Float64Array} a - Input vector
 * @returns {number} Variance
 */
function variance(a) {
  return accelerate.variance(a);
}

/**
 * Standard deviation of vector elements
 * @param {Float64Array} a - Input vector
 * @returns {number} Standard deviation
 */
function stddev(a) {
  return accelerate.stddev(a);
}

/**
 * Min and max of vector elements
 * @param {Float64Array} a - Input vector
 * @returns {{min: number, max: number}} Object with min and max
 */
function minmax(a) {
  return accelerate.minmax(a);
}

/**
 * Vector sine: b = sin(a)
 * @param {Float64Array} a - Input vector
 * @param {Float64Array} b - Output vector
 * @returns {Float64Array} b
 */
function vsin(a, b) {
  return accelerate.vsin(a, b);
}

/**
 * Vector cosine: b = cos(a)
 * @param {Float64Array} a - Input vector
 * @param {Float64Array} b - Output vector
 * @returns {Float64Array} b
 */
function vcos(a, b) {
  return accelerate.vcos(a, b);
}

/**
 * Vector tangent: b = tan(a)
 * @param {Float64Array} a - Input vector
 * @param {Float64Array} b - Output vector
 * @returns {Float64Array} b
 */
function vtan(a, b) {
  return accelerate.vtan(a, b);
}

/**
 * Vector exponential: b = exp(a)
 * @param {Float64Array} a - Input vector
 * @param {Float64Array} b - Output vector
 * @returns {Float64Array} b
 */
function vexp(a, b) {
  return accelerate.vexp(a, b);
}

/**
 * Vector natural logarithm: b = log(a)
 * @param {Float64Array} a - Input vector
 * @param {Float64Array} b - Output vector
 * @returns {Float64Array} b
 */
function vlog(a, b) {
  return accelerate.vlog(a, b);
}

/**
 * Vector base-10 logarithm: b = log10(a)
 * @param {Float64Array} a - Input vector
 * @param {Float64Array} b - Output vector
 * @returns {Float64Array} b
 */
function vlog10(a, b) {
  return accelerate.vlog10(a, b);
}

/**
 * Vector power: c = a^b (element-wise)
 * @param {Float64Array} a - Base vector
 * @param {Float64Array} b - Exponent vector
 * @param {Float64Array} c - Output vector
 * @returns {Float64Array} c
 */
function vpow(a, b, c) {
  return accelerate.vpow(a, b, c);
}

/**
 * Vector clip: b = clip(a, min, max)
 * @param {Float64Array} a - Input vector
 * @param {Float64Array} b - Output vector
 * @param {number} min - Minimum value
 * @param {number} max - Maximum value
 * @returns {Float64Array} b
 */
function vclip(a, b, min, max) {
  return accelerate.vclip(a, b, min, max);
}

/**
 * Vector threshold: b = a where a > threshold, else 0
 * @param {Float64Array} a - Input vector
 * @param {Float64Array} b - Output vector
 * @param {number} threshold - Threshold value
 * @returns {Float64Array} b
 */
function vthreshold(a, b, threshold) {
  return accelerate.vthreshold(a, b, threshold);
}

/**
 * Convolution: result = signal * kernel
 * @param {Float64Array} signal - Input signal
 * @param {Float64Array} kernel - Convolution kernel
 * @param {Float64Array} result - Output (length = signal.length - kernel.length + 1)
 * @returns {Float64Array} result
 */
function conv(signal, kernel, result) {
  return accelerate.conv(signal, kernel, result);
}

/**
 * Cross-correlation: result = correlate(a, b)
 * @param {Float64Array} a - First signal
 * @param {Float64Array} b - Second signal
 * @param {Float64Array} result - Output (length = a.length + b.length - 1)
 * @returns {Float64Array} result
 */
function xcorr(a, b, result) {
  return accelerate.xcorr(a, b, result);
}

/**
 * Generate Hamming window
 * @param {number} length - Window length
 * @returns {Float64Array} Window coefficients
 */
function hamming(length) {
  return accelerate.hamming(length);
}

/**
 * Generate Hanning window
 * @param {number} length - Window length
 * @returns {Float64Array} Window coefficients
 */
function hanning(length) {
  return accelerate.hanning(length);
}

/**
 * Generate Blackman window
 * @param {number} length - Window length
 * @returns {Float64Array} Window coefficients
 */
function blackman(length) {
  return accelerate.blackman(length);
}

/**
 * Matrix transpose: B = A^T
 * @param {Float64Array} A - Input matrix (rows × cols, row-major)
 * @param {Float64Array} B - Output matrix (cols × rows, row-major)
 * @param {number} rows - Number of rows in A
 * @param {number} cols - Number of columns in A
 * @returns {Float64Array} B
 */
function transpose(A, B, rows, cols) {
  return accelerate.transpose(A, B, rows, cols);
}

/**
 * Inverse Fast Fourier Transform
 * @param {Float64Array} real - Real part of frequency domain
 * @param {Float64Array} imag - Imaginary part of frequency domain
 * @returns {Float64Array} Time domain signal
 */
function ifft(real, imag) {
  if (!(real instanceof Float64Array) || !(imag instanceof Float64Array)) {
    throw new TypeError('Arguments must be Float64Arrays');
  }
  if (real.length !== imag.length) {
    throw new RangeError('Real and imaginary arrays must have same length');
  }
  
  return accelerate.ifft(real, imag);
}

/**
 * Linear interpolation
 * @param {Float64Array} x - X coordinates of data points
 * @param {Float64Array} y - Y coordinates of data points
 * @param {Float64Array} xi - X coordinates to interpolate at
 * @param {Float64Array} yi - Output interpolated Y values
 * @returns {Float64Array} yi
 */
function interp1d(x, y, xi, yi) {
  return accelerate.interp1d(x, y, xi, yi);
}

/**
 * Vector reverse: b = reverse(a)
 * @param {Float64Array} a - Input vector
 * @param {Float64Array} b - Output vector
 * @returns {Float64Array} b
 */
function vreverse(a, b) {
  return accelerate.vreverse(a, b);
}

/**
 * Vector negate: b = -a
 * @param {Float64Array} a - Input vector
 * @param {Float64Array} b - Output vector
 * @returns {Float64Array} b
 */
function vneg(a, b) {
  return accelerate.vneg(a, b);
}

/**
 * Sum of squares: sum(a[i]^2)
 * @param {Float64Array} a - Input vector
 * @returns {number} Sum of squares
 */
function sumOfSquares(a) {
  return accelerate.sumOfSquares(a);
}

/**
 * Mean magnitude: mean(|a[i]|)
 * @param {Float64Array} a - Input vector
 * @returns {number} Mean magnitude
 */
function meanMagnitude(a) {
  return accelerate.meanMagnitude(a);
}

/**
 * Mean square: mean(a[i]^2)
 * @param {Float64Array} a - Input vector
 * @returns {number} Mean square
 */
function meanSquare(a) {
  return accelerate.meanSquare(a);
}

// ============================================================================
// ADDITIONAL BLAS OPERATIONS
// ============================================================================

/**
 * Vector copy: y = x
 * @param {Float64Array} x - Source vector
 * @param {Float64Array} y - Destination vector
 * @returns {Float64Array} y
 */
function copy(x, y) {
  return accelerate.copy(x, y);
}

/**
 * Vector swap: exchange x and y
 * @param {Float64Array} x - First vector
 * @param {Float64Array} y - Second vector
 * @returns {Float64Array} x
 */
function swap(x, y) {
  return accelerate.swap(x, y);
}

/**
 * Vector norm (L2 norm / Euclidean length): ||x||
 * @param {Float64Array} x - Input vector
 * @returns {number} L2 norm
 */
function norm(x) {
  return accelerate.norm(x);
}

/**
 * Sum of absolute values: sum(|x[i]|)
 * @param {Float64Array} x - Input vector
 * @returns {number} Sum of absolute values
 */
function abssum(x) {
  return accelerate.abssum(x);
}

/**
 * Index of maximum absolute value
 * @param {Float64Array} x - Input vector
 * @returns {number} Index of element with maximum absolute value
 */
function maxAbsIndex(x) {
  return accelerate.maxAbsIndex(x);
}

/**
 * Apply Givens rotation
 * @param {Float64Array} x - First vector
 * @param {Float64Array} y - Second vector
 * @param {number} c - Cosine of rotation angle
 * @param {number} s - Sine of rotation angle
 * @returns {Float64Array} x
 */
function rot(x, y, c, s) {
  return accelerate.rot(x, y, c, s);
}

// ============================================================================
// ADDITIONAL vDSP OPERATIONS
// ============================================================================

/**
 * Fill vector with scalar value
 * @param {number} scalar - Value to fill with
 * @param {Float64Array} vec - Output vector
 * @returns {Float64Array} vec
 */
function vfill(scalar, vec) {
  return accelerate.vfill(scalar, vec);
}

/**
 * Generate linear ramp: vec[i] = start + i * step
 * @param {number} start - Starting value
 * @param {number} step - Step size
 * @param {Float64Array} vec - Output vector
 * @returns {Float64Array} vec
 */
function vramp(start, step, vec) {
  return accelerate.vramp(start, step, vec);
}

/**
 * Add scalar to vector: c = a + scalar
 * @param {Float64Array} a - Input vector
 * @param {number} scalar - Scalar to add
 * @param {Float64Array} c - Output vector
 * @returns {Float64Array} c
 */
function vaddScalar(a, scalar, c) {
  return accelerate.vaddScalar(a, scalar, c);
}

/**
 * Multiply and add: d = (a * b) + c
 * @param {Float64Array} a - First vector
 * @param {Float64Array} b - Second vector
 * @param {Float64Array} c - Third vector
 * @param {Float64Array} d - Output vector
 * @returns {Float64Array} d
 */
function vma(a, b, c, d) {
  return accelerate.vma(a, b, c, d);
}

/**
 * Multiply and scalar add: d = (a * b) + c (scalar)
 * @param {Float64Array} a - First vector
 * @param {Float64Array} b - Second vector
 * @param {number} c - Scalar to add
 * @param {Float64Array} d - Output vector
 * @returns {Float64Array} d
 */
function vmsa(a, b, c, d) {
  return accelerate.vmsa(a, b, c, d);
}

/**
 * Linear interpolation between vectors: c = a + t * (b - a)
 * @param {Float64Array} a - Start vector
 * @param {Float64Array} b - End vector
 * @param {number} t - Interpolation parameter (0 to 1)
 * @param {Float64Array} c - Output vector
 * @returns {Float64Array} c
 */
function vlerp(a, b, t, c) {
  return accelerate.vlerp(a, b, t, c);
}

/**
 * Clear vector (set to zero)
 * @param {Float64Array} vec - Vector to clear
 * @returns {Float64Array} vec
 */
function vclear(vec) {
  return accelerate.vclear(vec);
}

/**
 * Limit vector values (saturate): c = limit(a, low, high)
 * @param {Float64Array} a - Input vector
 * @param {number} low - Lower limit
 * @param {number} high - Upper limit
 * @param {Float64Array} c - Output vector
 * @returns {Float64Array} c
 */
function vlimit(a, low, high, c) {
  return accelerate.vlimit(a, low, high, c);
}

/**
 * Maximum magnitude in vector: max(|a[i]|)
 * @param {Float64Array} a - Input vector
 * @returns {number} Maximum magnitude
 */
function maxMagnitude(a) {
  return accelerate.maxMagnitude(a);
}

/**
 * Minimum magnitude in vector: min(|a[i]|)
 * @param {Float64Array} a - Input vector
 * @returns {number} Minimum magnitude
 */
function minMagnitude(a) {
  return accelerate.minMagnitude(a);
}

// ============================================================================
// MORE MATH FUNCTIONS
// ============================================================================

/**
 * Reciprocal: b = 1/a
 * @param {Float64Array} a - Input vector
 * @param {Float64Array} b - Output vector
 * @returns {Float64Array} b
 */
function vreciprocal(a, b) {
  return accelerate.vreciprocal(a, b);
}

/**
 * Inverse square root: b = 1/sqrt(a)
 * @param {Float64Array} a - Input vector
 * @param {Float64Array} b - Output vector
 * @returns {Float64Array} b
 */
function vrsqrt(a, b) {
  return accelerate.vrsqrt(a, b);
}

/**
 * Hyperbolic sine: b = sinh(a)
 * @param {Float64Array} a - Input vector
 * @param {Float64Array} b - Output vector
 * @returns {Float64Array} b
 */
function vsinh(a, b) {
  return accelerate.vsinh(a, b);
}

/**
 * Hyperbolic cosine: b = cosh(a)
 * @param {Float64Array} a - Input vector
 * @param {Float64Array} b - Output vector
 * @returns {Float64Array} b
 */
function vcosh(a, b) {
  return accelerate.vcosh(a, b);
}

/**
 * Hyperbolic tangent: b = tanh(a)
 * @param {Float64Array} a - Input vector
 * @param {Float64Array} b - Output vector
 * @returns {Float64Array} b
 */
function vtanh(a, b) {
  return accelerate.vtanh(a, b);
}

/**
 * Inverse sine: b = asin(a)
 * @param {Float64Array} a - Input vector
 * @param {Float64Array} b - Output vector
 * @returns {Float64Array} b
 */
function vasin(a, b) {
  return accelerate.vasin(a, b);
}

/**
 * Inverse cosine: b = acos(a)
 * @param {Float64Array} a - Input vector
 * @param {Float64Array} b - Output vector
 * @returns {Float64Array} b
 */
function vacos(a, b) {
  return accelerate.vacos(a, b);
}

/**
 * Inverse tangent: b = atan(a)
 * @param {Float64Array} a - Input vector
 * @param {Float64Array} b - Output vector
 * @returns {Float64Array} b
 */
function vatan(a, b) {
  return accelerate.vatan(a, b);
}

/**
 * Two-argument inverse tangent: c = atan2(a, b)
 * @param {Float64Array} a - Y coordinates
 * @param {Float64Array} b - X coordinates
 * @param {Float64Array} c - Output vector (angles)
 * @returns {Float64Array} c
 */
function vatan2(a, b, c) {
  return accelerate.vatan2(a, b, c);
}

/**
 * Ceiling: b = ceil(a)
 * @param {Float64Array} a - Input vector
 * @param {Float64Array} b - Output vector
 * @returns {Float64Array} b
 */
function vceil(a, b) {
  return accelerate.vceil(a, b);
}

/**
 * Floor: b = floor(a)
 * @param {Float64Array} a - Input vector
 * @param {Float64Array} b - Output vector
 * @returns {Float64Array} b
 */
function vfloor(a, b) {
  return accelerate.vfloor(a, b);
}

/**
 * Truncate (round toward zero): b = trunc(a)
 * @param {Float64Array} a - Input vector
 * @param {Float64Array} b - Output vector
 * @returns {Float64Array} b
 */
function vtrunc(a, b) {
  return accelerate.vtrunc(a, b);
}

/**
 * Copy sign: c = copysign(a, b) - magnitude of a with sign of b
 * @param {Float64Array} a - Magnitude source
 * @param {Float64Array} b - Sign source
 * @param {Float64Array} c - Output vector
 * @returns {Float64Array} c
 */
function vcopysign(a, b, c) {
  return accelerate.vcopysign(a, b, c);
}

module.exports = {
  // Matrix operations
  matmul,
  matmulFloat,
  matvec,
  transpose,
  
  // BLAS operations
  axpy,
  copy,
  swap,
  norm,
  abssum,
  maxAbsIndex,
  rot,
  
  // Vector arithmetic
  dot,
  vadd,
  vsub,
  vmul,
  vdiv,
  vscale,
  vneg,
  vaddScalar,
  vma,
  vmsa,
  
  // Vector functions
  vabs,
  vsquare,
  vsqrt,
  normalize,
  vreverse,
  vfill,
  vramp,
  vlerp,
  vclear,
  vlimit,
  
  // Trigonometric
  vsin,
  vcos,
  vtan,
  vasin,
  vacos,
  vatan,
  vatan2,
  
  // Hyperbolic
  vsinh,
  vcosh,
  vtanh,
  
  // Exponential/Logarithmic
  vexp,
  vlog,
  vlog10,
  vpow,
  vreciprocal,
  vrsqrt,
  
  // Rounding
  vceil,
  vfloor,
  vtrunc,
  vcopysign,
  
  // Clipping/Thresholding
  vclip,
  vthreshold,
  
  // Reductions
  sum,
  mean,
  max,
  min,
  minmax,
  rms,
  variance,
  stddev,
  sumOfSquares,
  meanMagnitude,
  meanSquare,
  maxMagnitude,
  minMagnitude,
  
  // Distance metrics
  euclidean,
  
  // Signal processing
  fft,
  ifft,
  conv,
  xcorr,
  
  // Window functions
  hamming,
  hanning,
  blackman,
  
  // Interpolation
  interp1d,
  
  // Raw native bindings (for advanced use)
  _native: accelerate
};
