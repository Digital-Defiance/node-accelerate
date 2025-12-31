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

module.exports = {
  // Matrix operations
  matmul,
  matmulFloat,
  matvec,
  
  // BLAS operations
  axpy,
  
  // Vector arithmetic
  dot,
  vadd,
  vsub,
  vmul,
  vdiv,
  vscale,
  
  // Vector functions
  vabs,
  vsquare,
  vsqrt,
  normalize,
  
  // Reductions
  sum,
  mean,
  max,
  min,
  rms,
  
  // Distance metrics
  euclidean,
  
  // Signal processing
  fft,
  
  // Raw native bindings (for advanced use)
  _native: accelerate
};
