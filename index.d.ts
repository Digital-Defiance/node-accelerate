/**
 * TypeScript definitions for node-accelerate
 * Apple Accelerate framework bindings for Node.js
 */

/**
 * Perform matrix multiplication: C = A × B
 * Uses Apple's BLAS (Basic Linear Algebra Subprograms) for hardware-accelerated computation
 * 
 * @param A - First matrix (M × K) as Float64Array in row-major order
 * @param B - Second matrix (K × N) as Float64Array in row-major order
 * @param C - Output matrix (M × N) as Float64Array in row-major order
 * @param M - Number of rows in A and C
 * @param K - Number of columns in A and rows in B
 * @param N - Number of columns in B and C
 * @returns The output matrix C
 * 
 * @example
 * const M = 100, K = 100, N = 100;
 * const A = new Float64Array(M * K);
 * const B = new Float64Array(K * N);
 * const C = new Float64Array(M * N);
 * 
 * // Fill A and B with data
 * for (let i = 0; i < A.length; i++) A[i] = Math.random();
 * for (let i = 0; i < B.length; i++) B[i] = Math.random();
 * 
 * // C = A × B (hardware-accelerated)
 * accelerate.matmul(A, B, C, M, K, N);
 */
export function matmul(
  A: Float64Array,
  B: Float64Array,
  C: Float64Array,
  M: number,
  K: number,
  N: number
): Float64Array;

/**
 * Compute dot product of two vectors: result = sum(a[i] * b[i])
 * Uses Apple's vDSP for hardware-accelerated computation
 * 
 * @param a - First vector as Float64Array
 * @param b - Second vector as Float64Array (must be same length as a)
 * @returns The dot product as a number
 * 
 * @example
 * const a = new Float64Array([1, 2, 3, 4]);
 * const b = new Float64Array([5, 6, 7, 8]);
 * const result = accelerate.dot(a, b); // 70
 */
export function dot(a: Float64Array, b: Float64Array): number;

/**
 * Compute sum of all elements in a vector: result = sum(vec[i])
 * Uses Apple's vDSP for hardware-accelerated computation
 * 
 * @param vec - Input vector as Float64Array
 * @returns The sum of all elements
 * 
 * @example
 * const vec = new Float64Array([1, 2, 3, 4, 5]);
 * const result = accelerate.sum(vec); // 15
 */
export function sum(vec: Float64Array): number;

/**
 * Compute mean (average) of all elements in a vector
 * Uses Apple's vDSP for hardware-accelerated computation
 * 
 * @param vec - Input vector as Float64Array
 * @returns The mean of all elements
 * 
 * @example
 * const vec = new Float64Array([1, 2, 3, 4, 5]);
 * const result = accelerate.mean(vec); // 3
 */
export function mean(vec: Float64Array): number;

/**
 * Element-wise vector addition: out[i] = a[i] + b[i]
 * Uses Apple's vDSP for hardware-accelerated computation
 * 
 * @param a - First vector as Float64Array
 * @param b - Second vector as Float64Array (must be same length as a)
 * @param out - Output vector as Float64Array (must be same length as a)
 * @returns The output vector
 * 
 * @example
 * const a = new Float64Array([1, 2, 3]);
 * const b = new Float64Array([4, 5, 6]);
 * const out = new Float64Array(3);
 * accelerate.vadd(a, b, out); // out = [5, 7, 9]
 */
export function vadd(
  a: Float64Array,
  b: Float64Array,
  out: Float64Array
): Float64Array;

/**
 * Element-wise vector multiplication: out[i] = a[i] * b[i]
 * Uses Apple's vDSP for hardware-accelerated computation
 * 
 * @param a - First vector as Float64Array
 * @param b - Second vector as Float64Array (must be same length as a)
 * @param out - Output vector as Float64Array (must be same length as a)
 * @returns The output vector
 * 
 * @example
 * const a = new Float64Array([2, 3, 4]);
 * const b = new Float64Array([5, 6, 7]);
 * const out = new Float64Array(3);
 * accelerate.vmul(a, b, out); // out = [10, 18, 28]
 */
export function vmul(
  a: Float64Array,
  b: Float64Array,
  out: Float64Array
): Float64Array;

/**
 * Vector scaling: out[i] = vec[i] * scalar
 * Uses Apple's vDSP for hardware-accelerated computation
 * 
 * @param vec - Input vector as Float64Array
 * @param scalar - Scalar value to multiply by
 * @param out - Output vector as Float64Array (must be same length as vec)
 * @returns The output vector
 * 
 * @example
 * const vec = new Float64Array([1, 2, 3]);
 * const out = new Float64Array(3);
 * accelerate.vscale(vec, 2.0, out); // out = [2, 4, 6]
 */
export function vscale(
  vec: Float64Array,
  scalar: number,
  out: Float64Array
): Float64Array;

/**
 * Find maximum value in a vector
 * Uses Apple's vDSP for hardware-accelerated computation
 * 
 * @param vec - Input vector as Float64Array
 * @returns The maximum value
 * 
 * @example
 * const vec = new Float64Array([1, 5, 3, 2, 4]);
 * const result = accelerate.max(vec); // 5
 */
export function max(vec: Float64Array): number;

/**
 * Find minimum value in a vector
 * Uses Apple's vDSP for hardware-accelerated computation
 * 
 * @param vec - Input vector as Float64Array
 * @returns The minimum value
 * 
 * @example
 * const vec = new Float64Array([1, 5, 3, 2, 4]);
 * const result = accelerate.min(vec); // 1
 */
export function min(vec: Float64Array): number;

/**
 * Fast Fourier Transform (FFT) of a real signal
 * Uses Apple's vDSP for hardware-accelerated computation
 * 
 * @param signal - Input signal as Float64Array (length must be power of 2)
 * @returns Object with real and imaginary components of the frequency spectrum
 * 
 * @example
 * const signal = new Float64Array(1024);
 * for (let i = 0; i < signal.length; i++) {
 *   signal[i] = Math.sin(2 * Math.PI * i / signal.length);
 * }
 * const spectrum = accelerate.fft(signal);
 * console.log(spectrum.real, spectrum.imag);
 */
export function fft(signal: Float64Array): {
  real: Float64Array;
  imag: Float64Array;
};

/**
 * Matrix-vector multiplication: y = A × x
 * Uses Apple's BLAS for hardware-accelerated computation
 * 
 * @param A - Matrix (M × N) as Float64Array in row-major order
 * @param x - Vector (N elements) as Float64Array
 * @param y - Output vector (M elements) as Float64Array
 * @param M - Number of rows in A
 * @param N - Number of columns in A
 * @returns The output vector y
 */
export function matvec(
  A: Float64Array,
  x: Float64Array,
  y: Float64Array,
  M: number,
  N: number
): Float64Array;

/**
 * AXPY operation: y = alpha*x + y
 * Uses Apple's BLAS for hardware-accelerated computation
 * 
 * @param alpha - Scalar multiplier
 * @param x - Input vector as Float64Array
 * @param y - Input/output vector as Float64Array
 * @returns The output vector y
 */
export function axpy(
  alpha: number,
  x: Float64Array,
  y: Float64Array
): Float64Array;

/**
 * Vector absolute value: b = |a|
 * Uses Apple's vDSP for hardware-accelerated computation
 * 
 * @param a - Input vector as Float64Array
 * @param b - Output vector as Float64Array
 * @returns The output vector b
 */
export function vabs(a: Float64Array, b: Float64Array): Float64Array;

/**
 * Vector square: b = a^2 (element-wise)
 * Uses Apple's vDSP for hardware-accelerated computation
 * 
 * @param a - Input vector as Float64Array
 * @param b - Output vector as Float64Array
 * @returns The output vector b
 */
export function vsquare(a: Float64Array, b: Float64Array): Float64Array;

/**
 * Vector square root: b = sqrt(a) (element-wise)
 * Uses Apple's vForce for hardware-accelerated computation
 * 
 * @param a - Input vector as Float64Array
 * @param b - Output vector as Float64Array
 * @returns The output vector b
 */
export function vsqrt(a: Float64Array, b: Float64Array): Float64Array;

/**
 * Normalize vector to unit length: b = a / ||a||
 * Uses Apple's vDSP for hardware-accelerated computation
 * 
 * @param a - Input vector as Float64Array
 * @param b - Output vector as Float64Array (unit vector)
 * @returns The output vector b
 */
export function normalize(a: Float64Array, b: Float64Array): Float64Array;

/**
 * Euclidean distance between two vectors: sqrt(sum((a - b)^2))
 * Uses Apple's vDSP for hardware-accelerated computation
 * 
 * @param a - First vector as Float64Array
 * @param b - Second vector as Float64Array
 * @returns The Euclidean distance
 */
export function euclidean(a: Float64Array, b: Float64Array): number;

/**
 * Root Mean Square of vector: sqrt(sum(a^2) / n)
 * Uses Apple's vDSP for hardware-accelerated computation
 * 
 * @param a - Input vector as Float64Array
 * @returns The RMS value
 */
export function rms(a: Float64Array): number;

/**
 * All exported functions
 */
declare const accelerate: {
  // Matrix operations
  matmul: typeof matmul;
  matvec: typeof matvec;
  
  // BLAS operations
  axpy: typeof axpy;
  
  // Vector arithmetic
  dot: typeof dot;
  sum: typeof sum;
  mean: typeof mean;
  vadd: typeof vadd;
  vsub: typeof vsub;
  vmul: typeof vmul;
  vdiv: typeof vdiv;
  
  // Vector functions
  vabs: typeof vabs;
  vsquare: typeof vsquare;
  vsqrt: typeof vsqrt;
  normalize: typeof normalize;
  
  // Reductions
  max: typeof max;
  min: typeof min;
  rms: typeof rms;
  
  // Distance metrics
  euclidean: typeof euclidean;
  
  // Signal processing
  fft: typeof fft;
};

export default accelerate;
