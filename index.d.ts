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
 * Variance of vector elements
 * Uses Apple's vDSP for hardware-accelerated computation
 * 
 * @param a - Input vector as Float64Array
 * @returns The variance
 */
export function variance(a: Float64Array): number;

/**
 * Standard deviation of vector elements
 * Uses Apple's vDSP for hardware-accelerated computation
 * 
 * @param a - Input vector as Float64Array
 * @returns The standard deviation
 */
export function stddev(a: Float64Array): number;

/**
 * Find both minimum and maximum values in a vector
 * Uses Apple's vDSP for hardware-accelerated computation
 * 
 * @param vec - Input vector as Float64Array
 * @returns Object with min and max values
 */
export function minmax(vec: Float64Array): { min: number; max: number };

/**
 * Element-wise sine: b = sin(a)
 * Uses Apple's vForce for hardware-accelerated computation
 * 
 * @param a - Input vector as Float64Array
 * @param b - Output vector as Float64Array
 * @returns The output vector b
 */
export function vsin(a: Float64Array, b: Float64Array): Float64Array;

/**
 * Element-wise cosine: b = cos(a)
 * Uses Apple's vForce for hardware-accelerated computation
 * 
 * @param a - Input vector as Float64Array
 * @param b - Output vector as Float64Array
 * @returns The output vector b
 */
export function vcos(a: Float64Array, b: Float64Array): Float64Array;

/**
 * Element-wise tangent: b = tan(a)
 * Uses Apple's vForce for hardware-accelerated computation
 * 
 * @param a - Input vector as Float64Array
 * @param b - Output vector as Float64Array
 * @returns The output vector b
 */
export function vtan(a: Float64Array, b: Float64Array): Float64Array;

/**
 * Element-wise exponential: b = exp(a)
 * Uses Apple's vForce for hardware-accelerated computation
 * 
 * @param a - Input vector as Float64Array
 * @param b - Output vector as Float64Array
 * @returns The output vector b
 */
export function vexp(a: Float64Array, b: Float64Array): Float64Array;

/**
 * Element-wise natural logarithm: b = log(a)
 * Uses Apple's vForce for hardware-accelerated computation
 * 
 * @param a - Input vector as Float64Array
 * @param b - Output vector as Float64Array
 * @returns The output vector b
 */
export function vlog(a: Float64Array, b: Float64Array): Float64Array;

/**
 * Element-wise base-10 logarithm: b = log10(a)
 * Uses Apple's vForce for hardware-accelerated computation
 * 
 * @param a - Input vector as Float64Array
 * @param b - Output vector as Float64Array
 * @returns The output vector b
 */
export function vlog10(a: Float64Array, b: Float64Array): Float64Array;

/**
 * Element-wise power: c = a^b
 * Uses Apple's vForce for hardware-accelerated computation
 * 
 * @param a - Base vector as Float64Array
 * @param b - Exponent vector as Float64Array
 * @param c - Output vector as Float64Array
 * @returns The output vector c
 */
export function vpow(
  a: Float64Array,
  b: Float64Array,
  c: Float64Array
): Float64Array;

/**
 * Clip vector values to range [min, max]
 * Uses Apple's vDSP for hardware-accelerated computation
 * 
 * @param a - Input vector as Float64Array
 * @param b - Output vector as Float64Array
 * @param min - Minimum value
 * @param max - Maximum value
 * @returns The output vector b
 */
export function vclip(
  a: Float64Array,
  b: Float64Array,
  min: number,
  max: number
): Float64Array;

/**
 * Threshold vector: b[i] = a[i] if a[i] > threshold, else 0
 * Uses Apple's vDSP for hardware-accelerated computation
 * 
 * @param a - Input vector as Float64Array
 * @param b - Output vector as Float64Array
 * @param threshold - Threshold value
 * @returns The output vector b
 */
export function vthreshold(
  a: Float64Array,
  b: Float64Array,
  threshold: number
): Float64Array;

/**
 * 1D Convolution
 * Uses Apple's vDSP for hardware-accelerated computation
 * 
 * @param signal - Input signal as Float64Array
 * @param kernel - Convolution kernel as Float64Array
 * @param result - Output as Float64Array (length = signal.length - kernel.length + 1)
 * @returns The output result
 */
export function conv(
  signal: Float64Array,
  kernel: Float64Array,
  result: Float64Array
): Float64Array;

/**
 * Cross-correlation of two signals
 * Uses Apple's vDSP for hardware-accelerated computation
 * 
 * @param a - First signal as Float64Array
 * @param b - Second signal as Float64Array
 * @param result - Output as Float64Array (length = a.length + b.length - 1)
 * @returns The output result
 */
export function xcorr(
  a: Float64Array,
  b: Float64Array,
  result: Float64Array
): Float64Array;

/**
 * Generate Hamming window
 * Uses Apple's vDSP for hardware-accelerated computation
 * 
 * @param length - Window length
 * @returns Window coefficients as Float64Array
 */
export function hamming(length: number): Float64Array;

/**
 * Generate Hanning window
 * Uses Apple's vDSP for hardware-accelerated computation
 * 
 * @param length - Window length
 * @returns Window coefficients as Float64Array
 */
export function hanning(length: number): Float64Array;

/**
 * Generate Blackman window
 * Uses Apple's vDSP for hardware-accelerated computation
 * 
 * @param length - Window length
 * @returns Window coefficients as Float64Array
 */
export function blackman(length: number): Float64Array;

/**
 * Matrix transpose: B = A^T
 * Uses Apple's vDSP for hardware-accelerated computation
 * 
 * @param A - Input matrix (rows × cols) as Float64Array in row-major order
 * @param B - Output matrix (cols × rows) as Float64Array in row-major order
 * @param rows - Number of rows in A
 * @param cols - Number of columns in A
 * @returns The output matrix B
 */
export function transpose(
  A: Float64Array,
  B: Float64Array,
  rows: number,
  cols: number
): Float64Array;

/**
 * Inverse Fast Fourier Transform (IFFT)
 * Uses Apple's vDSP for hardware-accelerated computation
 * 
 * @param real - Real part of frequency domain as Float64Array
 * @param imag - Imaginary part of frequency domain as Float64Array
 * @returns Time domain signal as Float64Array
 */
export function ifft(real: Float64Array, imag: Float64Array): Float64Array;

/**
 * Linear interpolation
 * Uses Apple's vDSP for hardware-accelerated computation
 * 
 * @param x - X coordinates of data points as Float64Array
 * @param y - Y coordinates of data points as Float64Array
 * @param xi - X coordinates to interpolate at as Float64Array
 * @param yi - Output interpolated Y values as Float64Array
 * @returns The output yi
 */
export function interp1d(
  x: Float64Array,
  y: Float64Array,
  xi: Float64Array,
  yi: Float64Array
): Float64Array;

/**
 * Reverse vector order: b = reverse(a)
 * Uses Apple's vDSP for hardware-accelerated computation
 * 
 * @param a - Input vector as Float64Array
 * @param b - Output vector as Float64Array
 * @returns The output vector b
 */
export function vreverse(a: Float64Array, b: Float64Array): Float64Array;

/**
 * Negate vector: b = -a
 * Uses Apple's vDSP for hardware-accelerated computation
 * 
 * @param a - Input vector as Float64Array
 * @param b - Output vector as Float64Array
 * @returns The output vector b
 */
export function vneg(a: Float64Array, b: Float64Array): Float64Array;

/**
 * Sum of squares: sum(a[i]^2)
 * Uses Apple's vDSP for hardware-accelerated computation
 * 
 * @param a - Input vector as Float64Array
 * @returns The sum of squares
 */
export function sumOfSquares(a: Float64Array): number;

/**
 * Mean magnitude: mean(|a[i]|)
 * Uses Apple's vDSP for hardware-accelerated computation
 * 
 * @param a - Input vector as Float64Array
 * @returns The mean magnitude
 */
export function meanMagnitude(a: Float64Array): number;

/**
 * Mean square: mean(a[i]^2)
 * Uses Apple's vDSP for hardware-accelerated computation
 * 
 * @param a - Input vector as Float64Array
 * @returns The mean square
 */
export function meanSquare(a: Float64Array): number;

/**
 * Element-wise vector subtraction: out[i] = a[i] - b[i]
 * Uses Apple's vDSP for hardware-accelerated computation
 * 
 * @param a - First vector as Float64Array
 * @param b - Second vector as Float64Array (must be same length as a)
 * @param out - Output vector as Float64Array (must be same length as a)
 * @returns The output vector
 */
export function vsub(
  a: Float64Array,
  b: Float64Array,
  out: Float64Array
): Float64Array;

/**
 * Element-wise vector division: out[i] = a[i] / b[i]
 * Uses Apple's vDSP for hardware-accelerated computation
 * 
 * @param a - First vector as Float64Array
 * @param b - Second vector as Float64Array (must be same length as a)
 * @param out - Output vector as Float64Array (must be same length as a)
 * @returns The output vector
 */
export function vdiv(
  a: Float64Array,
  b: Float64Array,
  out: Float64Array
): Float64Array;

/**
 * Copy vector: y = x
 * Uses Apple's BLAS for hardware-accelerated computation
 * 
 * @param x - Input vector as Float64Array
 * @param y - Output vector as Float64Array
 * @returns The output vector y
 */
export function copy(x: Float64Array, y: Float64Array): Float64Array;

/**
 * Swap two vectors: x <-> y
 * Uses Apple's BLAS for hardware-accelerated computation
 * 
 * @param x - First vector as Float64Array
 * @param y - Second vector as Float64Array
 * @returns The first vector x
 */
export function swap(x: Float64Array, y: Float64Array): Float64Array;

/**
 * L2 norm (Euclidean length): ||x||
 * Uses Apple's BLAS for hardware-accelerated computation
 * 
 * @param x - Input vector as Float64Array
 * @returns The L2 norm
 */
export function norm(x: Float64Array): number;

/**
 * Sum of absolute values: sum(|x[i]|)
 * Uses Apple's BLAS for hardware-accelerated computation
 * 
 * @param x - Input vector as Float64Array
 * @returns The sum of absolute values
 */
export function abssum(x: Float64Array): number;

/**
 * Index of maximum absolute value
 * Uses Apple's BLAS for hardware-accelerated computation
 * 
 * @param x - Input vector as Float64Array
 * @returns The index of the maximum absolute value
 */
export function maxAbsIndex(x: Float64Array): number;

/**
 * Givens rotation: apply rotation to vectors x and y
 * Uses Apple's BLAS for hardware-accelerated computation
 * 
 * @param x - First vector as Float64Array
 * @param y - Second vector as Float64Array
 * @param c - Cosine of rotation angle
 * @param s - Sine of rotation angle
 * @returns The first vector x
 */
export function rot(
  x: Float64Array,
  y: Float64Array,
  c: number,
  s: number
): Float64Array;

/**
 * Fill vector with scalar value
 * Uses Apple's vDSP for hardware-accelerated computation
 * 
 * @param scalar - Value to fill with
 * @param vec - Output vector as Float64Array
 * @returns The output vector
 */
export function vfill(scalar: number, vec: Float64Array): Float64Array;

/**
 * Generate linear ramp: vec[i] = start + i * step
 * Uses Apple's vDSP for hardware-accelerated computation
 * 
 * @param start - Starting value
 * @param step - Step size
 * @param vec - Output vector as Float64Array
 * @returns The output vector
 */
export function vramp(
  start: number,
  step: number,
  vec: Float64Array
): Float64Array;

/**
 * Add scalar to vector: c[i] = a[i] + scalar
 * Uses Apple's vDSP for hardware-accelerated computation
 * 
 * @param a - Input vector as Float64Array
 * @param scalar - Scalar value to add
 * @param c - Output vector as Float64Array
 * @returns The output vector c
 */
export function vaddScalar(
  a: Float64Array,
  scalar: number,
  c: Float64Array
): Float64Array;

/**
 * Multiply-add: d[i] = (a[i] * b[i]) + c[i]
 * Uses Apple's vDSP for hardware-accelerated computation
 * 
 * @param a - First vector as Float64Array
 * @param b - Second vector as Float64Array
 * @param c - Third vector as Float64Array
 * @param d - Output vector as Float64Array
 * @returns The output vector d
 */
export function vma(
  a: Float64Array,
  b: Float64Array,
  c: Float64Array,
  d: Float64Array
): Float64Array;

/**
 * Multiply-scalar-add: d[i] = (a[i] * b) + c[i]
 * Uses Apple's vDSP for hardware-accelerated computation
 * 
 * @param a - Input vector as Float64Array
 * @param b - Scalar multiplier
 * @param c - Vector to add as Float64Array
 * @param d - Output vector as Float64Array
 * @returns The output vector d
 */
export function vmsa(
  a: Float64Array,
  b: number,
  c: Float64Array,
  d: Float64Array
): Float64Array;

/**
 * Linear interpolation: c[i] = a[i] + t * (b[i] - a[i])
 * Uses Apple's vDSP for hardware-accelerated computation
 * 
 * @param a - Start vector as Float64Array
 * @param b - End vector as Float64Array
 * @param t - Interpolation parameter (0 to 1)
 * @param c - Output vector as Float64Array
 * @returns The output vector c
 */
export function vlerp(
  a: Float64Array,
  b: Float64Array,
  t: number,
  c: Float64Array
): Float64Array;

/**
 * Clear vector (set all elements to zero)
 * Uses Apple's vDSP for hardware-accelerated computation
 * 
 * @param vec - Vector to clear as Float64Array
 * @returns The cleared vector
 */
export function vclear(vec: Float64Array): Float64Array;

/**
 * Limit/saturate values to range [low, high]
 * Uses Apple's vDSP for hardware-accelerated computation
 * 
 * @param a - Input vector as Float64Array
 * @param low - Lower bound
 * @param high - Upper bound
 * @param c - Output vector as Float64Array
 * @returns The output vector c
 */
export function vlimit(
  a: Float64Array,
  low: number,
  high: number,
  c: Float64Array
): Float64Array;

/**
 * Maximum magnitude (absolute value)
 * Uses Apple's vDSP for hardware-accelerated computation
 * 
 * @param vec - Input vector as Float64Array
 * @returns The maximum magnitude
 */
export function maxMagnitude(vec: Float64Array): number;

/**
 * Minimum magnitude (absolute value)
 * Uses Apple's vDSP for hardware-accelerated computation
 * 
 * @param vec - Input vector as Float64Array
 * @returns The minimum magnitude
 */
export function minMagnitude(vec: Float64Array): number;

/**
 * Element-wise inverse sine: b[i] = asin(a[i])
 * Uses Apple's vForce for hardware-accelerated computation
 * 
 * @param a - Input vector as Float64Array
 * @param b - Output vector as Float64Array
 * @returns The output vector b
 */
export function vasin(a: Float64Array, b: Float64Array): Float64Array;

/**
 * Element-wise inverse cosine: b[i] = acos(a[i])
 * Uses Apple's vForce for hardware-accelerated computation
 * 
 * @param a - Input vector as Float64Array
 * @param b - Output vector as Float64Array
 * @returns The output vector b
 */
export function vacos(a: Float64Array, b: Float64Array): Float64Array;

/**
 * Element-wise inverse tangent: b[i] = atan(a[i])
 * Uses Apple's vForce for hardware-accelerated computation
 * 
 * @param a - Input vector as Float64Array
 * @param b - Output vector as Float64Array
 * @returns The output vector b
 */
export function vatan(a: Float64Array, b: Float64Array): Float64Array;

/**
 * Two-argument arctangent: out[i] = atan2(y[i], x[i])
 * Uses Apple's vForce for hardware-accelerated computation
 * 
 * @param y - Y coordinates as Float64Array
 * @param x - X coordinates as Float64Array
 * @param out - Output vector as Float64Array
 * @returns The output vector
 */
export function vatan2(
  y: Float64Array,
  x: Float64Array,
  out: Float64Array
): Float64Array;

/**
 * Element-wise hyperbolic sine: b[i] = sinh(a[i])
 * Uses Apple's vForce for hardware-accelerated computation
 * 
 * @param a - Input vector as Float64Array
 * @param b - Output vector as Float64Array
 * @returns The output vector b
 */
export function vsinh(a: Float64Array, b: Float64Array): Float64Array;

/**
 * Element-wise hyperbolic cosine: b[i] = cosh(a[i])
 * Uses Apple's vForce for hardware-accelerated computation
 * 
 * @param a - Input vector as Float64Array
 * @param b - Output vector as Float64Array
 * @returns The output vector b
 */
export function vcosh(a: Float64Array, b: Float64Array): Float64Array;

/**
 * Element-wise hyperbolic tangent: b[i] = tanh(a[i])
 * Uses Apple's vForce for hardware-accelerated computation
 * 
 * @param a - Input vector as Float64Array
 * @param b - Output vector as Float64Array
 * @returns The output vector b
 */
export function vtanh(a: Float64Array, b: Float64Array): Float64Array;

/**
 * Element-wise reciprocal: b[i] = 1 / a[i]
 * Uses Apple's vForce for hardware-accelerated computation
 * 
 * @param a - Input vector as Float64Array
 * @param b - Output vector as Float64Array
 * @returns The output vector b
 */
export function vreciprocal(a: Float64Array, b: Float64Array): Float64Array;

/**
 * Element-wise inverse square root: b[i] = 1 / sqrt(a[i])
 * Uses Apple's vForce for hardware-accelerated computation
 * 
 * @param a - Input vector as Float64Array
 * @param b - Output vector as Float64Array
 * @returns The output vector b
 */
export function vrsqrt(a: Float64Array, b: Float64Array): Float64Array;

/**
 * Element-wise ceiling: b[i] = ceil(a[i])
 * Uses Apple's vForce for hardware-accelerated computation
 * 
 * @param a - Input vector as Float64Array
 * @param b - Output vector as Float64Array
 * @returns The output vector b
 */
export function vceil(a: Float64Array, b: Float64Array): Float64Array;

/**
 * Element-wise floor: b[i] = floor(a[i])
 * Uses Apple's vForce for hardware-accelerated computation
 * 
 * @param a - Input vector as Float64Array
 * @param b - Output vector as Float64Array
 * @returns The output vector b
 */
export function vfloor(a: Float64Array, b: Float64Array): Float64Array;

/**
 * Element-wise truncate (round toward zero): b[i] = trunc(a[i])
 * Uses Apple's vForce for hardware-accelerated computation
 * 
 * @param a - Input vector as Float64Array
 * @param b - Output vector as Float64Array
 * @returns The output vector b
 */
export function vtrunc(a: Float64Array, b: Float64Array): Float64Array;

/**
 * Copy sign: c[i] = |a[i]| * sign(b[i])
 * Uses Apple's vForce for hardware-accelerated computation
 * 
 * @param a - Magnitude vector as Float64Array
 * @param b - Sign vector as Float64Array
 * @param c - Output vector as Float64Array
 * @returns The output vector c
 */
export function vcopysign(
  a: Float64Array,
  b: Float64Array,
  c: Float64Array
): Float64Array;

/**
 * All exported functions
 */
declare const accelerate: {
  // Matrix operations
  matmul: typeof matmul;
  matvec: typeof matvec;
  transpose: typeof transpose;
  
  // BLAS operations
  axpy: typeof axpy;
  copy: typeof copy;
  swap: typeof swap;
  norm: typeof norm;
  abssum: typeof abssum;
  maxAbsIndex: typeof maxAbsIndex;
  rot: typeof rot;
  
  // Vector arithmetic
  dot: typeof dot;
  sum: typeof sum;
  mean: typeof mean;
  vadd: typeof vadd;
  vsub: typeof vsub;
  vmul: typeof vmul;
  vdiv: typeof vdiv;
  vscale: typeof vscale;
  vneg: typeof vneg;
  vaddScalar: typeof vaddScalar;
  vma: typeof vma;
  vmsa: typeof vmsa;
  
  // Vector functions
  vabs: typeof vabs;
  vsquare: typeof vsquare;
  vsqrt: typeof vsqrt;
  normalize: typeof normalize;
  vreverse: typeof vreverse;
  vfill: typeof vfill;
  vramp: typeof vramp;
  vlerp: typeof vlerp;
  vclear: typeof vclear;
  vlimit: typeof vlimit;
  
  // Trigonometric
  vsin: typeof vsin;
  vcos: typeof vcos;
  vtan: typeof vtan;
  vasin: typeof vasin;
  vacos: typeof vacos;
  vatan: typeof vatan;
  vatan2: typeof vatan2;
  
  // Hyperbolic
  vsinh: typeof vsinh;
  vcosh: typeof vcosh;
  vtanh: typeof vtanh;
  
  // Exponential/Logarithmic
  vexp: typeof vexp;
  vlog: typeof vlog;
  vlog10: typeof vlog10;
  vpow: typeof vpow;
  vreciprocal: typeof vreciprocal;
  vrsqrt: typeof vrsqrt;
  
  // Rounding
  vceil: typeof vceil;
  vfloor: typeof vfloor;
  vtrunc: typeof vtrunc;
  vcopysign: typeof vcopysign;
  
  // Clipping/Thresholding
  vclip: typeof vclip;
  vthreshold: typeof vthreshold;
  
  // Reductions
  max: typeof max;
  min: typeof min;
  minmax: typeof minmax;
  rms: typeof rms;
  variance: typeof variance;
  stddev: typeof stddev;
  sumOfSquares: typeof sumOfSquares;
  meanMagnitude: typeof meanMagnitude;
  meanSquare: typeof meanSquare;
  maxMagnitude: typeof maxMagnitude;
  minMagnitude: typeof minMagnitude;
  
  // Distance metrics
  euclidean: typeof euclidean;
  
  // Signal processing
  fft: typeof fft;
  ifft: typeof ifft;
  conv: typeof conv;
  xcorr: typeof xcorr;
  
  // Window functions
  hamming: typeof hamming;
  hanning: typeof hanning;
  blackman: typeof blackman;
  
  // Interpolation
  interp1d: typeof interp1d;
};

export default accelerate;
