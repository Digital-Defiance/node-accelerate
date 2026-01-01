# node-accelerate

High-performance Apple Accelerate framework bindings for Node.js. Get **up to 305x faster** matrix operations and **5-10x faster** vector operations on Apple Silicon (M1/M2/M3/M4).

**80+ hardware-accelerated functions** including BLAS, vDSP, and vForce operations.

[![npm version](https://badge.fury.io/js/node-accelerate.svg)](https://www.npmjs.com/package/@digitaldefiance/node-accelerate)
[![GitHub](https://img.shields.io/github/license/Digital-Defiance/node-accelerate)](https://github.com/Digital-Defiance/node-accelerate/blob/main/LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/Digital-Defiance/node-accelerate?style=social)](https://github.com/Digital-Defiance/node-accelerate)
[![Node.js CI](https://github.com/Digital-Defiance/node-accelerate/workflows/Test/badge.svg)](https://github.com/Digital-Defiance/node-accelerate/actions)
[![Platform](https://img.shields.io/badge/platform-macOS-lightgrey)](https://github.com/Digital-Defiance/node-accelerate)
[![Node](https://img.shields.io/badge/node-%3E%3D18-brightgreen)](https://nodejs.org/)

----

## Why?

Node.js doesn't natively use Apple's Accelerate framework, which provides hardware-optimized routines for numerical computing. This addon exposes Accelerate's BLAS (matrix operations) and vDSP (vector/signal processing) to JavaScript, giving you direct access to:

- **AMX (Apple Matrix coprocessor)** - Hardware matrix acceleration
- **NEON SIMD** - Vector processing
- **Optimized FFT** - Fast Fourier Transform

**Note**: This package only works on macOS because it uses Apple's Accelerate framework. If you're on Linux ARM64 (Raspberry Pi, AWS Graviton, etc.), consider using OpenBLAS, Eigen, or BLIS instead.

## Performance

Real benchmarks on Apple M4 Max:

| Operation | Pure JavaScript | node-accelerate | Speedup |
|-----------|----------------|-----------------|---------|
| Matrix Multiply (500×500) | 90.5 ms | 0.30 ms | **305x** |
| Vector Dot Product (1M) | 0.65 ms | 0.13 ms | **4.9x** |
| Vector Sum (1M) | 0.58 ms | 0.07 ms | **7.8x** |
| Vector Add (1M) | 0.44 ms | 0.19 ms | **2.2x** |
| Vector Sin (10k) | 0.062 ms | 0.008 ms | **7.6x** |

## Features

### Matrix Operations (BLAS)
- **Matrix multiplication** (double & single precision)
- **Matrix-vector multiplication**
- **Matrix transpose**
- **AXPY, copy, swap** operations
- **Vector norms** and rotations

### Vector Arithmetic (vDSP)
- **Basic operations**: add, subtract, multiply, divide, scale, negate
- **Dot product, norm, absolute sum**
- **Element-wise operations**: abs, square, sqrt, power, reciprocal
- **Multiply-add operations**: vma, vmsa
- **Normalization** to unit length
- **Vector utilities**: fill, ramp, clear, reverse, copy, swap

### Trigonometric Functions (vForce)
- **Standard trig**: sin, cos, tan (5-10x faster than Math functions)
- **Inverse trig**: asin, acos, atan, atan2
- **Hyperbolic**: sinh, cosh, tanh
- Process 1000s of values in microseconds

### Exponential & Logarithmic Functions (vForce)
- **exp** - Natural exponential
- **log, log10** - Natural and base-10 logarithms
- **pow** - Element-wise power
- **Reciprocal and inverse square root**

### Statistical Functions (vDSP)
- **Basic stats**: sum, mean, min, max, minmax
- **Variance & standard deviation**
- **RMS** (Root Mean Square)
- **Sum of squares, mean magnitude, mean square**
- **Max/min magnitude**

### Signal Processing (vDSP)
- **FFT/IFFT** - Fast Fourier Transform (forward & inverse)
- **Convolution** - 1D convolution for filtering
- **Cross-correlation** - Signal similarity analysis
- **Window functions**: Hamming, Hanning, Blackman
- 10-50x faster than pure JavaScript FFT

### Data Processing (vDSP & vForce)
- **Clipping & limiting** - Constrain values to range
- **Thresholding** - Apply threshold to data
- **Interpolation** - Linear interpolation and lerp
- **Rounding**: ceil, floor, trunc
- **Sign manipulation**: copysign

### All Operations Support
- ✅ Hardware acceleration via AMX & NEON
- ✅ Double precision (Float64Array)
- ✅ Single precision (Float32Array) for matrix ops
- ✅ Zero-copy operations where possible
- ✅ Optimized for Apple Silicon
- ✅ **80+ functions** total

## Installation

```bash
npm install @digitaldefiance/node-accelerate
```

**Requirements:**
- macOS (Apple Silicon: M1/M2/M3/M4 or Intel)
- Node.js >= 18.0.0
- Xcode Command Line Tools

### First-time Setup

If you don't have Xcode Command Line Tools installed:

```bash
xcode-select --install
```

### Platform Check

The package will automatically check your platform during installation. If you see errors:

**"node-accelerate requires macOS"**
- This package only works on macOS due to Apple's Accelerate framework
- Not supported on Linux or Windows

**"Xcode Command Line Tools may not be installed"**
- Run: `xcode-select --install`
- Follow the prompts to install

**"Failed to load native module"**
- Try rebuilding: `npm rebuild @digitaldefiance/node-accelerate`
- Ensure Xcode Command Line Tools are installed

### Verifying Installation

```bash
node -e "const a = require('@digitaldefiance/node-accelerate'); console.log('✓ Works!')"
```

## Quick Start

```javascript
const accelerate = require('@digitaldefiance/node-accelerate');

// Matrix multiplication: C = A × B
const M = 1000, K = 1000, N = 1000;
const A = new Float64Array(M * K);
const B = new Float64Array(K * N);
const C = new Float64Array(M * N);

// Fill with random data
for (let i = 0; i < A.length; i++) A[i] = Math.random();
for (let i = 0; i < B.length; i++) B[i] = Math.random();

// Hardware-accelerated matrix multiplication
accelerate.matmul(A, B, C, M, K, N);

// Vector operations
const vec1 = new Float64Array(1000000);
const vec2 = new Float64Array(1000000);
const result = new Float64Array(1000000);

for (let i = 0; i < vec1.length; i++) {
  vec1[i] = Math.random();
  vec2[i] = Math.random();
}

accelerate.vadd(vec1, vec2, result);  // result = vec1 + vec2
accelerate.vmul(vec1, vec2, result);  // result = vec1 * vec2

const dotProduct = accelerate.dot(vec1, vec2);
const sum = accelerate.sum(vec1);
const mean = accelerate.mean(vec1);

// Statistical operations
const { min, max } = accelerate.minmax(vec1);
const variance = accelerate.variance(vec1);
const stddev = accelerate.stddev(vec1);

// Trigonometric functions (vectorized)
const angles = new Float64Array(1000);
const sines = new Float64Array(1000);
const cosines = new Float64Array(1000);

for (let i = 0; i < 1000; i++) {
  angles[i] = (i / 1000) * 2 * Math.PI;
}

accelerate.vsin(angles, sines);
accelerate.vcos(angles, cosines);

// Signal processing
const signal = new Float64Array(65536);
for (let i = 0; i < signal.length; i++) {
  signal[i] = Math.sin(2 * Math.PI * i / signal.length);
}

// Apply window and compute FFT
const window = accelerate.hanning(signal.length);
const windowed = new Float64Array(signal.length);
accelerate.vmul(signal, window, windowed);

const spectrum = accelerate.fft(windowed);
console.log(spectrum.real, spectrum.imag);

// Inverse FFT
const reconstructed = accelerate.ifft(spectrum.real, spectrum.imag);

// Convolution for filtering
const kernel = new Float64Array([0.25, 0.5, 0.25]); // Moving average
const filtered = new Float64Array(signal.length - kernel.length + 1);
accelerate.conv(signal, kernel, filtered);

// Data processing
const data = new Float64Array(1000);
for (let i = 0; i < data.length; i++) {
  data[i] = Math.random() * 200 - 100;
}

// Clip outliers
const clipped = new Float64Array(1000);
accelerate.vclip(data, clipped, -50, 50);

// Matrix transpose
const matrix = new Float64Array([1, 2, 3, 4, 5, 6]); // 2×3
const transposed = new Float64Array(6); // 3×2
accelerate.transpose(matrix, transposed, 2, 3);
```

## More Examples

Check out the `examples/` directory for complete working examples:

- **`machine-learning.js`** - Neural network operations, softmax, ReLU
- **`signal-processing.js`** - FFT, filtering, spectral analysis
- **`statistical-operations.js`** - Mean, variance, std dev, z-scores
- **`trigonometric-functions.js`** - Vectorized trig operations
- **`signal-processing-advanced.js`** - Convolution, correlation, windowing
- **`mathematical-functions.js`** - Exp, log, power functions
- **`data-processing.js`** - Clipping, thresholding, interpolation
- **`matrix-multiply.js`** - Matrix operations and benchmarks
- **`vector-operations.js`** - Vector arithmetic examples

Run any example:
```bash
node examples/statistical-operations.js
node examples/signal-processing-advanced.js
```

## API Reference

### Matrix Operations (BLAS)

#### `matmul(A, B, C, M, K, N)`

Matrix multiplication: C = A × B

- `A`: Float64Array - First matrix (M × K) in row-major order
- `B`: Float64Array - Second matrix (K × N) in row-major order
- `C`: Float64Array - Output matrix (M × N) in row-major order
- `M`: number - Rows in A and C
- `K`: number - Columns in A, rows in B
- `N`: number - Columns in B and C
- Returns: Float64Array (C)

**Example:**
```javascript
const M = 100, K = 100, N = 100;
const A = new Float64Array(M * K);
const B = new Float64Array(K * N);
const C = new Float64Array(M * N);

// Fill A and B...
accelerate.matmul(A, B, C, M, K, N);
```

#### `matmulFloat(A, B, C, M, K, N)`

Single-precision matrix multiplication (uses Float32Array)

Same parameters as `matmul` but with Float32Array instead of Float64Array.

#### `matvec(A, x, y, M, N)`

Matrix-vector multiplication: y = A × x

- `A`: Float64Array - Matrix (M × N) in row-major order
- `x`: Float64Array - Input vector (N elements)
- `y`: Float64Array - Output vector (M elements)
- `M`: number - Rows in A
- `N`: number - Columns in A
- Returns: Float64Array (y)

#### `transpose(A, B, rows, cols)`

Matrix transpose: B = A^T

- `A`: Float64Array - Input matrix (rows × cols) in row-major order
- `B`: Float64Array - Output matrix (cols × rows) in row-major order
- `rows`: number - Number of rows in A
- `cols`: number - Number of columns in A
- Returns: Float64Array (B)

**Example:**
```javascript
const A = new Float64Array([1, 2, 3, 4, 5, 6]); // 2×3 matrix
const B = new Float64Array(6); // 3×2 matrix
accelerate.transpose(A, B, 2, 3);
```

#### `axpy(alpha, x, y)`

AXPY operation: y = alpha*x + y

- `alpha`: number - Scalar multiplier
- `x`: Float64Array - Input vector
- `y`: Float64Array - Input/output vector
- Returns: Float64Array (y)

#### `copy(x, y)`

Copy vector: y = x

- `x`: Float64Array - Input vector
- `y`: Float64Array - Output vector
- Returns: Float64Array (y)

#### `swap(x, y)`

Swap two vectors: x <-> y

- `x`: Float64Array - First vector
- `y`: Float64Array - Second vector
- Returns: Float64Array (x)

#### `norm(x)`

L2 norm (Euclidean length): ||x||

- `x`: Float64Array - Input vector
- Returns: number

**Example:**
```javascript
const vec = new Float64Array([3, 4]);
const length = accelerate.norm(vec); // 5
```

#### `abssum(x)`

Sum of absolute values: sum(|x[i]|)

- `x`: Float64Array - Input vector
- Returns: number

#### `maxAbsIndex(x)`

Index of maximum absolute value

- `x`: Float64Array - Input vector
- Returns: number (index)

**Example:**
```javascript
const vec = new Float64Array([1, -5, 3, -2]);
const idx = accelerate.maxAbsIndex(vec); // 1 (value is -5)
```

#### `rot(x, y, c, s)`

Givens rotation: apply rotation to vectors x and y

- `x`: Float64Array - First vector
- `y`: Float64Array - Second vector
- `c`: number - Cosine of rotation angle
- `s`: number - Sine of rotation angle
- Returns: Float64Array (x)

### Vector Arithmetic

#### `dot(a, b)`

Dot product: sum(a[i] * b[i])

- `a`: Float64Array - First vector
- `b`: Float64Array - Second vector (same length as a)
- Returns: number

#### `vadd(a, b, out)`

Element-wise addition: out[i] = a[i] + b[i]

- `a`: Float64Array - First vector
- `b`: Float64Array - Second vector
- `out`: Float64Array - Output vector
- Returns: Float64Array (out)

#### `vsub(a, b, out)`

Element-wise subtraction: out[i] = a[i] - b[i]

- `a`: Float64Array - First vector
- `b`: Float64Array - Second vector
- `out`: Float64Array - Output vector
- Returns: Float64Array (out)

#### `vmul(a, b, out)`

Element-wise multiplication: out[i] = a[i] * b[i]

- `a`: Float64Array - First vector
- `b`: Float64Array - Second vector
- `out`: Float64Array - Output vector
- Returns: Float64Array (out)

#### `vdiv(a, b, out)`

Element-wise division: out[i] = a[i] / b[i]

- `a`: Float64Array - First vector
- `b`: Float64Array - Second vector
- `out`: Float64Array - Output vector
- Returns: Float64Array (out)

#### `vscale(a, scalar, b)`

Vector scaling: b = a * scalar

- `a`: Float64Array - Input vector
- `scalar`: number - Scalar multiplier
- `b`: Float64Array - Output vector
- Returns: Float64Array (b)

#### `vneg(a, b)`

Vector negation: b = -a

- `a`: Float64Array - Input vector
- `b`: Float64Array - Output vector
- Returns: Float64Array (b)

#### `vaddScalar(a, scalar, c)`

Add scalar to vector: c[i] = a[i] + scalar

- `a`: Float64Array - Input vector
- `scalar`: number - Scalar value to add
- `c`: Float64Array - Output vector
- Returns: Float64Array (c)

#### `vma(a, b, c, d)`

Multiply-add: d[i] = (a[i] * b[i]) + c[i]

- `a`: Float64Array - First vector
- `b`: Float64Array - Second vector
- `c`: Float64Array - Third vector
- `d`: Float64Array - Output vector
- Returns: Float64Array (d)

**Example:**
```javascript
const a = new Float64Array([2, 3, 4]);
const b = new Float64Array([5, 6, 7]);
const c = new Float64Array([1, 1, 1]);
const d = new Float64Array(3);
accelerate.vma(a, b, c, d); // d = [11, 19, 29]
```

#### `vmsa(a, b, c, d)`

Multiply-scalar-add: d[i] = (a[i] * b) + c[i]

- `a`: Float64Array - Input vector
- `b`: number - Scalar multiplier
- `c`: Float64Array - Vector to add
- `d`: Float64Array - Output vector
- Returns: Float64Array (d)

### Vector Functions

#### `vabs(a, b)`

Element-wise absolute value: b[i] = |a[i]|

- `a`: Float64Array - Input vector
- `b`: Float64Array - Output vector
- Returns: Float64Array (b)

#### `vsquare(a, b)`

Element-wise square: b[i] = a[i]^2

- `a`: Float64Array - Input vector
- `b`: Float64Array - Output vector
- Returns: Float64Array (b)

#### `vsqrt(a, b)`

Element-wise square root: b[i] = sqrt(a[i])

- `a`: Float64Array - Input vector
- `b`: Float64Array - Output vector
- Returns: Float64Array (b)

#### `normalize(a, b)`

Normalize vector to unit length: b = a / ||a||

- `a`: Float64Array - Input vector
- `b`: Float64Array - Output vector (unit vector)
- Returns: Float64Array (b)

#### `vreverse(a, b)`

Reverse vector order: b = reverse(a)

- `a`: Float64Array - Input vector
- `b`: Float64Array - Output vector
- Returns: Float64Array (b)

#### `vfill(scalar, vec)`

Fill vector with scalar value

- `scalar`: number - Value to fill with
- `vec`: Float64Array - Output vector
- Returns: Float64Array (vec)

**Example:**
```javascript
const vec = new Float64Array(100);
accelerate.vfill(3.14, vec); // All elements = 3.14
```

#### `vramp(start, step, vec)`

Generate linear ramp: vec[i] = start + i * step

- `start`: number - Starting value
- `step`: number - Step size
- `vec`: Float64Array - Output vector
- Returns: Float64Array (vec)

**Example:**
```javascript
const vec = new Float64Array(5);
accelerate.vramp(0, 2, vec); // vec = [0, 2, 4, 6, 8]
```

#### `vlerp(a, b, t, c)`

Linear interpolation: c[i] = a[i] + t * (b[i] - a[i])

- `a`: Float64Array - Start vector
- `b`: Float64Array - End vector
- `t`: number - Interpolation parameter (0 to 1)
- `c`: Float64Array - Output vector
- Returns: Float64Array (c)

**Example:**
```javascript
const start = new Float64Array([0, 0, 0]);
const end = new Float64Array([10, 20, 30]);
const result = new Float64Array(3);
accelerate.vlerp(start, end, 0.5, result); // result = [5, 10, 15]
```

#### `vclear(vec)`

Clear vector (set all elements to zero)

- `vec`: Float64Array - Vector to clear
- Returns: Float64Array (vec)

#### `vlimit(a, low, high, c)`

Limit/saturate values to range [low, high]

- `a`: Float64Array - Input vector
- `low`: number - Lower bound
- `high`: number - Upper bound
- `c`: Float64Array - Output vector
- Returns: Float64Array (c)

### Trigonometric Functions

#### `vsin(a, b)`

Element-wise sine: b[i] = sin(a[i])

- `a`: Float64Array - Input vector (radians)
- `b`: Float64Array - Output vector
- Returns: Float64Array (b)

#### `vcos(a, b)`

Element-wise cosine: b[i] = cos(a[i])

- `a`: Float64Array - Input vector (radians)
- `b`: Float64Array - Output vector
- Returns: Float64Array (b)

#### `vtan(a, b)`

Element-wise tangent: b[i] = tan(a[i])

- `a`: Float64Array - Input vector (radians)
- `b`: Float64Array - Output vector
- Returns: Float64Array (b)

**Example:**
```javascript
const angles = new Float64Array(1000);
const sines = new Float64Array(1000);
for (let i = 0; i < 1000; i++) {
  angles[i] = (i / 1000) * 2 * Math.PI;
}
accelerate.vsin(angles, sines);
```

#### `vasin(a, b)`

Element-wise inverse sine: b[i] = asin(a[i])

- `a`: Float64Array - Input vector (values in [-1, 1])
- `b`: Float64Array - Output vector (radians)
- Returns: Float64Array (b)

#### `vacos(a, b)`

Element-wise inverse cosine: b[i] = acos(a[i])

- `a`: Float64Array - Input vector (values in [-1, 1])
- `b`: Float64Array - Output vector (radians)
- Returns: Float64Array (b)

#### `vatan(a, b)`

Element-wise inverse tangent: b[i] = atan(a[i])

- `a`: Float64Array - Input vector
- `b`: Float64Array - Output vector (radians)
- Returns: Float64Array (b)

#### `vatan2(y, x, out)`

Two-argument arctangent: out[i] = atan2(y[i], x[i])

- `y`: Float64Array - Y coordinates
- `x`: Float64Array - X coordinates
- `out`: Float64Array - Output vector (radians)
- Returns: Float64Array (out)

**Example:**
```javascript
const y = new Float64Array([1, 1, -1, -1]);
const x = new Float64Array([1, -1, -1, 1]);
const angles = new Float64Array(4);
accelerate.vatan2(y, x, angles); // Angles in all four quadrants
```

### Hyperbolic Functions

#### `vsinh(a, b)`

Element-wise hyperbolic sine: b[i] = sinh(a[i])

- `a`: Float64Array - Input vector
- `b`: Float64Array - Output vector
- Returns: Float64Array (b)

#### `vcosh(a, b)`

Element-wise hyperbolic cosine: b[i] = cosh(a[i])

- `a`: Float64Array - Input vector
- `b`: Float64Array - Output vector
- Returns: Float64Array (b)

#### `vtanh(a, b)`

Element-wise hyperbolic tangent: b[i] = tanh(a[i])

- `a`: Float64Array - Input vector
- `b`: Float64Array - Output vector
- Returns: Float64Array (b)

**Example:**
```javascript
// tanh is commonly used as an activation function in neural networks
const logits = new Float64Array(1000);
const activations = new Float64Array(1000);
// ... fill logits ...
accelerate.vtanh(logits, activations);
```

### Exponential and Logarithmic Functions

#### `vexp(a, b)`

Element-wise exponential: b[i] = exp(a[i])

- `a`: Float64Array - Input vector
- `b`: Float64Array - Output vector
- Returns: Float64Array (b)

#### `vlog(a, b)`

Element-wise natural logarithm: b[i] = log(a[i])

- `a`: Float64Array - Input vector
- `b`: Float64Array - Output vector
- Returns: Float64Array (b)

#### `vlog10(a, b)`

Element-wise base-10 logarithm: b[i] = log10(a[i])

- `a`: Float64Array - Input vector
- `b`: Float64Array - Output vector
- Returns: Float64Array (b)

#### `vpow(a, b, c)`

Element-wise power: c[i] = a[i]^b[i]

- `a`: Float64Array - Base vector
- `b`: Float64Array - Exponent vector
- `c`: Float64Array - Output vector
- Returns: Float64Array (c)

#### `vreciprocal(a, b)`

Element-wise reciprocal: b[i] = 1 / a[i]

- `a`: Float64Array - Input vector
- `b`: Float64Array - Output vector
- Returns: Float64Array (b)

**Example:**
```javascript
const values = new Float64Array([2, 4, 5, 10]);
const reciprocals = new Float64Array(4);
accelerate.vreciprocal(values, reciprocals); // [0.5, 0.25, 0.2, 0.1]
```

#### `vrsqrt(a, b)`

Element-wise inverse square root: b[i] = 1 / sqrt(a[i])

- `a`: Float64Array - Input vector
- `b`: Float64Array - Output vector
- Returns: Float64Array (b)

**Example:**
```javascript
// Fast normalization using inverse square root
const vec = new Float64Array([3, 4]);
const sumSq = accelerate.sumOfSquares(vec); // 25
const invLen = new Float64Array([sumSq]);
const invLenResult = new Float64Array(1);
accelerate.vrsqrt(invLen, invLenResult); // 0.2 (1/5)
```

### Rounding Functions

#### `vceil(a, b)`

Element-wise ceiling: b[i] = ceil(a[i])

- `a`: Float64Array - Input vector
- `b`: Float64Array - Output vector
- Returns: Float64Array (b)

#### `vfloor(a, b)`

Element-wise floor: b[i] = floor(a[i])

- `a`: Float64Array - Input vector
- `b`: Float64Array - Output vector
- Returns: Float64Array (b)

#### `vtrunc(a, b)`

Element-wise truncate (round toward zero): b[i] = trunc(a[i])

- `a`: Float64Array - Input vector
- `b`: Float64Array - Output vector
- Returns: Float64Array (b)

**Example:**
```javascript
const values = new Float64Array([1.7, -1.7, 2.3, -2.3]);
const ceiled = new Float64Array(4);
const floored = new Float64Array(4);
const truncated = new Float64Array(4);

accelerate.vceil(values, ceiled);       // [2, -1, 3, -2]
accelerate.vfloor(values, floored);     // [1, -2, 2, -3]
accelerate.vtrunc(values, truncated);   // [1, -1, 2, -2]
```

#### `vcopysign(a, b, c)`

Copy sign: c[i] = |a[i]| * sign(b[i])

- `a`: Float64Array - Magnitude vector
- `b`: Float64Array - Sign vector
- `c`: Float64Array - Output vector
- Returns: Float64Array (c)

**Example:**
```javascript
const magnitudes = new Float64Array([1, 2, 3, 4]);
const signs = new Float64Array([-1, 1, -1, 1]);
const result = new Float64Array(4);
accelerate.vcopysign(magnitudes, signs, result); // [-1, 2, -3, 4]
```

### Clipping and Thresholding

#### `vclip(a, b, min, max)`

Clip values to range [min, max]

- `a`: Float64Array - Input vector
- `b`: Float64Array - Output vector
- `min`: number - Minimum value
- `max`: number - Maximum value
- Returns: Float64Array (b)

**Example:**
```javascript
const data = new Float64Array([-10, -5, 0, 5, 10]);
const clipped = new Float64Array(5);
accelerate.vclip(data, clipped, -3, 3);
// clipped = [-3, -3, 0, 3, 3]
```

#### `vthreshold(a, b, threshold)`

Threshold values: b[i] = a[i] if a[i] > threshold, else threshold

- `a`: Float64Array - Input vector
- `b`: Float64Array - Output vector
- `threshold`: number - Threshold value
- Returns: Float64Array (b)

### Statistical Functions

#### `sum(vec)`

Sum of all elements

- `vec`: Float64Array - Input vector
- Returns: number

#### `mean(vec)`

Mean (average) of all elements

- `vec`: Float64Array - Input vector
- Returns: number

#### `variance(vec)`

Variance of all elements

- `vec`: Float64Array - Input vector
- Returns: number

#### `stddev(vec)`

Standard deviation of all elements

- `vec`: Float64Array - Input vector
- Returns: number

#### `max(vec)`

Maximum element

- `vec`: Float64Array - Input vector
- Returns: number

#### `min(vec)`

Minimum element

- `vec`: Float64Array - Input vector
- Returns: number

#### `minmax(vec)`

Both minimum and maximum elements

- `vec`: Float64Array - Input vector
- Returns: {min: number, max: number}

**Example:**
```javascript
const data = new Float64Array([1, 5, 3, 9, 2]);
const stats = accelerate.minmax(data);
console.log(stats.min, stats.max); // 1, 9
```

#### `rms(vec)`

Root Mean Square: sqrt(sum(vec[i]^2) / n)

- `vec`: Float64Array - Input vector
- Returns: number

#### `sumOfSquares(vec)`

Sum of squares: sum(vec[i]^2)

- `vec`: Float64Array - Input vector
- Returns: number

#### `meanMagnitude(vec)`

Mean magnitude: mean(|vec[i]|)

- `vec`: Float64Array - Input vector
- Returns: number

#### `meanSquare(vec)`

Mean square: mean(vec[i]^2)

- `vec`: Float64Array - Input vector
- Returns: number

#### `maxMagnitude(vec)`

Maximum magnitude (absolute value)

- `vec`: Float64Array - Input vector
- Returns: number

**Example:**
```javascript
const vec = new Float64Array([1, -5, 3, -2]);
const maxMag = accelerate.maxMagnitude(vec); // 5
```

#### `minMagnitude(vec)`

Minimum magnitude (absolute value)

- `vec`: Float64Array - Input vector
- Returns: number

**Example:**
```javascript
const vec = new Float64Array([1, -5, 3, -2]);
const minMag = accelerate.minMagnitude(vec); // 1
```

### Distance Metrics

#### `euclidean(a, b)`

Euclidean distance: sqrt(sum((a[i] - b[i])^2))

- `a`: Float64Array - First vector
- `b`: Float64Array - Second vector
- Returns: number

**Example:**
```javascript
const point1 = new Float64Array([0, 0, 0]);
const point2 = new Float64Array([3, 4, 0]);
const distance = accelerate.euclidean(point1, point2); // 5
```

### Signal Processing

#### `fft(signal)`

Fast Fourier Transform (real to complex)

- `signal`: Float64Array - Input signal (length must be power of 2)
- Returns: {real: Float64Array, imag: Float64Array}

**Example:**
```javascript
const signal = new Float64Array(1024);
for (let i = 0; i < signal.length; i++) {
  signal[i] = Math.sin(2 * Math.PI * i / signal.length);
}
const spectrum = accelerate.fft(signal);
console.log(spectrum.real.length); // 512
console.log(spectrum.imag.length); // 512
```

#### `ifft(real, imag)`

Inverse Fast Fourier Transform (complex to real)

- `real`: Float64Array - Real part of frequency domain
- `imag`: Float64Array - Imaginary part of frequency domain
- Returns: Float64Array - Time domain signal

**Example:**
```javascript
const signal = new Float64Array(256);
// ... fill signal ...
const spectrum = accelerate.fft(signal);
const reconstructed = accelerate.ifft(spectrum.real, spectrum.imag);
// reconstructed ≈ signal
```

#### `conv(signal, kernel, result)`

1D Convolution

- `signal`: Float64Array - Input signal
- `kernel`: Float64Array - Convolution kernel
- `result`: Float64Array - Output (length = signal.length - kernel.length + 1)
- Returns: Float64Array (result)

**Example:**
```javascript
const signal = new Float64Array([1, 2, 3, 4, 5]);
const kernel = new Float64Array([0.25, 0.5, 0.25]); // Moving average
const result = new Float64Array(3);
accelerate.conv(signal, kernel, result);
```

#### `xcorr(a, b, result)`

Cross-correlation

- `a`: Float64Array - First signal
- `b`: Float64Array - Second signal
- `result`: Float64Array - Output (length = a.length + b.length - 1)
- Returns: Float64Array (result)

### Window Functions

#### `hamming(length)`

Generate Hamming window

- `length`: number - Window length
- Returns: Float64Array - Window coefficients

#### `hanning(length)`

Generate Hanning window

- `length`: number - Window length
- Returns: Float64Array - Window coefficients

#### `blackman(length)`

Generate Blackman window

- `length`: number - Window length
- Returns: Float64Array - Window coefficients

**Example:**
```javascript
const window = accelerate.hanning(256);
const signal = new Float64Array(256);
const windowed = new Float64Array(256);

// Apply window to signal
accelerate.vmul(signal, window, windowed);
const spectrum = accelerate.fft(windowed);
```

### Interpolation

#### `interp1d(x, y, xi, yi)`

Linear interpolation

- `x`: Float64Array - X coordinates of data points
- `y`: Float64Array - Y coordinates of data points
- `xi`: Float64Array - X coordinates to interpolate at
- `yi`: Float64Array - Output interpolated Y values
- Returns: Float64Array (yi)

**Example:**
```javascript
const x = new Float64Array([0, 1, 2, 3]);
const y = new Float64Array([0, 1, 4, 9]);
const xi = new Float64Array([0.5, 1.5, 2.5]);
const yi = new Float64Array(3);
accelerate.interp1d(x, y, xi, yi);
```

## TypeScript Support

Full TypeScript definitions included:

```typescript
import * as accelerate from 'node-accelerate';

const A = new Float64Array(100 * 100);
const B = new Float64Array(100 * 100);
const C = new Float64Array(100 * 100);

accelerate.matmul(A, B, C, 100, 100, 100);
```

## Use Cases

### Machine Learning Inference

```javascript
// Neural network dense layer with activation
function denseLayerWithReLU(input, weights, bias, output) {
  const M = 1, K = input.length, N = output.length;
  
  // Matrix multiplication: output = input × weights
  accelerate.matmul(input, weights, output, M, K, N);
  
  // Add bias
  accelerate.vadd(output, bias, output);
  
  // ReLU activation: max(0, x)
  const zeros = new Float64Array(N);
  accelerate.vclip(output, output, 0, Infinity);
  
  return output;
}

// Softmax activation
function softmax(logits, output) {
  // Subtract max for numerical stability
  const maxVal = accelerate.max(logits);
  const shifted = new Float64Array(logits.length);
  const negMax = -maxVal;
  
  for (let i = 0; i < logits.length; i++) {
    shifted[i] = logits[i] + negMax;
  }
  
  // Compute exp
  accelerate.vexp(shifted, output);
  
  // Normalize
  const sum = accelerate.sum(output);
  accelerate.vscale(output, 1.0 / sum, output);
  
  return output;
}
```

### Signal Processing & Audio

```javascript
// Apply windowed FFT for spectral analysis
function spectralAnalysis(audioBuffer, windowSize = 2048) {
  const window = accelerate.hanning(windowSize);
  const windowed = new Float64Array(windowSize);
  
  // Apply window
  accelerate.vmul(audioBuffer, window, windowed);
  
  // Compute FFT
  const spectrum = accelerate.fft(windowed);
  
  // Compute magnitude spectrum
  const magnitudes = new Float64Array(spectrum.real.length);
  for (let i = 0; i < magnitudes.length; i++) {
    magnitudes[i] = Math.sqrt(
      spectrum.real[i] ** 2 + spectrum.imag[i] ** 2
    );
  }
  
  // Convert to dB
  const dB = new Float64Array(magnitudes.length);
  accelerate.vlog10(magnitudes, dB);
  accelerate.vscale(dB, 20, dB);
  
  return dB;
}

// Low-pass filter using convolution
function lowPassFilter(signal, cutoffFreq, sampleRate) {
  // Design simple FIR filter kernel
  const kernelSize = 51;
  const kernel = new Float64Array(kernelSize);
  
  // Sinc function kernel
  const fc = cutoffFreq / sampleRate;
  for (let i = 0; i < kernelSize; i++) {
    const x = i - kernelSize / 2;
    if (x === 0) {
      kernel[i] = 2 * fc;
    } else {
      kernel[i] = Math.sin(2 * Math.PI * fc * x) / (Math.PI * x);
    }
  }
  
  // Apply Hamming window to kernel
  const window = accelerate.hamming(kernelSize);
  accelerate.vmul(kernel, window, kernel);
  
  // Normalize
  const sum = accelerate.sum(kernel);
  accelerate.vscale(kernel, 1.0 / sum, kernel);
  
  // Convolve
  const filtered = new Float64Array(signal.length - kernelSize + 1);
  accelerate.conv(signal, kernel, filtered);
  
  return filtered;
}
```

### Scientific Computing

```javascript
// Numerical integration using trapezoidal rule
function trapezoidalIntegration(f, a, b, n) {
  const h = (b - a) / n;
  const x = new Float64Array(n + 1);
  const y = new Float64Array(n + 1);
  
  // Generate points
  for (let i = 0; i <= n; i++) {
    x[i] = a + i * h;
    y[i] = f(x[i]);
  }
  
  // Trapezoidal rule: h * (y[0]/2 + y[1] + ... + y[n-1] + y[n]/2)
  const sum = accelerate.sum(y);
  return h * (sum - (y[0] + y[n]) / 2);
}

// Compute correlation coefficient
function correlationCoefficient(x, y) {
  const n = x.length;
  
  // Compute means
  const meanX = accelerate.mean(x);
  const meanY = accelerate.mean(y);
  
  // Center the data
  const xCentered = new Float64Array(n);
  const yCentered = new Float64Array(n);
  
  for (let i = 0; i < n; i++) {
    xCentered[i] = x[i] - meanX;
    yCentered[i] = y[i] - meanY;
  }
  
  // Compute correlation
  const numerator = accelerate.dot(xCentered, yCentered);
  const denomX = Math.sqrt(accelerate.sumOfSquares(xCentered));
  const denomY = Math.sqrt(accelerate.sumOfSquares(yCentered));
  
  return numerator / (denomX * denomY);
}

// Polynomial evaluation using Horner's method (vectorized)
function polyval(coefficients, x, result) {
  const n = x.length;
  const degree = coefficients.length - 1;
  
  // Initialize with highest degree coefficient
  for (let i = 0; i < n; i++) {
    result[i] = coefficients[degree];
  }
  
  // Horner's method: result = result * x + coeff
  for (let i = degree - 1; i >= 0; i--) {
    accelerate.vmul(result, x, result);
    
    const coeff = new Float64Array(n);
    coeff.fill(coefficients[i]);
    accelerate.vadd(result, coeff, result);
  }
  
  return result;
}
```

### Data Analysis & Statistics

```javascript
// Compute z-scores (standardization)
function zScore(data, output) {
  const mean = accelerate.mean(data);
  const std = accelerate.stddev(data);
  
  // z = (x - mean) / std
  for (let i = 0; i < data.length; i++) {
    output[i] = data[i] - mean;
  }
  accelerate.vscale(output, 1.0 / std, output);
  
  return output;
}

// Moving average filter
function movingAverage(data, windowSize) {
  const kernel = new Float64Array(windowSize);
  kernel.fill(1.0 / windowSize);
  
  const result = new Float64Array(data.length - windowSize + 1);
  accelerate.conv(data, kernel, result);
  
  return result;
}

// Outlier detection using IQR method
function detectOutliers(data) {
  const sorted = new Float64Array(data);
  // Note: You'd need to implement sorting or use JS sort
  
  const q1Index = Math.floor(data.length * 0.25);
  const q3Index = Math.floor(data.length * 0.75);
  
  const q1 = sorted[q1Index];
  const q3 = sorted[q3Index];
  const iqr = q3 - q1;
  
  const lowerBound = q1 - 1.5 * iqr;
  const upperBound = q3 + 1.5 * iqr;
  
  const outliers = [];
  for (let i = 0; i < data.length; i++) {
    if (data[i] < lowerBound || data[i] > upperBound) {
      outliers.push({ index: i, value: data[i] });
    }
  }
  
  return outliers;
}
```

### Image Processing

```javascript
// Gaussian blur (separable convolution)
function gaussianBlur(image, width, height, sigma) {
  // Generate 1D Gaussian kernel
  const kernelSize = Math.ceil(sigma * 6) | 1; // Ensure odd
  const kernel = new Float64Array(kernelSize);
  const center = Math.floor(kernelSize / 2);
  
  for (let i = 0; i < kernelSize; i++) {
    const x = i - center;
    kernel[i] = Math.exp(-(x * x) / (2 * sigma * sigma));
  }
  
  // Normalize
  const sum = accelerate.sum(kernel);
  accelerate.vscale(kernel, 1.0 / sum, kernel);
  
  // Horizontal pass
  const temp = new Float64Array(width * height);
  for (let y = 0; y < height; y++) {
    const row = image.subarray(y * width, (y + 1) * width);
    const outRow = temp.subarray(y * width, (y + 1) * width);
    // Convolve row (simplified - needs padding)
    accelerate.conv(row, kernel, outRow);
  }
  
  // Vertical pass (similar logic)
  // ...
  
  return temp;
}

// Edge detection using Sobel operator
function sobelEdgeDetection(image, width, height) {
  const sobelX = new Float64Array([-1, 0, 1, -2, 0, 2, -1, 0, 1]);
  const sobelY = new Float64Array([-1, -2, -1, 0, 0, 0, 1, 2, 1]);
  
  const gradX = new Float64Array(width * height);
  const gradY = new Float64Array(width * height);
  const magnitude = new Float64Array(width * height);
  
  // Apply Sobel kernels (simplified)
  // ... convolution logic ...
  
  // Compute gradient magnitude
  const gradXSq = new Float64Array(width * height);
  const gradYSq = new Float64Array(width * height);
  
  accelerate.vsquare(gradX, gradXSq);
  accelerate.vsquare(gradY, gradYSq);
  accelerate.vadd(gradXSq, gradYSq, magnitude);
  accelerate.vsqrt(magnitude, magnitude);
  
  return magnitude;
}
```

### Financial Analysis

```javascript
// Calculate returns and volatility
function calculateReturns(prices) {
  const returns = new Float64Array(prices.length - 1);
  
  for (let i = 1; i < prices.length; i++) {
    returns[i - 1] = (prices[i] - prices[i - 1]) / prices[i - 1];
  }
  
  const meanReturn = accelerate.mean(returns);
  const volatility = accelerate.stddev(returns);
  
  return { returns, meanReturn, volatility };
}

// Exponential moving average
function exponentialMovingAverage(data, alpha) {
  const ema = new Float64Array(data.length);
  ema[0] = data[0];
  
  for (let i = 1; i < data.length; i++) {
    ema[i] = alpha * data[i] + (1 - alpha) * ema[i - 1];
  }
  
  return ema;
}

// Bollinger Bands
function bollingerBands(prices, period, numStdDev) {
  const ma = movingAverage(prices, period);
  const upper = new Float64Array(ma.length);
  const lower = new Float64Array(ma.length);
  
  for (let i = 0; i < ma.length; i++) {
    const window = prices.subarray(i, i + period);
    const std = accelerate.stddev(window);
    upper[i] = ma[i] + numStdDev * std;
    lower[i] = ma[i] - numStdDev * std;
  }
  
  return { middle: ma, upper, lower };
}
```

## Benchmarking

Run the included benchmarks:

```bash
npm run benchmark
```

Run tests:

```bash
npm test
```

Compare with pure JavaScript:

```bash
npm run compare
```

## Performance Tips

1. **Reuse buffers** - Allocate Float64Arrays once and reuse them
2. **Batch operations** - Process large arrays instead of many small ones
3. **Use appropriate sizes** - Accelerate shines with larger data (1000+ elements)
4. **Profile your code** - Not all operations benefit equally
5. **Use windows for FFT** - Apply Hanning/Hamming windows before FFT for better spectral analysis
6. **Leverage vectorized trig** - Use vsin/vcos/vtan instead of loops with Math.sin/cos/tan
7. **Chain operations** - Minimize intermediate allocations

## Complete Function Reference

### Matrix Operations (BLAS)
- `matmul(A, B, C, M, K, N)` - Matrix multiplication (double precision)
- `matmulFloat(A, B, C, M, K, N)` - Matrix multiplication (single precision)
- `matvec(A, x, y, M, N)` - Matrix-vector multiplication
- `transpose(A, B, rows, cols)` - Matrix transpose
- `axpy(alpha, x, y)` - AXPY operation (y = alpha*x + y)
- `copy(x, y)` - Vector copy
- `swap(x, y)` - Vector swap
- `norm(x)` - L2 norm (Euclidean length)
- `abssum(x)` - Sum of absolute values
- `maxAbsIndex(x)` - Index of maximum absolute value
- `rot(x, y, c, s)` - Givens rotation

### Vector Arithmetic
- `dot(a, b)` - Dot product
- `vadd(a, b, out)` - Element-wise addition
- `vsub(a, b, out)` - Element-wise subtraction
- `vmul(a, b, out)` - Element-wise multiplication
- `vdiv(a, b, out)` - Element-wise division
- `vscale(a, scalar, b)` - Scalar multiplication
- `vneg(a, b)` - Negation
- `vaddScalar(a, scalar, c)` - Add scalar to vector
- `vma(a, b, c, d)` - Multiply-add: d = (a*b) + c
- `vmsa(a, b, c, d)` - Multiply-scalar-add: d = (a*b) + c

### Vector Functions
- `vabs(a, b)` - Absolute value
- `vsquare(a, b)` - Square
- `vsqrt(a, b)` - Square root
- `normalize(a, b)` - Normalize to unit length
- `vreverse(a, b)` - Reverse order
- `vfill(scalar, vec)` - Fill with scalar
- `vramp(start, step, vec)` - Generate linear ramp
- `vlerp(a, b, t, c)` - Linear interpolation
- `vclear(vec)` - Clear (set to zero)
- `vlimit(a, low, high, c)` - Limit/saturate values

### Trigonometric (Vectorized)
- `vsin(a, b)` - Sine
- `vcos(a, b)` - Cosine
- `vtan(a, b)` - Tangent
- `vasin(a, b)` - Inverse sine
- `vacos(a, b)` - Inverse cosine
- `vatan(a, b)` - Inverse tangent
- `vatan2(y, x, out)` - Two-argument arctangent

### Hyperbolic Functions
- `vsinh(a, b)` - Hyperbolic sine
- `vcosh(a, b)` - Hyperbolic cosine
- `vtanh(a, b)` - Hyperbolic tangent

### Exponential & Logarithmic
- `vexp(a, b)` - Natural exponential
- `vlog(a, b)` - Natural logarithm
- `vlog10(a, b)` - Base-10 logarithm
- `vpow(a, b, c)` - Power (c = a^b)
- `vreciprocal(a, b)` - Reciprocal (1/x)
- `vrsqrt(a, b)` - Inverse square root (1/sqrt(x))

### Rounding Functions
- `vceil(a, b)` - Ceiling
- `vfloor(a, b)` - Floor
- `vtrunc(a, b)` - Truncate (round toward zero)
- `vcopysign(a, b, c)` - Copy sign

### Data Processing
- `vclip(a, b, min, max)` - Clip to range
- `vthreshold(a, b, threshold)` - Apply threshold

### Statistical Functions
- `sum(vec)` - Sum of elements
- `mean(vec)` - Mean (average)
- `variance(vec)` - Variance
- `stddev(vec)` - Standard deviation
- `max(vec)` - Maximum value
- `min(vec)` - Minimum value
- `minmax(vec)` - Both min and max
- `rms(vec)` - Root mean square
- `sumOfSquares(vec)` - Sum of squares
- `meanMagnitude(vec)` - Mean of absolute values
- `meanSquare(vec)` - Mean of squares
- `maxMagnitude(vec)` - Maximum magnitude
- `minMagnitude(vec)` - Minimum magnitude

### Distance Metrics
- `euclidean(a, b)` - Euclidean distance

### Signal Processing
- `fft(signal)` - Fast Fourier Transform
- `ifft(real, imag)` - Inverse FFT
- `conv(signal, kernel, result)` - Convolution
- `xcorr(a, b, result)` - Cross-correlation

### Window Functions
- `hamming(length)` - Hamming window
- `hanning(length)` - Hanning window
- `blackman(length)` - Blackman window

### Interpolation
- `interp1d(x, y, xi, yi)` - Linear interpolation

**Total: 80+ functions** - All hardware-accelerated via Apple's Accelerate framework

## Limitations

- **macOS only** - Requires Apple's Accelerate framework
- **Float64Array only** - Currently supports double precision only
- **Row-major order** - Matrices must be in row-major format
- **FFT size** - Must be power of 2

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT © Jessica Mulein

## Acknowledgments

Built on Apple's Accelerate framework. Inspired by the need for high-performance numerical computing in Node.js on Apple Silicon.

## Troubleshooting

### "Cannot find module '@digitaldefiance/node-accelerate'"

Make sure you installed it:
```bash
npm install @digitaldefiance/node-accelerate
```

### "Error: Module did not self-register"

Rebuild the addon:
```bash
npm rebuild @digitaldefiance/node-accelerate
```

### "node-accelerate requires macOS"

This package only works on macOS because it uses Apple's Accelerate framework. It cannot run on Linux or Windows.

### Build fails with "gyp: No Xcode or CLT version detected"

Install Xcode Command Line Tools:
```bash
xcode-select --install
```

### "Unsupported architecture"

node-accelerate supports:
- ARM64 (Apple Silicon: M1/M2/M3/M4)
- x64 (Intel Macs)

If you're on an older Mac with a different architecture, this package won't work.

### Performance seems slow

1. Make sure you're using large arrays (1000+ elements)
2. Reuse buffers instead of allocating new ones
3. Run `npm run compare` to see actual speedups on your machine
4. Check that you're not running in a VM or emulator

## Related Projects

- [Apple Accelerate Documentation](https://developer.apple.com/documentation/accelerate)
- [BLAS Reference](http://www.netlib.org/blas/)
- [vDSP Reference](https://developer.apple.com/documentation/accelerate/vdsp)

## Support

- [GitHub Issues](https://github.com/Digital-Defiance/node-accelerate/issues)
- [npm Package](https://www.npmjs.com/package/@digitaldefiance/node-accelerate)

---

**Made with ❤️ for Apple Silicon**
