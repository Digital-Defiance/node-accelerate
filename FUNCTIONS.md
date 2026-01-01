# Complete Function Reference

This document lists all functions available in node-accelerate v2.0.0.

## Matrix Operations (BLAS)

| Function | Description | Signature |
|----------|-------------|-----------|
| `matmul` | Matrix multiplication (double) | `(A, B, C, M, K, N) => C` |
| `matmulFloat` | Matrix multiplication (single) | `(A, B, C, M, K, N) => C` |
| `matvec` | Matrix-vector multiplication | `(A, x, y, M, N) => y` |
| `transpose` | Matrix transpose | `(A, B, rows, cols) => B` |
| `axpy` | AXPY: y = alpha*x + y | `(alpha, x, y) => y` |
| `copy` | Vector copy | `(x, y) => y` |
| `swap` | Vector swap | `(x, y) => x` |
| `norm` | L2 norm (Euclidean length) | `(x) => number` |
| `abssum` | Sum of absolute values | `(x) => number` |
| `maxAbsIndex` | Index of max absolute value | `(x) => number` |
| `rot` | Givens rotation | `(x, y, c, s) => x` |

## Vector Arithmetic

| Function | Description | Signature |
|----------|-------------|-----------|
| `dot` | Dot product | `(a, b) => number` |
| `vadd` | Element-wise addition | `(a, b, out) => out` |
| `vsub` | Element-wise subtraction | `(a, b, out) => out` |
| `vmul` | Element-wise multiplication | `(a, b, out) => out` |
| `vdiv` | Element-wise division | `(a, b, out) => out` |
| `vscale` | Scalar multiplication | `(a, scalar, b) => b` |
| `vneg` | Negation | `(a, b) => b` |
| `vaddScalar` | Add scalar to vector | `(a, scalar, c) => c` |
| `vma` | Multiply-add: d = (a*b) + c | `(a, b, c, d) => d` |
| `vmsa` | Multiply-scalar-add | `(a, b, c, d) => d` |

## Vector Functions

| Function | Description | Signature |
|----------|-------------|-----------|
| `vabs` | Absolute value | `(a, b) => b` |
| `vsquare` | Square | `(a, b) => b` |
| `vsqrt` | Square root | `(a, b) => b` |
| `normalize` | Normalize to unit length | `(a, b) => b` |
| `vreverse` | Reverse order | `(a, b) => b` |
| `vfill` | Fill with scalar | `(scalar, vec) => vec` |
| `vramp` | Generate linear ramp | `(start, step, vec) => vec` |
| `vlerp` | Linear interpolation | `(a, b, t, c) => c` |
| `vclear` | Clear (set to zero) | `(vec) => vec` |
| `vlimit` | Limit/saturate values | `(a, low, high, c) => c` |

## Trigonometric Functions (Vectorized)

| Function | Description | Signature |
|----------|-------------|-----------|
| `vsin` | Sine | `(a, b) => b` |
| `vcos` | Cosine | `(a, b) => b` |
| `vtan` | Tangent | `(a, b) => b` |
| `vasin` | Inverse sine | `(a, b) => b` |
| `vacos` | Inverse cosine | `(a, b) => b` |
| `vatan` | Inverse tangent | `(a, b) => b` |
| `vatan2` | Two-argument arctangent | `(y, x, out) => out` |

## Hyperbolic Functions

| Function | Description | Signature |
|----------|-------------|-----------|
| `vsinh` | Hyperbolic sine | `(a, b) => b` |
| `vcosh` | Hyperbolic cosine | `(a, b) => b` |
| `vtanh` | Hyperbolic tangent | `(a, b) => b` |

## Exponential & Logarithmic Functions

| Function | Description | Signature |
|----------|-------------|-----------|
| `vexp` | Natural exponential | `(a, b) => b` |
| `vlog` | Natural logarithm | `(a, b) => b` |
| `vlog10` | Base-10 logarithm | `(a, b) => b` |
| `vpow` | Power (c = a^b) | `(a, b, c) => c` |
| `vreciprocal` | Reciprocal (1/x) | `(a, b) => b` |
| `vrsqrt` | Inverse square root | `(a, b) => b` |

## Rounding Functions

| Function | Description | Signature |
|----------|-------------|-----------|
| `vceil` | Ceiling | `(a, b) => b` |
| `vfloor` | Floor | `(a, b) => b` |
| `vtrunc` | Truncate (round toward zero) | `(a, b) => b` |
| `vcopysign` | Copy sign | `(a, b, c) => c` |

## Data Processing

| Function | Description | Signature |
|----------|-------------|-----------|
| `vclip` | Clip to range | `(a, b, min, max) => b` |
| `vthreshold` | Apply threshold | `(a, b, threshold) => b` |

## Statistical Functions

| Function | Description | Signature |
|----------|-------------|-----------|
| `sum` | Sum of elements | `(vec) => number` |
| `mean` | Mean (average) | `(vec) => number` |
| `variance` | Variance | `(vec) => number` |
| `stddev` | Standard deviation | `(vec) => number` |
| `max` | Maximum value | `(vec) => number` |
| `min` | Minimum value | `(vec) => number` |
| `minmax` | Both min and max | `(vec) => {min, max}` |
| `rms` | Root mean square | `(vec) => number` |
| `sumOfSquares` | Sum of squares | `(vec) => number` |
| `meanMagnitude` | Mean of absolute values | `(vec) => number` |
| `meanSquare` | Mean of squares | `(vec) => number` |
| `maxMagnitude` | Maximum magnitude | `(vec) => number` |
| `minMagnitude` | Minimum magnitude | `(vec) => number` |

## Distance Metrics

| Function | Description | Signature |
|----------|-------------|-----------|
| `euclidean` | Euclidean distance | `(a, b) => number` |

## Signal Processing

| Function | Description | Signature |
|----------|-------------|-----------|
| `fft` | Fast Fourier Transform | `(signal) => {real, imag}` |
| `ifft` | Inverse FFT | `(real, imag) => signal` |
| `conv` | Convolution | `(signal, kernel, result) => result` |
| `xcorr` | Cross-correlation | `(a, b, result) => result` |

## Window Functions

| Function | Description | Signature |
|----------|-------------|-----------|
| `hamming` | Hamming window | `(length) => window` |
| `hanning` | Hanning window | `(length) => window` |
| `blackman` | Blackman window | `(length) => window` |

## Interpolation

| Function | Description | Signature |
|----------|-------------|-----------|
| `interp1d` | Linear interpolation | `(x, y, xi, yi) => yi` |

## Total Functions: 80+

All functions use Apple's Accelerate framework for hardware-optimized performance on macOS.

### Performance Characteristics

- **Matrix operations**: 100-300x faster than pure JavaScript
- **Vector operations**: 3-10x faster than pure JavaScript
- **Trigonometric**: 5-10x faster than Math.sin/cos/tan in loops
- **FFT**: 10-50x faster than pure JavaScript implementations

### Memory Efficiency

- Zero-copy operations where possible
- In-place operations supported
- Efficient use of SIMD instructions
- Optimized for Apple Silicon (M1/M2/M3/M4)
