# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.0.0] - 2024-12-31

### Added - Major Feature Expansion (80+ Functions)

#### Matrix Operations (BLAS)
- `transpose(A, B, rows, cols)` - Matrix transpose operation
- `copy(x, y)` - Vector copy
- `swap(x, y)` - Vector swap
- `norm(x)` - L2 norm (Euclidean length)
- `abssum(x)` - Sum of absolute values
- `maxAbsIndex(x)` - Index of maximum absolute value
- `rot(x, y, c, s)` - Givens rotation

#### Vector Arithmetic (vDSP)
- `vneg(a, b)` - Vector negation
- `vaddScalar(a, scalar, c)` - Add scalar to vector
- `vma(a, b, c, d)` - Multiply-add: d = (a * b) + c
- `vmsa(a, b, c, d)` - Multiply-scalar-add: d = (a * b) + c

#### Vector Utilities (vDSP)
- `vfill(scalar, vec)` - Fill vector with scalar
- `vramp(start, step, vec)` - Generate linear ramp
- `vlerp(a, b, t, c)` - Linear interpolation between vectors
- `vclear(vec)` - Clear vector (set to zero)
- `vlimit(a, low, high, c)` - Limit/saturate values
- `vreverse(a, b)` - Reverse vector order

#### Trigonometric Functions (vForce)
- `vsin(a, b)`, `vcos(a, b)`, `vtan(a, b)` - Standard trig
- `vasin(a, b)`, `vacos(a, b)`, `vatan(a, b)` - Inverse trig
- `vatan2(y, x, out)` - Two-argument arctangent
- `vsinh(a, b)`, `vcosh(a, b)`, `vtanh(a, b)` - Hyperbolic functions

#### Exponential & Logarithmic Functions (vForce)
- `vexp(a, b)` - Natural exponential
- `vlog(a, b)`, `vlog10(a, b)` - Logarithms
- `vpow(a, b, c)` - Element-wise power
- `vreciprocal(a, b)` - Reciprocal (1/x)
- `vrsqrt(a, b)` - Inverse square root (1/sqrt(x))

#### Rounding Functions (vForce)
- `vceil(a, b)` - Ceiling
- `vfloor(a, b)` - Floor
- `vtrunc(a, b)` - Truncate (round toward zero)
- `vcopysign(a, b, c)` - Copy sign

#### Statistical Functions (vDSP)
- `variance(vec)` - Variance
- `stddev(vec)` - Standard deviation
- `minmax(vec)` - Both min and max
- `sumOfSquares(vec)` - Sum of squared elements
- `meanMagnitude(vec)` - Mean of absolute values
- `meanSquare(vec)` - Mean of squared values
- `maxMagnitude(vec)` - Maximum magnitude
- `minMagnitude(vec)` - Minimum magnitude

#### Signal Processing (vDSP)
- `ifft(real, imag)` - Inverse Fast Fourier Transform
- `conv(signal, kernel, result)` - 1D convolution
- `xcorr(a, b, result)` - Cross-correlation
- `hamming(length)`, `hanning(length)`, `blackman(length)` - Window functions

#### Data Processing
- `vclip(a, b, min, max)` - Clip values to range
- `vthreshold(a, b, threshold)` - Apply threshold
- `interp1d(x, y, xi, yi)` - Linear interpolation

### Testing
- Comprehensive test suite with 89 tests
- All 80+ functions tested and verified
- 100% test pass rate

### Documentation
- Complete API reference for all 80+ functions
- 8 example files demonstrating real-world use cases
- Added `FUNCTIONS.md` - Complete function reference table
- Expanded README with detailed use cases

### Performance
- All functions leverage Apple's Accelerate framework
- Hardware-accelerated via AMX and NEON SIMD
- 5-10x speedup for trigonometric operations
- 10-50x speedup for FFT operations
- Optimized for Apple Silicon (M1/M2/M3/M4)

### Breaking Changes
None - All existing APIs remain unchanged and backward compatible

## [1.0.0] - 2024-12-30

### Added
- Initial release
- Matrix operations:
  - Matrix multiplication (BLAS GEMM) with 283x speedup
  - Matrix-vector multiplication (BLAS GEMV)
  - AXPY operation (y = alpha*x + y)
- Vector arithmetic (vDSP):
  - Dot product (5x speedup)
  - Addition, subtraction, multiplication, division
  - Scalar multiplication
- Vector functions (vDSP):
  - Absolute value
  - Square, square root
  - Normalize to unit length
- Reductions (vDSP):
  - Sum (7.6x speedup)
  - Mean, max, min
  - Root Mean Square (RMS)
- Distance metrics:
  - Euclidean distance
- Signal processing:
  - Fast Fourier Transform (FFT)
- Full TypeScript definitions
- Comprehensive test suite (26 tests)
- Performance benchmarks
- Complete documentation

### Performance
- Matrix multiply (500×500): 93ms → 0.33ms (283x faster)
- Vector dot product (1M): 0.66ms → 0.13ms (5x faster)
- Vector sum (1M): 0.59ms → 0.08ms (7.6x faster)
- Vector add (1M): 0.74ms → 0.20ms (3.7x faster)

### Requirements
- macOS (Apple Silicon or Intel)
- Node.js >= 18.0.0
- Xcode Command Line Tools

## [Unreleased]

### Planned
- Complex number operations
- Sparse matrix support
- Additional BLAS Level 2 & 3 operations
- GPU acceleration via Metal
- Streaming/chunked operations for large datasets

---

[2.0.0]: https://github.com/Digital-Defiance/node-accelerate/releases/tag/v2.0.0
[1.0.0]: https://github.com/Digital-Defiance/node-accelerate/releases/tag/v1.0.0
