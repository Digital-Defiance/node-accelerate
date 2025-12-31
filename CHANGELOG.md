# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
- Float32 support for single-precision operations
- Additional BLAS operations (GEMV, triangular solve)
- Additional vDSP operations (convolution, correlation)
- Improved error handling and validation
- More comprehensive benchmarks
- CI/CD pipeline

---

[1.0.0]: https://github.com/Digital-Defiance/node-accelerate/releases/tag/v1.0.0
