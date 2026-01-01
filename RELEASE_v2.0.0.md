# node-accelerate v2.0.0 Release

## 🚀 Major Release: 80+ Hardware-Accelerated Functions

node-accelerate v2.0.0 is a massive expansion bringing **80+ hardware-accelerated functions** from Apple's Accelerate framework to Node.js.

## What's New

### Complete Function Suite

- **11 BLAS operations** - Matrix operations, vector norms, rotations
- **10 vDSP utilities** - Fill, ramp, multiply-add, interpolation
- **10 Trigonometric functions** - Standard, inverse, and hyperbolic
- **6 Exponential/logarithmic** - exp, log, pow, reciprocal, inverse sqrt
- **4 Rounding functions** - ceil, floor, trunc, copysign
- **13 Statistical functions** - Complete statistical analysis suite
- **7 Signal processing** - FFT/IFFT, convolution, correlation, windows
- **Plus**: Data processing, interpolation, and more

### Testing

- **89 comprehensive tests**
- **100% pass rate**
- Every function tested and verified

### Performance

All functions leverage Apple's Accelerate framework:
- **Matrix operations**: 100-300x faster
- **Vector operations**: 3-10x faster
- **Trigonometric**: 5-10x faster than Math functions
- **FFT**: 10-50x faster than pure JavaScript

### Compatibility

- ✅ **100% backward compatible** with v1.x
- ✅ macOS only (Apple Silicon & Intel)
- ✅ Node.js >= 18.0.0
- ✅ Full TypeScript support

## Installation

```bash
npm install @digitaldefiance/node-accelerate@2.0.0
```

## Quick Example

```javascript
const accelerate = require('@digitaldefiance/node-accelerate');

// Trigonometric operations (5-10x faster)
const angles = new Float64Array(1000);
const sines = new Float64Array(1000);
accelerate.vsin(angles, sines);

// Statistical analysis
const data = new Float64Array(10000);
const stats = {
  mean: accelerate.mean(data),
  stddev: accelerate.stddev(data),
  variance: accelerate.variance(data),
  ...accelerate.minmax(data)
};

// Signal processing
const signal = new Float64Array(2048);
const window = accelerate.hanning(2048);
const windowed = new Float64Array(2048);
accelerate.vmul(signal, window, windowed);
const spectrum = accelerate.fft(windowed);
```

## Documentation

- [Complete API Reference](README.md#api-reference)
- [Function Reference Table](FUNCTIONS.md)
- [Quick Reference](QUICK_REFERENCE.md)
- [Examples](examples/)
- [Changelog](CHANGELOG.md)

## Links

- [GitHub](https://github.com/Digital-Defiance/node-accelerate)
- [npm](https://www.npmjs.com/package/@digitaldefiance/node-accelerate)
- [Showcase](https://digital-defiance.github.io/node-accelerate/)

---

**Made with ❤️ for Apple Silicon**
