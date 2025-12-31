# node-accelerate

High-performance Apple Accelerate framework bindings for Node.js. Get **283x faster** matrix operations and **5-8x faster** vector operations on Apple Silicon (M1/M2/M3/M4).

[![npm version](https://badge.fury.io/js/node-accelerate.svg)](https://www.npmjs.com/package/node-accelerate)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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
| Matrix Multiply (500×500) | 93 ms | 0.33 ms | **283x** |
| Vector Dot Product (1M) | 0.66 ms | 0.13 ms | **5x** |
| Vector Sum (1M) | 0.59 ms | 0.08 ms | **7.6x** |
| Vector Add (1M) | 0.74 ms | 0.20 ms | **3.7x** |

## Installation

```bash
npm install node-accelerate
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
- Try rebuilding: `npm rebuild node-accelerate`
- Ensure Xcode Command Line Tools are installed

### Verifying Installation

```bash
node -e "const a = require('node-accelerate'); console.log('✓ Works!')"
```

## Quick Start

```javascript
const accelerate = require('node-accelerate');

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

// FFT
const signal = new Float64Array(65536);
for (let i = 0; i < signal.length; i++) {
  signal[i] = Math.sin(2 * Math.PI * i / signal.length);
}
const spectrum = accelerate.fft(signal);
console.log(spectrum.real, spectrum.imag);
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

#### `matvec(A, x, y, M, N)`

Matrix-vector multiplication: y = A × x

- `A`: Float64Array - Matrix (M × N) in row-major order
- `x`: Float64Array - Input vector (N elements)
- `y`: Float64Array - Output vector (M elements)
- `M`: number - Rows in A
- `N`: number - Columns in A
- Returns: Float64Array (y)

**Example:**
```javascript
const M = 100, N = 50;
const A = new Float64Array(M * N);
const x = new Float64Array(N);
const y = new Float64Array(M);

accelerate.matvec(A, x, y, M, N);
```

#### `axpy(alpha, x, y)`

AXPY operation: y = alpha*x + y

- `alpha`: number - Scalar multiplier
- `x`: Float64Array - Input vector
- `y`: Float64Array - Input/output vector
- Returns: Float64Array (y)

**Example:**
```javascript
const x = new Float64Array([1, 2, 3]);
const y = new Float64Array([4, 5, 6]);
accelerate.axpy(2.0, x, y); // y = [6, 9, 12]
```

### Vector Operations (vDSP)

#### `dot(a, b)`

Dot product: sum(a[i] * b[i])

- `a`: Float64Array - First vector
- `b`: Float64Array - Second vector (same length as a)
- Returns: number

**Example:**
```javascript
const a = new Float64Array([1, 2, 3, 4]);
const b = new Float64Array([5, 6, 7, 8]);
const result = accelerate.dot(a, b); // 70
```

#### `sum(vec)`

Sum of all elements

- `vec`: Float64Array - Input vector
- Returns: number

**Example:**
```javascript
const vec = new Float64Array([1, 2, 3, 4, 5]);
const result = accelerate.sum(vec); // 15
```

#### `mean(vec)`

Mean (average) of all elements

- `vec`: Float64Array - Input vector
- Returns: number

**Example:**
```javascript
const vec = new Float64Array([1, 2, 3, 4, 5]);
const result = accelerate.mean(vec); // 3
```

#### `vadd(a, b, out)`

Element-wise addition: out[i] = a[i] + b[i]

- `a`: Float64Array - First vector
- `b`: Float64Array - Second vector
- `out`: Float64Array - Output vector
- Returns: Float64Array (out)

**Example:**
```javascript
const a = new Float64Array([1, 2, 3]);
const b = new Float64Array([4, 5, 6]);
const out = new Float64Array(3);
accelerate.vadd(a, b, out); // out = [5, 7, 9]
```

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

#### `vabs(a, b)`

Element-wise absolute value: b[i] = |a[i]|

- `a`: Float64Array - Input vector
- `b`: Float64Array - Output vector
- Returns: Float64Array (b)

**Example:**
```javascript
const a = new Float64Array([-1, -2, 3, -4]);
const b = new Float64Array(4);
accelerate.vabs(a, b); // b = [1, 2, 3, 4]
```

#### `vsquare(a, b)`

Element-wise square: b[i] = a[i]^2

- `a`: Float64Array - Input vector
- `b`: Float64Array - Output vector
- Returns: Float64Array (b)

**Example:**
```javascript
const a = new Float64Array([2, 3, 4]);
const b = new Float64Array(3);
accelerate.vsquare(a, b); // b = [4, 9, 16]
```

#### `vsqrt(a, b)`

Element-wise square root: b[i] = sqrt(a[i])

- `a`: Float64Array - Input vector
- `b`: Float64Array - Output vector
- Returns: Float64Array (b)

**Example:**
```javascript
const a = new Float64Array([4, 9, 16]);
const b = new Float64Array(3);
accelerate.vsqrt(a, b); // b = [2, 3, 4]
```

#### `normalize(a, b)`

Normalize vector to unit length: b = a / ||a||

- `a`: Float64Array - Input vector
- `b`: Float64Array - Output vector (unit vector)
- Returns: Float64Array (b)

**Example:**
```javascript
const a = new Float64Array([3, 4, 0]);
const b = new Float64Array(3);
accelerate.normalize(a, b); // b = [0.6, 0.8, 0]
```

### Reductions

#### `rms(vec)`

Root Mean Square: sqrt(sum(vec[i]^2) / n)

- `vec`: Float64Array - Input vector
- Returns: number

**Example:**
```javascript
const vec = new Float64Array([1, 2, 3, 4, 5]);
const result = accelerate.rms(vec); // 3.317
```

### Distance Metrics

#### `euclidean(a, b)`

Euclidean distance: sqrt(sum((a[i] - b[i])^2))

- `a`: Float64Array - First vector
- `b`: Float64Array - Second vector
- Returns: number

**Example:**
```javascript
const a = new Float64Array([0, 0, 0]);
const b = new Float64Array([3, 4, 0]);
const distance = accelerate.euclidean(a, b); // 5
```

### Signal Processing

#### `fft(signal)`

Fast Fourier Transform

- `signal`: Float64Array - Input signal (length must be power of 2)
- Returns: Object with `real` and `imag` Float64Arrays

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
// Matrix multiplication for neural network layers
function denseLayer(input, weights, bias) {
  const output = new Float64Array(weights.length / input.length);
  accelerate.matmul(
    input, weights, output,
    1, input.length, output.length
  );
  // Add bias...
  return output;
}
```

### Signal Processing

```javascript
// Analyze audio spectrum
function analyzeAudio(audioBuffer) {
  const spectrum = accelerate.fft(audioBuffer);
  const magnitudes = new Float64Array(spectrum.real.length);
  
  for (let i = 0; i < magnitudes.length; i++) {
    magnitudes[i] = Math.sqrt(
      spectrum.real[i] ** 2 + spectrum.imag[i] ** 2
    );
  }
  
  return magnitudes;
}
```

### Scientific Computing

```javascript
// Numerical integration using vector operations
function integrate(f, a, b, n) {
  const h = (b - a) / n;
  const x = new Float64Array(n);
  const y = new Float64Array(n);
  
  for (let i = 0; i < n; i++) {
    x[i] = a + i * h;
    y[i] = f(x[i]);
  }
  
  return h * accelerate.sum(y);
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

## Performance Tips

1. **Reuse buffers** - Allocate Float64Arrays once and reuse them
2. **Batch operations** - Process large arrays instead of many small ones
3. **Use appropriate sizes** - Accelerate shines with larger data (1000+ elements)
4. **Profile your code** - Not all operations benefit equally

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

### "Cannot find module 'node-accelerate'"

Make sure you installed it:
```bash
npm install node-accelerate
```

### "Error: Module did not self-register"

Rebuild the addon:
```bash
npm rebuild node-accelerate
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
