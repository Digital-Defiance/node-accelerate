# Quick Start Guide

Get up and running with `node-accelerate` in 5 minutes.

## Installation

```bash
npm install node-accelerate
```

## Your First Program

Create `test.js`:

```javascript
const accelerate = require('node-accelerate');

// Matrix multiplication
const M = 100, K = 100, N = 100;
const A = new Float64Array(M * K);
const B = new Float64Array(K * N);
const C = new Float64Array(M * N);

// Fill with random data
for (let i = 0; i < A.length; i++) A[i] = Math.random();
for (let i = 0; i < B.length; i++) B[i] = Math.random();

// Multiply!
console.time('matmul');
accelerate.matmul(A, B, C, M, K, N);
console.timeEnd('matmul');

console.log('✓ Matrix multiplication complete!');
```

Run it:

```bash
node test.js
```

## Common Operations

### Vector Math

```javascript
const a = new Float64Array([1, 2, 3, 4, 5]);
const b = new Float64Array([6, 7, 8, 9, 10]);
const result = new Float64Array(5);

// Addition
accelerate.vadd(a, b, result);
console.log(result); // [7, 9, 11, 13, 15]

// Dot product
const dot = accelerate.dot(a, b);
console.log(dot); // 130

// Sum
const sum = accelerate.sum(a);
console.log(sum); // 15

// Mean
const mean = accelerate.mean(a);
console.log(mean); // 3
```

### Signal Processing

```javascript
// Create a signal
const signal = new Float64Array(1024);
for (let i = 0; i < signal.length; i++) {
  signal[i] = Math.sin(2 * Math.PI * i / signal.length);
}

// Perform FFT
const spectrum = accelerate.fft(signal);
console.log(spectrum.real.length); // 512
console.log(spectrum.imag.length); // 512
```

## Performance Tips

1. **Reuse buffers** - Don't allocate new arrays in loops
2. **Use large arrays** - Overhead is amortized over more data
3. **Batch operations** - Process 1M elements once, not 1 element 1M times

## Examples

Check the `examples/` directory:

```bash
node examples/matrix-multiply.js
node examples/vector-operations.js
node examples/signal-processing.js
```

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

### Build fails

Make sure you have Xcode Command Line Tools:
```bash
xcode-select --install
```

## Next Steps

- Read the [full documentation](README.md)
- Run the [benchmarks](benchmark.js)
- Check out [examples](examples/)
- Read the [API reference](README.md#api-reference)

## Need Help?

- [GitHub Issues](https://github.com/Digital-Defiance/node-accelerate/issues)
- [npm Package](https://www.npmjs.com/package/@digitaldefiance/node-accelerate)

---

**Happy computing!** 🚀
