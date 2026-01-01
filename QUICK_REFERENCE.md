# node-accelerate Quick Reference

## Installation
```bash
npm install @digitaldefiance/node-accelerate
```

## Import
```javascript
const accelerate = require('@digitaldefiance/node-accelerate');
```

## Matrix Operations
```javascript
// Matrix multiply: C = A × B
accelerate.matmul(A, B, C, M, K, N);

// Matrix-vector: y = A × x
accelerate.matvec(A, x, y, M, N);

// Transpose: B = A^T
accelerate.transpose(A, B, rows, cols);

// AXPY: y = alpha*x + y
accelerate.axpy(2.0, x, y);
```

## Vector Arithmetic
```javascript
accelerate.dot(a, b);              // Dot product
accelerate.vadd(a, b, out);        // out = a + b
accelerate.vsub(a, b, out);        // out = a - b
accelerate.vmul(a, b, out);        // out = a * b
accelerate.vdiv(a, b, out);        // out = a / b
accelerate.vscale(a, 2.0, out);    // out = a * 2.0
accelerate.vneg(a, out);           // out = -a
```

## Vector Functions
```javascript
accelerate.vabs(a, out);           // out = |a|
accelerate.vsquare(a, out);        // out = a²
accelerate.vsqrt(a, out);          // out = √a
accelerate.normalize(a, out);      // out = a / ||a||
accelerate.vreverse(a, out);       // out = reverse(a)
```

## Trigonometry (Vectorized)
```javascript
accelerate.vsin(angles, out);      // out = sin(angles)
accelerate.vcos(angles, out);      // out = cos(angles)
accelerate.vtan(angles, out);      // out = tan(angles)
```

## Exponential & Logarithmic
```javascript
accelerate.vexp(a, out);           // out = e^a
accelerate.vlog(a, out);           // out = ln(a)
accelerate.vlog10(a, out);         // out = log₁₀(a)
accelerate.vpow(a, b, out);        // out = a^b
```

## Statistics
```javascript
accelerate.sum(vec);               // Sum
accelerate.mean(vec);              // Mean
accelerate.variance(vec);          // Variance
accelerate.stddev(vec);            // Standard deviation
accelerate.max(vec);               // Maximum
accelerate.min(vec);               // Minimum
accelerate.minmax(vec);            // {min, max}
accelerate.rms(vec);               // Root mean square
accelerate.sumOfSquares(vec);      // Σ(x²)
accelerate.meanMagnitude(vec);     // mean(|x|)
accelerate.meanSquare(vec);        // mean(x²)
```

## Data Processing
```javascript
// Clip to range
accelerate.vclip(data, out, -10, 10);

// Threshold
accelerate.vthreshold(data, out, 5.0);

// Interpolate
accelerate.interp1d(x, y, xi, yi);
```

## Signal Processing
```javascript
// FFT
const spectrum = accelerate.fft(signal);
// spectrum.real, spectrum.imag

// Inverse FFT
const signal = accelerate.ifft(real, imag);

// Convolution
accelerate.conv(signal, kernel, result);

// Cross-correlation
accelerate.xcorr(sig1, sig2, result);
```

## Window Functions
```javascript
const hamming = accelerate.hamming(256);
const hanning = accelerate.hanning(256);
const blackman = accelerate.blackman(256);

// Apply window
accelerate.vmul(signal, window, windowed);
```

## Distance
```javascript
const dist = accelerate.euclidean(point1, point2);
```

## Common Patterns

### Normalize Data (Z-score)
```javascript
const mean = accelerate.mean(data);
const std = accelerate.stddev(data);
for (let i = 0; i < data.length; i++) {
  normalized[i] = (data[i] - mean) / std;
}
```

### Softmax
```javascript
const max = accelerate.max(logits);
const shifted = new Float64Array(logits.length);
for (let i = 0; i < logits.length; i++) {
  shifted[i] = logits[i] - max;
}
accelerate.vexp(shifted, probs);
const sum = accelerate.sum(probs);
accelerate.vscale(probs, 1.0 / sum, probs);
```

### ReLU
```javascript
accelerate.vclip(input, output, 0, Infinity);
```

### Windowed FFT
```javascript
const window = accelerate.hanning(signal.length);
const windowed = new Float64Array(signal.length);
accelerate.vmul(signal, window, windowed);
const spectrum = accelerate.fft(windowed);
```

### Moving Average
```javascript
const kernel = new Float64Array([0.25, 0.5, 0.25]);
const filtered = new Float64Array(signal.length - 2);
accelerate.conv(signal, kernel, filtered);
```

## Performance Tips

1. **Reuse buffers** - Allocate once, reuse many times
2. **Batch operations** - Process large arrays, not many small ones
3. **Use vectorized trig** - 5-10x faster than Math.sin/cos/tan
4. **Apply windows before FFT** - Better spectral analysis
5. **Profile your code** - Measure before optimizing

## Requirements

- macOS (Apple Silicon or Intel)
- Node.js >= 18.0.0
- Xcode Command Line Tools

## Links

- [Full Documentation](README.md)
- [Examples](examples/)
- [GitHub](https://github.com/Digital-Defiance/node-accelerate)
- [npm](https://www.npmjs.com/package/@digitaldefiance/node-accelerate)
