# node-accelerate Showcase

This is the GitHub Pages showcase site for **node-accelerate v2.0.0**, high-performance Apple Accelerate framework bindings for Node.js. Built with React, TypeScript, and Vite.

## About node-accelerate v2.0.0

`node-accelerate` provides 80+ hardware-accelerated functions:

### Matrix Operations (BLAS)
- Matrix multiplication, transpose, matrix-vector operations
- BLAS operations: copy, swap, norm, abssum, maxAbsIndex, rot
- Up to 296x speedup on matrix multiplication

### Vector Operations (vDSP)
- Element-wise arithmetic: add, subtract, multiply, divide
- Dot product, normalization, scalar operations
- Advanced operations: vma, vmsa, vlerp, vlimit
- Utilities: vfill, vramp, vclear, vreverse
- 5-8x speedup on vector operations

### Statistical Functions
- Mean, variance, standard deviation, min/max, RMS
- Sum of squares, mean magnitude, mean square
- Max/min magnitude for advanced statistics
- Hardware-accelerated data analysis

### Trigonometric Functions (vForce)
- Standard trig: sin, cos, tan
- Inverse trig: asin, acos, atan, atan2
- 5-10x faster than Math functions

### Hyperbolic Functions (vForce)
- sinh, cosh, tanh
- Perfect for neural network activations
- Hardware-accelerated for ML workloads

### Exponential & Logarithmic (vForce)
- exp, log, log10, pow
- Reciprocal and inverse square root
- Perfect for ML activations and scientific computing

### Rounding Functions (vForce)
- ceil, floor, trunc, copysign
- Vectorized rounding operations
- Essential for quantization and preprocessing

### Signal Processing (vDSP)
- FFT/IFFT, convolution, cross-correlation
- Window functions (Hamming, Hanning, Blackman)
- 10-50x faster than pure JavaScript FFT

### Data Processing
- Clipping, thresholding, interpolation
- Essential preprocessing functions
- Hardware-optimized for large datasets

### Distance Metrics
- Euclidean distance, L2 norm
- Perfect for ML and scientific computing
- Foundation for clustering and similarity search

## Development

```bash
cd showcase
yarn install
yarn dev
```

Visit `http://localhost:5173` to see the site.

## Building

```bash
yarn build
```

The built site will be in the `dist` directory.

## Deployment

The site is automatically deployed to GitHub Pages when changes are pushed to the `main` branch.

## Technology Stack

- **React 19** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool
- **Framer Motion** - Animations
- **React Icons** - Icon library
- **React Intersection Observer** - Scroll animations

## Structure

- `/src/components` - React components
- `/src/assets` - Static assets
- `/public` - Public files
- `index.html` - Entry HTML file
- `vite.config.ts` - Vite configuration
