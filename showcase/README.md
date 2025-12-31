# node-accelerate Showcase

This is the GitHub Pages showcase site for **node-accelerate**, high-performance Apple Accelerate framework bindings for Node.js. Built with React, TypeScript, and Vite.

## About node-accelerate

`node-accelerate` provides:
- Hardware-accelerated matrix operations (BLAS) with 283x speedup
- SIMD-optimized vector processing (vDSP) with 5-8x speedup
- Fast Fourier Transform (FFT) for signal processing
- Direct access to Apple's AMX matrix coprocessor and NEON SIMD
- Full TypeScript support with complete type definitions
- Perfect for ML inference, scientific computing, and signal processing

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
