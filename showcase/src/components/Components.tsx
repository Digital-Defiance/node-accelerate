import { motion } from "framer-motion";
import { useInView } from "react-intersection-observer";
import "./Components.css";

interface Feature {
  title: string;
  description: string;
  icon: string;
  tech: string[];
  highlights: string[];
  category: "Matrix" | "Vector" | "Signal" | "Performance" | "Core";
}

const features: Feature[] = [
  {
    title: "Matrix Operations (BLAS)",
    icon: "🔢",
    description:
      "Hardware-accelerated matrix operations using Apple's AMX coprocessor. Get up to 296x speedup on matrix multiplication compared to pure JavaScript.",
    tech: ["BLAS", "AMX", "Linear Algebra"],
    category: "Matrix",
    highlights: [
      "matmul() - Matrix multiplication (C = A × B) with up to 296x speedup",
      "matvec() - Matrix-vector multiplication (y = A × x)",
      "transpose() - Matrix transpose (B = A^T)",
      "axpy() - AXPY operation (y = alpha*x + y)",
      "copy(), swap(), norm(), abssum(), maxAbsIndex(), rot()",
    ],
  },
  {
    title: "Vector Operations (vDSP)",
    icon: "📊",
    description:
      "SIMD-accelerated vector operations using NEON instructions. Element-wise operations run 3-8x faster than pure JavaScript.",
    tech: ["vDSP", "NEON SIMD", "Vector Math"],
    category: "Vector",
    highlights: [
      "vadd(), vsub(), vmul(), vdiv() - Element-wise arithmetic (3-8x faster)",
      "dot() - Dot product with 5x speedup",
      "vscale(), vneg(), vaddScalar() - Scalar operations",
      "vabs(), vsquare(), vsqrt() - Element-wise functions",
      "normalize(), vreverse(), vfill(), vramp(), vlerp(), vclear(), vlimit()",
    ],
  },
  {
    title: "Advanced Vector Operations",
    icon: "⚡",
    description:
      "Multiply-add operations and vector utilities for high-performance computing. Fused operations reduce memory bandwidth and improve performance.",
    tech: ["vDSP", "Fused Operations", "Performance"],
    category: "Vector",
    highlights: [
      "vma() - Multiply-add: d = (a*b) + c (fused operation)",
      "vmsa() - Multiply-scalar-add: d = (a*b) + c",
      "vlerp() - Linear interpolation between vectors",
      "vlimit() - Saturate/limit values to range",
      "Perfect for physics simulations and numerical algorithms",
    ],
  },
  {
    title: "Statistical Functions",
    icon: "📈",
    description:
      "Hardware-accelerated statistical operations for data analysis. Compute mean, variance, standard deviation, and more with vDSP optimization.",
    tech: ["Statistics", "vDSP", "Data Analysis"],
    category: "Core",
    highlights: [
      "mean(), variance(), stddev() - Central tendency and spread",
      "sum(), min(), max(), minmax() - Basic reductions",
      "rms(), sumOfSquares() - Power and energy metrics",
      "meanMagnitude(), meanSquare() - Advanced statistics",
      "maxMagnitude(), minMagnitude() - Magnitude extrema",
    ],
  },
  {
    title: "Trigonometric Functions",
    icon: "📐",
    description:
      "Vectorized trigonometric operations using vForce. Process thousands of angles 5-10x faster than Math.sin/cos/tan in loops.",
    tech: ["vForce", "Trigonometry", "SIMD"],
    category: "Core",
    highlights: [
      "vsin(), vcos(), vtan() - Standard trig functions",
      "vasin(), vacos(), vatan(), vatan2() - Inverse trig",
      "5-10x faster than JavaScript Math functions",
      "Process entire arrays in single operations",
      "Perfect for graphics, physics, and signal processing",
    ],
  },
  {
    title: "Hyperbolic Functions",
    icon: "〰️",
    description:
      "Hardware-accelerated hyperbolic functions using vForce. Essential for neural network activations (tanh), physics simulations, and mathematical modeling.",
    tech: ["vForce", "Hyperbolic", "Neural Networks"],
    category: "Core",
    highlights: [
      "vsinh(), vcosh(), vtanh() - Hyperbolic functions",
      "tanh() commonly used as activation function in ML",
      "Hardware-accelerated for maximum performance",
      "Perfect for neural networks and scientific computing",
      "Process thousands of values in microseconds",
    ],
  },
  {
    title: "Exponential & Logarithmic",
    icon: "📉",
    description:
      "Fast exponential and logarithmic operations using vForce. Essential for ML activations, probability calculations, and scientific computing.",
    tech: ["vForce", "Math Functions", "ML"],
    category: "Core",
    highlights: [
      "vexp() - Natural exponential (e^x)",
      "vlog(), vlog10() - Natural and base-10 logarithms",
      "vpow() - Element-wise power (a^b)",
      "vreciprocal(), vrsqrt() - Reciprocal and inverse square root",
      "Perfect for softmax, log-likelihood, and ML operations",
    ],
  },
  {
    title: "Rounding Functions",
    icon: "🔄",
    description:
      "Vectorized rounding operations using vForce. Efficiently round, floor, ceil, and truncate entire arrays with hardware acceleration.",
    tech: ["vForce", "Rounding", "Data Processing"],
    category: "Core",
    highlights: [
      "vceil(), vfloor(), vtrunc() - Rounding operations",
      "vcopysign() - Copy sign between vectors",
      "Hardware-accelerated for large datasets",
      "Perfect for quantization and data preprocessing",
      "Process thousands of values simultaneously",
    ],
  },
  {
    title: "Signal Processing",
    icon: "📡",
    description:
      "Complete signal processing toolkit with FFT, convolution, and correlation. Hardware-optimized for audio, communications, and time-series analysis.",
    tech: ["FFT", "vDSP", "Signal Processing"],
    category: "Signal",
    highlights: [
      "fft(), ifft() - Fast Fourier Transform (forward & inverse)",
      "conv() - 1D convolution for filtering",
      "xcorr() - Cross-correlation for signal alignment",
      "hamming(), hanning(), blackman() - Window functions",
      "10-50x faster than pure JavaScript FFT",
    ],
  },
  {
    title: "Data Processing",
    icon: "🔧",
    description:
      "Essential data manipulation functions for preprocessing and feature engineering. Clip, threshold, interpolate, and transform your data efficiently.",
    tech: ["Data Processing", "vDSP", "Preprocessing"],
    category: "Core",
    highlights: [
      "vclip() - Clip values to range [min, max]",
      "vthreshold() - Apply threshold to data",
      "interp1d() - Linear interpolation",
      "vlimit() - Saturate values to range",
      "Perfect for data preprocessing and normalization",
    ],
  },
  {
    title: "Distance Metrics",
    icon: "📏",
    description:
      "Accelerated distance calculations for machine learning and scientific computing. Essential for clustering, nearest neighbor, and similarity algorithms.",
    tech: ["Distance Metrics", "ML", "Scientific Computing"],
    category: "Core",
    highlights: [
      "euclidean() - Euclidean distance between vectors",
      "norm() - L2 norm (vector length)",
      "Hardware-accelerated for large datasets",
      "Perfect for k-NN, clustering, and similarity search",
      "Foundation for ML inference pipelines",
    ],
  },
  {
    title: "Extreme Performance",
    icon: "🚀",
    description:
      "Real benchmarks on Apple M4 Max show dramatic speedups. Matrix operations are up to 296x faster, vector operations 5-8x faster than pure JavaScript.",
    tech: ["Benchmarks", "Performance", "Optimization"],
    category: "Performance",
    highlights: [
      "Matrix Multiply (500×500): 86ms → 0.30ms (290x faster typical)",
      "Vector Dot Product (1M): 0.63ms → 0.13ms (5x faster)",
      "Vector Sum (1M): 0.59ms → 0.07ms (8x faster)",
      "Trigonometric ops: 5-10x faster than Math functions",
      "Direct access to Apple Silicon hardware acceleration",
    ],
  },
  {
    title: "TypeScript Support",
    icon: "📘",
    description:
      "Full TypeScript definitions included for type-safe numerical computing. Catch errors at compile time and get excellent IDE autocomplete.",
    tech: ["TypeScript", "Type Safety", "DX"],
    category: "Core",
    highlights: [
      "Complete .d.ts type definitions for all 80+ functions",
      "Type-safe function signatures for all operations",
      "Excellent IDE autocomplete and documentation",
      "Catch dimension mismatches at compile time",
      "Seamless integration with TypeScript projects",
    ],
  },
  {
    title: "Machine Learning Ready",
    icon: "🤖",
    description:
      "Perfect for ML inference in Node.js. Accelerate neural network layers, embeddings, and transformations with hardware-optimized operations.",
    tech: ["Machine Learning", "Neural Networks", "Inference"],
    category: "Core",
    highlights: [
      "Fast matrix multiplication for dense layers",
      "Vectorized activations (ReLU, softmax, sigmoid, tanh)",
      "Statistical functions for normalization",
      "Distance metrics for similarity search",
      "80+ functions for complete ML pipelines",
    ],
  },
  {
    title: "Scientific Computing",
    icon: "🔬",
    description:
      "Essential tools for numerical computing, simulations, and data analysis. From numerical integration to signal processing, all hardware-accelerated.",
    tech: ["Scientific Computing", "Numerical Methods", "Data Science"],
    category: "Core",
    highlights: [
      "Vector operations for numerical algorithms",
      "FFT for signal and frequency analysis",
      "Matrix operations for linear algebra",
      "Statistical functions for data analysis",
      "Perfect for data science and scientific applications",
    ],
  },
];

const Components = () => {
  const [ref, inView] = useInView({
    triggerOnce: true,
    threshold: 0.1,
  });

  return (
    <section className="components section" id="components" ref={ref}>
      <motion.div
        className="components-container"
        initial={{ opacity: 0 }}
        animate={inView ? { opacity: 1 } : {}}
        transition={{ duration: 0.6 }}
      >
        <h2 className="section-title">
          Hardware-Accelerated <span className="gradient-text">Operations</span>
        </h2>
        <p className="components-subtitle">
          Direct access to Apple's Accelerate framework from Node.js
        </p>

        <motion.div
          className="suite-intro"
          initial={{ opacity: 0, y: 20 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.6, delay: 0.2 }}
        >
          <h3>
            Unlock <em>hardware acceleration</em> for <em>numerical computing</em> in{" "}
            <em>Node.js</em> on Apple Silicon.
          </h3>
          <p>
            <strong>
              node-accelerate brings Apple's Accelerate framework to JavaScript
            </strong>{" "}
            — giving you direct access to 80+ hardware-optimized functions including
            BLAS (matrix operations), vDSP (vector/signal processing), and vForce
            (math functions). Get up to 296x speedup on matrix multiplication and 5-10x
            faster vector operations compared to pure JavaScript.
          </p>
          <div className="problem-solution">
            <div className="problem">
              <h4>❌ The Challenge: JavaScript Is Slow for Math</h4>
              <ul>
                <li>Pure JavaScript matrix operations are extremely slow</li>
                <li>No native access to Apple's hardware acceleration</li>
                <li>ML inference and scientific computing bottlenecked by CPU</li>
                <li>FFT and signal processing limited by single-threaded JS</li>
                <li>Apple Silicon's AMX and NEON go unused in Node.js</li>
              </ul>
              <p>
                <strong>Result:</strong> Your numerical code runs 100-300x slower
                than it could with hardware acceleration.
              </p>
            </div>
            <div className="solution">
              <h4>✅ The Solution: Native Accelerate Bindings</h4>
              <p>
                <strong>node-accelerate</strong> exposes{" "}
                <strong>Apple's Accelerate framework</strong> to JavaScript through
                native bindings. This gives you direct access to the{" "}
                <strong>AMX matrix coprocessor</strong>, <strong>NEON SIMD</strong>{" "}
                instructions, and <strong>optimized FFT</strong> routines.
              </p>
              <p>
                Perfect for <strong>machine learning inference</strong>,{" "}
                <strong>signal processing</strong>, <strong>scientific computing</strong>,
                and any application that needs fast numerical operations. Works on both
                Apple Silicon (M1/M2/M3/M4) and Intel Macs with zero configuration.
              </p>
            </div>
          </div>
          <div className="value-props">
            <div className="value-prop">
              <strong>⚡ Up to 296x Faster</strong>
              <p>
                Matrix multiplication runs up to 296x faster than pure JavaScript on
                Apple M4 Max
              </p>
            </div>
            <div className="value-prop">
              <strong>🔢 80+ Functions</strong>
              <p>
                Complete matrix, vector, signal processing, statistics, and math
                functions
              </p>
            </div>
            <div className="value-prop">
              <strong>🍎 Apple Silicon</strong>
              <p>
                Direct access to AMX matrix coprocessor and NEON SIMD on M1/M2/M3/M4
              </p>
            </div>
            <div className="value-prop">
              <strong>📘 TypeScript</strong>
              <p>
                Full type definitions included for type-safe numerical computing
              </p>
            </div>
          </div>
        </motion.div>

        <div className="components-grid">
          {features.map((feature, index) => (
            <motion.div
              key={feature.title}
              className="component-card card"
              initial={{ opacity: 0, y: 50 }}
              animate={inView ? { opacity: 1, y: 0 } : {}}
              transition={{ delay: index * 0.1, duration: 0.6 }}
            >
              <div className="component-header">
                <div className="component-icon">{feature.icon}</div>
                <h3>{feature.title}</h3>
                <span
                  className={`component-badge ${feature.category.toLowerCase()}`}
                >
                  {feature.category}
                </span>
              </div>

              <p className="component-description">{feature.description}</p>

              <ul className="component-highlights">
                {feature.highlights.map((highlight, i) => (
                  <li key={i}>{highlight}</li>
                ))}
              </ul>

              <div className="component-tech">
                {feature.tech.map((tech) => (
                  <span key={tech} className="tech-badge">
                    {tech}
                  </span>
                ))}
              </div>
            </motion.div>
          ))}
        </div>
      </motion.div>
    </section>
  );
};

export default Components;
