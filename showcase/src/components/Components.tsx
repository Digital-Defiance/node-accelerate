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
      "Hardware-accelerated matrix operations using Apple's AMX coprocessor. Get 283x speedup on matrix multiplication compared to pure JavaScript.",
    tech: ["BLAS", "AMX", "Linear Algebra"],
    category: "Matrix",
    highlights: [
      "matmul() - Matrix multiplication (C = A × B) with 283x speedup",
      "matvec() - Matrix-vector multiplication (y = A × x)",
      "axpy() - AXPY operation (y = alpha*x + y)",
      "Supports Float64Array for double precision",
      "Row-major order for seamless JavaScript integration",
    ],
  },
  {
    title: "Vector Operations (vDSP)",
    icon: "�",
    description:
      "SIMD-accelerated vector operations using NEON instructions. Element-wise operations run 3-8x faster than pure JavaScript.",
    tech: ["vDSP", "NEON SIMD", "Vector Math"],
    category: "Vector",
    highlights: [
      "vadd(), vsub(), vmul(), vdiv() - Element-wise arithmetic (3-8x faster)",
      "dot() - Dot product with 5x speedup",
      "sum(), mean(), rms() - Fast reductions",
      "vabs(), vsquare(), vsqrt() - Element-wise functions",
      "normalize() - Vector normalization to unit length",
    ],
  },
  {
    title: "Fast Fourier Transform",
    icon: "📡",
    description:
      "Hardware-optimized FFT implementation for signal processing. Analyze frequency spectra with Apple's vDSP FFT routines.",
    tech: ["FFT", "vDSP", "Signal Processing"],
    category: "Signal",
    highlights: [
      "fft() - Fast Fourier Transform for frequency analysis",
      "Returns real and imaginary components",
      "Supports power-of-2 signal lengths",
      "Perfect for audio processing and spectrum analysis",
      "Hardware-accelerated for maximum performance",
    ],
  },
  {
    title: "Distance Metrics",
    icon: "�",
    description:
      "Accelerated distance calculations for machine learning and scientific computing. Essential for clustering, nearest neighbor, and similarity algorithms.",
    tech: ["Distance Metrics", "ML", "Scientific Computing"],
    category: "Core",
    highlights: [
      "euclidean() - Euclidean distance between vectors",
      "Hardware-accelerated for large datasets",
      "Perfect for k-NN, clustering, and similarity search",
      "Works with Float64Array for precision",
      "Foundation for ML inference pipelines",
    ],
  },
  {
    title: "Extreme Performance",
    icon: "🚀",
    description:
      "Real benchmarks on Apple M4 Max show dramatic speedups. Matrix operations are 283x faster, vector operations 5-8x faster than pure JavaScript.",
    tech: ["Benchmarks", "Performance", "Optimization"],
    category: "Performance",
    highlights: [
      "Matrix Multiply (500×500): 93ms → 0.33ms (283x faster)",
      "Vector Dot Product (1M): 0.66ms → 0.13ms (5x faster)",
      "Vector Sum (1M): 0.59ms → 0.08ms (7.6x faster)",
      "Vector Add (1M): 0.74ms → 0.20ms (3.7x faster)",
      "Direct access to Apple Silicon hardware acceleration",
    ],
  },
  {
    title: "TypeScript Support",
    icon: "�",
    description:
      "Full TypeScript definitions included for type-safe numerical computing. Catch errors at compile time and get excellent IDE autocomplete.",
    tech: ["TypeScript", "Type Safety", "DX"],
    category: "Core",
    highlights: [
      "Complete .d.ts type definitions included",
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
      "Perfect for ML inference in Node.js. Accelerate neural network layers, embeddings, and transformations with hardware-optimized matrix operations.",
    tech: ["Machine Learning", "Neural Networks", "Inference"],
    category: "Core",
    highlights: [
      "Fast matrix multiplication for dense layers",
      "Vector operations for activations and normalization",
      "Distance metrics for similarity search",
      "Ideal for running ML models in Node.js",
      "Dramatically faster than pure JavaScript implementations",
    ],
  },
  {
    title: "Scientific Computing",
    icon: "�",
    description:
      "Essential tools for numerical computing, simulations, and data analysis. From numerical integration to signal processing, all hardware-accelerated.",
    tech: ["Scientific Computing", "Numerical Methods", "Data Science"],
    category: "Core",
    highlights: [
      "Vector operations for numerical algorithms",
      "FFT for signal and frequency analysis",
      "Matrix operations for linear algebra",
      "Reductions (sum, mean, RMS) for statistics",
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
            — giving you direct access to hardware-optimized BLAS (matrix operations)
            and vDSP (vector/signal processing) routines. Get 283x speedup on matrix
            multiplication and 5-8x faster vector operations compared to pure JavaScript.
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
              <strong>⚡ 283x Faster</strong>
              <p>
                Matrix multiplication runs 283x faster than pure JavaScript on
                Apple M4 Max
              </p>
            </div>
            <div className="value-prop">
              <strong>🔢 BLAS & vDSP</strong>
              <p>
                Complete matrix operations (BLAS) and vector/signal processing
                (vDSP) support
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
