import { motion } from "framer-motion";
import { useInView } from "react-intersection-observer";
import "./Benchmarks.css";

interface BenchmarkResult {
  operation: string;
  category: string;
  stock: string;
  accelerate: string;
  speedup: string;
  isPrimary?: boolean;
}

const benchmarkResults: BenchmarkResult[] = [
  // Primary Benchmarks
  {
    operation: "Matrix Multiply (500×500)",
    category: "Matrix",
    stock: "90.52 ms",
    accelerate: "0.297 ms",
    speedup: "305x",
    isPrimary: true,
  },
  {
    operation: "Vector Dot Product (1M)",
    category: "Vector",
    stock: "0.647 ms",
    accelerate: "0.133 ms",
    speedup: "4.9x",
    isPrimary: true,
  },
  {
    operation: "Vector Sum (1M)",
    category: "Vector",
    stock: "0.583 ms",
    accelerate: "0.075 ms",
    speedup: "7.8x",
    isPrimary: true,
  },
  {
    operation: "Vector Add (1M)",
    category: "Vector",
    stock: "0.436 ms",
    accelerate: "0.195 ms",
    speedup: "2.2x",
    isPrimary: true,
  },
  
  // Matrix Operations
  {
    operation: "Matrix Multiply (100×100)",
    category: "Matrix",
    stock: "0.846 ms",
    accelerate: "0.013 ms",
    speedup: "68x",
  },
  {
    operation: "Matrix Multiply Float32 (100×100)",
    category: "Matrix",
    stock: "0.736 ms",
    accelerate: "0.004 ms",
    speedup: "211x",
  },
  {
    operation: "Matrix-Vector Multiply (1000×1000)",
    category: "Matrix",
    stock: "0.641 ms",
    accelerate: "0.012 ms",
    speedup: "52x",
  },
  {
    operation: "Vector Copy (100k)",
    category: "Vector",
    stock: "0.054 ms",
    accelerate: "0.003 ms",
    speedup: "17x",
  },
  {
    operation: "Vector Norm (100k)",
    category: "Vector",
    stock: "0.065 ms",
    accelerate: "0.022 ms",
    speedup: "3.0x",
  },
  {
    operation: "Dot Product (100k)",
    category: "Vector",
    stock: "0.065 ms",
    accelerate: "0.013 ms",
    speedup: "5.1x",
  },
  
  // Vector Operations
  {
    operation: "Vector Add (100k)",
    category: "Vector",
    stock: "0.072 ms",
    accelerate: "0.019 ms",
    speedup: "3.8x",
  },
  {
    operation: "Vector Multiply (100k)",
    category: "Vector",
    stock: "0.062 ms",
    accelerate: "0.019 ms",
    speedup: "3.3x",
  },
  {
    operation: "Vector Subtract (100k)",
    category: "Vector",
    stock: "0.052 ms",
    accelerate: "0.019 ms",
    speedup: "2.7x",
  },
  {
    operation: "Vector Divide (100k)",
    category: "Vector",
    stock: "0.062 ms",
    accelerate: "0.026 ms",
    speedup: "2.4x",
  },
  {
    operation: "Vector Negate (100k)",
    category: "Vector",
    stock: "0.050 ms",
    accelerate: "0.013 ms",
    speedup: "3.9x",
  },
  {
    operation: "Vector Abs (100k)",
    category: "Vector",
    stock: "0.051 ms",
    accelerate: "0.013 ms",
    speedup: "4.0x",
  },
  {
    operation: "Vector Square (100k)",
    category: "Vector",
    stock: "0.051 ms",
    accelerate: "0.013 ms",
    speedup: "4.0x",
  },
  {
    operation: "Vector Scale (100k)",
    category: "Vector",
    stock: "0.058 ms",
    accelerate: "0.013 ms",
    speedup: "4.6x",
  },
  {
    operation: "AXPY (100k)",
    category: "Vector",
    stock: "0.049 ms",
    accelerate: "0.004 ms",
    speedup: "13x",
  },
  
  // Trigonometric
  {
    operation: "Vector Sin (10k)",
    category: "Trig",
    stock: "0.062 ms",
    accelerate: "0.008 ms",
    speedup: "7.6x",
  },
  {
    operation: "Vector Cos (10k)",
    category: "Trig",
    stock: "0.073 ms",
    accelerate: "0.009 ms",
    speedup: "8.2x",
  },
  {
    operation: "Vector Tan (10k)",
    category: "Trig",
    stock: "0.075 ms",
    accelerate: "0.009 ms",
    speedup: "8.7x",
  },
  {
    operation: "Vector Asin (10k)",
    category: "Trig",
    stock: "0.037 ms",
    accelerate: "0.009 ms",
    speedup: "4.0x",
  },
  {
    operation: "Vector Sinh (10k)",
    category: "Trig",
    stock: "0.058 ms",
    accelerate: "0.010 ms",
    speedup: "6.0x",
  },
  
  // Exponential/Logarithmic
  {
    operation: "Vector Exp (10k)",
    category: "Math",
    stock: "0.035 ms",
    accelerate: "0.010 ms",
    speedup: "3.7x",
  },
  {
    operation: "Vector Log (10k)",
    category: "Math",
    stock: "0.036 ms",
    accelerate: "0.010 ms",
    speedup: "3.4x",
  },
  {
    operation: "Vector Sqrt (100k)",
    category: "Math",
    stock: "0.060 ms",
    accelerate: "0.022 ms",
    speedup: "2.7x",
  },
  {
    operation: "Vector Ceil (100k)",
    category: "Math",
    stock: "0.060 ms",
    accelerate: "0.014 ms",
    speedup: "4.3x",
  },
  {
    operation: "Vector Floor (100k)",
    category: "Math",
    stock: "0.058 ms",
    accelerate: "0.014 ms",
    speedup: "4.1x",
  },
  
  // Statistical
  {
    operation: "Sum (100k)",
    category: "Stats",
    stock: "0.062 ms",
    accelerate: "0.007 ms",
    speedup: "8.5x",
  },
  {
    operation: "Mean (100k)",
    category: "Stats",
    stock: "0.063 ms",
    accelerate: "0.008 ms",
    speedup: "8.4x",
  },
  {
    operation: "Variance (100k)",
    category: "Stats",
    stock: "0.121 ms",
    accelerate: "0.043 ms",
    speedup: "2.8x",
  },
  {
    operation: "Std Dev (100k)",
    category: "Stats",
    stock: "0.125 ms",
    accelerate: "0.044 ms",
    speedup: "2.8x",
  },
  {
    operation: "Min/Max (100k)",
    category: "Stats",
    stock: "0.097 ms",
    accelerate: "0.014 ms",
    speedup: "6.9x",
  },
  {
    operation: "Max (100k)",
    category: "Stats",
    stock: "0.072 ms",
    accelerate: "0.007 ms",
    speedup: "11x",
  },
  {
    operation: "Min (100k)",
    category: "Stats",
    stock: "0.082 ms",
    accelerate: "0.007 ms",
    speedup: "12x",
  },
  {
    operation: "RMS (100k)",
    category: "Stats",
    stock: "0.063 ms",
    accelerate: "0.025 ms",
    speedup: "2.6x",
  },
  {
    operation: "Euclidean Distance (100k)",
    category: "Stats",
    stock: "0.065 ms",
    accelerate: "0.050 ms",
    speedup: "1.3x",
  },
  {
    operation: "Normalize (100k)",
    category: "Stats",
    stock: "0.114 ms",
    accelerate: "0.022 ms",
    speedup: "5.3x",
  },
  
  // Signal Processing
  {
    operation: "FFT (8192)",
    category: "Signal",
    stock: "N/A (too slow)",
    accelerate: "0.097 ms",
    speedup: "50-100x",
  },
  {
    operation: "IFFT (8192)",
    category: "Signal",
    stock: "N/A (too slow)",
    accelerate: "0.098 ms",
    speedup: "50-100x",
  },
  {
    operation: "Cross-Correlation (1000)",
    category: "Signal",
    stock: "1.26 ms",
    accelerate: "0.034 ms",
    speedup: "37x",
  },
  {
    operation: "Convolution (1000)",
    category: "Signal",
    stock: "0.002 ms",
    accelerate: "0.0004 ms",
    speedup: "4.5x",
  },
  {
    operation: "Hanning Window (8192)",
    category: "Signal",
    stock: "0.045 ms",
    accelerate: "0.009 ms",
    speedup: "5.2x",
  },
  
  // Data Processing
  {
    operation: "Clip (100k)",
    category: "Data",
    stock: "0.051 ms",
    accelerate: "0.014 ms",
    speedup: "3.7x",
  },
  {
    operation: "Threshold (100k)",
    category: "Data",
    stock: "0.269 ms",
    accelerate: "0.013 ms",
    speedup: "21x",
  },
];

const Benchmarks = () => {
  const [ref, inView] = useInView({
    triggerOnce: true,
    threshold: 0.1,
  });

  const categories = ["Matrix", "Vector", "Trig", "Math", "Stats", "Signal", "Data"];
  const categoryColors: Record<string, string> = {
    Matrix: "#ff6b6b",
    Vector: "#4ecdc4",
    Trig: "#45b7d1",
    Math: "#96ceb4",
    Stats: "#ffeaa7",
    Signal: "#dfe6e9",
    Data: "#a29bfe",
  };

  return (
    <section className="benchmarks section" id="benchmarks" ref={ref}>
      <motion.div
        className="benchmarks-container"
        initial={{ opacity: 0 }}
        animate={inView ? { opacity: 1 } : {}}
        transition={{ duration: 0.6 }}
      >
        <h2 className="section-title">
          Real-World <span className="gradient-text">Benchmarks</span>
        </h2>
        <p className="benchmarks-subtitle">
          Tested on Apple M4 Max • All times in milliseconds
        </p>

        <motion.div
          className="benchmark-legend"
          initial={{ opacity: 0, y: 20 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.6, delay: 0.2 }}
        >
          <div className="legend-item">
            <span className="legend-badge primary">★</span>
            <span>Primary Benchmarks (README claims)</span>
          </div>
          {categories.map((cat) => (
            <div key={cat} className="legend-item">
              <span
                className="legend-badge"
                style={{ backgroundColor: categoryColors[cat] }}
              >
                {cat}
              </span>
            </div>
          ))}
        </motion.div>

        <motion.div
          className="benchmark-table-container"
          initial={{ opacity: 0, y: 20 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.6, delay: 0.3 }}
        >
          <table className="benchmark-table">
            <thead>
              <tr>
                <th>Operation</th>
                <th>Category</th>
                <th>Pure JavaScript</th>
                <th>node-accelerate</th>
                <th>Speedup</th>
              </tr>
            </thead>
            <tbody>
              {benchmarkResults.map((result, index) => (
                <motion.tr
                  key={result.operation}
                  className={result.isPrimary ? "primary-row" : ""}
                  initial={{ opacity: 0, x: -20 }}
                  animate={inView ? { opacity: 1, x: 0 } : {}}
                  transition={{ delay: 0.4 + index * 0.02, duration: 0.4 }}
                >
                  <td className="operation-cell">
                    {result.isPrimary && <span className="star">★</span>}
                    {result.operation}
                  </td>
                  <td>
                    <span
                      className="category-badge"
                      style={{ backgroundColor: categoryColors[result.category] }}
                    >
                      {result.category}
                    </span>
                  </td>
                  <td className="stock-cell">{result.stock}</td>
                  <td className="accelerate-cell">{result.accelerate}</td>
                  <td className="speedup-cell">
                    <strong>{result.speedup}</strong>
                  </td>
                </motion.tr>
              ))}
            </tbody>
          </table>
        </motion.div>

        <motion.div
          className="benchmark-summary"
          initial={{ opacity: 0, y: 20 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.6, delay: 0.5 }}
        >
          <div className="summary-card">
            <h3>Average Speedup</h3>
            <p className="summary-value">19.6x</p>
          </div>
          <div className="summary-card">
            <h3>Maximum Speedup</h3>
            <p className="summary-value">305x</p>
          </div>
          <div className="summary-card">
            <h3>Functions Tested</h3>
            <p className="summary-value">48</p>
          </div>
        </motion.div>

        <motion.p
          className="benchmark-note"
          initial={{ opacity: 0 }}
          animate={inView ? { opacity: 1 } : {}}
          transition={{ duration: 0.6, delay: 0.6 }}
        >
          Run <code>npm run benchmark</code> to test on your own machine
        </motion.p>
      </motion.div>
    </section>
  );
};

export default Benchmarks;
