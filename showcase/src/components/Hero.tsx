import { motion } from "framer-motion";
import { FaGithub, FaNpm } from "react-icons/fa";
import "./Hero.css";

interface HeroProps {
  scrollY: number;
}

const Hero = ({ scrollY }: HeroProps) => {
  const parallaxOffset = scrollY * 0.5;

  return (
    <section className="hero" id="home">
      <div
        className="hero-background"
        style={{ transform: `translateY(${parallaxOffset}px)` }}
      >
        <div className="particles" />
      </div>

      <motion.div
        className="hero-content"
        initial={{ opacity: 0, y: 50 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8 }}
      >
        <motion.div
          className="hero-badge"
          initial={{ opacity: 0, scale: 0.8 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ delay: 0.1, duration: 0.6 }}
        >
          <span className="badge-text">⚡ Hardware-Accelerated Computing for Node.js</span>
        </motion.div>

        <motion.h1
          className="hero-title"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.2, duration: 0.8 }}
        >
          node-accelerate
        </motion.h1>

        <motion.h2
          className="hero-subtitle gradient-text"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.4, duration: 0.8 }}
        >
          Apple Accelerate Framework for Node.js
        </motion.h2>

        <motion.p
          className="hero-description"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.6, duration: 0.8 }}
        >
          50+ hardware-accelerated functions for matrix operations, signal processing, and statistics
          <br />
          on Apple Silicon with hardware-optimized BLAS, vDSP, and vForce routines.
          <br />
          <span className="hero-highlight">
            🚀 Up to 296x Faster • 🔢 80+ Functions • 📊 Statistics • 📡 FFT/IFFT • 🌊 Convolution • 🍎 Apple Silicon
          </span>
        </motion.p>

        <motion.div
          className="hero-cta"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.9, duration: 0.8 }}
        >
          <a
            href="https://www.npmjs.com/package/@digitaldefiance/node-accelerate"
            className="btn btn-primary"
            target="_blank"
            rel="noopener noreferrer"
          >
            <FaNpm />
            View on NPM
          </a>
          <a
            href="https://github.com/Digital-Defiance/node-accelerate"
            className="btn btn-secondary"
            target="_blank"
            rel="noopener noreferrer"
          >
            <FaGithub />
            View on GitHub
          </a>
        </motion.div>
      </motion.div>

      <motion.div
        className="scroll-indicator"
        animate={{ y: [0, 10, 0] }}
        transition={{ repeat: Infinity, duration: 1.5 }}
      >
        <div className="mouse">
          <div className="wheel" />
        </div>
      </motion.div>
    </section>
  );
};

export default Hero;
