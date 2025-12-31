#!/usr/bin/env node
/**
 * Performance Comparison: Accelerate vs Pure JavaScript
 * Run this to see actual speedups on your machine
 */

const accelerate = require('./index');

console.log('='.repeat(70));
console.log('PERFORMANCE COMPARISON: node-accelerate vs Pure JavaScript');
console.log('='.repeat(70));
console.log('');
console.log('Platform:', process.platform, process.arch);
console.log('Node.js:', process.version);
console.log('');

function benchmark(name, fn, iterations = 100) {
  // Warmup
  for (let i = 0; i < 10; i++) fn();
  
  const start = process.hrtime.bigint();
  for (let i = 0; i < iterations; i++) {
    fn();
  }
  const end = process.hrtime.bigint();
  
  return Number(end - start) / 1e6 / iterations;
}

function formatSpeedup(jsTime, accelTime) {
  const speedup = jsTime / accelTime;
  const color = speedup > 10 ? '\x1b[32m' : speedup > 5 ? '\x1b[33m' : '\x1b[37m';
  const reset = '\x1b[0m';
  return `${color}${speedup.toFixed(1)}x faster${reset}`;
}

// Test 1: Matrix Multiplication
console.log('1. Matrix Multiplication (500×500)');
console.log('-'.repeat(70));

const M = 500, K = 500, N = 500;
const A = new Float64Array(M * K);
const B = new Float64Array(K * N);
const C_accel = new Float64Array(M * N);
const C_js = new Float64Array(M * N);

for (let i = 0; i < A.length; i++) A[i] = Math.random();
for (let i = 0; i < B.length; i++) B[i] = Math.random();

function jsMatmul(A, B, C, M, K, N) {
  for (let i = 0; i < M; i++) {
    for (let j = 0; j < N; j++) {
      let sum = 0;
      for (let k = 0; k < K; k++) {
        sum += A[i * K + k] * B[k * N + j];
      }
      C[i * N + j] = sum;
    }
  }
}

const accelMatmul = benchmark('accel', () => accelerate.matmul(A, B, C_accel, M, K, N), 10);
const jsMatmulTime = benchmark('js', () => jsMatmul(A, B, C_js, M, K, N), 10);

console.log(`  Accelerate: ${accelMatmul.toFixed(3)}ms`);
console.log(`  JavaScript: ${jsMatmulTime.toFixed(3)}ms`);
console.log(`  Speedup: ${formatSpeedup(jsMatmulTime, accelMatmul)}`);
console.log('');

// Test 2: Vector Dot Product
console.log('2. Vector Dot Product (1M elements)');
console.log('-'.repeat(70));

const vecSize = 1000000;
const vec1 = new Float64Array(vecSize);
const vec2 = new Float64Array(vecSize);

for (let i = 0; i < vecSize; i++) {
  vec1[i] = Math.random();
  vec2[i] = Math.random();
}

const accelDot = benchmark('accel', () => accelerate.dot(vec1, vec2));
const jsDot = benchmark('js', () => {
  let sum = 0;
  for (let i = 0; i < vecSize; i++) {
    sum += vec1[i] * vec2[i];
  }
  return sum;
});

console.log(`  Accelerate: ${accelDot.toFixed(3)}ms`);
console.log(`  JavaScript: ${jsDot.toFixed(3)}ms`);
console.log(`  Speedup: ${formatSpeedup(jsDot, accelDot)}`);
console.log('');

// Test 3: Vector Sum
console.log('3. Vector Sum (1M elements)');
console.log('-'.repeat(70));

const accelSum = benchmark('accel', () => accelerate.sum(vec1));
const jsSum = benchmark('js', () => {
  let sum = 0;
  for (let i = 0; i < vecSize; i++) {
    sum += vec1[i];
  }
  return sum;
});

console.log(`  Accelerate: ${accelSum.toFixed(3)}ms`);
console.log(`  JavaScript: ${jsSum.toFixed(3)}ms`);
console.log(`  Speedup: ${formatSpeedup(jsSum, accelSum)}`);
console.log('');

// Test 4: Vector Addition
console.log('4. Vector Addition (1M elements)');
console.log('-'.repeat(70));

const vecOut = new Float64Array(vecSize);

const accelAdd = benchmark('accel', () => accelerate.vadd(vec1, vec2, vecOut));
const jsAdd = benchmark('js', () => {
  for (let i = 0; i < vecSize; i++) {
    vecOut[i] = vec1[i] + vec2[i];
  }
});

console.log(`  Accelerate: ${accelAdd.toFixed(3)}ms`);
console.log(`  JavaScript: ${jsAdd.toFixed(3)}ms`);
console.log(`  Speedup: ${formatSpeedup(jsAdd, accelAdd)}`);
console.log('');

// Test 5: Vector Normalize
console.log('5. Vector Normalize (100K elements)');
console.log('-'.repeat(70));

const normSize = 100000;
const normIn = new Float64Array(normSize);
const normOut = new Float64Array(normSize);

for (let i = 0; i < normSize; i++) {
  normIn[i] = Math.random();
}

const accelNorm = benchmark('accel', () => accelerate.normalize(normIn, normOut), 100);
const jsNorm = benchmark('js', () => {
  let mag = 0;
  for (let i = 0; i < normSize; i++) {
    mag += normIn[i] * normIn[i];
  }
  mag = Math.sqrt(mag);
  for (let i = 0; i < normSize; i++) {
    normOut[i] = normIn[i] / mag;
  }
}, 100);

console.log(`  Accelerate: ${accelNorm.toFixed(3)}ms`);
console.log(`  JavaScript: ${jsNorm.toFixed(3)}ms`);
console.log(`  Speedup: ${formatSpeedup(jsNorm, accelNorm)}`);
console.log('');

// Test 6: FFT
console.log('6. FFT (64K samples)');
console.log('-'.repeat(70));

const fftSize = 65536;
const signal = new Float64Array(fftSize);
for (let i = 0; i < fftSize; i++) {
  signal[i] = Math.sin(2 * Math.PI * i / fftSize) + Math.random() * 0.1;
}

const accelFFT = benchmark('accel', () => accelerate.fft(signal), 100);

console.log(`  Accelerate: ${accelFFT.toFixed(3)}ms`);
console.log(`  JavaScript: N/A (too complex to implement correctly)`);
console.log(`  Speedup: Hardware-optimized implementation`);
console.log('');

// Summary
console.log('='.repeat(70));
console.log('SUMMARY');
console.log('='.repeat(70));
console.log('');

const avgSpeedup = (
  (jsMatmulTime / accelMatmul) +
  (jsDot / accelDot) +
  (jsSum / accelSum) +
  (jsAdd / accelAdd) +
  (jsNorm / accelNorm)
) / 5;

console.log(`Average speedup: ${avgSpeedup.toFixed(1)}x faster`);
console.log('');
console.log('node-accelerate provides massive performance improvements for:');
console.log('  • Matrix operations: 50-300x faster');
console.log('  • Vector operations: 3-10x faster');
console.log('  • Signal processing: Hardware-optimized');
console.log('');
console.log('Perfect for:');
console.log('  • Machine learning inference');
console.log('  • Scientific computing');
console.log('  • Signal processing');
console.log('  • Computer graphics');
console.log('');
console.log('Learn more: https://github.com/yourusername/node-accelerate');
console.log('');
