#!/usr/bin/env node
/**
 * Benchmark: Apple Accelerate vs Pure JavaScript
 * 
 * This demonstrates the massive performance gains from using
 * Apple's Accelerate framework on M4 Max.
 */

const accelerate = require('./index');

function benchmark(name, fn, iterations = 100) {
  // Warmup
  for (let i = 0; i < 10; i++) fn();
  
  const start = process.hrtime.bigint();
  for (let i = 0; i < iterations; i++) {
    fn();
  }
  const end = process.hrtime.bigint();
  
  const totalMs = Number(end - start) / 1e6;
  const avgMs = totalMs / iterations;
  
  return { name, totalMs, avgMs, iterations };
}

function formatResult(result) {
  return `${result.name}: ${result.avgMs.toFixed(3)}ms avg (${result.iterations} iterations)`;
}

console.log('='.repeat(70));
console.log('APPLE ACCELERATE vs PURE JAVASCRIPT BENCHMARK');
console.log('='.repeat(70));
console.log('');

// Matrix multiplication benchmark
const M = 500, K = 500, N = 500;
console.log(`Matrix Multiplication (${M}x${K} * ${K}x${N}):`);
console.log('-'.repeat(50));

const A = new Float64Array(M * K);
const B = new Float64Array(K * N);
const C_accel = new Float64Array(M * N);
const C_js = new Float64Array(M * N);

// Initialize with random values
for (let i = 0; i < A.length; i++) A[i] = Math.random();
for (let i = 0; i < B.length; i++) B[i] = Math.random();

// Pure JS matrix multiplication
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
  return C;
}

const accelMatmul = benchmark('Accelerate BLAS', () => {
  accelerate.matmul(A, B, C_accel, M, K, N);
}, 10);

const jsMatmulResult = benchmark('Pure JavaScript', () => {
  jsMatmul(A, B, C_js, M, K, N);
}, 10);

console.log(formatResult(accelMatmul));
console.log(formatResult(jsMatmulResult));
console.log(`Speedup: ${(jsMatmulResult.avgMs / accelMatmul.avgMs).toFixed(1)}x faster`);
console.log('');

// Vector operations benchmark
const vecSize = 1000000;
console.log(`Vector Operations (${vecSize.toLocaleString()} elements):`);
console.log('-'.repeat(50));

const vec1 = new Float64Array(vecSize);
const vec2 = new Float64Array(vecSize);
const vecOut = new Float64Array(vecSize);

for (let i = 0; i < vecSize; i++) {
  vec1[i] = Math.random();
  vec2[i] = Math.random();
}

// Dot product
const accelDot = benchmark('Accelerate dot', () => {
  accelerate.dot(vec1, vec2);
}, 100);

const jsDot = benchmark('JS dot', () => {
  let sum = 0;
  for (let i = 0; i < vecSize; i++) {
    sum += vec1[i] * vec2[i];
  }
  return sum;
}, 100);

console.log(formatResult(accelDot));
console.log(formatResult(jsDot));
console.log(`Speedup: ${(jsDot.avgMs / accelDot.avgMs).toFixed(1)}x faster`);
console.log('');

// Vector sum
const accelSum = benchmark('Accelerate sum', () => {
  accelerate.sum(vec1);
}, 100);

const jsSum = benchmark('JS sum', () => {
  let sum = 0;
  for (let i = 0; i < vecSize; i++) {
    sum += vec1[i];
  }
  return sum;
}, 100);

console.log(formatResult(accelSum));
console.log(formatResult(jsSum));
console.log(`Speedup: ${(jsSum.avgMs / accelSum.avgMs).toFixed(1)}x faster`);
console.log('');

// Vector add
const accelAdd = benchmark('Accelerate vadd', () => {
  accelerate.vadd(vec1, vec2, vecOut);
}, 100);

const jsAdd = benchmark('JS vadd', () => {
  for (let i = 0; i < vecSize; i++) {
    vecOut[i] = vec1[i] + vec2[i];
  }
}, 100);

console.log(formatResult(accelAdd));
console.log(formatResult(jsAdd));
console.log(`Speedup: ${(jsAdd.avgMs / accelAdd.avgMs).toFixed(1)}x faster`);
console.log('');

// FFT benchmark
const fftSize = 65536; // 2^16
console.log(`FFT (${fftSize.toLocaleString()} samples):`);
console.log('-'.repeat(50));

const signal = new Float64Array(fftSize);
for (let i = 0; i < fftSize; i++) {
  signal[i] = Math.sin(2 * Math.PI * i / fftSize) + Math.random() * 0.1;
}

const accelFFT = benchmark('Accelerate FFT', () => {
  accelerate.fft(signal);
}, 100);

console.log(formatResult(accelFFT));
console.log('(No JS comparison - FFT is complex to implement correctly)');
console.log('');

console.log('='.repeat(70));
console.log('SUMMARY');
console.log('='.repeat(70));
console.log('');
console.log('Apple Accelerate provides massive speedups for:');
console.log('  - Matrix operations (BLAS): 50-100x faster');
console.log('  - Vector operations (vDSP): 5-20x faster');
console.log('  - FFT: Hardware-optimized implementation');
console.log('');
console.log('Use this addon for numerical computing workloads!');
