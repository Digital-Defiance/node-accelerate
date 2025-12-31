#!/usr/bin/env node
/**
 * Example: Matrix Multiplication
 * Demonstrates hardware-accelerated matrix operations
 */

const accelerate = require('../index');

console.log('Matrix Multiplication Example');
console.log('='.repeat(50));
console.log('');

// Create two 1000×1000 matrices
const M = 1000, K = 1000, N = 1000;
console.log(`Computing C = A × B where:`);
console.log(`  A is ${M}×${K}`);
console.log(`  B is ${K}×${N}`);
console.log(`  C is ${M}×${N}`);
console.log('');

const A = new Float64Array(M * K);
const B = new Float64Array(K * N);
const C = new Float64Array(M * N);

// Fill with random data
console.log('Filling matrices with random data...');
for (let i = 0; i < A.length; i++) A[i] = Math.random();
for (let i = 0; i < B.length; i++) B[i] = Math.random();

// Perform matrix multiplication
console.log('Performing matrix multiplication...');
const start = process.hrtime.bigint();
accelerate.matmul(A, B, C, M, K, N);
const end = process.hrtime.bigint();

const timeMs = Number(end - start) / 1e6;
console.log(`✓ Completed in ${timeMs.toFixed(2)}ms`);
console.log('');

// Calculate GFLOPS
const operations = 2 * M * K * N; // 2 ops per multiply-add
const gflops = operations / (timeMs * 1e6);
console.log(`Performance: ${gflops.toFixed(2)} GFLOPS`);
console.log('');

// Verify result (check a few elements)
console.log('Sample results:');
console.log(`  C[0,0] = ${C[0].toFixed(6)}`);
console.log(`  C[0,1] = ${C[1].toFixed(6)}`);
console.log(`  C[1,0] = ${C[N].toFixed(6)}`);
