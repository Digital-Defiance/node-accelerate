#!/usr/bin/env node
/**
 * Example: Vector Operations
 * Demonstrates hardware-accelerated vector math
 */

const accelerate = require('../index');

console.log('Vector Operations Example');
console.log('='.repeat(50));
console.log('');

// Create large vectors
const size = 1000000;
console.log(`Vector size: ${size.toLocaleString()} elements`);
console.log('');

const a = new Float64Array(size);
const b = new Float64Array(size);
const result = new Float64Array(size);

// Fill with random data
for (let i = 0; i < size; i++) {
  a[i] = Math.random() * 100;
  b[i] = Math.random() * 100;
}

// Dot product
console.log('Computing dot product...');
let start = process.hrtime.bigint();
const dotProduct = accelerate.dot(a, b);
let end = process.hrtime.bigint();
console.log(`  Result: ${dotProduct.toFixed(2)}`);
console.log(`  Time: ${(Number(end - start) / 1e6).toFixed(3)}ms`);
console.log('');

// Sum
console.log('Computing sum...');
start = process.hrtime.bigint();
const sum = accelerate.sum(a);
end = process.hrtime.bigint();
console.log(`  Result: ${sum.toFixed(2)}`);
console.log(`  Time: ${(Number(end - start) / 1e6).toFixed(3)}ms`);
console.log('');

// Mean
console.log('Computing mean...');
start = process.hrtime.bigint();
const mean = accelerate.mean(a);
end = process.hrtime.bigint();
console.log(`  Result: ${mean.toFixed(2)}`);
console.log(`  Time: ${(Number(end - start) / 1e6).toFixed(3)}ms`);
console.log('');

// Vector addition
console.log('Computing vector addition...');
start = process.hrtime.bigint();
accelerate.vadd(a, b, result);
end = process.hrtime.bigint();
console.log(`  Sample: ${a[0].toFixed(2)} + ${b[0].toFixed(2)} = ${result[0].toFixed(2)}`);
console.log(`  Time: ${(Number(end - start) / 1e6).toFixed(3)}ms`);
console.log('');

// Vector multiplication
console.log('Computing vector multiplication...');
start = process.hrtime.bigint();
accelerate.vmul(a, b, result);
end = process.hrtime.bigint();
console.log(`  Sample: ${a[0].toFixed(2)} × ${b[0].toFixed(2)} = ${result[0].toFixed(2)}`);
console.log(`  Time: ${(Number(end - start) / 1e6).toFixed(3)}ms`);
console.log('');

console.log('✓ All operations completed successfully!');
