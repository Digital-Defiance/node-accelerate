/**
 * Mathematical Functions Examples
 * Demonstrates exp, log, power, and other math operations
 */

const accelerate = require('..');

console.log('=== Mathematical Functions ===\n');

const n = 1000;

// 1. Exponential and Logarithm
console.log('--- Exponential and Logarithm ---');
const x = new Float64Array(n);
const expX = new Float64Array(n);
const logExpX = new Float64Array(n);

for (let i = 0; i < n; i++) {
  x[i] = i / 100; // 0 to 10
}

accelerate.vexp(x, expX);
accelerate.vlog(expX, logExpX);

// Verify log(exp(x)) = x
let maxError = 0;
for (let i = 0; i < n; i++) {
  const error = Math.abs(x[i] - logExpX[i]);
  if (error > maxError) maxError = error;
}

console.log('log(exp(x)) = x verification');
console.log('Max error:', maxError.toFixed(10));

// 2. Power functions
console.log('\n--- Power Functions ---');
const base = new Float64Array(100);
const exponent = new Float64Array(100);
const result = new Float64Array(100);

for (let i = 0; i < 100; i++) {
  base[i] = i / 10 + 1; // 1 to 11
  exponent[i] = 2; // Square everything
}

accelerate.vpow(base, exponent, result);

console.log('First 5 values squared:');
for (let i = 0; i < 5; i++) {
  console.log(`  ${base[i].toFixed(1)}^2 = ${result[i].toFixed(2)}`);
}

// 3. Square root verification
console.log('\n--- Square Root ---');
const squares = new Float64Array(100);
const sqrtResult = new Float64Array(100);

for (let i = 0; i < 100; i++) {
  squares[i] = i * i;
}

accelerate.vsqrt(squares, sqrtResult);

console.log('sqrt(x²) = x verification:');
console.log('  sqrt(25) =', sqrtResult[5].toFixed(4));
console.log('  sqrt(100) =', sqrtResult[10].toFixed(4));
console.log('  sqrt(10000) =', sqrtResult[100-1].toFixed(4));

// 4. Log base 10
console.log('\n--- Logarithm Base 10 ---');
const powers10 = new Float64Array([1, 10, 100, 1000, 10000]);
const log10Result = new Float64Array(5);

accelerate.vlog10(powers10, log10Result);

console.log('log10 of powers of 10:');
for (let i = 0; i < 5; i++) {
  console.log(`  log10(${powers10[i]}) = ${log10Result[i].toFixed(4)}`);
}

// 5. Absolute value
console.log('\n--- Absolute Value ---');
const mixed = new Float64Array([-5, -3, -1, 0, 1, 3, 5]);
const absResult = new Float64Array(7);

accelerate.vabs(mixed, absResult);

console.log('Absolute values:');
console.log('  Input:', Array.from(mixed));
console.log('  Output:', Array.from(absResult));

// 6. Negation
console.log('\n--- Negation ---');
const positive = new Float64Array([1, 2, 3, 4, 5]);
const negative = new Float64Array(5);

accelerate.vneg(positive, negative);

console.log('Negation:');
console.log('  Input:', Array.from(positive));
console.log('  Output:', Array.from(negative));
