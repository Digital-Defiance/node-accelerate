#!/usr/bin/env node
/**
 * Test suite for node-accelerate
 * Verifies all functions work correctly
 */

const accelerate = require('./index');

let passed = 0;
let failed = 0;

function assert(condition, message) {
  if (condition) {
    console.log('✓', message);
    passed++;
  } else {
    console.error('✗', message);
    failed++;
  }
}

function assertClose(actual, expected, tolerance, message) {
  const diff = Math.abs(actual - expected);
  if (diff < tolerance) {
    console.log('✓', message, `(${actual} ≈ ${expected})`);
    passed++;
  } else {
    console.error('✗', message, `(${actual} != ${expected}, diff: ${diff})`);
    failed++;
  }
}

function assertArrayClose(actual, expected, tolerance, message) {
  if (actual.length !== expected.length) {
    console.error('✗', message, '(length mismatch)');
    failed++;
    return;
  }
  
  let allClose = true;
  for (let i = 0; i < actual.length; i++) {
    if (Math.abs(actual[i] - expected[i]) > tolerance) {
      allClose = false;
      break;
    }
  }
  
  if (allClose) {
    console.log('✓', message);
    passed++;
  } else {
    console.error('✗', message);
    failed++;
  }
}

console.log('='.repeat(70));
console.log('NODE-ACCELERATE TEST SUITE');
console.log('='.repeat(70));
console.log('');

// Test 1: Matrix multiplication
console.log('Testing matrix multiplication...');
const M = 3, K = 3, N = 3;
const A = new Float64Array([
  1, 2, 3,
  4, 5, 6,
  7, 8, 9
]);
const B = new Float64Array([
  1, 0, 0,
  0, 1, 0,
  0, 0, 1
]);
const C = new Float64Array(M * N);

accelerate.matmul(A, B, C, M, K, N);
assertArrayClose(C, A, 1e-10, 'Matrix multiply by identity');

// Test 2: Dot product
console.log('');
console.log('Testing dot product...');
const vec1 = new Float64Array([1, 2, 3, 4]);
const vec2 = new Float64Array([5, 6, 7, 8]);
const dotResult = accelerate.dot(vec1, vec2);
assertClose(dotResult, 70, 1e-10, 'Dot product [1,2,3,4] · [5,6,7,8] = 70');

// Test 3: Sum
console.log('');
console.log('Testing sum...');
const vec3 = new Float64Array([1, 2, 3, 4, 5]);
const sumResult = accelerate.sum(vec3);
assertClose(sumResult, 15, 1e-10, 'Sum [1,2,3,4,5] = 15');

// Test 4: Mean
console.log('');
console.log('Testing mean...');
const meanResult = accelerate.mean(vec3);
assertClose(meanResult, 3, 1e-10, 'Mean [1,2,3,4,5] = 3');

// Test 5: Vector addition
console.log('');
console.log('Testing vector addition...');
const va = new Float64Array([1, 2, 3]);
const vb = new Float64Array([4, 5, 6]);
const vout = new Float64Array(3);
accelerate.vadd(va, vb, vout);
assertArrayClose(vout, new Float64Array([5, 7, 9]), 1e-10, 'Vector add [1,2,3] + [4,5,6] = [5,7,9]');

// Test 6: Vector subtraction
console.log('');
console.log('Testing vector subtraction...');
const vsub_out = new Float64Array(3);
accelerate.vsub(vb, va, vsub_out);
assertArrayClose(vsub_out, new Float64Array([3, 3, 3]), 1e-10, 'Vector sub [4,5,6] - [1,2,3] = [3,3,3]');

// Test 7: Vector multiplication
console.log('');
console.log('Testing vector multiplication...');
const vmul_out = new Float64Array(3);
accelerate.vmul(va, vb, vmul_out);
assertArrayClose(vmul_out, new Float64Array([4, 10, 18]), 1e-10, 'Vector mul [1,2,3] * [4,5,6] = [4,10,18]');

// Test 8: Vector division
console.log('');
console.log('Testing vector division...');
const vdiv_a = new Float64Array([10, 20, 30]);
const vdiv_b = new Float64Array([2, 4, 5]);
const vdiv_out = new Float64Array(3);
accelerate.vdiv(vdiv_a, vdiv_b, vdiv_out);
assertArrayClose(vdiv_out, new Float64Array([5, 5, 6]), 1e-10, 'Vector div [10,20,30] / [2,4,5] = [5,5,6]');

// Test 9: Vector scaling
console.log('');
console.log('Testing vector scaling...');
const vscale_out = new Float64Array(3);
accelerate.vscale(va, 2.0, vscale_out);
assertArrayClose(vscale_out, new Float64Array([2, 4, 6]), 1e-10, 'Vector scale [1,2,3] * 2 = [2,4,6]');

// Test 10: Vector max
console.log('');
console.log('Testing vector max...');
const maxResult = accelerate.max(vec3);
assertClose(maxResult, 5, 1e-10, 'Max [1,2,3,4,5] = 5');

// Test 11: Vector min
console.log('');
console.log('Testing vector min...');
const minResult = accelerate.min(vec3);
assertClose(minResult, 1, 1e-10, 'Min [1,2,3,4,5] = 1');

// Test 12: FFT
console.log('');
console.log('Testing FFT...');
const fftSize = 64;
const signal = new Float64Array(fftSize);
// Create a simple sine wave
for (let i = 0; i < fftSize; i++) {
  signal[i] = Math.sin(2 * Math.PI * i / fftSize);
}
const spectrum = accelerate.fft(signal);
assert(spectrum.real instanceof Float64Array, 'FFT returns real component');
assert(spectrum.imag instanceof Float64Array, 'FFT returns imaginary component');
assert(spectrum.real.length === fftSize / 2, 'FFT real component has correct length');
assert(spectrum.imag.length === fftSize / 2, 'FFT imaginary component has correct length');

// Test 13: Large matrix multiplication
console.log('');
console.log('Testing large matrix multiplication...');
const largeM = 100, largeK = 100, largeN = 100;
const largeA = new Float64Array(largeM * largeK);
const largeB = new Float64Array(largeK * largeN);
const largeC = new Float64Array(largeM * largeN);

for (let i = 0; i < largeA.length; i++) largeA[i] = Math.random();
for (let i = 0; i < largeB.length; i++) largeB[i] = Math.random();

try {
  accelerate.matmul(largeA, largeB, largeC, largeM, largeK, largeN);
  assert(true, 'Large matrix multiplication (100×100) completes without error');
} catch (e) {
  assert(false, 'Large matrix multiplication (100×100) throws error: ' + e.message);
}

// Test 14: Large vector operations
console.log('');
console.log('Testing large vector operations...');
const largeVecSize = 1000000;
const largeVec1 = new Float64Array(largeVecSize);
const largeVec2 = new Float64Array(largeVecSize);
const largeVecOut = new Float64Array(largeVecSize);

for (let i = 0; i < largeVecSize; i++) {
  largeVec1[i] = Math.random();
  largeVec2[i] = Math.random();
}

try {
  accelerate.vadd(largeVec1, largeVec2, largeVecOut);
  assert(true, 'Large vector addition (1M elements) completes without error');
  
  const largeDot = accelerate.dot(largeVec1, largeVec2);
  assert(!isNaN(largeDot) && isFinite(largeDot), 'Large dot product (1M elements) returns valid result');
} catch (e) {
  assert(false, 'Large vector operations throw error: ' + e.message);
}

// Test 15: Matrix-vector multiplication
console.log('');
console.log('Testing matrix-vector multiplication...');
const mvM = 4, mvN = 3;
const mvA = new Float64Array([
  1, 2, 3,
  4, 5, 6,
  7, 8, 9,
  10, 11, 12
]);
const mvx = new Float64Array([1, 1, 1]);
const mvy = new Float64Array(mvM);
accelerate.matvec(mvA, mvx, mvy, mvM, mvN);
assertArrayClose(mvy, new Float64Array([6, 15, 24, 33]), 1e-10, 'Matrix-vector multiply');

// Test 16: AXPY
console.log('');
console.log('Testing AXPY...');
const axpy_x = new Float64Array([1, 2, 3]);
const axpy_y = new Float64Array([4, 5, 6]);
accelerate.axpy(2.0, axpy_x, axpy_y);
assertArrayClose(axpy_y, new Float64Array([6, 9, 12]), 1e-10, 'AXPY: y = 2*x + y');

// Test 17: Vector absolute value
console.log('');
console.log('Testing vector absolute value...');
const vabs_in = new Float64Array([-1, -2, 3, -4]);
const vabs_out = new Float64Array(4);
accelerate.vabs(vabs_in, vabs_out);
assertArrayClose(vabs_out, new Float64Array([1, 2, 3, 4]), 1e-10, 'Vector abs');

// Test 18: Vector square
console.log('');
console.log('Testing vector square...');
const vsq_in = new Float64Array([2, 3, 4]);
const vsq_out = new Float64Array(3);
accelerate.vsquare(vsq_in, vsq_out);
assertArrayClose(vsq_out, new Float64Array([4, 9, 16]), 1e-10, 'Vector square');

// Test 19: Vector square root
console.log('');
console.log('Testing vector square root...');
const vsqrt_in = new Float64Array([4, 9, 16]);
const vsqrt_out = new Float64Array(3);
accelerate.vsqrt(vsqrt_in, vsqrt_out);
assertArrayClose(vsqrt_out, new Float64Array([2, 3, 4]), 1e-10, 'Vector sqrt');

// Test 20: Vector normalize
console.log('');
console.log('Testing vector normalize...');
const vnorm_in = new Float64Array([3, 4, 0]);
const vnorm_out = new Float64Array(3);
accelerate.normalize(vnorm_in, vnorm_out);
assertArrayClose(vnorm_out, new Float64Array([0.6, 0.8, 0]), 1e-10, 'Vector normalize');

// Test 21: Euclidean distance
console.log('');
console.log('Testing Euclidean distance...');
const dist_a = new Float64Array([0, 0, 0]);
const dist_b = new Float64Array([3, 4, 0]);
const distance = accelerate.euclidean(dist_a, dist_b);
assertClose(distance, 5, 1e-10, 'Euclidean distance');

// Test 22: RMS
console.log('');
console.log('Testing RMS...');
const rms_vec = new Float64Array([1, 2, 3, 4, 5]);
const rmsResult = accelerate.rms(rms_vec);
const expectedRMS = Math.sqrt((1 + 4 + 9 + 16 + 25) / 5);
assertClose(rmsResult, expectedRMS, 1e-10, 'RMS calculation');

// Summary
console.log('');
console.log('='.repeat(70));
console.log('TEST RESULTS');
console.log('='.repeat(70));
console.log(`Passed: ${passed}`);
console.log(`Failed: ${failed}`);
console.log(`Total:  ${passed + failed}`);
console.log('');

if (failed === 0) {
  console.log('✓ All tests passed!');
  process.exit(0);
} else {
  console.error('✗ Some tests failed');
  process.exit(1);
}
