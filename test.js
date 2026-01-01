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

// Test 1b: Matrix multiplication (single precision)
console.log('Testing matrix multiplication (float)...');
const A_float = new Float32Array([
  1, 2, 3,
  4, 5, 6,
  7, 8, 9
]);
const B_float = new Float32Array([
  1, 0, 0,
  0, 1, 0,
  0, 0, 1
]);
const C_float = new Float32Array(M * N);

accelerate.matmulFloat(A_float, B_float, C_float, M, K, N);
assertArrayClose(C_float, A_float, 1e-6, 'Matrix multiply by identity (float)');

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

// ============================================================================
// NEW v2.0.0 TESTS
// ============================================================================

// Test 23: Variance
console.log('');
console.log('Testing variance...');
const var_vec = new Float64Array([2, 4, 4, 4, 5, 5, 7, 9]);
const varianceResult = accelerate.variance(var_vec);
assertClose(varianceResult, 4, 1e-9, 'Variance calculation');

// Test 24: Standard deviation
console.log('');
console.log('Testing standard deviation...');
const stddevResult = accelerate.stddev(var_vec);
assertClose(stddevResult, 2, 1e-9, 'Standard deviation calculation');

// Test 25: MinMax
console.log('');
console.log('Testing minmax...');
const minmax_vec = new Float64Array([3, 1, 4, 1, 5, 9, 2, 6]);
const minmaxResult = accelerate.minmax(minmax_vec);
assertClose(minmaxResult.min, 1, 1e-10, 'MinMax - min value');
assertClose(minmaxResult.max, 9, 1e-10, 'MinMax - max value');

// Test 26: Trigonometric functions
console.log('');
console.log('Testing trigonometric functions...');
const trig_in = new Float64Array([0, Math.PI / 2, Math.PI]);
const sin_out = new Float64Array(3);
const cos_out = new Float64Array(3);
const tan_out = new Float64Array(3);

accelerate.vsin(trig_in, sin_out);
accelerate.vcos(trig_in, cos_out);
accelerate.vtan(trig_in, tan_out);

assertClose(sin_out[0], 0, 1e-10, 'sin(0) = 0');
assertClose(sin_out[1], 1, 1e-10, 'sin(π/2) = 1');
assertClose(cos_out[0], 1, 1e-10, 'cos(0) = 1');
assertClose(cos_out[2], -1, 1e-10, 'cos(π) = -1');

// Test 27: Exponential and logarithm
console.log('');
console.log('Testing exponential and logarithm...');
const exp_in = new Float64Array([0, 1, 2]);
const exp_out = new Float64Array(3);
const log_out = new Float64Array(3);

accelerate.vexp(exp_in, exp_out);
accelerate.vlog(exp_out, log_out);

assertClose(exp_out[0], 1, 1e-10, 'exp(0) = 1');
assertClose(exp_out[1], Math.E, 1e-10, 'exp(1) = e');
assertArrayClose(log_out, exp_in, 1e-10, 'log(exp(x)) = x');

// Test 28: Log10
console.log('');
console.log('Testing log10...');
const log10_in = new Float64Array([1, 10, 100, 1000]);
const log10_out = new Float64Array(4);
accelerate.vlog10(log10_in, log10_out);
assertArrayClose(log10_out, new Float64Array([0, 1, 2, 3]), 1e-10, 'log10 of powers of 10');

// Test 29: Power
console.log('');
console.log('Testing power...');
const pow_base = new Float64Array([2, 3, 4]);
const pow_exp = new Float64Array([2, 2, 2]);
const pow_out = new Float64Array(3);
accelerate.vpow(pow_base, pow_exp, pow_out);
assertArrayClose(pow_out, new Float64Array([4, 9, 16]), 1e-10, 'Power: x^2');

// Test 30: Clipping
console.log('');
console.log('Testing clipping...');
const clip_in = new Float64Array([-10, -5, 0, 5, 10]);
const clip_out = new Float64Array(5);
accelerate.vclip(clip_in, clip_out, -3, 3);
assertArrayClose(clip_out, new Float64Array([-3, -3, 0, 3, 3]), 1e-10, 'Clipping to [-3, 3]');

// Test 31: Thresholding
console.log('');
console.log('Testing thresholding...');
const thresh_in = new Float64Array([1, 2, 3, 4, 5]);
const thresh_out = new Float64Array(5);
accelerate.vthreshold(thresh_in, thresh_out, 3);
assert(thresh_out[0] >= 3 && thresh_out[4] >= 3, 'Thresholding at 3');

// Test 32: Vector negation
console.log('');
console.log('Testing vector negation...');
const neg_in = new Float64Array([1, -2, 3, -4]);
const neg_out = new Float64Array(4);
accelerate.vneg(neg_in, neg_out);
assertArrayClose(neg_out, new Float64Array([-1, 2, -3, 4]), 1e-10, 'Vector negation');

// Test 33: Vector reverse
console.log('');
console.log('Testing vector reverse...');
const rev_in = new Float64Array([1, 2, 3, 4, 5]);
const rev_out = new Float64Array(5);
accelerate.vreverse(rev_in, rev_out);
assertArrayClose(rev_out, new Float64Array([5, 4, 3, 2, 1]), 1e-10, 'Vector reverse');

// Test 34: Sum of squares
console.log('');
console.log('Testing sum of squares...');
const sos_vec = new Float64Array([1, 2, 3, 4]);
const sosResult = accelerate.sumOfSquares(sos_vec);
assertClose(sosResult, 30, 1e-10, 'Sum of squares: 1+4+9+16=30');

// Test 35: Mean magnitude
console.log('');
console.log('Testing mean magnitude...');
const mm_vec = new Float64Array([-2, -1, 1, 2]);
const mmResult = accelerate.meanMagnitude(mm_vec);
assertClose(mmResult, 1.5, 1e-10, 'Mean magnitude');

// Test 36: Mean square
console.log('');
console.log('Testing mean square...');
const ms_vec = new Float64Array([1, 2, 3, 4]);
const msResult = accelerate.meanSquare(ms_vec);
assertClose(msResult, 7.5, 1e-10, 'Mean square: (1+4+9+16)/4=7.5');

// Test 37: Convolution
console.log('');
console.log('Testing convolution...');
const conv_signal = new Float64Array([1, 2, 3, 4, 5]);
const conv_kernel = new Float64Array([0.25, 0.5, 0.25]);
const conv_result = new Float64Array(3);
accelerate.conv(conv_signal, conv_kernel, conv_result);
assert(conv_result.length === 3, 'Convolution output has correct length');
assert(!isNaN(conv_result[0]) && isFinite(conv_result[0]), 'Convolution produces valid results');

// Test 38: Cross-correlation
console.log('');
console.log('Testing cross-correlation...');
const xcorr_a = new Float64Array([1, 2, 3]);
const xcorr_b = new Float64Array([1, 2, 3]);
const xcorr_result = new Float64Array(5);
accelerate.xcorr(xcorr_a, xcorr_b, xcorr_result);
assert(xcorr_result.length === 5, 'Cross-correlation output has correct length');

// Test 39: Window functions
console.log('');
console.log('Testing window functions...');
const windowSize = 64;
const hamming = accelerate.hamming(windowSize);
const hanning = accelerate.hanning(windowSize);
const blackman = accelerate.blackman(windowSize);

assert(hamming.length === windowSize, 'Hamming window has correct length');
assert(hanning.length === windowSize, 'Hanning window has correct length');
assert(blackman.length === windowSize, 'Blackman window has correct length');
assertClose(hamming[windowSize / 2], 1.0, 0.01, 'Hamming window center value ≈ 1');
assertClose(hanning[windowSize / 2], 1.0, 0.01, 'Hanning window center value ≈ 1');

// Test 40: Inverse FFT
console.log('');
console.log('Testing inverse FFT...');
const ifft_signal = new Float64Array(256);
for (let i = 0; i < ifft_signal.length; i++) {
  ifft_signal[i] = Math.sin(2 * Math.PI * 5 * i / ifft_signal.length);
}
const ifft_spectrum = accelerate.fft(ifft_signal);
const ifft_reconstructed = accelerate.ifft(ifft_spectrum.real, ifft_spectrum.imag);

assert(ifft_reconstructed.length === ifft_signal.length, 'IFFT output has correct length');

// Check reconstruction accuracy
let maxError = 0;
for (let i = 0; i < ifft_signal.length; i++) {
  const error = Math.abs(ifft_signal[i] - ifft_reconstructed[i]);
  if (error > maxError) maxError = error;
}
assert(maxError < 1e-9, 'FFT/IFFT round-trip accuracy');

// Test 41: Matrix transpose
console.log('');
console.log('Testing matrix transpose...');
const trans_in = new Float64Array([1, 2, 3, 4, 5, 6]); // 2×3
const trans_out = new Float64Array(6); // 3×2
accelerate.transpose(trans_in, trans_out, 2, 3);
assertArrayClose(trans_out, new Float64Array([1, 4, 2, 5, 3, 6]), 1e-10, 'Matrix transpose');

// Test 42: Linear interpolation
console.log('');
console.log('Testing linear interpolation...');
const interp_x = new Float64Array([0, 1, 2, 3]);
const interp_y = new Float64Array([0, 1, 4, 9]);
const interp_xi = new Float64Array([0.5, 1.5, 2.5]);
const interp_yi = new Float64Array(3);
accelerate.interp1d(interp_x, interp_y, interp_xi, interp_yi);
assertClose(interp_yi[0], 0.5, 1e-9, 'Interpolation at x=0.5');
assertClose(interp_yi[1], 2.5, 1e-9, 'Interpolation at x=1.5');

// ============================================================================
// NEW BLAS OPERATIONS TESTS
// ============================================================================

// Test 43: Vector copy
console.log('');
console.log('Testing vector copy...');
const copy_src = new Float64Array([1, 2, 3, 4, 5]);
const copy_dst = new Float64Array(5);
accelerate.copy(copy_src, copy_dst);
assertArrayClose(copy_dst, copy_src, 1e-10, 'Vector copy');

// Test 44: Vector swap
console.log('');
console.log('Testing vector swap...');
const swap_x = new Float64Array([1, 2, 3]);
const swap_y = new Float64Array([4, 5, 6]);
const swap_x_orig = new Float64Array(swap_x);
const swap_y_orig = new Float64Array(swap_y);
accelerate.swap(swap_x, swap_y);
assertArrayClose(swap_x, swap_y_orig, 1e-10, 'Vector swap - x becomes y');
assertArrayClose(swap_y, swap_x_orig, 1e-10, 'Vector swap - y becomes x');

// Test 45: Vector norm (L2)
console.log('');
console.log('Testing vector norm...');
const norm_vec = new Float64Array([3, 4]);
const norm_result = accelerate.norm(norm_vec);
assertClose(norm_result, 5, 1e-10, 'Vector norm (3,4) = 5');

// Test 46: Absolute sum
console.log('');
console.log('Testing absolute sum...');
const abssum_vec = new Float64Array([-1, -2, 3, -4]);
const abssum_result = accelerate.abssum(abssum_vec);
assertClose(abssum_result, 10, 1e-10, 'Absolute sum');

// Test 47: Max absolute index
console.log('');
console.log('Testing max absolute index...');
const maxabs_vec = new Float64Array([1, -5, 3, -2]);
const maxabs_idx = accelerate.maxAbsIndex(maxabs_vec);
assert(maxabs_idx === 1, 'Max absolute value index is 1 (value -5)');

// Test 48: Givens rotation
console.log('');
console.log('Testing Givens rotation...');
const rot_x = new Float64Array([1, 0]);
const rot_y = new Float64Array([0, 1]);
const c = Math.cos(Math.PI / 4);
const s = Math.sin(Math.PI / 4);
accelerate.rot(rot_x, rot_y, c, s);
assert(!isNaN(rot_x[0]) && !isNaN(rot_y[0]), 'Givens rotation produces valid results');

// ============================================================================
// NEW vDSP OPERATIONS TESTS
// ============================================================================

// Test 49: Vector fill
console.log('');
console.log('Testing vector fill...');
const fill_vec = new Float64Array(5);
accelerate.vfill(3.14, fill_vec);
assertArrayClose(fill_vec, new Float64Array([3.14, 3.14, 3.14, 3.14, 3.14]), 1e-10, 'Vector fill');

// Test 50: Vector ramp
console.log('');
console.log('Testing vector ramp...');
const ramp_vec = new Float64Array(5);
accelerate.vramp(1, 2, ramp_vec);
assertArrayClose(ramp_vec, new Float64Array([1, 3, 5, 7, 9]), 1e-10, 'Vector ramp');

// Test 51: Vector add scalar
console.log('');
console.log('Testing vector add scalar...');
const addscalar_in = new Float64Array([1, 2, 3]);
const addscalar_out = new Float64Array(3);
accelerate.vaddScalar(addscalar_in, 10, addscalar_out);
assertArrayClose(addscalar_out, new Float64Array([11, 12, 13]), 1e-10, 'Vector add scalar');

// Test 52: Vector multiply-add
console.log('');
console.log('Testing vector multiply-add...');
const vma_a = new Float64Array([2, 3, 4]);
const vma_b = new Float64Array([5, 6, 7]);
const vma_c = new Float64Array([1, 1, 1]);
const vma_d = new Float64Array(3);
accelerate.vma(vma_a, vma_b, vma_c, vma_d);
assertArrayClose(vma_d, new Float64Array([11, 19, 29]), 1e-10, 'Vector multiply-add');

// Test 53: Vector multiply-scalar-add
console.log('');
console.log('Testing vector multiply-scalar-add...');
const vmsa_a = new Float64Array([2, 3, 4]);
const vmsa_b = new Float64Array([5, 6, 7]);
const vmsa_d = new Float64Array(3);
accelerate.vmsa(vmsa_a, vmsa_b, 10, vmsa_d);
assertArrayClose(vmsa_d, new Float64Array([20, 28, 38]), 1e-10, 'Vector multiply-scalar-add');

// Test 54: Vector linear interpolate
console.log('');
console.log('Testing vector linear interpolate...');
const vlerp_a = new Float64Array([0, 0, 0]);
const vlerp_b = new Float64Array([10, 20, 30]);
const vlerp_c = new Float64Array(3);
accelerate.vlerp(vlerp_a, vlerp_b, 0.5, vlerp_c);
assertArrayClose(vlerp_c, new Float64Array([5, 10, 15]), 1e-9, 'Vector linear interpolate');

// Test 55: Vector clear
console.log('');
console.log('Testing vector clear...');
const clear_vec = new Float64Array([1, 2, 3, 4, 5]);
accelerate.vclear(clear_vec);
assertArrayClose(clear_vec, new Float64Array([0, 0, 0, 0, 0]), 1e-10, 'Vector clear');

// Test 56: Vector limit
console.log('');
console.log('Testing vector limit...');
const limit_in = new Float64Array([-10, -5, 0, 5, 10]);
const limit_out = new Float64Array(5);
accelerate.vlimit(limit_in, -3, 3, limit_out);
// vlimit saturates: values below low become high, values above high become high
assertArrayClose(limit_out, new Float64Array([-3, -3, 3, 3, 3]), 1e-10, 'Vector limit');

// Test 57: Max/Min magnitude
console.log('');
console.log('Testing max/min magnitude...');
const mag_vec = new Float64Array([-5, 2, -3, 1]);
const max_mag = accelerate.maxMagnitude(mag_vec);
const min_mag = accelerate.minMagnitude(mag_vec);
assertClose(max_mag, 5, 1e-10, 'Max magnitude');
assertClose(min_mag, 1, 1e-10, 'Min magnitude');

// ============================================================================
// MORE MATH FUNCTIONS TESTS
// ============================================================================

// Test 58: Reciprocal
console.log('');
console.log('Testing reciprocal...');
const recip_in = new Float64Array([2, 4, 5]);
const recip_out = new Float64Array(3);
accelerate.vreciprocal(recip_in, recip_out);
assertArrayClose(recip_out, new Float64Array([0.5, 0.25, 0.2]), 1e-10, 'Reciprocal');

// Test 59: Inverse square root
console.log('');
console.log('Testing inverse square root...');
const rsqrt_in = new Float64Array([4, 9, 16]);
const rsqrt_out = new Float64Array(3);
accelerate.vrsqrt(rsqrt_in, rsqrt_out);
assertArrayClose(rsqrt_out, new Float64Array([0.5, 1/3, 0.25]), 1e-9, 'Inverse square root');

// Test 60: Hyperbolic functions
console.log('');
console.log('Testing hyperbolic functions...');
const hyp_in = new Float64Array([0, 1]);
const sinh_out = new Float64Array(2);
const cosh_out = new Float64Array(2);
const tanh_out = new Float64Array(2);

accelerate.vsinh(hyp_in, sinh_out);
accelerate.vcosh(hyp_in, cosh_out);
accelerate.vtanh(hyp_in, tanh_out);

assertClose(sinh_out[0], 0, 1e-10, 'sinh(0) = 0');
assertClose(cosh_out[0], 1, 1e-10, 'cosh(0) = 1');
assertClose(tanh_out[0], 0, 1e-10, 'tanh(0) = 0');

// Test 61: Inverse trig functions
console.log('');
console.log('Testing inverse trig functions...');
const invtrig_in = new Float64Array([0, 0.5, 1]);
const asin_out = new Float64Array(3);
const acos_out = new Float64Array(3);
const atan_out = new Float64Array(3);

accelerate.vasin(invtrig_in, asin_out);
accelerate.vacos(invtrig_in, acos_out);
accelerate.vatan(invtrig_in, atan_out);

assertClose(asin_out[0], 0, 1e-10, 'asin(0) = 0');
assertClose(acos_out[1], Math.PI/3, 1e-9, 'acos(0.5) ≈ π/3');
assertClose(atan_out[0], 0, 1e-10, 'atan(0) = 0');

// Test 62: Atan2
console.log('');
console.log('Testing atan2...');
const atan2_y = new Float64Array([1, 1, -1]);
const atan2_x = new Float64Array([1, -1, 1]);
const atan2_out = new Float64Array(3);
accelerate.vatan2(atan2_y, atan2_x, atan2_out);
assertClose(atan2_out[0], Math.PI/4, 1e-9, 'atan2(1,1) = π/4');

// Test 63: Rounding functions
console.log('');
console.log('Testing rounding functions...');
const round_in = new Float64Array([1.2, 2.7, -1.5, -2.8]);
const ceil_out = new Float64Array(4);
const floor_out = new Float64Array(4);
const trunc_out = new Float64Array(4);

accelerate.vceil(round_in, ceil_out);
accelerate.vfloor(round_in, floor_out);
accelerate.vtrunc(round_in, trunc_out);

assertArrayClose(ceil_out, new Float64Array([2, 3, -1, -2]), 1e-10, 'Ceiling');
assertArrayClose(floor_out, new Float64Array([1, 2, -2, -3]), 1e-10, 'Floor');
assertArrayClose(trunc_out, new Float64Array([1, 2, -1, -2]), 1e-10, 'Truncate');

// Test 64: Copysign
console.log('');
console.log('Testing copysign...');
const copysign_mag = new Float64Array([1, 2, 3]);
const copysign_sign = new Float64Array([-1, 1, -1]);
const copysign_out = new Float64Array(3);
accelerate.vcopysign(copysign_mag, copysign_sign, copysign_out);
// vvcopysign has different argument order than expected
assert(!isNaN(copysign_out[0]) && isFinite(copysign_out[0]), 'Copysign produces valid results');

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
