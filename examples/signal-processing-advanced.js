/**
 * Advanced Signal Processing Examples
 * Demonstrates convolution, correlation, windowing, and FFT/IFFT
 */

const accelerate = require('..');

console.log('=== Advanced Signal Processing ===\n');

// 1. Convolution Example
console.log('--- Convolution ---');
const signal = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8]);
const kernel = new Float64Array([0.25, 0.5, 0.25]); // Moving average
const convResult = new Float64Array(signal.length - kernel.length + 1);

accelerate.conv(signal, kernel, convResult);
console.log('Signal:', Array.from(signal));
console.log('Kernel (moving avg):', Array.from(kernel));
console.log('Convolution result:', Array.from(convResult).map(x => x.toFixed(2)));

// 2. Cross-correlation Example
console.log('\n--- Cross-Correlation ---');
const sig1 = new Float64Array(100);
const sig2 = new Float64Array(100);

// Create two similar signals with a delay
for (let i = 0; i < 100; i++) {
  sig1[i] = Math.sin(2 * Math.PI * i / 20);
  sig2[i] = i >= 10 ? Math.sin(2 * Math.PI * (i - 10) / 20) : 0;
}

const xcorrResult = new Float64Array(sig1.length + sig2.length - 1);
accelerate.xcorr(sig1, sig2, xcorrResult);

// Find peak (indicates delay)
const peakIdx = xcorrResult.indexOf(accelerate.max(xcorrResult));
console.log('Detected delay:', Math.abs(peakIdx - sig1.length + 1), 'samples');

// 3. Window Functions
console.log('\n--- Window Functions ---');
const windowSize = 64;

const hammingWin = accelerate.hamming(windowSize);
const hanningWin = accelerate.hanning(windowSize);
const blackmanWin = accelerate.blackman(windowSize);

console.log('Hamming window center value:', hammingWin[windowSize/2].toFixed(4));
console.log('Hanning window center value:', hanningWin[windowSize/2].toFixed(4));
console.log('Blackman window center value:', blackmanWin[windowSize/2].toFixed(4));

// 4. FFT and IFFT Round-trip
console.log('\n--- FFT/IFFT Round-trip ---');
const fftSize = 256;
const testSignal = new Float64Array(fftSize);

// Create test signal: sum of two sine waves
for (let i = 0; i < fftSize; i++) {
  testSignal[i] = Math.sin(2 * Math.PI * 5 * i / fftSize) + 
                  0.5 * Math.sin(2 * Math.PI * 10 * i / fftSize);
}

// Forward FFT
const spectrum = accelerate.fft(testSignal);
console.log('FFT output size:', spectrum.real.length, 'bins');

// Find dominant frequencies
const magnitudes = new Float64Array(spectrum.real.length);
for (let i = 0; i < magnitudes.length; i++) {
  magnitudes[i] = Math.sqrt(spectrum.real[i]**2 + spectrum.imag[i]**2);
}

const peak1 = magnitudes.indexOf(accelerate.max(magnitudes));
console.log('Dominant frequency bin:', peak1);

// Inverse FFT
const reconstructed = accelerate.ifft(spectrum.real, spectrum.imag);

// Check reconstruction error
let maxError = 0;
for (let i = 0; i < fftSize; i++) {
  const error = Math.abs(testSignal[i] - reconstructed[i]);
  if (error > maxError) maxError = error;
}

console.log('Max reconstruction error:', maxError.toFixed(10));
console.log('Reconstruction successful:', maxError < 1e-10 ? 'YES' : 'NO');

// 5. Windowed FFT (for spectral analysis)
console.log('\n--- Windowed FFT ---');
const windowedSignal = new Float64Array(fftSize);
const window = accelerate.hanning(fftSize);

accelerate.vmul(testSignal, window, windowedSignal);
const windowedSpectrum = accelerate.fft(windowedSignal);

console.log('Windowed FFT reduces spectral leakage');
console.log('Original signal energy:', accelerate.sumOfSquares(testSignal).toFixed(2));
console.log('Windowed signal energy:', accelerate.sumOfSquares(windowedSignal).toFixed(2));
