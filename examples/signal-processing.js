#!/usr/bin/env node
/**
 * Example: Signal Processing with FFT
 * Demonstrates frequency analysis of audio signals
 */

const accelerate = require('../index');

console.log('Signal Processing Example');
console.log('='.repeat(50));
console.log('');

// Create a composite signal: 440 Hz (A4) + 880 Hz (A5)
const sampleRate = 44100; // CD quality
const duration = 1; // 1 second
const fftSize = 16384; // Must be power of 2

console.log(`Sample rate: ${sampleRate} Hz`);
console.log(`FFT size: ${fftSize}`);
console.log(`Frequency resolution: ${(sampleRate / fftSize).toFixed(2)} Hz`);
console.log('');

// Generate signal
console.log('Generating signal (440 Hz + 880 Hz)...');
const signal = new Float64Array(fftSize);
for (let i = 0; i < fftSize; i++) {
  const t = i / sampleRate;
  signal[i] = 
    Math.sin(2 * Math.PI * 440 * t) +  // A4
    Math.sin(2 * Math.PI * 880 * t);   // A5
}

// Perform FFT
console.log('Performing FFT...');
const start = process.hrtime.bigint();
const spectrum = accelerate.fft(signal);
const end = process.hrtime.bigint();

const timeMs = Number(end - start) / 1e6;
console.log(`✓ Completed in ${timeMs.toFixed(3)}ms`);
console.log('');

// Calculate magnitudes
const magnitudes = new Float64Array(spectrum.real.length);
for (let i = 0; i < magnitudes.length; i++) {
  magnitudes[i] = Math.sqrt(
    spectrum.real[i] ** 2 + spectrum.imag[i] ** 2
  );
}

// Find peaks
console.log('Finding frequency peaks...');
const peaks = [];
const threshold = Math.max(...magnitudes) * 0.5;

for (let i = 1; i < magnitudes.length - 1; i++) {
  if (magnitudes[i] > threshold &&
      magnitudes[i] > magnitudes[i - 1] &&
      magnitudes[i] > magnitudes[i + 1]) {
    const frequency = i * sampleRate / fftSize;
    peaks.push({ frequency, magnitude: magnitudes[i] });
  }
}

peaks.sort((a, b) => b.magnitude - a.magnitude);

console.log('Top frequency components:');
peaks.slice(0, 5).forEach((peak, i) => {
  console.log(`  ${i + 1}. ${peak.frequency.toFixed(1)} Hz (magnitude: ${peak.magnitude.toFixed(2)})`);
});
