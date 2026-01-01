/**
 * Complete Machine Learning Pipeline Example
 * Demonstrates data preprocessing, neural network inference, and evaluation
 */

const accelerate = require('..');

console.log('=== Machine Learning Pipeline ===\n');

// ============================================================================
// 1. DATA PREPROCESSING
// ============================================================================

console.log('--- Step 1: Data Preprocessing ---');

// Generate synthetic dataset
const numSamples = 1000;
const numFeatures = 10;
const rawData = new Float64Array(numSamples * numFeatures);

for (let i = 0; i < rawData.length; i++) {
  rawData[i] = Math.random() * 200 - 100; // Range: -100 to 100
}

console.log('Raw data shape:', [numSamples, numFeatures]);
console.log('Raw data range:', [
  accelerate.min(rawData).toFixed(2),
  accelerate.max(rawData).toFixed(2)
]);

// Step 1a: Clip outliers
const clipped = new Float64Array(rawData.length);
accelerate.vclip(rawData, clipped, -50, 50);

// Step 1b: Standardization (z-score normalization)
function standardize(data, output) {
  const mean = accelerate.mean(data);
  const std = accelerate.stddev(data);
  
  for (let i = 0; i < data.length; i++) {
    output[i] = (data[i] - mean) / std;
  }
  
  return output;
}

const normalized = new Float64Array(clipped.length);
standardize(clipped, normalized);

console.log('After preprocessing:');
console.log('  Mean:', accelerate.mean(normalized).toFixed(6));
console.log('  Std Dev:', accelerate.stddev(normalized).toFixed(6));
console.log('  Range:', [
  accelerate.min(normalized).toFixed(2),
  accelerate.max(normalized).toFixed(2)
]);

// ============================================================================
// 2. NEURAL NETWORK INFERENCE
// ============================================================================

console.log('\n--- Step 2: Neural Network Inference ---');

// Network architecture: 10 -> 64 -> 32 -> 3 (classification)
const layer1Weights = new Float64Array(10 * 64);
const layer1Bias = new Float64Array(64);
const layer2Weights = new Float64Array(64 * 32);
const layer2Bias = new Float64Array(32);
const layer3Weights = new Float64Array(32 * 3);
const layer3Bias = new Float64Array(3);

// Initialize with random weights (Xavier initialization)
function initializeWeights(weights, fanIn, fanOut) {
  const limit = Math.sqrt(6.0 / (fanIn + fanOut));
  for (let i = 0; i < weights.length; i++) {
    weights[i] = (Math.random() * 2 - 1) * limit;
  }
}

initializeWeights(layer1Weights, 10, 64);
initializeWeights(layer2Weights, 64, 32);
initializeWeights(layer3Weights, 32, 3);

// ReLU activation
function relu(input, output) {
  accelerate.vclip(input, output, 0, Infinity);
  return output;
}

// Softmax activation
function softmax(logits, output) {
  // Subtract max for numerical stability
  const maxVal = accelerate.max(logits);
  const shifted = new Float64Array(logits.length);
  
  for (let i = 0; i < logits.length; i++) {
    shifted[i] = logits[i] - maxVal;
  }
  
  // Compute exp
  accelerate.vexp(shifted, output);
  
  // Normalize
  const sum = accelerate.sum(output);
  accelerate.vscale(output, 1.0 / sum, output);
  
  return output;
}

// Forward pass for one sample
function forward(input) {
  // Layer 1: input (10) -> hidden1 (64)
  const hidden1 = new Float64Array(64);
  accelerate.matvec(layer1Weights, input, hidden1, 64, 10);
  accelerate.vadd(hidden1, layer1Bias, hidden1);
  relu(hidden1, hidden1);
  
  // Layer 2: hidden1 (64) -> hidden2 (32)
  const hidden2 = new Float64Array(32);
  accelerate.matvec(layer2Weights, hidden1, hidden2, 32, 64);
  accelerate.vadd(hidden2, layer2Bias, hidden2);
  relu(hidden2, hidden2);
  
  // Layer 3: hidden2 (32) -> output (3)
  const logits = new Float64Array(3);
  accelerate.matvec(layer3Weights, hidden2, logits, 3, 32);
  accelerate.vadd(logits, layer3Bias, logits);
  
  // Softmax
  const probs = new Float64Array(3);
  softmax(logits, probs);
  
  return probs;
}

// Run inference on first sample
const sampleInput = normalized.subarray(0, 10);
const predictions = forward(sampleInput);

console.log('Network architecture: 10 -> 64 -> 32 -> 3');
console.log('Sample prediction:', Array.from(predictions).map(x => x.toFixed(4)));
console.log('Predicted class:', predictions.indexOf(accelerate.max(predictions)));

// Batch inference
console.log('\nRunning batch inference...');
console.time('Batch inference (1000 samples)');

const allPredictions = [];
for (let i = 0; i < numSamples; i++) {
  const input = normalized.subarray(i * numFeatures, (i + 1) * numFeatures);
  const pred = forward(input);
  allPredictions.push(Array.from(pred));
}

console.timeEnd('Batch inference (1000 samples)');

// ============================================================================
// 3. FEATURE ENGINEERING
// ============================================================================

console.log('\n--- Step 3: Feature Engineering ---');

// Compute polynomial features (x^2)
const squaredFeatures = new Float64Array(normalized.length);
accelerate.vsquare(normalized, squaredFeatures);

// Compute interaction features (for first two features)
const feature1 = normalized.subarray(0, numSamples);
const feature2 = normalized.subarray(numSamples, numSamples * 2);
const interaction = new Float64Array(numSamples);
accelerate.vmul(feature1, feature2, interaction);

console.log('Original features:', numFeatures);
console.log('Added squared features:', numFeatures);
console.log('Added interaction features: 1');
console.log('Total features:', numFeatures * 2 + 1);

// ============================================================================
// 4. DISTANCE-BASED METHODS
// ============================================================================

console.log('\n--- Step 4: Distance Computations ---');

// K-nearest neighbors: find distance to first sample
const query = normalized.subarray(0, numFeatures);
const distances = [];

for (let i = 1; i < Math.min(100, numSamples); i++) {
  const sample = normalized.subarray(i * numFeatures, (i + 1) * numFeatures);
  const dist = accelerate.euclidean(query, sample);
  distances.push({ index: i, distance: dist });
}

// Sort by distance
distances.sort((a, b) => a.distance - b.distance);

console.log('K-Nearest Neighbors (k=5):');
for (let i = 0; i < 5; i++) {
  console.log(`  Neighbor ${i + 1}: sample ${distances[i].index}, distance ${distances[i].distance.toFixed(4)}`);
}

// ============================================================================
// 5. DIMENSIONALITY REDUCTION (PCA-like projection)
// ============================================================================

console.log('\n--- Step 5: Dimensionality Reduction ---');

// Simple random projection (approximates PCA)
const projectionMatrix = new Float64Array(numFeatures * 2); // Project to 2D
initializeWeights(projectionMatrix, numFeatures, 2);

// Normalize projection matrix columns
for (let col = 0; col < 2; col++) {
  const column = new Float64Array(numFeatures);
  for (let row = 0; row < numFeatures; row++) {
    column[row] = projectionMatrix[row * 2 + col];
  }
  
  const normalized = new Float64Array(numFeatures);
  accelerate.normalize(column, normalized);
  
  for (let row = 0; row < numFeatures; row++) {
    projectionMatrix[row * 2 + col] = normalized[row];
  }
}

// Project first sample
const projected = new Float64Array(2);
accelerate.matvec(projectionMatrix, sampleInput, projected, 2, numFeatures);

console.log('Original dimensions:', numFeatures);
console.log('Reduced dimensions: 2');
console.log('Sample projection:', Array.from(projected).map(x => x.toFixed(4)));

// ============================================================================
// 6. SIGNAL PROCESSING FOR TIME SERIES
// ============================================================================

console.log('\n--- Step 6: Time Series Analysis ---');

// Generate synthetic time series
const timeSeriesLength = 1024;
const timeSeries = new Float64Array(timeSeriesLength);

for (let i = 0; i < timeSeriesLength; i++) {
  // Mix of frequencies
  timeSeries[i] = 
    Math.sin(2 * Math.PI * 5 * i / timeSeriesLength) +
    0.5 * Math.sin(2 * Math.PI * 10 * i / timeSeriesLength) +
    0.3 * Math.sin(2 * Math.PI * 20 * i / timeSeriesLength);
}

// Apply Hanning window
const window = accelerate.hanning(timeSeriesLength);
const windowed = new Float64Array(timeSeriesLength);
accelerate.vmul(timeSeries, window, windowed);

// Compute FFT
const spectrum = accelerate.fft(windowed);

// Find dominant frequencies
const magnitudes = new Float64Array(spectrum.real.length);
for (let i = 0; i < magnitudes.length; i++) {
  magnitudes[i] = Math.sqrt(spectrum.real[i]**2 + spectrum.imag[i]**2);
}

// Find peaks
const peaks = [];
for (let i = 1; i < magnitudes.length - 1; i++) {
  if (magnitudes[i] > magnitudes[i-1] && magnitudes[i] > magnitudes[i+1] && magnitudes[i] > 10) {
    peaks.push({ bin: i, magnitude: magnitudes[i] });
  }
}

peaks.sort((a, b) => b.magnitude - a.magnitude);

console.log('Time series length:', timeSeriesLength);
console.log('Dominant frequency bins:', peaks.slice(0, 3).map(p => p.bin));

// ============================================================================
// 7. PERFORMANCE SUMMARY
// ============================================================================

console.log('\n--- Performance Summary ---');

// Benchmark key operations
const benchSize = 10000;
const benchVec1 = new Float64Array(benchSize);
const benchVec2 = new Float64Array(benchSize);
const benchResult = new Float64Array(benchSize);

for (let i = 0; i < benchSize; i++) {
  benchVec1[i] = Math.random();
  benchVec2[i] = Math.random();
}

console.time('Vector addition (10k elements)');
for (let i = 0; i < 100; i++) {
  accelerate.vadd(benchVec1, benchVec2, benchResult);
}
console.timeEnd('Vector addition (10k elements)');

console.time('Dot product (10k elements)');
for (let i = 0; i < 100; i++) {
  accelerate.dot(benchVec1, benchVec2);
}
console.timeEnd('Dot product (10k elements)');

console.time('Statistical operations (10k elements)');
for (let i = 0; i < 100; i++) {
  accelerate.mean(benchVec1);
  accelerate.stddev(benchVec1);
  accelerate.minmax(benchVec1);
}
console.timeEnd('Statistical operations (10k elements)');

console.log('\n✓ ML Pipeline Complete!');
