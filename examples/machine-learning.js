#!/usr/bin/env node
/**
 * Example: Machine Learning Operations
 * Demonstrates using Accelerate for ML inference
 */

const accelerate = require('../index');

console.log('Machine Learning Example');
console.log('='.repeat(50));
console.log('');

// Simple neural network layer: y = activation(W*x + b)
function denseLayer(input, weights, bias, output) {
  const inputSize = input.length;
  const outputSize = output.length;
  
  // Matrix-vector multiply: output = weights * input
  accelerate.matvec(weights, input, output, outputSize, inputSize);
  
  // Add bias: output = output + bias
  accelerate.axpy(1.0, bias, output);
  
  // ReLU activation: output = max(0, output)
  const zeros = new Float64Array(outputSize);
  for (let i = 0; i < outputSize; i++) {
    output[i] = Math.max(0, output[i]);
  }
  
  return output;
}

// Example: 2-layer network
console.log('Building 2-layer neural network...');
console.log('  Input: 784 (28×28 image)');
console.log('  Hidden: 128');
console.log('  Output: 10 (digits 0-9)');
console.log('');

// Layer 1: 784 → 128
const input = new Float64Array(784);
for (let i = 0; i < input.length; i++) {
  input[i] = Math.random();
}

const weights1 = new Float64Array(128 * 784);
const bias1 = new Float64Array(128);
const hidden = new Float64Array(128);

for (let i = 0; i < weights1.length; i++) weights1[i] = Math.random() - 0.5;
for (let i = 0; i < bias1.length; i++) bias1[i] = Math.random() - 0.5;

console.log('Layer 1: Forward pass...');
let start = process.hrtime.bigint();
denseLayer(input, weights1, bias1, hidden);
let end = process.hrtime.bigint();
console.log(`  Time: ${(Number(end - start) / 1e6).toFixed(3)}ms`);
console.log('');

// Layer 2: 128 → 10
const weights2 = new Float64Array(10 * 128);
const bias2 = new Float64Array(10);
const output = new Float64Array(10);

for (let i = 0; i < weights2.length; i++) weights2[i] = Math.random() - 0.5;
for (let i = 0; i < bias2.length; i++) bias2[i] = Math.random() - 0.5;

console.log('Layer 2: Forward pass...');
start = process.hrtime.bigint();
denseLayer(hidden, weights2, bias2, output);
end = process.hrtime.bigint();
console.log(`  Time: ${(Number(end - start) / 1e6).toFixed(3)}ms`);
console.log('');

// Softmax (normalize to probabilities)
console.log('Softmax activation...');
const expOutput = new Float64Array(10);
for (let i = 0; i < output.length; i++) {
  expOutput[i] = Math.exp(output[i]);
}
const sumExp = accelerate.sum(expOutput);
for (let i = 0; i < expOutput.length; i++) {
  expOutput[i] /= sumExp;
}

console.log('Output probabilities:');
for (let i = 0; i < 10; i++) {
  console.log(`  Digit ${i}: ${(expOutput[i] * 100).toFixed(2)}%`);
}
console.log('');

// K-means clustering example
console.log('K-means Clustering Example');
console.log('-'.repeat(50));

const numPoints = 1000;
const numClusters = 3;
const dimensions = 2;

// Generate random points
const points = new Float64Array(numPoints * dimensions);
for (let i = 0; i < points.length; i++) {
  points[i] = Math.random() * 100;
}

// Initialize centroids
const centroids = new Float64Array(numClusters * dimensions);
for (let i = 0; i < numClusters; i++) {
  const idx = Math.floor(Math.random() * numPoints);
  centroids[i * dimensions] = points[idx * dimensions];
  centroids[i * dimensions + 1] = points[idx * dimensions + 1];
}

console.log('Finding nearest centroids for 1000 points...');
start = process.hrtime.bigint();

const assignments = new Int32Array(numPoints);
const point = new Float64Array(dimensions);
const centroid = new Float64Array(dimensions);

for (let i = 0; i < numPoints; i++) {
  point[0] = points[i * dimensions];
  point[1] = points[i * dimensions + 1];
  
  let minDist = Infinity;
  let minCluster = 0;
  
  for (let j = 0; j < numClusters; j++) {
    centroid[0] = centroids[j * dimensions];
    centroid[1] = centroids[j * dimensions + 1];
    
    const dist = accelerate.euclidean(point, centroid);
    if (dist < minDist) {
      minDist = dist;
      minCluster = j;
    }
  }
  
  assignments[i] = minCluster;
}

end = process.hrtime.bigint();
console.log(`  Time: ${(Number(end - start) / 1e6).toFixed(3)}ms`);
console.log('');

// Count assignments
const counts = [0, 0, 0];
for (let i = 0; i < numPoints; i++) {
  counts[assignments[i]]++;
}

console.log('Cluster assignments:');
for (let i = 0; i < numClusters; i++) {
  console.log(`  Cluster ${i}: ${counts[i]} points`);
}
console.log('');

console.log('✓ Machine learning operations completed!');
