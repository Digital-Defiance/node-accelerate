/**
 * Statistical Operations Examples
 * Demonstrates statistical functions using Apple Accelerate
 */

const accelerate = require('..');

console.log('=== Statistical Operations ===\n');

// Create sample data
const data = new Float64Array(1000);
for (let i = 0; i < data.length; i++) {
  data[i] = Math.random() * 100;
}

console.log('Sample size:', data.length);

// Basic statistics
console.log('\n--- Basic Statistics ---');
console.log('Mean:', accelerate.mean(data).toFixed(2));
console.log('Variance:', accelerate.variance(data).toFixed(2));
console.log('Standard Deviation:', accelerate.stddev(data).toFixed(2));
console.log('RMS:', accelerate.rms(data).toFixed(2));

// Min/Max
const { min, max } = accelerate.minmax(data);
console.log('\n--- Range ---');
console.log('Min:', min.toFixed(2));
console.log('Max:', max.toFixed(2));
console.log('Range:', (max - min).toFixed(2));

// Magnitude statistics
console.log('\n--- Magnitude Statistics ---');
console.log('Sum:', accelerate.sum(data).toFixed(2));
console.log('Sum of Squares:', accelerate.sumOfSquares(data).toFixed(2));
console.log('Mean Magnitude:', accelerate.meanMagnitude(data).toFixed(2));
console.log('Mean Square:', accelerate.meanSquare(data).toFixed(2));

// Normalized data
const normalized = new Float64Array(data.length);
accelerate.normalize(data, normalized);
console.log('\n--- Normalized Data ---');
console.log('Normalized mean:', accelerate.mean(normalized).toFixed(6));
console.log('Normalized magnitude:', Math.sqrt(accelerate.sumOfSquares(normalized)).toFixed(6));
