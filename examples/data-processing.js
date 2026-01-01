/**
 * Data Processing Examples
 * Demonstrates clipping, thresholding, interpolation, and matrix operations
 */

const accelerate = require('..');

console.log('=== Data Processing ===\n');

// 1. Clipping
console.log('--- Clipping ---');
const data = new Float64Array(10);
for (let i = 0; i < 10; i++) {
  data[i] = (i - 5) * 10; // -50 to 40
}

const clipped = new Float64Array(10);
accelerate.vclip(data, clipped, -20, 20);

console.log('Original:', Array.from(data));
console.log('Clipped [-20, 20]:', Array.from(clipped));

// 2. Thresholding
console.log('\n--- Thresholding ---');
const signal = new Float64Array(10);
for (let i = 0; i < 10; i++) {
  signal[i] = Math.random() * 100;
}

const thresholded = new Float64Array(10);
accelerate.vthreshold(signal, thresholded, 50);

console.log('Original:', Array.from(signal).map(x => x.toFixed(1)));
console.log('Threshold > 50:', Array.from(thresholded).map(x => x.toFixed(1)));

// 3. Vector Reversal
console.log('\n--- Vector Reversal ---');
const sequence = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
const reversed = new Float64Array(10);

accelerate.vreverse(sequence, reversed);

console.log('Original:', Array.from(sequence));
console.log('Reversed:', Array.from(reversed));

// 4. Matrix Transpose
console.log('\n--- Matrix Transpose ---');
const rows = 3, cols = 4;
const matrix = new Float64Array(rows * cols);

// Fill matrix
for (let i = 0; i < rows; i++) {
  for (let j = 0; j < cols; j++) {
    matrix[i * cols + j] = i * cols + j + 1;
  }
}

const transposed = new Float64Array(rows * cols);
accelerate.transpose(matrix, transposed, rows, cols);

console.log('Original matrix (3×4):');
for (let i = 0; i < rows; i++) {
  const row = [];
  for (let j = 0; j < cols; j++) {
    row.push(matrix[i * cols + j].toFixed(0).padStart(3));
  }
  console.log('  [' + row.join(' ') + ']');
}

console.log('Transposed matrix (4×3):');
for (let i = 0; i < cols; i++) {
  const row = [];
  for (let j = 0; j < rows; j++) {
    row.push(transposed[i * rows + j].toFixed(0).padStart(3));
  }
  console.log('  [' + row.join(' ') + ']');
}

// 5. Linear Interpolation
console.log('\n--- Linear Interpolation ---');
const xData = new Float64Array([0, 1, 2, 3, 4]);
const yData = new Float64Array([0, 1, 4, 9, 16]); // y = x²

const xiData = new Float64Array([0.5, 1.5, 2.5, 3.5]);
const yiData = new Float64Array(4);

accelerate.interp1d(xData, yData, xiData, yiData);

console.log('Known points (x, y):');
for (let i = 0; i < xData.length; i++) {
  console.log(`  (${xData[i]}, ${yData[i]})`);
}

console.log('Interpolated points:');
for (let i = 0; i < xiData.length; i++) {
  console.log(`  x=${xiData[i]} → y=${yiData[i].toFixed(2)}`);
}

// 6. Data Normalization Pipeline
console.log('\n--- Data Normalization Pipeline ---');
const rawData = new Float64Array(100);
for (let i = 0; i < 100; i++) {
  rawData[i] = Math.random() * 200 - 100; // -100 to 100
}

// Step 1: Clip outliers
const clippedData = new Float64Array(100);
accelerate.vclip(rawData, clippedData, -50, 50);

// Step 2: Take absolute value
const absData = new Float64Array(100);
accelerate.vabs(clippedData, absData);

// Step 3: Normalize to [0, 1]
const maxVal = accelerate.max(absData);
const normalized = new Float64Array(100);
accelerate.vscale(absData, 1.0 / maxVal, normalized);

console.log('Pipeline statistics:');
console.log('  Raw data range:', [accelerate.min(rawData).toFixed(2), accelerate.max(rawData).toFixed(2)]);
console.log('  After clipping:', [accelerate.min(clippedData).toFixed(2), accelerate.max(clippedData).toFixed(2)]);
console.log('  After abs:', [accelerate.min(absData).toFixed(2), accelerate.max(absData).toFixed(2)]);
console.log('  After normalization:', [accelerate.min(normalized).toFixed(2), accelerate.max(normalized).toFixed(2)]);
