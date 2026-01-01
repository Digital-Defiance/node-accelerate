/**
 * Trigonometric Functions Examples
 * Demonstrates vectorized trig operations using Apple Accelerate
 */

const accelerate = require('..');

console.log('=== Trigonometric Functions ===\n');

// Create angle array (0 to 2π)
const n = 1000;
const angles = new Float64Array(n);
for (let i = 0; i < n; i++) {
  angles[i] = (i / n) * 2 * Math.PI;
}

// Compute sin, cos, tan
const sinValues = new Float64Array(n);
const cosValues = new Float64Array(n);
const tanValues = new Float64Array(n);

console.time('Vectorized trig operations');
accelerate.vsin(angles, sinValues);
accelerate.vcos(angles, cosValues);
accelerate.vtan(angles, tanValues);
console.timeEnd('Vectorized trig operations');

// Verify with known values
console.log('\n--- Verification at key angles ---');
const testIndices = [0, n/4, n/2, 3*n/4];
const testAngles = ['0', 'π/2', 'π', '3π/2'];

for (let i = 0; i < testIndices.length; i++) {
  const idx = Math.floor(testIndices[i]);
  console.log(`\nAngle: ${testAngles[i]}`);
  console.log(`  sin: ${sinValues[idx].toFixed(4)}`);
  console.log(`  cos: ${cosValues[idx].toFixed(4)}`);
}

// Compute sin²(x) + cos²(x) = 1 (identity check)
const sinSquared = new Float64Array(n);
const cosSquared = new Float64Array(n);
const identity = new Float64Array(n);

accelerate.vsquare(sinValues, sinSquared);
accelerate.vsquare(cosValues, cosSquared);
accelerate.vadd(sinSquared, cosSquared, identity);

console.log('\n--- Pythagorean Identity Check ---');
console.log('sin²(x) + cos²(x) should equal 1.0');
console.log('Mean value:', accelerate.mean(identity).toFixed(10));
console.log('Max deviation:', Math.abs(1.0 - accelerate.max(identity)).toFixed(10));
