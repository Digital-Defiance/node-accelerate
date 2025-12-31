#!/usr/bin/env node
/**
 * Pre-install platform check
 * Validates that the system can run node-accelerate
 */

const os = require('os');
const fs = require('fs');

const platform = process.platform;
const arch = process.arch;

console.log('Checking platform compatibility for node-accelerate...');
console.log(`  Platform: ${platform}`);
console.log(`  Architecture: ${arch}`);
console.log(`  Node.js: ${process.version}`);

// Check platform
if (platform !== 'darwin') {
  console.error('');
  console.error('❌ ERROR: node-accelerate requires macOS');
  console.error('');
  
  if (platform === 'linux' && arch === 'arm64') {
    console.error('Detected: Linux ARM64 (e.g., Raspberry Pi, AWS Graviton)');
    console.error('');
    console.error('This package uses Apple\'s Accelerate framework, which is only');
    console.error('available on macOS. Linux ARM64 systems are not supported.');
    console.error('');
    console.error('For Linux ARM64, consider using:');
    console.error('  • OpenBLAS: https://www.openblas.net/');
    console.error('  • Eigen: https://eigen.tuxfamily.org/');
    console.error('  • BLIS: https://github.com/flame/blis');
  } else {
    console.error('This package uses Apple\'s Accelerate framework, which is only');
    console.error('available on macOS (darwin).');
    console.error('');
    console.error(`Your platform: ${platform}`);
    console.error('Supported platforms: darwin (macOS only)');
  }
  
  console.error('');
  process.exit(1);
}

// Check architecture
if (arch !== 'arm64' && arch !== 'x64') {
  console.error('');
  console.error('❌ ERROR: Unsupported architecture');
  console.error('');
  console.error(`Your architecture: ${arch}`);
  console.error('Supported architectures: arm64 (Apple Silicon), x64 (Intel)');
  console.error('');
  process.exit(1);
}

// Check for Accelerate framework
const acceleratePath = '/System/Library/Frameworks/Accelerate.framework';
if (!fs.existsSync(acceleratePath)) {
  console.error('');
  console.error('❌ ERROR: Apple Accelerate framework not found');
  console.error('');
  console.error(`Expected location: ${acceleratePath}`);
  console.error('');
  console.error('This is unusual for macOS. Please ensure you\'re running on a');
  console.error('standard macOS system with system frameworks intact.');
  console.error('');
  process.exit(1);
}

// Check Node.js version
const nodeVersion = process.versions.node;
const [major] = nodeVersion.split('.').map(Number);

if (major < 18) {
  console.error('');
  console.error('❌ ERROR: Node.js version too old');
  console.error('');
  console.error(`Your version: ${nodeVersion}`);
  console.error('Required: >= 18.0.0');
  console.error('');
  console.error('Please upgrade Node.js: https://nodejs.org/');
  console.error('');
  process.exit(1);
}

// Check for Xcode Command Line Tools
const xcodePath = '/Library/Developer/CommandLineTools';
if (!fs.existsSync(xcodePath)) {
  console.warn('');
  console.warn('⚠️  WARNING: Xcode Command Line Tools may not be installed');
  console.warn('');
  console.warn('node-accelerate requires Xcode Command Line Tools to build.');
  console.warn('');
  console.warn('If the build fails, install them with:');
  console.warn('  xcode-select --install');
  console.warn('');
}

console.log('');
console.log('✓ Platform check passed!');
console.log('');
