# Contributing to node-accelerate

Thank you for your interest in contributing! This document provides guidelines for contributing to node-accelerate.

## Development Setup

### Prerequisites

- macOS (Apple Silicon or Intel)
- Node.js >= 18.0.0
- Xcode Command Line Tools
- Git

### Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/Digital-Defiance/node-accelerate.git
   cd node-accelerate
   ```

3. Install dependencies:
   ```bash
   npm install
   ```

4. Build the addon:
   ```bash
   npm run build
   ```

5. Run tests:
   ```bash
   npm test
   ```

## Project Structure

```
node-accelerate/
├── accelerate.cc      # C++ implementation
├── binding.gyp        # Build configuration
├── index.js           # JavaScript wrapper
├── index.d.ts         # TypeScript definitions
├── test.js            # Test suite
├── benchmark.js       # Performance benchmarks
└── README.md          # Documentation
```

## Making Changes

### Code Style

**C++ Code:**
- Follow Google C++ Style Guide
- Use 2-space indentation
- Add comments for complex operations
- Include error handling

**JavaScript Code:**
- Use 2-space indentation
- Use `const` and `let`, not `var`
- Add JSDoc comments for functions
- Follow Node.js best practices

### Adding New Functions

1. **Add C++ implementation** in `accelerate.cc`:
   ```cpp
   Napi::Value YourFunction(const Napi::CallbackInfo& info) {
     Napi::Env env = info.Env();
     
     // Validate arguments
     if (info.Length() < 1) {
       Napi::TypeError::New(env, "Expected 1 argument").ThrowAsJavaScriptException();
       return env.Null();
     }
     
     // Your implementation using Accelerate framework
     
     return result;
   }
   ```

2. **Export the function** in `Init()`:
   ```cpp
   exports.Set("yourFunction", Napi::Function::New(env, YourFunction));
   ```

3. **Add TypeScript definition** in `index.d.ts`:
   ```typescript
   export function yourFunction(arg: Float64Array): number;
   ```

4. **Add tests** in `test.js`:
   ```javascript
   console.log('Testing your function...');
   const result = accelerate.yourFunction(testData);
   assertClose(result, expectedValue, 1e-10, 'Your function test');
   ```

5. **Update documentation** in `README.md`

### Testing

Run the test suite:
```bash
npm test
```

Add tests for:
- Correct results with known inputs
- Edge cases (empty arrays, single elements)
- Large inputs (performance validation)
- Error handling (invalid arguments)

### Benchmarking

Run benchmarks:
```bash
npm run benchmark
```

When adding new functions, include benchmarks comparing:
- Pure JavaScript implementation
- Accelerate-based implementation
- Speedup factor

## Pull Request Process

1. **Create a branch** for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following the code style guidelines

3. **Add tests** for new functionality

4. **Run tests** to ensure everything works:
   ```bash
   npm test
   npm run benchmark
   ```

5. **Commit your changes** with clear messages:
   ```bash
   git commit -m "Add feature: description"
   ```

6. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a Pull Request** with:
   - Clear description of changes
   - Test results
   - Benchmark results (if applicable)
   - Documentation updates

## What to Contribute

### High Priority

- **More BLAS operations**: Matrix-vector multiply, triangular solve, etc.
- **More vDSP operations**: Convolution, correlation, windowing
- **Float32 support**: Add single-precision variants
- **Error handling**: Improve validation and error messages
- **Documentation**: More examples and use cases

### Medium Priority

- **Performance optimizations**: Reduce overhead, optimize memory usage
- **Additional tests**: Edge cases, stress tests
- **CI/CD**: GitHub Actions for automated testing
- **Benchmarks**: More comprehensive performance tests

### Low Priority

- **Additional platforms**: Explore other ARM64 platforms
- **Advanced features**: Sparse matrices, complex numbers
- **Utilities**: Helper functions for common patterns

## Code Review

All submissions require review. We'll look for:

- **Correctness**: Does it work as intended?
- **Performance**: Does it maintain or improve performance?
- **Tests**: Are there adequate tests?
- **Documentation**: Is it well-documented?
- **Style**: Does it follow the style guide?

## Reporting Issues

When reporting issues, include:

1. **Environment**:
   - macOS version
   - Node.js version
   - Chip (M1/M2/M3/M4/Intel)

2. **Description**: Clear description of the issue

3. **Reproduction**: Minimal code to reproduce

4. **Expected vs Actual**: What you expected vs what happened

5. **Logs**: Any error messages or logs

## Questions?

- Open an issue for questions
- Check existing issues first
- Be respectful and constructive

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Acknowledgments

Thank you for contributing to node-accelerate! Your efforts help make high-performance numerical computing accessible to the Node.js community.
