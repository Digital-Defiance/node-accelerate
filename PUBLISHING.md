# Publishing Guide

This guide walks through publishing `node-accelerate` to npm.

## Pre-Publication Checklist

### 1. Code Quality
- [ ] All tests pass (`npm test`)
- [ ] Benchmarks run successfully (`npm run benchmark`)
- [ ] No compiler warnings
- [ ] Code follows style guidelines
- [ ] Documentation is complete and accurate

### 2. Package Configuration
- [ ] `package.json` version is correct
- [ ] `package.json` repository URL is correct
- [ ] `package.json` author information is correct
- [ ] `LICENSE` file is present
- [ ] `README.md` is complete
- [ ] `.npmignore` excludes unnecessary files

### 3. Testing
- [ ] Test on multiple Node.js versions (18, 20, 22)
- [ ] Test on Apple Silicon (M1/M2/M3/M4)
- [ ] Test on Intel Mac (if available)
- [ ] Verify TypeScript definitions work
- [ ] Test installation from tarball

### 4. Documentation
- [ ] README has installation instructions
- [ ] README has usage examples
- [ ] API documentation is complete
- [ ] CHANGELOG is updated
- [ ] GitHub repository URL is correct

## Publishing Steps

### 1. Prepare the Release

Update version in `package.json`:
```bash
npm version patch  # or minor, or major
```

This will:
- Update `package.json` version
- Create a git commit
- Create a git tag

### 2. Test the Package

Create a tarball and test it:
```bash
npm pack
```

This creates `node-accelerate-1.0.0.tgz`. Test it in another directory:
```bash
mkdir test-install
cd test-install
npm init -y
npm install ../node-accelerate/node-accelerate-1.0.0.tgz
node -e "const a = require('@digitaldefiance/node-accelerate'); console.log(a)"
```

### 3. Login to npm

```bash
npm login
```

Enter your npm credentials.

### 4. Publish to npm

For first release:
```bash
npm publish --access public
```

For subsequent releases:
```bash
npm publish
```

### 5. Verify Publication

Check the package page:
```
https://www.npmjs.com/package/node-accelerate
```

Test installation:
```bash
npm install node-accelerate
```

### 6. Create GitHub Release

1. Push tags to GitHub:
   ```bash
   git push origin main --tags
   ```

2. Go to GitHub releases page
3. Click "Create a new release"
4. Select the version tag
5. Add release notes from CHANGELOG
6. Publish release

## Post-Publication

### 1. Announce

- Tweet about the release
- Post on Reddit (r/node, r/programming)
- Share on LinkedIn
- Submit to Hacker News

### 2. Monitor

- Watch for issues on GitHub
- Monitor npm download stats
- Respond to questions

### 3. Update Documentation

- Update blog post with npm install instructions
- Update any external documentation

## Troubleshooting

### Build Fails on User's Machine

Common issues:
- Missing Xcode Command Line Tools
- Wrong Node.js version
- Not on macOS

Solution: Improve error messages in `binding.gyp`

### TypeScript Definitions Don't Work

- Verify `index.d.ts` is in `files` array in `package.json`
- Test with `tsc --noEmit` in a TypeScript project

### Package Size Too Large

Check what's included:
```bash
npm pack --dry-run
```

Update `.npmignore` to exclude unnecessary files.

## Version Strategy

Follow Semantic Versioning:

- **Patch** (1.0.x): Bug fixes, documentation updates
- **Minor** (1.x.0): New features, backward compatible
- **Major** (x.0.0): Breaking changes

## npm Scripts

```json
{
  "scripts": {
    "prepublishOnly": "npm test",
    "version": "npm run build && git add -A",
    "postversion": "git push && git push --tags"
  }
}
```

These ensure:
- Tests run before publishing
- Build is up to date
- Tags are pushed to GitHub

## Security

### Audit Dependencies

```bash
npm audit
```

Fix any vulnerabilities before publishing.

### Two-Factor Authentication

Enable 2FA on your npm account:
```bash
npm profile enable-2fa auth-and-writes
```

## Maintenance

### Regular Updates

- Update dependencies quarterly
- Test with new Node.js versions
- Update documentation as needed
- Respond to issues promptly

### Deprecation

If you need to deprecate a version:
```bash
npm deprecate node-accelerate@1.0.0 "Use version 1.1.0 or higher"
```

### Unpublishing

Only possible within 72 hours:
```bash
npm unpublish node-accelerate@1.0.0
```

After 72 hours, you can only deprecate.

## Support

- GitHub Issues: Bug reports and feature requests
- npm: Package distribution
- Documentation: Keep README and examples up to date

---

**Ready to publish?** Follow the checklist above and you're good to go!
