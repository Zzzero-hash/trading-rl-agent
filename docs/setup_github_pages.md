# Setting Up GitHub Pages for Documentation

This guide will help you set up GitHub Pages to automatically host your Sphinx documentation.

## 1. Enable GitHub Pages

1. Go to your repository on GitHub
2. Click on **Settings** tab
3. Scroll down to **Pages** section in the left sidebar
4. Under **Source**, select **GitHub Actions**

## 2. Configure Repository Secrets (Optional)

If you want to use Netlify for preview deployments, add these secrets:

1. Go to **Settings** → **Secrets and variables** → **Actions**
2. Add the following secrets:
   - `NETLIFY_AUTH_TOKEN`: Your Netlify auth token
   - `NETLIFY_SITE_ID`: Your Netlify site ID

## 3. Test the Documentation Workflow

The documentation workflow will automatically run when you:

- Push to `main` or `develop` branches
- Create pull requests
- Modify files in `src/`, `docs/`, or `requirements*.txt`

## 4. Manual Trigger

You can manually trigger the documentation build:

1. Go to **Actions** tab
2. Select **Documentation Build and Deploy**
3. Click **Run workflow**
4. Choose branch and options

## 5. View Your Documentation

Once deployed, your documentation will be available at:

- **GitHub Pages**: `https://[username].github.io/[repository-name]/`
- **Netlify Preview** (for PRs): Automatically generated link in PR comments

## 6. Custom Domain (Optional)

To use a custom domain:

1. Go to **Settings** → **Pages**
2. Under **Custom domain**, enter your domain
3. Add a CNAME record pointing to `[username].github.io`
4. Check **Enforce HTTPS**

## 7. Troubleshooting

### Build Failures

- Check the Actions tab for detailed error logs
- Ensure all dependencies are in `requirements-dev.txt`
- Verify Sphinx configuration in `docs/conf.py`

### Missing Documentation

- Run `python docs/build_docs.py --check` locally to identify issues
- Check that all Python modules have proper docstrings
- Verify that `.rst` files are generated for all modules

### Deployment Issues

- Ensure GitHub Pages is enabled
- Check that the workflow has proper permissions
- Verify that the `gh-pages` branch is created

## 8. Local Development

For local development:

```bash
# Build documentation
cd docs
python build_docs.py

# Serve locally
python build_docs.py --serve

# Check for issues
python build_docs.py --check
```

## 9. Documentation Structure

Your documentation is organized into:

- **Getting Started**: Quick start guides
- **Features**: Detailed feature documentation
- **Development**: Setup and development guides
- **Deployment**: Production deployment guides
- **API Reference**: Auto-generated from code
- **Examples**: Code examples and use cases

## 10. Adding New Documentation

1. Create new `.md` files in the appropriate directory
2. Add them to the `toctree` in `docs/index.rst`
3. Commit and push - documentation will be built automatically
4. Check the Actions tab for build status

## 11. Best Practices

- Keep documentation close to code
- Use clear, concise language
- Include examples where helpful
- Update documentation with code changes
- Use proper docstrings in Python code
- Test documentation builds locally before pushing
