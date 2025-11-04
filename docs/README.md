# PencilFlows.jl Documentation

This directory contains the documentation source files for PencilFlows.jl.

## Building Documentation Locally

### Prerequisites

```julia
using Pkg
Pkg.activate("docs")
Pkg.instantiate()
```

### Build

```bash
julia --project=docs docs/make.jl
```

The generated documentation will be in `docs/build/`.

### Preview

Open `docs/build/index.html` in your web browser.

## Documentation Structure

```
docs/
├── make.jl              # Documentation build script
├── Project.toml         # Documentation dependencies
└── src/                 # Documentation source files
    ├── index.md         # Home page
    ├── installation.md  # Installation guide
    ├── quickstart.md    # Quick start tutorial
    ├── guide/           # User guides
    │   ├── concepts.md
    │   ├── boundary_conditions.md
    │   ├── mpi.md
    │   ├── poisson.md
    │   └── timestepping.md
    ├── examples/        # Example tutorials
    │   ├── basic_flow.md
    │   ├── parallel.md
    │   └── custom_bc.md
    └── api/             # API reference
        ├── core.md
        ├── solvers.md
        ├── io.md
        ├── physics.md
        └── utilities.md
```

## Deploying to GitHub Pages

Documentation is automatically deployed to `https://subhk.github.io/PencilFlows.jl` when you push to the `main` branch.

### Setup (One-time)

1. Generate SSH key for Documenter:
   ```bash
   julia -e 'using DocumenterTools; DocumenterTools.genkeys(user="subhk", repo="PencilFlows.jl")'
   ```

2. Add the public key to GitHub repository:
   - Go to repository Settings → Deploy Keys
   - Add the public key
   - Enable "Allow write access"

3. Add the private key as a secret:
   - Go to repository Settings → Secrets and variables → Actions
   - Create new secret named `DOCUMENTER_KEY`
   - Paste the private key

### Manual Deployment

If you need to deploy manually:

```bash
julia --project=docs -e '
    using Pkg;
    Pkg.develop(PackageSpec(path=pwd()));
    Pkg.instantiate();
    include("docs/make.jl")'
```

## Contributing to Documentation

### Adding New Pages

1. Create a new `.md` file in the appropriate directory
2. Add it to `docs/make.jl` in the `pages` array
3. Build and check the output

### Style Guide

- Use clear, simple language (assume no prior CFD knowledge)
- Include code examples with explanations
- Add visual diagrams when helpful (ASCII art is fine!)
- Link to related sections using `[@ref]` syntax

### Example Template

```markdown
# Page Title

Brief introduction explaining what this page covers.

## Section 1

Explanation with simple analogy.

\```julia
# Code example
code_here()
\```

**What this does:**
Step-by-step explanation.

## Common Issues

### Problem 1
**Symptom:** What user sees
**Solution:** How to fix
```

## Testing Documentation

Before committing:

1. Build locally and check for warnings
2. Verify all links work
3. Test code examples
4. Check rendering in browser

## Questions?

File an issue on GitHub or contact the maintainers.
