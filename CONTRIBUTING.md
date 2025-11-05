### Contributing

Thanks for contributing! This project follows a lightweight gitflow.

- Branch from `develop` using `feature/<short-name>`
- Run `make init` once, then use `pre-commit` to keep code clean
- Ensure `make lint` and `make test` pass before opening a PR
- Open PRs into `develop`; maintainers will squash-merge

Releases:

- When ready, maintainers fast-forward merge `develop` -> `main`
- Release automation: Release Please opens a PR; merging it creates a `vX.Y.Z` tag
- Publishing uses PyPI Trusted Publishers (OIDC), no PyPI token required

Trusted Publishers setup (one-time in PyPI):

1. In PyPI project settings, add a Trusted Publisher for GitHub Actions
2. Repository: this repo; Workflow: `.github/workflows/publish.yml`; Environment: leave blank
3. Docs: https://docs.pypi.org/trusted-publishers/

Secrets required (GitHub repo → Settings → Secrets and variables → Actions):

- `HF_TOKEN`: Hugging Face write token for the dataset workflow

Local development commands:

```bash
make init      # install dev deps and pre-commit
make format    # auto-fix formatting
make lint      # static checks
make test      # run tests
make build     # build sdist+wheel
```

