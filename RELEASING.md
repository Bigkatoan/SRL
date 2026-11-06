# Releasing SRL

Releases are fully automated by [`.github/workflows/release.yml`](.github/workflows/release.yml)
and triggered by pushing a `vX.Y.Z` tag. The package version is derived from
that tag by [setuptools-scm](https://setuptools-scm.readthedocs.io/) — there is
no version string to bump in the source tree.

Pipeline: build sdist + wheel → `twine check --strict` → verify the built
version equals the tag → smoke-test the wheel → *(optional)* publish to
**TestPyPI** → **PyPI** → create a **GitHub Release** with the dist files and
the matching `CHANGELOG.md` section as notes.

> **Current status: PyPI publishing is disabled.** Releases are published as
> GitHub Releases only; users install from GitHub, e.g.
> `pip install "srl-rl @ git+https://github.com/Bigkatoan/SRL.git@v0.3.0"` or the
> attached wheel. To enable PyPI later, complete the one-time setup below and
> set the repository variable `PUBLISH_TO_PYPI=true`
> (*Settings → Secrets and variables → Actions → Variables*).

Publishing uses PyPI [trusted publishing](https://docs.pypi.org/trusted-publishers/)
(OpenID Connect): no API tokens are stored in the repository.

## One-time setup (repository owner)

These steps only need to be done once, before the first release.

1. **GitHub environments.** In *Settings → Environments* create two
   environments named exactly `testpypi` and `pypi`. For `pypi`, add
   yourself as a *required reviewer* so every production upload needs a
   manual approval, and restrict deployment to tags matching `v*`.
2. **PyPI pending publisher.** Since `srl-rl` does not exist on PyPI yet,
   register a *pending* trusted publisher (this also reserves the name):
   go to <https://pypi.org/manage/account/publishing/> → *Add a new pending
   publisher* → GitHub, and enter:
   - PyPI project name: `srl-rl`
   - Owner: `Bigkatoan`
   - Repository name: `SRL`
   - Workflow name: `release.yml`
   - Environment name: `pypi`
3. **TestPyPI pending publisher.** Repeat step 2 on
   <https://test.pypi.org/manage/account/publishing/> (separate account),
   with environment name `testpypi`.
4. **Tag protection (recommended).** In *Settings → Rules → Rulesets*, add a
   tag ruleset for `v*` that restricts creation to maintainers.

After the first successful upload the pending publishers become regular
trusted publishers of the `srl-rl` project.

## Cutting a release

1. Make sure `main` is green (Tests, Lint, Docs) and, ideally, that the
   nightly `learning` job passed.
2. In `CHANGELOG.md`, rename `## [Unreleased]` to `## [X.Y.Z] - YYYY-MM-DD`
   and add a fresh empty `## [Unreleased]` section above it. Merge that PR.
3. Tag the merge commit and push the tag:

   ```bash
   git switch main && git pull
   git tag -a vX.Y.Z -m "SRL vX.Y.Z"
   git push origin vX.Y.Z
   ```

4. Watch the *Release* workflow. Approve the `pypi` environment deployment
   when prompted, after checking the TestPyPI upload:

   ```bash
   pip install --index-url https://test.pypi.org/simple/ \
       --extra-index-url https://pypi.org/simple/ srl-rl==X.Y.Z
   ```

Pre-releases (`vX.Y.ZrcN`, `vX.Y.ZaN`, `vX.Y.ZbN`) follow the same path and are
marked as pre-releases on GitHub.

## Notes

- The repository currently has **no tags**, so untagged builds report a
  version like `0.0.1.devN+g<sha>`. Tagging the commit that corresponds to the
  existing 0.2.0 code (or simply releasing `v0.3.0`) fixes this.
- Building locally: `python -m build && twine check --strict dist/*`.
- A failed upload can be retried by re-running the failed job; TestPyPI
  uses `skip-existing`, PyPI never allows re-uploading the same version.
