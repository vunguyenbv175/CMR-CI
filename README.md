# CMR Linux-focused CI verification snapshot

This is a **public testing snapshot**, not the CMR product, not a release, and not a
clinical system. It exists solely to run a bounded Linux verification of a
byte-identical allowlisted dependency/test closure from the private CMR repository.

- Source commit (merged main): `7877cca83e7a56c6349200b5406c7151b2830182`
- Source tree: `0be52a71ad2ca6875186501f33ae1eeeef22b93c`
- Export manifest: `PUBLIC_MANIFEST.json` (maps every exported source/test/static file
  to its upstream path and SHA-256)
- Manifest verifier: `ci/verify_manifest.py`
- Selected tests: see `PUBLIC_MANIFEST.json` `"selection"` and the workflow command

## Scope

The workflow runs only the selected workstation tests:

```text
tests/workstation/test_case_workspace_recent_api.py
tests/workstation/test_case_workspace_open.py
tests/workstation/test_case_workspace_switch.py
tests/workstation/test_case_workspace_switch_study.py
tests/workstation/test_server_api.py
tests/workstation/test_static_assets.py
tests/workstation/test_series_selection_api.py
tests/workstation/test_frame_pixels_endpoint.py
```

They exercise the recent-cases store/probe/stage/reopen path, source scan/open,
switch/switch-study behavior, server/static neighbors, series selection and frame
transport. The recent API file includes POSIX security witnesses (symlinked
store/sidecar, `dir_fd`, `O_DIRECTORY | O_NOFOLLOW`, ancestor swaps) that are
skipped on Windows and execute on `ubuntu-latest`.

## Known residual boundary (not fixed, not waived)

The corrected sidecar replacement witness forces a genuinely different identity:
it holds the original inode open and renames it outside the validated path before
creating the replacement. It does **not** cover same-inode reuse. Production
`_same_file` compares only `(st_dev, st_ino)`, so a same-path substitution that
reuses the inode can still be accepted. That is a residual local same-user
integrity limitation, not a remote/loopback bypass. This snapshot does not fix or
waive it; a stronger content/version binding is a separate bounded scope decision.

## Explicit non-claims

Linux results apply **only** to this byte-identical exported closure. They do not
establish:

- that the original source commit `5472ea706b31d5bc8d2c7df339933466dd40df51`
  was green, or that the full private Linux suite passed;
- full-suite, whole-repository, Windows, or desktop-native results;
- milestone acceptance, release qualification, clinical validation, or physician
  visual acceptance;
- correctness of omitted packages/services, BAAI/ML/GPU paths, or any real DICOM
  or patient data path;
- any security waiver: the same-inode residual above remains open.

The selected tests use synthetic generated DICOM only. Do not infer clinical
correctness from CI.

## License / provenance

The upstream repository has no LICENSE file; `pyproject.toml` marks the project
proprietary. **No open-source license is granted by this snapshot.** It is published
only so the authorized CI verification can run; all rights remain with the owner.
Third-party Python/Node dependencies are installed from their own package indexes at
CI runtime and are not vendored here. See `NOTICE.md`.

## Reproduce locally

```bash
python -m pip install -r requirements-ci.txt
python ci/verify_manifest.py
python -m pytest -q \
  tests/workstation/test_case_workspace_recent_api.py \
  tests/workstation/test_case_workspace_open.py \
  tests/workstation/test_case_workspace_switch.py \
  tests/workstation/test_case_workspace_switch_study.py \
  tests/workstation/test_server_api.py \
  tests/workstation/test_static_assets.py \
  tests/workstation/test_series_selection_api.py \
  tests/workstation/test_frame_pixels_endpoint.py \
  --junitxml=artifacts/junit.xml
```

`PYTHONPATH` is not required when running pytest from the repository root because
`tests/conftest.py` adds `packages/`, `services/` and `tests/`.

## Workflow security

`.github/workflows/linux-focused.yml` uses `ubuntu-latest`, `permissions:
contents: read`, checkout with `persist-credentials: false`, bounded
`timeout-minutes`, a concurrency group, no secrets, no `pull_request_target`, no
self-hosted runners, and no paid larger runners. Actions are pinned by commit SHA.
