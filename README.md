# CMR Linux-focused CI verification snapshot

This is a **public testing snapshot**, not the CMR product, not a release, and not a
clinical system. It exists solely to run a bounded Linux verification of a
byte-identical allowlisted dependency/test closure from the private CMR repository.

- Source commit: `5472ea706b31d5bc8d2c7df339933466dd40df51`
- Source tree: `b7945784f4a5413ec796477f0ca493aba0c87b2b`
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

## Explicit non-claims

Linux results apply **only** to this byte-identical exported closure. They do not
establish:

- full-suite, whole-repository, Windows, or desktop-native results;
- milestone acceptance, release qualification, clinical validation, or physician
  visual acceptance;
- correctness of omitted packages/services, BAAI/ML/GPU paths, or any real DICOM
  or patient data path.

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
