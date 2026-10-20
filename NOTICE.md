# NOTICE

## Upstream status

The private upstream CMR repository at source commit `7877cca83e7a56c6349200b5406c7151b2830182` contains no
LICENSE or COPYRIGHT file. Its `pyproject.toml` declares the project proprietary.
This snapshot is published only for the authorized Linux CI verification and grants
no license, express or implied, to use, copy, modify, distribute, or commercialize
the first-party code. All rights remain with the owner.

## Snapshot provenance

This snapshot is exported from merged main `7877cca83e7a56c6349200b5406c7151b2830182`,
not a re-run of the original `5472ea706b31d5bc8d2c7df339933466dd40df51` tree. The
original public run at `5472ea7` failed four Linux tests; the green claim, if the
workflow passes, applies only to this corrected exported closure.

## Third-party material

No third-party source code, model weights, datasets, or clinical data are included.
The CI installs `numpy`, `pydicom`, `matplotlib`, `pillow`, `pytest`, Node.js and
GitHub Actions from their own distribution channels under their own terms. This
repository does not redistribute those packages.

## Known residual boundary

The corrected test tree forces a genuinely different sidecar identity for the
replacement witness. It does not cover same-inode reuse: production identity checks
use `(st_dev, st_ino)` only, and a same-path substitution can reuse the inode. This
is a documented local same-user integrity limitation, not fixed or waived by this
snapshot.

## Clinical boundary

This is not a medical device, clinical product, or release. It must not be used for
diagnosis, treatment, or patient care. The tests are synthetic and machine-level;
they are not clinical validation or physician signoff.
