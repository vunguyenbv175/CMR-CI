# NOTICE

## Upstream status

The private upstream CMR repository at source commit `5472ea706b31d5bc8d2c7df339933466dd40df51` contains no
LICENSE or COPYRIGHT file. Its `pyproject.toml` declares the project proprietary.
This snapshot is published only for the authorized Linux CI verification and grants
no license, express or implied, to use, copy, modify, distribute, or commercialize
the first-party code. All rights remain with the owner.

## Third-party material

No third-party source code, model weights, datasets, or clinical data are included.
The CI installs `numpy`, `pydicom`, `matplotlib`, `pillow`, `pytest`, Node.js and
GitHub Actions from their own distribution channels under their own terms. This
repository does not redistribute those packages.

## Clinical boundary

This is not a medical device, clinical product, or release. It must not be used for
diagnosis, treatment, or patient care. The tests are synthetic and machine-level;
they are not clinical validation or physician signoff.
