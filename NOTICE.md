# NOTICE

## Upstream status

The private upstream CMR repository at source commit `e44bbe01f259a2698e584d56ff3f4da3dbd38e43` contains no
LICENSE or COPYRIGHT file. Its `pyproject.toml` declares the project proprietary.
This snapshot is published only for the authorized Linux CI verification and grants
no license, express or implied, to use, copy, modify, distribute, or commercialize
the first-party code. All rights remain with the owner.

## Snapshot provenance

This snapshot is exported from merged main `e44bbe01f259a2698e584d56ff3f4da3dbd38e43`, not a re-run of the
original `5472ea706b31d5bc8d2c7df339933466dd40df51` tree. That original public
run failed four Linux tests. The merged tests-only portability correction
(`7877cca83e7a56c6349200b5406c7151b2830182`) and the merged sidecar same-inode narrowing
(`e44bbe01f259a2698e584d56ff3f4da3dbd38e43`) are both included here. A green result, if the workflow
passes, applies only to this corrected exported closure.

## Third-party material

No third-party source code, model weights, datasets, or clinical data are included.
The CI installs `numpy`, `pydicom`, `matplotlib`, `pillow`, `pytest`, Node.js and
GitHub Actions from their own distribution channels under their own terms. This
repository does not redistribute those packages.

## Known residual boundary

The merged source compares dev/inode/size (and `st_ctime_ns` on POSIX) when it
compares two file observations, and it re-stats the open sidecar after the
bounded read. A real same-inode unlink/recreate at the validated path is refused,
and a mutation during the read is an unusable sidecar. This remains a narrowing
witness, not an unforgeable identity proof: a replacement that reproduces size
and ctime, privileged ctime manipulation, and a mutation after the final
observation are outside the guarantee. On Windows `st_ctime` is not compared
because it is not a stable lstat/fstat observation there. The residual is not
waived.

## Clinical boundary

This is not a medical device, clinical product, or release. It must not be used for
diagnosis, treatment, or patient care. The tests are synthetic and machine-level;
they are not clinical validation or physician signoff.
