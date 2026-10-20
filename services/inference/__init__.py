"""inference — SegmentationProvider implementations (dummy/manual) + registry.

BAAI/nnU-Net/remote providers plug in here later without touching core.
"""

from .providers import (
    PROVIDERS,
    DummySegmentationProvider,
    ManualSegmentationProvider,
    get_provider,
    list_providers,
)

__all__ = [
    "DummySegmentationProvider",
    "ManualSegmentationProvider",
    "PROVIDERS",
    "get_provider",
    "list_providers",
]
