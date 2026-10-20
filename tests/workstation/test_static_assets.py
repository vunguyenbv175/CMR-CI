"""T4 static asset, CSP, and cache-contract tests."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from tests.workstation.support import make_seeded_study, request, running_server

ASSETS = {
    "index.html": "text/html; charset=utf-8",
    "app.js": "text/javascript; charset=utf-8",
    "style.css": "text/css; charset=utf-8",
    "kbmap.js": "text/javascript; charset=utf-8",
}


def test_static_source_is_local_and_csp_safe() -> None:
    root = Path(__file__).parents[2] / "services" / "workstation" / "static"
    source = "\n".join(path.read_text(encoding="utf-8") for path in root.iterdir())
    assert re.search(r"https?://", source, flags=re.IGNORECASE) is None
    assert re.search(r"\son[a-z]+\s*=", source, flags=re.IGNORECASE) is None
    assert "eval(" not in source
    assert "new Function(" not in source
    assert "document.write(" not in source


def test_static_assets_have_explicit_mime_etag_and_immutable_cache(tmp_path: Path) -> None:
    study = make_seeded_study(tmp_path, seed_history=False)
    with running_server(study.app) as (server, base_url):
        root = request(server, base_url, "/")
        assert root.status == 200
        assert root.headers["Content-Security-Policy"] == "default-src 'self'; img-src 'self' data:"

        for name, content_type in ASSETS.items():
            response = request(server, base_url, f"/static/{name}")
            assert response.status == 200
            assert response.headers["Content-Type"] == content_type
            assert response.headers["Cache-Control"] == "public, max-age=31536000, immutable"
            digest = hashlib.sha256(response.body).hexdigest()[:12]
            assert response.headers["ETag"] == f'"{digest}"'
            assert int(response.headers["Content-Length"]) == len(response.body)

        traversal = request(server, base_url, "/static/%2e%2e/RUNBOOK.md")
        assert traversal.status in {400, 404}


def test_static_assets_carry_true_window_level_controls_without_css_filters() -> None:
    """S3-008B §3.3/§5.7: true W/L controls replace the CSS pseudo-W/L.

    The S3-008 brightness/contrast sliders were a local canvas CSS filter and
    were never medical window/level; the frozen spec mandates their removal.
    """
    root = Path(__file__).parents[2] / "services" / "workstation" / "static"
    html = (root / "index.html").read_text(encoding="utf-8")
    app_js = (root / "app.js").read_text(encoding="utf-8")
    css = (root / "style.css").read_text(encoding="utf-8")

    # True W/L controls, labelled exactly as the spec mandates.
    assert "Window / Level (display only)" in html
    for element_id in ("wl-center", "wl-width", "wl-apply", "wl-reset"):
        assert f'id="{element_id}"' in html, element_id
    # Visibility controls.
    for element_id in ("visibility-grid", "visibility-reset"):
        assert f'id="{element_id}"' in html, element_id

    # NO CSS pseudo-W/L remnant anywhere.
    for forbidden in ("brightness", "contrast"):
        assert forbidden not in html, forbidden
        assert forbidden not in css, forbidden
        assert forbidden not in app_js, forbidden
    assert "setDisplayFilter" not in app_js
    assert "style.filter" not in app_js
    assert re.search(r"\bfilter\s*:", css) is None

    # Visibility checkboxes are generated in the FROZEN Structure order.
    assert 'const STRUCTURE_ORDER = ["lv_cavity", "lv_myocardium", "rv_cavity"];' in app_js
    # Still no external dependency and no inline handler.
    assert re.search(r"https?://", html, flags=re.IGNORECASE) is None
    assert re.search(r"\son[a-z]+\s*=", html, flags=re.IGNORECASE) is None
