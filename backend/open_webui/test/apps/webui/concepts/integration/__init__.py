"""Integration tests for concept-graph production wiring (Phase 2)."""

import os

# docker exec does not load start.sh's secret-key file; config import requires this.
if not os.environ.get('WEBUI_SECRET_KEY'):
    os.environ['WEBUI_SECRET_KEY'] = 'pytest-concept-graph-integration'
