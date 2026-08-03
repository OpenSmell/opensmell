"""MOX thermodynamic feasibility chain (Smellability).

Placeholder for the Python port of `osmograph-web/lib/smellability/` — the
4-step chain (identity -> volatility -> headspace concentration -> MOX redox
check). Lives under `opensmell.mox` because it encodes MOX-specific physics
(MOX response floor, redox activity at 300C), not generic chemistry.

Populated in M3 of the opensmell port plan; modules to land here:
chain, transport, constants, types, compounds, composites, ontology, groups,
enrichment (PubChem), provisional, search, user_dictionary, index.
"""

from __future__ import annotations
