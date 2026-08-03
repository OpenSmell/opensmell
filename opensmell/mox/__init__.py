"""MOX sensor family implementation of the opensmell framework.

Everything that assumes metal-oxide (SnO2) sensors lives under
`opensmell.mox`: R0 normalization, kinetic feature extraction, the MOX quality
scorer, and the MOX thermodynamic feasibility chain (`opensmell.mox.smellability`).
"""

from . import features, normalize, quality, smellability  # noqa: F401
