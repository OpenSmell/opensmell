import os
import sys
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import opensmell


def test_same_substance_different_sessions():
    csv1 = os.path.join(os.path.dirname(__file__), "cinnamon_1.csv")
    csv2 = os.path.join(os.path.dirname(__file__), "cinnamon_2.csv")

    r1 = opensmell.process(csv1)
    r2 = opensmell.process(csv2)

    sim = float(cosine_similarity(r1.latent.reshape(1, -1), r2.latent.reshape(1, -1))[0, 0])
    assert sim > 0.8, (
        f"Expected cosine similarity > 0.8 for same substance "
        f"different sessions, got {sim:.4f}"
    )
    assert r1.substance == r2.substance, (
        f"Expected same substance, got '{r1.substance}' vs '{r2.substance}'"
    )
    print(f"PASS: cinnamon session 1 vs 2 cosine sim = {sim:.4f} > 0.8")

if __name__ == "__main__":
    test_same_substance_different_sessions()
