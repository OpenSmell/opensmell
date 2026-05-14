import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import opensmell


def test_process_returns_correct_shapes():
    csv_path = os.path.join(os.path.dirname(__file__), "cinnamon_6.csv")
    result = opensmell.process(csv_path)
    assert result.chemoprint.shape == (29,), f"Expected (29,), got {result.chemoprint.shape}"
    assert result.latent.shape == (256,), f"Expected (256,), got {result.latent.shape}"
    assert isinstance(result.substance, str), f"Expected str, got {type(result.substance)}"
    assert result.confidence > 0, f"Expected confidence > 0, got {result.confidence}"
    assert result.confidence <= 1.0, f"Expected confidence <= 1, got {result.confidence}"
    print(f"PASS: process() returns correct shapes")
    print(f"  substance={result.substance}, confidence={result.confidence:.4f}")

if __name__ == "__main__":
    test_process_returns_correct_shapes()
