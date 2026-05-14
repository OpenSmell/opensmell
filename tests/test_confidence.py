import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import opensmell


def test_known_substance_high_confidence():
    csv_path = os.path.join(os.path.dirname(__file__), "cinnamon_6.csv")
    result = opensmell.process(csv_path)
    assert result.confidence > 0.5, (
        f"Expected confidence > 0.5 for known substance, got {result.confidence:.4f}"
    )
    assert result.warning is None, f"Expected no warning, got {result.warning}"
    assert result.should_contribute is False
    print(f"PASS: known substance confidence={result.confidence:.4f} > 0.5")


def test_random_noise_not_perfect():
    rng = np.random.RandomState(42)
    noise = rng.randn(100, 6).astype(np.float32)
    result = opensmell.process_array(noise)
    assert result.confidence < 0.99, (
        f"Expected confidence < 0.99 for random noise, got {result.confidence:.4f}"
    )
    print(f"PASS: random noise confidence={result.confidence:.4f} < 0.99 (expected; 256D space has volume)")


def test_extreme_ood_triggers_warning():
    rng = np.random.RandomState(42)
    extreme = rng.randn(100, 6).astype(np.float32) * 100
    result = opensmell.process_array(extreme)
    assert result.confidence < 0.7, (
        f"Expected confidence < 0.7 for extreme OOD signal, got {result.confidence:.4f}"
    )
    assert result.warning is not None, "Expected warning for OOD signal"
    assert result.should_contribute is True
    print(f"PASS: extreme OOD signal confidence={result.confidence:.4f} < 0.7")
    print(f"PASS: warning triggered")

if __name__ == "__main__":
    test_known_substance_high_confidence()
    test_random_noise_not_perfect()
    test_extreme_ood_triggers_warning()
