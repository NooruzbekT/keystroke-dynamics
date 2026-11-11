import numpy as np
import pandas as pd
import pytest

from keystroke.data import FEATURE_COLUMNS


def make_cmu_like(n_subjects: int = 51, seed: int = 0) -> pd.DataFrame:
    """Synthetic table with the CMU layout; subject ids are non-contiguous like the real ones."""
    rng = np.random.default_rng(seed)
    ids = [f"s{i:03d}" for i in range(2, 2 + n_subjects + 3) if i not in (6, 9, 23)][:n_subjects]
    meta = pd.DataFrame(
        [(s, sess, rep) for s in ids for sess in range(1, 9) for rep in range(1, 51)],
        columns=["subject", "sessionIndex", "rep"],
    )
    feats = pd.DataFrame(rng.gamma(2.0, 0.05, size=(len(meta), len(FEATURE_COLUMNS))), columns=FEATURE_COLUMNS)
    df = pd.concat([meta, feats], axis=1)
    # Shuffle rows so tests do not depend on file order.
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


@pytest.fixture(scope="session")
def cmu_like() -> pd.DataFrame:
    return make_cmu_like()
