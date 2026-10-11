"""Checks that the data match the published UCI cohort descriptions."""

import numpy as np
import pytest

from heartdisease.data import FEATURES, SITES, TARGET, load_all, load_site

# (n patients, n without disease) from the UCI dataset documentation.
EXPECTED = {
    "cleveland": (303, 164),
    "hungarian": (294, 188),
    "switzerland": (123, 8),
    "va": (200, 51),
}


@pytest.mark.parametrize("site", list(SITES))
def test_cohort_sizes_match_uci(site):
    df = load_site(site)
    n, n_healthy = EXPECTED[site]
    assert len(df) == n
    assert (df[TARGET] == 0).sum() == n_healthy


def test_target_is_binary_and_derived_from_num():
    """The original notebook's bug: `num` is 0-4 but was fed to a binary loss."""
    df = load_all()
    assert set(df["num"].unique()) <= {0, 1, 2, 3, 4}
    assert set(df[TARGET].unique()) == {0, 1}
    assert (df[TARGET] == (df["num"] > 0)).all()


def test_zero_cholesterol_is_treated_as_missing():
    df = load_all()
    assert (df["chol"] == 0).sum() == 0
    assert df.loc[df.site == "switzerland", "chol"].isna().all()


def test_cleveland_missingness_matches_original_notebook():
    df = load_site("cleveland")
    missing = df[FEATURES].isna().sum()
    assert missing["ca"] == 4
    assert missing["thal"] == 2
    assert missing.drop(["ca", "thal"]).sum() == 0


def test_category_codes_are_uci_codes():
    df = load_all()
    assert set(df["cp"].dropna().unique()) == {1, 2, 3, 4}
    assert set(df["thal"].dropna().unique()) == {3, 6, 7}
    assert set(df["slope"].dropna().unique()) == {1, 2, 3}
    assert set(df["restecg"].dropna().unique()) == {0, 1, 2}
    assert np.isin(df["sex"].unique(), [0, 1]).all()
