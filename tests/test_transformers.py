"""Tests for Quant_engine.transformers.
 
Map of the file, grouped by what each test checks. Update it when a
test is added, renamed or removed.
 
1. Market price
   - test_only_rows_with_valid_quote_are_kept: a row reaches the solver
     only with a two-sided quote (bid > 0, ask >= bid), because the
     market price is the bid/ask mid.
2. Moneyness
   - test_keep_otm_uses_forward: only the OTM option of each strike is
     kept, with moneyness measured against the forward, not the spot.
"""

import math

import pandas as pd
import pytest

from Quant_engine.instruments import EuropeanCall, EuropeanPut
from Quant_engine.transformers import YahooDataTransformer, keep_otm


# ---------------------------------------------------------------------
# Shared parameters
# ---------------------------------------------------------------------

# Fixed pricing date and expiry, so time to maturity does not depend on
# the day the tests are run.
PRICING_DATE = "2026-09-30"
EXPIRY = "2026-12-18"
SPOT = 100.0
RATE = 0.05

# ---------------------------------------------------------------------
# 1. Market price
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "bid, ask, expected_rows",
    [
        # Kept: the other side of the filter. Without these, a filter
        # that drops every row would pass the test.
        pytest.param(4.0, 6.2, 1, id="valid_quote"),
        pytest.param(5.0, 5.0, 1, id="locked_market"), # bid = ask case
        # Dropped: no two-sided quote, so no mid. The NaN rows catch a
        # filter written as "drop if bid <= 0", which keeps NaN.
        pytest.param(math.nan, 6.2, 0, id="bid_missing"),
        pytest.param(4.0, math.nan, 0, id="ask_missing"),
        pytest.param(0.0, 6.2, 0, id="bid_zero"),
        pytest.param(4.0, 0.0, 0, id="ask_zero"),
        pytest.param(6.2, 4.0, 0, id="crossed_market") # ask < bid
    ],
)
def test_only_rows_with_valid_quote_are_kept(bid, ask, expected_rows):
    """A contract is kept only if its quote gives a valid mid price.
 
    Does not check the mid value itself, a one-line average.
    """
    # volume > 0 on purpose: with volume 0 the row would be dropped by
    # the volume filter, and the dropped cases would pass for the
    # wrong reason.
    raw_chain = pd.DataFrame(
        {
            "strike": [100.0],
            "bid": [bid],
            "ask": [ask],
            "option_type": ["call"],
        }
    )
    transformer = YahooDataTransformer(PRICING_DATE)
 
    options = transformer.transform_to_objects(raw_chain, SPOT, EXPIRY)
 
    assert len(options) == expected_rows


# ---------------------------------------------------------------------
# 2. Moneyness
# ---------------------------------------------------------------------

# With SPOT = 100, RATE = 5% and T = 0.25 the forward is about 101.26.
MATURITY = 0.25


@pytest.mark.parametrize(
    "option_class, strike, is_kept",
    [
        pytest.param(EuropeanPut, 95.0, True, id="put_below_spot"),
        pytest.param(EuropeanCall, 95.0, False, id="call_below_spot"),
        # K = 101: above the spot, below the forward.
        pytest.param(EuropeanPut, 101.0, True, id="put_between"),
        pytest.param(EuropeanCall, 101.0, False, id="call_between"),
        pytest.param(EuropeanPut, 105.0, False, id="put_above_forward"),
        pytest.param(EuropeanCall, 105.0, True, id="call_above_forward"),
    ],
)
def test_keep_otm_uses_forward(option_class, strike, is_kept):
    """Only the option that is OTM relative to the forward is kept."""
    options = [{"instruments": option_class(SPOT, strike, MATURITY)}]

    kept = keep_otm(options, RATE)

    assert (len(kept) == 1) == is_kept
 