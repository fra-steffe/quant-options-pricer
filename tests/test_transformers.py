"""Tests for Quant_engine.transformers.
 
Map of the file, grouped by what each test checks. Update it when a
test is added, renamed or removed.
 
1. Market price
   - test_only_rows_with_valid_quote_are_kept: a row reaches the solver
     only with a two-sided quote (bid > 0, ask >= bid), because the
     market price is the bid/ask mid.
"""

import math

import pandas as pd
import pytest

from    Quant_engine.transformers import YahooDataTransformer


# ---------------------------------------------------------------------
# Shared parameters
# ---------------------------------------------------------------------

# Fixed pricing date and expiry, so time to maturity does not depend on
# the day the tests are run.
PRICING_DATE = "2026-09-30"
EXPIRY = "2026-12-18"
SPOT = 100.0

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
            "volume": [10.0],
            "option_type": ["call"],
        }
    )
    transformer = YahooDataTransformer(PRICING_DATE)
 
    options = transformer.transform_to_objects(raw_chain, SPOT, EXPIRY)
 
    assert len(options) == expected_rows
 