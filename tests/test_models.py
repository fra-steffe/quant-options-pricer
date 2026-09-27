"""Tests for Quant_engine.models."""

import math

import pytest

from Quant_engine.instruments import EuropeanCall, EuropeanPut
from Quant_engine.models import BlackScholesModel

#decorator that attaches to the function below a mark with the names and values of the arguments
#so that pytest creates a test for each set of values
@pytest.mark.parametrize(
    "spot, strike, maturity, rate, vol",
    [
        #tests different moneyness scenarios
        pytest.param(100.0, 100.0, 1.0, 0.05, 0.20, id="atm"),
        pytest.param(100.0, 80.0, 1.0, 0.05, 0.20, id="call_itm"),
        pytest.param(100.0, 120.0, 1.0, 0.05, 0.20, id="call_otm"),

        #tests different maturities
        pytest.param(100.0, 100.0, 0.05, 0.05, 0.20, id="short_maturity"),
        pytest.param(100.0, 100.0, 5.0, 0.05, 0.20, id="long_maturity"),

        #tests edge case of no discount
        pytest.param(100.0, 100.0, 1.0, 0.0, 0.20, id="zero_rate"),

        #tests extreme volatility
        pytest.param(100.0, 100.0, 1.0, 0.05, 0.80, id="high_vol"),
    ],
)
def test_put_call_parity(spot, strike, maturity, rate, vol):
    """C - P must equal S - K * exp(-rT).

    Catches errors in discounting and in the structure of the call and
    put formulas. Blind to errors inside d1 and d2, because
    N(x) + N(-x) = 1 for any x: those need a reference-value test.
    """
    model = BlackScholesModel(risk_free_rate=rate, volatility=vol)
    call = EuropeanCall(underlying=spot, strike=strike, maturity=maturity)
    put = EuropeanPut(underlying=spot, strike=strike, maturity=maturity)

    call_price = model.price(call)
    put_price = model.price(put)

    expected = spot - strike * math.exp(-rate * maturity)
    assert call_price - put_price == pytest.approx(expected) 
