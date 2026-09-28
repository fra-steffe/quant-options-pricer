"""Tests for Quant_engine.models."""

import math

import pytest

from Quant_engine.instruments import EuropeanCall, EuropeanPut
from Quant_engine.models import BlackScholesModel

#Put-call parity test

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

        # tests edge case of no discount
        pytest.param(100.0, 100.0, 1.0, 0.0, 0.20, id="zero_rate"),

        # tests extreme volatility
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


# reference test, against the values from: 'Hull, Options, Futures, and Other Derivatives'
# 8th ed. (Global Edition), Example 14.6, pp. 315-316
@pytest.mark.parametrize(
    "option_class, spot, strike, maturity, rate, vol, expected",
    [
        pytest.param(
            EuropeanCall, 42.0, 40.0, 0.5, 0.10, 0.20, 4.76, id="hull_call"
        ),
        pytest.param(
            EuropeanPut, 42.0, 40.0, 0.5, 0.10, 0.20, 0.81, id="hull_put"
        ),
    ],
)
def test_price_matches_reference(
    option_class, spot, strike, maturity, rate, vol, expected
    ):
    """Prices must match published textbook values.

    Complements the parity test: catches errors inside d1 and d2.
    The reference is rounded, so the tolerance is half a unit in its
    last digit (2 decimals in Hull -> abs=0.005).
    """
    model = BlackScholesModel(risk_free_rate=rate, volatility=vol)
    option = option_class(underlying=spot, strike=strike, maturity=maturity)

    price = model.price(option)

    assert price == pytest.approx(expected, abs=0.005)


# test the condition if T <= BlackScholesModel.TIME_EPSILON: return option.payoff(S)
@pytest.mark.parametrize(
    "maturity",
    [
        pytest.param(0.0, id = "t_zero"),
        #boundary condition check: verifies the <= check
        pytest.param(BlackScholesModel.TIME_EPSILON, id = "t_epsilon")
    ],
)
@pytest.mark.parametrize(
    "option_class, spot, strike, expected",
    [
        pytest.param(EuropeanCall, 110.0, 100.0, 10.0, id="call_itm"),
        pytest.param(EuropeanCall, 100.0, 100.0, 0.0, id="call_atm"),
        pytest.param(EuropeanCall, 90.0, 100.0, 0.0, id="call_otm"),
        pytest.param(EuropeanPut, 90.0, 100.0, 10.0, id="put_itm"),
        pytest.param(EuropeanPut, 100.0, 100.0, 0.0, id="put_atm"),
        pytest.param(EuropeanPut, 110.0, 100.0, 0.0, id="put_otm"),
    ],
)
def test_price_at_expiry_equals_payoff(
    option_class, spot, strike, expected, maturity
):
    """At or below TIME_EPSILON, price() must return the intrinsic payoff."""
    model = BlackScholesModel(risk_free_rate=0.05, volatility=0.20) # rate and vol are constants since they don't influence payoff 
    option = option_class(underlying=spot, strike=strike, maturity=maturity)

    price = model.price(option)

    # == is used instead of approx: this branch computes only max(S - K, 0)
    # on round numbers, so there is no rounding error to tolerate.
    assert price == expected
