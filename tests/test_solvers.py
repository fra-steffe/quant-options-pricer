"""Tests for Quant_engine.solvers.

Map of the file, grouped by what each test checks. Update it when a
test is added, renamed or removed.

1. The solver recovers sigma (well-posed cases)
   - test_newton_raphson_round_trip: price with a known sigma, then the
     solver must recover it, across moneyness, maturity and volatility.
   - test_newton_raphson_forward_atm: the same at the forward ATM point,
     where the Manaster-Koehler initial guess is exactly zero.

2. The solver fails explicitly (sigma not identifiable)
   - test_newton_raphson_returns_none_at_expiry: at or below
     BlackScholesModel.TIME_EPSILON the price does not depend on sigma,
     so the solver must return None.
   - test_newton_raphson_returns_none_when_ill_posed: when vega is so
     small that the price pins sigma only within a band wider than the
     solver's vol_tol, the solver must return None instead of a sigma
     picked from inside the band.

3. The initial guess
   - test_manaster_koehler_guess_maximizes_vega: the Manaster-Koehler
     guess is the sigma at which vega is maximal, the property that
     makes it a safe starting point for Newton.
"""

import pytest

from Quant_engine.instruments import EuropeanCall, EuropeanPut
from Quant_engine.models import BlackScholesModel
from Quant_engine.solvers import manaster_koehler_guess, newton_raphson

# ---------------------------------------------------------------------
# Shared parameters
# ---------------------------------------------------------------------

SPOT = 100.0
RATE = 0.05

# Tolerance on the recovered volatility. The solver stops on a price
# tolerance (1e-4), which translates into a volatility error of about
# 1e-4 / vega. Every case below has vega above 0.3 (the lowest,
# call_low_vega, on purpose), so the expected volatility error is at
# most ~3e-4: 1e-3 still leaves a margin.
VOL_TOLERANCE = 1e-3

# Same value as the solver's default price tolerance (tol=1e-4).
PRICE_TOLERANCE = 1e-4

# ---------------------------------------------------------------------
# 1. The solver recovers sigma
# ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "option_class, strike, maturity, true_vol",
    [
        # Regime 1: well-posed problem, the solver must recover sigma.
        pytest.param(EuropeanCall, 100.0, 0.02, 0.20, id="call_atm_short"),
        pytest.param(EuropeanCall, 100.0, 0.25, 0.20, id="call_atm"),
        pytest.param(EuropeanCall, 90.0, 0.25, 0.20, id="call_itm"),
        pytest.param(EuropeanCall, 110.0, 0.25, 0.20, id="call_otm"),
        pytest.param(EuropeanCall, 70.0, 1.0, 0.20, id="call_deep_itm_long"),
        pytest.param(EuropeanCall, 160.0, 1.0, 0.50, id="call_deep_otm_hivol"),
        pytest.param(EuropeanPut, 100.0, 0.25, 0.20, id="put_atm"),
        pytest.param(EuropeanPut, 110.0, 0.25, 0.20, id="put_itm"),
        # Low but sufficient vega (0.35): sigma is pinned within
        # 1e-4 / 0.35 = 3e-4, below the solver's vol_tol (1e-3). Guards
        # against a check for ill-posed cases that rejects too much.
        pytest.param(EuropeanCall, 85.0, 0.1, 0.20, id="call_low_vega"),
        # Regime 2: well-posed but far OTM.
        pytest.param(EuropeanCall, 130.0, 0.25, 0.20, id="call_otm_far"),
        pytest.param(EuropeanPut, 90.0, 0.25, 0.20, id="put_otm_far"),
    ],
)
def test_newton_raphson_round_trip(option_class, strike, maturity, true_vol):
    """Price with a known sigma, then the solver must recover it.

    Only covers cases where sigma is identifiable from the price.
    Ill-posed cases (price insensitive to sigma) need a separate test
    on how the solver reports failure.
    """
    option = option_class(underlying=SPOT, strike=strike, maturity=maturity)

    # Two separate model instances: newton_raphson overwrites model.sigma
    # (side effect), so the model that generates the price must not be
    # the one handed to the solver.
    pricing_model = BlackScholesModel(risk_free_rate=RATE, volatility=true_vol)
    market_price = pricing_model.price(option)

    # The starting volatility is a placeholder: the solver replaces it
    # with its own initial guess (see initial_guess) before iterating.
    solver_model = BlackScholesModel(risk_free_rate=RATE, volatility=0.0)
    implied_vol = newton_raphson(solver_model, option, market_price)

    # Checked first and on its own, so that a solver failure reads as
    # "returned None" instead of a confusing TypeError in the next line.
    assert implied_vol is not None

    # TEST1: the solver recovers the volatility used to generate the price.
    assert implied_vol == pytest.approx(true_vol, abs=VOL_TOLERANCE)

    # TEST2: repricing with the recovered volatility gives back the market
    # price, i.e. the solver keeps its own promise (price tolerance).
    # A fresh model is used instead of solver_model, so the check does
    # not rely on the solver's side effect.
    check_model = BlackScholesModel(
        risk_free_rate=RATE, volatility=implied_vol
    )
    assert check_model.price(option) == pytest.approx(
        market_price, abs=PRICE_TOLERANCE
    )


# degenerate point of the Manaster-Koehler initial guess. Its formula,
# sqrt(2 * |ln(S/K) + r*T| / T), is exactly zero when K equals the
# forward price S*exp(r*T), and at sigma = 0 d1 is 0/0. With r = 0 and
# K = S the log-moneyness is exactly 0.0 in floating point, so this is
# the case that reaches the degenerate point for sure.
FORWARD_ATM_RATE = 0.0


@pytest.mark.parametrize(
    "option_class",
    [
        pytest.param(EuropeanCall, id="call"),
        pytest.param(EuropeanPut, id="put"),
    ],
)
def test_newton_raphson_forward_atm(option_class):
    """At the forward ATM point the solver must still recover sigma.

    Kept apart from the round-trip test because it needs a different
    rate (r = 0), while the round-trip test uses one shared RATE.
    Guards the special case of the solver's initial guess: 
    here the Manaster-Koehler guess is exactly 0 and Brenner is used instead."
    Does not catch: points near, but not exactly at, the forward, where
    the log-moneyness is a tiny non-zero number and the guess is valid.
    """
    true_vol = 0.20
    option = option_class(underlying=SPOT, strike=SPOT, maturity=0.25)

    pricing_model = BlackScholesModel(
        risk_free_rate=FORWARD_ATM_RATE, volatility=true_vol
    )
    market_price = pricing_model.price(option)

    solver_model = BlackScholesModel(
        risk_free_rate=FORWARD_ATM_RATE, volatility=0.0
    )
    implied_vol = newton_raphson(solver_model, option, market_price)

    assert implied_vol is not None
    assert implied_vol == pytest.approx(true_vol, abs=VOL_TOLERANCE)


# ---------------------------------------------------------------------
# 2. The solver fails explicitly
# ---------------------------------------------------------------------

@pytest.mark.parametrize(
    "maturity, market_price",
    [
        # Market price different from the payoff (10.0): no sigma can
        # reproduce it.
        pytest.param(0.0, 12.0, id="expired_price_off_payoff"),
        pytest.param(0.0, 10.0, id="expired_price_on_payoff"),
        # Boundary test: exactly at the threshold the option counts as
        # expired (the model uses <=), so the solver must agree.
        pytest.param(BlackScholesModel.TIME_EPSILON, 10.0, id="at_threshold_price_on_payoff")
    ],
)
def test_newton_raphson_returns_none_at_expiry(maturity, market_price):
    """At expiry sigma is not identifiable: the solver must return None.

    At or below BlackScholesModel.TIME_EPSILON the model prices the
    option at its payoff whatever sigma is, so any sigma the solver
    returned would be arbitrary. Uses an ITM call (S=110, K=100) so the
    payoff, 10.0, is not zero.
    Does not catch: ill-posed cases before expiry (regime 3), which
    need their own test.
    """
    option = EuropeanCall(underlying=110.0, strike=100.0, maturity=maturity)
    solver_model = BlackScholesModel(risk_free_rate=RATE, volatility=0.2)

    assert newton_raphson(solver_model, option, market_price) is None


@pytest.mark.parametrize(
    "option_class, strike, maturity, true_vol",
    [
        # Short-dated OTM: the case the Manaster-Koehler guess turned
        # from None into a wrong sigma. Band half-width ~4e-3.
        pytest.param(EuropeanCall, 110.0, 0.02, 0.20, id="call_otm_short"),
        pytest.param(EuropeanPut, 90.0, 0.02, 0.20, id="put_otm_short"),
        # Deep ITM: the price (~41) is not small, but almost all of it is
        # intrinsic value, which does not depend on sigma.
        pytest.param(EuropeanCall, 65.0, 2.0, 0.10, id="call_deep_itm"),
        # Deep OTM: the price is essentially zero (~1e-24).
        pytest.param(EuropeanCall, 130.0, 0.25, 0.05, id="call_deep_otm")
    ],
)
def test_newton_raphson_returns_none_when_ill_posed(
    option_class, strike, maturity, true_vol
):
    """When the price does not pin sigma down, the solver returns None.

    The price is generated with a known sigma, but vega is so small that
    a whole band of sigmas reprices within the solver's price tolerance.
    Any sigma from that band would look like an answer: the solver must
    refuse to pick one. Cases are chosen well inside the ill-posed
    region (band at least 2x the solver's vol_tol), not at its border.
    Does not catch: a threshold that rejects well-posed cases (see
    call_low_vega in the round-trip test).
    """
    option = option_class(underlying=SPOT, strike=strike, maturity=maturity)
    pricing_model = BlackScholesModel(risk_free_rate=RATE, volatility=true_vol)
    market_price = pricing_model.price(option)

    solver_model = BlackScholesModel(risk_free_rate=RATE, volatility=0.0)

    assert newton_raphson(solver_model, option, market_price) is None


# ---------------------------------------------------------------------
# 3. The initial guess
# ---------------------------------------------------------------------
# Relative bump used to check that vega is maximal at the guess. 
# Limit: an error that moves the guess by less than about 1% can pass.
VEGA_BUMP = 0.01


@pytest.mark.parametrize(
    "strike, maturity",
    [
        pytest.param(130.0, 0.25, id="otm"),
        pytest.param(70.0, 1.0, id="itm"),
        # ln(S/K) and r*T nearly cancel: the guess depends mostly on the
        # rate term, so an error in r*T (e.g. its sign) shows up here.
        pytest.param(105.0, 1.0, id="near_forward"),
        pytest.param(160.0, 3.0, id="otm_long"),
    ],
)
def test_manaster_koehler_guess_maximizes_vega(strike, maturity):
    """Vega is maximal at the Manaster-Koehler guess."""
    option = EuropeanCall(underlying=SPOT, strike=strike, maturity=maturity)
    guess = manaster_koehler_guess(option, RATE)

    def vega_at(sigma):
        model = BlackScholesModel(risk_free_rate=RATE, volatility=sigma)
        return model.vega(option)

    assert vega_at(guess) >= vega_at(guess * (1 + VEGA_BUMP))
    assert vega_at(guess) >= vega_at(guess * (1 - VEGA_BUMP))
