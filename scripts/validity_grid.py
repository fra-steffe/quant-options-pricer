"""Validity domain of the implied-volatility solver on synthetic data.

For every case of a grid (option type x strike x maturity x volatility)
the script prices the option with a known volatility, runs the solver on
that price and compares the answer with the known volatility.

Each case is judged twice:
- before solving, whether it is identifiable: the price pins sigma down
  within VOL_TOL, i.e. TOL / vega(true sigma) <= VOL_TOL;
- after solving, what the solver did: correct sigma, None, or a wrong
  sigma returned as if it were right.

Run from the project root (so that Quant_engine can be imported):

    python -m scripts.validity_grid

Limits:
- Results hold for SPOT = 100. Black-Scholes prices scale with the spot,
  but TOL is an absolute price tolerance, so with a different spot the
  border of the identifiable region moves.
- The solver is run with TOL and VOL_TOL defined here, passed
  explicitly. If the library defaults change, this script keeps
  measuring this configuration, not the one used by main.py.
"""

import contextlib
import io
from collections import Counter
from itertools import product

from Quant_engine.instruments import EuropeanCall, EuropeanPut
from Quant_engine.models import BlackScholesModel
from Quant_engine.solvers import newton_raphson

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
SPOT = 100.0
RATE = 0.05

# Solver configuration, passed explicitly to newton_raphson and used to
# judge the cases, so the two cannot disagree. Same values as the
# library defaults on 05/10/2026.
TOL = 1e-4
VOL_TOL = 1e-3

# The grid: 2 x 31 x 8 x 5 = 2480 cases.
OPTION_CLASSES = (EuropeanCall, EuropeanPut)
STRIKES = tuple(float(k) for k in range(50, 205, 5))  # 50, 55, ..., 200
MATURITIES = (1 / 365, 0.02, 0.1, 0.25, 0.5, 1.0, 2.0, 3.0)  # 1 day to 3y
VOLS = (0.05, 0.1, 0.2, 0.5, 1.0)

# Subset printed as a strike x maturity table. Every value must also be
# in STRIKES and MATURITIES: the table reads results already computed.
SLICE_STRIKES = (50.0, 70.0, 90.0, 100.0, 110.0, 130.0, 160.0, 200.0)
SLICE_MATURITIES = (0.02, 0.25, 1.0, 3.0)

# Possible outcomes of the solver on one case.
CORRECT = "correct"
NO_ANSWER = "None"
WRONG = "wrong"
OUTCOMES = (CORRECT, NO_ANSWER, WRONG)
SYMBOLS = {CORRECT: "ok", NO_ANSWER: "-", WRONG: "WRONG"}

# ---------------------------------------------------------------------
# One case
# ---------------------------------------------------------------------
def solve(option, market_price: float) -> float | None:
    """Run the solver: implied sigma, or None.

    Works around two known issues of the library, kept in this one
    function so that only it changes when they are fixed:
    - newton_raphson writes into model.sigma (side effect), so every
      call gets a fresh model and no case inherits state from another;
    - newton_raphson prints a line at every convergence, which would
      bury the results, so its output is discarded (until logging).
    """
    # The starting volatility is a placeholder: the solver replaces it
    # with its own initial guess.
    model = BlackScholesModel(risk_free_rate=RATE, volatility=0.0)
    # Inside the with block anything printed goes to an in-memory buffer
    # that is then thrown away; stdout is restored on exit, even if the
    # solver raises.
    with contextlib.redirect_stdout(io.StringIO()):
        return newton_raphson(
            model, option, market_price, tol=TOL, vol_tol=VOL_TOL
        )


def classify(option_class, strike: float, maturity: float,
             true_vol: float) -> tuple[bool, str]:
    """Judge one case: (is it identifiable?, what did the solver do?)."""
    option = option_class(underlying=SPOT, strike=strike, maturity=maturity)

    # The "market" price, generated with the volatility we know.
    pricing_model = BlackScholesModel(risk_free_rate=RATE, volatility=true_vol)
    market_price = pricing_model.price(option)

    # Judged before solving, with the true sigma, so the verdict on the
    # case does not depend on what the solver does. Same rule as the
    # solver (tol / vega <= vol_tol), written as a product so that
    # vega = 0 cannot divide by zero.
    identifiable = pricing_model.vega(option) * VOL_TOL >= TOL

    implied_vol = solve(option, market_price)
    if implied_vol is None:
        outcome = NO_ANSWER
    elif abs(implied_vol - true_vol) <= VOL_TOL:
        outcome = CORRECT
    else:
        outcome = WRONG
    return identifiable, outcome


# ---------------------------------------------------------------------
# The grid
# ---------------------------------------------------------------------
def run_grid() -> list[dict]:
    """Classify every case of the grid; one dict per case."""
    results = []
    # product() yields every combination of the four sequences, like
    # four nested for loops.
    for option_class, strike, maturity, true_vol in product(
        OPTION_CLASSES, STRIKES, MATURITIES, VOLS
    ):
        identifiable, outcome = classify(
            option_class, strike, maturity, true_vol
        )
        results.append({
            "option_class": option_class,
            "strike": strike,
            "maturity": maturity,
            "true_vol": true_vol,
            "identifiable": identifiable,
            "outcome": outcome,
        })
    return results


# ---------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------
def print_summary(results: list[dict]) -> None:
    """Counts of cases by (identifiable, outcome)."""
    counts = Counter((r["identifiable"], r["outcome"]) for r in results)

    print("| Case | " + " | ".join(OUTCOMES) + " | Total |")
    print("|---" * (len(OUTCOMES) + 2) + "|")
    for identifiable, label in ((True, "Identifiable"),
                                (False, "Not identifiable")):
        row = [counts[(identifiable, outcome)] for outcome in OUTCOMES]
        cells = " | ".join(str(n) for n in row)
        print(f"| {label} | {cells} | {sum(row)} |")


def print_slice_table(results: list[dict], option_class,
                      true_vol: float) -> None:
    """Outcome for each strike x maturity, one option type and sigma."""
    outcome_at = {
        (r["strike"], r["maturity"]): r["outcome"]
        for r in results
        if r["option_class"] is option_class and r["true_vol"] == true_vol
    }

    header = " | ".join(f"{k:g}" for k in SLICE_STRIKES)
    print(f"| T \\ K | {header} |")
    print("|---" * (len(SLICE_STRIKES) + 1) + "|")
    for maturity in SLICE_MATURITIES:
        cells = " | ".join(
            SYMBOLS[outcome_at[(strike, maturity)]]
            for strike in SLICE_STRIKES
        )
        print(f"| {maturity:g} | {cells} |")


if __name__ == "__main__":
    results = run_grid()

    print(f"Cases: {len(results)} | S = {SPOT:g}, r = {RATE:g}, "
          f"tol = {TOL:g}, vol_tol = {VOL_TOL:g}")
    print()
    print_summary(results)
    for option_class in OPTION_CLASSES:
        print()
        print(f"{option_class.__name__}, true sigma = 20% "
              f"(ok: correct, -: None, WRONG: wrong sigma)")
        print()
        print_slice_table(results, option_class, true_vol=0.2)
