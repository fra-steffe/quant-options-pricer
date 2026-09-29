import numpy as np

from Quant_engine.models import BlackScholesModel

def Brenner_guess(option, market_price: float) -> float:
    #Stima iniziale della volatilità usando la formula di Brenner-Subrahmanyam, che è una stima rapida e robusta per opzioni at-the-money
    S = option.underlying
    T = option.T
   # Assumes T > BlackScholesModel.TIME_EPSILON: newton_raphson returns
   # None for expired options before calling this function. 
    return np.sqrt(2 * np.pi / T) * (market_price / S)

def manaster_koehler_guess(option, rate: float) -> float:
    """Initial guess of Manaster and Koehler (1982).

    Returns the sigma at which vega is maximal, sqrt(2|ln(S/K) + rT| / T).
    There the Black-Scholes price, as a function of sigma, changes from
    convex to concave (inflection point), and Newton-Raphson started from
    it converges monotonically to the implied volatility whenever the
    price is within the no-arbitrage bounds. Unlike Brenner, it does not
    use the market price: it chooses where to start, it is not an
    estimate of the answer.
    Returns exactly 0.0 when K equals the forward S*exp(rT): the caller
    must handle that point. Assumes T > BlackScholesModel.TIME_EPSILON.
    """
    S = option.underlying
    K = option.strike
    T = option.T
    # ln(S/K) + rT = ln(F/K): log-moneyness with respect to the forward.
    forward_log_moneyness = np.log(S / K) + rate * T
    return np.sqrt(2 * abs(forward_log_moneyness) / T)

#function to deal with the initial guess, which will be manaster everywhere but the 
#single point where Manaster is 0
def _initial_guess(model, option, market_price: float) -> float:
    """Starting sigma for newton_raphson: Manaster-Koehler, or Brenner
    at the single point where Manaster-Koehler is zero."""
    sigma0 = manaster_koehler_guess(option, model.r)
    # Exact comparison on purpose: it does not check that two computed
    # values are equal, it guards the only point where the guess is
    # unusable. At sigma = 0, d1 is 0/0 (nan) and Newton never recovers,
    # while any positive guess, however small, works. At the forward ATM
    # point Brenner is accurate, since it is an ATM approximation.
    if sigma0 == 0.0:
        return Brenner_guess(option, market_price)
    return sigma0
    
#Metodo di Newton-Raphson per trovare la volatilità implicita dato un prezzo di mercato
def newton_raphson(model, option, market_price: float, tol = 1e-4, max_iter = 100):

    # At or below the model's expiry threshold price() returns the payoff
    # whatever sigma is, so sigma is not identifiable. Checked here, before
    # the initial guess, so that no guess formula ever sees T = 0 (Brenner
    # and Manaster-Koehler both divide by T). The threshold is read from
    # BlackScholesModel instead of being repeated, so the two cannot drift.
    if option.T <= BlackScholesModel.TIME_EPSILON:
        return None
    
    model.sigma = _initial_guess(model, option, market_price)
    #Iteriamo fino a raggiungere la convergenza o il numero massimo di iterazioni
    
    for i in range(max_iter):
        
        #consegno l'opzione al modello con la volatilità attuale e calcolo il prezzo teorico
        current_price = model.price(option)
        
        #calcolo la differenza tra il prezzo teorico e quello di mercato
        error = current_price - market_price
        
        # se l'errore è abbastanza piccolo, consideriamo la soluzione trovata
        if abs(error) < tol:
            print(f"Converged in {i} iterations. Implied Volatility: {model.sigma:.4f}")
            return model.sigma
        
        #consegno l'opzione al modello con la volatilità attuale e calcolo la vega (sensibilità del prezzo alla volatilità)
        current_vega = model.vega(option)

        # se il vega è troppo piccolo si ferma
        if current_vega < 1e-8:
            return None 
        #Aggiornamento di newton-raphson: nuova stima della volatilità basata sull'errore e sulla vega
        model.sigma -= error / current_vega

    # se non converge entro il massimo di iterazioni, l'algoritmo ha fallito
    return None


    