import numpy as np

def Brenner_guess(option, market_price: float) -> float:
    #Stima iniziale della volatilità usando la formula di Brenner-Subrahmanyam, che è una stima rapida e robusta per opzioni at-the-money
    S = option.underlying
    T = option.T
    if T <= 1e-6:
        return None #se l'opzione è scaduta, la volatilità implicita è zero perché il prezzo non cambia più con la volatilità
    return np.sqrt(2 * np.pi / T) * (market_price / S)
    
#Metodo di Newton-Raphson per trovare la volatilità implicita dato un prezzo di mercato
def newton_raphson(model, option, market_price: float, tol = 1e-4, max_iter = 100):
    
    model.sigma = Brenner_guess(option, market_price) #partiamo da una stima iniziale della volatilità (valore default, meglio implementare la formula di brenner per gesitre casi estremi OTM)
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


    