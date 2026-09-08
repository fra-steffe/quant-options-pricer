import numpy as np
from scipy.stats import norm
from Quant_engine.instruments import EuropeanCall, EuropeanPut

class BlackScholesModel:
   # Definiamo un epsilon (circa mezz'ora di tempo) sotto il quale l'opzione è "scaduta"
   # Questo è importante per evitare problemi numerici quando T è molto piccolo, poiché d1 e d2 possono diventare instabili.
    TIME_EPSILON = 1e-6
   
    def __init__(self, risk_free_rate: float, volatility: float):
        self.r = risk_free_rate
        self.sigma = volatility
        
    def _d1(self, S:float, K: float, T: float) -> float:
        #calcola il parametro d1 (assume T > TIME_EPSILON)
        return (np.log(S/K) + (self.r + 0.5 * self.sigma**2) * T) / (self.sigma * np.sqrt(T))
                                                                     
    def _d2( self, d1_value: float, T: float) -> float:
        #calcola il parametro d2 (assume T > TIME_EPSILON)
        return d1_value - self.sigma * np.sqrt(T)
    def price(self, option) -> float:
        #calcola il prezzo dell'opzione usando la formula di Black-Scholes
        S = option.underlying
        K = option.strike
        T = option.T
        # Gestione robusta della scadenza: se T è quasi zero, restituisci il payoff intrinseco
        if T <= BlackScholesModel.TIME_EPSILON:
            return option.payoff(S)
        d1 = self._d1(S, K, T)
        d2 = self._d2(d1, T)

        if isinstance(option, EuropeanCall): #controlla se option (argomento di price) è un'istanza di european call
            return S * norm.cdf(d1) - K * np.exp(-self.r * T) * norm.cdf(d2)
        elif isinstance(option, EuropeanPut): 
            return K * np.exp(-self.r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        else:
            raise ValueError("Unsupported option type")
    def vega(self,option) -> float:
        #calcola la vega dell'opzione usando la formula di Black-Scholes
        S = option.underlying
        K = option.strike
        T = option.T
        if T <= BlackScholesModel.TIME_EPSILON:
            return 0.0 #se l'opzione è scaduta, la vega è zero perché il prezzo non cambia più con la volatilità
        d1 = self._d1(S, K, T)
        #restituisce il valore assoluto della vega, che è lo stesso per call e put, poiché dipende solo da d1
        vega_value = S * norm.pdf(d1) * np.sqrt(T)
        
        return vega_value



       
class MonteCarloPricer:
    #motore di pricing Monte Carlo per opzioni senza formule chiuse
    def __init__(self, risk_free_rate: float, volatility: float, num_paths: int= 10000, num_steps: int= 100):

        self.r = risk_free_rate
        self.sigma = volatility
        self.num_paths = num_paths #numero di simulazioni
        self.num_steps = num_steps #numero di step temporali per ogni simulazione
    def _generate_z(self)-> np.ndarray:
        #Creiamo una matrice di dimensioni (n_paths, n_steps) riempita di numeri estratti da una Normale Standard. la useremo per vettorizzare i calcoli
        #Questi numeri rappresentano i "colpi di scena" casuali che influenzeranno l'evoluzione del prezzo dell'asset sottostante in ogni simulazione.
        # Righe = percorsi (paths), Colonne = step temporali (steps)
        half_paths = int(self.num_paths / 2)#Calcolo la metà esatta dei percorsi necessari
        z_half = np.random.standard_normal((half_paths, self.num_steps)) #Genera la prima metà dei percorsi con numeri casuali standard
        z_anti = -z_half #Creo la seconda metà dei percorsi usando l'antithetic variate (inversione dei numeri casuali) per ridurre la varianza
        Z = np.concatenate((z_half, z_anti), axis=0) #Combino le due metà per ottenere la matrice completa di shock casuali
        if self.num_paths % 2 == 1: #Se il numero totale di percorsi è dispari, genero un percorso extra per completare la matrice
            extra_z = np.random.standard_normal((1, self.num_steps)) #Genera un percorso extra
            Z = np.concatenate((Z, extra_z), axis=0) #Aggiunge il percorso extra alla matrice Z
        return np.random.standard_normal((self.num_paths, self.num_steps))
    def _simulate_paths(self, S0: float, T: float) -> np.ndarray:
        #Simuliamo i percorsi del prezzo dell'asset sottostante usando il modello di moto browniano geometrico.
        dt = T / self.num_steps #calcoliamo il passo temporale
        Z = self._generate_z() #matrice di shock casuali
        drift = (self.r - 0.5 * self.sigma**2) * dt #componente deterministica del movimento del prezzo
        shock = self.sigma * np.sqrt(dt) * Z #componente stocastica
        daily_returns = drift + shock #ritorno giornaliero totale
        accumulated_returns = np.cumsum(daily_returns, axis=1) #ritorno cumulativo lungo i passi temporali, matrice NxP
        S_t = S0 * np.exp(accumulated_returns) #prezzo dell'asset sottostante in ogni percorso e passo temporale
        return S_t
    def _get_final_prices(self, S_t: np.ndarray) -> np.ndarray:
        #estraiamo i prezzi finali (alla scadenza) da ogni percorso simulato
        final_prices = S_t[:, -1] #prendiamo l'ultima colonna della matrice S_t, che contiene i prezzi alla scadenza
        return final_prices #vettore di dimensione (n_paths,) con i prezzi finali di ogni percorso
    def price(self, option) -> float:
        #calcolo il prezzo usando la media dei payoff scontati 
        S0 = option.underlying
        T = option.T #estraggo i dati del contratto da Option
        if T <= BlackScholesModel.TIME_EPSILON: #gestione robusta della scadenza: se T è quasi zero, restituisci il payoff intrinseco
            return option.payoff(S0) #se l'opzione è scaduta, restituisci il payoff intrinseco. uso lo stesso criterio di scadenza del modello di Black-Scholes per coerenza
        S_t = self._simulate_paths(S0, T) #simuliamo i percorsi del prezzo sottostante
        final_prices = self._get_final_prices(S_t) #estraiamo i prezzi finali alla scadenza
        final_payoffs = option.payoff(final_prices) #calcoliamo il payoff finale per ogni percorso
        expected_payoff = np.mean(final_payoffs) #calcoliamo la media dei payoff
        discount_factor = np.exp(-self.r * T) #calcoliamo il fattore di sconto per portare il valore attuale
        theoretical_price = expected_payoff * discount_factor #prezzo teorico come payoff atteso scontato risk-neutral
        return float(theoretical_price) #restituiamo il prezzo come float, anche se è già un float, per coerenza con il tipo di ritorno dichiarato
