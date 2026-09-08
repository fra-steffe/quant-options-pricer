import numpy as np

class Option: #classe base astratta per tutte le opzioni
    def __init__(self, underlying: float, strike: float, maturity: float):
        self.underlying = underlying
        self.strike = strike
        self.T = maturity
    def __repr__(self):
        return f"{self.__class__.__name__}(S = {self.underlying}, K = {self.strike}, T = {self.T})"
    

class EuropeanCall (Option): #Classe specifica per la Call Europea  eredita S,K,T dalla madre
    def payoff(self, S_t: float) ->float: #questa ci serve per caalcolare opzioni senza formule chiuse
        return np.maximum(S_t - self.strike, 0.0) #il payoff è max(S-K,0)
    

class EuropeanPut (Option) :  #Classe specifica per la Put Europea.
    #La classe specializzata per le put. eredita S,K,T dalla madre
    def payoff(self, S_t: float) ->float: 
        return np.maximum(self.strike - S_t, 0.0) #il payoff qui  è max(K-S,0)





