import math

import pandas as pd
from datetime import datetime
from Quant_engine.instruments import EuropeanCall, EuropeanPut

class YahooDataTransformer:
    
    def __init__(self, current_date: str = None): # la data di riferimento per il pricing
       # se non viene specificata una data usiamo la data di oggi, altrimenti accetta una data in formato stringa 'YYYY-MM-DD' (come da yahoo finance) e la converte in un oggetto datetime
        if current_date is not None:
            self.current_date = datetime.strptime(current_date, "%Y-%m-%d")
        else:
            #oggi è la data di riferimento
            self.current_date = datetime.now()
    
    def _calculate_time_to_maturity(self, expiry_str: str) -> float: #resituisce un float
        #convertiamo la stringa di scadenza in un oggetto datetime
        expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d")
        
        #calcola la differenza 
        delta = expiry_date - self.current_date

        #estrae il numero intero dei giorni e calcola gli anni
        T = delta.days / 365.25 #teniamo in conto gli anni bisestili
        
        return T
    
    #pulisce il dataframe
    def _clean_data(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        #Crea una copia indipendente del DataFrame per non alterare i dati originali in memoria.
        clean_df = raw_df.copy()

        # elimina le righe con valori nulli sulle colonne d'interesse
        # Missing bid/ask are handled by the quote filter below.
        clean_df = clean_df.dropna(subset = ["strike"])

        # Cast to float before comparing and averaging the quotes.
        clean_df['strike'] = clean_df['strike'].astype(float)
        clean_df['bid'] = clean_df['bid'].astype(float)
        clean_df['ask'] = clean_df['ask'].astype(float)

        # Market price = mid ((bid + ask) / 2). we also filter for options 
        # with bid > 0 and ask >= bid. Written as "keep if valid"
        # so NaN quotes are dropped too: comparisons with NaN are False

        valid_quote = (
            (clean_df['bid'] > 0)
            & (clean_df['ask'] >= clean_df['bid'])
        )
        clean_df = clean_df[valid_quote]
        clean_df['mid'] = (clean_df['bid'] + clean_df['ask']) / 2

        return clean_df
    
    #il metodo principale
    #prende il dataframe grezzo e resituisce una lista di dizionari,ognuno contenente 
    #l'oggetto option e il prezzo di mercato
    def transform_to_objects(self, raw_df: pd.DataFrame, underlying: float, expiry_str: str) -> list:

        #Pulizia: passiamo raw_df 
        clean_df = self._clean_data(raw_df)

        #Normalizzazione del tempo:
        T = self._calculate_time_to_maturity(expiry_str)

        #lista di opzioni che riempiamo
        options_list = []

        if T <= 0: #condizione di sicurezza
            return []

        
        for index, row in clean_df.iterrows():
            strike = row['strike']
            price = row["mid"]
            option_type = str(row["option_type"]).strip().lower()

            #instanziazione dell'oggetto corretto
            if option_type == 'call':
                option_obj = EuropeanCall(underlying, strike, T)
            else:
                option_obj = EuropeanPut(underlying, strike, T)
            
            
            
            #mette assieme oggetto e il suo prezzo reale in un dizionario che mette nella lista
            options_list.append({"instruments": option_obj, 
                                 "target_price": price, 
                                 "option_type": option_type.capitalize()#salva come 'Call' o 'Put'
                                 })
        return options_list


def keep_otm(options:list, rate:float) -> list:
    """Keep calls with K >= F and puts with K < F, F = S * exp(r * T), where 
    the forward is the ATM point.
    Call and put with the same strike have the same implied volatility,
    so one contract per strike loses nothing. The OTM one is almost all
    time value, trades with tighter spreads and carries almost no
    early-exercise premium when the option is American.
    """
    otm_options = []
    for item in options:
        option = item["instruments"]
        forward = option.underlying * math.exp(rate * option.T)
        if isinstance(option, EuropeanCall):
            is_otm = option.strike >= forward
        else:
            is_otm = option.strike < forward
        if is_otm:
            otm_options.append(item)
    return otm_options




            
