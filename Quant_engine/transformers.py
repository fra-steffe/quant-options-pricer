import pandas as pd
from datetime import datetime
from Quant_engine.instruments import EuropeanCall, EuropeanPut

class YahooDataTransformer:
    
    def __init__(self, current_date: str = None): # la data di riferimento per il pricing
       # se non viene specificata una data usiamo la data di oggi, altrimenti accetta una data in formato stringa 'YYYY-MM-DD' (come da yahoo finance) e la converte in un oggetto datetime
        if current_date is not None:
            #converte il testo (stringa con formato 'YYYY-MM-DD') in un oggetto temporale datetime su cui possiamo fare operazioni di data
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
    def _clean_data(self, raw_df: pd.DataFrame) -> pd.DataFrame: #metodo privato
        #Crea una copia indipendente del DataFrame per non alterare i dati originali in memoria.
        clean_df = raw_df.copy()

        #elimina i le righe con valori nulli sulle colonne d'interesse
        clean_df = clean_df.dropna(subset = ["strike", "lastPrice", 'volume'])

        #Mantieni solo le righe per opzioni liquide
        clean_df = clean_df[clean_df['volume'] > 0]

        #Forza la conversione delle colonne 'strike' e 'lastPrice' al tipo di dato float
        clean_df['strike'] = clean_df['strike'].astype(float)
        clean_df['lastPrice'] = clean_df['lastPrice'].astype(float)

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
            price = row["lastPrice"]
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
            
