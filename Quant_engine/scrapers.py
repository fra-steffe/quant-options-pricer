import abc
import numpy as np
import pandas as pd
import yfinance as yf

class DataProvider(abc.ABC): #classe astratta per i data scraper, definisce l'interfaccia che tutti i data scraper devono implementare
    @abc.abstractmethod
    def get_option_chain(self, ticker: str, expiry_str: str) -> pd.DataFrame:
        pass 
    @abc.abstractmethod
    def get_expiration_dates(self, ticker: str) -> list:
        pass

#implementazione specifica per yaohoo finance
class YahooProvider(DataProvider): #eredita da DataProvider
    
    def get_expiration_dates(self, ticker: str) -> list:
        try: 
            stock = yf.Ticker(ticker) #crea un oggetto Ticker di yfinance per il ticker specificato
            return list(stock.options) #restituisce la lista delle date di scadenza disponibili per quel ticker
        except Exception as e:
            print(f"Errore nel recupero delle scadenze per {ticker}: {e}")
            return[] #restituisce una lista vuota in caso di errore
    
    def get_option_chain(self, ticker: str, expiry_str: str) -> pd.DataFrame:
        try:
            stock = yf.Ticker(ticker)

            # Scarichiamo solo la catena per la data richiesta
            chain = stock.option_chain(expiry_str)

            calls = chain.calls.copy() #crea una copia del DataFrame delle call per evitare modifiche non intenzionali
            puts = chain.puts.copy()

            #etichettiamo le opzioni con un tipo (call o put) per facilitare l'identificazione
            calls['option_type'] = 'call'
            puts['option_type'] = 'put'

            #uniamo le call e le put in un unico DataFrame per quella scadenza
            full_chain = pd.concat([calls, puts], ignore_index=True) #crea un nuovo indice per il DataFrame unito
            return full_chain
        except Exception as e:
            print(f"Errore nell'estrazione dei dati per {ticker} in data {expiry_str}: {e}")
            return pd.DataFrame() # Restituisce tabella vuota in caso di errore
    

     #recupera l'ultimo prezzo di mercato per il sottostante   
    def get_underlying_price(self, ticker: str) -> float:

        print(f"Recupero il prezzo attuale per {ticker}")

        #scarica ultimo giorno di contrattazione
        stock = yf.Ticker(ticker)
        data = stock.history(period = "1d")

        if data.empty:
            raise ValueError(f"Errore: impossibile recuperare il prezzo per il ticker")

        #prendi prezzo di chiusura (close) disponibile
        current_price = float(data['Close'].iloc[-1])  

        return current_price




            
         


