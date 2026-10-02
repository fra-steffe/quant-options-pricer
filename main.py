#NOMENCLATURA VARIABILI LOCALI: 
# expiry_str = "2024-12-31" #scadenza in formato stringa, da convertire in datetime
# expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d") #scadenza convertita in oggetto datetime
#T = il tempo in frazione di anno, calcolato come (expiry_date - current_date).days / 365.25, che è il formato richiesto dai modelli di pricing. T è un float che rappresenta il tempo alla scadenza in anni, con precisione fino a 4 cifre decimali (ad esempio, 0.25 per 3 mesi, 0.5 per 6 mesi, ecc.).

from datetime import date

from Quant_engine.instruments import EuropeanCall, EuropeanPut
from Quant_engine.models import BlackScholesModel, MonteCarloPricer
from Quant_engine.solvers import newton_raphson
from Quant_engine.scrapers import YahooProvider
from Quant_engine.transformers import YahooDataTransformer, keep_otm
import pandas as pd

#initial configuration
TICKER = "AAPL"
MIN_DAYS_TO_EXPIRY = 30
RISK_FREE_RATE = 0.05


#definiamo la funzione principale
def main():
    print(f"--- Avvio Analisi Opzioni per {TICKER} ---")


    #istanziazione moduli
    provider = YahooProvider()
    tranformer = YahooDataTransformer()

    #otteniamo il prezzo 
    try: 
        UNDERLYING_PRICE = provider.get_underlying_price(TICKER)
        print(f"Prezzo Spot Rilevato per {TICKER}: ${UNDERLYING_PRICE: .2f}")
    except Exception as e:
        print(f"Errore critico: {e}")
        return

    #Setup modello BS, la volatilità è casuale e verra sovrascritta dal solver, ma è necessario metterla
    #per instanziare la classe
    bs_model = BlackScholesModel(risk_free_rate = RISK_FREE_RATE, volatility = 0.10)

    print("finding expiration dates...")
    dates = provider.get_expiration_dates(TICKER) # tutte le date disponibili per le opzioni di quel ticker
    
    if not dates:
        print("Errore: Nessuna data trovata. Termino il programma.")
        return
    
    # Test su una singola data
    #prendiamo una scadenza a medio termine con buona liquidità. 
    # First expiry at least MIN_DAYS_TO_EXPIRY away (dates are sorted).
    today = date.today()
    expiry = None
    for d in dates:
        if (date.fromisoformat(d) - today).days >= MIN_DAYS_TO_EXPIRY:
            expiry = d
            break
    if expiry is None:
        print("Errore: nessuna scadenza abbastanza lontana.")
        return
    print(f"Scadenza selezionata: {expiry}")

    # ESTRAZIONE
    raw_data = provider.get_option_chain(TICKER, expiry)

    # TRASFORMAZIONE
    option_list = tranformer.transform_to_objects(raw_data, UNDERLYING_PRICE, expiry)

    # save number of valid contracts in chain before filtering for OTM contracts
    n_quoted = len(option_list)

    # keep only OTM options
    option_list = keep_otm(option_list, RISK_FREE_RATE)

    # MOTORE DI CALCOLO
    results = []
    print(f"inizio calcolo volatilità implicita per {len(option_list)} contratti...")

    # iteriamo attraverso la lista di dizionari di opzioni
    for item in option_list:
        # estraiamo i valori associati alle due chiavi
        option = item["instruments"]
        market_price = item["target_price"]

        # chiamiamo il solver
        iv_computed = newton_raphson(bs_model, option, market_price)

        # se fallisce salto la riga
        if iv_computed is None:
            continue

        # salviamo i risultati 
        results.append({"Strike": option.strike, 
                        "option_type": "Call" if isinstance(option, EuropeanCall) else "Put",
                        "IV": iv_computed,
                        "Market_price": market_price,
                        "Underlying_Price": UNDERLYING_PRICE
        })


    # raccolta metriche della pipeline
    n_chain = len(raw_data)
    n_otm = len(option_list)
    n_solved = len(results)
    print(f"Contracts in chain:          {n_chain:4d}") 
    print(f"  dropped, no valid quote:   {n_chain - n_quoted:4d}")
    print(f"  dropped, ITM:              {n_quoted - n_otm:4d}")
    print(f"  rejected by solver:        {n_otm - n_solved:4d}")
    print(f"Implied vols computed:       {n_solved:4d}")



    # ESPORTAZIONE CSV PER ANALISI ESPLORATIVA
    # trasformiamo la lsita di dizionari in un dataframe
    df_results = pd.DataFrame(results)

    # salviamo in un file CSV
    nome_file = f"superficie_iv_{TICKER}_{expiry}.csv"
    df_results.to_csv(nome_file, index = False)
    print(f"dati esportati con successo in: {nome_file}")

if __name__ == "__main__":
    main()

    






