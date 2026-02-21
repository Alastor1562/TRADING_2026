import pandas as pd
import ta

def add_indicators(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Calcula RSI, MACD y Bollinger Bands usando la librería `ta`
    y los agrega como columnas al dataframe.

    Parámetros esperados en params:
      rsi_window   : periodo del RSI (int)
      macd_fast    : EMA rápida del MACD (int)
      macd_slow    : EMA lenta del MACD (int)
      macd_signal  : EMA de la señal MACD (int)
      bb_window    : periodo de las Bandas de Bollinger (int)
      bb_std       : número de desviaciones estándar (float)
    """
    df = df.copy()

    # ── RSI ───────────────────────────────────────────────────
    rsi_ind    = ta.momentum.RSIIndicator(
        close=df['Close'],
        window=params['rsi_window']
    )
    df['rsi']  = rsi_ind.rsi()

    # ── MACD ──────────────────────────────────────────────────
    macd_ind         = ta.trend.MACD(
        close=df['Close'],
        window_fast=params['macd_fast'],
        window_slow=params['macd_slow'],
        window_sign=params['macd_signal']
    )
    df['macd']       = macd_ind.macd()
    df['macd_signal']= macd_ind.macd_signal()
    df['macd_hist']  = macd_ind.macd_diff()   # histograma = MACD − Signal

    # ── Bollinger Bands ───────────────────────────────────────
    bb_ind        = ta.volatility.BollingerBands(
        close=df['Close'],
        window=params['bb_window'],
        window_dev=params['bb_std']
    )
    df['bb_upper']= bb_ind.bollinger_hband()
    df['bb_mid']  = bb_ind.bollinger_mavg()
    df['bb_lower']= bb_ind.bollinger_lband()
    df['bb_pctb'] = bb_ind.bollinger_pband()  # %B: 0 = banda inf, 1 = banda sup

    return df.dropna()


def generate_signals(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    Genera señales individuales por indicador y la señal combinada 2/3.

    Cada indicador vota:
      +1  →  señal LONG
      -1  →  señal SHORT
       0  →  neutral

    signal final:
      +1  si la suma de votos >= 2  (al menos 2 de 3 votan long)
      -1  si la suma de votos <= -2 (al menos 2 de 3 votan short)
       0  en caso contrario (sin confirmación)
    """
    df = df.copy()

    # ── Voto RSI ──────────────────────────────────────────────
    df['sig_rsi'] = 0
    df.loc[df['rsi'] < params['rsi_lower'], 'sig_rsi'] =  1
    df.loc[df['rsi'] > params['rsi_upper'], 'sig_rsi'] = -1

    # ── Voto MACD (dirección del histograma) ──────────────────
    df['macd_hist_diff'] = df['macd_hist'].diff()
    df['sig_macd'] = 0
    df.loc[df['macd_hist_diff'] > 0, 'sig_macd'] =  1
    df.loc[df['macd_hist_diff'] < 0, 'sig_macd'] = -1

    # ── Voto Bollinger Bands (%B) ─────────────────────────────
    df['sig_bb'] = 0
    df.loc[df['bb_pctb'] < params['bb_lower_pct'], 'sig_bb'] =  1
    df.loc[df['bb_pctb'] > params['bb_upper_pct'], 'sig_bb'] = -1

    # ── Confirmación 2/3 ──────────────────────────────────────
    df['votes']  = df['sig_rsi'] + df['sig_macd'] + df['sig_bb']
    df['signal'] = 0
    df.loc[df['votes'] >=  2, 'signal'] =  1
    df.loc[df['votes'] <= -2, 'signal'] = -1

    return df