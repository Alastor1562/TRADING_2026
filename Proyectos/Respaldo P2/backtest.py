import pandas as pd
import metrics
import signals

def run_backtest(data, params, initial_cash, COM):
    """
    Ejecuta el backtest completo sobre el dataframe dado.

    Flujo por barra:
      1. Si hay posición abierta → chequear SL / TP y cerrar si aplica.
      2. Si no hay posición → abrir long o short según la señal 2/3.
      3. Calcular el valor del portfolio al precio actual.

    NOTA: El índice es entero (posición en la serie continua de precios).
    No se usan timestamps para ninguna lógica de trading.

    Retorna dict con:
      portfolio : pd.DataFrame con columna 'portfolio_value'
      trades    : pd.DataFrame con el historial de operaciones
      metrics   : dict de métricas de rendimiento
    """
    data = data.copy()

    data = signals.add_indicators(data, params)
    data = signals.generate_signals(data, params)
    data = data.dropna()  # quedarnos solo con filas donde se pudo generar señal
    data = data.reset_index(drop=True)  # asegurar índice entero limpio

    sl    = params['sl']
    tp    = params['tp']
    n_btc = params['n_btc']

    lost_to_com = 0.0

    cash             = initial_cash
    position         = None          # None | dict con info de la posición activa
    portfolio_values = []
    trades           = []

    for idx, row in data.iterrows():
        price = row['Close']

        # ── 1. Gestión de riesgo: cerrar posición activa ──────
        if position is not None:

            if position['type'] == 'long':
                pnl_pct = (price - position['entry']) / position['entry']

                if pnl_pct <= -sl or pnl_pct >= tp:
                    proceeds  = price * position['n_btc'] * (1 - COM)
                    lost_to_com += position['n_btc'] * price * COM
                    pnl       = proceeds
                    cash     += proceeds
                    trades.append({
                        'bar': idx, 'type': 'long',
                        'entry': position['entry'], 'exit': price, 'pnl': pnl
                    })
                    position = None

            elif position['type'] == 'short':
                pnl_pct = (position['entry'] - price) / position['entry']

                if pnl_pct <= -sl or pnl_pct >= tp:
                    cost_cover = price * position['n_btc'] * (COM)
                    lost_to_com += cost_cover
                    pnl        = (position['entry'] - price) * position['n_btc'] - cost_cover
                    cash      += pnl
                    trades.append({
                        'bar': idx, 'type': 'short',
                        'entry': position['entry'], 'exit': price, 'pnl': pnl
                    })
                    position = None

        # ── 2. Abrir nueva posición si no hay una activa ──────
        sig = row['signal']

        if position is None:
            if sig == 1:                           # LONG
                cost = price * n_btc * (1 + COM)

                if cash >= cost:
                    lost_to_com += price * n_btc * COM
                    cash    -= cost
                    position = {
                        'type': 'long', 'entry': price,
                        'n_btc': n_btc, 'cost': cost
                    }


            elif sig == -1:                        # SHORT (vender primero)
                colateral = price * n_btc * (1 + COM)
                if cash >= colateral:
                    contract_cost  = price * n_btc * (COM)
                    lost_to_com   += contract_cost
                    cash     -= contract_cost
                    position  = {
                        'type': 'short', 'entry': price,
                        'n_btc': n_btc, 'coms': contract_cost,
                        'colateral': colateral,
                    }

        # ── 3. Valor del portfolio en este instante ───────────
        if position is None:
            pv = cash
        elif position['type'] == 'long':
            pv = cash + price * position['n_btc']
        else:
            pv = cash +  (position['entry'] - price) * position['n_btc']

        portfolio_values.append(pv)

    # ── Cierre forzado al final del período ───────────────────
    if position is not None:
        last_price = data['Close'].iloc[-1]
        if position['type'] == 'long':
            cash += last_price * position['n_btc'] * (1 - COM)
            lost_to_com += last_price * position['n_btc'] * COM
        else:
            cash += (position['entry'] - last_price) * position['n_btc'] - (last_price * position['n_btc'] * (COM))
            lost_to_com += last_price * position['n_btc'] * COM

    tr_df  = pd.DataFrame(trades) if trades else \
             pd.DataFrame(columns=['bar', 'type', 'entry', 'exit', 'pnl'])
    
    metricas = metrics.calculate_metrics(portfolio_values, tr_df, rf=0.0)
    metricas['lost_to_com'] = lost_to_com

    return {'portfolio': portfolio_values, 'trades': tr_df, 'metrics': metricas}



