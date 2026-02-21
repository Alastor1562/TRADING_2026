import optuna
import backtest

def make_objective(data, metric= 'calmar'):
    """
    Devuelve la función objetivo para Optuna.

    El closure captura `data` y `metric`, de modo que Optuna
    solo recibe el objeto `trial` como argumento.

    Restricciones aplicadas dentro de la función:
      - macd_fast < macd_slow  (condición obligatoria de MACD)
      - rsi_lower < rsi_upper  (lógica de la estrategia)
    """
    def objective(trial: optuna.Trial) -> float:

        # ── RSI ───────────────────────────────────────────────
        rsi_window = trial.suggest_int('rsi_window', 5, 28)
        rsi_lower  = trial.suggest_int('rsi_lower',  20, 45)
        rsi_upper  = trial.suggest_int('rsi_upper',  55, 85)

        # ── MACD ──────────────────────────────────────────────
        macd_fast   = trial.suggest_int('macd_fast',   3, 15)
        macd_slow   = trial.suggest_int('macd_slow',  10, 31)
        macd_signal = trial.suggest_int('macd_signal',  3, 15)

        # Restricción: fast < slow (poda del trial si no se cumple)
        if macd_fast >= macd_slow:
            raise optuna.exceptions.TrialPruned()

        # ── Bollinger Bands ───────────────────────────────────
        bb_window    = trial.suggest_int  ('bb_window',    10, 30)
        bb_std       = trial.suggest_float('bb_std',       1.0, 3.0)
        bb_lower_pct = trial.suggest_float('bb_lower_pct', 0.0, 0.25)
        bb_upper_pct = trial.suggest_float('bb_upper_pct', 0.80, 1.5)

        # ── Risk management ───────────────────────────────────
        sl    = trial.suggest_float('sl',    0.05, 0.10)
        tp    = trial.suggest_float('tp',    0.05, 0.10)
        n_btc = trial.suggest_float('n_btc', 0.5, 5.0)

        params = {
            'rsi_window': rsi_window, 'rsi_lower': rsi_lower, 'rsi_upper': rsi_upper,
            'macd_fast': macd_fast, 'macd_slow': macd_slow, 'macd_signal': macd_signal,
            'bb_window': bb_window, 'bb_std': bb_std,
            'bb_lower_pct': bb_lower_pct, 'bb_upper_pct': bb_upper_pct,
            'sl': sl, 'tp': tp, 'n_btc': n_btc,
        }

        result = backtest.run_backtest(data, params, initial_cash=1_000_000.0, COM=0.00125)
        score  = result['metrics'].get(metric)

        return score

    return objective