# portfolio_full_with_extras_html_sync.py
"""
Pipeline completo revisado:
- optimizaciones, MC/factor sims, blends/ensembles
- backtest (gráfico en $), drawdowns, rolling metrics
- stress tests (arreglados y robustos)
- rebalance alerts (opción de pasar cartera del usuario)
- generación PDF y HTML viewer (rutas relativas)
- arreglos: HTML format error, alerts layout, stress plots without lines
- estética mejorada básica en plots/tables
"""
import os
os.environ["MPLBACKEND"] = "Agg"
import gc
import json
import shutil
import webbrowser
import sys
import argparse
import yaml
import logging
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
from datetime import datetime, timedelta
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from PIL import Image
import plotly.graph_objects as go
from plotly.colors import qualitative
import warnings
warnings.filterwarnings("ignore")

# basic logger configuration; updated later once CLI/config parsed
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Try to import tkinter optionally (viewer)
try:
    import tkinter as tk
    from tkinter import simpledialog, messagebox, ttk
    from PIL import ImageTk
    try:
        _root = tk.Tk()
        _root.withdraw()
        ttk.Style(_root).theme_use('clam')
        _root.update()
        _root.destroy()
        TK_AVAILABLE = True
    except Exception:
        TK_AVAILABLE = False
except Exception:
    TK_AVAILABLE = False

# basic ttk theme initialization
def init_ttk_theme(master=None):
    if not TK_AVAILABLE:
        return
    try:
        style = ttk.Style(master) if master is not None else ttk.Style()
    except Exception:
        # fallback: try without master
        style = ttk.Style()
    try:
        style.theme_use('clam')
    except Exception:
        pass
    base_font = ('Segoe UI', 10)
    try:
        style.configure('.', font=base_font)
        style.configure('TButton', padding=6)
        style.configure('Primary.TButton', padding=6, background='#0d6efd', foreground='white')
        style.configure('TEntry', padding=4)
    except Exception:
        pass

# ---------- CONFIG ----------
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)
SNAPSHOT_DIR = os.path.join(OUT_DIR, "snapshots")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)
PORTFOLIOS_DIR = os.path.join(OUT_DIR, "portfolios")
os.makedirs(PORTFOLIOS_DIR, exist_ok=True)

TRADING_DAYS = 252
RISK_FREE_ANNUAL = 0.08
SEED = 42
np.random.seed(SEED)

USE_LEDOIT = True
MEAN_SHRINK_TAU = 0.6
MEAN_SHRINK_TARGET = 0.0
ALLOW_ZERO_WEIGHTS = True

# Número máximo de activos a seleccionar (modificable por el usuario)
MAX_SELECTED_ASSETS = 12
CAPITAL_TO_DEPLOY = 535.35

ROLLING_WINDOW = 63

# Rebalance (passive) configuration
REBALANCE_INTERVAL_DAYS = 30
REBALANCE_THRESHOLD = 0.05

# Matplotlib aesthetics
plt.style.use('seaborn-v0_8')
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "font.family": "sans-serif"
})
PALETTE = plt.get_cmap("tab10")

# default configuration values; may be overridden by YAML/CLI
DEFAULT_CONFIG = {
    'risk_free': RISK_FREE_ANNUAL,
    'max_selected_assets': MAX_SELECTED_ASSETS,
    'capital': CAPITAL_TO_DEPLOY,
    'seed': SEED,
    'benchmark': 'SPY',
    'costs_bps': 0.0,
    'turnover_max': None,
    'lam_tc': 0.0,
    'target_vol': None,
    'sim_mode': 'mc',
    'band_base': REBALANCE_THRESHOLD,
    'cash_available': None,
    'with_alpha': False,
    'rebalance_mode': 'passive',
    'colorblind_mode': True,
    'lot_size': 1,
    'min_trade_value': 0.0
}

def parse_args_and_config():
    parser = argparse.ArgumentParser(description='Portfolio pipeline')
    parser.add_argument('--config', type=str, help='config YAML file')
    parser.add_argument('--sim_mode', choices=['mc','bootstrap'])
    parser.add_argument('--lam_tc', type=float)
    parser.add_argument('--turnover_max', type=float)
    parser.add_argument('--target_vol', type=float)
    parser.add_argument('--band_base', type=float)
    parser.add_argument('--costs_bps', type=float)
    parser.add_argument('--with_alpha', action='store_true')
    parser.add_argument('--cash_available', type=float)
    parser.add_argument('--rebalance_mode', choices=['passive','active'])
    parser.add_argument('--colorblind_mode', type=lambda s: str(s).lower() in ['true','1','yes','on'])
    parser.add_argument('--lot_size', type=int)
    parser.add_argument('--min_trade_value', type=float)
    args, unknown = parser.parse_known_args()

    config = DEFAULT_CONFIG.copy()
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as fh:
            file_cfg = yaml.safe_load(fh) or {}
        config.update(file_cfg)
    for k in ['sim_mode','lam_tc','turnover_max','target_vol','band_base','cash_available',
              'rebalance_mode','colorblind_mode','lot_size','min_trade_value','costs_bps']:
        v = getattr(args, k)
        if v is not None:
            config[k] = v
    if args.with_alpha:
        config['with_alpha'] = True
    return config, args

# ---------- UTILS ----------
def download_prices(tickers, start, end, auto_adjust=False):
    chunk = 20
    all_frames = []
    failed = []
    seen = set()
    tickers = [t for t in tickers if not (t in seen or seen.add(t))]
    print(f"Descargando {len(tickers)} tickers en bloques...")
    for i in tqdm(range(0, len(tickers), chunk), desc="Descargando"):
        block = tickers[i:i+chunk]
        try:
            df = yf.download(block, start=start, end=end, progress=False, threads=True,
                             auto_adjust=auto_adjust, group_by='column')
            if isinstance(df.columns, pd.MultiIndex):
                if 'Adj Close' in df.columns.get_level_values(0):
                    df_block = df['Adj Close']
                elif 'Close' in df.columns.get_level_values(0):
                    df_block = df['Close']
                else:
                    df_block = df
            else:
                if 'Adj Close' in df.columns:
                    df_block = df['Adj Close']
                elif 'Close' in df.columns:
                    df_block = df['Close']
                else:
                    df_block = df
            if isinstance(df_block, pd.Series):
                df_block = df_block.to_frame()
            all_frames.append(df_block)
        except Exception as e:
            failed.extend(block)
            print(f"[warning] fallo descarga bloque {block}: {e}")
    if len(all_frames) == 0:
        raise ValueError("No se pudieron descargar datos para ningún ticker.")
    prices = pd.concat(all_frames, axis=1)
    prices = prices.loc[~prices.index.duplicated()]
    prices = prices.dropna(how='all').sort_index()
    prices = prices.dropna(axis=1, how='all')
    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        print(f"[info] Se omiten por descarga fallida o sin datos: {missing}")
    return prices

def compute_returns(prices):
    return prices.pct_change().dropna(how='all')

def annualize_returns(returns_daily):
    mean_daily = returns_daily.mean()
    vol_daily = returns_daily.std(ddof=1)
    return mean_daily * TRADING_DAYS, vol_daily * np.sqrt(TRADING_DAYS)

def compute_betas_historical(returns_assets, returns_bench):
    var_bench = returns_bench.var(ddof=1)
    betas = {}
    for col in returns_assets.columns:
        cov = returns_assets[col].cov(returns_bench)
        betas[col] = cov / var_bench if var_bench > 0 else np.nan
    return pd.Series(betas)


def _standardize(series):
    """Z-score normalization with NaN-safe handling."""
    s = series.astype(float)
    if s.std(ddof=0) == 0 or s.isna().all():
        return pd.Series(0.0, index=series.index)
    return ((s - s.mean()) / s.std(ddof=0)).fillna(0.0)


def _max_drawdown(prices):
    """Compute max drawdown for each column in price DataFrame."""
    mdd = {}
    for col in prices.columns:
        p = prices[col].dropna()
        if p.empty:
            mdd[col] = np.nan
            continue
        roll_max = p.cummax()
        drawdown = p / roll_max - 1.0
        mdd[col] = drawdown.min()
    return pd.Series(mdd)


def compute_asset_scores(price_df, returns_df):
    """Compute composite scores for assets using multiple factors."""
    mean_ann, vol_ann = annualize_returns(returns_df)
    sharpe = (mean_ann - RISK_FREE_ANNUAL) / vol_ann
    mdd = _max_drawdown(price_df)
    risk_score = _standardize(sharpe) + _standardize(-mdd)

    infos = {}
    for t in price_df.columns:
        try:
            infos[t] = yf.Ticker(t).info
        except Exception:
            infos[t] = {}
    avg_volume = pd.Series({t: infos[t].get('averageVolume', np.nan) for t in price_df.columns})
    market_cap = pd.Series({t: infos[t].get('marketCap', np.nan) for t in price_df.columns})
    liquidity_score = _standardize(avg_volume) + _standardize(market_cap)

    pe = pd.Series({t: infos[t].get('trailingPE', np.nan) for t in price_df.columns})
    pb = pd.Series({t: infos[t].get('priceToBook', np.nan) for t in price_df.columns})
    div_y = pd.Series({t: infos[t].get('dividendYield', np.nan) for t in price_df.columns})
    growth = pd.Series({t: infos[t].get('earningsGrowth', np.nan) for t in price_df.columns})
    fundamentals_score = _standardize(-pe) + _standardize(-pb) + _standardize(div_y) + _standardize(growth)

    corr_matrix = returns_df.corr().abs()
    avg_corr = corr_matrix.mean()
    divers_score = _standardize(-avg_corr)

    mom_12m = price_df.pct_change(252).iloc[-1]
    mom_3m = price_df.pct_change(63).iloc[-1]
    ma_ratio = price_df.iloc[-1] / price_df.rolling(200).mean().iloc[-1] - 1.0
    momentum_score = _standardize(mom_12m) + _standardize(mom_3m) + _standardize(ma_ratio)

    total_score = risk_score + liquidity_score + fundamentals_score + divers_score + momentum_score
    return total_score.sort_values(ascending=False)

# ---------- INPUT WINDOWS ----------
def prompt_initial_capital(default=100000.0):
    if TK_AVAILABLE:
        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes('-topmost', True)
        except Exception:
            pass
        value = simpledialog.askfloat(
            "Capital inicial",
            "Introduce el monto inicial:",
            initialvalue=default,
            parent=root,
        )
        root.destroy()
        return value if value is not None else default
    # fallback en consola
    try:
        txt = input(f"Introduce el monto inicial [{default}]: ")
        return float(txt) if txt else default
    except Exception:
        return default

def prompt_risk_free_rate(default=0.105):
    """Solicita la Tasa Libre de Riesgo (anual). Default 10.5% (México)."""
    if TK_AVAILABLE:
        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes('-topmost', True)
        except Exception:
            pass
        # Pedimos el valor como porcentaje (ej: 10.5) para que sea más amigable
        value = simpledialog.askfloat(
            "Tasa Libre de Riesgo",
            f"Tasa anual (%): (Ej: 10.5 para México)",
            initialvalue=default*100,
            parent=root,
        )
        root.destroy()
        return (value / 100.0) if value is not None else default
    
    # Fallback consola
    try:
        txt = input(f"Introduce Tasa Libre de Riesgo anual (decimal) [{default}]: ")
        return float(txt) if txt else default
    except Exception:
        return default

def prompt_max_assets(default=12):
    """Solicita al usuario el número máximo de activos a seleccionar."""
    if TK_AVAILABLE:
        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes('-topmost', True)
        except Exception:
            pass
        value = simpledialog.askinteger(
            "Activos máx.",
            "¿Cuántos activos deseas como máximo?",
            initialvalue=default,
            minvalue=1,
            parent=root,
        )
        root.destroy()
        return value if value is not None else default
    # fallback consola
    try:
        txt = input(f"¿Cuántos activos deseas como máximo? [{default}]: ")
        return int(txt) if txt else default
    except Exception:
        return default

def prompt_user_portfolio(scenarios=None, assets=None):
    """Solicita al usuario los datos de su cartera.

    Si `tkinter` está disponible se emplean ventanas para capturar los datos;
    en caso contrario se recurre a la consola."""

    if scenarios and assets and TK_AVAILABLE:
        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes('-topmost', True)
        except Exception:
            pass
        scenario_names = list(scenarios.keys())
        choice = {"value": None}

        win = tk.Toplevel(root)
        win.title("Escenario para rebalanceo")
        tk.Label(win, text="Selecciona escenario:").pack(padx=10, pady=5)
        lb = tk.Listbox(win, height=min(10, len(scenario_names)))
        for name in scenario_names:
            lb.insert(tk.END, name)
        lb.pack(padx=10, pady=5)

        def on_ok():
            sel = lb.curselection()
            if sel:
                choice["value"] = scenario_names[sel[0]]
            win.destroy()

        tk.Button(win, text="OK", command=on_ok).pack(pady=5)
        win.grab_set()
        root.wait_window(win)
        scenario = choice["value"]
        holdings = {}
        tickers = []
        if scenario:
            weights = scenarios.get(scenario, [])
            tickers = [a for a, w in zip(assets, weights) if float(w) > 0]
            for t in tickers:
                price = simpledialog.askfloat("Precio compra", f"Precio de compra para {t}:", parent=root, minvalue=0.0)
                qty = simpledialog.askfloat("Acciones", f"Acciones compradas de {t}:", parent=root, minvalue=0.0)
                if price is not None and qty is not None:
                    holdings[t] = {"BuyPrice": price, "Shares": qty}
        if not tickers:
            while True:
                t = simpledialog.askstring("Ticker", "Ticker (cancelar para terminar):", parent=root)
                if not t:
                    break
                t = t.strip().upper()
                price = simpledialog.askfloat("Precio compra", f"Precio de compra para {t}:", parent=root, minvalue=0.0)
                if price is None:
                    continue
                qty = simpledialog.askfloat("Acciones", f"Acciones compradas de {t}:", parent=root, minvalue=0.0)
                if qty is None:
                    continue
                holdings[t] = {"BuyPrice": price, "Shares": qty}
        root.destroy()
        df_hold = pd.DataFrame.from_dict(holdings, orient='index')
        return scenario, (df_hold if not df_hold.empty else None)

    # fallback en consola
    scenario = None
    if scenarios:
        print("Escenarios disponibles:", ", ".join(scenarios.keys()))
        try:
            scenario = input("Selecciona escenario: ") or None
        except Exception:
            scenario = None
        if scenario not in scenarios:
            scenario = None

    holdings = {}
    while True:
        try:
            ticker = input("Ticker (enter = terminar): ").strip().upper()
        except Exception:
            ticker = ''
        if not ticker:
            break
        try:
            price_txt = input(f"Precio de compra para {ticker}: ")
            buy_price = float(price_txt)
        except Exception:
            print("Precio inválido, intenta de nuevo.")
            continue
        try:
            shares_txt = input(f"Número de acciones de {ticker}: ")
            shares = float(shares_txt)
        except Exception:
            print("Cantidad inválida, intenta de nuevo.")
            continue
        holdings[ticker] = {"BuyPrice": buy_price, "Shares": shares}

    df_hold = pd.DataFrame.from_dict(holdings, orient='index')
    return scenario, (df_hold if not df_hold.empty else None)


def load_rebalance_snapshots(snapshot_dir=SNAPSHOT_DIR):
    snaps = {}
    if not os.path.isdir(snapshot_dir):
        return snaps
    for fn in os.listdir(snapshot_dir):
        if fn.endswith('.json'):
            path = os.path.join(snapshot_dir, fn)
            try:
                with open(path, 'r') as f:
                    data = json.load(f)
                name = data.get('scenario', os.path.splitext(fn)[0])
                df = pd.DataFrame(data.get('holdings', []))
                if 'Ticker' in df.columns:
                    df = df.set_index('Ticker')
                last_reb = data.get('last_rebalance_date') or data.get('timestamp')
                try:
                    last_reb = datetime.fromisoformat(last_reb).date()
                except Exception:
                    last_reb = None
                snaps[name] = {"df": df, "last_rebalance": last_reb, "path": path}
            except Exception as e:
                print(f"[warning] no se pudo cargar snapshot {fn}: {e}")
    return snaps


def save_rebalance_snapshot(scenarios, latest_prices, capital_total, snapshot_dir=SNAPSHOT_DIR):
    if not scenarios or latest_prices is None:
        return
    if TK_AVAILABLE:
        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes('-topmost', True)
        except Exception:
            pass
        scenario_names = list(scenarios.keys())
        choice = {"value": None}
        win = tk.Toplevel(root)
        win.title("Escenario a guardar")
        tk.Label(win, text="Selecciona escenario a guardar:").pack(padx=10, pady=5)
        lb = tk.Listbox(win, height=min(10, len(scenario_names)))
        for name in scenario_names:
            lb.insert(tk.END, name)
        lb.pack(padx=10, pady=5)

        def on_ok():
            sel = lb.curselection()
            if sel:
                choice["value"] = scenario_names[sel[0]]
            win.destroy()

        tk.Button(win, text="OK", command=on_ok).pack(pady=5)
        win.grab_set()
        root.wait_window(win)
        scenario = choice["value"]
        holdings = {}
        if scenario:
            w_ser = pd.Series(scenarios.get(scenario, []), index=latest_prices.index).fillna(0.0)
            tickers = [t for t, w in w_ser.items() if float(w) > 0]
            for t in tickers:
                default_price = float(latest_prices.get(t, 0.0))
                alloc = float(w_ser[t] * capital_total)
                default_sh = alloc / default_price if default_price > 0 else 0.0
                price = simpledialog.askfloat("Precio compra", f"Precio de compra para {t}:", parent=root, initialvalue=default_price, minvalue=0.0)
                qty = simpledialog.askfloat("Acciones compradas", f"Acciones compradas de {t}:", parent=root, initialvalue=default_sh, minvalue=0.0)
                if qty is not None and price is not None:
                    holdings[t] = {"TargetWeight": float(w_ser[t]),
                                   "BaseWeight": float(w_ser[t]),
                                   "BuyPrice": price,
                                   "InitialShares": qty}
        root.destroy()
    else:
        resp = input("¿Guardar snapshot de rebalanceo? [s/N]: ")
        if resp.lower() != 's':
            return
        print("Escenarios disponibles:", ", ".join(scenarios.keys()))
        scenario = input("Escenario a guardar: ")
        w_ser = pd.Series(scenarios.get(scenario, []), index=latest_prices.index).fillna(0.0)
        holdings = {}
        tickers = [t for t, w in w_ser.items() if float(w) > 0]
        for t in tickers:
            txt = input(f"Precio y acciones para {t} (price@shares): ")
            if txt and '@' in txt:
                pr, sh = txt.split('@', 1)
                try:
                    holdings[t] = {"TargetWeight": float(w_ser[t]),
                                   "BaseWeight": float(w_ser[t]),
                                   "BuyPrice": float(pr),
                                   "InitialShares": float(sh)}
                except ValueError:
                    pass
    if holdings:
        payload = {
            "scenario": scenario,
            "timestamp": datetime.now().isoformat(),
            "last_rebalance_date": datetime.now().date().isoformat(),
            "holdings": [{"Ticker": k, **v} for k, v in holdings.items()]
        }
        fname = f"{scenario}_{datetime.now().strftime('%Y%m%d')}.json"
        path = os.path.join(snapshot_dir, fname)
        with open(path, 'w') as f:
            json.dump(payload, f, indent=2)
        print(f"[saved] snapshot {path}")


def _portfolio_path(name):
    return os.path.join(PORTFOLIOS_DIR, f"{name}.json")


def list_portfolios():
    return sorted(os.path.splitext(f)[0] for f in os.listdir(PORTFOLIOS_DIR) if f.endswith('.json'))


def load_portfolio(name):
    path = _portfolio_path(name)
    with open(path, 'r') as f:
        return json.load(f)


def save_portfolio(obj):
    name = obj.get('name')
    if not name:
        raise ValueError('portfolio must have name')
    now = datetime.now().isoformat()
    obj.setdefault('created', now)
    obj['updated'] = now
    obj.setdefault('base_currency', 'USD')
    obj.setdefault('notes', '')
    obj.setdefault('last_rebalance_date', None)
    path = _portfolio_path(name)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)


def delete_portfolio(name):
    path = _portfolio_path(name)
    if os.path.exists(path):
        os.remove(path)


def _edit_portfolio_gui(port=None, assets_universe=None, parent=None):
    lots = [] if port is None else port.get('lots', [])[:]
    top = tk.Toplevel(parent) if parent else tk.Toplevel()
    # ensure ttk style binds to an existing toplevel/root to avoid extra blank windows
    try:
        init_ttk_theme(top)
    except Exception:
        pass
    top.title('Editar Cartera')
    ttk.Label(top, text='Nombre:').grid(row=0, column=0, sticky='w')
    name_var = tk.StringVar(value='' if port is None else port.get('name', ''))
    ttk.Entry(top, textvariable=name_var).grid(row=0, column=1, sticky='we')
    ttk.Label(top, text='Notas:').grid(row=1, column=0, sticky='w')
    notes_var = tk.StringVar(value='' if port is None else port.get('notes', ''))
    ttk.Entry(top, textvariable=notes_var).grid(row=1, column=1, sticky='we')

    tree = ttk.Treeview(top, columns=('Ticker','Date','Price','Shares'), show='headings', height=6)
    for col in ('Ticker','Date','Price','Shares'):
        tree.heading(col, text=col)
    tree.grid(row=2, column=0, columnspan=3, pady=5)
    for lot in lots:
        tree.insert('', 'end', values=(lot['Ticker'], lot['Date'], lot['BuyPrice'], lot['Shares']))

    def add_lot():
        t = simpledialog.askstring('Ticker', 'Ticker:', parent=top)
        if not t:
            return
        t = t.strip().upper()
        if assets_universe and t not in assets_universe:
            messagebox.showerror('Ticker', 'Ticker no válido')
            return
        d = simpledialog.askstring('Fecha', 'Fecha (YYYY-MM-DD):', parent=top, initialvalue=datetime.now().date().isoformat())
        if not d:
            return
        p = simpledialog.askfloat('Precio', 'Precio de compra:', parent=top, minvalue=0.0)
        if p is None:
            return
        s = simpledialog.askfloat('Acciones', 'Número de acciones:', parent=top, minvalue=0.0)
        if s is None:
            return
        lot = {"Ticker": t, "Date": d, "BuyPrice": float(p), "Shares": float(s)}
        lots.append(lot)
        tree.insert('', 'end', values=(t, d, p, s))

    def remove_lot():
        sel = tree.selection()
        if sel:
            idx = tree.index(sel[0])
            tree.delete(sel[0])
            lots.pop(idx)

    def on_ok():
        if not name_var.get():
            messagebox.showerror('Nombre', 'Nombre requerido')
            return
        top.portfolio = {
            'name': name_var.get(),
            'notes': notes_var.get(),
            'lots': lots,
            'base_currency': port.get('base_currency', 'USD') if port else 'USD',
            'created': port.get('created') if port else None,
            'last_rebalance_date': port.get('last_rebalance_date') if port else None
        }
        top.destroy()

    ttk.Button(top, text='Agregar lote', command=add_lot).grid(row=3, column=0, pady=5)
    ttk.Button(top, text='Quitar lote', command=remove_lot).grid(row=3, column=1, pady=5)
    ttk.Button(top, text='Guardar', style='Primary.TButton', command=on_ok).grid(row=4, column=0, columnspan=2, pady=5)
    ttk.Button(top, text='Cancelar', command=top.destroy).grid(row=4, column=2, pady=5)
    top.grab_set()
    top.mainloop()
    return getattr(top, 'portfolio', None)


def portfolio_manager_gui(assets_universe):
    if TK_AVAILABLE:
        root = tk.Tk()
        try:
            init_ttk_theme(root)
        except Exception:
            pass
        root.title('Administrador de Carteras')
        names = list_portfolios()

        def refresh():
            lb.delete(0, tk.END)
            for n in list_portfolios():
                lb.insert(tk.END, n)

        def on_load():
            s = lb.curselection()
            if s:
                name = lb.get(s[0])
                root.result = load_portfolio(name)
                root.destroy()

        def on_create():
            port = _edit_portfolio_gui(None, assets_universe, root)
            if port:
                save_portfolio(port)
                refresh()

        def on_edit():
            s = lb.curselection()
            if s:
                name = lb.get(s[0])
                port = load_portfolio(name)
                edited = _edit_portfolio_gui(port, assets_universe, root)
                if edited:
                    edited['name'] = name
                    save_portfolio(edited)
                    refresh()

        def on_delete():
            s = lb.curselection()
            if s:
                name = lb.get(s[0])
                if messagebox.askyesno('Confirmar', f'¿Borrar cartera {name}?'):
                    delete_portfolio(name)
                    refresh()

        def on_use_temp():
            port = _edit_portfolio_gui(None, assets_universe, root)
            root.result = port
            root.destroy()

        lb = tk.Listbox(root, width=40)
        lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        refresh()
        btn_frame = ttk.Frame(root)
        btn_frame.pack(side=tk.RIGHT, fill=tk.Y)
        ttk.Button(btn_frame, text='Crear', command=on_create).pack(fill=tk.X)
        ttk.Button(btn_frame, text='Cargar', command=on_load).pack(fill=tk.X)
        ttk.Button(btn_frame, text='Editar', command=on_edit).pack(fill=tk.X)
        ttk.Button(btn_frame, text='Borrar', command=on_delete).pack(fill=tk.X)
        ttk.Button(btn_frame, text='Usar sin guardar', command=on_use_temp).pack(fill=tk.X)
        ttk.Button(btn_frame, text='Cancelar', command=root.destroy).pack(fill=tk.X)
        root.result = None
        root.mainloop()
        return root.result

    # Console fallback
    while True:
        names = list_portfolios()
        print("Carteras disponibles:", ', '.join(names) if names else '(ninguna)')
        cmd = input("[c]rear, [l]oad, [e]dit, [d]elete, [u]sar sin guardar, [q]salir: ").strip().lower()
        if cmd == 'c':
            port = _edit_portfolio_console(None, assets_universe)
            if port:
                save_portfolio(port)
        elif cmd == 'l':
            name = input('Nombre de la cartera: ').strip()
            if name in names:
                return load_portfolio(name)
        elif cmd == 'e':
            name = input('Nombre a editar: ').strip()
            if name in names:
                port = load_portfolio(name)
                edited = _edit_portfolio_console(port, assets_universe)
                if edited:
                    edited['name'] = name
                    save_portfolio(edited)
        elif cmd == 'd':
            name = input('Nombre a borrar: ').strip()
            if name in names:
                delete_portfolio(name)
        elif cmd == 'u':
            return _edit_portfolio_console(None, assets_universe)
        elif cmd == 'q':
            return None


def _edit_portfolio_console(port=None, assets_universe=None):
    lots = [] if port is None else port.get('lots', [])[:]
    name = port.get('name') if port else input('Nombre de cartera: ')
    notes = port.get('notes', '') if port else input('Notas (opcional): ')
    while True:
        t = input('Ticker (enter para terminar): ').strip().upper()
        if not t:
            break
        d = input('Fecha (YYYY-MM-DD): ').strip() or datetime.now().date().isoformat()
        try:
            p = float(input('Precio de compra: '))
            s = float(input('Acciones: '))
        except Exception:
            print('Valores inválidos')
            continue
        lots.append({'Ticker': t, 'Date': d, 'BuyPrice': p, 'Shares': s})
    if not name:
        return None
    return {
        'name': name,
        'notes': notes,
        'lots': lots,
        'base_currency': port.get('base_currency', 'USD') if port else 'USD',
        'created': port.get('created') if port else None,
        'last_rebalance_date': port.get('last_rebalance_date') if port else None
    }

def prompt_portfolio_from_snapshot(snapshots, interval_days=REBALANCE_INTERVAL_DAYS):
    if not snapshots:
        return None, None, None, None
    scenario_names = list(snapshots.keys())
    if TK_AVAILABLE:
        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes('-topmost', True)
        except Exception:
            pass
        choice = {"value": None}
        win = tk.Toplevel(root)
        win.title("Escenario guardado")
        tk.Label(win, text="Selecciona escenario guardado:").pack(padx=10, pady=5)
        lb = tk.Listbox(win, height=min(10, len(scenario_names)))
        for name in scenario_names:
            lb.insert(tk.END, name)
        lb.pack(padx=10, pady=5)

        def on_ok():
            sel = lb.curselection()
            if sel:
                choice["value"] = scenario_names[sel[0]]
            win.destroy()

        tk.Button(win, text="OK", command=on_ok).pack(pady=5)
        win.grab_set()
        root.wait_window(win)
        scenario = choice["value"]
        holdings = {}
        weights = None
        last_reb = None
        if scenario:
            snap_info = snapshots[scenario]
            snap_df = snap_info.get('df', pd.DataFrame())
            weights = snap_df['BaseWeight'] if 'BaseWeight' in snap_df.columns else snap_df.get('TargetWeight')
            last_reb = snap_info.get('last_rebalance')
            today = datetime.now().date()
            need_update = True
            if last_reb and (today - last_reb).days < interval_days:
                resp = messagebox.askyesno(
                    "Actualizar valores?",
                    f"No ha pasado el intervalo de {interval_days} días.\n¿Actualizar valores de todas formas?",
                    parent=root,
                )
                need_update = resp
            if need_update:
                for t, row in snap_df.iterrows():
                    qty = simpledialog.askfloat(
                        "Cartera actual",
                        f"Acciones actuales para {t}:",
                        parent=root,
                        initialvalue=float(row.get('InitialShares', 0.0)),
                        minvalue=0.0,
                    )
                    holdings[t] = {
                        "Shares": qty if qty is not None else float(row.get('InitialShares', 0.0)),
                        "BuyPrice": float(row.get('BuyPrice', 0.0)),
                        "BaseWeight": float(row.get('BaseWeight', row.get('TargetWeight', 0.0)))
                    }
                # actualizar snapshot en disco
                new_df = pd.DataFrame.from_dict(holdings, orient='index').rename(columns={'Shares':'InitialShares'})
                new_df_reset = new_df.reset_index().rename(columns={'index':'Ticker'})
                payload = {
                    "scenario": scenario,
                    "timestamp": datetime.now().isoformat(),
                    "last_rebalance_date": today.isoformat(),
                    "holdings": new_df_reset.to_dict(orient='records')
                }
                try:
                    with open(snap_info.get('path'), 'w') as f:
                        json.dump(payload, f, indent=2)
                except Exception as e:
                    print(f"[warning] no se pudo actualizar snapshot: {e}")
                last_reb = today
            else:
                holdings = snap_df[['InitialShares','BuyPrice','BaseWeight']].rename(columns={'InitialShares':'Shares'}).to_dict('index')
        root.destroy()
    else:
        print("Escenarios guardados:", ", ".join(scenario_names))
        scenario = input("Selecciona escenario: ")
        snap_info = snapshots.get(scenario)
        snap_df = snap_info.get('df', pd.DataFrame()) if snap_info else pd.DataFrame()
        weights = snap_df['BaseWeight'] if 'BaseWeight' in snap_df.columns else snap_df.get('TargetWeight')
        last_reb = snap_info.get('last_rebalance') if snap_info else None
        holdings = {}
        today = datetime.now().date()
        need_update = True
        if last_reb and (today - last_reb).days < interval_days:
            resp = input(f"No ha pasado el intervalo de {interval_days} días. ¿Actualizar valores? [s/N]: ")
            need_update = (resp.lower() == 's')
        if need_update:
            for t, row in snap_df.iterrows():
                txt = input(f"Acciones actuales para {t} [{row.get('InitialShares', 0.0)}]: ")
                if txt:
                    try:
                        holdings[t] = {
                            "Shares": float(txt),
                            "BuyPrice": float(row.get('BuyPrice', 0.0)),
                            "BaseWeight": float(row.get('BaseWeight', row.get('TargetWeight', 0.0)))
                        }
                    except ValueError:
                        pass
                else:
                    holdings[t] = {
                        "Shares": float(row.get('InitialShares', 0.0)),
                        "BuyPrice": float(row.get('BuyPrice', 0.0)),
                        "BaseWeight": float(row.get('BaseWeight', row.get('TargetWeight', 0.0)))
                    }
            new_df = pd.DataFrame.from_dict(holdings, orient='index').rename(columns={'Shares':'InitialShares'})
            new_df_reset = new_df.reset_index().rename(columns={'index':'Ticker'})
            payload = {
                "scenario": scenario,
                "timestamp": datetime.now().isoformat(),
                "last_rebalance_date": today.isoformat(),
                "holdings": new_df_reset.to_dict(orient='records')
            }
            try:
                with open(snap_info.get('path'), 'w') as f:
                    json.dump(payload, f, indent=2)
            except Exception as e:
                print(f"[warning] no se pudo actualizar snapshot: {e}")
            last_reb = today
        else:
            holdings = snap_df[['InitialShares','BuyPrice','BaseWeight']].rename(columns={'InitialShares':'Shares'}).to_dict('index')
    df_hold = pd.DataFrame.from_dict(holdings, orient='index')
    return scenario, weights, (df_hold if not df_hold.empty else None), last_reb


def should_rebalance(last_rebalance_date, diff_items, threshold=REBALANCE_THRESHOLD,
                     interval_days=REBALANCE_INTERVAL_DAYS, today=None):
    """Decide if a passive rebalance alert should be emitted."""
    if today is None:
        today = datetime.now().date()
    if last_rebalance_date is None:
        return True
    if isinstance(last_rebalance_date, datetime):
        last_rebalance_date = last_rebalance_date.date()
    days_since = (today - last_rebalance_date).days
    max_dev = 0.0
    for item in diff_items:
        try:
            max_dev = max(max_dev, abs(float(item[3])))
        except Exception:
            continue
    if max_dev >= 1.5 * threshold:
        return True
    return (days_since >= interval_days) and (max_dev > threshold)


def plot_rebalance_bars(df, out_path, date_label='', threshold=REBALANCE_THRESHOLD,
                        show_all=False, subtitle=None):
    plt.figure(figsize=(8,4))
    data = df.copy() if df is not None else pd.DataFrame()
    if not show_all:
        data = data[data['Diff'].abs() > threshold]
    if data is None or data.empty:
        plt.text(0.5, 0.5, 'Sin datos para graficar',
                 ha='center', va='center', fontsize=12)
        plt.axis('off')
        plt.savefig(out_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"[saved] {out_path} (sin datos)")
        return
    labels = data['Scenario'] + ' | ' + data['Ticker'] if 'Scenario' in data.columns else data['Ticker']
    colors = [PALETTE(3) if v < 0 else PALETTE(0) for v in data['Diff']]
    plt.bar(labels, data['Diff'] * 100, color=colors)
    plt.axhline(0, color='black', linewidth=0.8)
    plt.ylabel('Actual - Target (%)')
    title = 'Alertas de Rebalanceo'
    if subtitle:
        title += f"\n{subtitle}"
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    for i, (lbl, diff_val) in enumerate(zip(labels, data['Diff'])):
        pct = diff_val * 100
        va = 'bottom' if pct >= 0 else 'top'
        plt.text(i, pct, f"{date_label}\n{pct:.2f}%", ha='center', va=va, fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[saved] {out_path}")

# ---------- REBALANCE SUGGESTIONS ----------
def compute_rebalance_actions(diff_items, latest_prices, holdings_df,
                              threshold=REBALANCE_THRESHOLD,
                              costs_bps=0.0, cash_available=None,
                              lot_size=1, min_trade_value=0.0):
    """Genera sugerencias de rebalanceo y un resumen de impacto con costos.

    Se redondean las acciones a enteros y se respeta un presupuesto de caja si
    se especifica ``cash_available``. ``costs_bps`` representa los costos de
    transacción por lado.
    """
    if holdings_df is None or latest_prices is None:
        return pd.DataFrame(), pd.Series(dtype=float)

    df_hold = holdings_df.copy()
    price_series = latest_prices.reindex(df_hold.index).astype(float)
    df_hold['Price'] = price_series
    df_hold['Value'] = df_hold['Shares'] * df_hold['Price']
    total_value = df_hold['Value'].sum()

    rows = []
    total_buy = 0.0
    total_sell = 0.0

    for t, tgt, act, diff in diff_items:
        price = float(latest_prices.get(t, np.nan))
        gap = float(tgt) - float(act)
        if abs(gap) <= threshold or np.isnan(price):
            action = 'HOLD'
            trade_shares = 0.0
            trade_value = 0.0
        elif gap > 0:
            action = 'BUY'
            trade_value = gap * total_value
            trade_shares = trade_value / price if price > 0 else 0.0
            total_buy += trade_value
        else:
            action = 'SELL'
            trade_value = -gap * total_value
            trade_shares = trade_value / price if price > 0 else 0.0
            total_sell += trade_value
        rows.append((t, tgt, act, diff, action, trade_value, trade_shares))

    df_actions = pd.DataFrame(rows, columns=['Ticker','Target','Actual','Diff',
                                             'Action','TradeValue','TradeShares'])
    if not df_actions.empty:
        df_actions['TradeShares'] = (df_actions['TradeShares'] / lot_size).round() * lot_size
        df_actions['TradeShares'] = df_actions['TradeShares'].astype(int)
        df_actions['TradeValue'] = df_actions['TradeShares'] * df_actions.apply(lambda r: latest_prices.get(r['Ticker'], np.nan), axis=1)
        df_actions.loc[df_actions['TradeValue'].abs() < min_trade_value, ['Action','TradeShares','TradeValue']] = ['HOLD',0.0,0.0]
        df_actions['TradeShares'] = df_actions['TradeShares'].astype(int)
        df_actions['Cost'] = df_actions['TradeValue'].abs() * (costs_bps/10000.0)

        if cash_available is not None:
            buys_mask = df_actions['Action'] == 'BUY'
            total_cost = (df_actions.loc[buys_mask, 'TradeValue'] + df_actions.loc[buys_mask, 'Cost']).sum()
            if total_cost > cash_available and total_cost > 0:
                scale = cash_available / total_cost
                df_actions.loc[buys_mask, 'TradeShares'] = (df_actions.loc[buys_mask, 'TradeShares'] * scale / lot_size).round() * lot_size
                df_actions.loc[buys_mask, 'TradeShares'] = df_actions.loc[buys_mask, 'TradeShares'].astype(int)
                df_actions['TradeValue'] = df_actions['TradeShares'] * df_actions.apply(lambda r: latest_prices.get(r['Ticker'], np.nan), axis=1)
                df_actions.loc[df_actions['TradeValue'].abs() < min_trade_value, ['Action','TradeShares','TradeValue']] = ['HOLD',0.0,0.0]
                df_actions['TradeShares'] = df_actions['TradeShares'].astype(int)
                df_actions['Cost'] = df_actions['TradeValue'].abs() * (costs_bps/10000.0)

    summary = pd.Series({'TotalBuy': df_actions.loc[df_actions['Action']=='BUY','TradeValue'].sum(),
                         'TotalSell': df_actions.loc[df_actions['Action']=='SELL','TradeValue'].sum(),
                         'Costs': df_actions['Cost'].sum(),
                         'NetImpact': df_actions.loc[df_actions['Action']=='BUY','TradeValue'].sum() -
                                      df_actions.loc[df_actions['Action']=='SELL','TradeValue'].sum() -
                                      df_actions['Cost'].sum()})
    return df_actions, summary

def compute_active_rebalance_trades(target_weights, latest_prices, holdings_df,
                                    costs_bps=0.0, cash_available=None, turnover_max=None,
                                    only_topup=False, allow_fractional_shares=False,
                                    min_trade_value=5.0):
    """Calcula operaciones para alcanzar ``target_weights``.

    Maneja modo solo-aportes ``only_topup`` y opción de fraccionales.
    Devuelve DataFrame con [Ticker, Action, Shares, TradeValue, Cost] y un resumen.
    """
    if holdings_df is None or latest_prices is None or target_weights is None:
        return pd.DataFrame(), pd.Series(dtype=float)

    # normalizar índices
    lp = pd.Series(latest_prices).dropna()
    lp.index = lp.index.map(lambda x: str(x).upper())
    tw = pd.Series(target_weights)
    tw.index = tw.index.map(lambda x: str(x).upper())
    hold = holdings_df.copy()
    hold.index = hold.index.map(lambda x: str(x).upper())

    universe = sorted(set(lp.index) | set(tw.index) | set(hold.index))
    lp = lp.reindex(universe).astype(float).fillna(0.0)
    hold = hold.reindex(universe).fillna(0.0)
    tw = tw.reindex(universe).fillna(0.0)

    # normalizar pesos objetivo
    tw_sum = tw.sum()
    if tw_sum != 0 and abs(tw_sum - 1) > 1e-6:
        tw = tw / tw_sum

    hold['Price'] = lp
    hold['Value'] = hold['Shares'] * hold['Price']
    total_equity = hold['Value'].sum()
    cash_add = max(0.0, cash_available or 0.0)

    target_values = tw * (total_equity + cash_add)
    current_values = hold['Value']
    delta_value = target_values - current_values
    if only_topup:
        delta_value = delta_value.clip(lower=0.0)

    diff_shares = np.where(lp > 0, delta_value / lp, 0.0)
    if allow_fractional_shares:
        diff_shares = np.round(diff_shares, 4)
    else:
        diff_shares = np.round(diff_shares)

    actions = np.where(diff_shares > 0, 'BUY', np.where(diff_shares < 0, 'SELL', 'HOLD'))
    df = pd.DataFrame({'Ticker': universe,
                       'Shares': diff_shares,
                       'Price': lp.values,
                       'Action': actions})

    df['TradeValue'] = df['Shares'] * df['Price']
    df['Cost'] = df['TradeValue'].abs() * (costs_bps / 10000.0)

    # presupuesto de compras
    buys = df[df['Action'] == 'BUY']
    total_need = (buys['TradeValue'] + buys['Cost']).sum()
    cash_av = cash_add if cash_available is not None else cash_available
    if cash_av is None:
        cash_av = cash_add
    if total_need > cash_av and total_need > 0:
        scale = cash_av / total_need
        df.loc[buys.index, 'Shares'] *= scale
        df.loc[buys.index, 'TradeValue'] = df.loc[buys.index, 'Shares'] * df.loc[buys.index, 'Price']
        df.loc[buys.index, 'Cost'] = df.loc[buys.index, 'TradeValue'].abs() * (costs_bps/10000.0)

    # redondeo final
    if allow_fractional_shares:
        df['Shares'] = df['Shares'].round(4)
    else:
        df['Shares'] = df['Shares'].round().astype(int)
    df['TradeValue'] = df['Shares'] * df['Price']
    df['Cost'] = df['TradeValue'].abs() * (costs_bps/10000.0)

    # turnover constraint solo si se permiten ventas
    if (not only_topup) and turnover_max is not None and turnover_max > 0 and total_equity > 0:
        turnover = df['TradeValue'].abs().sum() / total_equity
        if turnover > turnover_max and turnover > 0:
            scale = (turnover_max * total_equity) / df['TradeValue'].abs().sum()
            df['Shares'] *= scale
            if allow_fractional_shares:
                df['Shares'] = df['Shares'].round(4)
            else:
                df['Shares'] = df['Shares'].round().astype(int)
            df['TradeValue'] = df['Shares'] * df['Price']
            df['Cost'] = df['TradeValue'].abs() * (costs_bps/10000.0)

    # eliminar precios faltantes o muy pequeños
    df.loc[df['Price'] <= 0, ['Shares','TradeValue','Cost','Action']] = [0,0,0,'HOLD']

    # aplicar umbral mínimo
    df.loc[df['TradeValue'].abs() < min_trade_value, ['Action','Shares','TradeValue','Cost']] = ['HOLD',0,0,0]

    df = df[df['Shares'] != 0].copy()
    if only_topup:
        df = df[df['Shares'] > 0]

    summary = pd.Series({'TotalBuy': df.loc[df['Action']=='BUY','TradeValue'].sum(),
                         'TotalSell': df.loc[df['Action']=='SELL','TradeValue'].sum(),
                         'Costs': df['Cost'].sum()})
    summary['NetImpact'] = summary['TotalBuy'] - summary['TotalSell'] - summary['Costs']

    # métricas L1
    base_total = total_equity + cash_add
    current_w = current_values / base_total if base_total > 0 else pd.Series(0, index=universe)
    L1_before = (current_w - tw).abs().sum()
    trade_series = df.set_index('Ticker')['TradeValue'].reindex(universe).fillna(0.0)
    new_values = current_values + trade_series
    cash_unused = max(0.0, cash_add - (df[df['Action']=='BUY']['TradeValue'] + df[df['Action']=='BUY']['Cost']).sum())
    new_total = base_total - cash_unused
    new_w = new_values / new_total if new_total > 0 else pd.Series(0, index=universe)
    L1_after = (new_w - tw).abs().sum()
    summary['L1_before'] = L1_before
    summary['L1_after'] = L1_after
    summary['CashUnused'] = cash_unused

    return df[['Ticker','Action','Shares','TradeValue','Cost']], summary


def holdings_from_portfolio(portfolio, latest_prices):
    lots = portfolio.get('lots', []) if portfolio else []
    agg = {}
    for lot in lots:
        t = lot.get('Ticker')
        sh = float(lot.get('Shares', 0.0))
        bp = float(lot.get('BuyPrice', 0.0))
        if t not in agg:
            agg[t] = {'Shares': 0.0, 'BuyCost': 0.0}
        agg[t]['Shares'] += sh
        agg[t]['BuyCost'] += bp * sh
    rows = []
    for t, v in agg.items():
        shares = v['Shares']
        buy_avg = v['BuyCost']/shares if shares else 0.0
        price = float(latest_prices.get(t, 0.0))
        value = price * shares
        rows.append((t, buy_avg, shares, price, value))
    df = pd.DataFrame(rows, columns=['Ticker','BuyPriceAvg','Shares','Price','Value']).set_index('Ticker')
    return df


def actual_weights_from_holdings(df_hold):
    total = df_hold['Value'].sum()
    if total == 0:
        return pd.Series(0.0, index=df_hold.index)
    return df_hold['Value'] / total


def compute_active_rebalance_from_cash(actual_w, latest_prices, df_hold, add_cash,
                                       costs_bps=0.0, lot_size=1, min_trade_value=0.0):
    rows = []
    for t in df_hold.index:
        w = float(actual_w.get(t, 0.0))
        if w <= 0:
            continue
        price = float(latest_prices.get(t, 0.0))
        buy_val = add_cash * w
        shares = round(buy_val / price / lot_size) * lot_size if price > 0 else 0
        trade_val = shares * price
        if abs(trade_val) < min_trade_value or shares == 0:
            continue
        cost = abs(trade_val) * (costs_bps/10000.0)
        rows.append((t, 'BUY', shares, trade_val, cost))
    df = pd.DataFrame(rows, columns=['Ticker','Action','Shares','TradeValue','Cost'])
    summary = pd.Series({'TotalBuy': df['TradeValue'].sum(),
                         'TotalSell': 0.0,
                         'Costs': df['Cost'].sum(),
                         'NetImpact': df['TradeValue'].sum() + df['Cost'].sum()})
    return df, summary


def rebalance_flow(prices_assets, scenarios, initial_capital, config):
    alerts_path = ''
    alerts_plot_path = ''
    actions_path = ''
    impact_path = ''
    trades_csv = ''
    portfolio = portfolio_manager_gui(list(prices_assets.columns))
    if not portfolio:
        return alerts_path, alerts_plot_path, actions_path, impact_path, trades_csv
    latest_prices = prices_assets.ffill().iloc[-1]
    df_hold = holdings_from_portfolio(portfolio, latest_prices)
    actual_w = actual_weights_from_holdings(df_hold)
    # Preguntar si desea realizar rebalanceo antes de mostrar opciones
    try:
        if TK_AVAILABLE:
            root = tk.Tk(); root.withdraw()
            do_reb = messagebox.askyesno('Rebalanceo', '¿Deseas hacer rebalanceo ahora?', parent=root)
            root.destroy()
        else:
            do_reb = (input('¿Deseas hacer rebalanceo ahora? [s/N]: ').strip().lower() == 's')
        if not do_reb:
            return alerts_path, alerts_plot_path, actions_path, impact_path, trades_csv
    except Exception:
        pass
    if TK_AVAILABLE:
        root = tk.Tk(); root.withdraw()
        mode = simpledialog.askstring('Tipo de rebalanceo','Opciones: pasivo, activo_topup, activo_completo', parent=root, initialvalue='pasivo') or 'pasivo'
        root.destroy()
    else:
        mode = input('Tipo de rebalanceo [pasivo/activo_topup/activo_completo]: ') or 'pasivo'
    mode = mode.lower()

    min_tv = config.get('min_trade_value',0.0)

    if mode.startswith('pas'):
        if TK_AVAILABLE:
            root = tk.Tk(); root.withdraw()
            thresh = simpledialog.askfloat('Umbral','Umbral de desviación (0-1):', parent=root, initialvalue=config.get('band_base', REBALANCE_THRESHOLD))
            costs_bps = simpledialog.askfloat('Costos bps','Costos en bps:', parent=root, initialvalue=config.get('costs_bps',0.0))
            cash_available = simpledialog.askfloat('Caja disponible','Caja disponible (USD, opcional):', parent=root)
            turnover_max = simpledialog.askfloat('Turnover máx','Turnover máximo (0-1, opcional):', parent=root)
            root.destroy()
        else:
            thresh = float(input(f'Umbral de desviación [{REBALANCE_THRESHOLD}]: ') or REBALANCE_THRESHOLD)
            costs_bps = float(input(f'Costos bps [{config.get("costs_bps",0.0)}]: ') or config.get('costs_bps',0.0))
            cash_txt = input('Caja disponible (USD, opcional): ')
            cash_available = float(cash_txt) if cash_txt else None
            to_txt = input('Turnover máximo (0-1, opcional): ')
            turnover_max = float(to_txt) if to_txt else None
        lot_size = config.get('lot_size',1)
        scenario_names = list(scenarios.keys())
        if TK_AVAILABLE:
            root = tk.Tk(); root.withdraw()
            scen = simpledialog.askstring('Estrategia objetivo', f"Escenario ({', '.join(scenario_names)}):", parent=root, initialvalue=scenario_names[0] if scenario_names else '')
            root.destroy()
        else:
            scen = input(f"Estrategia objetivo ({', '.join(scenario_names)}): ")
        if scen in scenarios:
            raw = scenarios[scen]
            if isinstance(raw, (pd.Series, dict)):
                target_w = pd.Series(raw)
            else:
                target_w = pd.Series(raw, index=prices_assets.columns)
            target_w.index = target_w.index.astype(str)
            df_hold = df_hold.copy()
            df_hold.index = df_hold.index.astype(str)
            actual_w = actual_w.reindex(df_hold.index).fillna(0.0)
            latest_prices = latest_prices.reindex(df_hold.index).fillna(0.0)
            universe = sorted(set(target_w.index).union(df_hold.index))
            target_w = target_w.reindex(universe).fillna(0.0)
            df_hold = df_hold.reindex(universe).fillna(0.0)
            actual_w = actual_w.reindex(universe).fillna(0.0)
            latest_prices = latest_prices.reindex(universe).fillna(0.0)
            diff = actual_w - target_w
            diff_items = [(t, target_w[t], actual_w[t], diff[t]) for t in universe]
            if should_rebalance(portfolio.get('last_rebalance_date'), diff_items, threshold=thresh, interval_days=REBALANCE_INTERVAL_DAYS) and any(abs(d)>thresh for (_,_,_,d) in diff_items):
                alerts_df = pd.DataFrame(diff_items, columns=['Ticker','Target','Actual','Diff'])
                alerts_df_num = alerts_df.copy()
                diffs_csv = os.path.join(OUT_DIR, 'rebalance_diffs.csv')
                alerts_df.to_csv(diffs_csv, index=False)
                alerts_df['Target'] = alerts_df['Target'].apply(lambda x: f"{x:.1%}")
                alerts_df['Actual'] = alerts_df['Actual'].apply(lambda x: f"{x:.1%}")
                alerts_df['Diff'] = alerts_df_num['Diff'].apply(lambda x: f"{x:.1%}")
                alerts_df = alerts_df.set_index('Ticker')
                alerts_path = os.path.join(OUT_DIR, 'rebalance_alerts.png')
                save_table_image(alerts_df, alerts_path, title=f"Alertas de Rebalanceo (|diff| > {thresh*100:.0f}%)")
                alerts_plot_path = os.path.join(OUT_DIR, 'rebalance_alerts_selected.png')
                flagged = alerts_df_num[alerts_df_num['Diff'].abs() > thresh]
                if flagged.empty:
                    plot_rebalance_bars(alerts_df_num, alerts_plot_path, date_label=datetime.now().strftime('%Y-%m-%d'), threshold=thresh, show_all=True, subtitle='Sin alertas (todos ≤ umbral)')
                else:
                    plot_rebalance_bars(flagged, alerts_plot_path, date_label=datetime.now().strftime('%Y-%m-%d'), threshold=thresh, show_all=False)
                holdings_df = df_hold[['Shares']]
                actions_df, impact_summary = compute_rebalance_actions(diff_items, latest_prices, holdings_df, threshold=thresh, costs_bps=costs_bps, cash_available=cash_available, lot_size=lot_size, min_trade_value=min_tv)
                if not actions_df.empty:
                    actions_fmt = actions_df.copy()
                    for col in ['Target','Actual','Diff']:
                        actions_fmt[col] = actions_fmt[col].apply(lambda x: f"{x:.1%}")
                    actions_fmt['TradeValue'] = actions_fmt['TradeValue'].apply(lambda x: f"${x:,.2f}")
                    actions_fmt['TradeShares'] = actions_fmt['TradeShares'].apply(lambda x: f"{x:,.0f}")
                    actions_fmt = actions_fmt.set_index('Ticker')[['Action','Target','Actual','Diff','TradeShares','TradeValue']]
                    actions_path = os.path.join(OUT_DIR, 'rebalance_suggestions.png')
                    save_table_image(actions_fmt, actions_path, title='Sugerencias de Rebalanceo (Pasivo)')
                    trades_csv = os.path.join(OUT_DIR, 'trades.csv')
                    actions_df.to_csv(trades_csv, index=False)
                    impact_df = impact_summary.to_frame(name='Amount')
                    impact_df['Amount'] = impact_df['Amount'].apply(lambda x: f"${x:,.2f}")
                    impact_path = os.path.join(OUT_DIR, 'rebalance_impact_summary.png')
                    save_table_image(impact_df, impact_path, title='Resumen Impacto Rebalanceo')
                    if TK_AVAILABLE:
                        resp = messagebox.askyesno('Guardar', '¿Aplicar y guardar cambios en cartera?')
                    else:
                        resp = input('¿Aplicar y guardar cambios? [s/N]: ').lower()=='s'
                    if resp:
                        portfolio['last_rebalance_date'] = datetime.now().date().isoformat()
                        save_portfolio(portfolio)
            else:
                msg_df = pd.DataFrame({'Info':['No es necesario rebalancear']})
                alerts_path = os.path.join(OUT_DIR,'rebalance_suggestions.png')
                save_table_image(msg_df, alerts_path, title='Sugerencias de Rebalanceo')
    elif mode.startswith('activo_top'):
        if TK_AVAILABLE:
            root = tk.Tk(); root.withdraw()
            costs_bps = simpledialog.askfloat('Costos bps','Costos en bps:', parent=root, initialvalue=config.get('costs_bps',0.0))
            cash_add = simpledialog.askfloat('Capital adicional','Capital a aportar (USD):', parent=root, minvalue=0.0) or 0.0
            frac = messagebox.askyesno('Fraccionales','¿Permitir fraccionales?')
            root.destroy()
        else:
            costs_bps = float(input(f'Costos bps [{config.get("costs_bps",0.0)}]: ') or config.get('costs_bps',0.0))
            cash_add = float(input('Capital adicional (USD): ') or 0.0)
            frac = input('¿Permitir fraccionales? [s/N]: ').lower() == 's'
        df_actions, impact_summary = compute_active_rebalance_trades(actual_w, latest_prices, df_hold[['Shares']],
                                                                    costs_bps=costs_bps, cash_available=cash_add,
                                                                    turnover_max=None, only_topup=True,
                                                                    allow_fractional_shares=frac,
                                                                    min_trade_value=min_tv)
        if not df_actions.empty:
            actions_fmt = df_actions.copy()
            actions_fmt['TradeValue'] = actions_fmt['TradeValue'].apply(lambda x: f"${x:,.2f}")
            if frac:
                actions_fmt['Shares'] = actions_fmt['Shares'].apply(lambda x: f"{x:.4f}")
            else:
                actions_fmt['Shares'] = actions_fmt['Shares'].apply(lambda x: f"{x:,.0f}")
            actions_fmt['Cost'] = actions_fmt['Cost'].apply(lambda x: f"${x:,.2f}")
            actions_fmt = actions_fmt.set_index('Ticker')[['Action','Shares','TradeValue','Cost']]
            actions_path = os.path.join(OUT_DIR, 'rebalance_active_suggestions.png')
            save_table_image(actions_fmt, actions_path, title='Sugerencias de Rebalanceo (Activo Top-up)')
            trades_csv = os.path.join(OUT_DIR, 'trades_active.csv')
            df_actions.to_csv(trades_csv, index=False)
            impact_df = impact_summary.to_frame(name='Amount')
            impact_df['Amount'] = impact_df['Amount'].apply(lambda x: f"${x:,.2f}")
            if impact_summary.get('CashUnused',0) > 0:
                impact_df.loc['Efectivo no usado'] = f"${impact_summary['CashUnused']:,.2f}"
            impact_path = os.path.join(OUT_DIR, 'rebalance_impact_summary.png')
            save_table_image(impact_df, impact_path, title='Resumen Impacto Rebalanceo')
            # validaciones
            if (df_actions['Action']=='SELL').any() or impact_summary.get('TotalSell',0)!=0:
                raise ValueError('Top-up produjo ventas')
            if impact_summary.get('L1_after',0) >= impact_summary.get('L1_before',0):
                print('Advertencia: L1_after no mejora')
            if TK_AVAILABLE:
                resp = messagebox.askyesno('Guardar', '¿Guardar cambios en cartera?')
            else:
                resp = input('¿Guardar cambios en cartera? [s/N]: ').lower()=='s'
            if resp:
                for _, row in df_actions.iterrows():
                    portfolio.setdefault('lots', []).append({'Ticker': row['Ticker'], 'Date': datetime.now().date().isoformat(), 'BuyPrice': float(latest_prices.get(row['Ticker'],0.0)), 'Shares': row['Shares']})
                save_portfolio(portfolio)
    else:
        # activo completo
        if TK_AVAILABLE:
            root = tk.Tk(); root.withdraw()
            costs_bps = simpledialog.askfloat('Costos bps','Costos en bps:', parent=root, initialvalue=config.get('costs_bps',0.0))
            cash_available = simpledialog.askfloat('Caja disponible','Caja disponible (USD, opcional):', parent=root)
            turnover_max = simpledialog.askfloat('Turnover máx','Turnover máximo (0-1, opcional):', parent=root)
            scenario_names = list(scenarios.keys())
            scen = simpledialog.askstring('Estrategia objetivo', f"Escenario ({', '.join(scenario_names)}):", parent=root, initialvalue=scenario_names[0] if scenario_names else '')
            frac = messagebox.askyesno('Fraccionales','¿Permitir fraccionales?')
            root.destroy()
        else:
            costs_bps = float(input(f'Costos bps [{config.get("costs_bps",0.0)}]: ') or config.get('costs_bps',0.0))
            cash_txt = input('Caja disponible (USD, opcional): ')
            cash_available = float(cash_txt) if cash_txt else None
            to_txt = input('Turnover máximo (0-1, opcional): ')
            turnover_max = float(to_txt) if to_txt else None
            scenario_names = list(scenarios.keys())
            scen = input(f"Estrategia objetivo ({', '.join(scenario_names)}): ")
            frac = input('¿Permitir fraccionales? [s/N]: ').lower()=='s'
        if scen in scenarios:
            raw = scenarios[scen]
            if isinstance(raw, (pd.Series, dict)):
                target_w = pd.Series(raw)
            else:
                target_w = pd.Series(raw, index=prices_assets.columns)
            target_w.index = target_w.index.astype(str)
            df_hold = df_hold.copy(); df_hold.index = df_hold.index.astype(str)
            latest_prices = latest_prices.reindex(df_hold.index).fillna(0.0)
            actions_df, impact_summary = compute_active_rebalance_trades(target_w, latest_prices, df_hold[['Shares']],
                                                                        costs_bps=costs_bps, cash_available=cash_available,
                                                                        turnover_max=turnover_max, only_topup=False,
                                                                        allow_fractional_shares=frac,
                                                                        min_trade_value=min_tv)
            if not actions_df.empty:
                actions_fmt = actions_df.copy()
                actions_fmt['TradeValue'] = actions_fmt['TradeValue'].apply(lambda x: f"${x:,.2f}")
                if frac:
                    actions_fmt['Shares'] = actions_fmt['Shares'].apply(lambda x: f"{x:.4f}")
                else:
                    actions_fmt['Shares'] = actions_fmt['Shares'].apply(lambda x: f"{x:,.0f}")
                actions_fmt['Cost'] = actions_fmt['Cost'].apply(lambda x: f"${x:,.2f}")
                actions_fmt = actions_fmt.set_index('Ticker')[['Action','Shares','TradeValue','Cost']]
                actions_path = os.path.join(OUT_DIR, 'rebalance_active_suggestions.png')
                save_table_image(actions_fmt, actions_path, title='Sugerencias de Rebalanceo (Activo)')
                trades_csv = os.path.join(OUT_DIR, 'trades_active.csv')
                actions_df.to_csv(trades_csv, index=False)
                impact_df = impact_summary.to_frame(name='Amount')
                impact_df['Amount'] = impact_df['Amount'].apply(lambda x: f"${x:,.2f}")
                if impact_summary.get('CashUnused',0) > 0:
                    impact_df.loc['Efectivo no usado'] = f"${impact_summary['CashUnused']:,.2f}"
                impact_path = os.path.join(OUT_DIR, 'rebalance_impact_summary.png')
                save_table_image(impact_df, impact_path, title='Resumen Impacto Rebalanceo')
                if TK_AVAILABLE:
                    resp = messagebox.askyesno('Guardar', '¿Guardar cambios en cartera?')
                else:
                    resp = input('¿Guardar cambios en cartera? [s/N]: ').lower()=='s'
                if resp:
                    for _, row in actions_df.iterrows():
                        portfolio.setdefault('lots', []).append({'Ticker': row['Ticker'], 'Date': datetime.now().date().isoformat(), 'BuyPrice': float(latest_prices.get(row['Ticker'],0.0)), 'Shares': row['Shares']})
                    save_portfolio(portfolio)
    return alerts_path, alerts_plot_path, actions_path, impact_path, trades_csv

# ---------- OPTIMIZATION ----------
def obj_min_var(w, mean_returns, cov_matrix, betas=None):
    return float(w.T @ cov_matrix @ w)

def obj_max_sharpe(w, mean_returns, cov_matrix, betas=None):
    port_ret = float(np.dot(w, mean_returns))
    port_vol = float(np.sqrt(w.T @ cov_matrix @ w))
    if port_vol <= 0:
        return 1e10
    return - (port_ret - RISK_FREE_ANNUAL) / port_vol

def obj_min_beta(w, mean_returns, cov_matrix, betas):
    return float(np.dot(w, betas))

def obj_max_return(w, mean_returns, cov_matrix=None, betas=None):
    return - float(np.dot(w, mean_returns))

def obj_min_tracking_error(w, mean_returns, cov_matrix, bench_w):
    diff = w - bench_w
    return float(diff.T @ cov_matrix @ diff)

def risk_parity_weights(cov_matrix):
    n = cov_matrix.shape[0]
    x0 = np.repeat(1/n, n)
    bounds = tuple((0,1) for _ in range(n))
    constraints = ({'type':'eq', 'fun': lambda w: np.sum(w) - 1},)

    def obj(w):
        port_var = float(w.T @ cov_matrix @ w)
        mrc = cov_matrix @ w
        rc = w * mrc
        target = port_var / n
        return np.sum((rc - target)**2)

    res = minimize(obj, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    if not res.success:
        raise ValueError('Risk parity optimization failed: ' + res.message)
    return res.x

def scale_to_target_vol(w, cov_matrix, target_vol):
    current = np.sqrt(float(w.T @ cov_matrix @ w))
    if current == 0:
        return w
    return w * (target_vol / current)

def risk_contributions(cov_matrix, weights):
    cov_matrix = np.asarray(cov_matrix)
    weights = np.asarray(weights)
    port_var = float(weights.T @ cov_matrix @ weights)
    if port_var <= 0:
        return np.zeros_like(weights), np.zeros_like(weights)
    mrc = cov_matrix @ weights
    rc = weights * mrc
    rc_pct = rc / port_var
    return rc, rc_pct

def optimize_portfolio(mean_returns, cov_matrix, betas, bounds, constraints, objective_fn, x0=None,
                       lam_tc=0.0, w_prev=None, turnover_max=None):
    """Generic optimizer with optional turnover penalties/constraints."""
    mean_returns = np.asarray(mean_returns)
    cov_matrix = np.asarray(cov_matrix)
    betas = np.asarray(betas)
    n = len(mean_returns)
    if x0 is None:
        x0 = np.repeat(1/n, n)
    if w_prev is None:
        w_prev = np.zeros(n)

    args = (mean_returns, cov_matrix, betas)

    def obj(w):
        base = objective_fn(w, *args)
        penalty = lam_tc * np.sum(np.abs(w - w_prev)) if lam_tc > 0 else 0.0
        return base + penalty

    cons = list(constraints)
    if turnover_max is not None:
        cons.append({'type': 'ineq', 'fun': lambda w: turnover_max - np.sum(np.abs(w - w_prev))})

    res = minimize(obj, x0, method='SLSQP', bounds=bounds, constraints=cons)
    if not res.success:
        raise ValueError("Optimización falló: " + res.message)
    return res.x

# ---------- BACKTEST ----------
def backtest_fixed(prices, weights, initial_capital=100000):
    if not isinstance(weights, pd.Series):
        weights = pd.Series(weights, index=prices.columns)
    weights = weights.reindex(prices.columns).fillna(0.0)
    first_prices = prices.ffill().iloc[0]
    allocation = initial_capital * weights
    with np.errstate(divide='ignore', invalid='ignore'):
        shares = (allocation / first_prices).fillna(0).astype(int)
    cash = initial_capital - (shares * first_prices).sum()
    portfolio_vals = (prices * shares).sum(axis=1) + cash
    return portfolio_vals, shares

def make_backtest_plotly(backtest_results, bench_series, start_capital, out_html,
                         colorblind_mode=True):
    """Exporta un gráfico interactivo Plotly con acabados profesionales.

    Se normalizan todas las series a ``start_capital`` y se muestra un tooltip
    con el rendimiento acumulado.  El HTML resultante es autónomo y usa una
    paleta apta para daltónicos cuando ``colorblind_mode`` es ``True``.
    """

    palette = qualitative.Safe if colorblind_mode else qualitative.Plotly
    dashes = ['solid', 'dash', 'dot', 'dashdot', 'longdash', 'longdashdot']

    fig = go.Figure()
    for i, (name, series) in enumerate(backtest_results.items()):
        series = series.dropna()
        if series.empty:
            continue
        scaled = (series / series.iloc[0]) * start_capital
        cum_ret = scaled / start_capital - 1
        final_ret = cum_ret.iloc[-1]
        fig.add_trace(
            go.Scatter(
                x=scaled.index,
                y=scaled.values,
                name=f"{name} ({final_ret:+.2%})",
                line=dict(color=palette[i % len(palette)],
                          dash=dashes[i % len(dashes)], width=2.5),
                customdata=cum_ret.values,
                meta=name,
                hovertemplate=(
                    "Escenario: %{meta}<br>Fecha: %{x|%Y-%m-%d}<br>Valor: $%{y:,.2f}<br>"
                    "Rend. acumulado: %{customdata:.2%}<extra></extra>"
                ),
            )
        )

    if bench_series is not None and not bench_series.dropna().empty:
        bench = (bench_series / bench_series.iloc[0]) * start_capital
        cum_ret = bench / start_capital - 1
        final_ret = cum_ret.iloc[-1]
        fig.add_trace(
            go.Scatter(
                x=bench.index,
                y=bench.values,
                name=f"Benchmark ({final_ret:+.2%})",
                line=dict(color='black', dash='solid', width=2.5),
                customdata=cum_ret.values,
                meta='Benchmark',
                hovertemplate=(
                    "Escenario: %{meta}<br>Fecha: %{x|%Y-%m-%d}<br>Valor: $%{y:,.2f}<br>"
                    "Rend. acumulado: %{customdata:.2%}<extra></extra>"
                ),
            )
        )

    fig.update_traces(
        unselected=dict(marker=dict(opacity=0.35)),
        selector=dict(type="scatter")
    )

    fig.update_layout(
        hovermode='closest',
        legend=dict(orientation='h', x=0, y=1.05, itemclick='toggleothers',
                    font=dict(size=12)),
        xaxis=dict(showspikes=True, spikemode='across', spikesnap='cursor',
                   spikethickness=0.6, spikedash='dot'),
        margin=dict(l=40, r=20, t=50, b=40),
        font=dict(family="Inter, 'Segoe UI', Roboto, Arial, sans-serif", size=13),
        hoverlabel=dict(font=dict(family="Inter, 'Segoe UI', Roboto, Arial, sans-serif", size=12)),
        paper_bgcolor='white',
        plot_bgcolor='white',
        colorway=palette
    )

    fig.write_html(
        out_html,
        include_plotlyjs='inline',  # mete Plotly dentro del archivo, sin depender de internet
        full_html=True,
        config={
            'displaylogo': False,
            'modeBarButtonsToRemove': ['autoScale2d', 'lasso2d', 'select2d'],
            'modeBarButtonsToAdd': ['toggleSpikelines', 'toImage'],
            'scrollZoom': True,
        }
    )


    return out_html

# ---------- MONTECARLO / FACTOR ----------
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import Ridge

def mc_simulate_multivariate_logreturns(returns_all, days=TRADING_DAYS, n_sims=500, seed=None,
                                       mean_shrink_tau=MEAN_SHRINK_TAU,
                                       mean_shrink_target=MEAN_SHRINK_TARGET,
                                       use_ledoit=USE_LEDOIT):
    if seed is not None:
        np.random.seed(seed)
    logret = np.log1p(returns_all).dropna(how='all')
    mean_log = logret.mean().values
    if mean_shrink_tau is not None and mean_shrink_tau > 0:
        target = np.repeat(mean_shrink_target, len(mean_log))
        mean_log = (1 - mean_shrink_tau) * mean_log + mean_shrink_tau * target

    if use_ledoit:
        try:
            lw = LedoitWolf().fit(logret.values)
            cov_log = lw.covariance_
        except Exception:
            cov_log = logret.cov().values
    else:
        cov_log = logret.cov().values

    sims = np.random.multivariate_normal(mean_log, cov_log, size=(n_sims, days))
    return sims

def block_bootstrap_logreturns(returns_all, days=TRADING_DAYS, n_sims=500, block_size=21, seed=None):
    """Simulación alternativa mediante block bootstrap que preserva dependencias."""
    if seed is not None:
        np.random.seed(seed)
    logret = np.log1p(returns_all).dropna(how='all').values
    T, n = logret.shape
    blocks = []
    for _ in range(n_sims):
        idx = np.random.randint(0, T - block_size + 1, size=int(np.ceil(days / block_size)))
        samp = np.vstack([logret[i:i+block_size] for i in idx])
        samp = samp[:days]
        blocks.append(samp)
    return np.stack(blocks, axis=0)

def walk_forward_analysis(prices, scenario_fn, splits=3):
    """Simple walk-forward: entrena en cada fold y evalúa en el siguiente."""
    idx_splits = np.array_split(prices.index, splits)
    rows = []
    for i in range(len(idx_splits) - 1):
        train_idx = idx_splits[i]
        test_idx = idx_splits[i+1]
        train = prices.loc[train_idx]
        test = prices.loc[test_idx]
        r_train = compute_returns(train)
        mean_train, _ = annualize_returns(r_train)
        cov_train = r_train.cov() * TRADING_DAYS
        w = scenario_fn(mean_train.values, cov_train.values)
        port_train, _ = backtest_fixed(train, pd.Series(w, index=prices.columns))
        port_test, _ = backtest_fixed(test, pd.Series(w, index=prices.columns), initial_capital=port_train.iloc[-1])
        ret_is = port_train.pct_change().add(1).prod() - 1
        ret_oos = port_test.pct_change().add(1).prod() - 1
        vol_is = port_train.pct_change().std() * np.sqrt(TRADING_DAYS)
        vol_oos = port_test.pct_change().std() * np.sqrt(TRADING_DAYS)
        sharpe_is = (ret_is - RISK_FREE_ANNUAL) / vol_is if vol_is > 0 else np.nan
        sharpe_oos = (ret_oos - RISK_FREE_ANNUAL) / vol_oos if vol_oos > 0 else np.nan
        dd_is = compute_drawdown(port_train)[1] if len(port_train) > 0 else np.nan
        dd_oos = compute_drawdown(port_test)[1] if len(port_test) > 0 else np.nan
        rows.append({'Fold': i+1,
                     'IS_Return': ret_is,
                     'OOS_Return': ret_oos,
                     'IS_Vol': vol_is,
                     'OOS_Vol': vol_oos,
                     'IS_Sharpe': sharpe_is,
                     'OOS_Sharpe': sharpe_oos,
                     'IS_MaxDD': dd_is,
                     'OOS_MaxDD': dd_oos})
    return pd.DataFrame(rows)

def fit_factor_model_log(returns_assets, factors, alpha=1.0):
    lr_assets = np.log1p(returns_assets)
    lr_factors = np.log1p(factors)
    df = lr_assets.join(lr_factors, how='inner').dropna()
    if df.shape[0] < 2:
        raise ValueError("No hay suficientes observaciones alineadas entre activos y factores (log-returns).")
    Y = df[lr_assets.columns].values
    X = df[lr_factors.columns].values
    model = Ridge(alpha=alpha, fit_intercept=True)
    Bs = []
    for i in range(Y.shape[1]):
        model.fit(X, Y[:, i])
        Bs.append(model.coef_)
    B = np.vstack(Bs).T
    preds = X @ B
    residuals = Y - preds
    resid_df = pd.DataFrame(residuals, index=df.index, columns=lr_assets.columns)
    return pd.DataFrame(B, index=lr_factors.columns, columns=lr_assets.columns), resid_df

def simulate_via_factors_with_bench_log(B, resid_df, factors, days=TRADING_DAYS, n_sims=400, seed=None, use_ledoit=USE_LEDOIT):
    if seed is not None:
        np.random.seed(seed)
    lr_factors = np.log1p(factors).dropna()
    fac_mean = lr_factors.mean().values
    if use_ledoit:
        try:
            lw = LedoitWolf().fit(lr_factors.values)
            fac_cov = lw.covariance_
        except Exception:
            fac_cov = lr_factors.cov().values
    else:
        fac_cov = lr_factors.cov().values

    K = len(fac_mean)
    N = B.shape[1]
    resid_cov = resid_df.cov().values
    sims_assets = np.zeros((n_sims, days, N))
    sims_factor = np.zeros((n_sims, days))
    for s in tqdm(range(n_sims), desc="Simulando Factores"):
        factor_path = np.random.multivariate_normal(fac_mean, fac_cov, size=days)
        asset_path = factor_path @ B.values
        noise = np.random.multivariate_normal(np.zeros(N), resid_cov, size=days)
        sims_assets[s] = asset_path + noise
        sims_factor[s] = factor_path[:, 0] if K == 1 else factor_path.mean(axis=1)
    return sims_assets, sims_factor

# ---------- SUMMARIZERS ----------
def summarise_simulation_for_weights_from_log(sims_assets_log, sims_bench_log, weights, rf=RISK_FREE_ANNUAL):
    n_sims, days, N = sims_assets_log.shape
    w = np.asarray(weights)
    cum_assets = np.exp(np.sum(sims_assets_log, axis=1)) - 1.0
    cum_port = cum_assets @ w
    exp_return = float(np.mean(cum_port))
    vol = float(np.std(cum_port, ddof=1))
    sharpe = (exp_return - rf) / vol if vol > 0 else np.nan
    var95 = -np.percentile(cum_port, 5.0)
    betas = []
    for s in range(n_sims):
        asset_daily_log = sims_assets_log[s]
        bench_daily_log = sims_bench_log[s]
        port_daily_log = asset_daily_log @ w
        port_daily = np.expm1(port_daily_log)
        bench_daily = np.expm1(bench_daily_log)
        var_b = np.var(bench_daily, ddof=1)
        if var_b > 0:
            beta_s = np.cov(port_daily, bench_daily, ddof=1)[0,1] / var_b
        else:
            beta_s = np.nan
        betas.append(beta_s)
    beta_mean = float(np.nanmean(betas))
    return {"ExpReturn": exp_return, "AnnVol": vol, "Sharpe": sharpe, "BetaSim": beta_mean, "VaR95": var95}

def summarise_assets_from_sims_log(sims_assets_log, sims_bench_log, asset_names=None):
    n_sims, days, N = sims_assets_log.shape
    cum_assets = np.exp(np.sum(sims_assets_log, axis=1)) - 1.0
    exp_returns = np.mean(cum_assets, axis=0)
    vols = np.std(cum_assets, axis=0, ddof=1)
    betas = np.zeros(N)
    for j in tqdm(range(N), desc="Calculando Betas Simuladas"):
        betas_j = []
        for s in range(n_sims):
            asset_daily = sims_assets_log[s,:,j]
            bench_daily = sims_bench_log[s]
            var_b = np.var(np.expm1(bench_daily), ddof=1)
            if var_b > 0:
                betas_j.append(np.cov(np.expm1(asset_daily), np.expm1(bench_daily), ddof=1)[0,1] / var_b)
            else:
                betas_j.append(np.nan)
        betas[j] = np.nanmean(betas_j)
    df = pd.DataFrame({"BetaSim": betas, "ExpReturn_1Y": exp_returns, "Vol_1Y": vols})
    if asset_names is not None:
        df.index = asset_names
    return df

# ---------- ROBUST OPT OVER SIMS ----------
def optimize_over_simulations_local(sims_log, method='avg_sharpe', bounds=None, constraints=None, risk_free=RISK_FREE_ANNUAL):
    n_sims, days, n_assets = sims_log.shape
    print("Calculando estadísticas sobre simulaciones...")
    mean_annual_sims = np.array([np.expm1(sims_log[i]).mean(axis=0) * TRADING_DAYS for i in tqdm(range(n_sims), desc="Means")])
    cov_annual_sims = np.array([np.cov(np.expm1(sims_log[i]).T) * TRADING_DAYS for i in tqdm(range(n_sims), desc="Covs")])
    def objective_avg_sharpe(w):
        total = 0.0
        for i in range(n_sims):
            m = mean_annual_sims[i]; C = cov_annual_sims[i]
            ret = float(np.dot(w, m)); vol = float(np.sqrt(w.T @ C @ w))
            sharpe = (ret - risk_free) / vol if vol > 0 else -1e6
            total += -sharpe
        return total / n_sims
    def objective_worst_sharpe(w):
        worst = 1e9
        for i in range(n_sims):
            m = mean_annual_sims[i]; C = cov_annual_sims[i]
            ret = float(np.dot(w, m)); vol = float(np.sqrt(w.T @ C @ w))
            sharpe = (ret - risk_free) / vol if vol > 0 else -1e6
            worst = min(worst, -sharpe)
        return worst
    if bounds is None:
        bounds = tuple((0.0, 1.0) for _ in range(n_assets))
    if constraints is None:
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1},)
    if method == 'avg_sharpe':
        res = minimize(objective_avg_sharpe, x0=np.repeat(1/n_assets, n_assets), bounds=bounds, constraints=constraints, method='SLSQP')
    elif method == 'worst_sharpe':
        res = minimize(objective_worst_sharpe, x0=np.repeat(1/n_assets, n_assets), bounds=bounds, constraints=constraints, method='SLSQP')
    else:
        raise ValueError("Método desconocido")
    if not res.success:
        raise ValueError("Optimización robusta falló: " + res.message)
    return res.x

# ---------- RISK METRICS & PLOTS ----------
def var95_from_sims_horizon(sims_assets_log, weights, h_days=1):
    n_sims, days, N = sims_assets_log.shape
    w = np.asarray(weights)
    samples = []
    if h_days > days:
        raise ValueError("h_days cannot be larger than simulation days")
    for s in range(n_sims):
        for start in range(0, days - h_days + 1):
            seg = sims_assets_log[s, start:start + h_days, :]
            cum_assets = np.exp(np.sum(seg, axis=0)) - 1.0
            cum_port = float(cum_assets @ w)
            samples.append(cum_port)
    samples = np.array(samples)
    return float(-np.percentile(samples, 5.0))

def cvar95_from_sims_horizon(sims_assets_log, weights, h_days=1):
    n_sims, days, N = sims_assets_log.shape
    w = np.asarray(weights)
    samples = []
    if h_days > days:
        raise ValueError("h_days cannot be larger than simulation days")
    for s in range(n_sims):
        for start in range(0, days - h_days + 1):
            seg = sims_assets_log[s, start:start + h_days, :]
            cum_assets = np.exp(np.sum(seg, axis=0)) - 1.0
            cum_port = float(cum_assets @ w)
            samples.append(cum_port)
    samples = np.array(samples)
    thresh = np.percentile(samples, 5.0)
    tail = samples[samples <= thresh]
    if tail.size == 0:
        return 0.0
    return float(-np.mean(tail))

def get_cum_port_fullyear_per_sim(sims_assets_log, weights):
    n_sims, days, N = sims_assets_log.shape
    w = np.asarray(weights)
    cum = np.exp(np.sum(sims_assets_log, axis=1)) - 1.0
    cum_port = cum @ w
    return cum_port

# ---------- TABLE -> IMAGE ----------
def save_table_image(df, filename, title=None, round_digits=6):
    try:
        df_print = df.copy()
        numeric = df_print.select_dtypes(include=[np.number]).columns
        for c in numeric:
            df_print[c] = df_print[c].round(round_digits)
    except Exception:
        df_print = df.copy()
    if df_print is None or len(df_print) == 0:
        fig, ax = plt.subplots(figsize=(8, 1.2))
        ax.axis('off')
        ax.text(0.5, 0.5, "No hay datos", ha='center', va='center', fontsize=12)
        if title:
            plt.title(title, fontsize=12)
        plt.tight_layout()
        fig.savefig(filename, dpi=200, bbox_inches='tight')
        plt.close(fig)
        print(f"[saved-empty] {filename}")
        return
    df_print = df_print.fillna("").astype(str)
    rows, cols = df_print.shape if hasattr(df_print, "shape") else (0,0)
    height = max(1.2, 0.35*rows + 1.0)
    width = max(8, 0.9*cols + 2)
    fig, ax = plt.subplots(figsize=(width, height))
    ax.axis('off')
    col_labels = [str(c).replace(' | ', '\n').replace(' |', '\n').replace('| ', '\n') for c in df_print.columns]
    the_table = ax.table(cellText=df_print.values,
                     colLabels=col_labels,
                     rowLabels=df_print.index,
                     loc='center',
                     cellLoc='left')
    the_table.auto_set_font_size(False)
    the_table.set_fontsize(9)
    the_table.scale(1, 1.4)
    if title:
        plt.title(title, fontsize=12)
    plt.tight_layout()
    fig.savefig(filename, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[saved] {filename}")

def images_to_pdf(image_paths, out_pdf_path):
    try:
        imgs = []
        for p in image_paths:
            im = Image.open(p).convert('RGB')
            imgs.append(im)
        if len(imgs) == 0:
            print("No images to save into PDF.")
            return
        first, rest = imgs[0], imgs[1:]
        first.save(out_pdf_path, save_all=True, append_images=rest)
        try:
            first.close()
        except:
            pass
        for r in rest:
            try:
                r.close()
            except:
                pass
        print(f"[saved] {out_pdf_path}")
    except Exception as e:
        print("No se pudo generar PDF. Instala pillow o revisa imágenes. Error:", e)

# ---------- COMPACT WEIGHTS ----------
def make_alloc_pairs(weights, assets, fmt="{:.4f}"):
    pairs = []
    for a, w in zip(assets, weights):
        try:
            wv = float(w)
        except Exception:
            continue
        if wv <= 0 or np.isnan(wv):
            continue
        pairs.append((a, wv))
    pairs_sorted = sorted(pairs, key=lambda x: x[1], reverse=True)
    return [f"{t}:{fmt.format(w)}" for t, w in pairs_sorted]

def save_weights_compact_table(scenarios_dict, assets, filename, title=None):
    rows = {}
    max_allocs = 0
    for name, w in scenarios_dict.items():
        w_arr = np.asarray(w)
        pairs = make_alloc_pairs(w_arr, assets, fmt="{:.4f}")
        max_allocs = max(max_allocs, len(pairs))
        rows[name] = pairs
    col_names = [f"Alloc_{i+1}" for i in range(max_allocs)]
    data = []
    idx = []
    for name, pairs in rows.items():
        row = pairs + [""]*(max_allocs - len(pairs))
        data.append(row)
        idx.append(name)
    df = pd.DataFrame(data, index=idx, columns=col_names)
    save_table_image(df, filename, title=title)

# ---------- INVESTMENT PLANS ----------
def compute_investment_plan_per_scenario(scenarios_dict, latest_prices, capital_total):
    results = {}
    assets = list(latest_prices.index)
    for name, w in scenarios_dict.items():
        w_arr = np.asarray(w)
        if len(w_arr) != len(assets):
            if isinstance(w, pd.Series):
                w_ser = w.reindex(assets).fillna(0.0)
                w_arr = w_ser.values
            else:
                tmp = np.zeros(len(assets))
                tmp[:min(len(tmp), len(w_arr))] = w_arr[:min(len(tmp), len(w_arr))]
                w_arr = tmp
        alloc_amounts = w_arr * capital_total
        prices = latest_prices.values
        with np.errstate(divide='ignore', invalid='ignore'):
            shares_frac = np.where(prices > 0, alloc_amounts / prices, 0.0)
        df = pd.DataFrame({
            "Ticker": assets,
            "Weight": w_arr,
            "Price": prices,
            "Allocation_$": alloc_amounts,
            "Shares_frac": shares_frac
        }).set_index("Ticker")
        df_display = df.copy()
        df_display = df_display.rename(columns={
            "Weight": "Weight\n(%)",
            "Price": "Price\n($)",
            "Allocation_$": "Allocation\n($)",
            "Shares_frac": "Shares\n(frac)"
        })
        results[name] = df_display
    return results

def save_investment_tables(investment_dict, out_dir):
    paths = []
    all_tables = []
    for name, df in investment_dict.items():
        df_nonzero = df[df.iloc[:,2].abs() > 1e-8].copy() if df.shape[1] >= 3 else df.copy()
        fname = os.path.join(out_dir, f"investments_{name}.png")
        if df_nonzero.shape[0] == 0:
            df_msg = pd.DataFrame({"Info": [f"No allocation >0 for {name}"]})
            save_table_image(df_msg, fname, title=f"Inversiones - {name}")
            paths.append(fname)
            continue
        save_table_image(df_nonzero, fname, title=f"Inversiones - {name}")
        paths.append(fname)
        display_df = df_nonzero.copy()
        display_df.columns = pd.MultiIndex.from_product([[name], display_df.columns])
        all_tables.append(display_df)
    if len(all_tables) > 0:
        consolidated = pd.concat(all_tables, axis=1)
        consolidated.columns = ["\n".join(map(str,c)) for c in consolidated.columns]
        cons_path = os.path.join(out_dir, "investments_all_scenarios.png")
        save_table_image(consolidated, cons_path, title="Inversiones - Todos los Escenarios")
        paths.append(cons_path)
    return paths

# ---------- DRAWDOWN & ROLLING ----------
def compute_drawdown(series):
    wealth = series.dropna()
    if wealth.empty:
        return wealth * 0, 0.0, None, None, None
    running_max = wealth.cummax()
    dd = (wealth - running_max) / running_max
    max_dd = dd.min()
    trough_idx = dd.idxmin()
    peak_idx = wealth.loc[:trough_idx].idxmax()
    recovery_days = None
    post_trough = wealth.loc[trough_idx:]
    target = running_max.loc[peak_idx]
    recover_dates = post_trough[post_trough >= target]
    if not recover_dates.empty:
        recovery_days = (recover_dates.index[0] - trough_idx).days
    return dd, float(-max_dd), peak_idx, trough_idx, recovery_days

def plot_drawdown_summary(series, out_path, title="Drawdown Summary"):
    dd, max_dd, peak, trough, rec_days = compute_drawdown(series)
    plt.figure(figsize=(10,5))
    plt.plot(dd.index, dd.values, label='Drawdown', color=PALETTE(2))
    plt.fill_between(dd.index, dd.values, 0, alpha=0.15)
    plt.title(f"{title}\nMax Drawdown: {max_dd:.2%} | Peak: {peak} | Trough: {trough} | Recovery days: {rec_days}")
    plt.ylabel("Drawdown")
    plt.xlabel("Date")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[saved] {out_path}")

def plot_rolling_metrics(returns_series, out_path_sharpe, out_path_vol, window=ROLLING_WINDOW):
    r = returns_series.dropna()
    if r.empty:
        plt.figure(figsize=(10,4))
        plt.text(0.5,0.5,"No data for rolling metrics", ha='center')
        plt.axis('off')
        plt.savefig(out_path_sharpe, dpi=150, bbox_inches='tight'); plt.close()
        plt.savefig(out_path_vol, dpi=150, bbox_inches='tight'); plt.close()
        return None, None
    roll_mean = r.rolling(window).mean() * TRADING_DAYS
    roll_std = r.rolling(window).std(ddof=1) * np.sqrt(TRADING_DAYS)
    roll_sharpe = (roll_mean - RISK_FREE_ANNUAL) / roll_std
    plt.figure(figsize=(10,4))
    roll_sharpe.plot(color=PALETTE(0))
    plt.title(f"Rolling Sharpe ({window} days)")
    plt.axhline(0, color='k', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_path_sharpe, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[saved] {out_path_sharpe}")
    plt.figure(figsize=(10,4))
    roll_std.plot(color=PALETTE(1))
    plt.title(f"Rolling Volatility ({window} days)")
    plt.tight_layout()
    plt.savefig(out_path_vol, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[saved] {out_path_vol}")
    return out_path_sharpe, out_path_vol

# ---------- CORRELATION HEATMAP ----------
def save_correlation_heatmap(returns_df, out_path, title="Correlation Heatmap"):
    corr = returns_df.corr()
    plt.figure(figsize=(8,6))
    im = plt.imshow(corr, interpolation='nearest', cmap='RdBu_r', vmin=-1, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    ticks = np.arange(len(corr.columns))
    plt.xticks(ticks, corr.columns, rotation=90, fontsize=8)
    plt.yticks(ticks, corr.columns, fontsize=8)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[saved] {out_path}")

# ---------- STRESS TEST (ROBUSTO, REEMPLAZAR LA VERSIÓN ANTERIOR) ----------
def stress_test_crises(prices_for_stress,
                       scenarios,
                       crises=None,
                       out_dir=OUT_DIR,
                       reference_index=None,
                       palette=None,
                       debug_save=True):
    """
    prices_for_stress: DataFrame (precio ajustado) con history larga.
    scenarios: dict name->weights (arrays, pd.Series, or dict)
    reference_index: optional list/Index with the 'canonical' asset order used when weights are plain arrays.
                     En main llamar con reference_index=prices_assets.columns
    palette: optional list of colors (tu PALETTE). Si None, usa tab10.
    debug_save: guarda CSV con info cuando no se ploteó nada (útil para debugging).
    """
    import csv

    if crises is None:
        crises = [
            ("GlobalFinCrisis_2008", "2007-10-01", "2009-12-31"),
            ("COVID_2020", "2020-02-01", "2020-06-30")
        ]

    if palette is None:
        try:
            palette = list(plt.get_cmap('tab10').colors)
        except Exception:
            palette = None

    saved = []
    idx_min = prices_for_stress.index.min()
    idx_max = prices_for_stress.index.max()

    for label, sstart, send in crises:
        sstart_dt = pd.to_datetime(sstart)
        send_dt = pd.to_datetime(send)
        clipped_start = max(sstart_dt, idx_min)
        clipped_end = min(send_dt, idx_max)

        if clipped_start > clipped_end:
            print(f"[stress_test] sin intersección de datos para {label} (requested {sstart}–{send}), saltando.")
            continue

        window = prices_for_stress.loc[
            (prices_for_stress.index >= clipped_start) &
            (prices_for_stress.index <= clipped_end)
        ].copy()

        # Rellenos robustos para evitar NaNs al calcular returns
        # forward-fill luego back-fill para mitigar huecos al inicio/medio
        window = window.ffill().bfill()
        # Tras rellenar, si todo sigue NaN o muy pocas filas: saltar
        if window.dropna(how='all').shape[0] < 2 or window.shape[1] == 0:
            print(f"[stress_test] ventana muy pequeña o sin columnas útiles para {label}, saltando.")
            continue

        # calculo retornos; si hay NaNs en algunas celdas, los reemplazamos por 0
        ret = compute_returns(window).fillna(0.0)

        print(f"[stress_test] {label}: periodo usado {clipped_start.date()}–{clipped_end.date()} | ret.shape = {ret.shape}")

        plt.figure(figsize=(10, 6))
        plotted = False
        debug_rows = []

        for idx, (name, w) in enumerate(scenarios.items()):
            # normalizar/transformar weights a pd.Series indexadas en ret.columns
            w_ser = None
            try:
                # Si es pd.Series o dict o tiene index -> reindex sobre ret.columns
                if isinstance(w, pd.Series):
                    w_ser = w.reindex(ret.columns).fillna(0.0).astype(float)
                elif isinstance(w, dict):
                    w_ser = pd.Series(w).reindex(ret.columns).fillna(0.0).astype(float)
                else:
                    w_arr = np.asarray(w)
                    # caso ideal: longitudes iguales -> asumimos mismo orden que ret.columns
                    if len(w_arr) == len(ret.columns):
                        w_ser = pd.Series(w_arr, index=ret.columns).astype(float)
                    else:
                        # si se pasó referencia, la usamos para mapear
                        if reference_index is not None:
                            try:
                                ref = pd.Index(reference_index)
                                # crear serie con index reference_index y luego reindex a ret.columns
                                w_ser = pd.Series(w_arr, index=ref[:len(w_arr)]).reindex(ret.columns).fillna(0.0).astype(float)
                            except Exception:
                                w_ser = pd.Series(0.0, index=ret.columns)
                        else:
                            # fallback: ceros (no podemos inferir correspondencia)
                            w_ser = pd.Series(0.0, index=ret.columns)
            except Exception as e:
                print(f"[stress_test] error al construir serie de pesos para {name}: {e}")
                w_ser = pd.Series(0.0, index=ret.columns)

            # Diagnostics
            nonzero = int((w_ser.abs() > 0).sum())
            total_w = float(w_ser.sum() if hasattr(w_ser, 'sum') else np.nansum(w_ser))
            debug_rows.append({
                "scenario": name,
                "weights_len": len(w_ser),
                "nonzero_weights": nonzero,
                "sum_weights": total_w
            })

            if nonzero == 0:
                print(f"[stress_test] escenario '{name}' sin pesos válidos en {label}, saltando.")
                continue

            # compute portfolio daily returns robustly (ret has shape [T, M])
            port_daily = ret.values @ w_ser.values  # shape (T,)
            # si hay NaNs en port_daily, sustituir por 0 para poder plotear
            port_daily = np.nan_to_num(port_daily, nan=0.0, posinf=0.0, neginf=0.0)
            cum = (1 + pd.Series(port_daily, index=ret.index)).cumprod() - 1.0

            # plot visible y con estilo
            color = palette[idx % len(palette)] if palette is not None else None
            plt.plot(cum.index, cum.values, label=f"{name}", linewidth=2.0, alpha=0.9, color=color)
            plotted = True

        # si no se ploteó nada, escribir un mensaje grande en la figura y salvar
        if not plotted:
            plt.text(0.5, 0.5, "No hay escenarios válidos para plotear en este periodo.",
                     ha='center', va='center', transform=plt.gca().transAxes, fontsize=12, color='red')
            print(f"[stress_test] no se graficó ningún escenario para {label}.")

        plt.title(f'Stress Test: {label} ({clipped_start.date()} → {clipped_end.date()})')
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small')
        plt.xlabel("Date")
        plt.ylabel("Cumulative return")
        plt.grid(axis='y', alpha=0.15)
        plt.tight_layout()

        path = os.path.join(out_dir, f'stress_{label}.png')
        plt.savefig(path, dpi=200, bbox_inches='tight')
        plt.close()

        saved.append(path)
        print(f"[saved] {path}")

        # guardar debug CSV si se solicitó (útil para ver por qué un scenario quedó out)
        if debug_save:
            dbg_path = os.path.join(out_dir, f'stress_debug_{label}.csv')
            with open(dbg_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=["scenario", "weights_len", "nonzero_weights", "sum_weights"])
                writer.writeheader()
                for r in debug_rows:
                    writer.writerow(r)
            print(f"[saved] debug CSV: {dbg_path}")

    return saved

# ---------- REBALANCE ALERTS (mejorada) ----------
def parse_user_portfolio_csv(path):
    df = pd.read_csv(path).copy()
    df.columns = [c.strip() for c in df.columns]
    if 'Ticker' not in df.columns:
        df.rename(columns={df.columns[0]: 'Ticker'}, inplace=True)
    if 'Shares' not in df.columns and 'Value' not in df.columns and len(df.columns) >= 2:
        df.rename(columns={df.columns[1]: 'Shares'}, inplace=True)
    df = df.set_index('Ticker')
    if 'Shares' in df.columns:
        df['Shares'] = pd.to_numeric(df['Shares'], errors='coerce').fillna(0.0)
    if 'Value' in df.columns:
        df['Value'] = pd.to_numeric(df['Value'], errors='coerce').fillna(0.0)
    return df

def rebalance_alerts_for_scenarios(scenarios, prices, initial_capital=100000,
                                   threshold=0.05, user_portfolio=None, target_w=None):
    alerts = {}
    diffs_all = {}
    latest_map = {}
    holdings_map = {}
    prices_ffill = prices.ffill()
    base_latest = prices_ffill.iloc[-1]
    if target_w is not None and (scenarios is None or len(scenarios)==0):
        scenarios = {'target': target_w}
    if scenarios is None:
        scenarios = {'actual': None}
    for name, w in scenarios.items():
        universe = set(prices.columns)
        if w is not None:
            w_keys = list(w.keys()) if isinstance(w, dict) else list(prices.columns)
            universe.update(w_keys)
        if user_portfolio is not None:
            if isinstance(user_portfolio, pd.DataFrame):
                universe.update(user_portfolio.index)
            elif isinstance(user_portfolio, dict):
                universe.update(user_portfolio.keys())
            elif isinstance(user_portfolio, str):
                try:
                    tmp_df = parse_user_portfolio_csv(user_portfolio)
                    universe.update(tmp_df.index)
                except Exception:
                    pass
        universe = sorted(universe)
        latest = base_latest.reindex(universe)
        missing = latest[latest.isna()].index.tolist()
        if missing:
            print(f"[warning] precios faltantes: {missing}, omitiendo en {name}")
            latest = latest.dropna()
            universe = list(latest.index)
        if w is not None:
            try:
                w_ser = pd.Series(w, index=universe).fillna(0.0)
            except Exception:
                w_ser = pd.Series(0.0, index=universe)
        else:
            w_ser = pd.Series(0.0, index=universe)
        if user_portfolio is None:
            pf_vals, shares = backtest_fixed(prices[universe], w_ser, initial_capital)
            values = shares * latest
            total = values.sum()
            actual_w = values / total if total != 0 else pd.Series(0.0, index=universe)
            holdings_map[name] = pd.DataFrame({'Shares': shares}, index=universe)
        else:
            if isinstance(user_portfolio, pd.DataFrame):
                up_df = user_portfolio.copy()
            elif isinstance(user_portfolio, dict):
                up_df = pd.DataFrame.from_dict(user_portfolio, orient='index')
            else:
                up_df = pd.DataFrame()
            up_df = up_df.reindex(universe).fillna(0.0)
            if 'Value' in up_df.columns:
                values_series = up_df['Value']
            else:
                shares_ser = up_df.get('Shares', pd.Series(0.0, index=universe)).reindex(universe).fillna(0.0)
                values_series = shares_ser * latest
            total = values_series.sum()
            actual_w = values_series / total if total != 0 else pd.Series(0.0, index=universe)
            holdings_map[name] = pd.DataFrame({'Shares': up_df.get('Shares', pd.Series(0.0, index=universe)).reindex(universe).fillna(0.0),
                                               'BuyPrice': up_df.get('BuyPrice', pd.Series(0.0, index=universe)).reindex(universe).fillna(0.0)})
        diffs = actual_w - w_ser
        flagged = []
        all_items = []
        for t in universe:
            d = float(diffs.get(t, 0.0))
            tgt = float(w_ser.get(t, 0.0))
            act = float(actual_w.get(t, 0.0))
            all_items.append((t, tgt, act, d))
            if abs(d) > threshold:
                flagged.append((t, tgt, act, d))
        alerts[name] = flagged
        diffs_all[name] = all_items
        latest_map[name] = latest
    return alerts, diffs_all, latest_map, holdings_map

# ---------- HTML VIEWER (sin error de format con llaves) ----------
def generate_html_viewer(image_index, out_html_path, out_dir=OUT_DIR, run_id=None):
    """
    Genera un visor HTML profesional (Modo Oscuro, Búsqueda, Animaciones, Modal con navegación).
    """
    # 1. Preparar lista de items
    page_items = []
    
    # Procesar el índice de imágenes
    for cat, items in (image_index or {}).items():
        for title, path in items:
            if not path or not os.path.exists(path):
                continue
            
            filename = os.path.basename(path)
            dest = os.path.join(out_dir, filename)
            
            # Copiar archivo si no está en la carpeta de salida
            try:
                if os.path.abspath(path) != os.path.abspath(dest):
                    shutil.copy2(path, dest)
            except Exception:
                pass 
            
            # Determinar ruta relativa
            rel_path = filename if os.path.exists(dest) else os.path.relpath(path, start=out_dir)

            is_html = filename.lower().endswith('.html')
            is_csv = filename.lower().endswith('.csv')
            
            item = {
                "category": cat,
                "title": title,
                "path": rel_path,
                "filename": filename,
                "type": "html" if is_html else ("csv" if is_csv else "img")
            }
            page_items.append(item)

    run_id = run_id or datetime.now().strftime('%Y-%m-%d %H:%M')

    # 2. Plantilla HTML Moderna
    html_template = """<!DOCTYPE html>
<html lang="es" data-theme="light">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Portfolio Analytics</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            /* Light Theme */
            --bg-body: #f3f4f6;
            --bg-sidebar: #ffffff;
            --bg-card: #ffffff;
            --text-main: #1f2937;
            --text-muted: #6b7280;
            --primary: #4f46e5; /* Indigo */
            --primary-hover: #4338ca;
            --border: #e5e7eb;
            --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
            --shadow-hover: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        }

        [data-theme="dark"] {
            /* Dark Theme */
            --bg-body: #0f172a;
            --bg-sidebar: #1e293b;
            --bg-card: #1e293b;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --primary: #6366f1;
            --primary-hover: #818cf8;
            --border: #334155;
            --shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
            --shadow-hover: 0 10px 15px -3px rgba(0, 0, 0, 0.4);
        }

        * { box-sizing: border-box; transition: background-color 0.2s, border-color 0.2s, color 0.1s; }
        
        body {
            font-family: 'Inter', sans-serif;
            margin: 0;
            background-color: var(--bg-body);
            color: var(--text-main);
            display: flex;
            height: 100vh;
            overflow: hidden;
        }

        /* --- Sidebar --- */
        aside {
            width: 280px;
            background-color: var(--bg-sidebar);
            border-right: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            flex-shrink: 0;
            z-index: 20;
        }

        .brand {
            padding: 1.5rem;
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .brand h1 {
            font-size: 1.25rem;
            font-weight: 700;
            color: var(--text-main);
            margin: 0;
            letter-spacing: -0.025em;
        }
        .brand-icon {
            width: 32px; height: 32px;
            background: linear-gradient(135deg, var(--primary), #ec4899);
            border-radius: 8px;
            display: flex; align-items: center; justify-content: center;
            color: white; font-weight: bold; font-size: 18px;
        }

        .nav-scroll {
            flex: 1;
            overflow-y: auto;
            padding: 1rem;
        }
        
        .nav-label {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: var(--text-muted);
            margin-bottom: 0.75rem;
            margin-top: 1.5rem;
            font-weight: 600;
            padding-left: 0.75rem;
        }

        .nav-btn {
            display: flex;
            justify-content: space-between;
            align-items: center;
            width: 100%;
            padding: 0.75rem 1rem;
            margin-bottom: 0.25rem;
            border: 1px solid transparent;
            background: transparent;
            color: var(--text-muted);
            cursor: pointer;
            border-radius: 0.5rem;
            font-size: 0.95rem;
            font-weight: 500;
            text-align: left;
            transition: all 0.2s ease;
        }
        
        .nav-btn:hover {
            background-color: var(--bg-body);
            color: var(--text-main);
        }
        
        .nav-btn.active {
            background-color: var(--primary);
            color: #ffffff;
            box-shadow: 0 4px 12px rgba(79, 70, 229, 0.3);
        }
        .nav-btn.active .badge {
            background-color: rgba(255,255,255,0.2);
            color: white;
        }

        .badge {
            font-size: 0.75rem;
            background-color: var(--bg-body);
            padding: 2px 8px;
            border-radius: 12px;
            color: var(--text-muted);
            transition: all 0.2s;
        }

        .sidebar-footer {
            padding: 1rem;
            border-top: 1px solid var(--border);
            font-size: 0.8rem;
            color: var(--text-muted);
            text-align: center;
        }

        /* --- Main Content --- */
        main {
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
            position: relative;
        }

        header {
            background-color: var(--bg-sidebar);
            border-bottom: 1px solid var(--border);
            padding: 1rem 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .header-left h2 {
            margin: 0;
            font-size: 1.5rem;
            font-weight: 700;
        }
        .run-info {
            font-size: 0.85rem;
            color: var(--text-muted);
            margin-top: 4px;
        }

        .header-controls {
            display: flex;
            gap: 1rem;
            align-items: center;
        }

        /* Search Input */
        .search-wrapper {
            position: relative;
        }
        .search-input {
            background: var(--bg-body);
            border: 1px solid var(--border);
            padding: 0.5rem 1rem 0.5rem 2.2rem;
            border-radius: 8px;
            color: var(--text-main);
            width: 240px;
            font-size: 0.9rem;
            outline: none;
        }
        .search-input:focus {
            border-color: var(--primary);
            box-shadow: 0 0 0 2px rgba(79, 70, 229, 0.1);
        }
        .search-icon {
            position: absolute;
            left: 0.75rem;
            top: 50%;
            transform: translateY(-50%);
            color: var(--text-muted);
            font-size: 0.9rem;
        }

        /* Theme Toggle */
        .theme-toggle {
            background: none;
            border: 1px solid var(--border);
            padding: 0.5rem;
            border-radius: 8px;
            cursor: pointer;
            color: var(--text-main);
            display: flex; align-items: center; justify-content: center;
        }
        .theme-toggle:hover {
            background: var(--bg-body);
        }

        /* Content Area */
        .content-scroll {
            flex: 1;
            overflow-y: auto;
            padding: 2rem;
            background-color: var(--bg-body);
        }

        /* Grid System */
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(320px, 1fr));
            gap: 1.5rem;
            padding-bottom: 3rem;
        }

        /* Card Design */
        .card {
            background-color: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            overflow: hidden;
            box-shadow: var(--shadow);
            display: flex;
            flex-direction: column;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
            animation: fadeIn 0.4s ease-out backwards;
        }
        
        .card:hover {
            transform: translateY(-4px);
            box-shadow: var(--shadow-hover);
            border-color: var(--primary);
        }

        .card-preview {
            height: 220px;
            background-color: var(--bg-body);
            display: flex;
            align-items: center;
            justify-content: center;
            border-bottom: 1px solid var(--border);
            cursor: pointer;
            overflow: hidden;
            position: relative;
        }
        
        .card-preview img {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
            transition: transform 0.3s;
        }
        .card:hover .card-preview img {
            transform: scale(1.03);
        }

        .card-icon { font-size: 3rem; opacity: 0.5; }

        .card-body {
            padding: 1.25rem;
            display: flex;
            flex-direction: column;
            flex: 1;
        }
        
        .card-title {
            font-weight: 600;
            font-size: 1rem;
            margin-bottom: 0.5rem;
            color: var(--text-main);
            line-height: 1.4;
        }
        
        .card-meta {
            font-size: 0.75rem;
            color: var(--text-muted);
            margin-bottom: 1rem;
            flex: 1;
        }

        .card-actions {
            display: flex;
            gap: 0.5rem;
            margin-top: auto;
        }
        
        .btn {
            flex: 1;
            text-align: center;
            padding: 0.5rem;
            border-radius: 6px;
            font-size: 0.85rem;
            font-weight: 500;
            text-decoration: none;
            transition: all 0.2s;
            cursor: pointer;
        }
        
        .btn-primary {
            background-color: var(--primary);
            color: white;
            border: 1px solid var(--primary);
        }
        .btn-primary:hover {
            background-color: var(--primary-hover);
        }
        
        .btn-outline {
            background-color: transparent;
            border: 1px solid var(--border);
            color: var(--text-main);
        }
        .btn-outline:hover {
            border-color: var(--text-muted);
            background-color: var(--bg-body);
        }

        /* Iframe */
        .iframe-wrap { width: 100%; height: 100%; }
        iframe { width: 100%; height: 100%; border: none; }

        /* Modal */
        .modal {
            display: none;
            position: fixed;
            inset: 0;
            z-index: 1000;
            background: rgba(0, 0, 0, 0.85);
            backdrop-filter: blur(4px);
            align-items: center;
            justify-content: center;
            opacity: 0;
            transition: opacity 0.3s;
        }
        .modal.show {
            display: flex;
            opacity: 1;
        }
        
        .modal-content {
            max-width: 90vw;
            max-height: 90vh;
            border-radius: 8px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
        }

        .modal-btn {
            position: absolute;
            background: rgba(255,255,255,0.1);
            border: none;
            color: white;
            padding: 1rem;
            cursor: pointer;
            border-radius: 50%;
            font-size: 1.5rem;
            display: flex; align-items: center; justify-content: center;
            transition: background 0.2s;
        }
        .modal-btn:hover { background: rgba(255,255,255,0.2); }
        .modal-close { top: 1.5rem; right: 1.5rem; }
        .modal-prev { left: 1.5rem; }
        .modal-next { right: 1.5rem; }

        .empty-state {
            grid-column: 1 / -1;
            text-align: center;
            padding: 4rem;
            color: var(--text-muted);
            font-style: italic;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        /* Mobile responsive */
        @media (max-width: 768px) {
            body { flex-direction: column; overflow: auto; }
            aside { width: 100%; height: auto; border-right: none; border-bottom: 1px solid var(--border); }
            .nav-scroll { display: none; } /* Simplified for mobile */
            main { height: auto; }
        }
    </style>
</head>
<body>

<aside>
    <div class="brand">
        <div class="brand-icon">P</div>
        <h1>Portfolio Maker</h1>
    </div>
    
    <div class="nav-scroll" id="nav-container">
        </div>

    <div class="sidebar-footer">
        Run ID: __RUN_ID__
    </div>
</aside>

<main>
    <header>
        <div class="header-left">
            <h2 id="page-title">Dashboard</h2>
            <div class="run-info" id="item-count">Cargando...</div>
        </div>
        
        <div class="header-controls">
            <div class="search-wrapper">
                <span class="search-icon">🔍</span>
                <input type="text" id="search-input" class="search-input" placeholder="Buscar gráfico...">
            </div>
            
            <button class="theme-toggle" id="theme-btn" title="Cambiar tema">
                🌓
            </button>
        </div>
    </header>

    <div class="content-scroll">
        <div class="grid" id="grid">
            </div>
    </div>
</main>

<div class="modal" id="modal">
    <button class="modal-btn modal-close" onclick="closeModal()">✕</button>
    <button class="modal-btn modal-prev" onclick="navModal(-1)">❮</button>
    <button class="modal-btn modal-next" onclick="navModal(1)">❯</button>
    <img id="modal-img" class="modal-content" src="" alt="Vista previa">
</div>

<script>
    // --- Data ---
    const ITEMS = __ITEMS_JSON__;
    
    // --- State ---
    let state = {
        category: 'Todas',
        search: '',
        currentImageIndex: -1,
        filteredItems: []
    };

    // --- DOM Elements ---
    const grid = document.getElementById('grid');
    const navContainer = document.getElementById('nav-container');
    const pageTitle = document.getElementById('page-title');
    const itemCount = document.getElementById('item-count');
    const searchInput = document.getElementById('search-input');
    const modal = document.getElementById('modal');
    const modalImg = document.getElementById('modal-img');

    // --- Initialization ---
    function init() {
        // Theme init
        const savedTheme = localStorage.getItem('theme') || 'light';
        document.documentElement.setAttribute('data-theme', savedTheme);
        
        // Render Nav
        renderNav();
        
        // Initial Filter & Render
        filterAndRender();

        // Event Listeners
        searchInput.addEventListener('input', (e) => {
            state.search = e.target.value.toLowerCase();
            filterAndRender();
        });

        document.getElementById('theme-btn').addEventListener('click', toggleTheme);
        
        // Keyboard Nav
        document.addEventListener('keydown', (e) => {
            if (modal.classList.contains('show')) {
                if (e.key === 'Escape') closeModal();
                if (e.key === 'ArrowLeft') navModal(-1);
                if (e.key === 'ArrowRight') navModal(1);
            }
        });
    }

    // --- Core Logic ---
    function renderNav() {
        // Count categories
        const counts = {};
        ITEMS.forEach(i => counts[i.category] = (counts[i.category] || 0) + 1);
        
        // Always add 'Todas'
        let navHTML = `
            <div class="nav-label">General</div>
            <button class="nav-btn ${state.category === 'Todas' ? 'active' : ''}" onclick="setCategory('Todas')">
                <span>📑 Todas</span>
                <span class="badge">${ITEMS.length}</span>
            </button>
        `;

        // Sort keys
        const cats = Object.keys(counts).sort();
        
        // Separate logic can be added here to group specific categories if needed
        // For now, simple list
        if(cats.length > 0) navHTML += `<div class="nav-label">Categorías</div>`;
        
        cats.forEach(cat => {
            let icon = '📊';
            if(cat.includes('Riesgo')) icon = '⚠️';
            if(cat.includes('Rebalanceo')) icon = '⚖️';
            if(cat.includes('Comparativa')) icon = '📈';
            
            navHTML += `
                <button class="nav-btn ${state.category === cat ? 'active' : ''}" onclick="setCategory('${cat}')">
                    <span>${icon} ${cat}</span>
                    <span class="badge">${counts[cat]}</span>
                </button>
            `;
        });

        navContainer.innerHTML = navHTML;
    }

    window.setCategory = function(cat) {
        state.category = cat;
        renderNav(); // Re-render to update active class
        filterAndRender();
        // Reset scroll
        document.querySelector('.content-scroll').scrollTop = 0;
    }

    function filterAndRender() {
        // Filter logic
        state.filteredItems = ITEMS.filter(item => {
            const matchCat = state.category === 'Todas' || item.category === state.category;
            const matchSearch = item.title.toLowerCase().includes(state.search) || 
                                item.filename.toLowerCase().includes(state.search);
            return matchCat && matchSearch;
        });

        // Update Header
        pageTitle.innerText = state.category;
        itemCount.innerText = `${state.filteredItems.length} elementos encontrados`;

        // Clear Grid
        grid.innerHTML = '';

        if (state.filteredItems.length === 0) {
            grid.innerHTML = '<div class="empty-state">No se encontraron resultados para tu búsqueda.</div>';
            return;
        }

        // Render Cards with Animation Stagger
        state.filteredItems.forEach((item, index) => {
            const card = document.createElement('div');
            card.className = 'card';
            card.style.animationDelay = `${index * 50}ms`; // Stagger effect
            
            let preview = '';
            if (item.type === 'img') {
                preview = `<div class="card-preview" onclick="openModal('${item.path}')">
                             <img src="${item.path}" alt="${item.title}" loading="lazy">
                           </div>`;
            } else if (item.type === 'html') {
                 preview = `<div class="card-preview">
                             <div class="iframe-wrap"><iframe src="${item.path}" scrolling="no"></iframe></div>
                           </div>`;
            } else {
                preview = `<div class="card-preview"><span class="card-icon">📄</span></div>`;
            }

            card.innerHTML = `
                ${preview}
                <div class="card-body">
                    <div class="card-title" title="${item.title}">${item.title}</div>
                    <div class="card-meta">${item.filename}</div>
                    <div class="card-actions">
                        <a href="${item.path}" target="_blank" class="btn btn-primary">Abrir</a>
                        <a href="${item.path}" download class="btn btn-outline">Descargar</a>
                    </div>
                </div>
            `;
            grid.appendChild(card);
        });
    }

    // --- Modal Logic ---
    window.openModal = function(path) {
        // Find index in filtered items if possible, else global
        const idx = state.filteredItems.findIndex(i => i.path === path);
        if (idx !== -1) {
            state.currentImageIndex = idx;
            modalImg.src = path;
            modal.classList.add('show');
        }
    }

    window.closeModal = function() {
        modal.classList.remove('show');
        setTimeout(() => modalImg.src = '', 300);
    }

    window.navModal = function(dir) {
        // Filter only images from current view
        const imagesOnly = state.filteredItems.filter(i => i.type === 'img');
        if (imagesOnly.length === 0) return;
        
        // Find current image in this sub-list
        const currentSrc = modalImg.getAttribute('src');
        let localIdx = imagesOnly.findIndex(i => i.path === currentSrc);
        
        if (localIdx === -1) localIdx = 0;
        
        let newIdx = localIdx + dir;
        if (newIdx < 0) newIdx = imagesOnly.length - 1;
        if (newIdx >= imagesOnly.length) newIdx = 0;
        
        const nextItem = imagesOnly[newIdx];
        
        // Fade out slightly
        modalImg.style.opacity = 0.5;
        setTimeout(() => {
            modalImg.src = nextItem.path;
            modalImg.style.opacity = 1;
        }, 150);
    }
    
    // Close modal on background click
    modal.addEventListener('click', (e) => {
        if (e.target === modal) closeModal();
    });

    // --- Utilities ---
    function toggleTheme() {
        const current = document.documentElement.getAttribute('data-theme');
        const next = current === 'dark' ? 'light' : 'dark';
        document.documentElement.setAttribute('data-theme', next);
        localStorage.setItem('theme', next);
    }

    // Run
    init();

</script>
</body>
</html>
"""
    json_data = json.dumps(page_items, ensure_ascii=False)
    html_content = html_template.replace('__ITEMS_JSON__', json_data)
    html_content = html_content.replace('__RUN_ID__', run_id)

    try:
        with open(out_html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"[viewer] HTML profesional generado: {out_html_path}")
        return out_html_path
    except Exception as e:
        print(f"[error] No se pudo guardar el HTML: {e}")
        return None

# ---------------------------------------------------------
# CLASE PRINCIPAL: PORTFOLIO ENGINE (CORREGIDA Y COMPLETA)
# ---------------------------------------------------------
class PortfolioEngine:
    def __init__(self, config):
        self.config = config
        self.capital = config.get('capital', 100000.0)
        self.risk_free = config.get('risk_free', 0.08)
        self.max_assets = config.get('max_selected_assets', 12)
        self.seed = config.get('seed', 42)
        
        # Estado interno (Datos)
        self.prices_all = None
        self.prices_assets = None
        self.prices_bench = None
        self.returns_daily = None
        self.bench_returns = None
        
        # Estado interno (Métricas para reportes)
        self.mean_annual = None
        self.vol_annual = None
        self.betas_hist = None
        self.cov_annual = None
        
        # Resultados
        self.scenarios = {}
        self.backtest_results = {}
        self.sims_assets_all_log = None
        self.sims_bench_all_log = None
        self.sims_assets_factor_log = None
        self.sims_factor_log = None
        
        # Metadata de pronósticos
        self.cum_ports_by_scenario = {}
        
        np.random.seed(self.seed)

    def fetch_and_filter_data(self, tickers, benchmark, start_date, end_date):
        print(f"\n[1/6] 📥 Descargando datos ({len(tickers)} tickers)...")
        self.prices_all = download_prices(tickers + [benchmark], start_date, end_date)
        
        if benchmark in self.prices_all.columns:
            self.prices_bench = self.prices_all[benchmark].dropna()
        else:
            self.prices_bench = None

        valid_cols = [c for c in tickers if c in self.prices_all.columns]
        assets_data = self.prices_all[valid_cols].dropna(how='all').dropna(axis=1, how='all')
        
        print(f"[2/6] 🔍 Seleccionando Top-{self.max_assets} activos...")
        r_temp = compute_returns(assets_data)
        scores = compute_asset_scores(assets_data, r_temp)
        top_assets = scores.head(self.max_assets).index.tolist()
        
        self.prices_assets = assets_data[top_assets].copy()
        
        if self.prices_bench is not None:
            common = self.prices_assets.index.intersection(self.prices_bench.index)
            self.prices_assets = self.prices_assets.loc[common]
            self.prices_bench = self.prices_bench.loc[common]
        
        self.returns_daily = compute_returns(self.prices_assets)
        self.bench_returns = compute_returns(self.prices_bench.to_frame()).iloc[:, 0] if self.prices_bench is not None else pd.Series(0, index=self.returns_daily.index)
        
        # Calcular métricas base necesarias para optimización y reportes
        self.mean_annual, self.vol_annual = annualize_returns(self.returns_daily)
        self.betas_hist = compute_betas_historical(self.returns_daily, self.bench_returns)
        self.cov_annual = self.returns_daily.cov() * TRADING_DAYS

    def run_optimizations(self):
        print(f"\n[3/6] 🧠 Ejecutando optimizaciones y simulaciones...")
        
        n = len(self.prices_assets.columns)
        # Configuración de límites (bounds)
        min_w, max_w = 0.03, 0.20
        if ALLOW_ZERO_WEIGHTS: min_w = 0.0
        bounds = tuple((min_w, max_w) for _ in range(n))
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1},)
        
        params = {
            'mean_returns': self.mean_annual.values, 
            'cov_matrix': self.cov_annual.values, 
            'betas': self.betas_hist.values,
            'bounds': bounds, 'constraints': constraints, 
            'lam_tc': self.config.get('lam_tc', 0), 
            'turnover_max': self.config.get('turnover_max')
        }
        
        # 1. Optimizaciones Estáticas
        self.scenarios['MinVar'] = optimize_portfolio(**params, objective_fn=obj_min_var)
        self.scenarios['MaxSharpe'] = optimize_portfolio(**params, objective_fn=obj_max_sharpe)
        self.scenarios['MaxReturn'] = optimize_portfolio(**params, objective_fn=obj_max_return)
        self.scenarios['MinBeta'] = optimize_portfolio(**params, objective_fn=obj_min_beta)
        self.scenarios['ERC'] = risk_parity_weights(self.cov_annual.values)
        
        # 2. Simulaciones Monte Carlo (Usando tqdm)
        print("      🎲 Generando simulaciones Monte Carlo...")
        returns_all = self.returns_daily.join(self.bench_returns.rename(self.config.get('benchmark','SPY')), how='inner')
        n_sims = 500
        sims_log = mc_simulate_multivariate_logreturns(returns_all, n_sims=n_sims)
        
        n_assets = len(self.prices_assets.columns)
        self.sims_assets_all_log = sims_log[:, :, :n_assets]
        self.sims_bench_all_log = sims_log[:, :, n_assets]
        
        # Optimización Robusta
        print("      🛡️ Optimizando Robust (Avg & Worst)...")
        # Nota: Aquí podrías envolver con tqdm si quisieras, como vimos antes
        self.scenarios['RobustAvgMC'] = optimize_over_simulations_local(self.sims_assets_all_log, method='avg_sharpe', bounds=bounds, constraints=constraints, risk_free=self.risk_free)
        self.scenarios['RobustWorstMC'] = optimize_over_simulations_local(self.sims_assets_all_log, method='worst_sharpe', bounds=bounds, constraints=constraints, risk_free=self.risk_free)
        
        # 3. Factores
        print("      Factor Model Simulation...")
        factors = pd.DataFrame({'MKT': self.bench_returns}).loc[self.returns_daily.index]
        B, resid_df = fit_factor_model_log(self.returns_daily, factors)
        self.sims_assets_factor_log, self.sims_factor_log = simulate_via_factors_with_bench_log(B, resid_df, factors, n_sims=400)
        self.scenarios['FactorRobust'] = optimize_over_simulations_local(self.sims_assets_factor_log, method='avg_sharpe', bounds=bounds, constraints=constraints, risk_free=self.risk_free)

    def run_backtest(self):
        print(f"\n[4/6] 📉 Ejecutando Backtest histórico...")
        self.backtest_results = {}
        self.cum_ports_by_scenario = {}
        
        # Backtest histórico
        for name, weights in self.scenarios.items():
            w_series = pd.Series(weights, index=self.prices_assets.columns)
            vals, _ = backtest_fixed(self.prices_assets, w_series, self.capital)
            self.backtest_results[name] = vals
            
            # También pre-calculamos la proyección a futuro (para los percentiles)
            if name == 'FactorRobust':
                cum = get_cum_port_fullyear_per_sim(self.sims_assets_factor_log, weights)
            else:
                cum = get_cum_port_fullyear_per_sim(self.sims_assets_all_log, weights)
            self.cum_ports_by_scenario[name] = cum

# ---------- MAIN (pipeline) ----------
def main():
    # ---------------------------------------------------------
    # 1. CONFIGURACIÓN E INPUTS
    # ---------------------------------------------------------
    config, args = parse_args_and_config()
    
    # Input: Tasa Libre de Riesgo (Dinámica / México)
    rf_input = prompt_risk_free_rate(default=0.105)
    config['risk_free'] = rf_input
    
    # Input: Capital y Assets
    cap_input = prompt_initial_capital(default=config.get('capital', 100000.0))
    config['capital'] = cap_input
    max_assets = prompt_max_assets(default=config.get('max_selected_assets', 12))
    config['max_selected_assets'] = max_assets

    # Actualizar globales para compatibilidad
    global RISK_FREE_ANNUAL, CAPITAL_TO_DEPLOY, MAX_SELECTED_ASSETS
    RISK_FREE_ANNUAL = rf_input
    CAPITAL_TO_DEPLOY = cap_input
    MAX_SELECTED_ASSETS = max_assets
    
    # ---------------------------------------------------------
    # 2. INICIAR MOTOR Y CÁLCULOS
    # ---------------------------------------------------------
    engine = PortfolioEngine(config)
    
    tickers = ['AAPL','MSFT','AMZN','GOOGL','TSLA','NVDA','AMD','LLY','META','UBER','COST','ORCL','JPM','BAC','MA','SPOT',
    'PEP','KO','PG','JNJ','MRK','PFE','DIS','NFLX','INTC','QCOM','TXN','CRM','ADBE','SHOP','V','AXP',
    'BA','CAT','GE','HON','MMM','UPS','FDX','NKE','SBUX','MCD','WMT','TGT','HD','LOW','DE','CVX','XOM','SLB',
    'BP','RIO','BHP','PLTR','PYPL','DKNG','TTD','ABNB','MAR','HLT','DAL','UAL','CCL','RCL']
    
    start_date = (datetime.today() - timedelta(days=365*5)).strftime('%Y-%m-%d')
    end_date = datetime.today().strftime('%Y-%m-%d')
    
    try:
        # A. Descarga y Optimización
        engine.fetch_and_filter_data(tickers, config.get('benchmark', 'SPY'), start_date, end_date)
        engine.run_optimizations()
        engine.run_backtest()
        
        # ---------------------------------------------------------
        # 3. GENERACIÓN DE REPORTES (Restaurando TODO lo perdido)
        # ---------------------------------------------------------
        print(f"\n[5/6] 📊 Generando TODOS los reportes y gráficos...")
        
        # A. Pronósticos y Métricas (Forecasts)
        scenario_forecasts = []
        for name, w in engine.scenarios.items():
            if name == 'FactorRobust':
                stats = summarise_simulation_for_weights_from_log(engine.sims_assets_factor_log, engine.sims_factor_log, w, rf=RISK_FREE_ANNUAL)
            else:
                stats = summarise_simulation_for_weights_from_log(engine.sims_assets_all_log, engine.sims_bench_all_log, w, rf=RISK_FREE_ANNUAL)
            scenario_forecasts.append([name, stats['ExpReturn'], stats['AnnVol'], stats['Sharpe'], stats['BetaSim'], stats['VaR95']])
        
        forecast_df = pd.DataFrame(scenario_forecasts, columns=['Escenario','ExpReturn_1Y','AnnVol_1Y','Sharpe','BetaSim','VaR95']).set_index('Escenario')
        save_table_image(forecast_df.round(6), os.path.join(OUT_DIR, 'forecast_by_scenario.png'), title='Pronóstico 1A por Escenario')

        # B. Tabla por Activos
        assets_forecast_df = summarise_assets_from_sims_log(engine.sims_assets_all_log, engine.sims_bench_all_log, asset_names=engine.prices_assets.columns)
        assets_forecast_df['BetaHist'] = engine.betas_hist.reindex(assets_forecast_df.index).values
        assets_forecast_df['Hist_AnnReturn'] = engine.mean_annual.reindex(assets_forecast_df.index).values
        assets_forecast_df['Hist_AnnVol'] = engine.vol_annual.reindex(assets_forecast_df.index).values
        assets_forecast_df = assets_forecast_df[['BetaHist','BetaSim','Hist_AnnReturn','ExpReturn_1Y','Hist_AnnVol','Vol_1Y']]
        save_table_image(assets_forecast_df.round(6), os.path.join(OUT_DIR, 'assets_forecast_table.png'), title='Detalle por Activo')

        # C. Pesos y Contribuciones
        save_weights_compact_table(engine.scenarios, list(engine.prices_assets.columns), os.path.join(OUT_DIR, 'weights_compact.png'), title='Pesos Compactos')
        weights_table = pd.DataFrame({name: np.asarray(w) for name,w in engine.scenarios.items()}, index=engine.prices_assets.columns).T
        save_table_image(weights_table.round(6), os.path.join(OUT_DIR, 'weights_per_scenario_full.png'), title='Pesos Completos')
        
        # Risk Contributions (CSV)
        rc_tables = {}
        for name, w in engine.scenarios.items():
            rc, rc_pct = risk_contributions(engine.cov_annual.values, np.asarray(w))
            rc_tables[name] = pd.DataFrame({'RC': rc, 'RC_Pct': rc_pct}, index=engine.prices_assets.columns)
        if rc_tables:
            pd.concat(rc_tables, axis=0).to_csv(os.path.join(OUT_DIR, 'risk_contributions.csv'))

        # D. Gráficos de Backtest (Estático e Interactivo)
        plt.figure(figsize=(12,8))
        for name, series in engine.backtest_results.items():
            plt.plot(series.index, series.values, label=name)
        if engine.prices_bench is not None:
            b_line = (engine.prices_bench / engine.prices_bench.iloc[0]) * engine.capital
            plt.plot(b_line.index, b_line.values, 'k--', label='Benchmark', alpha=0.7)
        plt.title(f'Backtest Comparativo (Inicio ${engine.capital:,.2f})')
        plt.legend()
        plt.tight_layout()
        graph_path = os.path.join(OUT_DIR, 'backtest_comparative.png')
        plt.savefig(graph_path, dpi=200, bbox_inches='tight'); plt.close()
        
        interactive_path = os.path.join(OUT_DIR, 'backtest_interactive.html')
        bench_val = (engine.prices_bench / engine.prices_bench.iloc[0]) * engine.capital if engine.prices_bench is not None else None
        make_backtest_plotly(engine.backtest_results, bench_val, engine.capital, interactive_path, config.get('colorblind_mode', True))

        # E. Diagnósticos (Correlación y Walk Forward)
        corr_path = os.path.join(OUT_DIR, 'correlation_heatmap.png')
        save_correlation_heatmap(engine.returns_daily, corr_path, title="Correlaciones")
        
        # F. Rolling Metrics y Drawdowns (Loop)
        rolling_paths = []
        dd_paths = []
        dd_summary_rows = []
        
        # Usamos tqdm para ver progreso de generación de imágenes
        print("      Generando gráficos individuales...")
        for name, series in tqdm(engine.backtest_results.items(), desc="Plots"):
            ret = series.pct_change().dropna()
            sharpe_p = os.path.join(OUT_DIR, f'rolling_sharpe_{name}.png')
            vol_p = os.path.join(OUT_DIR, f'rolling_vol_{name}.png')
            dd_p = os.path.join(OUT_DIR, f'drawdown_{name}.png')
            
            plot_rolling_metrics(ret, sharpe_p, vol_p, window=ROLLING_WINDOW)
            plot_drawdown_summary(series, dd_p, title=f"Drawdown - {name}")
            
            rolling_paths.extend([sharpe_p, vol_p])
            dd_paths.append(dd_p)
            
            # Datos para tabla resumen de DD
            dd, max_dd, peak, trough, rec_days = compute_drawdown(series)
            dd_summary_rows.append({"Scenario": name, "MaxDrawdown": max_dd, "Peak": peak, "RecoveryDays": rec_days})

        # Tabla resumen Drawdown
        if dd_summary_rows:
            dd_df = pd.DataFrame(dd_summary_rows).set_index('Scenario')
            dd_df['MaxDrawdown'] = dd_df['MaxDrawdown'].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "")
            dd_sum_path = os.path.join(OUT_DIR, 'drawdown_summary_by_scenario.png')
            save_table_image(dd_df, dd_sum_path, title="Resumen de Drawdowns")
        else:
            dd_sum_path = None

        # G. VaR, CVaR y Percentiles
        perc_rows = {name: np.percentile(cum, [5,25,50,75,95]) for name, cum in engine.cum_ports_by_scenario.items()}
        perc_df = pd.DataFrame(perc_rows, index=['p5','p25','p50','p75','p95']).T
        perc_path = os.path.join(OUT_DIR, 'percentiles_by_scenario.png')
        save_table_image(perc_df, perc_path, title='Percentiles Simulados (1 Año)')
        
        # VaR/CVaR Horizons
        horizons = {'VaR_1d':1, 'VaR_5d':5, 'VaR_21d':21}
        var_rows = {}
        for name, w in engine.scenarios.items():
            vals = []
            s_log = engine.sims_assets_factor_log if name=='FactorRobust' else engine.sims_assets_all_log
            for h in horizons.values():
                vals.append(var95_from_sims_horizon(s_log, w, h))
            var_rows[name] = vals
        var_df = pd.DataFrame(var_rows, index=horizons.keys()).T
        var_path = os.path.join(OUT_DIR, 'var_by_horizon.png')
        save_table_image(var_df, var_path, title='VaR 95% por Horizonte')

        # H. Investment Plans (Cuántas acciones comprar)
        latest_prices = engine.prices_assets.ffill().iloc[-1]
        inv_plans = compute_investment_plan_per_scenario(engine.scenarios, latest_prices, engine.capital)
        inv_paths = save_investment_tables(inv_plans, OUT_DIR)

        # ---------------------------------------------------------
        # 4. ADMINISTRADOR DE CARTERAS (RESTAURADO)
        # ---------------------------------------------------------
        print(f"\n[6/6] 💼 Iniciando Administrador de Carteras y Rebalanceo...")
        # Esta es la función que permite Guardar/Cargar/Editar carteras y genera alertas
        alerts_path, alerts_plot_path, actions_path, impact_path, trades_csv = rebalance_flow(
            engine.prices_assets, engine.scenarios, engine.capital, config
        )

        # I. Stress Tests (Extended)
        print("      Ejecutando Stress Tests históricos...")
        try:
            stress_prices = download_prices(list(engine.prices_assets.columns), "2007-01-01", end_date)
            stress_paths = stress_test_crises(stress_prices, engine.scenarios, out_dir=OUT_DIR, reference_index=engine.prices_assets.columns)
        except Exception as e:
            print(f"      [!] No se pudieron ejecutar Stress Tests completos: {e}")
            stress_paths = []

        # ---------------------------------------------------------
        # 5. GENERAR VISOR HTML FINAL
        # ---------------------------------------------------------
        image_index = {}
        def add(cat, path, title=None):
            if path and os.path.exists(path):
                t = title if title else os.path.basename(path)
                image_index.setdefault(cat, []).append((t, path))
        
        # Llenar índice para el visor
        add('Comparativa', interactive_path, 'Interactivo')
        add('Comparativa', graph_path, 'Estático')
        add('Forecasts', os.path.join(OUT_DIR, 'forecast_by_scenario.png'), 'Escenarios')
        add('Forecasts', os.path.join(OUT_DIR, 'assets_forecast_table.png'), 'Activos')
        add('Pesos', os.path.join(OUT_DIR, 'weights_compact.png'), 'Compacto')
        add('Pesos', os.path.join(OUT_DIR, 'weights_per_scenario_full.png'), 'Completo')
        add('Riesgo', var_path, 'VaR')
        add('Riesgo', perc_path, 'Percentiles')
        if dd_sum_path: add('Riesgo', dd_sum_path, 'Drawdowns Resumen')
        
        for p in stress_paths: add('Stress Tests', p)
        
        # Agregar gráficos de rebalanceo si se generaron
        if alerts_path: add('Rebalanceo', alerts_path, 'Alertas')
        if actions_path: add('Rebalanceo', actions_path, 'Sugerencias')
        if impact_path: add('Rebalanceo', impact_path, 'Impacto')
        
        # Agregar planes de inversión
        for p in inv_paths: add('Inversiones', p)
        
        # Agregar gráficos individuales
        for name in engine.scenarios.keys():
            add(name, os.path.join(OUT_DIR, f'drawdown_{name}.png'), 'Drawdown')
            add(name, os.path.join(OUT_DIR, f'rolling_sharpe_{name}.png'), 'Sharpe')

        # Generar HTML
        html_path = os.path.join(OUT_DIR, 'viewer.html')
        generate_html_viewer(image_index, html_path, run_id=datetime.now().strftime('%H:%M'))
        
        print(f"\n✅ ¡Todo listo! Abre el visor: {html_path}")
        try:
            webbrowser.open('file://' + os.path.abspath(html_path))
        except:
            pass

    except KeyboardInterrupt:
        print("\n❌ Interrumpido.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()

    
