import os
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import tempfile
from datetime import datetime
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from ta.trend import ADXIndicator

# --- Setup Google Sheets Access ---
def authorize_gspread(creds_json_path="credentials.json"):
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    # Ambil JSON credential dari environment variable
    creds_json = os.getenv("GOOGLE_CREDENTIALS_JSON")
    if not creds_json:
        raise ValueError("Environment variable GOOGLE_CREDENTIALS_JSON tidak ditemukan!")

    # Buat temporary file untuk credential
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp:
        temp.write(creds_json)
        temp_path = temp.name

    creds = ServiceAccountCredentials.from_json_keyfile_name(temp_path, scope)
    client = gspread.authorize(creds)
    return client
    
# --- Setup Telegram ID ---    
BOT_TOKEN = os.getenv('BOT_TOKEN')
CHAT_ID = os.getenv('CHAT_ID')

client = authorize_gspread()

# --- Load Sheets Data ---
spreadsheet = client.open("config_screener")

# Load list_stock sheet
ws_stock = spreadsheet.worksheet("list_stock")
list_stock = pd.DataFrame(ws_stock.get_all_records())

# Load sinyal_jual_screener sheet
ws_signal = spreadsheet.worksheet("sinyal_jual_screener")
df_signals = pd.DataFrame(ws_signal.get_all_records())

# Parse signals config: enabled and score weights
def to_bool(val):
    return str(val).strip().lower() == "true"

enabled_signals = {row['key']: to_bool(row['value']) for idx, row in df_signals.iterrows()}
signal_weights = {row['key']: float(row['score_weight']) for idx, row in df_signals.iterrows()}
signal_labels = {row['key']: row['keterangan'] for idx, row in df_signals.iterrows()}

# --- Indicator Calculation Helpers ---

def compute_stoch_rsi(close, rsi_length=14, stoch_length=14, k=3, d=3):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=rsi_length).mean()
    avg_loss = loss.rolling(window=rsi_length).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    min_rsi = rsi.rolling(window=stoch_length).min()
    max_rsi = rsi.rolling(window=stoch_length).max()
    stoch_rsi = (rsi - min_rsi) / (max_rsi - min_rsi)
    k_line = stoch_rsi.rolling(window=k).mean() * 100
    d_line = k_line.rolling(window=d).mean()
    return k_line, d_line

def is_doji_or_hammer(candle):
    body = abs(candle['Close'] - candle['Open'])
    candle_range = candle['High'] - candle['Low']
    upper_shadow = candle['High'] - max(candle['Close'], candle['Open'])
    lower_shadow = min(candle['Close'], candle['Open']) - candle['Low']

    is_doji = body <= 0.1 * candle_range
    is_hammer = (lower_shadow >= 2 * body) and (upper_shadow <= 0.1 * body)
    return is_doji or is_hammer

# --- Define signal check functions ---
def check_macd_cross_down(prev, latest):
    return prev['MACD'] > prev['MACD_signal'] and latest['MACD'] <= latest['MACD_signal']

def check_ema9_cross_down_ema21(prev, latest):
    return prev['EMA9'] > prev['EMA21'] and latest['EMA9'] <= latest['EMA21']

def check_rsi_above_70_then_down(prev, latest):
    return prev['RSI'] > 70 and latest['RSI'] < prev['RSI']

def check_volume_drop(prev, latest):
    return latest['Volume'] < 0.7 * prev['Volume']

def check_price_above_target_profit(latest, target):
    if target is None or pd.isna(target):
        return False
    return latest['Close'] >= target

def check_price_below_cut_loss(latest, cutloss):
    if cutloss is None or pd.isna(cutloss):
        return False
    return latest['Close'] <= cutloss

def check_macd_histogram_divergence(prev, latest):
    # Simplified divergence check: histogram decreasing while price increasing
    hist_prev = prev['MACD'] - prev['MACD_signal']
    hist_latest = latest['MACD'] - latest['MACD_signal']
    price_increase = latest['Close'] > prev['Close']
    hist_decrease = hist_latest < hist_prev
    return price_increase and hist_decrease

def check_fibonacci_retracement_support(latest):
    # Placeholder: implement actual fib retracement logic if data available
    return False

def check_adx_di_crossover(prev, latest):
    return (prev['DI+'] < prev['DI-']) and (latest['DI+'] > latest['DI-']) and (latest['ADX'] > 20)

def check_doji_or_hammer(latest):
    return is_doji_or_hammer(latest)

# --- Screening Function ---
def screen_sell_stock(ticker, buy_price, target_profit, cut_loss):
    try:
        # Fungsi bantu konversi ke float aman
        def safe_float(val):
            try:
                if val is None or (isinstance(val, float) and pd.isna(val)):
                    return None
                return float(val)
            except (ValueError, TypeError):
                return None

        target_profit = safe_float(target_profit)
        cut_loss = safe_float(cut_loss)

        ticker_obj = yf.Ticker(ticker + ".JK")
        df_hist = ticker_obj.history(period="3mo", interval="1d")
        if df_hist.empty or len(df_hist) < 5:
            print(f"Data tidak cukup untuk {ticker}")
            return None

        # Hitung indikator
        df_hist['EMA9'] = df_hist['Close'].ewm(span=9, adjust=False).mean()
        df_hist['EMA21'] = df_hist['Close'].ewm(span=21, adjust=False).mean()
        ema_12 = df_hist['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df_hist['Close'].ewm(span=26, adjust=False).mean()
        df_hist['MACD'] = ema_12 - ema_26
        df_hist['MACD_signal'] = df_hist['MACD'].ewm(span=9, adjust=False).mean()
        df_hist['RSI'] = df_hist['Close'].rolling(window=14).apply(
            lambda x: (100 - (100 / (1 + ((x.diff().clip(lower=0).mean()) / abs(x.diff().clip(upper=0).mean() if x.diff().clip(upper=0).mean() != 0 else 1)))))
        )
        df_hist['Vol_SMA5'] = df_hist['Volume'].rolling(window=5).mean()

        # ADX & DI
        adx_indicator = ADXIndicator(df_hist['High'], df_hist['Low'], df_hist['Close'], window=14)
        df_hist['ADX'] = adx_indicator.adx()
        df_hist['DI+'] = adx_indicator.adx_pos()
        df_hist['DI-'] = adx_indicator.adx_neg()

        latest = df_hist.iloc[-1]
        prev = df_hist.iloc[-2]

        score = 0
        matched_signals = []
        action = None

        for key, enabled in enabled_signals.items():
            if not enabled:
                continue

            if key == "macd_cross_down" and check_macd_cross_down(prev, latest):
                score += signal_weights[key]
                matched_signals.append(signal_labels[key])

            elif key == "ema9_cross_down_ema21" and check_ema9_cross_down_ema21(prev, latest):
                score += signal_weights[key]
                matched_signals.append(signal_labels[key])

            elif key == "rsi_above_70_then_down" and check_rsi_above_70_then_down(prev, latest):
                score += signal_weights[key]
                matched_signals.append(signal_labels[key])

            elif key == "volume_drop" and check_volume_drop(prev, latest):
                score += signal_weights[key]
                matched_signals.append(signal_labels[key])

            elif key == "macd_histogram_divergence" and check_macd_histogram_divergence(prev, latest):
                score += signal_weights[key]
                matched_signals.append(signal_labels[key])

            elif key == "fibonacci_retracement_support" and check_fibonacci_retracement_support(latest):
                score += signal_weights[key]
                matched_signals.append(signal_labels[key])

            elif key == "adx_di_crossover" and check_adx_di_crossover(prev, latest):
                score += signal_weights[key]
                matched_signals.append(signal_labels[key])

            elif key == "doji_or_hammer" and check_doji_or_hammer(latest):
                score += signal_weights[key]
                matched_signals.append(signal_labels[key])

        # Evaluasi Action berdasarkan posisi harga terakhir
        if latest['Close'] >= buy_price:
          action = "Take profit"
        else:
          action = "Cut Loss"

        if score > 0:
            return {
                "Stock": ticker,
                "Buy Price": buy_price,
                "Target Profit": target_profit,
                "Cut Loss": cut_loss,
                "Last Price": latest['Close'],
                "Score": round(score, 2),
                "Signals": ", ".join(matched_signals),
                "Action": action or ""
            }
        else:
            return None

    except Exception as e:
        print(f"Error screening {ticker}: {e}")
        return None

# --- Run Screening for All Stocks ---
results = []
for idx, row in list_stock.iterrows():
    ticker = row['stock']
    buy_price = row['buy_price']
    target_profit = row.get('target_price_profit', None)
    cut_loss = row.get('target_price_cutloss', None)
    res = screen_sell_stock(ticker, buy_price, target_profit, cut_loss)
    if res:
        results.append(res)

# --- Display Results ---
df_results = pd.DataFrame(results)
if not df_results.empty:
    df_results = df_results.sort_values(by="Score", ascending=False).reset_index(drop=True)
    print(df_results[["Stock", "Buy Price", "Target Profit", "Cut Loss", "Last Price", "Score", "Action", "Signals"]])
else:
    print("Tidak ada sinyal jual yang terpenuhi saat ini.")

# --- Definisikan fungsi kirim pesan Telegram ---
def send_telegram_message(bot_token, chat_id, message):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    data = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
    response = requests.post(url, data=data)
    if response.status_code == 200:
        print("✅ Notifikasi Telegram terkirim!")
    else:
        print(f"❌ Gagal kirim notifikasi: {response.text}")

# --- Format pesan Telegram ---
def format_telegram_message_sell(df_result, max_items=25):
    if df_result.empty:
        return "ℹ️ Tidak ada saham yang harus dijual saat ini."

    today = datetime.now().strftime("%d %B %Y")
    message = f"📈 Rekomendasi *JUAL SAHAM* Saham Hari Ini ({today}):\n\n"

    for i, row in df_result.head(max_items).iterrows():
        stock = row['Stock']
        score = row['Score']
        action = row['Action']
        signals = row['Signals']
        message += f"{i+1}. {stock} | {score:.2f} | {action}\n   ↳ {signals}\n\n"
    return message

# --- Kirim pesan Telegram ---
message = format_telegram_message_sell(df_results)
send_telegram_message(BOT_TOKEN, CHAT_ID, message)


