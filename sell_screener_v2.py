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

def compute_stoch_rsi(close, rsi_period=14, stoch_period=14, k_period=3, d_period=3):
    # Hitung perubahan harga harian
    delta = close.diff()

    # Hitung gain dan loss
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Rata-rata gain dan loss selama rsi_period (simple moving average atau exponential?)
    avg_gain = gain.rolling(rsi_period).mean()
    avg_loss = loss.rolling(rsi_period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    # Hitung stochastic RSI %K
    min_rsi = rsi.rolling(stoch_period).min()
    max_rsi = rsi.rolling(stoch_period).max()
    k_line = (rsi - min_rsi) / (max_rsi - min_rsi) * 100

    # Hitung stochastic RSI %D sebagai SMA dari %K
    d_line = k_line.rolling(k_period).mean().rolling(d_period).mean()

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

def check_take_profit_tp1(latest, tp1):
    return latest['Close'] >= tp1

def check_take_profit_tp2(latest, tp2):
    return latest['Close'] >= tp2

def check_cut_loss(latest, cut_loss):
    return latest['Close'] <= cut_loss

def check_macd_histogram_divergence(prev, latest):
    hist_prev = prev['MACD'] - prev['MACD_signal']
    hist_latest = latest['MACD'] - latest['MACD_signal']
    price_increase = latest['Close'] > prev['Close']
    hist_decrease = hist_latest < hist_prev
    return price_increase and hist_decrease

def check_fibonacci_retracement_support(latest):
    return False

def check_adx_di_crossover(prev, latest):
    return (prev['DI+'] < prev['DI-']) and (latest['DI+'] > latest['DI-']) and (latest['ADX'] > 20)

def check_doji_or_hammer(latest):
    return is_doji_or_hammer(latest)

# --- Screening Function ---
def screen_sell_stock(ticker, buy_price, enabled_signals, signal_weights, signal_labels):
    try:
        ticker_obj = yf.Ticker(ticker + ".JK")
        df_hist = ticker_obj.history(period="3mo", interval="1d")
        if df_hist.empty or len(df_hist) < 20:
            print(f"Data tidak cukup untuk {ticker}")
            return None

        # Hitung ATR(14)
        df_hist['H-L'] = df_hist['High'] - df_hist['Low']
        df_hist['H-PC'] = abs(df_hist['High'] - df_hist['Close'].shift(1))
        df_hist['L-PC'] = abs(df_hist['Low'] - df_hist['Close'].shift(1))
        df_hist['TR'] = df_hist[['H-L', 'H-PC', 'L-PC']].max(axis=1)
        df_hist['ATR14'] = df_hist['TR'].rolling(window=14).mean()

        latest = df_hist.iloc[-1]
        prev = df_hist.iloc[-2]
        atr_14 = latest['ATR14']

        # Hitung Cut Loss dan Take Profit
        cut_loss = buy_price - 1.5 * atr_14
        tp1 = buy_price + 2 * (buy_price - cut_loss)
        tp2 = buy_price + 3 * (buy_price - cut_loss)

        # Hitung indikator teknikal
        df_hist['EMA9'] = df_hist['Close'].ewm(span=9, adjust=False).mean()
        df_hist['EMA21'] = df_hist['Close'].ewm(span=21, adjust=False).mean()
        ema_12 = df_hist['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df_hist['Close'].ewm(span=26, adjust=False).mean()
        df_hist['MACD'] = ema_12 - ema_26
        df_hist['MACD_signal'] = df_hist['MACD'].ewm(span=9, adjust=False).mean()
        df_hist['RSI'] = df_hist['Close'].rolling(window=14).apply(
            lambda x: (100 - (100 / (1 + ((x.diff().clip(lower=0).mean()) /
                                          abs(x.diff().clip(upper=0).mean() if x.diff().clip(upper=0).mean() != 0 else 1)))))
        )

        # ADX dan DI
        adx_indicator = ADXIndicator(df_hist['High'], df_hist['Low'], df_hist['Close'], window=14)
        df_hist['ADX'] = adx_indicator.adx()
        df_hist['DI+'] = adx_indicator.adx_pos()
        df_hist['DI-'] = adx_indicator.adx_neg()

        latest = df_hist.iloc[-1]
        prev = df_hist.iloc[-2]

        score = 0
        matched_signals = []

        # Cek sinyal sesuai enabled_signals dan bobotnya
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
            elif key == "adx_di_crossover" and check_adx_di_crossover(prev, latest):
                score += signal_weights[key]
                matched_signals.append(signal_labels[key])
            elif key == "doji_or_hammer" and check_doji_or_hammer(latest):
                score += signal_weights[key]
                matched_signals.append(signal_labels[key])

      # Tambah sinyal TP dan CL otomatis
        if check_take_profit_tp1(latest, tp1):
            score += 1
            matched_signals.append("Take Profit TP1 (Harga >= TP1)")

        if check_take_profit_tp2(latest, tp2):
            score += 1
            matched_signals.append("Take Profit TP2 (Harga >= TP2)")

        if check_cut_loss(latest, cut_loss):
            score += 1
            matched_signals.append("Cut Loss (Harga <= Cut Loss)")

        # Tentukan action berdasarkan prioritas harga dan skor
        action = None
        if latest['Close'] >= tp2:
            action = "Take Profit TP2"
        elif latest['Close'] >= tp1:
            action = "Take Profit TP1"
        elif latest['Close'] <= cut_loss:
            action = "Cut Loss"
        elif score > 0:
            action = "Take Profit"

        if score > 0:
            return {
                "Stock": ticker,
                "Buy Price": buy_price,
                "ATR14": round(atr_14, 2),
                "Cut Loss": round(cut_loss, 2),
                "Take Profit TP1": round(tp1, 2),
                "Take Profit TP2": round(tp2, 2),
                "Last Price": round(latest['Close'], 2),
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
    res = screen_sell_stock(ticker, buy_price, enabled_signals, signal_weights, signal_labels)
    if res:
        results.append(res)


# --- Display Results ---
df_results = pd.DataFrame(results)
if not df_results.empty:
    df_results = df_results.sort_values(by="Score", ascending=False).reset_index(drop=True)
    print(df_results[[
        "Stock",
        "Buy Price",
        "Take Profit TP1",
        "Take Profit TP2",
        "Cut Loss",
        "Last Price",
        "Score",
        "Action",
        "Signals"
    ]])
else:
    print("Tidak ada sinyal jual yang terpenuhi saat ini.")


# --- Definisikan fungsi kirim pesan Telegram ---
def send_telegram_message(bot_token, chat_id, message):
    url = f"https://api.telegram.org/bot7487407302:AAGe6I8fLFGt19BfU7AFh31YMHaFVHUmu7U/sendMessage"
    data = {"chat_id": 652946372, "text": message, "parse_mode": "Markdown"}
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
        buy_price = row['Buy Price']
        last_price = row['Last Price']
        cut_loss = row['Cut Loss']
        tp1 = row['Take Profit TP1']
        tp2 = row['Take Profit TP2']
        score = row['Score']
        action = row['Action']
        signals = row['Signals']

        # Tentukan target CL_TP sesuai kondisi
        if buy_price >= last_price:
            target = f"{last_price}/{cut_loss} (CL)"
        elif buy_price < last_price and buy_price <= tp1:
            target = f"{last_price}/{tp1} (TP1)"
        else:  # buy_price > tp1
            target = f"{last_price}/{tp2} (TP2)"

        message += (
            f"{i+1}. {stock} | {buy_price} |{score:.2f} | {action}\n"
            f"   ↳ {signals}\n"
            f"   🎯 Target: *{target}*\n\n"
        )
    return message
# --- Kirim pesan Telegram ---
message = format_telegram_message_sell(df_results)
send_telegram_message(BOT_TOKEN, CHAT_ID, message)
