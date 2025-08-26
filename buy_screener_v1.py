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
import json # Added missing import for json

# Load Excel from the provided URL
df = pd.read_excel('https://github.com/fauzilrizqi14/stock_screener/raw/main/Daftar%20Saham%20%20-%2020250826.xlsx')

print("\n--- DEBUG DATA EXCEL ---")
print("DEBUG: Isi awal DataFrame dari Excel (5 baris pertama):")
print(df.head())
print("\nDEBUG: dtypes awal DataFrame:")
print(df.dtypes)

# Ensure 'Saham' column is numeric and drop invalid data
# Pertama, ubah kolom 'Saham' menjadi string untuk memastikan operasi string bisa dilakukan
df['Saham'] = df['Saham'].astype(str)
# Hapus titik sebagai pemisah ribuan
df['Saham'] = df['Saham'].str.replace('.', '', regex=False)
# Sekarang konversi ke numerik, menggunakan errors='coerce' untuk nilai yang masih tidak valid
df['Saham'] = pd.to_numeric(df['Saham'], errors='coerce')

print("\nDEBUG: dtypes DataFrame setelah pembersihan dan konversi 'Saham':")
print(df.dtypes)

# Drop rows where 'Saham' is NaN after conversion
df_shares = df[['Kode', 'Saham']].dropna(subset=['Saham'])

print("\nDEBUG: DataFrame setelah konversi 'Saham' ke numerik dan drop NaN (5 baris pertama):")
print(df_shares.head())
if df_shares.empty:
    print("DEBUG: PERHATIAN: df_shares KOSONG! Pastikan kolom 'Saham' di Excel berisi angka valid.")


# Buat dict kode saham -> jumlah saham
# Ensure 'Kode' is treated as string for dictionary keys
market_caps_shares = dict(zip(df_shares['Kode'].astype(str), df_shares['Saham']))

print("\nDEBUG: Isi dari market_caps_shares (beberapa entri pertama):")
# Print a limited number of dictionary items for debugging
for i, (k, v) in enumerate(market_caps_shares.items()):
    if i < 10: # Print first 10 items
        print(f"  '{k}': {v}")
    else:
        break
if not market_caps_shares:
    print("  (Kamus market_caps_shares kosong!)")


# List ticker buat yfinance
tickers = df['Kode'].dropna().astype(str) + '.JK'
tickers = tickers.tolist()

print(f"\nDEBUG: Contoh ticker untuk yfinance (5 ticker pertama): {tickers[:5]}")
print(f"DEBUG: Total jumlah ticker yang akan diproses: {len(tickers)}")


# --- Setup Telegram ID ---
BOT_TOKEN = os.getenv('BOT_TOKEN')
CHAT_ID = os.getenv('CHAT_ID')
CONFIG_SHEET_NAME = 'config_screener' # NAMA FILE GOOGLE SHEET ANDA

def authorize_gspread_from_env():
    """Membaca kredensial dari environment variable dan mengotorisasi gspread."""
    # Ambil JSON string dari environment variable yang di-set oleh GitHub Secrets
    creds_json_str = os.getenv("GOOGLE_CREDENTIALS_JSON")
    
    if not creds_json_str:
        print("FATAL ERROR: Environment variable 'GOOGLE_CREDENTIALS_JSON' tidak ditemukan!")
        print("Pastikan Anda sudah mengaturnya di bagian Secrets pada repository GitHub Anda.")
        return None

    try:
        # Ubah JSON string menjadi dictionary
        creds_dict = json.loads(creds_json_str)
        
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        
        # Otorisasi menggunakan dictionary, bukan file
        creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)
        client = gspread.authorize(creds)
        
        print("    -> Berhasil terhubung ke Google API menggunakan environment variable.")
        return client
    except Exception as e:
        print(f"    -> FATAL ERROR: Gagal melakukan otorisasi Google Sheets: {e}")
        return None

def load_config_from_sheet(client, sheet_name):
    """
    Loads configuration settings, enabled signals, and signal weights from a Google Sheet.

    Args:
        client (gspread.Client): An authorized gspread client.
        sheet_name (str): The title of the Google Spreadsheet.

    Returns:
        tuple: A tuple containing MIN_SCORE_THRESHOLD, SEND_ONLY_ABOVE_THRESHOLD,
               MAX_ITEM_TELE, K_THRESHOLD_VALUE, MIN_MARKET_CAP,
               enabled_signals dictionary, and signal_weights dictionary.

    Raises:
        gspread.exceptions.WorksheetNotFound: If the specific worksheet is not found.
        Exception: For general errors during Google Sheet access or parsing.
    """
    try:
        spreadsheet = client.open(sheet_name)
        sheet = spreadsheet.worksheet("sinyal_beli_screener") # Assuming this is the specific worksheet
        records = sheet.get_all_records()

        def to_bool(val): return str(val).strip().lower() == "true"

        config_dict = {str(item['key']).strip(): str(item['value']).strip() for item in records if item.get('key') and item.get('value') is not None}

        MIN_SCORE_THRESHOLD = float(config_dict.get("MIN_SCORE_THRESHOLD", 1.0))
        SEND_ONLY_ABOVE_THRESHOLD = to_bool(config_dict.get("SEND_ONLY_ABOVE_THRESHOLD", "True"))
        MAX_ITEM_TELE = int(config_dict.get("MAX_ITEM_TELE", 25))
        K_THRESHOLD_VALUE = int(config_dict.get("K_THRESHOLD_VALUE", 25))
        MIN_MARKET_CAP = float(config_dict.get("MIN_MARKET_CAP", 5_000_000_000_000))

        enabled_signals, signal_weights = {}, {}
        for item in records:
            key = item.get('key')
            if key is None or not isinstance(key, str): continue
            key = str(key).strip()
            if not key: continue

            enabled_signals[key] = to_bool(item.get('value', 'False'))
            try:
                signal_weights[key] = float(item.get('score_weight', 0))
            except (TypeError, ValueError):
                signal_weights[key] = 0.0

        print("    -> Berhasil memuat konfigurasi dari sheet 'sinyal_beli_screener'.")
        return MIN_SCORE_THRESHOLD, SEND_ONLY_ABOVE_THRESHOLD, MAX_ITEM_TELE, K_THRESHOLD_VALUE, MIN_MARKET_CAP, enabled_signals, signal_weights
    except gspread.exceptions.WorksheetNotFound:
        print("    -> ERROR: Sheet dengan nama 'sinyal_beli_screener' tidak ditemukan. Menggunakan konfigurasi default.")
        return 1.0, True, 25, 25, 5_000_000_000_000, {}, {}
    except Exception as e:
        print(f"    -> ERROR saat memuat konfigurasi: {e}. Menggunakan konfigurasi default.")
        return 1.0, True, 25, 25, 5_000_000_000_000, {}, {}


# --- Main execution flow ---
# Authorize gspread first
gspread_client = authorize_gspread_from_env()

if gspread_client is None:
    print("Script terminated due to Google Sheets authorization failure.")
    exit() # Exit the script if authorization fails

# Load configurations using the authorized client
MIN_SCORE_THRESHOLD, SEND_ONLY_ABOVE_THRESHOLD, MAX_ITEM_TELE, K_THRESHOLD_VALUE, MIN_MARKET_CAP, enabled_signals, signal_weights = load_config_from_sheet(gspread_client, CONFIG_SHEET_NAME)

print(f"\nDEBUG: Konfigurasi yang dimuat:")
print(f"  MIN_SCORE_THRESHOLD: {MIN_SCORE_THRESHOLD}")
print(f"  SEND_ONLY_ABOVE_THRESHOLD: {SEND_ONLY_ABOVE_THRESHOLD}")
print(f"  MAX_ITEM_TELE: {MAX_ITEM_TELE}")
print(f"  K_THRESHOLD_VALUE: {K_THRESHOLD_VALUE}")
print(f"  MIN_MARKET_CAP: {MIN_MARKET_CAP}")
print(f"  Enabled Signals (some): {dict(list(enabled_signals.items())[:5])}...") # Print first 5 enabled signals
print(f"  Signal Weights (some): {dict(list(signal_weights.items())[:5])}...") # Print first 5 signal weights


def format_market_cap_trillion(market_cap):
    # market_cap dalam rupiah, bagi 1 triliun (1_000_000_000_000)
    val = market_cap / 1_000_000_000_000
    return f"{val:.2f} Tn"

def classify_entry_level(conditions, score):
    cond = set(conditions)

    # Safe Entry
    safe_core = {"MACD cross up 0", "EMA9 cross up EMA21", "Volume breakout", "Higher High & Higher Low"}
    if safe_core.issubset(cond):
        return "Safe Entry"

    # Normal Entry
    normal_mandatory = {"MACD cross up 0", "EMA9 cross up EMA21"}
    normal_optional = {"Volume breakout", "RSI naik dari bawah 30"}
    if normal_mandatory.issubset(cond) and cond.intersection(normal_optional):
        return "Normal Entry"

    # Early Entry
    # K_THRESHOLD_VALUE is now defined globally when this function is called
    early_indicators = {
        f"StochRSI K cross D & K < {K_THRESHOLD_VALUE}",
        "StochRSI K cross up 0",
        "RSI naik dari bawah 30",
        "MACD cross up signal"
    }
    if cond.intersection(early_indicators):
        return "Early Entry"

    # Watchlist fallback
    if score > 2.0:
        return "Watchlist"

    return "Unclassified"

# Define signal conditions and their default weights.
# THIS SECTION MUST BE AFTER load_config_from_sheet CALL
# to ensure K_THRESHOLD_VALUE is defined for the f-string in its label.
signal_conditions = {
    "macd_cross_0": {
        "check": lambda prev, latest: prev['MACD'] < 0 and latest['MACD'] >= 0,
        "bobot": 0.85,
        "label": "MACD cross up 0"
    },
    "macd_cross_signal": {
        "check": lambda prev, latest: prev['MACD'] < prev['MACD_signal'] and latest['MACD'] >= latest['MACD_signal'] and latest['MACD'] < 0,
        "bobot": 0.65,
        "label": "MACD cross up signal"
    },
    "stoch_kd_cross": {
        "check": lambda prev, latest: prev['K'] < prev['D'] and latest['K'] >= latest['D'] and latest['K'] < K_THRESHOLD_VALUE,
        "bobot": 0.70,
        "label": f"StochRSI K cross D & K < {K_THRESHOLD_VALUE}" # K_THRESHOLD_VALUE from loaded config
    },
    "stoch_k_cross_0": {
        "check": lambda prev, latest: prev['K'] < 0 and latest['K'] >= 0,
        "bobot": 0.55,
        "label": "StochRSI K cross up 0"
    },
    "volume_breakout": {
        "check": lambda prev, latest: latest['Volume'] > latest['Vol_SMA20'],
        "bobot": 0.75,
        "label": "Volume breakout"
    },
    "rsi_cross_30": {
        "check": lambda prev, latest: prev['RSI'] < 30 and latest['RSI'] >= 30,
        "bobot": 0.60,
        "label": "RSI naik dari bawah 30"
    },
    "higher_high_low": {
        "check": lambda prev, latest: prev['Low'] < latest['Low'] and prev['High'] < latest['High'],
        "bobot": 0.50,
        "label": "Higher High & Higher Low"
    },
    "ema9_cross_ema21": {
        "check": lambda prev, latest: prev['EMA9'] < prev['EMA21'] and latest['EMA9'] >= latest['EMA21'],
        "bobot": 0.70,
        "label": "EMA9 cross up EMA21"
    },
    "adx_above_20": {
        "check": lambda prev, latest: latest.get("ADX", 0) > 20,
        "bobot": 0.55,
        "label": "ADX > 20"
    },
    "price_above_ema50": {
        "check": lambda prev, latest: latest["Close"] > latest["EMA50"],
        "bobot": 0.60,
        "label": "Harga > EMA50"
    },
    "ema50_above_ema200": {
        "check": lambda prev, latest: latest["EMA50"] > latest["EMA200"],
        "bobot": 0.75,
        "label": "EMA50 > EMA200"
    },
    "macd_histogram_increasing": {
        "check": lambda prev, latest: latest["MACD"] - latest["MACD_signal"] > prev["MACD"] - prev["MACD_signal"],
        "bobot": 0.60,
        "label": "Histogram MACD naik"
    },
    "volume_above_avg_2x": {
        "check": lambda prev, latest: latest["Volume"] > 2 * latest["Vol_SMA20"],
        "bobot": 0.70,
        "label": "Volume > 2x SMA20"
    },
    "bullish_engulfing_last3": {
        "check": lambda prev, latest: (
            latest["Close"] > latest["Open"] and
            prev["Close"] < prev["Open"] and
            latest["Close"] > prev["Open"] and
            latest["Open"] < prev["Close"]
        ),
        "bobot": 0.65,
        "label": "Bullish Engulfing"
    }
}

# Override default signal weights with values loaded from the sheet
for key, cond in signal_conditions.items():
    cond["bobot"] = signal_weights.get(key, cond["bobot"]) # Use existing bobot as default if not found in sheet


def compute_stoch_rsi(close, rsi_period=14, stoch_period=14, k_period=3, d_period=3):
    """
    Computes the Stochastic RSI (%K and %D) for a given close price series.
    """
    # Calculate daily price changes
    delta = close.diff()

    # Calculate gain and loss
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Average gain and loss over rsi_period (using EMA for smoother RSI, typical practice)
    avg_gain = gain.ewm(span=rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(span=rsi_period, adjust=False).mean()

    # Avoid division by zero for RS
    rs = avg_gain / avg_loss.replace(0, np.nan) # Replace 0 with NaN to avoid division by zero
    rsi = 100 - (100 / (1 + rs))

    # Calculate stochastic RSI %K
    min_rsi = rsi.rolling(stoch_period).min()
    max_rsi = rsi.rolling(stoch_period).max()
    # Avoid division by zero if max_rsi == min_rsi
    k = (rsi - min_rsi) / (max_rsi - min_rsi).replace(0, np.nan) * 100
    k = k.fillna(0) # Fill NaN values that might arise from division by zero or initial periods

    # Calculate stochastic RSI %D as SMA of %K
    d = k.rolling(k_period).mean().rolling(d_period).mean()

    return k, d

def screen_stock(ticker, market_caps_shares, enabled_signals, signal_conditions, MIN_MARKET_CAP):
    """
    Screens a single stock based on technical indicators and predefined signals.
    """
    print(f"\n--- DEBUG: Memproses ticker: {ticker} ---") # Debug point 1

    try:
        ticker_code = ticker.replace('.JK', '')
        # DEBUG: Check if ticker_code is in market_caps_shares
        if ticker_code not in market_caps_shares:
            print(f"DEBUG: {ticker_code}: TIDAK DITEMUKAN di market_caps_shares. Melewati.") # More specific debug
            return None

        shares_outstanding = market_caps_shares.get(ticker_code, None)
        if shares_outstanding is None: # This check is now redundant if the above passes, but kept for safety
            print(f"DEBUG: {ticker_code}: Saham beredar tidak ditemukan (internal error). Melewati.") # Debug point 2
            return None

        ticker_obj = yf.Ticker(ticker)
        hist = ticker_obj.history(period="1d")
        if hist.empty:
            print(f"DEBUG: {ticker_code}: Data histori 1 hari kosong dari yfinance. Melewati.") # Debug point 3
            return None
        last_price = hist['Close'].iloc[-1]
        print(f"DEBUG: {ticker_code}: Harga terakhir: {last_price:.2f}")

        market_cap = shares_outstanding * last_price
        print(f"DEBUG: {ticker_code}: Kapitalisasi pasar: {format_market_cap_trillion(market_cap)} (Min: {format_market_cap_trillion(MIN_MARKET_CAP)})") # Formatted for readability
        if market_cap < MIN_MARKET_CAP:
            print(f"DEBUG: {ticker_code}: Kapitalisasi pasar di bawah minimum. Melewati.") # Debug point 4
            return None

        df_hist = ticker_obj.history(period="6mo", interval="1d")
        print(f"DEBUG: {ticker_code}: Jumlah data histori 6 bulan: {len(df_hist)} baris (Min 200)")
        if df_hist.empty or len(df_hist) < 200: # Need enough data for 200 EMA
            print(f"DEBUG: {ticker_code}: Data histori tidak cukup untuk indikator (membutuhkan setidaknya 200 hari untuk EMA200). Melewati.") # Debug point 5
            return None

        # Calculate fundamental indicators
        df_hist['SMA20'] = df_hist['Close'].rolling(window=20).mean()
        df_hist['K'], df_hist['D'] = compute_stoch_rsi(df_hist['Close'])

        # MACD
        ema_12 = df_hist['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df_hist['Close'].ewm(span=26, adjust=False).mean()
        df_hist['MACD'] = ema_12 - ema_26
        df_hist['MACD_signal'] = df_hist['MACD'].ewm(span=9, adjust=False).mean()
        df_hist['MACD_histogram'] = df_hist['MACD'] - df_hist['MACD_signal']

        # EMAs
        df_hist['EMA9'] = df_hist['Close'].ewm(span=9, adjust=False).mean()
        df_hist['EMA21'] = df_hist['Close'].ewm(span=21, adjust=False).mean()
        df_hist['EMA50'] = df_hist['Close'].ewm(span=50, adjust=False).mean()
        df_hist['EMA200'] = df_hist['Close'].ewm(span=200, adjust=False).mean()

        # RSI (using a more standard RSI calculation)
        delta = df_hist['Close'].diff(1)
        gain = delta.where(delta > 0, 0)
        loss = delta.where(delta < 0, 0).abs()
        avg_gain = gain.ewm(com=13, adjust=False).mean()
        avg_loss = loss.ewm(com=13, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df_hist['RSI'] = 100 - (100 / (1 + rs))

        # Volume SMA
        df_hist['Vol_SMA20'] = df_hist['Volume'].rolling(window=20).mean()

        # ADX
        adx_indicator = ADXIndicator(df_hist['High'], df_hist['Low'], df_hist['Close'], window=14)
        df_hist['ADX'] = adx_indicator.adx()


        # Ensure there are at least two rows for 'prev' and 'latest' comparison
        if len(df_hist) < 2:
            print(f"DEBUG: {ticker_code}: Data histori kurang dari 2 baris untuk perbandingan prev/latest. Melewati.")
            return None

        latest = df_hist.iloc[-1]
        prev = df_hist.iloc[-2]

        print(f"DEBUG: {ticker_code}: Harga Penutupan: {latest['Close']:.2f}, SMA20: {latest['SMA20']:.2f}")
        # Filter: latest Close price must be above SMA20 to be considered
        if latest['Close'] <= latest['SMA20']:
            print(f"DEBUG: {ticker_code}: Harga penutupan ({latest['Close']:.2f}) kurang dari atau sama dengan SMA20 ({latest['SMA20']:.2f}). Melewati.") # Debug point 6
            return None

        score = 0
        conditions_met = []

        # Evaluate signals based on enabled_signals and signal_conditions
        print(f"DEBUG: {ticker_code}: Mengevaluasi sinyal...")
        for key, enabled in enabled_signals.items():
            if not enabled:
                # print(f"DEBUG: {ticker_code}: Sinyal '{key}' dinonaktifkan.")
                continue

            cond = signal_conditions.get(key)
            if cond is None:
                print(f"DEBUG: {ticker_code}: Definisi sinyal '{key}' tidak ditemukan. Melewati.")
                continue

            try:
                if cond["check"](prev, latest):
                    score += cond.get("bobot", 0)
                    conditions_met.append(cond["label"])
                    print(f"DEBUG: {ticker_code}: Sinyal '{key}' TERPENUHI. Bobot: {cond.get('bobot', 0):.2f}. Skor saat ini: {score:.2f}")
                # else:
                    # print(f"DEBUG: {ticker_code}: Sinyal '{key}' TIDAK TERPENUHI.")
            except Exception as e:
                print(f"⚠️ Error evaluasi sinyal {key} pada {ticker}: {e}")
                continue
        print(f"DEBUG: {ticker_code}: Total skor setelah evaluasi sinyal: {score:.2f}") # Debug point 7

        # Only return results for stocks with a positive score
        if score > 0:
            entry_level = classify_entry_level(conditions_met, score)
            print(f"DEBUG: {ticker_code}: Saham lolos screening dengan skor {score:.2f} dan entry level '{entry_level}'.")
            return {
                "Kode": ticker_code,
                "Last Price": last_price,
                "MarketCap": market_cap,
                "Score": round(score, 3),
                "Entry Level": entry_level,
                "Keterangan": ", ".join(conditions_met)
            }
        else:
            print(f"DEBUG: {ticker_code}: Skor 0 atau kurang, tidak memenuhi kriteria akhir. Melewati.")

    except Exception as e:
        print(f"❌ Error saat screening untuk {ticker}: {e}")
        return None

    print(f"DEBUG: {ticker_code}: Akhir screening, tidak ada hasil.") # Debug point 8
    return None

def send_telegram_message(bot_token, chat_id, message):
    """
    Sends a message to a Telegram chat.
    """
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    data = {"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}
    response = requests.post(url, data=data)
    if response.status_code == 200:
        print("✅ Telegram notification sent successfully!")
    else:
        print(f"❌ Failed to send Telegram notification: {response.text}")

def format_telegram_message(df_result, max_items=MAX_ITEM_TELE):
    """
    Formats the screening results into a Telegram-friendly message.
    """
    today = datetime.now().strftime("%d %B %Y")
    message = f"📈 Rekomendasi *BELI SAHAM* Hari Ini ({today}):\n\n"

    display_df = df_result.head(max_items)

    if display_df.empty:
        message += "Tidak ada saham yang memenuhi kriteria screening."
        return message

    for i, row in display_df.iterrows():
        kode = row['Kode']
        score = row['Score']
        entry = row['Entry Level']
        keterangan = row['Keterangan']
        message += f"*{i+1}. {kode}* | Score: {score:.2f} | {entry}\n" \
                   f"    ↳ {keterangan}\n\n"

    return message

# --- Final processing and notification ---
results = []

print("\n--- Memulai proses screening saham ---")
for t in tickers:
    res = screen_stock(t, market_caps_shares, enabled_signals, signal_conditions, MIN_MARKET_CAP)
    if res is not None:
        results.append(res)
print("--- Proses screening saham selesai ---")


if len(results) == 0:
    print("\n❗ Tidak ada saham yang memenuhi kriteria screening.")
else:
    df_result = pd.DataFrame(results)
    df_result = df_result.sort_values(by=["Score", "MarketCap"], ascending=[False, False])
    df_result.reset_index(drop=True, inplace=True)

    if SEND_ONLY_ABOVE_THRESHOLD:
        df_filtered = df_result[df_result["Score"] >= MIN_SCORE_THRESHOLD]
        print(f"DEBUG: Hanya menampilkan saham dengan skor >= {MIN_SCORE_THRESHOLD:.2f}. Jumlah hasil: {len(df_filtered)}")
    else:
        df_filtered = df_result
        print(f"DEBUG: Menampilkan semua saham dengan skor positif. Jumlah hasil: {len(df_filtered)}")
    
    df_filtered['MarketCap_Tn'] = df_filtered['MarketCap'].apply(format_market_cap_trillion)

    print("\nHasil Screening Saham:")
    print(df_filtered[["Kode", "Last Price", "MarketCap_Tn", "Score", "Entry Level", "Keterangan"]])

    pesan = format_telegram_message(df_filtered)
    send_telegram_message(BOT_TOKEN, CHAT_ID, pesan)
