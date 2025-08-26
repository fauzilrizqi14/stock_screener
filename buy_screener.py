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

# Load Excel
df = pd.read_excel('https://github.com/fauzilrizqi14/stock_screener/raw/main/Daftar%20Saham%20%20-%2020250826.xlsx')

# Pastikan kolom 'Saham' numeric dan drop data invalid
df['Saham'] = pd.to_numeric(df['Saham'], errors='coerce')
df_shares = df[['Kode', 'Saham']].dropna(subset=['Saham'])

# Buat dict kode saham -> jumlah saham
market_caps_shares = dict(zip(df_shares['Kode'].astype(str), df_shares['Saham']))

# List ticker buat yfinance
tickers = df['Kode'].dropna().astype(str) + '.JK'
tickers = tickers.tolist()

# --- Setup Telegram ID ---    
BOT_TOKEN = os.getenv('BOT_TOKEN')
CHAT_ID = os.getenv('CHAT_ID')

def load_config_from_sheet(sheet_name="config_screener"):
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

    sheet = client.open(sheet_name).sheet1
    records = sheet.get_all_records()

    def to_bool(val):
        return str(val).strip().lower() == "true"

    # Ambil konfigurasi umum dari config_dict
    def load_config_from_sheet(client, sheet_name):
        try:
            spreadsheet = client.open(sheet_name)
            sheet = spreadsheet.worksheet("sinyal_beli_screener")
            records = sheet.get_all_records()
            def to_bool(val): return str(val).strip().lower() == "true"
            config_dict = {str(item['key']).strip(): item['value'] for item in records if item.get('key')}
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
            print("   -> Berhasil memuat konfigurasi dari sheet 'sinyal_beli_screener'.")
            return MIN_SCORE_THRESHOLD, SEND_ONLY_ABOVE_THRESHOLD, MAX_ITEM_TELE, K_THRESHOLD_VALUE, MIN_MARKET_CAP, enabled_signals, signal_weights
        except gspread.exceptions.WorksheetNotFound:
            print("   -> ERROR: Sheet dengan nama 'sinyal_beli_screener' tidak ditemukan.")
            return 1.0, True, 25, 25, 5e12, {}, {}
        except Exception as e:
            print(f"   -> ERROR saat memuat konfigurasi: {e}")
            return 1.0, True, 25, 25, 5e12, {}, {}

    # Ambil sinyal dan bobot dari sheet
    enabled_signals = {}
    signal_weights = {}

    for item in records:
        key = item.get('key')
        if key is None:
            continue
        enabled_signals[key] = to_bool(item.get('value', 'False'))
        try:
            signal_weights[key] = float(item.get('score_weight', 0))
        except (TypeError, ValueError):
            signal_weights[key] = 0.0

    return MIN_SCORE_THRESHOLD, SEND_ONLY_ABOVE_THRESHOLD, MAX_ITEM_TELE, K_THRESHOLD_VALUE, MIN_MARKET_CAP, enabled_signals, signal_weights

# Panggil fungsi
MIN_SCORE_THRESHOLD, SEND_ONLY_ABOVE_THRESHOLD, MAX_ITEM_TELE, K_THRESHOLD_VALUE, MIN_MARKET_CAP, enabled_signals, signal_weights = load_config_from_sheet()


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
    early_indicators = {
        "StochRSI K cross D & K < {K_THRESHOLD_VALUE}",
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

#signal condition
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
        "label": f"StochRSI K cross D & K < {K_THRESHOLD_VALUE}"
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

# Masukkan bobot dari signal_weights ke setiap signal_conditions
for key, cond in signal_conditions.items():
    cond["bobot"] = signal_weights.get(key, 0)

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
    k = (rsi - min_rsi) / (max_rsi - min_rsi) * 100

    # Hitung stochastic RSI %D sebagai SMA dari %K
    d = k.rolling(k_period).mean().rolling(d_period).mean()

    return k, d

def screen_stock(ticker, market_caps_shares, enabled_signals, signal_conditions, MIN_MARKET_CAP):
    try:
        ticker_code = ticker.replace('.JK', '')
        shares_outstanding = market_caps_shares.get(ticker_code, None)
        if shares_outstanding is None:
            return None

        ticker_obj = yf.Ticker(ticker)
        hist = ticker_obj.history(period="1d")
        if hist.empty:
            return None
        last_price = hist['Close'].iloc[-1]

        market_cap = shares_outstanding * last_price
        if market_cap < MIN_MARKET_CAP:
            return None

        df_hist = ticker_obj.history(period="6mo", interval="1d")
        if df_hist.empty or len(df_hist) < 50:
            return None

        # Hitung indikator dasar
        df_hist['SMA20'] = df_hist['Close'].rolling(window=20).mean()
        df_hist['K'], df_hist['D'] = compute_stoch_rsi(df_hist['Close'])
        ema_12 = df_hist['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df_hist['Close'].ewm(span=26, adjust=False).mean()
        df_hist['MACD'] = ema_12 - ema_26
        df_hist['MACD_signal'] = df_hist['MACD'].ewm(span=9, adjust=False).mean()
        df_hist['EMA9'] = df_hist['Close'].ewm(span=9, adjust=False).mean()
        df_hist['EMA21'] = df_hist['Close'].ewm(span=21, adjust=False).mean()
        df_hist['EMA50'] = df_hist['Close'].ewm(span=50, adjust=False).mean()
        df_hist['EMA200'] = df_hist['Close'].ewm(span=200, adjust=False).mean()
        df_hist['RSI'] = 100 - (100 / (1 + df_hist['Close'].pct_change().add(1).rolling(14).apply(
            lambda x: (x[x > 0].mean() / abs(x[x < 0].mean())) if abs(x[x < 0].mean()) != 0 else 0)))
        df_hist['Vol_SMA20'] = df_hist['Volume'].rolling(window=20).mean()
        df_hist['MACD_histogram'] = df_hist['MACD'] - df_hist['MACD_signal']

        # ADX
        adx = ADXIndicator(df_hist['High'], df_hist['Low'], df_hist['Close'], window=14)
        df_hist['ADX'] = adx.adx()

        latest = df_hist.iloc[-1]
        prev = df_hist.iloc[-2]

        if latest['Close'] <= latest['SMA20']:
            return None

        score = 0
        conditions_met = []

        for key, enabled in enabled_signals.items():
            if not enabled:
                continue
            cond = signal_conditions.get(key)
            if cond is None:
                continue
            try:
                if cond["check"](prev, latest):
                    score += cond.get("bobot", 0)
                    conditions_met.append(cond["label"])
            except Exception as e:
                print(f"⚠️ Error evaluasi sinyal {key} pada {ticker}: {e}")
                continue

        if score > 0:
            # classify_entry_level bisa kamu panggil di sini jika sudah didefinisikan
            entry_level = classify_entry_level(conditions_met, score)
            return {
                "Kode": ticker_code,
                "Last Price": last_price,
                "MarketCap": market_cap,
                "Score": round(score, 3),
                "Entry Level": entry_level,
                "Keterangan": ", ".join(conditions_met)
            }

    except Exception as e:
        print(f"❌ Error di {ticker}: {e}")
        return None

    return None

def send_telegram_message(bot_token, chat_id, message):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    data = {"chat_id": chat_id, "text": message}
    response = requests.post(url, data=data)
    if response.status_code == 200:
        print("✅ Notifikasi Telegram terkirim!")
    else:
        print(f"❌ Gagal kirim notifikasi: {response.text}")

def format_telegram_message(df_result, max_items=MAX_ITEM_TELE):
    today = datetime.now().strftime("%d %B %Y")
    message = f"📈 Rekomendasi *BELI SAHAM* Hari Ini ({today}):\n\n"

    for i, row in df_result.head(max_items).iterrows():
        kode = row['Kode']
        score = row['Score']
        entry = row['Entry Level']
        keterangan = row['Keterangan']
        message += f"{i+1}. {kode} | Score: {score:.2f} | {entry}\n   ↳ {keterangan}\n\n"

    return message

results = []

for t in tickers:
    res = screen_stock(t, market_caps_shares, enabled_signals, signal_conditions, MIN_MARKET_CAP)
    if res is not None:
        results.append(res)

if len(results) == 0:
    print("❗ Tidak ada saham yang memenuhi kriteria screening")
else:
    import pandas as pd
    df_result = pd.DataFrame(results)
    df_result = df_result.sort_values(by=["Score", "MarketCap"], ascending=[False, False])
    df_result.reset_index(drop=True, inplace=True)

    # === Filter berdasarkan skor minimal (opsional)
    if SEND_ONLY_ABOVE_THRESHOLD:
        df_filtered = df_result[df_result["Score"] >= MIN_SCORE_THRESHOLD]
    else:
        df_filtered = df_result
    
    df_filtered['MarketCap_Tn'] = df_filtered['MarketCap'].apply(format_market_cap_trillion)

    # Tampilkan hasil
    print(df_filtered[["Kode", "Last Price", "MarketCap_Tn", "Score", "Entry Level", "Keterangan"]])


    # Format pesan Telegram (sesuaikan sesuai fungsi format yang sudah kamu buat)
    pesan = format_telegram_message(df_filtered)

    # Kirim pesan Telegram
    send_telegram_message(BOT_TOKEN, CHAT_ID, pesan)

