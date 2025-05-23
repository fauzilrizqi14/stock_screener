import yfinance as yf
import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime

# Load Excel
df = pd.read_excel('https://github.com/fauzilrizqi14/stock_screener/raw/main/Daftar%20Saham%20%20-%2020250523.xlsx')

# Pastikan kolom 'Saham' numeric dan drop data invalid
df['Saham'] = pd.to_numeric(df['Saham'], errors='coerce')
df_shares = df[['Kode', 'Saham']].dropna(subset=['Saham'])

# Buat dict kode saham -> jumlah saham
market_caps_shares = dict(zip(df_shares['Kode'].astype(str), df_shares['Saham']))

# List ticker buat yfinance
tickers = df['Kode'].dropna().astype(str) + '.JK'
tickers = tickers.tolist()

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

def screen_stock(ticker):
    try:
        score = 0
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

        # Filter market cap minimal
        if market_cap < 5_000_000_000_000:
            return None

        df_hist = ticker_obj.history(period="6mo", interval="1d")
        if df_hist.empty or len(df_hist) < 50:
            return None

        df_hist['SMA20'] = df_hist['Close'].rolling(window=20).mean()
        df_hist['K'], df_hist['D'] = compute_stoch_rsi(df_hist['Close'])
        ema_12 = df_hist['Close'].ewm(span=12, adjust=False).mean()
        ema_26 = df_hist['Close'].ewm(span=26, adjust=False).mean()
        df_hist['MACD'] = ema_12 - ema_26
        df_hist['MACD_signal'] = df_hist['MACD'].ewm(span=9, adjust=False).mean()

        latest = df_hist.iloc[-1]
        prev = df_hist.iloc[-2]

        if latest['Close'].item() <= latest['SMA20'].item():
            return None

        conditions_met = []

        if (prev['K'].item() < prev['D'].item()) and (latest['K'].item() >= latest['D'].item()) and (latest['K'].item() < 50):
            score += 4
            conditions_met.append("StochRSI K cross D & K<50")

        if (prev['K'].item() < 0) and (latest['K'].item() >= 0):
            score += 3
            conditions_met.append("StochRSI K cross up 0")

        if (prev['MACD'].item() < 0) and (latest['MACD'].item() >= 0):
            score += 4
            conditions_met.append("MACD level cross up 0")

        if (prev['MACD'].item() < prev['MACD_signal'].item()) and (latest['MACD'].item() >= latest['MACD_signal'].item()):
            score += 5
            conditions_met.append("MACD cross signal")

        if score > 0:
            return {
                "Kode": ticker_code,
                "Last Price": last_price,
                "MarketCap": market_cap,
                "Score": score,
                "Keterangan": ", ".join(conditions_met)
            }

    except Exception as e:
        print(f"❌ Error di {ticker}: {e}")
        return None

    return None

def send_telegram_message(token, chat_id, message):
    token = os.getenv('BOT_TOKEN')
    chat_id = os.getenv('CHAT_ID')
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    data = {"chat_id": chat_id, "text": message}
    response = requests.post(url, data=data)
    if response.status_code == 200:
        print("✅ Notifikasi Telegram terkirim!")
    else:
        print(f"❌ Gagal kirim notifikasi: {response.text}")

def format_telegram_message(df_result, max_items=25):
    today = datetime.now().strftime("%d %B %Y")
    message = f"Ini rekomendasi Saham Buat kamu Hari ini {today}:\n"
    
    for i, row in df_result.head(max_items).iterrows():
        kode = row['Kode']
        score = row['Score']
        keterangan = row['Keterangan']
        message += f"{i+1}. {kode}, {score}. {keterangan}\n"
    return message

results = []
for t in tickers:
    res = screen_stock(t)
    if res is not None:
        results.append(res)

if len(results) == 0:
    print("❗ Tidak ada saham yang memenuhi kriteria screening")
else:
    df_result = pd.DataFrame(results)
    df_result = df_result.sort_values(by=["Score", "MarketCap"], ascending=[False, False])
    df_result.reset_index(drop=True, inplace=True)
    print(df_result)

    # Kirim notif Telegram
    BOT_TOKEN = os.getenv('BOT_TOKEN')  # Use the environment variable for token
    CHAT_ID = os.getenv('CHAT_ID')  # Use the environment variable for chat ID
    pesan = format_telegram_message(df_result)
    send_telegram_message(BOT_TOKEN, CHAT_ID, pesan)
