import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from time import sleep, time

# Import pustaka Technical Analysis (ta)
from ta.trend import MACD, ADXIndicator, ema_indicator
from ta.momentum import RSIIndicator, StochRSIIndicator

# ==============================================================================
# 1. KONFIGURASI UTAMA
# ==============================================================================
CREDS_FILE_PATH = os.getenv("GOOGLE_CREDENTIALS_JSON")
CONFIG_SHEET_NAME = 'config_screener' # NAMA FILE GOOGLE SHEET ANDA
BOT_TOKEN = os.getenv('BOT_TOKEN')
CHAT_ID = os.getenv('CHAT_ID')

# ==============================================================================
# 2. FUNGSI-FUNGSI
# ==============================================================================
def calculate_sr_levels(df_hist):
    if len(df_hist) < 2: return None, None
    prev_day = df_hist.iloc[-2]
    H, L, C = prev_day['High'], prev_day['Low'], prev_day['Close']
    P = (H + L + C) / 3
    R1, R2 = (2 * P) - L, P + (H - L)
    S1, S2 = (2 * P) - H, P - (H - L)
    latest = df_hist.iloc[-1]
    last_price = latest['Close']
    ma_levels = [latest.get(ma) for ma in ['SMA20', 'EMA21', 'EMA50'] if latest.get(ma) is not None]
    support_candidates = sorted([S1, S2] + [ma for ma in ma_levels if ma < last_price], reverse=True)
    resistance_candidates = sorted([R1, R2] + [ma for ma in ma_levels if ma > last_price])
    target_sl_price = last_price * 0.95
    target_tp_price = last_price * 1.10
    valid_supports = [s for s in support_candidates if s <= target_sl_price]
    final_support = max(valid_supports) if valid_supports else target_sl_price
    valid_resistances = [r for r in resistance_candidates if r >= target_tp_price]
    final_resistance = min(valid_resistances) if valid_resistances else target_tp_price
    return final_support, final_resistance

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

def load_saham_from_sheet(client, sheet_name):
    try:
        spreadsheet = client.open(sheet_name)
        worksheet = spreadsheet.worksheet("Daftar_Saham")
        records = worksheet.get_all_records()
        df_saham = pd.DataFrame(records)
        if 'Kode' not in df_saham.columns or 'Saham' not in df_saham.columns:
            print("   -> ERROR: Header kolom di sheet 'Daftar_Saham' harus 'Kode' dan 'Saham'.")
            return None, None
        df_saham['Saham'] = df_saham['Saham'].astype(str).str.replace('.', '', regex=False).str.replace(',', '', regex=False)
        df_saham['Saham'] = pd.to_numeric(df_saham['Saham'], errors='coerce')
        df_saham.dropna(subset=['Kode', 'Saham'], inplace=True)
        market_caps_shares = dict(zip(df_saham['Kode'].astype(str), df_saham['Saham']))
        tickers_list = df_saham['Kode'].astype(str) + '.JK'
        tickers_list = tickers_list.tolist()
        print(f"   -> Berhasil memuat {len(tickers_list)} saham dari sheet 'Daftar_Saham'.")
        return market_caps_shares, tickers_list
    except gspread.exceptions.WorksheetNotFound:
        print("   -> ERROR: Sheet dengan nama 'Daftar_Saham' tidak ditemukan.")
        return None, None
    except Exception as e:
        print(f"   -> ERROR saat memuat daftar saham: {e}")
        return None, None

def format_market_cap_trillion(market_cap):
    if pd.isna(market_cap): return "N/A"
    return f"{market_cap / 1_000_000_000_000:.2f} Tn"

def classify_entry_level(condition_keys, score, latest_indicators):
    cond = set(condition_keys)
    is_strong_uptrend = latest_indicators.get("EMA50", 0) > latest_indicators.get("EMA200", 0)
    if is_strong_uptrend:
        continuation_triggers = {"stoch_kd_cross", "ema9_cross_ema21", "macd_histogram_increasing"}
        if cond.intersection(continuation_triggers): return "Lanjutan Tren (Buy on Dip)"
    momentum_core = {"macd_cross_0", "ema9_cross_ema21"}
    momentum_support = {"volume_breakout", "volume_above_avg_2x", "higher_high_low"}
    if cond.issuperset(momentum_core) and cond.intersection(momentum_support): return "Momentum Awal (Breakout)"
    reversal_triggers = {"stoch_kd_cross", "rsi_cross_30", "macd_cross_signal", "bullish_engulfing"}
    if cond.intersection(reversal_triggers) and not cond.intersection(momentum_core): return "Pembalikan Arah (Spekulatif)"
    if score > 2.0: return "Watchlist (Skor Tinggi)"
    return "Unclassified"

# --- FUNGSI TELEGRAM FINAL DENGAN LOGIKA "SMART CHUNKING" ---
def kirim_notifikasi_telegram(df_to_send, token, chat_id, max_items):
    """Membungkus logika pengiriman Telegram dengan penanganan pesan panjang yang cerdas."""
    if token == "GANTI_DENGAN_TOKEN_BOT_ANDA" or chat_id == "GANTI_DENGAN_CHAT_ID_ANDA":
        print("\nPERINGATAN: BOT_TOKEN atau CHAT_ID belum diisi. Notifikasi Telegram dilewati.")
        return

    today_str = datetime.now().strftime("%d %B %Y")
    header = f"📈 *Rekomendasi Saham Hari Ini ({today_str})* 📈\n\n"
    
    # 1. Buat daftar "blok pesan" untuk setiap saham
    message_blocks = []
    for i, row in df_to_send.head(max_items).iterrows():
        price_str = f"{row['Last Price']:,.0f}".replace(',', '.')
        support_str = f"{row['Support']:,.0f}".replace(',', '.')
        resistance_str = f"{row['Resistance']:,.0f}".replace(',', '.')
        
        block = ""
        block += f"*{i+1}. {row['Kode']}* | Price: {price_str} | *{row['MarketCap_Tn']}*\n"
        block += f"   Score: *{row['Score']:.2f}* | Level: *{row['Entry Level']}*\n"
        block += f"   S/R: *{support_str}* / *{resistance_str}*\n"
        block += f"   ↳ _{row['Keterangan']}_\n\n"
        message_blocks.append(block)

    # 2. Rakit pesan ke dalam beberapa bagian (chunks) secara cerdas
    chunks = []
    current_chunk = header
    for block in message_blocks:
        # Jika penambahan blok baru akan melebihi batas, kirim chunk saat ini
        if len(current_chunk) + len(block) > 4096:
            chunks.append(current_chunk)
            current_chunk = "*(Lanjutan...)*\n\n" # Mulai chunk baru dengan header lanjutan
        current_chunk += block
    
    chunks.append(current_chunk) # Jangan lupa tambahkan sisa chunk terakhir

    # 3. Kirim setiap chunk satu per satu
    url = f"https://api.telegram.org/{token}/sendMessage"
    for i, chunk in enumerate(chunks):
        data = {"chat_id": chat_id, "text": chunk, "parse_mode": "Markdown"}
        try:
            response = requests.post(url, data=data)
            if response.status_code == 200:
                print(f"\n✅ Notifikasi Telegram (bagian {i+1}/{len(chunks)}) berhasil terkirim!")
            else:
                print(f"\n❌ Gagal mengirim notifikasi (bagian {i+1}): {response.status_code} - {response.text}")
        except Exception as e:
            print(f"\n❌ Error koneksi saat kirim ke Telegram: {e}")
        
        if len(chunks) > 1 and i < len(chunks) - 1:
            sleep(1) # Beri jeda 1 detik antar pengiriman jika ada lebih dari 1 pesan


def screen_stock_processor(ticker_code, df_hist, market_caps_shares, enabled_signals, signal_weights, MIN_MARKET_CAP, K_THRESHOLD_VALUE):
    try:
        if df_hist.empty or len(df_hist) < 200: return None
        shares_outstanding = market_caps_shares.get(ticker_code)
        if shares_outstanding is None or not isinstance(shares_outstanding, (int, float, np.number)): return None
        last_price = df_hist['Close'].iloc[-1]
        if pd.isna(last_price): return None
        market_cap = shares_outstanding * last_price
        if market_cap < MIN_MARKET_CAP: return None
        df_hist['EMA9'] = ema_indicator(df_hist['Close'], window=9)
        df_hist['EMA21'] = ema_indicator(df_hist['Close'], window=21)
        df_hist['EMA50'] = ema_indicator(df_hist['Close'], window=50)
        df_hist['EMA200'] = ema_indicator(df_hist['Close'], window=200)
        df_hist['SMA20'] = df_hist['Close'].rolling(window=20).mean()
        df_hist['SMA26'] = df_hist['Close'].rolling(window=26).mean()
        df_hist['Vol_SMA20'] = df_hist['Volume'].rolling(window=20).mean()
        macd = MACD(close=df_hist['Close'], window_slow=26, window_fast=12, window_sign=9)
        df_hist['MACD'], df_hist['MACD_signal'], df_hist['MACD_histogram'] = macd.macd(), macd.macd_signal(), macd.macd_diff()
        df_hist['RSI'] = RSIIndicator(close=df_hist['Close'], window=14).rsi()
        stoch_rsi = StochRSIIndicator(close=df_hist['Close'], window=14, smooth1=3, smooth2=3)
        df_hist['K'], df_hist['D'] = stoch_rsi.stochrsi_k(), stoch_rsi.stochrsi_d()
        df_hist['ADX'] = ADXIndicator(df_hist['High'], df_hist['Low'], df_hist['Close'], window=14).adx()
        df_hist.dropna(inplace=True)
        if len(df_hist) < 2: return None
        latest = df_hist.iloc[-1]
        prev = df_hist.iloc[-2]
        if latest['Close'] <= latest['SMA20']: return None
        main_support, main_resistance = calculate_sr_levels(df_hist)
        if main_support is None or main_resistance is None: return None
        score = 0
        conditions_met_labels, conditions_met_keys = [], []
        signal_conditions = {"macd_cross_0":{"check":lambda p,l: p['MACD'] < 0 and l['MACD'] >= 0,"label":"MACD cross up 0"},"macd_cross_signal":{"check":lambda p,l: p['MACD'] < p['MACD_signal'] and l['MACD'] >= l['MACD_signal'] and l['MACD'] < 0,"label":"MACD cross up signal"},"stoch_kd_cross":{"check":lambda p,l: p['K'] < p['D'] and l['K'] >= l['D'] and l['K'] < K_THRESHOLD_VALUE,"label":f"StochRSI K cross D & K < {K_THRESHOLD_VALUE}"},"stoch_k_cross_0":{"check":lambda p,l: p['K'] < 0 and l['K'] >= 0,"label":"StochRSI K cross up 0"},"volume_breakout":{"check":lambda p,l: l['Volume'] > l['Vol_SMA20'],"label":"Volume breakout"},"rsi_cross_30":{"check":lambda p,l: p['RSI'] < 30 and l['RSI'] >= 30,"label":"RSI naik dari bawah 30"},"higher_high_low":{"check":lambda p,l: p['Low'] < l['Low'] and p['High'] < l['High'],"label":"Higher High & Higher Low"},"ema9_cross_ema21":{"check":lambda p,l: p['EMA9'] < p['EMA21'] and l['EMA9'] >= l['EMA21'],"label":"EMA9 cross up EMA21"},"close_cross_sma26":{"check":lambda p,l: p["Close"] < p["SMA26"] and l["Close"] >= l["SMA26"],"label":"Close cross up SMA26"},"adx_above_20":{"check":lambda p,l: l.get("ADX", 0) > 20,"label":"ADX > 20"},"price_above_ema50":{"check":lambda p,l: l["Close"] > l["EMA50"],"label":"Harga > EMA50"},"ema50_above_ema200":{"check":lambda p,l: l["EMA50"] > l["EMA200"],"label":"EMA50 > EMA200 (Golden Cross)"},"macd_histogram_increasing":{"check":lambda p,l: l["MACD_histogram"] > p["MACD_histogram"],"label":"Histogram MACD naik"},"volume_above_avg_2x":{"check":lambda p,l: l["Volume"] > 2 * l["Vol_SMA20"],"label":"Volume > 2x SMA20"},"bullish_engulfing":{"check":lambda p,l: (l["Close"] > l["Open"] and p["Close"] < p["Open"] and l["Close"] > p["Open"] and l["Open"] < p["Close"]),"label":"Bullish Engulfing"}}
        for key, cond_details in signal_conditions.items():
            if enabled_signals.get(key, False):
                try:
                    if cond_details["check"](prev, latest):
                        score += signal_weights.get(key, 0)
                        conditions_met_labels.append(cond_details["label"])
                        conditions_met_keys.append(key)
                except Exception: continue
        if score > 0:
            entry_level = classify_entry_level(conditions_met_keys, score, latest)
            return {"Kode": ticker_code, "Last Price": last_price, "MarketCap": market_cap,"Score": round(score, 3), "Entry Level": entry_level,"Keterangan": ", ".join(conditions_met_labels),"Support": main_support, "Resistance": main_resistance}
    except Exception: return None
    return None

# ==============================================================================
# 4. BLOK EKSEKUSI UTAMA
# ==============================================================================
if __name__ == "__main__":
    start_time = time()
    print("Memulai screener saham...")
    print("1. Menghubungkan ke Google Sheets...")
    try:
        scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name(CREDS_FILE_PATH, scope)
        gspread_client = gspread.authorize(creds)
    except FileNotFoundError:
        print(f"   -> FATAL ERROR: File kredensial '{CREDS_FILE_PATH}' tidak ditemukan.")
        exit()
    except Exception as e:
        print(f"   -> FATAL ERROR: Gagal terhubung ke Google API: {e}.")
        exit()
    print("\n2. Memuat data dan konfigurasi...")
    MIN_SCORE_THRESHOLD, SEND_ONLY_ABOVE_THRESHOLD, MAX_ITEM_TELE, K_THRESHOLD_VALUE, MIN_MARKET_CAP, enabled_signals, signal_weights = load_config_from_sheet(gspread_client, CONFIG_SHEET_NAME)
    market_caps_shares, tickers_list = load_saham_from_sheet(gspread_client, CONFIG_SHEET_NAME)
    if not tickers_list:
        print("\nProgram dihentikan karena daftar saham tidak dapat dimuat.")
        exit()
    print(f"\n3. Mengunduh data historis untuk {len(tickers_list)} saham...")
    download_start = time()
    all_data = yf.download(tickers_list, period="1y", interval="1d", group_by='ticker', auto_adjust=True, progress=False)
    print(f"   -> Data diunduh dalam {time() - download_start:.2f} detik.")
    print("\n4. Memproses setiap saham...")
    processing_start = time()
    results = []
    for ticker in tickers_list:
        ticker_code = ticker.replace('.JK', '')
        try:
            hist_data = all_data[ticker].copy()
            hist_data.dropna(subset=['Close'], inplace=True)
            if hist_data.empty: continue
            res = screen_stock_processor(ticker_code, hist_data, market_caps_shares, enabled_signals, signal_weights, MIN_MARKET_CAP, K_THRESHOLD_VALUE)
            if res:
                results.append(res)
        except (KeyError, IndexError):
            continue
    print(f"   -> Semua saham diproses dalam {time() - processing_start:.2f} detik.")
    print("\n5. Menyusun dan menampilkan hasil...")
    if not results:
        print("\n❗ Tidak ada saham yang memenuhi kriteria screening hari ini.")
    else:
        df_result = pd.DataFrame(results)
        df_result['sort_priority'] = np.where(df_result['Entry Level'] == 'Unclassified', 1, 0)
        df_result.sort_values(by=['sort_priority', 'Score', 'MarketCap'], ascending=[True, False, False], inplace=True)
        df_result.drop(columns=['sort_priority'], inplace=True)
        df_result.reset_index(drop=True, inplace=True)
        if SEND_ONLY_ABOVE_THRESHOLD:
            df_filtered = df_result[df_result["Score"] >= MIN_SCORE_THRESHOLD].copy()
        else:
            df_filtered = df_result.copy()
        if df_filtered.empty:
             print(f"\n❗ Tidak ada saham yang lolos skor minimal ({MIN_SCORE_THRESHOLD}).")
        else:
            df_filtered['MarketCap_Tn'] = df_filtered['MarketCap'].apply(format_market_cap_trillion)
            print("\n==================== HASIL SCREENING SAHAM ====================")
            display_cols = ["Kode", "Last Price", "Support", "Resistance", "MarketCap_Tn", "Score", "Entry Level", "Keterangan"]
            format_dict = {'Last Price': lambda x: f'{x:,.0f}'.replace(',', '.'),'Support': lambda x: f'{x:,.0f}'.replace(',', '.'),'Resistance': lambda x: f'{x:,.0f}'.replace(',', '.')}
            print(df_filtered[display_cols].to_string(formatters=format_dict))
            print("=============================================================")
            # CUKUP UNCOMMENT SATU BARIS DI BAWAH INI UNTUK MENGAKTIFKAN NOTIFIKASI TELEGRAM
            kirim_notifikasi_telegram(df_filtered, BOT_TOKEN, CHAT_ID, MAX_ITEM_TELE)
    print(f"\nProses screening selesai dalam total {time() - start_time:.2f} detik.")
