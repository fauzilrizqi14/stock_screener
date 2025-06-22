import os
import requests
import json

# Ambil BOT_TOKEN dari environment variable, sama seperti skrip utama
BOT_TOKEN = os.getenv('BOT_TOKEN')

print("="*30)
print("Memulai Tes Koneksi Telegram...")
print("="*30)

if not BOT_TOKEN:
    print("HASIL: ❌ GAGAL")
    print("Penyebab: BOT_TOKEN tidak ditemukan di environment variable!")
else:
    # Kita hanya print beberapa karakter awal untuk memastikan token tidak kosong
    print(f"Token yang digunakan (sebagian): {BOT_TOKEN[:10]}...")

    # /getMe adalah cara paling dasar untuk mengecek validitas token
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/getMe"

    try:
        print("Menghubungi server Telegram...")
        response = requests.get(url)

        print(f"\nStatus Code diterima: {response.status_code}")
        print("Response Body:")
        # Tampilkan response dalam format JSON yang rapi
        print(json.dumps(response.json(), indent=2))

        if response.status_code == 200:
            print("\nHASIL: ✅ BERHASIL!")
            print("Penjelasan: Token Anda valid dan berhasil terhubung dengan bot.")
        else:
            print("\nHASIL: ❌ GAGAL!")
            print(f"Penjelasan: Server Telegram merespons dengan status {response.status_code}. Ini mengkonfirmasi token di Secret Anda tidak valid.")

    except Exception as e:
        print(f"\nHASIL: ❌ GAGAL")
        print(f"Penyebab: Terjadi error saat melakukan request ke Telegram: {e}")

print("="*30)
print("Tes Selesai.")
print("="*30)
