# run_simulation.py
import os
import pandas as pd
from strategy.my_strategy import SimpleBuyLowStrategy 

# --- 1. สร้างตัว Handler จำลอง ---
# ในโปรเจกต์จริง 'handler' จะเป็นตัวส่งคำสั่ง
# แต่ตอนนี้เราจำลองมันขึ้นมาง่ายๆ ให้มันแค่ print
class MockHandler:
    def buy(self, symbol, price, time):
        print(f"✅✅✅ [BUY ORDER] {time} | ซื้อ {symbol} ที่ราคา {price}")
    
    def sell(self, symbol, price, time):
        print(f"❌❌❌ [SELL ORDER] {time} | ขาย {symbol} ที่ราคา {price}")

# --- 2. ตั้งค่าและเตรียมการ ---
try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # ถ้าเกิดรันในโหมด interactive (ที่ไม่มี __file__) ให้ใช้ path ปัจจุบัน
    SCRIPT_DIR = os.getcwd()

PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
ticks_folder = os.path.join(PROJECT_ROOT, 'marketInfo', 'ticks')
DATA_FILE_PATH = os.path.join(ticks_folder, '2025-09-17.csv')
SYMBOL_TO_TRACK = 'ADVANC'

# สร้าง Handler
my_handler = MockHandler()

# สร้าง Strategy ของเรา
# สังเกตว่าเราส่ง my_handler เข้าไปใน strategy
my_strategy = SimpleBuyLowStrategy(
    owner='Korn', 
    strategy_name='Simple_ADVANC_Trader', 
    handler=my_handler,
    symbol_to_track=SYMBOL_TO_TRACK
)

# --- 3. อ่านข้อมูลและเริ่มจำลอง ---
print(f"--- เริ่มการจำลองโดยใช้ไฟล์: {DATA_FILE_PATH} ---")

try:
    # อ่านไฟล์ CSV ด้วย pandas
    df = pd.read_csv(DATA_FILE_PATH)
    
    # ตรวจสอบคอลัมน์ (เผื่อชื่อไม่ตรง)
    print(f"พบคอลัมน์: {list(df.columns)}")

    # วนลูปข้อมูลทีละแถว
    # df.iterrows() จะส่งค่า (index, row) ออกมา
    for index, row in df.iterrows():
        # นี่คือจุดที่ "เชื่อมต่อ" กันครับ
        # เราส่งข้อมูลทีละแถว (row) ไปให้กลยุทธ์ของเรา
        my_strategy.on_data(row)

    print("--- การจำลองเสร็จสิ้น ---")

except FileNotFoundError:
    print(f"!!! เกิดข้อผิดพลาด: ไม่พบไฟล์ {DATA_FILE_PATH}")
except Exception as e:
    print(f"!!! เกิดข้อผิดพลาด: {e}")