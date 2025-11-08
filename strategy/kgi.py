# -*- coding: utf-8 -*-
"""
KGI.py
Adapted for local execution in VS Code from the original Colab notebook.
"""

import os
import glob
import pandas as pd
import numpy as np
from datetime import datetime, time
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from collections import defaultdict

# ============================================================
# 1) การตั้งค่าและจัดการ Path สำหรับ Local (VS Code)
# ============================================================

try:
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # ถ้าเกิดรันในโหมด interactive (ที่ไม่มี __file__) ให้ใช้ path ปัจจุบัน
    SCRIPT_DIR = os.getcwd()

PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# 1. Path ไปยังโฟลเดอร์ผลลัพธ์
output_dir = os.path.join(PROJECT_ROOT, 'competition_api_results')

# 2. Path ไปยังโฟลเดอร์ที่เก็บไฟล์ Ticks
ticks_folder = os.path.join(PROJECT_ROOT, 'marketInfo', 'ticks')

# 3. Path ของไฟล์ Ticks ที่จะรวม
ticks_glob_path = os.path.join(ticks_folder, '*.csv')

# 4. Path ที่จะเซฟไฟล์ Ticks ที่รวมแล้ว
merged_ticks_path = os.path.join(ticks_folder, 'merged_ticks.csv')

print(f"Project Root: {PROJECT_ROOT}")
print(f"Output Dir: {output_dir}")
print(f"Ticks Folder: {ticks_folder}")

# ============================================================
# 2) การตั้งค่าและการจัดการไฟล์
# ============================================================

team_name = 'menemanemo'
trading_day = 1

# ตรวจสอบและสร้าง Directory หลักสำหรับผลลัพธ์
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Created main directory: {output_dir}")
else:
    print(f"Using existing directory: {output_dir}")

# ============================================================
# 3) ฟังก์ชันโหลดข้อมูลก่อนหน้า
# ============================================================
def load_previous(file_type, teamName):
    """โหลดข้อมูล portfolio/statement/summary จาก local drive ถ้ามี"""
    folder_path = os.path.join(output_dir, "Previous", file_type)
    file_path = os.path.join(folder_path, f"{teamName}_{file_type}.csv")

    if os.path.exists(file_path):
        try:
            data = pd.read_csv(file_path)
            print(f"Loaded '{file_type}' data for team {teamName}.")
            return data
        except Exception as e:
            print(f"Error loading file: {e}")
            return None
    else:
        print(f"File not found: {file_path}")
        return None

# ============================================================
# 4) ฟังก์ชันบันทึกผลลัพธ์ใหม่
# ============================================================
def save_output(data, file_type, teamName):
    """บันทึก DataFrame (portfolio, statement, summary) ลง local drive"""
    folder_path = os.path.join(output_dir, "Result", file_type)
    file_path = os.path.join(folder_path, f"{teamName}_{file_type}.csv")

    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        print(f"Directory created: '{folder_path}'")

    data.to_csv(file_path, index=False)
    print(f"{file_type} saved at {file_path}")

# ============================================================
# 5) รวมไฟล์ tick ทั้งหมดในโฟลเดอร์ /ticks
# ============================================================
# ใช้ Path ที่เราสร้างไว้ในข้อ 1
csv_files = glob.glob(ticks_glob_path)

print("Found CSV files:", csv_files)

if csv_files:
    df = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)
    
    # เปลี่ยน display() เป็น print()
    print("--- Merged DataFrame Head ---")
    print(df.head())
    print("-----------------------------")

    # บันทึกไฟล์ที่รวมแล้วกลับไป
    df.to_csv(merged_ticks_path, index=False)
    print(f"บันทึกไฟล์รวมแล้วที่: {merged_ticks_path}")
    print("จำนวนแถวทั้งหมด:", len(df))
else:
    print(f"ไม่พบไฟล์ CSV ในโฟลเดอร์: {ticks_glob_path}")
    # สร้าง DataFrame ว่างๆ เพื่อให้โค้ดส่วนต่อไปไม่ error
    df = pd.DataFrame(columns=['ShareCode', 'TradeDateTime', 'LastPrice', 'Volume', 'Value', 'Flag'])

# # ============================================================
# # 6) ทดลองโหลดข้อมูลวันก่อนหน้า (ถ้ามี)
# # ============================================================
# portfolio_prev = load_previous("portfolio", team_name)
# statement_prev = load_previous("statement", team_name)
# summary_prev = load_previous("Summary", team_name)

# # ============================================================
# # 🎯 TASK 2: การโหลดและเตรียมข้อมูล (Data Loading and Preprocessing)
# # (ลบส่วนที่ซ้ำซ้อนกับข้างบนออก)
# # ============================================================

# statements = [] # ย้าย 'statements = []' มาไว้ตรงนี้

# # กำหนดตัวแปรเริ่มต้น
# buy_close_time = time(16, 30)
# sell_close_time = time(16, 30)
# count_win = 0
# count_sell = 0
# initial_investment = 10000000
# start_vol = 0

# # Load the summary file and set initial balance
# prev_summary_df = load_previous("summary", team_name)

# Start_Line_available = initial_investment

# if prev_summary_df is not None:
#     if 'End Line available' in prev_summary_df.columns:
#         initial_balance_series = prev_summary_df['End Line available']

#         if not initial_balance_series.empty:
#             first_value = initial_balance_series.iloc[0]

#             try:
#                 initial_balance = float(str(first_value).replace(',', '').strip())
#                 Start_Line_available = initial_balance
#                 print("End Line available column loaded successfully.")
#                 print(f"Initial balance (first value): {initial_balance}")
#             except ValueError:
#                 print(f"Error converting '{first_value}' to a float.")
#                 initial_balance = initial_investment
#         else:
#             print("'End Line available' column is empty.")
#             initial_balance = initial_investment
#     else:
#         print("'End Line available' column not found in the file.")
#         initial_balance = initial_investment
# else:
#     initial_balance = initial_investment
#     print(f"Initial balance = initial_investment: {initial_investment}")


# # ตรวจสอบคอลัมน์ที่จำเป็น
# required_columns = ['ShareCode', 'TradeDateTime', 'LastPrice', 'Volume', 'Value', 'Flag']

# if not df.empty:
#     missing_columns = [col for col in required_columns if col not in df.columns]
#     if missing_columns:
#         raise ValueError(f"Missing required columns: {missing_columns}")

#     # แปลงข้อมูลวันที่และเวลาของการเทรด
#     df['TradeDateTime'] = pd.to_datetime(df['TradeDateTime'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
#     df.dropna(subset=['TradeDateTime'], inplace=True)
# else:
#     print("DataFrame is empty, skipping preprocessing.")


# # ฟังก์ชันสำหรับกรองข้อมูลเฉพาะหุ้น
# def filter_data(df, share_code):
#     data = df[df['ShareCode'] == share_code].copy()
#     data.sort_values('TradeDateTime', inplace=True)
#     data.reset_index(drop=True, inplace=True)
#     return data

# # รายชื่อหุ้นใน SET50
# set50_shares = [
#     'ADVANC', 'AOT', 'AWC', 'BBL', 'BCP', 'BDMS', 'BEM', 'BGRIM', 'BH', 'BJC',
#     'BTS', 'CBG', 'CENTEL', 'CPALL', 'CPF', 'CPN', 'CRC', 'DELTA', 'EA', 'EGCO',
#     'GLOBAL', 'GPSC', 'GULF', 'HMPRO', 'INTUCH', 'ITC', 'IVL', 'KBANK', 'KTB',
#     'KTC', 'LH', 'MINT', 'MTC', 'OR', 'OSP', 'PTT', 'PTTEP', 'PTTGC', 'RATCH',
#     'SCB', 'SCC', 'SCGP', 'TIDLOR', 'TISCO', 'TLI', 'TOP', 'TRUE', 'TTB', 'TU', 'WHA'
# ]

# # รายชื่อหุ้นใน SET50
# #set15_shares = [
# #    'BBL', 'DELTA', 'ADVANC', 'BH', 'BCP', 'KBANK', 'KTB', 'M', 'PTTEP', 'SIRI',
# #    'TTB', 'WHA', 'WHAUP', 'TCAP', 'COM7'
# #]

# # ใช้ลูปในการกรองข้อมูล
# filtered_data = {share: filter_data(df, share) for share in set50_shares}

# # กำหนดพอร์ตหุ้นที่ต้องการ (ใช้สำหรับวนซ้ำ)
# portfolio = {stock: 1 for stock in set50_shares}

# portfolio_volumes = {stock: 0 for stock in portfolio.keys()}
# portfolio_amount_cost = {stock: 0 for stock in portfolio}
# portfolio_average_cost = {stock: 0 for stock in portfolio}

# ################################################################################################################################
# ##  TASK 3: การสร้างสัญญาณซื้อขาย (Signal Generation) - แก้ไขแล้ว
# ################################################################################################################################

# # ฟังก์ชันคำนวณ Bollinger Bands (ถูกคอมเมนต์: ต้องแน่ใจว่าไม่ได้ถูกเรียกใช้)
# # def bollinger_bands(data, window=8, no_of_std=0.2):
# #     data['SMA'] = data['LastPrice'].rolling(window=window).mean()
# #     # ... (โค้ดส่วนอื่น) ...
# #     return data

# # ฟังก์ชันคำนวณ RSI (ใช้สำหรับการสร้างสัญญาณ)
# def relative_strength_index(data, period=14):
#     """Calculates the Relative Strength Index (RSI)."""
#     delta = data['LastPrice'].diff()
#     gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
#     loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

#     # Avoid division by zero
#     rs = gain / loss
#     data['RSI'] = 100 - (100 / (1 + rs))
#     return data

# # ฟังก์ชันคำนวณ Stochastic Oscillator (ถูกคอมเมนต์: ต้องแน่ใจว่าไม่ได้ถูกเรียกใช้)
# # def stochastic_oscillator(data, period=8):
# #     # ... (โค้ดส่วนอื่น) ...
# #     return data

# # ฟังก์ชันคำนวณ MACD (ใช้สำหรับการสร้างสัญญาณ)
# def macd_indicator(data, fast_period=5, slow_period=15, signal_period=5):
#     """Calculates the Moving Average Convergence Divergence (MACD)."""
#     # 1. Calculate EMAs
#     data['EMA_Fast'] = data['LastPrice'].ewm(span=fast_period, adjust=False).mean()
#     data['EMA_Slow'] = data['LastPrice'].ewm(span=slow_period, adjust=False).mean()

#     # 2. Calculate MACD Line
#     data['MACD'] = data['EMA_Fast'] - data['EMA_Slow']

#     # 3. Calculate Signal Line
#     data['Signal_Line'] = data['MACD'].ewm(span=signal_period, adjust=False).mean()

#     # 4. Calculate MACD Histogram
#     data['MACD_Hist'] = data['MACD'] - data['Signal_Line']
#     return data

# # ฟังก์ชันสร้างสัญญาณซื้อ/ขาย (ปรับปรุง: ใช้ RSI และ MACD)
# def generate_signals(data):
#     # ต้องเรียกใช้เฉพาะฟังก์ชันที่ไม่ถูกคอมเมนต์และใช้ในการสร้างสัญญาณ
#     data = relative_strength_index(data)
#     data = macd_indicator(data)

#     # 1. Buy Signal (RSI Oversold + MACD Bullish Crossover)
#     data['Buy Signal'] = np.where(
#         (data['RSI'] < 45) & # RSI อยู่ในโซน Oversold (ใช้ 40 เป็นระดับ Conservative)
#         (data['MACD'].shift(1) <= data['Signal_Line'].shift(1)) & # MACD อยู่ต่ำกว่า/เท่ากับ Signal Line ใน Tick ก่อนหน้า
#         (data['MACD'] > data['Signal_Line']),                       # MACD ตัดขึ้นเหนือ Signal Line ใน Tick ปัจจุบัน
#         1, 0
#     )

#     # 2. Sell Signal (RSI Overbought + MACD Bearish Crossover)
#     data['Sell Signal'] = np.where(
#         (data['RSI'] > 55) & # RSI อยู่ในโซน Overbought (ใช้ 60 เป็นระดับ Conservative)
#         (data['MACD'].shift(1) >= data['Signal_Line'].shift(1)) & # MACD อยู่สูงกว่า/เท่ากับ Signal Line ใน Tick ก่อนหน้า
#         (data['MACD'] < data['Signal_Line']),                       # MACD ตัดลงใต้ Signal Line ใน Tick ปัจจุบัน
#         -1, 0
#     )

#     return data

# # ประมวลผลสัญญาณการซื้อขาย
# stock_data = {}
# for stock in portfolio.keys():
#     data = df[df['ShareCode'] == stock].copy()
#     data.sort_values('TradeDateTime', inplace=True)
#     # ต้องมั่นใจว่าฟังก์ชัน generate_signals() ที่ถูกเรียกใช้เป็นตัวที่แก้ไขแล้ว
#     data = generate_signals(data)
#     stock_data[stock] = data

# # รวมสัญญาณซื้อขายทั้งหมดเรียงตามเวลา
# all_trades = []
# for stock in portfolio.keys():
#     data = stock_data[stock].copy()
#     data['Stock Name'] = stock
#     all_trades.append(data[['TradeDateTime', 'Stock Name', 'LastPrice', 'Volume', 'Flag','Buy Signal', 'Sell Signal']])

# all_trades_df = pd.concat(all_trades).sort_values('TradeDateTime').reset_index(drop=True)
# # ################################################################################################################################
# # ## TASK 4: การจำลองการซื้อขายและบันทึก Statement (Trading Simulation)
# # ################################################################################################################################
# # (ใช้เวอร์ชันที่ Active ล่าสุด)

# # ตัวแปรเริ่มต้น
# # ใช้ initial_balance ที่โหลดมาจาก TASK 2
# initial_balance = Start_Line_available 
# print(f"Starting simulation with balance: {initial_balance}")

# buy_close_time = time(10, 30) # เวลาที่ใช้ในโค้ดตัวอย่าง
# sell_close_time = time(14, 30) # เวลาที่ใช้ในโค้ดตัวอย่าง

# # portfolio
# portfolio_volumes = defaultdict(int)
# portfolio_amount_cost = defaultdict(float)
# portfolio_average_cost = defaultdict(float)

# # counters 
# count_sell = 0
# count_win = 0

# # dictionary สำหรับ statement 
# statement_data = {
#     'Table Name': [], 'File Name': [], 'Stock Name': [], 'Date': [], 'Time': [],
#     'Side': [], 'Volume': [], 'Price': [], 'Amount Cost': [], 'End Line Available': []
# }

# # ตรวจสอบว่า DataFrame ไม่ว่าง 
# if all_trades_df.empty:
#     print("all_trades_df ว่าง ไม่มีข้อมูล")
# else:
#     for index, row in all_trades_df.iterrows():
#         trade_date = row['TradeDateTime'].date()
#         trade_time = row['TradeDateTime'].time()
#         stock = row['Stock Name']
#         price = row['LastPrice']
#         volume = row['Volume']
#         buy = row['Buy Signal']
#         sell = row['Sell Signal']
#         flag = row['Flag']

#         # DEBUG (สามารถ comment ออกได้)
#         # print(f"Row {index}: Stock={stock}, Volume={volume}, Price={price}, Buy={buy}, Sell={sell}, Flag={flag}")

#         # การซื้อ (Buy) 
#         # ลบเงื่อนไข volume % 100 และเวลา เพื่อให้เกิด Buy แน่นอน (ตามโค้ดตัวอย่าง)
#         if buy == 1 and initial_balance >= (volume * price) and volume > 0: # เพิ่ม volume > 0
#             amount_cost = volume * price
#             initial_balance -= amount_cost

#             # อัปเดตพอร์ต
#             portfolio_volumes[stock] += volume
#             portfolio_amount_cost[stock] += amount_cost
#             portfolio_average_cost[stock] = portfolio_amount_cost[stock] / portfolio_volumes[stock]

#             # บันทึก statement
#             statement_data['Table Name'].append('Statement_file')
#             statement_data['File Name'].append(team_name)
#             statement_data['Stock Name'].append(stock)
#             statement_data['Date'].append(trade_date)
#             statement_data['Time'].append(trade_time)
#             statement_data['Side'].append('Buy')
#             statement_data['Volume'].append(volume)
#             statement_data['Price'].append(price)
#             statement_data['Amount Cost'].append(amount_cost)
#             statement_data['End Line Available'].append(initial_balance)

#         # การขาย (Sell)
#         # ลบเงื่อนไข volume % 100 และเวลา เพื่อให้เกิด Sell แน่นอน (ตามโค้ดตัวอย่าง)
#         elif sell == -1:
#             actual_vol = min(volume, portfolio_volumes[stock])
#             if actual_vol > 0:
#                 count_sell += 1
#                 amount_revenue = actual_vol * price
#                 initial_balance += amount_revenue

#                 # คำนวณ Realized P/L
#                 cost_of_sold_shares = actual_vol * portfolio_average_cost[stock]

#                 if price > portfolio_average_cost[stock]:
#                     count_win += 1

#                 # อัปเดตพอร์ต
#                 portfolio_volumes[stock] -= actual_vol
#                 portfolio_amount_cost[stock] -= cost_of_sold_shares
#                 portfolio_average_cost[stock] = (portfolio_amount_cost[stock] / portfolio_volumes[stock]) if portfolio_volumes[stock] > 0 else 0

#                 # บันทึก statement
#                 statement_data['Table Name'].append('Statement_file')
#                 statement_data['File Name'].append(team_name)
#                 statement_data['Stock Name'].append(stock)
#                 statement_data['Date'].append(trade_date)
#                 statement_data['Time'].append(trade_time)
#                 statement_data['Side'].append('Sell')
#                 statement_data['Volume'].append(actual_vol)
#                 statement_data['Price'].append(price)
#                 statement_data['Amount Cost'].append(amount_revenue)
#                 statement_data['End Line Available'].append(initial_balance)

# # ----- คำนวณ Win Rate -----
# win_rate = (count_win / count_sell) * 100 if count_sell > 0 else 0

# # ----- สร้าง DataFrame Statement -----
# statement_df = pd.DataFrame(statement_data)
# pd.set_option('display.max_columns', None)
# pd.options.display.float_format = '{:.2f}'.format

# print("\n=== Trading Statement ===")
# if not statement_df.empty:
#     print(statement_df.to_string(index=False))
# else:
#     print("No statements generated.")
# print(f"\nWin Rate: {win_rate:.2f}%")

# ####################################################################
# ## 🎯 TASK 5: การสร้างตาราง Portfolio (Portfolio Summary)
# ####################################################################
# # (ใช้เวอร์ชันที่ Active ล่าสุด)

# # dictionary สำหรับ portfolio
# portfolio_data = {
#     'Table Name': [], 'File Name': [], 'Stock name': [], 'Start Vol': [], 'Actual Vol': [],
#     'Avg Cost': [], 'Market Price': [], 'Market Value': [], 'Amount Cost': [],
#     'Unrealized P/L': [], '% Unrealized P/L': [], 'Realized P/L': []
# }

# # กำหนด start_vol สำหรับแต่ละหุ้น (สมมติว่าเท่ากับจำนวนหุ้นเริ่มต้น = 0)
# start_vol_dict = defaultdict(int) 

# # เติมข้อมูลลงใน portfolio_data
# # วนลูปจาก stock ทั้งหมดที่ *เคยมี* การเคลื่อนไหว (portfolio_volumes)
# for stock in portfolio_volumes.keys():
#     stock_df_data = filtered_data.get(stock) # ดึงข้อมูลดิบของหุ้นนี้
#     avg_cost = portfolio_average_cost.get(stock, 0)
#     actual_vol = portfolio_volumes.get(stock, 0)
#     start_vol = start_vol_dict.get(stock, 0)

#     # ใช้ LastPrice ของ Tick สุดท้ายของหุ้นนั้นเป็น Market Price
#     market_price = 0
#     if stock_df_data is not None and not stock_df_data.empty:
#         market_price = stock_df_data['LastPrice'].iloc[-1]

#     market_value = actual_vol * market_price
#     amount_cost = actual_vol * avg_cost
#     unrealized_pl = market_value - amount_cost
#     percent_unrealized_pl = (unrealized_pl / amount_cost * 100) if amount_cost != 0 else 0

#     realized_pl = 0 # Placeholder

#     # บันทึกลง dictionary
#     portfolio_data['Table Name'].append('Portfolio_file')
#     portfolio_data['File Name'].append(team_name)
#     portfolio_data['Stock name'].append(stock)
#     portfolio_data['Start Vol'].append(start_vol)
#     portfolio_data['Actual Vol'].append(actual_vol)
#     portfolio_data['Avg Cost'].append(avg_cost)
#     portfolio_data['Market Price'].append(market_price)
#     portfolio_data['Market Value'].append(market_value)
#     portfolio_data['Amount Cost'].append(amount_cost)
#     portfolio_data['Unrealized P/L'].append(unrealized_pl)
#     portfolio_data['% Unrealized P/L'].append(percent_unrealized_pl)
#     portfolio_data['Realized P/L'].append(realized_pl)

# # แปลงข้อมูลเป็น DataFrame
# portfolio_df = pd.DataFrame(portfolio_data)

# # ตั้งค่าการแสดงผลตัวเลข
# pd.set_option('display.max_columns', None)
# pd.options.display.float_format = '{:.2f}'.format

# # แสดงตาราง Portfolio
# print("\n=== Portfolio Summary ===")
# if not portfolio_df.empty:
#     print(portfolio_df.to_string(index=False))
# else:
#     print("Portfolio is empty.")


# ####################################################################
# ## 🎯 TASK 6: การสร้างตาราง Summary (Summary Calculation)
# ####################################################################
# # (ใช้เวอร์ชันที่ Active ล่าสุด)

# # กำหนดค่าเริ่มต้น
# # Start_Line_available ถูกกำหนดไว้แล้วใน Task 2 และ 4
# initial_investment = Start_Line_available 
# trading_day_str = pd.to_datetime("today").date()  # สามารถปรับตามจริง

# # ----- ค่าล่าสุดของ End Line Available -----
# last_end_line_available = statement_df['End Line Available'].iloc[-1] if not statement_df.empty else Start_Line_available

# # ----- คำนวณ NAV และ Realized P/L -----
# final_nav = portfolio_df['Market Value'].sum() + last_end_line_available
# total_realized_pl = final_nav - Start_Line_available - portfolio_df['Unrealized P/L'].sum()

# # ----- ตรวจสอบจำนวนรายการต่าง ๆ -----
# num_transactions = len(statement_df) if not statement_df.empty else 0
# max_end_line = statement_df['End Line Available'].max() if not statement_df.empty else Start_Line_available
# min_end_line = statement_df['End Line Available'].min() if not statement_df.empty else Start_Line_available

# # ----- dictionary สำหรับ summary -----
# summary_data = {
#     'Table Name': ['Sum_file'],
#     'File Name': [team_name],
#     'trading_day': [trading_day_str],
#     'NAV': [final_nav],
#     'Portfolio value': [portfolio_df['Market Value'].sum()],
#     'End Line available': [last_end_line_available],
#     'Start Line available':[Start_Line_available],
#     'Number of wins': [count_win],
#     'Number of matched trades': [count_sell],
#     'Number of transactions': [num_transactions],
#     'Net Amount': [statement_df['Amount Cost'].sum() if not statement_df.empty else 0],
#     'Unrealized P/L': [portfolio_df['Unrealized P/L'].sum()],
#     '% Unrealized P/L': [(portfolio_df['Unrealized P/L'].sum() / initial_investment * 100) if initial_investment else 0],
#     'Realized P/L': [total_realized_pl],
#     'Maximum value': [max_end_line],
#     'Minimum value': [min_end_line],
#     'Win rate': [win_rate],
#     'Calmar Ratio': [0],           # Placeholder
#     'Relative Drawdown': [0],      # Placeholder
#     'Maximum Drawdown': [0],       # Placeholder
#     '%Return': [((final_nav - Start_Line_available) / Start_Line_available * 100) if Start_Line_available else 0]
# }

# # ----- สร้าง Summary DataFrame -----
# summary_df = pd.DataFrame(summary_data)

# # ----- ตั้งค่าแสดงผล -----
# pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', None)
# pd.options.display.float_format = '{:.4f}'.format

# # ----- แสดงผล -----
# print("\n=== Trading Summary ===")
# print(summary_df.to_string(index=False, header=True))

# ################################################################################################################################
# ## 🎯 TASK 7: บันทึกผลลัพธ์ (Saving Output)
# ################################################################################################################################

# if not portfolio_df.empty:
#     save_output(portfolio_df, "portfolio", team_name)
# if not statement_df.empty:
#     save_output(statement_df, "statement", team_name)
# if not summary_df.empty:
#     save_output(summary_df, "summary", team_name)

# ################################################################################################################################
# ## 🎯 TASK 8: สร้างกราฟแสดงเงินสะสมของพอร์ต (Equity Curve)
# ################################################################################################################################

# # ตรวจสอบว่ามีข้อมูลใน statement_df หรือไม่
# if not statement_df.empty:
#     # 1. เตรียมข้อมูลสำหรับกราฟ
#     statement_df['TradeDateTime'] = pd.to_datetime(
#         statement_df['Date'].astype(str) + ' ' + statement_df['Time'].astype(str)
#     )

#     # ใช้ข้อมูล ณ สิ้นสุดแต่ละธุรกรรม และเรียงลำดับเวลา
#     equity_data = statement_df[['TradeDateTime', 'End Line Available']].copy()
#     equity_data.sort_values(by='TradeDateTime', inplace=True)
#     equity_data = equity_data.drop_duplicates(subset=['TradeDateTime'], keep='last').reset_index(drop=True)
    
#     # เพิ่มจุดเริ่มต้น (Start_Line_available)
#     start_point = pd.DataFrame({
#         'TradeDateTime': [equity_data['TradeDateTime'].min() - pd.Timedelta(seconds=1)],
#         'End Line Available': [Start_Line_available]
#     })
#     equity_data = pd.concat([start_point, equity_data], ignore_index=True)


#     # 2. พล็อตกราฟ
#     plt.figure(figsize=(12, 6))
#     plt.plot(
#         equity_data['TradeDateTime'],
#         equity_data['End Line Available'],
#         label='End Line Available (Equity)',
#         color='#007ACC', # สีฟ้าเข้ม
#         marker='o',
#         markersize=3,
#         linestyle='-'
#     )

#     # กำหนดรูปแบบแกน X ให้แสดงวันที่และเวลา
#     formatter = mdates.DateFormatter('%Y-%m-%d %H:%M')
#     plt.gca().xaxis.set_major_formatter(formatter)
#     plt.gcf().autofmt_xdate() # หมุนวันที่เพื่อให้อ่านง่าย

#     # กำหนดเงินลงทุนเริ่มต้นเพื่อเป็นเส้นอ้างอิง
#     initial_investment_value = Start_Line_available # ใช้ตัวแปรที่ถูกต้อง
#     plt.axhline(y=initial_investment_value, color='red', linestyle='--', linewidth=1, label='Initial Investment')


#     plt.title(f'Graph showing portfolio accumulation (Equity Curve) - {team_name}', fontsize=16)
#     plt.xlabel('(Trade Date & Time)', fontsize=12)
#     plt.ylabel('Account balance (End Line Available)', fontsize=12)
#     plt.ticklabel_format(style='plain', axis='y') # ปรับแกน Y ไม่ให้เป็น scientific notation
#     plt.grid(True, linestyle=':', alpha=0.7)
#     plt.legend(loc='upper left')
#     plt.tight_layout()

#     # 3. บันทึกรูปภาพ
#     graph_folder = os.path.join(output_dir, "Result", "Graphs")
#     os.makedirs(graph_folder, exist_ok=True)

#     graph_file_name = f"{team_name}_Accumulated_Balance_Graph.png"
#     full_graph_path = os.path.join(graph_folder, graph_file_name)

#     plt.savefig(full_graph_path)

#     print("\n" + "="*80)
#     print(f"The graph shows the accumulated balance of the portfolio recorded at: {full_graph_path}")
#     print("================================================================================\n")

#     # 4. แสดงกราฟ (สำคัญสำหรับ VS Code)
#     print("Displaying graph...")
#     plt.show()

# else:
#     print("\nNo data in statement_df to plot graph.")

# print("\n--- Script execution finished ---")