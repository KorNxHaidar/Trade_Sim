from strategy.Strategies_template import Strategy_template
import pandas as pd
import numpy as np
from collections import deque

class my_strategy(Strategy_template):
    def __init__(self, handler_from_framework):
        self.handler_obj = handler_from_framework
        super().__init__("TEMP_OWNER", "TEMP_STRATEGY", handler_from_framework)
        self.initialized = False

    # ===== Utility Functions =====
    def relative_strength_index(self, prices, period=14):
        delta = pd.Series(prices).diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        rs = gain / (loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if len(rsi) >= period else None

    def macd_indicator(self, prices, fast_period=5, slow_period=15, signal_period=5):
        price_series = pd.Series(prices)
        ema_fast = price_series.ewm(span=fast_period, adjust=False).mean()
        ema_slow = price_series.ewm(span=slow_period, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        hist = macd_line - signal_line
        if len(macd_line) >= slow_period:
            return macd_line.iloc[-1], signal_line.iloc[-1], hist.iloc[-1]
        else:
            return None, None, None

    # ===== Trading Logic =====
    def on_data(self, row):
        if not self.initialized:
            try:
                self.owner = self.handler_obj.get_owner()
                self.strategy_name = self.__class__.__name__
                self.handler = self.handler_obj

                self.symbol = row['ShareCode']
                self.prices = deque(maxlen=60)
                self.position = 0
                self.trade_volume = 0
                self.buy_price = 0.0
                self.stop_loss_price = None
                self.take_profit_price = None
                self.initialized = True

                # print(f"[{self.symbol}] ✅ Strategy Initialized")
            except Exception as e:
                print(f"[INIT ERROR] {e}")
                return

        try:
            price = row['LastPrice']
            self.prices.append(price)

            # ต้องมีข้อมูลอย่างน้อย 20 จุดเพื่อคำนวณ RSI + MACD
            if len(self.prices) < 20:
                return

            # === คำนวณ RSI และ MACD ===
            rsi_value = self.relative_strength_index(self.prices)
            macd_line, signal_line, hist = self.macd_indicator(self.prices)

            if rsi_value is None or macd_line is None:
                return

            # === Buy Signal ===
            buy_signal = (
                (rsi_value < 40)
                and (macd_line > signal_line)
                and self.position == 0
            )

            # === Sell Signal (จาก RSI/MACD) ===
            sell_signal = (
                (rsi_value > 60)
                and (macd_line < signal_line)
                and self.position == 1
            )

            # === Stop Loss / Take Profit Conditions ===
            stop_loss_trigger = (
                self.position == 1 and price <= self.stop_loss_price
                if self.stop_loss_price else False
            )
            take_profit_trigger = (
                self.position == 1 and price >= self.take_profit_price
                if self.take_profit_price else False
            )

            # === Execute Orders ===
            if buy_signal:
                cash = self.handler.get_cash_balance()
                allocation = np.random.uniform(0.02, 0.03)  # ใช้ 2–3% ของพอร์ตต่อไม้
                allocated_cash = cash * allocation
                self.trade_volume = int(allocated_cash / price)

                if self.trade_volume > 0:
                    self.handler.create_order_to_limit(
                        volume=self.trade_volume,
                        price=price,
                        side="Buy",
                        symbol=self.symbol,
                    )
                    self.position = 1
                    self.buy_price = price
                    self.stop_loss_price = price * 0.95   # stop loss -5%
                    self.take_profit_price = price * 1.02 # take profit +2%

                    # print(
                    #     f"🚀 [BUY] {self.symbol} at {price:.2f} | "
                    #     f"Vol={self.trade_volume:,} | RSI={rsi_value:.1f}"
                    # )

            elif sell_signal or stop_loss_trigger or take_profit_trigger:
                self.handler.create_order_to_limit(
                    volume=self.trade_volume,
                    price=price,
                    side="Sell",
                    symbol=self.symbol,
                )
                reason = (
                    "Stop Loss" if stop_loss_trigger else
                    "Take Profit" if take_profit_trigger else
                    "RSI/MACD Signal"
                )
                # print(f"💰 [SELL] {self.symbol} at {price:.2f} | {reason}")
                self.position = 0
                self.trade_volume = 0
                self.buy_price = 0.0
                self.stop_loss_price = None
                self.take_profit_price = None

        except Exception as e:
            print(f"[{self.symbol}] Error in logic: {e}")