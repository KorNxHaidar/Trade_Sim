from strategy.Strategies_template import Strategy_template
import pandas as pd
import numpy as np
from collections import deque

class longterm_strategy(Strategy_template):
    def __init__(self, handler_from_framework):
        self.handler_obj = handler_from_framework
        super().__init__("LONG_OWNER", "LONG_STRATEGY", handler_from_framework)
        self.initialized = False

    # ===== Utility Functions =====
    def relative_strength_index(self, prices, period=10):  # ลด period
        delta = pd.Series(prices).diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        rs = gain / (loss + 1e-9)
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if len(rsi) >= period else None

    def macd_indicator(self, prices, fast_period=5, slow_period=20, signal_period=5):
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
                self.prices = deque(maxlen=60)  # เก็บราคาย้อนหลัง ~1 เดือน
                self.position = 0
                self.trade_volume = 0
                self.buy_price = 0.0
                self.stop_loss_price = None
                self.take_profit_price = None
                self.initialized = True
            except Exception as e:
                print(f"[INIT ERROR] {e}")
                return

        try:
            price = row['LastPrice']
            self.prices.append(price)

            if len(self.prices) < 15:  # รอข้อมูลสั้นลง
                return

            rsi_value = self.relative_strength_index(self.prices)
            macd_line, signal_line, hist = self.macd_indicator(self.prices)

            if rsi_value is None or macd_line is None:
                return

            # === Buy Signal (Confident Long-Term) ===
            buy_signal = (
                (rsi_value < 40)       # oversold
                and (macd_line > signal_line)
                and self.position == 0
            )

            # === Sell Signal ===
            sell_signal = (
                (rsi_value > 65)       # overbought
                and (macd_line < signal_line)
                and self.position == 1
            )

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
                allocation = np.random.uniform(0.03, 0.05)  # เพิ่มสัดส่วน 3–5%
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
                    self.stop_loss_price = price * 0.94   # stop loss -6%
                    self.take_profit_price = price * 1.04 # take profit +4%

            elif sell_signal or stop_loss_trigger or take_profit_trigger:
                self.handler.create_order_to_limit(
                    volume=self.trade_volume,
                    price=price,
                    side="Sell",
                    symbol=self.symbol,
                )
                self.position = 0
                self.trade_volume = 0
                self.buy_price = 0.0
                self.stop_loss_price = None
                self.take_profit_price = None

        except Exception as e:
            print(f"[{self.symbol}] Error in logic: {e}")

