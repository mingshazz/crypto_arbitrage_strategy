# Crypto Arbitrage Backtest with Order Book and Bar Data
This project implements a crypto arbitrage backtesting system between two exchanges: Binance (BNC) and Huobi (HB). It simulates executing arbitrage opportunities based on order book depth and filters signals using bar-level data to avoid volatile or illiquid conditions. Post-trade analysis and visualization are included to evaluate strategy performance.

## Strategy Overview
### Objective:
Capture arbitrage opportunities between two crypto exchanges by identifying and executing profitable trades when the spread exceeds transaction costs and the market conditions are stable.
### Arbitrage Direction:
1. A to B: Buy from BNC, sell to HB.
2. B to A: Buy from HB, sell to BNC.
### Data Used:
+ Order Book Data (per second):
  + Top 10 bid prices (b1 to b10) and volumes (bv1 to bv10)
  + Top 10 ask prices (a1 to a10) and volumes (av1 to av10)
+ Bar Data (1s candles):
  + timestamp, symbol, open, high, low, close, volume
### Filtering Criteria:
+ Minimum Spread Threshold: The spread (difference between buy and sell VWAP) must exceed a defined threshold.
+ Volatility Filter: Trades are filtered out when price volatility in the 1-second bar exceeds a defined max_volatility threshold.

## Implementation Details
### Matching Logic
+ VWAP (Volume-Weighted Average Price) is computed for both sides using all matching price levels until available liquidity is exhausted or matching condition fails.
+ The trade proceeds only if:
  + VWAP spread > min_spread
  + volatility (high-low/close) < max_volatility on both exchanges
### Fee Handling
+ A fee_rate is applied to both buy and sell sides to account for trading fees (default: 0.1%).
### Trade Execution
+ Trades are executed using available liquidity and the matching price levels.
+ Trade logs are recorded for each execution including:
  + vTimestamp, direction, VWAPs, spread, cost, revenue, profit, volume traded

## Post-Trade Analysis
The system includes an analyze_results function to generate insights from the trade results:
### Metrics Computed:
+ Total Profit
+ Number of Trades
+ Win Rate (percentage of profitable trades)
+ Average and Median Profit per Trade
+ Sharpe Ratio (if possible)
### Visualizations:
+ Profit over time
+ Cumulative PnL curve
+ Histogram of trade profits
+ Volatility vs Trade Profit scatter plot
### Exports:
+ Results DataFrame to CSV
+ Metrics summary to CSV
+ Graphs exported as PNGs into the output/ folder

## How to Use
+ Prepare order book and bar data for both BNC and HB.
+ Configure thresholds: min_spread, max_volatility, and fee_rate
+ Run the backtest using backtest_arbitrage_with_bar_filter
+ Call analyze_results and plot_trades_analysis to evaluate performance
+ Optional: Export metrics and logs for reporting
