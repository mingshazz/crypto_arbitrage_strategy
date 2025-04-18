import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import logging
log = logging.getLogger(__name__)

def compute_vwap_with_spread_check(
    buy_prices, buy_volumes, sell_prices, sell_volumes,
    min_spread, max_volatility,
    bar_bnc_row, bar_hb_row,
    symbol_bnc, symbol_hb,
    fee_rate
):
    """
    Check arbitrage opportunity considering spread, volatility, and trading fees.
    Returns trade metadata if valid; else returns None.
    """
    # 1. Volatility check (skip if either is NaN)
    vol_bnc = bar_bnc_row.get('volatility', np.nan)
    vol_hb = bar_hb_row.get('volatility', np.nan)
    if not (np.isnan(vol_bnc) or np.isnan(vol_hb)):
        if vol_bnc > max_volatility or vol_hb > max_volatility:
            return None

    matched_volume = 0
    cost = 0
    revenue = 0

    i = j = 0
    while i < len(sell_prices) and j < len(buy_prices):
        sell_price = sell_prices[i]
        buy_price = buy_prices[j]

        if sell_price > buy_price:
            break  # No arbitrage profit beyond this point

        trade_volume = min(sell_volumes[i], buy_volumes[j])
        cost += sell_price * trade_volume
        revenue += buy_price * trade_volume
        matched_volume += trade_volume

        sell_volumes[i] -= trade_volume
        buy_volumes[j] -= trade_volume

        if sell_volumes[i] == 0:
            i += 1
        if buy_volumes[j] == 0:
            j += 1

    if matched_volume == 0:
        return None

    # 2. Compute gross metrics
    avg_sell_price = cost / matched_volume
    avg_buy_price = revenue / matched_volume
    gross_spread = avg_buy_price - avg_sell_price
    gross_profit = revenue - cost

    # 3. Apply fee on both sides
    fee_cost = fee_rate * cost
    fee_revenue = fee_rate * revenue
    net_profit = gross_profit - fee_cost - fee_revenue
    net_spread = net_profit / matched_volume

    if net_spread < min_spread:
        return None

    # 4. Determine direction
    direction = 'bnc_to_hb' if 'BNC' in symbol_bnc.upper() and 'HB' in symbol_hb.upper() else 'hb_to_bnc'

    return {
        'timestamp': bar_bnc_row['timestamp'],
        'buy_exchange': 'hb' if direction == 'bnc_to_hb' else 'bnc',
        'sell_exchange': 'bnc' if direction == 'bnc_to_hb' else 'hb',
        'symbol_bnc': symbol_bnc,
        'symbol_hb': symbol_hb,
        'executed_price_bnc': avg_sell_price if direction == 'bnc_to_hb' else avg_buy_price,
        'executed_price_hb': avg_buy_price if direction == 'bnc_to_hb' else avg_sell_price,
        'direction': direction,
        'volume': matched_volume,
        'spread': net_spread,
        'profit': net_profit,
        'fee_cost': fee_cost,
        'fee_revenue': fee_revenue
    }


def backtest_arbitrage_with_bar_filter(order_book_bnc, order_book_hb, bar_bnc, bar_hb, min_spread=0.1, max_volatility=0.05, fee_rate=0.001):
    import pandas as pd
    import numpy as np

    results = []

    # Convert bar data timestamps to set for intersection
    timestamps = set(bar_bnc['timestamp']).intersection(set(bar_hb['timestamp']))
    timestamps = sorted(timestamps)

    for timestamp in timestamps:
        ob_bnc = order_book_bnc[order_book_bnc['timestamp'] == timestamp]
        ob_hb = order_book_hb[order_book_hb['timestamp'] == timestamp]

        if ob_bnc.empty or ob_hb.empty:
            continue

        bar_bnc_row = bar_bnc[bar_bnc['timestamp'] == timestamp].iloc[0] if not bar_bnc[bar_bnc['timestamp'] == timestamp].empty else None
        bar_hb_row = bar_hb[bar_hb['timestamp'] == timestamp].iloc[0] if not bar_hb[bar_hb['timestamp'] == timestamp].empty else None

        if bar_bnc_row is None or bar_hb_row is None:
            continue

        # A to B: Buy from bnc, sell to hb
        sell_prices_bnc = [ob_bnc[f'a{i}'].values[0] for i in range(1, 11)]
        sell_volumes_bnc = [ob_bnc[f'av{i}'].values[0] for i in range(1, 11)]
        buy_prices_hb = [ob_hb[f'b{i}'].values[0] for i in range(1, 11)]
        buy_volumes_hb = [ob_hb[f'bv{i}'].values[0] for i in range(1, 11)]

        result_ab = compute_vwap_with_spread_check(
            buy_prices_hb, buy_volumes_hb,
            sell_prices_bnc, sell_volumes_bnc,
            min_spread, max_volatility,
            bar_bnc_row, bar_hb_row,
            ob_bnc['symbol'].values[0], ob_hb['symbol'].values[0],
            fee_rate=fee_rate
        )
        if result_ab:
            results.append(result_ab)
            continue

        # B to A: Buy from hb, sell to bnc
        sell_prices_hb = [ob_hb[f'a{i}'].values[0] for i in range(1, 11)]
        sell_volumes_hb = [ob_hb[f'av{i}'].values[0] for i in range(1, 11)]
        buy_prices_bnc = [ob_bnc[f'b{i}'].values[0] for i in range(1, 11)]
        buy_volumes_bnc = [ob_bnc[f'bv{i}'].values[0] for i in range(1, 11)]

        result_ba = compute_vwap_with_spread_check(
            buy_prices_bnc, buy_volumes_bnc,
            sell_prices_hb, sell_volumes_hb,
            min_spread, max_volatility,
            bar_bnc_row, bar_hb_row,
            ob_bnc['symbol'].values[0], ob_hb['symbol'].values[0],
            fee_rate=fee_rate
        )
        if result_ba:
            results.append(result_ba)

    results_df = pd.DataFrame(results)
    return results_df




def analyze_results(results_df):
    results_df = results_df.copy()
    results_df["timestamp"] = pd.to_datetime(results_df["timestamp"])

    # Cumulative profit
    results_df["cum_profit"] = results_df["profit"].cumsum()

    # Metrics
    total_profit = results_df["profit"].sum()
    num_trades = len(results_df)
    avg_profit = results_df["profit"].mean()
    max_profit = results_df['profit'].max()
    min_profit = results_df['profit'].min()
    win_rate = (results_df["profit"] > 0).mean()
    std_profit = results_df["profit"].std()
    sharpe_ratio = (avg_profit / std_profit) * np.sqrt(252 * 24 * 60) if std_profit > 0 else np.nan

    # Max drawdown
    cum_max = results_df["cum_profit"].cummax()
    drawdown = results_df["cum_profit"] - cum_max
    max_drawdown = drawdown.min()

    metrics = {
        "total_profit": total_profit,
        "num_trades": num_trades,
        "avg_profit": avg_profit,
        "win_rate": win_rate,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown
    }
    print("📊 Performance Summary:")
    print(f"Total Profit: ${total_profit:.2f}")
    print(f"Total Trades: {num_trades}")
    print(f"Average Profit per Trade: ${avg_profit:.4f}")
    print(f"Max Profit: ${max_profit:.4f}")
    print(f"Min Profit: ${min_profit:.4f}")
    print(f"Win Rate: {win_rate:.2%}")

    return results_df, metrics


def plot_trades_analysis(results_df, output_prefix='arbitrage_analysis'):
    """
    Plots performance analytics and saves figures to outputs/ folder.
    """
    if results_df.empty:
        print("No trades to plot.")
        return

    # Prepare directory
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)

    # Ensure timestamp is datetime and sorted
    results_df = results_df.copy()
    results_df['timestamp'] = pd.to_datetime(results_df['timestamp'])
    results_df = results_df.sort_values('timestamp')
    results_df['cumulative_profit'] = results_df['profit'].cumsum()

    # === Plot 1: Cumulative Profit ===
    plt.figure(figsize=(12, 6))
    plt.plot(results_df['timestamp'], results_df['cumulative_profit'], label='Cumulative Profit', color='blue')
    plt.axhline(0, color='gray', linestyle='--')
    plt.title('Cumulative Profit Over Time')
    plt.xlabel('Time')
    plt.ylabel('Profit')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{output_prefix}_cumulative_profit.png')
    plt.close()

    # === Plot 2: Drawdowns ===
    running_max = results_df['cumulative_profit'].cummax()
    drawdown = running_max - results_df['cumulative_profit']

    plt.figure(figsize=(12, 6))
    plt.plot(results_df['timestamp'], drawdown, label='Drawdown', color='red')
    plt.title('Drawdowns Over Time')
    plt.xlabel('Time')
    plt.ylabel('Drawdown')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{output_prefix}_drawdown.png')
    plt.close()

    # === Plot 3: Rolling Sharpe Ratio ===
    results_df.set_index('timestamp', inplace=True)
    returns = results_df['profit'].resample('1min').sum().fillna(0)
    rolling_mean = returns.rolling(window=60).mean()
    rolling_std = returns.rolling(window=60).std()
    rolling_sharpe = (rolling_mean / rolling_std).replace([np.inf, -np.inf], np.nan).fillna(0)

    plt.figure(figsize=(12, 6))
    plt.plot(rolling_sharpe.index, rolling_sharpe, label='Rolling Sharpe Ratio (1h window)', color='green')
    plt.axhline(0, color='gray', linestyle='--')
    plt.title('Rolling Sharpe Ratio')
    plt.xlabel('Time')
    plt.ylabel('Sharpe Ratio')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{output_prefix}_rolling_sharpe.png')
    plt.close()

    # === Plot 4: Trade Volume Over Time ===
    trade_volume = results_df['volume'].resample('5min').sum()

    plt.figure(figsize=(12, 6))
    plt.plot(trade_volume.index, trade_volume, label='Trade Volume (5-min)', color='purple')
    plt.title('Trade Volume Over Time')
    plt.xlabel('Time')
    plt.ylabel('Volume')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{output_prefix}_volume.png')
    plt.close()

    print(f"✅ All plots saved in: {os.path.abspath(output_dir)}")


def export_analysis_results(results_df, metrics: dict, output_prefix='arbitrage_analysis'):
    """
    Export trade-level results and performance metrics to CSV and Excel.
    """
    # Ensure directory exists
    os.makedirs('outputs', exist_ok=True)

    # Export trade results
    results_path_csv = f'outputs/{output_prefix}_results.csv'
    results_df.to_csv(results_path_csv, index=False)

    # Export performance metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_path_csv = f'outputs/{output_prefix}_metrics.csv'
    metrics_df.to_csv(metrics_path_csv, index=False)

    print(f"✅ Exported trade results and metrics to:\n- {results_path_csv}\n- {metrics_path_csv}")


def generate_trade_log(results_df, output_prefix='arbitrage_analysis'):
    """
    Generates and exports a clean trade log from results_df.
    """
    if results_df.empty:
        print("No trades to log.")
        return

    # Create outputs directory
    output_dir = 'outputs'
    os.makedirs(output_dir, exist_ok=True)

    # Select and rename columns for clarity
    trade_log_columns = [
        'timestamp', 'direction', 'symbol_bnc', 'symbol_hb',
        'executed_price_bnc', 'executed_price_hb',
        'volume', 'spread', 'profit'
    ]

    # Check availability
    available_cols = [col for col in trade_log_columns if col in results_df.columns]
    trade_log_df = results_df[available_cols].copy()

    # Clean column names
    trade_log_df.rename(columns={
        'timestamp': 'Timestamp',
        'direction': 'Side',
        'symbol_bnc': 'Symbol_BNC',
        'symbol_hb': 'Symbol_HB',
        'executed_price_bnc': 'Price_BNC',
        'executed_price_hb': 'Price_HB',
        'volume': 'Volume',
        'spread': 'Spread',
        'profit': 'PnL'
    }, inplace=True)

    # Reorder
    trade_log_df = trade_log_df[[
        'Timestamp', 'Side', 'Symbol_BNC', 'Symbol_HB',
        'Price_BNC', 'Price_HB', 'Volume', 'Spread', 'PnL'
    ]]

    # Save to CSV and Excel
    trade_log_df.to_csv(f'{output_dir}/{output_prefix}_trade_log.csv', index=False)

    print(f"✅ Trade log exported to:\n- {output_dir}/{output_prefix}_trade_log.csv")


if __name__ == "__main__":
    df_bnc = pd.read_csv("orderbook_bnc.csv")
    df_hb = pd.read_csv("orderbook_hb.csv")
    df_bnc_bar = pd.read_csv("bar_bnc.csv")
    df_hb_bar = pd.read_csv("bar_hb.csv")
    # Step 1: Backtest_wi
    results_df = backtest_arbitrage_with_bar_filter(df_bnc, df_hb, df_bnc_bar, df_hb_bar, fee_rate=0.0000001, min_spread=0.0, max_volatility=0.05)

    # Step 2: Analyze
    results_df, metrics = analyze_results(results_df)

    # Step 3: Export results and metrics
    plot_trades_analysis(results_df)
    export_analysis_results(results_df, metrics)
    generate_trade_log(results_df)




