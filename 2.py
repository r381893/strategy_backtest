import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title('📈 策略回測（現貨/期貨/加密貨幣 百分比版，爆倉即結束，最大回撤、最佳化、更多指標）')


def format_chinese_yaxis(val):
    """
    將數值格式化為中文風格的字符串，帶有億、千萬、百萬、萬等單位。
    """
    if val >= 1e8: return f"{val / 1e8:.1f}億"
    if val >= 1e7: return f"{val / 1e7:.1f}千萬"
    if val >= 1e6: return f"{val / 1e6:.1f}百萬"
    if val >= 1e4: return f"{val / 1e4:.1f}萬"
    return f"{int(val)}"


def profit_color(val):
    """
    應用 CSS 樣式，使正值利潤顯示為綠色，負值顯示為紅色。
    此函數不再直接返回 CSS 字符串，而是用於判斷顏色。
    實際應用樣式將在 Streamlit DataFrame.style.apply 中處理。
    """
    try:
        v = float(val)
    except ValueError:
        v = 0
    return 'green' if v >= 0 else 'red'


# 補回了 apply_profit_color_to_cell 函數，用於在 DataFrame 中應用顏色樣式
def apply_profit_color_to_cell(val):
    """
    為交易損益的單元格應用顏色樣式 (綠色為盈利，紅色為虧損)。
    此函數返回 CSS 樣式字符串，供 DataFrame.style.applymap 使用。
    """
    try:
        if isinstance(val, str) and '%' in val:
            v = float(val.strip('%')) / 100
        else:
            v = float(val)
    except (ValueError, TypeError):
        v = 0  # 處理非數值情況，例如NaN或空值
    color = '#4caf50' if v >= 0 else '#f44336'  # 綠色為正，紅色為負
    return f'color: {color}; font-weight: bold;'


# 新增：計算交易相關統計數據
def calculate_trade_statistics(trade_df):
    if trade_df.empty:
        return {
            "總交易筆數": 0, "獲利交易筆數": 0, "虧損交易筆數": 0,
            "勝率": 0.0, "平均獲利": 0.0, "平均虧損": 0.0, "盈虧比": 0.0
        }

    # Convert '損益' column to numeric for calculations
    # Assuming '損益' is already clean numeric or easily convertible
    trade_df['損益_numeric'] = trade_df['損益'].astype(float)

    winning_trades = trade_df[trade_df['損益_numeric'] > 0]
    losing_trades = trade_df[trade_df['損益_numeric'] < 0]

    total_trades = len(trade_df)
    win_trades_count = len(winning_trades)
    lose_trades_count = len(losing_trades)

    win_rate = (win_trades_count / total_trades) * 100 if total_trades > 0 else 0.0
    # 修正此處
    avg_win = winning_trades['損益_numeric'].mean() if win_trades_count > 0 else 0.0
    # avg_loss will be negative, take abs for ratio
    avg_loss = losing_trades['損益_numeric'].mean() if lose_trades_count > 0 else 0.0

    # Ensure avg_loss is not zero to avoid division by zero for profit_loss_ratio
    profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0.0

    return {
        "總交易筆數": total_trades,
        "獲利交易筆數": win_trades_count,
        "虧損交易筆數": lose_trades_count,
        "勝率": win_rate,
        "平均獲利": avg_win,
        "平均虧損": avg_loss,
        "盈虧比": profit_loss_ratio
    }


def annualized_return(equity_curve_df, initial_cash):
    """
    根據資產曲線計算每年的年化報酬率。
    參數:
        equity_curve_df (pd.DataFrame): 包含 'Date' 和 'Value' 列的 DataFrame。
        initial_cash (float): 回測的初始資金。
    回傳:
        pd.DataFrame: 包含 '年度' 和 '年化報酬率' 的 DataFrame。
    """
    df = equity_curve_df.copy()
    if 'Date' in df.columns and not df['Date'].empty:
        df['Year'] = pd.to_datetime(df['Date']).dt.year
    else:
        return pd.DataFrame({"年度": [], "年化報酬率": []})

    results = []
    for year, group in df.groupby('Year'):
        if group.empty:
            continue
        start_val = group['Value'].iloc[0]
        end_val = group['Value'].iloc[-1]

        rtn = (end_val / start_val) - 1 if start_val > 0 else 0
        results.append({"年度": year, "年化報酬率": rtn})
    return pd.DataFrame(results)


def calc_max_drawdown(equity_curve_df):
    """
    計算最大回撤及其開始和結束日期。
    處理資產曲線可能因爆倉而跌至零的情況。
    參數:
        equity_curve_df (pd.DataFrame): 包含 'Date' 和 'Value' 列的 DataFrame。
    回傳:
        tuple: (最大回撤百分比, 高點日期, 低點日期)
    """
    if equity_curve_df.empty:
        return 0, None, None

    values = equity_curve_df['Value'].values
    dates = pd.to_datetime(equity_curve_df['Date']).values

    if values[-1] == 0:
        non_zero_indices = np.where(values > 0)[0]
        if non_zero_indices.size > 0:
            last_non_zero_idx = non_zero_indices[-1]
            values_for_dd = values[:last_non_zero_idx + 1]
            dates_for_dd = dates[:last_non_zero_idx + 1]
        else:
            return 1.0, dates[0], dates[0]

        if values_for_dd.size == 0:
            return 0, None, None

        cummax = np.maximum.accumulate(values_for_dd)
        drawdowns = 1 - values_for_dd / cummax
        max_dd = np.max(drawdowns)  # Using np.max directly if values_for_dd is not empty
        max_dd_idx_relative = np.argmax(drawdowns)

        peak_idx = np.where(values_for_dd[:max_dd_idx_relative + 1] == cummax[max_dd_idx_relative])[0][-1]
        t1 = dates_for_dd[peak_idx]
        t2 = dates_for_dd[max_dd_idx_relative]

        if max_dd == 0 and values[-1] == 0:
            return 1.0, dates[0], dates[-1]
        return max_dd, t1, t2
    else:
        cummax = np.maximum.accumulate(values)
        drawdowns = 1 - values / cummax
        max_dd_idx = np.argmax(drawdowns)
        max_dd = drawdowns[max_dd_idx]
        if max_dd == 0:
            return 0, None, None
        peak_idx = np.where(values[:max_dd_idx + 1] == cummax[max_dd_idx])[0][-1]
        t1 = dates[peak_idx]
        t2 = dates[max_dd_idx]
        return max_dd, t1, t2


def yearly_max_drawdown(equity_curve_df):
    """
    計算每年最大回撤。
    參數:
        equity_curve_df (pd.DataFrame): 包含 'Date' 和 'Value' 列的 DataFrame。
    回傳:
        pd.DataFrame: 包含 '年度' 和 '最大回撤百分比' 的 DataFrame。
    """
    df = equity_curve_df.copy()
    if 'Date' in df.columns and not df['Date'].empty:
        df['Year'] = pd.to_datetime(df['Date']).dt.year
    else:
        return pd.DataFrame({"年度": [], "最大回撤百分比": []})

    results = []
    for year, group in df.groupby('Year'):
        if group.empty:
            results.append({"年度": year, "最大回撤百分比": 0.0})
            continue
        values = group['Value'].values
        if len(values) == 0:
            results.append({"年度": year, "最大回撤百分比": 0.0})
            continue

        if values[-1] == 0 and values[0] > 0:
            max_dd = 1.0
        else:
            cummax = np.maximum.accumulate(values)
            drawdowns = 1 - values / cummax
            max_dd = np.max(drawdowns)
        results.append({"年度": year, "最大回撤百分比": max_dd})
    return pd.DataFrame(results)


# 新增：計算夏普比率
def calculate_sharpe_ratio(equity_curve_df, risk_free_rate_annual=0.02):  # 假設年化無風險利率為 2%
    if equity_curve_df.empty or len(equity_curve_df) < 2:
        return 0.0

    # 計算每日報酬率
    returns = equity_curve_df['Value'].pct_change().dropna()
    if returns.empty:
        return 0.0

    # 將年化無風險利率轉換為每日無風險利率
    # 假設一年有 365 個日曆日
    daily_risk_free_rate = (1 + risk_free_rate_annual) ** (1 / 365) - 1

    excess_returns = returns - daily_risk_free_rate

    # 計算年化平均超額報酬和年化標準差
    annualization_factor = np.sqrt(365)

    avg_excess_return_annualized = excess_returns.mean() * 365
    std_dev_returns_annualized = excess_returns.std() * annualization_factor

    if std_dev_returns_annualized == 0:
        return 0.0  # 避免除以零

    sharpe_ratio = avg_excess_return_annualized / std_dev_returns_annualized
    return sharpe_ratio


# 新增：計算索提諾比率
def calculate_sortino_ratio(equity_curve_df, risk_free_rate_annual=0.02):
    if equity_curve_df.empty or len(equity_curve_df) < 2:
        return 0.0

    returns = equity_curve_df['Value'].pct_change().dropna()
    if returns.empty:
        return 0.0

    daily_risk_free_rate = (1 + risk_free_rate_annual) ** (1 / 365) - 1
    excess_returns = returns - daily_risk_free_rate

    # 只考慮負的超額報酬 (下行波動)
    downside_returns = excess_returns[excess_returns < 0]

    # 計算下行標準差
    downside_std_dev_annualized = downside_returns.std() * np.sqrt(365) if not downside_returns.empty else 0.0

    avg_excess_return_annualized = excess_returns.mean() * 365

    if downside_std_dev_annualized == 0:
        return 0.0

    sortino_ratio = avg_excess_return_annualized / downside_std_dev_annualized
    return sortino_ratio


# 新增：計算卡爾瑪比率
def calculate_calmar_ratio(equity_curve_df, initial_cash):
    if equity_curve_df.empty:
        return 0.0

    total_return = (equity_curve_df['Value'].iloc[-1] / initial_cash) - 1 if initial_cash > 0 else 0.0

    # 計算年化報酬率
    num_days = (equity_curve_df['Date'].iloc[-1] - equity_curve_df['Date'].iloc[0]).days
    if num_days <= 0:
        return 0.0

    annualized_total_return = (1 + total_return) ** (365.0 / num_days) - 1

    max_dd_pct, _, _ = calc_max_drawdown(equity_curve_df)

    if max_dd_pct == 0:  # 避免除以零
        return 0.0

    calmar_ratio = annualized_total_return / max_dd_pct
    return calmar_ratio


# 初始化會話狀態變數（如果不存在）
if 'do_optimize' not in st.session_state:
    st.session_state['do_optimize'] = False
if 'page' not in st.session_state:
    st.session_state['page'] = 1
if 'trade_df' not in st.session_state:
    st.session_state['trade_df'] = pd.DataFrame()
if 'equity_curve_df' not in st.session_state:
    st.session_state['equity_curve_df'] = pd.DataFrame()
if 'liquidation_info' not in st.session_state:
    st.session_state['liquidation_info'] = {'liquidated': False, 'date': None}
if 'start_date_display' not in st.session_state:
    st.session_state['start_date_display'] = None
if 'end_date_display' not in st.session_state:
    st.session_state['end_date_display'] = None


def trigger_optimize():
    """設定會話狀態中的最佳化標誌。"""
    st.session_state['do_optimize'] = True


def set_page(page_num):
    """更新交易明細的當前頁碼。"""
    st.session_state['page'] = page_num


uploaded_file = st.file_uploader("📂 請上傳每日成交價 Excel 檔案 (.xlsx)", type=["xlsx"])
if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    lower_cols = [c.lower() for c in df.columns]
    date_candidates = ["date", "data", "日期"]
    close_candidates = ["close", "收盤價"]

    date_col_name = next((df.columns[lower_cols.index(cand)] for cand in date_candidates if cand in lower_cols), None)
    close_col_name = next((df.columns[lower_cols.index(cand)] for cand in close_candidates if cand in lower_cols), None)

    if not date_col_name or not close_col_name:
        st.error("❌ 無法找到日期或收盤價欄位。請檢查Excel。")
        st.stop()

    df[date_col_name] = pd.to_datetime(df[date_col_name], errors='coerce')
    df = df.dropna(subset=[date_col_name, close_col_name])
    df = df.sort_values(date_col_name).reset_index(drop=True)

    # 確保 session state 中的日期在每次上傳新檔案時，都設定為當前檔案的有效範圍
    min_date_current_df = df[date_col_name].min().date()
    max_date_current_df = df[date_col_name].max().date()

    st.session_state['start_date_display'] = min_date_current_df
    st.session_state['end_date_display'] = max_date_current_df

    st.write(
        f"資料筆數：{len(df)}，日期範圍：{df[date_col_name].min().strftime('%Y/%m/%d')} ~ {df[date_col_name].max().strftime('%Y/%m/%d')}")
    st.dataframe(df.head())

    st.write("### 原始收盤價曲線（標記最大回撤區間）")
    price_curve_df = df[[date_col_name, close_col_name]].copy()
    price_curve_df = price_curve_df.dropna().reset_index(drop=True)
    price_curve_df.rename(columns={date_col_name: 'Date', close_col_name: 'Close'}, inplace=True)

    # 計算原始價格的總期間最大回撤
    dd_price, dd_price_t1, dd_price_t2 = calc_max_drawdown(
        price_curve_df.rename(columns={'Close': 'Value'})
    )
    fig_price = go.Figure()
    fig_price.add_trace(
        go.Scatter(
            x=price_curve_df["Date"],
            y=price_curve_df["Close"],
            mode="lines",
            name="收盤價",
            line=dict(color="#1976d2"),
        )
    )
    if dd_price_t1 is not None and dd_price_t2 is not None:
        peak_idx = price_curve_df.index[price_curve_df["Date"] == pd.to_datetime(dd_price_t1)]
        trough_idx = price_curve_df.index[price_curve_df["Date"] == pd.to_datetime(dd_price_t2)]
        if not peak_idx.empty and not trough_idx.empty:
            peak_idx = peak_idx[0]
            trough_idx = trough_idx[0]
            fig_price.add_trace(
                go.Scatter(
                    x=[price_curve_df["Date"].iloc[peak_idx]],
                    y=[price_curve_df["Close"].iloc[peak_idx]],
                    mode="markers+text",
                    marker=dict(color="orange", size=12, symbol="star"),
                    text=["高點"],
                    textposition="top right",
                    name="最大回撤高點"
                )
            )
            fig_price.add_trace(
                go.Scatter(
                    x=[price_curve_df["Date"].iloc[trough_idx]],
                    y=[price_curve_df["Close"].iloc[trough_idx]],
                    mode="markers+text",
                    marker=dict(color="red", size=12, symbol="star"),
                    text=["低點"],
                    textposition="bottom left",
                    name="最大回撤低點"
                )
            )
            fig_price.add_trace(
                go.Scatter(
                    x=[price_curve_df["Date"].iloc[peak_idx], price_curve_df["Date"].iloc[trough_idx]],
                    y=[price_curve_df["Close"].iloc[peak_idx], price_curve_df["Close"].iloc[trough_idx]],
                    mode="lines",
                    line=dict(color="red", width=2, dash="dash"),
                    name="最大回撤區間"
                )
            )
        st.info(
            f"原始價格最大回撤：{dd_price * 100:.2f}%  發生區間：{pd.to_datetime(dd_price_t1).strftime('%Y-%m-%d')} ~ {pd.to_datetime(dd_price_t2).strftime('%Y-%m-%d')}")
    else:
        st.info("原始價格沒有明顯回撤。")

    # 標記每年最大回撤區間 (針對原始收盤價)
    if not price_curve_df.empty:
        price_curve_df['Year'] = pd.to_datetime(price_curve_df['Date']).dt.year
        unique_years_price = price_curve_df['Year'].unique()

        year_price_dd_colors = [
            '#004d00',  # Darker Green
            '#5a2000',  # Darker Brown
            '#003366',  # Darker Blue
            '#4a004a',  # Darker Purple
            '#36454F',  # Charcoal
            '#556B2F',  # DarkOliveGreen
            '#8B4513',  # SaddleBrown
            '#2F4F4F',  # DarkSlateGray
            '#483D8B',  # DarkSlateBlue
            '#800000'  # Maroon
        ]
        color_idx_price = 0

        for year_p in unique_years_price:
            yearly_price_df = price_curve_df[price_curve_df['Year'] == year_p].copy()
            if not yearly_price_df.empty:
                temp_yearly_price_df_for_dd = yearly_price_df[['Date', 'Close']].rename(columns={'Close': 'Value'})

                dd_yearly_pct_price, dd_yearly_t1_price, dd_yearly_t2_price = calc_max_drawdown(
                    temp_yearly_price_df_for_dd)

                if dd_yearly_t1_price is not None and dd_yearly_t2_price is not None and dd_yearly_pct_price > 0:
                    peak_val_yearly_price = \
                        price_curve_df[price_curve_df['Date'] == pd.to_datetime(dd_yearly_t1_price)]['Close'].iloc[0]
                    trough_val_yearly_price = \
                        price_curve_df[price_curve_df['Date'] == pd.to_datetime(dd_yearly_t2_price)]['Close'].iloc[0]

                    fig_price.add_trace(go.Scatter(
                        x=[pd.to_datetime(dd_yearly_t1_price), pd.to_datetime(dd_yearly_t2_price)],
                        y=[peak_val_yearly_price, trough_val_yearly_price],
                        mode='lines',
                        line=dict(color=year_price_dd_colors[color_idx_price % len(year_price_dd_colors)], width=1.5,
                                  dash='dot'),
                        name=f'原始價格 {year_p} 年回撤 ({dd_yearly_pct_price * 100:.2f}%)',
                        showlegend=True
                    ))
                    color_idx_price += 1

    fig_price.update_layout(
        xaxis_title="日期",
        yaxis_title="收盤價",
        height=400,
        legend=dict(orientation='h')
    )
    st.plotly_chart(fig_price, use_container_width=True, key="raw_price_chart")

    # 移除選擇回測標的
    # target_type = st.selectbox("請選擇回測標的", ["台股期貨", "加密貨幣", "ETF/股票"])
    st.write("---")
    st.subheader("⚡ 回測模式選擇")
    mode = st.selectbox(
        "請選擇回測策略模式",
        [
            "買進抱到底（只做多不賣出）",
            "均線上做多，下空手（多次進出）",
            "均線上做多，均線下做空"
        ]
    )

    st.write("---")
    st.subheader("📅 回測日期範圍設定")
    col_date1, col_date2 = st.columns(2)
    with col_date1:
        # 使用已經設定好的 min_date_current_df 和 max_date_current_df 作為 min_value 和 max_value
        st.session_state['start_date_display'] = st.date_input("回測起始日",
                                                               value=st.session_state['start_date_display'],
                                                               min_value=min_date_current_df,
                                                               max_value=max_date_current_df,
                                                               key="start_date_picker")
    with col_date2:
        # 使用已經設定好的 min_date_current_df 和 max_date_current_df 作為 min_value 和 max_value
        st.session_state['end_date_display'] = st.date_input("回測結束日",
                                                             value=st.session_state['end_date_display'],
                                                             min_value=min_date_current_df,
                                                             max_value=max_date_current_df,
                                                             key="end_date_picker")

    st.write("---")
    st.subheader('⚙️ 自動最佳化參數設定')
    col1, col2 = st.columns(2)
    with col1:
        ma_min = st.number_input('均線最小天數', min_value=2, max_value=200, value=5)
    with col2:
        ma_max = st.number_input('均線最大天數', min_value=int(ma_min), max_value=200, value=60)
    leverage_list = st.multiselect(
        '槓桿倍數',
        [1, 2, 3, 5, 10, 20],
        default=[1, 2, 3, 5],
    )

    st.button('開始自動搜尋最佳組合', on_click=trigger_optimize, key="btn_optimize")


    def calc_profit(entry, exit, position, leverage, margin):
        """
        計算交易的利潤或損失。
        參數:
            entry (float): 進場價格。
            exit (float): 出場價格。
            position (int): 1 表示做多，-1 表示做空。
            leverage (float): 使用的槓桿倍數。
            margin (float): 交易投入的初始保證金。
        回傳:
            float: 計算出的利潤/損失。
        """
        if entry == 0:
            return 0
        if position == 1:  # 做多頭寸的利潤計算
            return ((exit - entry) / entry) * leverage * margin
        else:  # 做空頭寸的利潤計算
            return ((entry - exit) / entry) * leverage * margin


    # ======== 自動最佳化區塊 ========
    if st.session_state['do_optimize']:
        start_date_opt = st.session_state['start_date_display']
        end_date_opt = st.session_state['end_date_display']

        if start_date_opt is None or end_date_opt is None:
            st.warning("⚠️ 請先上傳 Excel 檔案並讓日期範圍初始化，再執行自動最佳化。")
            st.session_state['do_optimize'] = False
        else:
            opt_mask = (df[date_col_name] >= pd.to_datetime(start_date_opt)) & \
                       (df[date_col_name] <= pd.to_datetime(end_date_opt))
            df_for_optimization = df.loc[opt_mask].copy().reset_index(drop=True)

            if df_for_optimization.empty:
                st.warning("⚠️ 在選擇的日期範圍內沒有足夠的資料來執行最佳化，請調整日期範圍。")
                st.session_state['do_optimize'] = False
            else:
                result = []
                best_equity = -float('inf')
                best_ma, best_lev = None, None
                progress_text = "回測進度："
                my_bar = st.progress(0, text=progress_text)
                total = (int(ma_max) - int(ma_min) + 1) * len(leverage_list)
                now = 0

                for ma in range(int(ma_min), int(ma_max) + 1):
                    for lev in leverage_list:
                        now += 1
                        my_bar.progress(now / total, text=f"{progress_text}{now}/{total}")

                        df_opt = df_for_optimization.copy()

                        if len(df_opt) < ma:
                            equity = 1.0
                            is_liquidated_opt = False
                        else:
                            df_opt['MA'] = df_opt[close_col_name].rolling(window=ma).mean()
                            df_opt['Prev_Close'] = df_opt[close_col_name].shift(1)
                            df_opt['Prev_MA'] = df_opt['MA'].shift(1)

                            df_opt['Signal'] = 0
                            df_opt_slice = df_opt.loc[ma:].copy()

                            cond_long_opt = (df_opt_slice[close_col_name] > df_opt_slice['MA']) & \
                                            (df_opt_slice['Prev_Close'] <= df_opt_slice['Prev_MA'])
                            cond_short_opt = (df_opt_slice[close_col_name] < df_opt_slice['MA']) & \
                                             (df_opt_slice['Prev_Close'] >= df_opt_slice['Prev_MA'])

                            df_opt_slice['Signal'] = np.select(
                                [cond_long_opt, cond_short_opt],
                                [1, -1],
                                default=0
                            )
                            df_opt.loc[ma:, 'Signal'] = df_opt_slice['Signal']

                            cash = 100000.0
                            pos = 0
                            entry = 0.0
                            margin = 0.0
                            is_liquidated_opt = False

                            for i in range(ma, len(df_opt)):
                                sig = df_opt['Signal'].iloc[i]
                                price = df_opt[close_col_name].iloc[i]

                                if pos != 0:
                                    current_value_of_position = margin + calc_profit(entry, price, pos, lev, margin)
                                    if lev > 1 and current_value_of_position < margin * 0.20:
                                        cash = 0.0
                                        is_liquidated_opt = True
                                        break

                                if mode == "買進抱到底（只做多不賣出）":
                                    if pos == 0 and cash > 0 and sig == 1:
                                        pos, entry, margin, cash = 1, price, cash, 0.0
                                elif mode == "均線上做多，下空手（多次進出）":
                                    if pos == 0 and cash > 0 and sig == 1:
                                        pos, entry, margin, cash = 1, price, cash, 0.0
                                    elif pos == 1 and sig == -1:
                                        profit = calc_profit(entry, price, pos, lev, margin)
                                        if profit < -margin: profit = -margin
                                        cash = margin + profit
                                        pos, entry, margin = 0, 0.0, 0.0
                                elif mode == "均線上做多，均線下做空":
                                    if pos == 0 and cash > 0 and sig != 0:
                                        pos, entry, margin, cash = sig, price, cash, 0.0
                                    elif pos != 0 and sig != 0 and sig != pos:
                                        profit = calc_profit(entry, price, pos, lev, margin)
                                        if profit < -margin: profit = -margin
                                        cash = margin + profit
                                        if cash > 0:
                                            pos, entry, margin, cash = sig, price, cash, 0.0
                                        else:
                                            pos, entry, margin = 0, 0.0, 0.0

                            if pos != 0 and not is_liquidated_opt:
                                last_price = df_opt[close_col_name].iloc[-1]
                                profit = calc_profit(entry, last_price, pos, lev, margin)
                                if profit < -margin: profit = -margin
                                cash = margin + profit

                            equity = cash / 100000.0

                        result.append((ma, lev, equity, is_liquidated_opt))
                        if equity > best_equity:
                            best_equity = equity
                            best_ma = ma
                            best_lev = lev
                my_bar.empty()

                result_df = pd.DataFrame(result, columns=['MA天數', '槓桿倍數', '最終資產倍數', '是否爆倉'])


                def format_equity_display(row):
                    if row['是否爆倉']:
                        return f"{row['最終資產倍數']:.2f} (已爆倉)"
                    return f"{row['最終資產倍數']:.2f}"


                result_df['最終資產倍數_顯示'] = result_df.apply(format_equity_display, axis=1)

                st.write("🏆 回測結果")
                best_strategy_info = result_df[(result_df['MA天數'] == best_ma) & (result_df['槓桿倍數'] == best_lev)]
                if not best_strategy_info.empty and best_strategy_info['是否爆倉'].iloc[0]:
                    st.error(
                        f"⚠️ 最佳均線天數: {best_ma} 天, 最佳槓桿倍數: {best_lev} 倍, 最終資產倍數: {best_equity:.2f} (此最佳策略已爆倉！)")
                else:
                    st.success(
                        f"最佳均線天數: {best_ma} 天, 最佳槓桿倍數: {best_lev} 倍, 最終資產倍數: {best_equity:.2f}")

                st.dataframe(result_df[['MA天數', '槓桿倍數', '最終資產倍數_顯示']], use_container_width=True)

                st.write("### 槓桿/均線最佳化圖表")
                fig_opt = go.Figure()
                for lev_val in sorted(result_df['槓桿倍數'].unique()):
                    sub = result_df[result_df['槓桿倍數'] == lev_val]
                    is_all_liquidated_for_lev = sub['是否爆倉'].all()

                    line_name = f"{lev_val}倍"
                    if is_all_liquidated_for_lev:
                        line_name += " (已爆倉)"

                    fig_opt.add_trace(go.Scatter(x=sub['MA天數'], y=sub['最終資產倍數'],
                                                 mode='lines+markers', name=line_name))
                fig_opt.update_layout(
                    xaxis_title="均線天數",
                    yaxis_title="最終資產倍數",
                    height=400
                )
                st.plotly_chart(fig_opt, use_container_width=True, key="opt_chart")
                st.session_state['do_optimize'] = False

    st.write("---")
    st.subheader(f"🔍 {mode}")
    initial_cash = st.number_input("初始資金（TWD）", min_value=10000, max_value=10000000, value=100000, step=10000,
                                   key="initial_cash")
    colc1, colc2 = st.columns(2)
    with colc1:
        custom_ma = st.number_input("均線天數", min_value=2, max_value=200, value=20, key="custom_ma")
    with colc2:
        custom_leverage = st.selectbox("槓桿倍數", options=[1, 2, 3, 5, 10, 20], index=1, key="custom_leverage")

    custom_btn = st.button(f"執行{mode}回測", key="btn_run")

    if custom_btn or (st.session_state.get('trade_df').empty and not st.session_state['do_optimize']):
        st.session_state['liquidation_info'] = {'liquidated': False, 'date': None}

        mask = (df[date_col_name] >= pd.to_datetime(st.session_state['start_date_display'])) & \
               (df[date_col_name] <= pd.to_datetime(st.session_state['end_date_display']))
        df_detail = df.loc[mask].copy().reset_index(drop=True)

        if len(df_detail) < custom_ma:
            st.warning("⚠️ 選擇的回測日期範圍資料不足以計算均線，請調整日期或均線天數。")
            st.session_state['trade_df'] = pd.DataFrame()
            st.session_state['equity_curve_df'] = pd.DataFrame()
        else:
            ma_period = custom_ma
            leverage = custom_leverage

            df_detail['MA'] = df_detail[close_col_name].rolling(window=ma_period).mean()
            df_detail['Prev_Close'] = df_detail[close_col_name].shift(1)
            df_detail['Prev_MA'] = df_detail['MA'].shift(1)

            df_detail['Signal'] = 0
            df_detail_slice = df_detail.loc[ma_period:].copy()

            cond_long_detail = (df_detail_slice[close_col_name] > df_detail_slice['MA']) & \
                               (df_detail_slice['Prev_Close'] <= df_detail_slice['Prev_MA'])
            cond_short_detail = (df_detail_slice[close_col_name] < df_detail_slice['MA']) & \
                                (df_detail_slice['Prev_Close'] >= df_detail_slice['Prev_MA'])

            df_detail_slice['Signal'] = np.select(
                [cond_long_detail, cond_short_detail],
                [1, -1],
                default=0
            )
            df_detail.loc[ma_period:, 'Signal'] = df_detail_slice['Signal']

            trade_list = []
            equity_curve_data = []
            cash_balance = float(initial_cash)
            position = 0
            entry_price = 0.0
            entry_date_of_trade = None
            margin_in_trade = 0.0
            units = 0.0

            liquidated = False

            for i in range(len(df_detail)):
                signal = df_detail['Signal'].iloc[i]
                price = df_detail[close_col_name].iloc[i]
                cur_date = df_detail[date_col_name].iloc[i]

                if liquidated:
                    equity_curve_data.append({"Date": cur_date, "Value": 0.0})
                    continue

                current_value = cash_balance
                if position != 0:
                    unrealized_pnl = calc_profit(entry_price, price, position, leverage, margin_in_trade)
                    current_value_of_position = margin_in_trade + unrealized_pnl

                    if leverage > 1 and current_value_of_position < margin_in_trade * 0.20:
                        cash_balance = 0.0
                        liquidated = True
                        st.session_state['liquidation_info']['liquidated'] = True
                        st.session_state['liquidation_info']['date'] = cur_date.strftime("%Y-%m-%d")
                        equity_curve_data.append({"Date": cur_date, "Value": 0.0})
                        break
                    else:
                        current_value = current_value_of_position

                equity_curve_data.append({"Date": cur_date, "Value": current_value})

                if mode == "買進抱到底（只做多不賣出）":
                    if position == 0 and cash_balance > 0 and signal == 1:
                        position = 1
                        entry_price = price
                        entry_date_of_trade = cur_date
                        margin_in_trade = cash_balance
                        cash_balance = 0.0
                        units = margin_in_trade * leverage / entry_price if entry_price > 0 else 0
                elif mode == "均線上做多，下空手（多次進出）":
                    if position == 0 and cash_balance > 0 and signal == 1:
                        position = 1
                        entry_price = price
                        entry_date_of_trade = cur_date
                        margin_in_trade = cash_balance
                        cash_balance = 0.0
                        units = margin_in_trade * leverage / entry_price if entry_price > 0 else 0
                    elif position == 1 and signal == -1:
                        realized_pnl = calc_profit(entry_price, price, position, leverage, margin_in_trade)
                        if realized_pnl < -margin_in_trade:
                            realized_pnl = -margin_in_trade

                        pnl_pct = (realized_pnl / margin_in_trade) if margin_in_trade > 0 else 0
                        cash_balance = margin_in_trade + realized_pnl

                        trade_list.append({
                            "進場日": entry_date_of_trade.strftime("%Y-%m-%d"),
                            "出場日": cur_date.strftime("%Y-%m-%d"),
                            "方向": "做多",
                            "進場價": entry_price,
                            "進場資金": margin_in_trade,
                            "出場價": price,
                            "持有天數": (cur_date - entry_date_of_trade).days,
                            "進場單位": units,
                            "損益": realized_pnl,
                            "損益百分比": pnl_pct,
                            "出場總資產": cash_balance
                        })
                        position = 0
                        entry_price = 0.0
                        entry_date_of_trade = None
                        margin_in_trade = 0.0
                        units = 0.0
                elif mode == "均線上做多，均線下做空":
                    if position == 0 and cash_balance > 0 and signal != 0:
                        position = signal
                        entry_price = price
                        entry_date_of_trade = cur_date
                        margin_in_trade = cash_balance
                        cash_balance = 0.0
                        units = margin_in_trade * leverage / entry_price if entry_price > 0 else 0
                    elif position != 0 and signal != 0 and signal != position:
                        # 先平倉
                        realized_pnl = calc_profit(entry_price, price, position, leverage, margin_in_trade)
                        if realized_pnl < -margin_in_trade:
                            realized_pnl = -margin_in_trade

                        pnl_pct = (realized_pnl / margin_in_trade) if margin_in_trade > 0 else 0
                        cash_balance = margin_in_trade + realized_pnl

                        trade_list.append({
                            "進場日": entry_date_of_trade.strftime("%Y-%m-%d"),
                            "出場日": cur_date.strftime("%Y-%m-%d"),
                            "方向": "做多" if position == 1 else "做空",
                            "進場價": entry_price,
                            "進場資金": margin_in_trade,
                            "出場價": price,
                            "持有天數": (cur_date - entry_date_of_trade).days,
                            "進場單位": units,
                            "損益": realized_pnl,
                            "損益百分比": pnl_pct,
                            "出場總資產": cash_balance
                        })

                        # 再開新倉
                        if cash_balance > 0:
                            position = signal
                            entry_price = price
                            entry_date_of_trade = cur_date
                            margin_in_trade = cash_balance
                            cash_balance = 0.0
                            units = margin_in_trade * leverage / entry_price if entry_price > 0 else 0
                        else:
                            liquidated = True
                            st.session_state['liquidation_info']['liquidated'] = True
                            st.session_state['liquidation_info']['date'] = cur_date.strftime("%Y-%m-%d")
                            position = 0
                            entry_price = 0.0
                            entry_date_of_trade = None
                            margin_in_trade = 0.0
                            units = 0.0

            # 迴圈結束後，如果還有持倉，進行最後平倉
            if position != 0 and not liquidated:
                last_price = df_detail[close_col_name].iloc[-1]
                last_date = df_detail[date_col_name].iloc[-1]
                realized_pnl = calc_profit(entry_price, last_price, position, leverage, margin_in_trade)
                if realized_pnl < -margin_in_trade:
                    realized_pnl = -margin_in_trade

                pnl_pct = (realized_pnl / margin_in_trade) if margin_in_trade > 0 else 0
                cash_balance = margin_in_trade + realized_pnl

                trade_list.append({
                    "進場日": entry_date_of_trade.strftime("%Y-%m-%d"),
                    "出場日": last_date.strftime("%Y-%m-%d"),
                    "方向": "做多" if position == 1 else "做空",
                    "進場價": entry_price,
                    "進場資金": margin_in_trade,
                    "出場價": last_price,
                    "持有天數": (last_date - entry_date_of_trade).days,
                    "進場單位": units,
                    "損益": realized_pnl,
                    "損益百分比": pnl_pct,
                    "出場總資產": cash_balance
                })

            st.session_state['trade_df'] = pd.DataFrame(trade_list)
            st.session_state['equity_curve_df'] = pd.DataFrame(equity_curve_data)

    # ===== 績效總覽 =====
    if not st.session_state.get('equity_curve_df').empty:
        st.write("---")
        st.subheader("📊 績效總覽")
        end_value = st.session_state['equity_curve_df']['Value'].iloc[-1]
        total_return = (end_value / initial_cash - 1) if initial_cash > 0 else 0
        total_days = (st.session_state['equity_curve_df']['Date'].iloc[-1] -
                      st.session_state['equity_curve_df']['Date'].iloc[0]).days

        # 處理爆倉情況下的最大回撤
        max_dd_pct, dd_t1, dd_t2 = calc_max_drawdown(st.session_state['equity_curve_df'])

        # 計算夏普、索提諾、卡爾瑪
        sharpe = calculate_sharpe_ratio(st.session_state['equity_curve_df'])
        sortino = calculate_sortino_ratio(st.session_state['equity_curve_df'])
        calmar = calculate_calmar_ratio(st.session_state['equity_curve_df'], initial_cash)

        col_metrics1, col_metrics2, col_metrics3, col_metrics4 = st.columns(4)
        with col_metrics1:
            st.metric("最終資產", f"{end_value:,.2f} TWD")
        with col_metrics2:
            st.metric("總報酬率", f"{total_return:.2%}")
        with col_metrics3:
            st.metric("總回測天數", f"{total_days} 天")
        with col_metrics4:
            st.metric("最大回撤", f"{max_dd_pct:.2%}")

        col_ratios1, col_ratios2, col_ratios3 = st.columns(3)
        with col_ratios1:
            st.metric("夏普比率", f"{sharpe:.2f}")
        with col_ratios2:
            st.metric("索提諾比率", f"{sortino:.2f}")
        with col_ratios3:
            st.metric("卡爾瑪比率", f"{calmar:.2f}")

        if st.session_state['liquidation_info']['liquidated']:
            st.error(f"⚠️ **注意：** 策略已於 **{st.session_state['liquidation_info']['date']}** 爆倉，資產歸零。")

        # 繪製資產曲線
        st.write("### 資產曲線")
        fig_equity = go.Figure()
        fig_equity.add_trace(
            go.Scatter(x=st.session_state['equity_curve_df']['Date'], y=st.session_state['equity_curve_df']['Value'],
                       mode='lines', name='總資產', line=dict(color='blue')))

        # 標記最大回撤
        if dd_t1 is not None and dd_t2 is not None:
            dd_t1 = pd.to_datetime(dd_t1)
            dd_t2 = pd.to_datetime(dd_t2)
            peak_value = \
            st.session_state['equity_curve_df'][st.session_state['equity_curve_df']['Date'] == dd_t1]['Value'].iloc[0]
            trough_value = \
            st.session_state['equity_curve_df'][st.session_state['equity_curve_df']['Date'] == dd_t2]['Value'].iloc[0]

            fig_equity.add_trace(go.Scatter(
                x=[dd_t1, dd_t2],
                y=[peak_value, trough_value],
                mode='lines',
                line=dict(color='red', width=2, dash='dash'),
                name='最大回撤區間'
            ))
            fig_equity.add_trace(go.Scatter(
                x=[dd_t1],
                y=[peak_value],
                mode='markers',
                marker=dict(color='red', size=10, symbol='star'),
                name='最大回撤高點'
            ))
            fig_equity.add_trace(go.Scatter(
                x=[dd_t2],
                y=[trough_value],
                mode='markers',
                marker=dict(color='red', size=10, symbol='star'),
                name='最大回撤低點'
            ))

        fig_equity.update_layout(
            xaxis_title="日期",
            yaxis_title="總資產",
            yaxis=dict(tickformat=',.0f'),  # 格式化為千分位
            legend=dict(orientation='h')
        )
        st.plotly_chart(fig_equity, use_container_width=True, key="equity_chart")

    # ===== 交易統計與明細 =====
    if not st.session_state.get('trade_df').empty:
        st.write("---")
        st.subheader("📋 交易明細與統計")

        trade_stats = calculate_trade_statistics(st.session_state['trade_df'])

        col_stats1, col_stats2, col_stats3 = st.columns(3)
        with col_stats1:
            st.metric("總交易筆數", trade_stats["總交易筆數"])
            st.metric("獲利交易筆數", trade_stats["獲利交易筆數"])
            st.metric("虧損交易筆數", trade_stats["虧損交易筆數"])
        with col_stats2:
            st.metric("勝率", f'{trade_stats["勝率"]:.2f}%')
            st.metric("平均獲利", f'{trade_stats["平均獲利"]:.2f}')
            st.metric("平均虧損", f'{trade_stats["平均虧損"]:.2f}')
        with col_stats3:
            st.metric("盈虧比", f'{trade_stats["盈虧比"]:.2f}')

        # 分頁顯示交易明細
        page_size = 10
        total_trades = len(st.session_state['trade_df'])
        total_pages = (total_trades + page_size - 1) // page_size

        # 如果是剛回測完，預設到最後一頁
        if custom_btn and total_pages > 0:
            st.session_state['page'] = total_pages
        # 處理沒有交易的情況
        if total_trades == 0:
            st.info("⚠️ 沒有交易發生。")
        else:
            st.write("---")
            st.write("#### 交易紀錄")

            # 分頁導航
            col_page1, col_page2, col_page3 = st.columns([1, 1, 8])
            with col_page1:
                if st.button("上一頁", disabled=st.session_state['page'] <= 1):
                    st.session_state['page'] -= 1
            with col_page2:
                if st.button("下一頁", disabled=st.session_state['page'] >= total_pages):
                    st.session_state['page'] += 1
            with col_page3:
                st.write(f"第 {st.session_state['page']}/{total_pages} 頁，共 {total_trades} 筆交易")

            start_idx = (st.session_state['page'] - 1) * page_size
            end_idx = start_idx + page_size

            trade_df_paged = st.session_state['trade_df'].iloc[start_idx:end_idx].reset_index(drop=True)

            # 應用顏色樣式
            st.dataframe(
                trade_df_paged.style.applymap(
                    apply_profit_color_to_cell, subset=pd.IndexSlice[:, ['損益', '損益百分比']]
                ).format({
                    '進場價': '{:.2f}',
                    '出場價': '{:.2f}',
                    '進場資金': '{:,.2f}',
                    '進場單位': '{:,.2f}',
                    '損益': '{:,.2f}',
                    '損益百分比': '{:.2%}',
                    '出場總資產': '{:,.2f}'
                }),
                use_container_width=True
            )

        # 每年回撤與年化報酬率
        st.write("---")
        st.write("### 每年績效")

        yearly_returns_df = annualized_return(st.session_state['equity_curve_df'], initial_cash)
        yearly_drawdown_df = yearly_max_drawdown(st.session_state['equity_curve_df'])

        yearly_results = pd.merge(
            yearly_returns_df, yearly_drawdown_df, on='年度', how='outer'
        ).fillna(0)

        yearly_results['年化報酬率'] = yearly_results['年化報酬率'].apply(lambda x: f'{x:.2%}')
        yearly_results['最大回撤百分比'] = yearly_results['最大回撤百分比'].apply(lambda x: f'{x:.2%}')

        st.dataframe(yearly_results)

    elif st.session_state.get('equity_curve_df').empty and not st.session_state['do_optimize'] and custom_btn:
        st.warning("⚠️ 沒有交易發生，請檢查您的策略或回測參數。")

else:
    st.info("📤 請上傳含有每日成交價的 Excel 檔案（欄位需為「日期」、「收盤價」）。")