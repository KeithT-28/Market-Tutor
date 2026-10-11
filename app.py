import random
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(
    page_title="Market Tutor | Technical Analysis Academy",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Visual system ----------
st.markdown("""
<style>
:root { --navy:#101827; --panel:#182436; --gold:#d7b56d; --muted:#98a6b8; }
.stApp { background: #0b1220; color: #edf2f7; }
[data-testid="stSidebar"] { background: #101827; border-right: 1px solid #263449; }
[data-testid="stSidebar"] * { color: #e7edf5; }
h1,h2,h3 { letter-spacing: -0.025em; }
p, li { color: #c7d0dc; }
div[data-testid="stMetric"] { background:#152033; border:1px solid #263449; padding:16px; border-radius:14px; }
div[data-testid="stMetricLabel"] { color:#98a6b8; }
div[data-testid="stMetricValue"] { color:#f5f7fa; }
div.stButton > button { border-radius:10px; border:1px solid #d7b56d; background:#d7b56d; color:#101827; font-weight:700; }
div.stButton > button:hover { border-color:#f0d89d; background:#f0d89d; color:#101827; }
div[data-testid="stAlert"] { border-radius:12px; }
.mt-eyebrow { color:#d7b56d; font-size:0.78rem; letter-spacing:.12em; font-weight:700; text-transform:uppercase; }
.mt-card { background:#121d2d; border:1px solid #263449; border-radius:16px; padding:18px 20px; }
.mt-muted { color:#98a6b8; font-size:.92rem; }
</style>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
<div style="padding:8px 0 18px 0">
  <div style="font-size:12px;letter-spacing:.16em;color:#d7b56d;font-weight:800">MARKET TUTOR</div>
  <div style="font-size:12px;color:#98a6b8;margin-top:4px">TECHNICAL ANALYSIS ACADEMY</div>
</div>
""", unsafe_allow_html=True)

page = st.sidebar.radio(
    "WORKSPACE",
    ["Overview", "Learn", "Chart Quiz", "Progress"],
    label_visibility="visible",
)
st.sidebar.divider()
st.sidebar.caption("TRAINING MODE · NO LIVE ORDERS")
st.sidebar.caption("Market data is provided by Yahoo Finance and may be delayed or incomplete.")

# ---------- Data and indicators ----------
@st.cache_data(ttl=900, show_spinner=False)
def load_data(symbol: str, period: str, interval: str):
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=True, progress=False)
    if df is None or df.empty:
        return pd.DataFrame()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.title)
    required = ["Open", "High", "Low", "Close", "Volume"]
    if not all(c in df.columns for c in required):
        return pd.DataFrame()
    df = df[required].dropna().copy()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    return df.dropna().copy()

def price_chart(df, show_indicators=True, title="Price action"):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Candles", increasing_line_color="#22c55e", decreasing_line_color="#ef6464",
        increasing_fillcolor="#22c55e", decreasing_fillcolor="#ef6464"
    ))
    if show_indicators:
        fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], name="MA 20", line=dict(color="#d7b56d", width=1.7)))
        fig.add_trace(go.Scatter(x=df.index, y=df["MA50"], name="MA 50", line=dict(color="#60a5fa", width=1.7)))
    fig.update_layout(
        title=title, template="plotly_dark", height=470, margin=dict(l=8, r=8, t=48, b=8),
        paper_bgcolor="#121d2d", plot_bgcolor="#121d2d",
        xaxis_rangeslider_visible=False, legend=dict(orientation="h", y=1.02, x=0),
        font=dict(color="#dce4ee"), hovermode="x unified"
    )
    fig.update_xaxes(showgrid=True, gridcolor="#263449")
    fig.update_yaxes(showgrid=True, gridcolor="#263449", side="right")
    return fig

def indicator_chart(df, indicator):
    fig = go.Figure()
    if indicator == "RSI":
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI (14)", line=dict(color="#c084fc", width=2)))
        fig.add_hline(y=70, line_dash="dash", line_color="#ef6464")
        fig.add_hline(y=30, line_dash="dash", line_color="#22c55e")
        fig.update_yaxes(range=[0, 100], title="RSI")
    elif indicator == "MACD":
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD", line=dict(color="#60a5fa", width=2)))
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"], name="Signal", line=dict(color="#d7b56d", width=2)))
        fig.add_hline(y=0, line_dash="dash", line_color="#98a6b8")
    elif indicator == "Volume":
        fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume", marker_color="#60a5fa"))
    fig.update_layout(
        template="plotly_dark", height=220, margin=dict(l=8, r=8, t=28, b=8),
        paper_bgcolor="#121d2d", plot_bgcolor="#121d2d",
        font=dict(color="#dce4ee"), legend=dict(orientation="h", y=1.02, x=0)
    )
    fig.update_xaxes(showgrid=True, gridcolor="#263449")
    fig.update_yaxes(showgrid=True, gridcolor="#263449", side="right")
    return fig

def get_market_data(symbol, interval):
    # Yahoo Finance generally limits intraday history to a relatively short window.
    period = "60d" if interval in ("15m", "30m", "60m") else "2y"
    return load_data(symbol, period, interval)

def reset_challenge(df, symbol, interval, indicator):
    # Reserve enough future candles for a defined, observable outcome.
    if len(df) < 100:
        return False
    min_cut = max(65, int(len(df) * 0.45))
    max_cut = len(df) - 13
    if min_cut >= max_cut:
        min_cut = max(50, max_cut - 50)
    if min_cut >= max_cut:
        return False
    cut = random.randint(min_cut, max_cut)
    st.session_state.challenge = {
        "symbol": symbol, "interval": interval, "indicator": indicator,
        "cut": cut, "answer": None, "confidence": 50, "revealed": False,
        "data_signature": f"{symbol}|{interval}|{len(df)}"
    }
    return True

# ---------- Page: Overview ----------
if page == "Overview":
    st.markdown('<div class="mt-eyebrow">Your trading education workspace</div>', unsafe_allow_html=True)
    st.title("Learn to read the market.")
    st.markdown("Build chart-reading intuition through short lessons, historical chart challenges and structured feedback.")
    st.write("")
    c1, c2, c3 = st.columns(3)
    c1.metric("Learning modules", "6", "Start with the foundations")
    c2.metric("Quiz attempts this session", str(len(st.session_state.get("attempts", []))))
    c3.metric("Trading mode", "Paper only", "No live orders")
    st.write("")
    left, right = st.columns([1.25, 1])
    with left:
        st.markdown('<div class="mt-card"><div class="mt-eyebrow">Recommended starting point</div><h3>01 · Market structure</h3><p>Learn to identify trends, swing highs and swing lows before adding indicators.</p></div>', unsafe_allow_html=True)
        if st.button("Start learning", use_container_width=True):
            st.session_state["go_to_learn"] = True
            st.rerun()
        if st.session_state.pop("go_to_learn", False):
            st.info("Choose **Learn** from the sidebar to open the lessons.")
    with right:
        st.markdown('<div class="mt-card"><div class="mt-eyebrow">Daily practice</div><h3>Chart Quiz</h3><p>Read a chart, form a view, then reveal what happened in the following candles.</p></div>', unsafe_allow_html=True)
        if st.button("Open Chart Quiz", use_container_width=True):
            st.session_state["quiz_nudge"] = True
            st.rerun()
        if st.session_state.pop("quiz_nudge", False):
            st.info("Choose **Chart Quiz** from the sidebar to start a challenge.")
    st.subheader("Your learning path")
    modules = [
        ("01", "Market structure", "Trends, swing highs/lows and price context"),
        ("02", "Support & resistance", "Zones, reactions and level quality"),
        ("03", "Moving averages", "Trend filters and dynamic reference levels"),
        ("04", "RSI", "Momentum, ranges and divergence caveats"),
        ("05", "MACD & volume", "Momentum shifts and participation"),
        ("06", "Trade planning", "Invalidation, targets and risk/reward"),
    ]
    for n, name, desc in modules:
        st.markdown(f'<div class="mt-card" style="margin:8px 0;padding:14px 18px"><span style="color:#d7b56d;font-weight:800">{n}</span> &nbsp; <b>{name}</b><br><span class="mt-muted">{desc}</span></div>', unsafe_allow_html=True)

# ---------- Page: Learn ----------
elif page == "Learn":
    st.markdown('<div class="mt-eyebrow">Learn · Foundations to application</div>', unsafe_allow_html=True)
    st.title("Technical Analysis Library")
    lesson = st.selectbox("Choose a lesson", [
        "01 · Market structure", "02 · Support & resistance", "03 · Moving averages",
        "04 · RSI", "05 · MACD & volume", "06 · Trade planning"
    ])
    lessons = {
        "01 · Market structure": (
            "Read the sequence before the indicator",
            "A trend is a sequence, not a single candle.",
            [
                "**Uptrend:** price tends to form higher swing highs and higher swing lows.",
                "**Downtrend:** price tends to form lower swing highs and lower swing lows.",
                "**Range:** price repeatedly reacts within an area without sustained directional progress.",
                "**Context matters:** a pattern on one timeframe can sit inside a larger trend in the opposite direction."
            ],
            "Practice: mark two recent swing highs and two swing lows. Ask whether the sequence is rising, falling or overlapping."
        ),
        "02 · Support & resistance": (
            "Think in zones, not magic lines",
            "Support and resistance are areas where price has previously reacted; they are not guaranteed barriers.",
            [
                "Look for repeated reactions, clear swing points and areas where price moved away decisively.",
                "A level with several reactions may be more noticeable, but repeated tests can also weaken it.",
                "A breakout is more convincing when price closes beyond a zone and follow-through appears.",
                "A retest can hold, fail or never occur. Avoid assuming one outcome is certain."
            ],
            "Practice: identify a zone before looking at the next candles. Define what would invalidate your idea."
        ),
        "03 · Moving averages": (
            "Smooth price; do not forecast it",
            "A moving average summarises past prices and therefore lags the market.",
            [
                "A rising average can help describe positive recent momentum; a falling average describes negative momentum.",
                "A short moving average reacting faster than a long one can help frame trend context.",
                "Crossovers can whipsaw in sideways markets.",
                "Moving averages are reference points, not automatic buy or sell signals."
            ],
            "Practice: compare price location, average slope and the most recent swing structure together."
        ),
        "04 · RSI": (
            "Momentum needs context",
            "RSI compares recent average gains and losses on a bounded 0–100 scale.",
            [
                "The traditional 70/30 levels are reference zones, not reliable reversal commands.",
                "RSI can remain elevated in a strong uptrend or depressed in a strong downtrend.",
                "Divergence may signal momentum changing, but it can persist without an immediate reversal.",
                "Interpret RSI alongside price structure and the market regime."
            ],
            "Practice: ask whether RSI agrees with the price trend, and what price would need to do to confirm a change."
        ),
        "05 · MACD & volume": (
            "Look for momentum shifts and participation",
            "MACD compares two exponential moving averages; volume shows how much trading occurred in a period.",
            [
                "The MACD line, signal line and zero line describe different aspects of momentum.",
                "Crossovers are not guaranteed entries and can be late or noisy.",
                "Volume can help assess participation around a breakout, but volume alone does not establish direction.",
                "Indicators derived from price often tell overlapping stories; more indicators do not automatically mean more evidence."
            ],
            "Practice: describe what price is doing first, then note whether MACD and volume support or contradict that view."
        ),
        "06 · Trade planning": (
            "Define risk before a hypothetical entry",
            "A trade idea should include a thesis, an invalidation point and a plan for managing risk.",
            [
                "Entry: the condition that would make you consider a setup.",
                "Invalidation: the price behaviour that would show the idea is no longer valid.",
                "Target: a planned area to assess potential reward, not a promise that price will reach it.",
                "Position size should reflect the amount you are willing to lose if the invalidation level is reached."
            ],
            "Practice: write down the setup, invalidation and target before revealing subsequent candles."
        ),
    }
    heading, intro, bullets, exercise = lessons[lesson]
    st.markdown(f'<div class="mt-card"><div class="mt-eyebrow">{lesson.split(" · ")[0]} · Core concept</div><h2>{heading}</h2><p>{intro}</p></div>', unsafe_allow_html=True)
    st.subheader("What to remember")
    for bullet in bullets:
        st.markdown(f"- {bullet}")
    st.info(exercise)
    with st.expander("Quick knowledge check"):
        if lesson.startswith("01"):
            q = st.radio("Which pattern most directly describes an uptrend?", [
                "Lower swing highs and lower swing lows",
                "Higher swing highs and higher swing lows",
                "Repeatedly overlapping highs and lows"
            ])
            if st.button("Check answer", key="lesson_check"):
                st.success("Correct: higher swing highs and higher swing lows describe the classic uptrend structure.") if q.startswith("Higher") else st.warning("Not quite. Focus on the direction of successive swing highs and lows.")
        elif lesson.startswith("04"):
            q = st.radio("RSI above 70 means price must reverse immediately.", ["True", "False"])
            if st.button("Check answer", key="lesson_check"):
                st.success("Correct: False. RSI can remain elevated during a strong trend.") if q == "False" else st.warning("RSI can stay elevated in a strong uptrend; it is not an automatic reversal signal.")
        else:
            st.write("In your own words, describe one piece of evidence and one condition that would invalidate your interpretation.")
            st.text_area("Your notes", key=f"notes_{lesson}")

# ---------- Page: Chart Quiz ----------
elif page == "Chart Quiz":
    st.markdown('<div class="mt-eyebrow">Practice · Read, decide, review</div>', unsafe_allow_html=True)
    st.title("Chart Quiz")
    st.markdown("Historical practice only. The next candles stay hidden until you submit your interpretation.")
    controls1, controls2, controls3 = st.columns([1, 1, 1])
    with controls1:
        symbol = st.selectbox("Instrument", ["SPY", "QQQ", "AAPL", "MSFT"], index=0)
    with controls2:
        interval = st.selectbox("Timeframe", ["15m", "30m", "60m", "1d"], index=0)
    with controls3:
        indicator = st.selectbox("Focus indicator", ["Moving averages", "RSI", "MACD", "Volume"], index=0)

    with st.spinner("Loading historical market data..."):
        df = get_market_data(symbol, interval)
    if df.empty or len(df) < 100:
        st.error("There isn't enough data available for this instrument/timeframe right now. Try SPY on 15m or 1d.")
    else:
        challenge = st.session_state.get("challenge")
        signature = f"{symbol}|{interval}|{len(df)}"
        if not challenge or challenge.get("data_signature") != signature or challenge.get("indicator") != indicator:
            reset_challenge(df, symbol, interval, indicator)
            challenge = st.session_state.get("challenge")

        if challenge:
            cut = challenge["cut"]
            visible = df.iloc[max(0, cut - 80):cut]
            future = df.iloc[cut:cut + 12]
            st.caption(f"{symbol} · {interval} · {visible.index[0]} to {visible.index[-1]} · Future candles hidden")
            show_ma = indicator == "Moving averages"
            st.plotly_chart(price_chart(visible, show_indicators=show_ma, title="What do you see?"), use_container_width=True, config={"displaylogo": False})
            if indicator in ("RSI", "MACD", "Volume"):
                st.plotly_chart(indicator_chart(visible, indicator), use_container_width=True, config={"displaylogo": False})
            if indicator == "RSI":
                last_rsi = float(visible["RSI"].iloc[-1])
                st.caption(f"Latest RSI (14): {last_rsi:.1f}")
            elif indicator == "MACD":
                st.caption(f"Latest MACD: {visible['MACD'].iloc[-1]:.3f} · Signal: {visible['MACD_Signal'].iloc[-1]:.3f}")
            elif indicator == "Volume":
                st.caption("Compare activity on recent candles with earlier bars; volume does not indicate direction by itself.")

            st.subheader("Your interpretation")
            answer = st.radio(
                "Over the next 12 candles, what is your directional bias?",
                ["Bullish", "Bearish", "Sideways / unclear"],
                horizontal=True,
                key=f"answer_{challenge['cut']}_{symbol}_{interval}_{indicator}",
                disabled=challenge.get("revealed", False),
            )
            confidence = st.slider(
                "How confident are you?", 0, 100, 50, step=10,
                key=f"confidence_{challenge['cut']}_{symbol}_{interval}_{indicator}",
                disabled=challenge.get("revealed", False),
            )
            if not challenge.get("revealed", False):
                if st.button("Submit interpretation", type="primary", use_container_width=True):
                    start_close = float(visible["Close"].iloc[-1])
                    end_close = float(future["Close"].iloc[-1])
                    move_pct = (end_close / start_close - 1) * 100
                    # A 0.35% dead zone reduces false precision for very small moves.
                    outcome = "Bullish" if move_pct > 0.35 else ("Bearish" if move_pct < -0.35 else "Sideways / unclear")
                    challenge.update({
                        "answer": answer, "confidence": confidence, "revealed": True,
                        "outcome": outcome, "move_pct": move_pct,
                        "start_close": start_close, "end_close": end_close,
                    })
                    st.session_state.challenge = challenge
                    st.session_state.setdefault("attempts", []).append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "symbol": symbol, "interval": interval, "indicator": indicator,
                        "answer": answer, "outcome": outcome, "confidence": confidence,
                        "move_pct": move_pct, "correct": answer == outcome,
                    })
                    st.rerun()
            else:
                st.success(f"Historical outcome: **{challenge['outcome']}** · Price moved {challenge['move_pct']:+.2f}% over the next 12 candles.")
                if challenge["answer"] == challenge["outcome"]:
                    st.markdown("### Your directional read matched the defined outcome.")
                else:
                    st.markdown("### Review the evidence; one result is only one sample.")
                st.caption(f"Your answer: {challenge['answer']} · Confidence: {challenge['confidence']}%")
                st.plotly_chart(price_chart(df.iloc[max(0, cut - 80):cut + 12], show_indicators=show_ma, title="Revealed: subsequent candles"), use_container_width=True, config={"displaylogo": False})
                st.subheader("Review checklist")
                if indicator == "Moving averages":
                    st.markdown("- Did price structure agree with the moving-average slope?\n- Was price extended from the averages?\n- Where would the trend idea have been invalidated?")
                elif indicator == "RSI":
                    st.markdown("- Was RSI elevated, depressed or near the middle of its range?\n- Did price confirm the momentum reading?\n- Could the market have stayed overbought or oversold?")
                elif indicator == "MACD":
                    st.markdown("- Were MACD and signal line rising or falling?\n- Was the reading above or below zero?\n- Did price action confirm the momentum signal?")
                else:
                    st.markdown("- Did volume expand near a key move?\n- Was the candle direction clear?\n- What other price evidence was needed?")
                st.warning("This is a hindsight-learning exercise, not a trading signal. A directional match does not establish a profitable strategy.")
                if st.button("Next challenge", use_container_width=True):
                    reset_challenge(df, symbol, interval, indicator)
                    st.rerun()
            st.caption("Method: the outcome is the close-to-close move over the next 12 bars, with a ±0.35% neutral band. This simple rule is for learning only and ignores costs, slippage and path risk.")

# ---------- Page: Progress ----------
elif page == "Progress":
    st.markdown('<div class="mt-eyebrow">Review · Build consistency</div>', unsafe_allow_html=True)
    st.title("Your Progress")
    attempts = st.session_state.get("attempts", [])
    if not attempts:
        st.markdown('<div class="mt-card"><h3>Your journal starts with your first challenge.</h3><p>Complete a chart quiz to begin tracking directional reads, confidence and indicator-specific performance.</p></div>', unsafe_allow_html=True)
    else:
        hist = pd.DataFrame(attempts)
        c1, c2, c3 = st.columns(3)
        c1.metric("Attempts", len(hist))
        c2.metric("Directional accuracy", f"{hist['correct'].mean() * 100:.0f}%")
        c3.metric("Average confidence", f"{hist['confidence'].mean():.0f}%")
        st.subheader("Attempts")
        st.dataframe(hist[["timestamp", "symbol", "interval", "indicator", "answer", "outcome", "confidence", "move_pct", "correct"]], use_container_width=True, hide_index=True)
        st.caption("Progress currently lasts for this browser session. Persistent cross-device saving will be added in a later version.")
        csv = hist.to_csv(index=False).encode("utf-8")
        st.download_button("Download practice journal (CSV)", csv, file_name="market_tutor_practice.csv", mime="text/csv")
    st.subheader("How to interpret your score")
    st.markdown("- Use accuracy as feedback, not as proof that a strategy works.\n- Review low-confidence correct calls and high-confidence incorrect calls.\n- Aim to explain the evidence and invalidation point, not just guess the direction.")
