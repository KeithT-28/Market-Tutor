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

def lesson_illustration(lesson):
    """Build a clearly labelled hypothetical chart for the selected lesson."""
    n = 60
    x = np.arange(n)
    # Deterministic examples keep the teaching illustration stable between reruns.
    if lesson.startswith("01"):
        close = 100 + 0.22*x + 2.2*np.sin(x/3.8)
    elif lesson.startswith("02"):
        close = 100 + 2.0*np.sin(x/4.1) + 0.35*np.sin(x/1.7)
    elif lesson.startswith("03"):
        close = 100 + 0.18*x + 2.0*np.sin(x/5.0)
    elif lesson.startswith("04"):
        close = 100 + 0.16*x + 3.1*np.sin(x/4.2)
    elif lesson.startswith("05"):
        close = 100 + 0.12*x + 2.4*np.sin(x/4.6)
    else:
        close = 100 + 0.16*x + 1.8*np.sin(x/4.8)
    open_ = np.r_[close[0], close[:-1] + 0.25*np.sin(x[1:])]
    high = np.maximum(open_, close) + 0.35 + 0.18*(1 + np.sin(x/2.7))
    low = np.minimum(open_, close) - 0.35 - 0.18*(1 + np.cos(x/3.1))
    volume = 800 + 180*np.abs(np.diff(np.r_[close[0], close]))*100 + 120*(1 + np.sin(x/2.4))
    idx = pd.RangeIndex(1, n + 1, name="Illustrative bar")
    df = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=idx)
    df["MA20"] = df["Close"].rolling(20, min_periods=1).mean()
    df["MA50"] = df["Close"].rolling(50, min_periods=1).mean()
    delta = df["Close"].diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=1).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=1).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = (100 - 100 / (1 + rs)).fillna(50)
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    fig = go.Figure()
    if lesson.startswith("04"):
        fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI (illustrative)", line=dict(color="#c084fc", width=2)))
        fig.add_hline(y=70, line_dash="dash", line_color="#ef6464", annotation_text="70 reference")
        fig.add_hline(y=30, line_dash="dash", line_color="#22c55e", annotation_text="30 reference")
        fig.update_yaxes(range=[0, 100], title="RSI (0–100)")
        title = "Illustration: RSI is context, not an automatic reversal signal"
    elif lesson.startswith("05"):
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD line", line=dict(color="#60a5fa", width=2)))
        fig.add_trace(go.Scatter(x=df.index, y=df["MACD_Signal"], name="Signal line", line=dict(color="#d7b56d", width=2)))
        fig.add_hline(y=0, line_dash="dash", line_color="#98a6b8")
        title = "Illustration: MACD momentum lines (hypothetical data)"
    else:
        fig.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Hypothetical price", increasing_line_color="#22c55e", decreasing_line_color="#ef6464"))
        if lesson.startswith("03"):
            fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], name="20-period average", line=dict(color="#d7b56d", width=2)))
            fig.add_trace(go.Scatter(x=df.index, y=df["MA50"], name="50-period average", line=dict(color="#60a5fa", width=2)))
            title = "Illustration: moving averages lag price"
        elif lesson.startswith("02"):
            fig.add_hrect(y0=99, y1=101, fillcolor="#d7b56d", opacity=0.18, line_width=0, annotation_text="Example support zone")
            fig.add_hrect(y0=102.5, y1=104.5, fillcolor="#ef6464", opacity=0.15, line_width=0, annotation_text="Example resistance zone")
            title = "Illustration: support and resistance are zones"
        elif lesson.startswith("06"):
            fig.add_hline(y=102, line_dash="dash", line_color="#22c55e", annotation_text="Hypothetical target")
            fig.add_hline(y=100, line_dash="dash", line_color="#d7b56d", annotation_text="Hypothetical entry")
            fig.add_hline(y=98.5, line_dash="dash", line_color="#ef6464", annotation_text="Hypothetical invalidation")
            title = "Illustration: define entry, target and invalidation"
        else:
            title = "Illustration: higher swing highs and higher swing lows"
            fig.add_annotation(x=15, y=float(df["High"].iloc[15]), text="Swing high", showarrow=True, arrowhead=2)
            fig.add_annotation(x=24, y=float(df["Low"].iloc[24]), text="Higher low", showarrow=True, arrowhead=2)
    fig.update_layout(template="plotly_dark", title=title, height=390, margin=dict(l=8, r=8, t=55, b=8), paper_bgcolor="#121d2d", plot_bgcolor="#121d2d", font=dict(color="#dce4ee"), legend=dict(orientation="h", y=1.02, x=0), xaxis_rangeslider_visible=False, hovermode="x unified")
    fig.update_xaxes(title="Illustrative bar", showgrid=True, gridcolor="#263449")
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

    # Expanded lesson notes: practical definitions, examples, common traps and a repeatable workflow.
    lesson_details = {
        "01 · Market structure": {
            "Why it matters": "Market structure gives you the map before you choose a tool. Indicators can disagree or lag; swing points help describe what price has actually done.",
            "Key ideas": [
                ("Swing high / swing low", "A swing high is a local peak with lower highs around it; a swing low is a local trough with higher lows around it. There is no single universal candle-count rule, so use a consistent method when marking them."),
                ("Trend versus range", "An uptrend usually advances through higher highs and higher lows. A downtrend usually makes lower highs and lower lows. In a range, price repeatedly rotates between areas and directional follow-through is limited."),
                ("Impulse and pullback", "An impulse is a directional move; a pullback is a counter-move within the broader move. A pullback does not automatically mean the trend has reversed."),
                ("Break of structure", "A meaningful move beyond a prior swing can suggest a change in control, but a brief wick beyond a level may be a false break. Look at the close, follow-through and wider context."),
            ],
            "Worked example": "SPY makes a swing low near $500, rallies to $510, pulls back to $505, then rises through $510. The higher low at $505 and a new high support an uptrend interpretation. If price instead falls decisively below $505, that bullish structure is weakened; it does not guarantee a downtrend.",
            "Common mistakes": ["Calling a trend from one large candle.", "Changing the definition of a swing after seeing what happened next.", "Ignoring the higher timeframe or nearby resistance.", "Treating a break of one swing as certainty rather than evidence."],
            "Practice steps": ["Mark the last two clear swing highs and lows.", "Label the environment: uptrend, downtrend, range or unclear.", "Write down the price behaviour that would weaken your read.", "Only then check indicators for supporting or conflicting evidence."],
        },
        "02 · Support & resistance": {
            "Why it matters": "Support and resistance are decision areas, not invisible walls. Traders may watch prior turning points, consolidation edges and breakout areas, so reactions can cluster there.",
            "Key ideas": [
                ("Zones, not exact lines", "Price rarely respects a level to the exact cent. Mark a reasonable area around a cluster of reactions rather than drawing a line that implies false precision."),
                ("Evidence quality", "A clear move away, multiple visible reactions and agreement with the broader structure can make a zone worth watching. Too many touches do not automatically make it stronger; repeated tests can consume resting orders."),
                ("Breakout and retest", "A breakout is a move beyond a zone. A close beyond the zone and follow-through can add evidence. A retest is a return toward the broken area; it may hold, fail or never happen."),
                ("Role reversal", "A former resistance area may act as support after a successful break, and vice versa. This is a possibility to test against price behaviour, not a rule."),
            ],
            "Worked example": "Suppose SPY has rejected the $520–$522 area several times. A candle wicking above $522 but closing back inside the zone is weaker breakout evidence than a decisive close above it followed by continued buying. A later pullback that holds the zone may strengthen the bullish case, while a close back below it weakens it.",
            "Common mistakes": ["Drawing so many zones that every price looks important.", "Assuming the next touch must bounce.", "Entering only because price reached a line, without a trigger or invalidation.", "Ignoring scheduled news or unusually volatile conditions."],
            "Practice steps": ["Mark only the two or three clearest nearby zones.", "Record why each zone matters (swing, repeated reaction, consolidation edge).", "Define what a convincing break would look like before it occurs.", "Identify the invalidation condition and the next obstacle to price."],
        },
        "03 · Moving averages": {
            "Why it matters": "A moving average smooths past prices to help you see direction and context. Because it is calculated from historical data, it responds after price has moved rather than predicting the future.",
            "Key ideas": [
                ("SMA and EMA", "A simple moving average gives equal weight to observations in its window. An exponential moving average weights recent observations more heavily and generally reacts faster."),
                ("Slope and location", "A rising average with price above it can support a bullish trend description; a falling average with price below it can support a bearish one. Neither is sufficient by itself."),
                ("Fast and slow averages", "A shorter average reacts faster; a longer one changes more slowly. Crossovers summarise past price behaviour and can arrive late or whipsaw when price is ranging."),
                ("Dynamic reference", "A moving average can act as a reference area during a trend, but price can slice through it repeatedly. Treat it as context, not guaranteed support or resistance."),
            ],
            "Worked example": "If SPY is above a rising 20-period average and the 20-period average is above a rising 50-period average, the recent trend context is positive. If price is far above both averages after a sharp rally, chasing may mean accepting poor risk/reward. Wait for a defined setup rather than buying just because the averages look bullish.",
            "Common mistakes": ["Buying every moving-average touch.", "Treating a crossover as a complete strategy.", "Using an indicator setting without considering the chart timeframe.", "Forgetting averages are calculated from past prices."],
            "Practice steps": ["Read price structure first.", "Describe the slope and ordering of the 20- and 50-period averages.", "Assess whether price is near the averages or unusually extended.", "Write down what price would invalidate the trend idea."],
        },
        "04 · RSI": {
            "Why it matters": "The Relative Strength Index (RSI) is a momentum oscillator scaled from 0 to 100. It summarises the balance of recent gains and losses; it does not measure value or predict an exact turning point.",
            "Key ideas": [
                ("Traditional levels", "RSI above 70 is often called overbought and below 30 oversold. These are warning/context zones, not instructions to sell or buy."),
                ("Trend regimes", "During strong uptrends, RSI can remain high; during strong downtrends, it can remain low. Some traders watch whether RSI holds higher ranges in an uptrend or lower ranges in a downtrend."),
                ("Divergence", "Bullish divergence describes price making a lower low while RSI makes a higher low; bearish divergence is the opposite. Divergence can warn that momentum is changing, but it can persist and is not a timing signal on its own."),
                ("Confirmation", "Look for price evidence such as a swing break or reclaim of a zone before treating an RSI observation as a trade idea."),
            ],
            "Worked example": "SPY rises while RSI reaches 74. This tells you momentum has been strong; it does not mean price must reverse. If price continues making higher highs and higher lows, fading the trend solely because RSI is above 70 can be costly. Instead, monitor structure and define what would show that momentum has actually weakened.",
            "Common mistakes": ["Automatically shorting above 70 or buying below 30.", "Treating divergence as proof of an imminent reversal.", "Ignoring the price chart because RSI looks dramatic.", "Comparing RSI values without noting timeframe and market regime."],
            "Practice steps": ["State the price trend before reading RSI.", "Record the RSI value and whether it is rising or falling.", "Ask whether the reading confirms or conflicts with price structure.", "Specify the price evidence you would need before changing your view."],
        },
        "05 · MACD & volume": {
            "Why it matters": "MACD helps describe momentum using moving averages; volume describes trading activity. They answer different questions and should be interpreted alongside price rather than used as standalone signals.",
            "Key ideas": [
                ("MACD line", "The standard MACD line is the 12-period EMA minus the 26-period EMA. The signal line is commonly a 9-period EMA of MACD. Settings can vary."),
                ("Signal and zero-line crosses", "A MACD cross above its signal line can indicate improving momentum; a move above zero means the shorter EMA is above the longer EMA. Both can lag and generate false signals in ranges."),
                ("Histogram", "The histogram is the difference between MACD and its signal line. It can show whether the gap between them is widening or narrowing, but it is not a direct measure of future price movement."),
                ("Volume context", "A breakout with above-average volume may indicate greater participation. But volume does not reveal whether all participants are buying or selling, and high volume can appear at both continuation and exhaustion points."),
            ],
            "Worked example": "Price breaks above a range and MACD rises above its signal line while volume expands. Those observations may support the breakout thesis, but you still need to assess whether price closes above the zone, whether the next candles follow through, where the setup fails and whether the potential reward justifies the risk.",
            "Common mistakes": ["Counting MACD, RSI and moving averages as three independent confirmations when all are derived from price.", "Assuming high volume is automatically bullish.", "Ignoring the zero line or broader trend when reading MACD crosses.", "Using an indicator cross without a price-based invalidation."],
            "Practice steps": ["Describe price action without indicators.", "Describe MACD direction, signal relationship and zero-line location.", "Compare recent volume with its own recent baseline.", "List both confirming and conflicting evidence."],
        },
        "06 · Trade planning": {
            "Why it matters": "A chart idea is not a complete plan until you know what would prove it wrong and how much you could lose. Risk management cannot make a weak setup profitable, but it helps prevent one decision from causing disproportionate damage.",
            "Key ideas": [
                ("Setup and trigger", "A setup is the context you are watching; a trigger is the observable event that would prompt consideration. Example: an uptrend pulls back to a prior breakout zone, then forms a bullish close."),
                ("Invalidation", "Choose the price behaviour that contradicts the thesis. It should be tied to market structure or the setup logic, not placed at an arbitrary distance just to allow a larger position."),
                ("Risk and position size", "For a simplified example, if a paper account is $10,000 and a hypothetical risk limit is 0.5%, planned loss is $50. With a $2 entry-to-stop distance, a simplified share quantity is 25 shares before fees, slippage and gap risk. This is an illustration, not a recommendation."),
                ("Reward-to-risk", "If the planned reward is $3 per share and planned risk is $1 per share, the ratio is 3:1. The ratio alone does not make a trade attractive; probability of success, execution costs and actual outcomes matter."),
                ("Journal and review", "Record the thesis, trigger, invalidation, target, market conditions and outcome. Review a meaningful sample rather than drawing conclusions from one winning or losing trade."),
            ],
            "Worked example": "Hypothetical paper setup: entry at $520, invalidation at $518, and target area at $526. Planned risk is $2 per share and potential reward is $6 per share, or 3:1 before costs. Before taking it, ask whether the target is realistic given nearby resistance and whether the invalidation level genuinely contradicts the thesis.",
            "Common mistakes": ["Moving a stop farther away after the trade idea fails.", "Choosing a target only to create an attractive reward-to-risk ratio.", "Ignoring gaps, slippage, fees and liquidity.", "Confusing one successful outcome with proof of an edge."],
            "Practice steps": ["Write the market context and setup.", "Define the trigger and invalidation before the hypothetical entry.", "Estimate risk per share and position size from a chosen risk budget.", "Record the result and whether you followed your plan, separately from whether the trade won."],
        },
    }
    detail = lesson_details[lesson]
    st.subheader("See the concept on a chart")
    st.caption("Teaching illustration only: these are synthetic, hypothetical values, not live or historical SPY prices and not trading signals.")
    st.plotly_chart(lesson_illustration(lesson), use_container_width=True, config={"displaylogo": False})
    if lesson.startswith("01"):
        st.markdown("**How to read it:** compare successive peaks and troughs. A rising sequence of highs and lows supports an uptrend description; one candle alone does not establish a trend.")
    elif lesson.startswith("02"):
        st.markdown("**How to read it:** shaded bands represent areas where price might react, not exact prices that must hold. Watch closes and follow-through around the zones.")
    elif lesson.startswith("03"):
        st.markdown("**How to read it:** the faster average follows price more closely, while the slower average reacts later. Crossovers describe past movement and can whipsaw in ranges.")
    elif lesson.startswith("04"):
        st.markdown("**How to read it:** 70 and 30 are reference levels. RSI above 70 does not force an immediate decline, and RSI below 30 does not force a rebound.")
    elif lesson.startswith("05"):
        st.markdown("**How to read it:** compare the MACD and signal lines with the zero line. Crossovers can indicate momentum changes, but they are lagging and should be checked against price action.")
    else:
        st.markdown("**How to read it:** the example marks a hypothetical entry, target and invalidation. The target is not guaranteed; planned risk and reward should be assessed before a trade idea.")
    st.subheader("Understand it in depth")
    st.markdown(detail["Why it matters"])
    for concept_title, concept_text in detail["Key ideas"]:
        with st.expander(concept_title):
            st.write(concept_text)
    st.markdown("#### Worked example")
    st.markdown(f'<div class="mt-card">{detail["Worked example"]}</div>', unsafe_allow_html=True)
    st.markdown("#### Common mistakes")
    for mistake in detail["Common mistakes"]:
        st.markdown(f"- {mistake}")
    st.markdown("#### A repeatable practice routine")
    for step_no, step in enumerate(detail["Practice steps"], start=1):
        st.markdown(f"**{step_no}.** {step}")
    with st.expander("Quick knowledge check"):
        if lesson.startswith("01"):
            q = st.radio("Which pattern most directly describes an uptrend?", [
                "Lower swing highs and lower swing lows",
                "Higher swing highs and higher swing lows",
                "Repeatedly overlapping highs and lows"
            ])
            if st.button("Check answer", key="lesson_check"):
                if q.startswith("Higher"):
                    st.success("Correct: higher swing highs and higher swing lows describe the classic uptrend structure.")
                else:
                    st.warning("Not quite. Focus on the direction of successive swing highs and lows.")
        elif lesson.startswith("04"):
            q = st.radio("RSI above 70 means price must reverse immediately.", ["True", "False"])
            if st.button("Check answer", key="lesson_check"):
                if q == "False":
                    st.success("Correct: False. RSI can remain elevated during a strong trend.")
                else:
                    st.warning("RSI can stay elevated in a strong uptrend; it is not an automatic reversal signal.")
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
