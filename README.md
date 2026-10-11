# Market Tutor

A mobile-friendly technical analysis learning app built with Streamlit. It includes lessons, historical chart challenges, indicator visualisations and a session-based practice journal.

## Features
- Dark, professional trading-academy interface
- Lessons on market structure, support/resistance, moving averages, RSI, MACD, volume and trade planning
- Historical chart quizzes with the future candles hidden until an answer is submitted
- SPY, QQQ, AAPL and MSFT; 15m, 30m, 60m and daily timeframes
- Review checklist and downloadable CSV practice journal
- No live trading or order placement

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Create a new public GitHub repository named `market-tutor`.
2. Upload `app.py`, `requirements.txt` and `README.md` to the repository root.
3. Open Streamlit Community Cloud and choose **Create app**.
4. Select your GitHub repository, branch `main`, and main file path `app.py`.
5. Deploy. Your app will receive a `streamlit.app` URL.

## Important notes
- Market data is fetched through `yfinance` / Yahoo Finance and may be delayed, incomplete or unavailable. Intraday history is limited by the data provider.
- Quiz outcomes are simple educational labels based on the close-to-close return over the next 12 bars with a neutral band. They are not trading signals or evidence of a profitable strategy.
- Quiz history currently persists only in the active Streamlit session. A later version can add a database for cross-device progress.
- This project does not place live orders.
