# 📈 ML Stock Signal & Ranking System

A machine-learning–based stock **signal and ranking system** that predicts short-term direction, estimates expected returns, and ranks stocks based on confidence and risk-adjusted performance.

Built for:
-  Learning machine learning & quantitative finance
-  Interview and portfolio demonstration
-  Research-style stock experimentation (not live trading)

>  **Disclaimer**  
> This project is for educational and research purposes only.  
> It is **NOT financial advice**.

---

##  What This Project Does

This system does **NOT** predict exact stock prices.

Instead, it answers:
> “Is there a statistical edge *tomorrow* based on historical data?”

It provides:
- Direction prediction (**UP / DOWN**)
- Confidence score (probability)
- Expected return on UP days
- Market trend filtering using SPY
- Ranking of stocks by signal strength
- Frequent **NO TRADE** signals (by design)

---

##  Key Concepts Used

- Supervised machine learning
- Feature engineering
- Expected value
- Market regime filtering
- Risk-adjusted ranking

This mirrors real quantitative trading logic.

---

##  How Predictions Work

### Target Definition

Each trading day is labeled as:

- **1 (UP)** → next-day return > 0  
- **0 (DOWN)** → next-day return ≤ 0  

This makes the task a **classification problem**, not price prediction.

---

##  Features Used

For each stock:
- Daily return
- 5-day moving average
- 20-day moving average
- Short-term volatility

These features capture **trend, momentum, and risk**.

---

##  Model

- `RandomForestClassifier`
- One model per ticker
- Stored using `joblib`

Each saved model contains:
- Trained classifier
- Feature list
- Expected return on UP days

---

##  Expected Return (Important)

**Expected Return (UP days)** means:

> The *average next-day return* **only on days where the model correctly predicted UP**

Example:
Expected Return: 0.14%


This is **not guaranteed profit**.  
It is used for **ranking signals**, not promises.

---

##  Confidence (Important)

Confidence is the model’s probability estimate.

Example:


Confidence: 0.57


Meaning:
> Historically, the model was correct ~57% of the time in similar conditions.

---

##  Ranking Score

Stocks are ranked using:



Score = Confidence × Expected Return


This balances:
- Accuracy
- Profit potential

High score = better risk-adjusted signal.

---

##  Market Filter (SPY)

Before trading any stock:
- If SPY predicts **DOWN** → ❌ no long trades
- If SPY predicts **UP** → ✅ signals allowed

This avoids trading against overall market trend.

---

##  How To Run

###  Activate virtual environment (Windows)

```bash
cd C:\Projects\ml-stock-predictor
.\venv\Scripts\Activate.ps1

 Install dependencies
pip install -r requirements.txt

 Train models
python src/train.py


Downloads data automatically

Trains models

Saves them to /models/

4️ Predict a single stock
python src/predict.py SPY


Example output:

SPY Prediction for Tomorrow
Direction: UP 📈
Confidence: 57.23%
Expected Return (if UP): 0.09%

5️ Daily ranking command
python src/daily_signal.py


Example:

 DAILY MARKET SIGNAL

SPY: UP | Confidence: 61.2%
------------------------------------
NVDA | UP | Conf: 68.4% | ExpRet: 0.21% | Score: 0.0014
AAPL | UP | Conf: 61.0% | ExpRet: 0.12% | Score: 0.0007


If market is bearish:

 Market is bearish. No long trades today.

 Project Structure
ml-stock-predictor/
│
├── src/
│   ├── train.py
│   ├── predict.py
│   ├── daily_signal.py
│   ├── data_loader.py
│   └── features.py
│
├── models/
├── requirements.txt
├── README.md
└── LICENSE

 Tech Stack

Python

pandas

NumPy

scikit-learn

yfinance

joblib