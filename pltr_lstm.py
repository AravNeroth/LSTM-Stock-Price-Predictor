# Created by Arav Neroth 
# Date: 06/08/2025

import os
# quiet TensorFlow C++ logs and disable oneDNN optimizations (silence annoying stuff in terminal)
os.environ['TF_CPP_MIN_LOG_LEVEL']   = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Input, LSTM, Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import keras.backend as K

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.utils import class_weight

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def focal_loss(alpha=0.25, gamma=2.0):
    """
    Focal loss for binary classification.
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)        for y=1
             -(1-alpha) * p_t^gamma * log(1-p_t)         for y=0
    """
    def loss_fn(y_true, y_pred):
        # Clip predictions to prevent log(0) error and keep within [ε, 1−ε]
        y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1 - K.epsilon())

        # Compute p_t for true positives and negatives
        pt_pos = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_neg = tf.where(tf.equal(y_true, 0), 1 - y_pred, tf.ones_like(y_pred))

        # Focal loss components
        loss_pos = -alpha * tf.pow(1 - pt_pos, gamma) * tf.math.log(pt_pos)
        loss_neg = -(1 - alpha) * tf.pow(1 - pt_neg, gamma) * tf.math.log(pt_neg)

        # Combine losses and return mean
        return tf.reduce_mean(loss_pos + loss_neg)

    return loss_fn

def get_stock_data(symbol, start_date, end_date):
    df = yf.download(symbol, start=start_date, end=end_date, auto_adjust=True)

    # SMAs & price-change
    df['SMA5']  = df['Close'].rolling(5).mean()
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA50'] = df['Close'].rolling(50).mean()
    df['Price_Change'] = df['Close'].pct_change()

    # RSI14
    delta = df['Close'].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta).clip(lower=0).rolling(14).mean()
    rs    = gain / loss
    df['RSI14'] = 100 - (100 / (1 + rs))

    # On-Balance Volume
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

    # target
    df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    df.dropna(inplace=True)
    return df

def prepare_data(df, look_back=20):
    features = ['Close','SMA5','SMA20','SMA50','Price_Change','RSI14','OBV']
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[features])
    X, y = [], []
    for i in range(look_back, len(scaled)-1):
        X.append(scaled[i-look_back:i])
        y.append(df['Target'].iloc[i])
    return np.array(X), np.array(y)


def plot_stock_data(df, ticker):
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        subplot_titles=(f'{ticker} Price & SMAs', 'Volume')
    )
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'], name='OHLC'
    ), row=1, col=1)
    for sma in ['SMA5','SMA20','SMA50']:
        fig.add_trace(go.Scatter(x=df.index, y=df[sma], name=sma), row=1, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], name='Volume'), row=2, col=1)
    fig.update_layout(xaxis_rangeslider_visible=False, height=600)
    fig.show()

def build_model(input_shape):
    m = Sequential([
        Input(shape=input_shape),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dropout(0.3),
        Dense(1, activation='sigmoid')
    ])
    m.compile(
        optimizer=Adam(5e-4),
        loss=focal_loss(alpha=0.25, gamma=2.0),
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    return m

def main():
    print("=== Starting LSTM Stock Predictor ===")
    ticker = "AAPL"
    today  = pd.Timestamp.today()
    start  = (today - pd.DateOffset(years=10)).date().isoformat()
    end    = today.date().isoformat()
    print(f"Loading {ticker} from {start} → {end}")

    df = get_stock_data(ticker, start, end)
    plot_stock_data(df, ticker)

    look_back = 10
    X, y      = prepare_data(df, look_back)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # class weights
    cw = class_weight.compute_class_weight('balanced',
                                           classes=np.unique(y_tr),
                                           y=y_tr)
    cw_dict = dict(zip(np.unique(y_tr), cw))
    print("Class weights:", cw_dict)

    model = build_model((look_back, X.shape[2]))
    early = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

    print("Training…")
    model.fit(
        X_tr, y_tr,
        epochs=100,
        batch_size=64,
        validation_split=0.1,
        class_weight=cw_dict,
        callbacks=[early],
        verbose=2
    )

    loss, acc, auc = model.evaluate(X_te, y_te, verbose=0)
    print(f"\nTest Loss: {loss:.4f}, Accuracy: {acc:.4f}, AUC: {auc:.4f}")

    # threshold sweep for macro-F1
    probs = model.predict(X_te).flatten()
    best_thr, best_f1 = 0.5, 0.0
    for thr in np.linspace(0.1, 0.9, 81):
        preds = (probs > thr).astype(int)
        f1    = f1_score(y_te, preds, average='macro')
        if f1 > best_f1:
            best_f1, best_thr = f1, thr

    print(f"\nBest threshold = {best_thr:.2f} (macro-F1 = {best_f1:.3f})")
    final_preds = (probs > best_thr).astype(int)
    print("\nFinal Classification Report:")
    print(classification_report(y_te, final_preds, zero_division=0))

# call main() unconditionally so you always run it
main()
