# Created by Arav Neroth 
# Date: 06/08/2025
import numpy as np
import pandas as pd
import yfinance as yf

from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

import plotly.graph_objects as plotgo
from plotly.subplots import make_subplots

