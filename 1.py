import pandas as pd
import numpy as np
from base import detect_vertical_leakage, detect_horizontal_leakage
data = pd.read_csv('./data/daily_adjusted_AAPL.csv')
prediction_window = 5
data['target'] = data['open'].shift(-prediction_window)
def create_features1(data_input):
    data = data_input.copy()
    close_5days_before = data['close'].shift(5)
    data['return_5day'] = (data['close'] - close_5days_before)/close_5days_before

    # past 2 day return. Leaky uses price after 2 days
    close_2days_before = data['close'].shift(-2)
    data['return_2day_leaky'] = (data['close'] - close_2days_before)/close_2days_before
    data['open_10day_before_leaky'] = data['open'].shift(-10)

    return data
input_feature_cols = ['open', 'high', 'low', 'close']
output_feature_cols = ['return_5day', 'return_2day_leaky', 'open_10day_before_leaky']
detect_vertical_leakage(create_features1, data, input_feature_cols, output_feature_cols, only_nan=False, direction='upward')

def create_features2(data_input):
    data = data_input.copy()
    # past 5 day return
    close_5days_before = data['close'].shift(5)
    data['return_5day'] = (data['close'] - close_5days_before)/close_5days_before

    # past 2 day return. Leaky uses price after 2 days
    close_2days_before = data['close'].shift(-2)
    data['return_2day_leaky'] = ((data['close'] - close_2days_before)/close_2days_before).fillna(0.1)
    data['open_10day_before_leaky'] = data['open'].shift(-10).fillna(100)

    return data
input_feature_cols = ['open', 'high', 'low', 'close']
output_feature_cols = ['return_5day', 'return_2day_leaky', 'open_10day_before_leaky']
detect_vertical_leakage(create_features2, data, input_feature_cols, output_feature_cols, only_nan=False,
                       direction='upward')
data = pd.read_csv('./data/daily_adjusted_AAPL.csv')
# predict price after 5 days
prediction_window = 5
data['target'] = data['open'].shift(-prediction_window)

def create_features3(data_input):
    data = data_input.copy()
    # past 2 and 5 day return
    close_5days_before = data['close'].shift(5)
    close_2days_before = data['close'].shift(2)

    data['return_5day'] = (data['close'] - close_5days_before)/close_5days_before
    data['return_2day'] = (data['close'] - close_2days_before)/close_2days_before

    # leaky feature which uses target column
    data['open_1day_before_leaky'] = data['target'].shift(-1)

    return data
target_cols = ['target']
input_feature_cols = ['open', 'high', 'low', 'close']
output_feature_cols = ['return_5day', 'return_2day', 'open_1day_before_leaky']
# checks for leakage from target cols to feature cols and from input feature cols to target cols
detect_horizontal_leakage(create_features3, data, target_cols, output_feature_cols, input_feature_cols)
