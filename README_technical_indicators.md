# Technical Indicators for Cryptocurrency Price Prediction

This module provides a comprehensive set of technical indicators for cryptocurrency price analysis and prediction. It's designed to enhance machine learning models by providing engineered features that capture complex market dynamics.

## Features
The `technical_indicators.py` script implements the following technical indicators:

### 1. Moving Average Convergence Divergence (MACD)
- **Measures**: Trend direction, strength, momentum, and potential reversals
- **Signals**: Trend changes, momentum shifts, and potential reversals
- **Calculation**: Difference between 12-period and 26-period EMAs, with 9-period signal line

### 2. Bollinger Bands
- **Measures**: Volatility, relative price levels, and potential overbought/oversold conditions
- **Signals**: Volatility expansions/contractions and potential price reversals
- **Calculation**: 20-period SMA with upper/lower bands at 2 standard deviations

### 3. Stochastic Oscillator
- **Measures**: Momentum and relative position of closing price within a recent range
- **Signals**: Overbought/oversold conditions and momentum shifts
- **Calculation**: Compares current price to high-low range over 14 periods

### 4. On-Balance Volume (OBV)
- **Measures**: Cumulative buying and selling pressure based on volume
- **Signals**: Potential price breakouts and divergences
- **Calculation**: Running total of volume, added when price rises, subtracted when price falls

### 5. Average True Range (ATR)
- **Measures**: Market volatility
- **Signals**: Potential for large price movements
- **Calculation**: 14-period average of the true range (maximum of current high-low, high-previous close, previous close-low)

### 6. Rate of Change (ROC)
- **Measures**: Price momentum as percentage change
- **Signals**: Acceleration/deceleration in price movements
- **Calculation**: Percentage change in price over 10 periods

### 7. Commodity Channel Index (CCI)
- **Measures**: Cyclical overbought and oversold conditions
- **Signals**: Potential trend reversals and cyclical turning points
- **Calculation**: Deviation of typical price from its moving average, scaled by mean deviation

### 8. Relative Strength Index (RSI)
- **Measures**: Momentum and overbought/oversold conditions
- **Signals**: Potential price reversals and trend strength
- **Calculation**: Ratio of average gains to average losses over 14 periods

### 9. Ichimoku Cloud
- **Measures**: Support/resistance levels, trend direction, and momentum
- **Signals**: Trend direction, momentum shifts, and future support/resistance
- **Calculation**: Multiple components including Tenkan-sen, Kijun-sen, Senkou Span A/B, and Chikou Span

### 10. Money Flow Index (MFI)
- **Measures**: Buying and selling pressure incorporating both price and volume
- **Signals**: Overbought/oversold conditions and potential divergences
- **Calculation**: Volume-weighted RSI over 14 periods

## Usage

### Basic Usage

```python
import pandas as pd
import yfinance as yf
from technical_indicators import add_technical_indicators, plot_technical_indicators

# Download cryptocurrency data
symbol = "BTC-USD"
data = yf.download(symbol, start="2022-01-01", end="2023-01-01")

# Add technical indicators
data_with_indicators = add_technical_indicators(data)

# Print the first few rows
print(data_with_indicators.head())

# Plot the indicators
plot_technical_indicators(data_with_indicators, symbol)
```

### Generating Trading Signals

The module can generate buy/sell signals based on each technical indicator:

```python
from technical_indicators import generate_trading_signals, plot_signals

# Generate trading signals
data_with_signals = generate_trading_signals(data_with_indicators)

# Plot the signals
plot_signals(data_with_signals, symbol)

# View signal columns
signal_columns = [col for col in data_with_signals.columns if 'Flag' in col]
print(data_with_signals[signal_columns].head())
```

The `generate_trading_signals` function adds the following signal columns:

- `MACD_Signal_Flag`: Buy (1) when MACD crosses above Signal Line, Sell (-1) when crosses below
- `Bollinger_Signal_Flag`: Buy (1) when price touches lower band, Sell (-1) when price touches upper band
- `RSI_Signal_Flag`: Buy (1) when RSI < 30 (oversold), Sell (-1) when RSI > 70 (overbought)
- `Stochastic_Signal_Flag`: Buy (1) when %K < 20, Sell (-1) when %K > 80
- `OBV_Signal_Flag`: Buy (1) when OBV is rising while price is flat, Sell (-1) when OBV is falling while price is flat
- `CCI_Signal_Flag`: Buy (1) when CCI < -100, Sell (-1) when CCI > 100
- `MFI_Signal_Flag`: Buy (1) when MFI < 20, Sell (-1) when MFI > 80
- `Ichimoku_Signal_Flag`: Buy (1) when price is above the cloud, Sell (-1) when price is below the cloud
- `Combined_Signal_Flag`: Majority vote of all signals

### Evaluating Trading Strategy Performance

```python
# Calculate performance metrics
buy_and_hold_return = (data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1

# Filter for days with signals
signal_days = data_with_signals[data_with_signals['Combined_Signal_Flag'] != 0].copy()
signal_days['Next_Day_Return'] = data_with_signals['Close'].pct_change().shift(-1)

# Calculate strategy return
strategy_return = signal_days.loc[signal_days['Combined_Signal_Flag'] == 1, 'Next_Day_Return'].mean() * len(signal_days[signal_days['Combined_Signal_Flag'] == 1])
strategy_return -= signal_days.loc[signal_days['Combined_Signal_Flag'] == -1, 'Next_Day_Return'].mean() * len(signal_days[signal_days['Combined_Signal_Flag'] == -1])

print(f"Buy and Hold Return: {buy_and_hold_return:.2%}")
print(f"Strategy Return: {strategy_return:.2%}")
```

### Normalizing Indicators for Machine Learning

```python
from technical_indicators import normalize_indicators

# Normalize indicators to 0-1 range for model input
normalized_data = normalize_indicators(data_with_signals)
print(normalized_data.head())
```

## Integration with Machine Learning Models

These technical indicators and trading signals can significantly improve cryptocurrency price prediction models by capturing complex market dynamics that simple price and volume data alone cannot reveal.

### Example Integration

```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

# Prepare data
df = add_technical_indicators(data)
df['target'] = df['Close'].shift(-1)  # Predict next day's close
df = df.dropna()

# Select features
features = ['MACD', 'MACD_Signal', 'Bollinger_Width', 'Stoch_K', 'Stoch_D', 
            'RSI', 'ATR', 'ROC', 'CCI', 'OBV']
X = df[features]
y = df['target']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f"RMSE: {rmse:.2f}")

# Feature importance
importance = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)
print(importance)
```

## Best Practices

1. **Feature Selection**: Not all indicators are useful in all market conditions. Use feature importance analysis to select the most relevant indicators.

2. **Parameter Tuning**: Consider adjusting the default parameters of indicators based on the specific cryptocurrency and timeframe.

3. **Regime-Specific Models**: Train separate models for different market regimes (bull, bear, sideways) identified by these indicators.

4. **Lag Consideration**: Include lagged values of indicators to capture developing patterns.

5. **Indicator Combinations**: The module already provides a combined signal based on majority vote, but you can create custom combinations based on your specific strategy.

6. **Signal Filtering**: Consider filtering signals to reduce false positives, such as requiring confirmation from multiple indicators or using signal persistence (signal must be present for multiple days).

7. **Backtesting**: Always thoroughly backtest any trading strategy across different market conditions before using it for real trading.

## Requirements

- pandas
- numpy
- matplotlib
- scikit-learn (for normalization)

## License

[MIT License](LICENSE)
