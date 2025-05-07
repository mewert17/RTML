import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

def add_technical_indicators(df):
    """
    Add various technical indicators to the dataframe
    
    Parameters:
    df (pandas.DataFrame): DataFrame with OHLCV data
    
    Returns:
    pandas.DataFrame: DataFrame with added technical indicators
    """
    # Make a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Ensure the dataframe has the required columns
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"DataFrame must contain {col} column")
    
    # Moving Average Convergence Divergence (MACD)
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
    
    # Bollinger Bands
    window = 20
    df['SMA20'] = df['Close'].rolling(window=window).mean()
    df['STD20'] = df['Close'].rolling(window=window).std()
    df['Bollinger_Upper'] = df['SMA20'] + (df['STD20'] * 2)
    df['Bollinger_Lower'] = df['SMA20'] - (df['STD20'] * 2)
    df['Bollinger_Width'] = (df['Bollinger_Upper'] - df['Bollinger_Lower']) / df['SMA20']
    df['Bollinger_Pct'] = (df['Close'] - df['Bollinger_Lower']) / (df['Bollinger_Upper'] - df['Bollinger_Lower'])
    
    # Stochastic Oscillator
    n = 14
    df['Stoch_K'] = 100 * ((df['Close'] - df['Low'].rolling(window=n).min()) / 
                           (df['High'].rolling(window=n).max() - df['Low'].rolling(window=n).min()))
    df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
    
    # On-Balance Volume (OBV)
    df['OBV'] = np.nan
    df.loc[0, 'OBV'] = df.loc[0, 'Volume']
    for i in range(1, len(df)):
        if df.loc[i, 'Close'] > df.loc[i-1, 'Close']:
            df.loc[i, 'OBV'] = df.loc[i-1, 'OBV'] + df.loc[i, 'Volume']
        elif df.loc[i, 'Close'] < df.loc[i-1, 'Close']:
            df.loc[i, 'OBV'] = df.loc[i-1, 'OBV'] - df.loc[i, 'Volume']
        else:
            df.loc[i, 'OBV'] = df.loc[i-1, 'OBV']
    
    # Average True Range (ATR)
    df['TR'] = np.maximum(
        np.maximum(
            df['High'] - df['Low'],
            np.abs(df['High'] - df['Close'].shift(1))
        ),
        np.abs(df['Low'] - df['Close'].shift(1))
    )
    df['ATR'] = df['TR'].rolling(window=14).mean()
    
    # Rate of Change (ROC)
    df['ROC'] = ((df['Close'] / df['Close'].shift(10)) - 1) * 100
    
    # Commodity Channel Index (CCI)
    df['TP'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['SMA_TP'] = df['TP'].rolling(window=20).mean()
    df['MAD'] = df['TP'].rolling(window=20).apply(lambda x: pd.Series(x).mad())
    df['CCI'] = (df['TP'] - df['SMA_TP']) / (0.015 * df['MAD'])
    
    # Relative Strength Index (RSI)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Ichimoku Cloud
    df['Tenkan_Sen'] = (df['High'].rolling(window=9).max() + df['Low'].rolling(window=9).min()) / 2
    df['Kijun_Sen'] = (df['High'].rolling(window=26).max() + df['Low'].rolling(window=26).min()) / 2
    df['Senkou_Span_A'] = ((df['Tenkan_Sen'] + df['Kijun_Sen']) / 2).shift(26)
    df['Senkou_Span_B'] = ((df['High'].rolling(window=52).max() + df['Low'].rolling(window=52).min()) / 2).shift(26)
    df['Chikou_Span'] = df['Close'].shift(-26)
    
    # Money Flow Index (MFI)
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['Raw_Money_Flow'] = df['Typical_Price'] * df['Volume']
    df['Money_Flow_Positive'] = np.where(df['Typical_Price'] > df['Typical_Price'].shift(1), df['Raw_Money_Flow'], 0)
    df['Money_Flow_Negative'] = np.where(df['Typical_Price'] < df['Typical_Price'].shift(1), df['Raw_Money_Flow'], 0)
    df['Money_Flow_Positive_14'] = df['Money_Flow_Positive'].rolling(window=14).sum()
    df['Money_Flow_Negative_14'] = df['Money_Flow_Negative'].rolling(window=14).sum()
    df['Money_Ratio'] = df['Money_Flow_Positive_14'] / df['Money_Flow_Negative_14']
    df['MFI'] = 100 - (100 / (1 + df['Money_Ratio']))
    
    # Drop intermediate columns used for calculations
    columns_to_drop = ['EMA12', 'EMA26', 'TR', 'TP', 'SMA_TP', 'MAD', 'Typical_Price', 
                       'Raw_Money_Flow', 'Money_Flow_Positive', 'Money_Flow_Negative', 
                       'Money_Flow_Positive_14', 'Money_Flow_Negative_14', 'Money_Ratio']
    df = df.drop(columns=columns_to_drop, errors='ignore')
    
    return df

def plot_technical_indicators(df, symbol, save_path=None):
    """
    Plot the technical indicators for visual analysis
    
    Parameters:
    df (pandas.DataFrame): DataFrame with technical indicators
    symbol (str): Symbol of the cryptocurrency
    save_path (str, optional): Path to save the plots
    """
    # Create a figure with subplots
    fig, axs = plt.subplots(4, 1, figsize=(14, 20), gridspec_kw={'height_ratios': [3, 1, 1, 1]})
    
    # Plot 1: Price with Bollinger Bands
    axs[0].plot(df.index, df['Close'], label='Close Price', color='blue')
    axs[0].plot(df.index, df['SMA20'], label='SMA (20)', color='orange', alpha=0.7)
    axs[0].plot(df.index, df['Bollinger_Upper'], label='Upper Band', color='green', linestyle='--', alpha=0.7)
    axs[0].plot(df.index, df['Bollinger_Lower'], label='Lower Band', color='red', linestyle='--', alpha=0.7)
    axs[0].fill_between(df.index, df['Bollinger_Upper'], df['Bollinger_Lower'], color='gray', alpha=0.1)
    axs[0].set_title(f'{symbol} Price with Bollinger Bands')
    axs[0].set_ylabel('Price')
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)
    
    # Plot 2: MACD
    axs[1].plot(df.index, df['MACD'], label='MACD', color='blue')
    axs[1].plot(df.index, df['MACD_Signal'], label='Signal Line', color='red')
    axs[1].bar(df.index, df['MACD_Histogram'], label='Histogram', color='green', alpha=0.5)
    axs[1].set_title('MACD')
    axs[1].set_ylabel('Value')
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)
    
    # Plot 3: RSI
    axs[2].plot(df.index, df['RSI'], label='RSI', color='purple')
    axs[2].axhline(y=70, color='red', linestyle='--', alpha=0.7)
    axs[2].axhline(y=30, color='green', linestyle='--', alpha=0.7)
    axs[2].fill_between(df.index, df['RSI'], 70, where=(df['RSI'] >= 70), color='red', alpha=0.3)
    axs[2].fill_between(df.index, df['RSI'], 30, where=(df['RSI'] <= 30), color='green', alpha=0.3)
    axs[2].set_title('RSI')
    axs[2].set_ylabel('Value')
    axs[2].set_ylim(0, 100)
    axs[2].legend()
    axs[2].grid(True, alpha=0.3)
    
    # Plot 4: Stochastic Oscillator
    axs[3].plot(df.index, df['Stoch_K'], label='%K', color='blue')
    axs[3].plot(df.index, df['Stoch_D'], label='%D', color='red')
    axs[3].axhline(y=80, color='red', linestyle='--', alpha=0.7)
    axs[3].axhline(y=20, color='green', linestyle='--', alpha=0.7)
    axs[3].fill_between(df.index, df['Stoch_K'], 80, where=(df['Stoch_K'] >= 80), color='red', alpha=0.3)
    axs[3].fill_between(df.index, df['Stoch_K'], 20, where=(df['Stoch_K'] <= 20), color='green', alpha=0.3)
    axs[3].set_title('Stochastic Oscillator')
    axs[3].set_ylabel('Value')
    axs[3].set_ylim(0, 100)
    axs[3].legend()
    axs[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    plt.show()

def generate_trading_signals(df):
    """
    Generate buy/sell signals based on technical indicators
    
    Parameters:
    df (pandas.DataFrame): DataFrame with technical indicators
    
    Returns:
    pandas.DataFrame: DataFrame with added trading signals
    """
    # Make a copy to avoid modifying the original dataframe
    df = df.copy()
    
    # Initialize signal columns with neutral (0)
    df['MACD_Signal_Flag'] = 0
    df['Bollinger_Signal_Flag'] = 0
    df['RSI_Signal_Flag'] = 0
    df['Stochastic_Signal_Flag'] = 0
    df['OBV_Signal_Flag'] = 0
    df['CCI_Signal_Flag'] = 0
    df['MFI_Signal_Flag'] = 0
    df['Ichimoku_Signal_Flag'] = 0
    df['Combined_Signal_Flag'] = 0
    
    # MACD Signal: Buy when MACD crosses above Signal Line, Sell when crosses below
    df.loc[df['MACD'] > df['MACD_Signal'], 'MACD_Signal_Flag'] = 1  # Buy
    df.loc[df['MACD'] < df['MACD_Signal'], 'MACD_Signal_Flag'] = -1  # Sell
    
    # Bollinger Bands Signal: Buy when price touches lower band, Sell when price touches upper band
    df.loc[df['Close'] <= df['Bollinger_Lower'], 'Bollinger_Signal_Flag'] = 1  # Buy
    df.loc[df['Close'] >= df['Bollinger_Upper'], 'Bollinger_Signal_Flag'] = -1  # Sell
    
    # RSI Signal: Buy when RSI < 30 (oversold), Sell when RSI > 70 (overbought)
    df.loc[df['RSI'] < 30, 'RSI_Signal_Flag'] = 1  # Buy
    df.loc[df['RSI'] > 70, 'RSI_Signal_Flag'] = -1  # Sell
    
    # Stochastic Oscillator Signal: Buy when %K < 20, Sell when %K > 80
    df.loc[df['Stoch_K'] < 20, 'Stochastic_Signal_Flag'] = 1  # Buy
    df.loc[df['Stoch_K'] > 80, 'Stochastic_Signal_Flag'] = -1  # Sell
    
    # OBV Signal: Buy when OBV is rising while price is flat, Sell when OBV is falling while price is flat
    # Calculate OBV change and price change
    df['OBV_Change'] = df['OBV'].pct_change()
    df['Price_Change'] = df['Close'].pct_change()
    
    # Buy when OBV is rising significantly but price isn't
    df.loc[(df['OBV_Change'] > 0.01) & (abs(df['Price_Change']) < 0.005), 'OBV_Signal_Flag'] = 1
    
    # Sell when OBV is falling significantly but price isn't
    df.loc[(df['OBV_Change'] < -0.01) & (abs(df['Price_Change']) < 0.005), 'OBV_Signal_Flag'] = -1
    
    # Clean up temporary columns
    df = df.drop(columns=['OBV_Change', 'Price_Change'], errors='ignore')
    
    # CCI Signal: Buy when CCI < -100, Sell when CCI > 100
    df.loc[df['CCI'] < -100, 'CCI_Signal_Flag'] = 1  # Buy
    df.loc[df['CCI'] > 100, 'CCI_Signal_Flag'] = -1  # Sell
    
    # MFI Signal: Buy when MFI < 20, Sell when MFI > 80
    df.loc[df['MFI'] < 20, 'MFI_Signal_Flag'] = 1  # Buy
    df.loc[df['MFI'] > 80, 'MFI_Signal_Flag'] = -1  # Sell
    
    # Ichimoku Signal: Buy when price crosses above the cloud, Sell when price crosses below the cloud
    # Check if price is above the cloud (both Senkou Span A and B)
    df.loc[(df['Close'] > df['Senkou_Span_A']) & (df['Close'] > df['Senkou_Span_B']), 'Ichimoku_Signal_Flag'] = 1
    
    # Check if price is below the cloud (both Senkou Span A and B)
    df.loc[(df['Close'] < df['Senkou_Span_A']) & (df['Close'] < df['Senkou_Span_B']), 'Ichimoku_Signal_Flag'] = -1
    
    # Combined Signal: Simple majority vote of all signals
    signal_columns = [
        'MACD_Signal_Flag', 'Bollinger_Signal_Flag', 'RSI_Signal_Flag', 
        'Stochastic_Signal_Flag', 'OBV_Signal_Flag', 'CCI_Signal_Flag', 
        'MFI_Signal_Flag', 'Ichimoku_Signal_Flag'
    ]
    
    # Count buy and sell signals
    df['Buy_Count'] = (df[signal_columns] == 1).sum(axis=1)
    df['Sell_Count'] = (df[signal_columns] == -1).sum(axis=1)
    
    # Determine combined signal based on majority
    df.loc[df['Buy_Count'] > df['Sell_Count'], 'Combined_Signal_Flag'] = 1
    df.loc[df['Buy_Count'] < df['Sell_Count'], 'Combined_Signal_Flag'] = -1
    
    # Clean up temporary columns
    df = df.drop(columns=['Buy_Count', 'Sell_Count'], errors='ignore')
    
    return df

def plot_signals(df, symbol, save_path=None):
    """
    Plot price chart with buy/sell signals
    
    Parameters:
    df (pandas.DataFrame): DataFrame with trading signals
    symbol (str): Symbol of the cryptocurrency
    save_path (str, optional): Path to save the plot
    """
    plt.figure(figsize=(14, 7))
    
    # Plot price
    plt.plot(df.index, df['Close'], label='Close Price', color='blue', alpha=0.6)
    
    # Plot buy signals
    buy_signals = df[df['Combined_Signal_Flag'] == 1]
    plt.scatter(buy_signals.index, buy_signals['Close'], 
               color='green', label='Buy Signal', marker='^', s=100)
    
    # Plot sell signals
    sell_signals = df[df['Combined_Signal_Flag'] == -1]
    plt.scatter(sell_signals.index, sell_signals['Close'], 
               color='red', label='Sell Signal', marker='v', s=100)
    
    plt.title(f'{symbol} Price with Trading Signals')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    plt.tight_layout()
    plt.show()

def normalize_indicators(df):
    """
    Normalize technical indicators to a 0-1 range for model input
    
    Parameters:
    df (pandas.DataFrame): DataFrame with technical indicators
    
    Returns:
    pandas.DataFrame: DataFrame with normalized indicators
    """
    # Make a copy to avoid modifying the original dataframe
    df_norm = df.copy()
    
    # Select columns to normalize (exclude Date and categorical features)
    exclude_cols = ['Date'] + [col for col in df_norm.columns if 'Flag' in col]
    cols_to_normalize = [col for col in df_norm.columns if col not in exclude_cols]
    
    # Initialize the scaler
    scaler = MinMaxScaler()
    
    # Normalize the selected columns
    df_norm[cols_to_normalize] = scaler.fit_transform(df_norm[cols_to_normalize])
    
    return df_norm

if __name__ == "__main__":
    # Example usage
    import yfinance as yf
    
    # Download cryptocurrency data
    symbol = "BTC-USD"
    data = yf.download(symbol, start="2020-01-01", end="2023-01-01")
    
    # Add technical indicators
    data_with_indicators = add_technical_indicators(data)
    
    # Generate trading signals
    data_with_signals = generate_trading_signals(data_with_indicators)
    
    # Print the first few rows
    print(data_with_signals.head())
    
    # Plot the indicators
    plot_technical_indicators(data_with_signals, symbol)
    
    # Plot the trading signals
    plot_signals(data_with_signals, symbol)
    
    # Normalize the indicators
    normalized_data = normalize_indicators(data_with_signals)
    print("\nNormalized Data:")
    print(normalized_data.head())
    
    # Calculate performance metrics
    buy_and_hold_return = (data['Close'].iloc[-1] / data['Close'].iloc[0]) - 1
    
    # Filter for days with signals
    signal_days = data_with_signals[data_with_signals['Combined_Signal_Flag'] != 0].copy()
    
    if not signal_days.empty:
        # Calculate returns for signal days
        signal_days['Next_Day_Return'] = data_with_signals['Close'].pct_change().shift(-1)
        
        # Calculate strategy return (buy on buy signals, sell on sell signals)
        strategy_return = signal_days.loc[signal_days['Combined_Signal_Flag'] == 1, 'Next_Day_Return'].mean() * len(signal_days[signal_days['Combined_Signal_Flag'] == 1])
        strategy_return -= signal_days.loc[signal_days['Combined_Signal_Flag'] == -1, 'Next_Day_Return'].mean() * len(signal_days[signal_days['Combined_Signal_Flag'] == -1])
        
        print(f"\nStrategy Performance:")
        print(f"Buy and Hold Return: {buy_and_hold_return:.2%}")
        print(f"Strategy Return: {strategy_return:.2%}")
        print(f"Number of Buy Signals: {len(signal_days[signal_days['Combined_Signal_Flag'] == 1])}")
        print(f"Number of Sell Signals: {len(signal_days[signal_days['Combined_Signal_Flag'] == -1])}")
