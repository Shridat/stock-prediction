import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import time
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import seaborn as sns
import plotly.graph_objs as go
from datetime import datetime, timedelta
import numpy as np

def create_features(data):
    data['Year'] = data['Date'].dt.year
    data['Month'] = data['Date'].dt.month
    data['Day'] = (data['Date']-data['Date'].min()).dt.days
    data['DayOfWeek'] = data['Date'].dt.dayofweek
    data['DayOfYear'] = data['Date'].dt.dayofyear
    return data

def add_tech_indicators(data):
    data['SMA_20'] = data['Close/Last'].rolling(window=20).mean()
    data['EMA_20'] = data['Close/Last'].ewm(span=20,adjust=False).mean()
    data['Bollinger_Upper'] = data['SMA_20']+(data['Close/Last'].rolling(window=20).std()*2)
    data['Bollinger_Lower'] = data['EMA_20']-(data['Close/Last'].rolling(window=20).std()*2)
    return data

def filter(data,years):
    end_date = data['Date'].max()
    start_date = end_date - timedelta(days=365*years)
    return data[data['Date'] >= start_date]

df = pd.read_csv('new_data.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = create_features(df)
df = add_tech_indicators(df)

X = df[['Low','High','Volume']]
Y = df['Close/Last']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

st.title('Stock Prediction Analysis')
# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
tabs = st.tabs(["Historical Data Prediction", "Correlation Heatmap", "Technical Indicators", "CandleStick Chart", "Trading Practice"])


companies = df['Company'].unique()
selected_company = st.sidebar.selectbox('Select Company',companies)
st.sidebar.title("Configuration")
def user_input_features():
    low = st.sidebar.number_input('Low', value=float(df['Low'].mean()))
    high = st.sidebar.number_input('High', value=float(df['High'].mean()))
    volume = st.sidebar.number_input('Volume', value=float(df['Volume'].min()))
    data = {'Low': low,
            'High': high,
            'Volume': volume}
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()
prediction = model.predict(input_df)

filtered_data = df[df['Company'] == selected_company]
chart_placeholder = st.empty()
start_index = 0
end_index = 20  # Initial window size
increment = 1
years = df['year'].unique()
periods = [10, 5, 2, 1, 0.5, 0.25, 0.084]

st.sidebar.subheader('Predicted Closing Price')
st.sidebar.write(prediction[0])
mse = mean_squared_error(y_test, y_pred)
st.sidebar.subheader('Model Performance Metrics')
st.sidebar.write(f'Mean Squared Error: {mse}')

with tabs[0]:
    period_tabs = st.tabs(['10Y','5Y','2Y','1Y','6M','3M','1M'])
    for i,period in enumerate(periods):
        with period_tabs[i]:
            time.sleep(0.1)
            st.subheader(f'{selected_company} Stock Prices')
            data = filter(filtered_data,period)
            st.line_chart(data.set_index('Date')['Close/Last'])

with tabs[1]:
    time.sleep(0.1)
    st.subheader('Correlation Heatmap')
    corr = filtered_data[['Low','High','Volume','Close/Last']].corr()
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

with tabs[2]:
    time.sleep(0.1)
    st.subheader('Technical Indicators')
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(filtered_data['Date'], filtered_data['Close/Last'], label='Close Price')
    ax.plot(filtered_data['Date'], filtered_data['SMA_20'], label='SMA 20', linestyle='dashed')
    ax.plot(filtered_data['Date'], filtered_data['EMA_20'], label='EMA 20', linestyle='dashed')
    ax.plot(filtered_data['Date'], filtered_data['Bollinger_Upper'], label='Bollinger Upper', linestyle='dashed')
    ax.plot(filtered_data['Date'], filtered_data['Bollinger_Lower'], label='Bollinger Lower', linestyle='dashed')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.set_title('Technical Indicators')
    ax.legend()
    st.pyplot(fig)

st.sidebar.subheader("Export Data to CSV")
csv = filtered_data.to_csv(index=False)
st.sidebar.download_button('Download data as csv', data=csv, file_name='stock_data.csv')

with tabs[3]:
    fig = go.Figure(data=[go.Candlestick(x=filtered_data['Date'],
                                         open=filtered_data['Open'],
                                         high=filtered_data['High'],
                                         low=filtered_data['Low'],
                                         close=filtered_data['Close/Last'])])
    fig.update_layout(title=f'{selected_company} Candlestick Chart',
                        yaxis_title='Stock Price',
                        xaxis_title='Date',
                        width = 1000,
                        height=600,
                        xaxis_rangeslider_visible=False)
    st.plotly_chart(fig,use_container_width=True)

X = filtered_data[['Day', 'Open', 'High', 'Low', 'Volume']]
Y = filtered_data['Close/Last']
model1 = LinearRegression()
model1.fit(X,Y)

def update_plot(future_day):
    future_data = np.array([[future_day, today_data['Open'].values[0], today_data['High'].values[0], today_data['Low'].values[0], today_data['Volume'].values[0]]])
    future_price = model1.predict(future_data)[0]
    future_price = round(future_price,2)
    future_data = filtered_data['Date'].values[-1] + np.timedelta64(future_day - filtered_data['Day'].values[-1], 'D')
    #time.sleep(0.5)
    fig.add_trace(go.Scatter(x=[future_date], y=[future_price], mode='markers+lines', name='Predicted Price'))
    fig.update_layout(title=f'{selected_company} Trade Prediction',
                          xaxis_title='Date',
                          yaxis_title='Price',
                          width=1000, height=600)
    chart_placeholder.plotly_chart(fig, use_container_width=True)
    #st.plotly_chart(fig, use_container_width=True)
    return future_price

with tabs[4]:
    st.subheader("Welcome to Trading Practice")
    baseline = filtered_data['Close/Last'].values[-1]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=filtered_data['Date'], y=filtered_data['Close/Last'], mode='lines', name='Close Price'))
    fig.add_trace(go.Scatter(x=[filtered_data['Date'].min(), filtered_data['Date'].max()], y=[baseline, baseline], mode='lines', name='Baseline Strike Price', line=dict(dash='dash')))
    fig.update_layout(title=f'{selected_company} Stock Price and Baseline Strike Price',
                  xaxis_title='Date',
                  yaxis_title='Price',
                  width=1000, height=600)
    chart_placeholder = st.empty()
    chart_placeholder.plotly_chart(fig, use_container_width=True)
    #st.plotly_chart(fig, use_container_width=True)
    trade_action = st.selectbox("Action", ["Call","Put"])
    today = datetime.now().date()
    future_date = st.date_input("Days to predict ahead",today + timedelta(days=5))
    if st.button("Execute Trade"):
        predict_days = (future_date-today).days
        today_data = filtered_data[filtered_data['Date']==filtered_data['Date'].max()]
        success = False
        if not today_data.empty:
            today_day = (today_data['Date'].values[0] - filtered_data['Date'].min()).days
            future_day = today_day + predict_days
            future_price = update_plot(future_day)
            time.sleep(0.5)

        if trade_action=='Call':
            success = future_price > baseline
        elif trade_action=="Put":
            success = future_price< baseline
        
        if success:
            st.success(f"Successful {trade_action}! Predicted future price ({future_price:.2f}) is {'higher' if trade_action == 'Call' else 'lower'} than the baseline strike price ({baseline:.2f}).")
        else:
            st.error(f"Unsuccessful {trade_action}. Predicted future price ({future_price:.2f}) is not {'higher' if trade_action == 'Call' else 'lower'} than the baseline strike price ({baseline:.2f}).")
            
            
            
            
            




