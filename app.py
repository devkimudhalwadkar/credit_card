#working version 2
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Set page config
st.set_page_config(
    page_title="Fraud Detection 💳",
    page_icon="🔒",
    layout="wide"
)

def create_synthetic_data():
    np.random.seed(42)
    normal_transactions = pd.DataFrame({
        'Transaction_Amount': np.random.uniform(10, 1000, 800),
        'Distance_from_Home': np.random.uniform(0, 50, 800),
        'Time_of_Day': np.random.uniform(6, 22, 800),
        'Frequency_Last_24h': np.random.uniform(0, 3, 800),
        'Class': 0,
        'Timestamp': [datetime.now() - timedelta(days=x) for x in np.random.uniform(0, 30, 800)]
    })
    
    fraud_transactions = pd.DataFrame({
        'Transaction_Amount': np.random.uniform(500, 5000, 200),
        'Distance_from_Home': np.random.uniform(100, 1000, 200),
        'Time_of_Day': np.random.uniform(0, 5, 200),
        'Frequency_Last_24h': np.random.uniform(5, 20, 200),
        'Class': 1,
        'Timestamp': [datetime.now() - timedelta(days=x) for x in np.random.uniform(0, 30, 200)]
    })
    
    df = pd.concat([normal_transactions, fraud_transactions])
    df['Transaction_Date'] = df['Timestamp'].dt.date
    return df

def create_visualizations(df):
    # 1. Amount Distribution
    fig_amount = px.histogram(
        df, 
        x='Transaction_Amount',
        color='Class',
        nbins=50,
        title='Distribution of Transaction Amounts',
        color_discrete_map={0: 'blue', 1: 'red'},
        labels={'Class': 'Transaction Type', 'Transaction_Amount': 'Amount ($)'}
    )
    fig_amount.update_layout(bargap=0.1)
    
    # 2. Time of Day Pattern
    fig_time = px.scatter(
        df,
        x='Time_of_Day',
        y='Transaction_Amount',
        color='Class',
        title='Transaction Patterns by Time of Day',
        color_discrete_map={0: 'blue', 1: 'red'},
        labels={'Class': 'Transaction Type', 'Time_of_Day': 'Hour of Day', 'Transaction_Amount': 'Amount ($)'}
    )
    
    # 3. Distance vs Amount
    fig_distance = px.scatter(
        df,
        x='Distance_from_Home',
        y='Transaction_Amount',
        color='Class',
        title='Transaction Amount vs Distance from Home',
        color_discrete_map={0: 'blue', 1: 'red'},
        labels={'Class': 'Transaction Type', 'Distance_from_Home': 'Distance (miles)', 'Transaction_Amount': 'Amount ($)'}
    )
    
    # 4. Fraud Trend Over Time
    daily_fraud = df[df['Class']==1].groupby('Transaction_Date').size().reset_index()
    daily_fraud.columns = ['Date', 'Fraud_Count']
    fig_trend = px.line(
        daily_fraud,
        x='Date',
        y='Fraud_Count',
        title='Daily Fraud Transactions',
        labels={'Fraud_Count': 'Number of Fraudulent Transactions', 'Date': 'Date'}
    )
    
    return fig_amount, fig_time, fig_distance, fig_trend

def train_or_load_model():
    model_path = 'fraud_model_simple.joblib'
    scaler_path = 'scaler_simple.joblib'
    
    # Force retrain
    if os.path.exists(model_path):
        os.remove(model_path)
    if os.path.exists(scaler_path):
        os.remove(scaler_path)
    
    # Create and prepare data
    df = create_synthetic_data()
    X = df.drop(['Class', 'Timestamp', 'Transaction_Date'], axis=1)
    y = df['Class']
    
    # Scale and train
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = LogisticRegression(max_iter=1000, class_weight={0:1, 1:10})
    model.fit(X_scaled, y)
    
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    
    return model, scaler, df

def create_gauge_chart(score, title):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score * 100,
        title = {'text': title},
        domain = {'x': [0, 1], 'y': [0, 1]},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgreen"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    ))
    return fig

def main():
    st.title("💳 Advanced Credit Card Fraud Detection Dashboard")
    
    # Load or train model
    with st.spinner('Loading model and generating visualizations...'):
        model, scaler, df = train_or_load_model()
        fig_amount, fig_time, fig_distance, fig_trend = create_visualizations(df)
    
    # Create tabs
    tab1, tab2 = st.tabs(["Fraud Detection", "Analytics Dashboard"])
    
    with tab1:
        st.write("### Enter Transaction Details:")
        
        # Add a template selector
        template = st.selectbox(
            "Choose a template or enter custom values:",
            ["Custom Input", "Likely Fraud Example", "Normal Transaction Example"]
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            if template == "Likely Fraud Example":
                amount = st.number_input("Transaction Amount ($)", value=3000.0, min_value=0.0)
                distance = st.number_input("Distance from Home (miles)", value=500.0, min_value=0.0)
            elif template == "Normal Transaction Example":
                amount = st.number_input("Transaction Amount ($)", value=100.0, min_value=0.0)
                distance = st.number_input("Distance from Home (miles)", value=5.0, min_value=0.0)
            else:
                amount = st.number_input("Transaction Amount ($)", value=0.0, min_value=0.0)
                distance = st.number_input("Distance from Home (miles)", value=0.0, min_value=0.0)
        
        with col2:
            if template == "Likely Fraud Example":
                time = st.number_input("Time of Day (24h format)", value=3.0, min_value=0.0, max_value=24.0)
                frequency = st.number_input("Number of Transactions in Last 24h", value=10.0, min_value=0.0)
            elif template == "Normal Transaction Example":
                time = st.number_input("Time of Day (24h format)", value=14.0, min_value=0.0, max_value=24.0)
                frequency = st.number_input("Number of Transactions in Last 24h", value=2.0, min_value=0.0)
            else:
                time = st.number_input("Time of Day (24h format)", value=12.0, min_value=0.0, max_value=24.0)
                frequency = st.number_input("Number of Transactions in Last 24h", value=0.0, min_value=0.0)

        if st.button("Check Transaction 🔍"):
            # Prepare input data
            input_data = np.array([[amount, distance, time, frequency]])
            input_scaled = scaler.transform(input_data)
            
            # Make prediction
            prediction = model.predict(input_scaled)
            probability = model.predict_proba(input_scaled)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction[0] == 1:
                    st.error("⚠️ Warning: Potential Fraudulent Transaction Detected!")
                else:
                    st.success("✅ Transaction Appears Safe!")
                
                # Risk factors analysis
                st.write("### 🔍 Risk Factors:")
                if amount > 1000:
                    st.write("- ⚠️ High transaction amount")
                if distance > 100:
                    st.write("- ⚠️ Unusual distance from home")
                if time < 6 or time > 22:
                    st.write("- ⚠️ Suspicious transaction time")
                if frequency > 5:
                    st.write("- ⚠️ High transaction frequency")
            
            with col2:
                # Add gauge chart for risk score
                fig_gauge = create_gauge_chart(
                    probability[0][1],
                    "Risk Score"
                )
                st.plotly_chart(fig_gauge)
    
    with tab2:
        st.write("### 📊 Transaction Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(fig_amount, use_container_width=True)
            st.plotly_chart(fig_distance, use_container_width=True)
        
        with col2:
            st.plotly_chart(fig_time, use_container_width=True)
            st.plotly_chart(fig_trend, use_container_width=True)
        
        # Add summary statistics
        st.write("### 📈 Summary Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Average Transaction Amount",
                f"${df['Transaction_Amount'].mean():.2f}",
                f"{df['Transaction_Amount'].std():.2f}"
            )
        
        with col2:
            fraud_rate = (df['Class'].mean() * 100)
            st.metric(
                "Fraud Rate",
                f"{fraud_rate:.2f}%",
                f"{fraud_rate - 2.5:.2f}%"
            )
        
        with col3:
            st.metric(
                "Total Transactions",
                len(df),
                f"+{len(df) - 900}"
            )

if __name__ == "__main__":
    main()
