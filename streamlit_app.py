# streamlit_app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from numpy.polynomial.polynomial import Polynomial

# ---------------- Streamlit Settings ----------------
st.set_page_config(page_title="Transformer Health Dashboard", layout="wide")
st.title("🔌 Transformer Health Dashboard")

# ---------------- Custom ML Model ----------------
class TransformerHealthModel:
    """Custom ML model for transformer health prediction"""
    
    def __init__(self):
        self.feature_weights = {
            'voltage': 0.25,
            'current': 0.20,
            'temperature': 0.35,
            'vibration': 0.20
        }
        self.optimal_ranges = {
            'voltage': {'min': 105, 'max': 115, 'optimal': 110},
            'current': {'min': 100, 'max': 200, 'optimal': 150},   # NOTE: convert mA → A if needed
            'temperature': {'min': 50, 'max': 85, 'optimal': 65},
            'vibration': {'min': 0, 'max': 4, 'optimal': 1.5}
        }
    
    def predict(self, X):
        predictions = []
        for row in X.values:
            voltage, current, temperature, vibration = row
            health_score = 100.0

            # Voltage penalty
            if voltage < self.optimal_ranges['voltage']['min'] or voltage > self.optimal_ranges['voltage']['max']:
                dev = min(abs(voltage - self.optimal_ranges['voltage']['min']),
                          abs(voltage - self.optimal_ranges['voltage']['max']))
                health_score -= dev * self.feature_weights['voltage']
            
            # Current penalty
            current_opt = self.optimal_ranges['current']['optimal']
            health_score -= abs(current - current_opt)/current_opt * 25 * self.feature_weights['current']

            # Temperature penalty
            temp_opt = self.optimal_ranges['temperature']['optimal']
            if temperature > temp_opt:
                health_score -= ((temperature - temp_opt)/10)**1.5 * 40 * self.feature_weights['temperature']

            # Vibration penalty
            vib_opt = self.optimal_ranges['vibration']['optimal']
            if vibration > vib_opt:
                health_score -= ((vibration - vib_opt)/2)**2 * 35 * self.feature_weights['vibration']

            # Random variation
            health_score += np.random.normal(0, 1.5)
            health_score = np.clip(health_score, 0, 100)
            predictions.append(round(health_score, 1))
        return np.array(predictions)

# ---------------- Load Google Sheets Data ----------------
@st.cache_data
def load_data_from_gsheet(csv_url):
    try:
        data = pd.read_csv(csv_url)

        # Expected columns
        required_cols = ['Timestamp', 'Voltage (V)', 'Current (mA)', 'Temperature ( C )', 'Vibration (mm/s2)']
        if not all(col in data.columns for col in required_cols):
            st.error(f"❌ Google Sheet missing required columns: {required_cols}")
            return pd.DataFrame()

        return data

    except Exception as e:
        st.error(f"⚠️ Error loading Google Sheet: {e}")
        return pd.DataFrame()

# ---------------- Utility Functions ----------------
def health_category(score):
    if score >= 70:
        return 'Good'
    elif score >= 40:
        return 'Moderate'
    else:
        return 'Poor'

def severity_color(category):
    return {'Good':'green', 'Moderate':'orange', 'Poor':'red'}.get(category, 'gray')

def maintenance_advice(category):
    if category == 'Poor':
        return '🚨 Immediate Maintenance Required'
    elif category == 'Moderate':
        return '⚠️ Schedule Maintenance Soon'
    else:
        return '✅ Operating Normally'

# ---------------- Main App ----------------
def main():
    # Replace with your published Google Sheet CSV link
    url = "https://docs.google.com/spreadsheets/d/e/2PACX-1vQomV0L7gsqd69qDjWF8i8uhkPzR4ga6syUymJFvXap5Bx8FXAdSdjn33Ax3OG5LiBozH6aoOcTzfNj/pub?output=csv"
    data = load_data_from_gsheet(url)

    if data.empty:
        st.stop()

    # Initialize model
    model = TransformerHealthModel()

    # Predictions
    feature_cols = ['Voltage (V)', 'Current (mA)', 'Temperature ( C )', 'Vibration (mm/s2)']
    data['Health_Score'] = model.predict(data[feature_cols])
    data['Health_Category'] = [health_category(h) for h in data['Health_Score']]
    data['Maintenance_Advice'] = data['Health_Category'].apply(maintenance_advice)

    # ---------------- Estimate Transformer Lifespan ----------------
    BASE_LIFESPAN = 30  # max lifespan in years

    def estimate_lifespan(row):
        health_score = row['Health_Score']  # 0–100
        estimated_remaining = BASE_LIFESPAN * (health_score / 100)
        return round(estimated_remaining, 1)

    data['Estimated_Lifespan (years)'] = data.apply(estimate_lifespan, axis=1)

    # ---------------- Summary Metrics ----------------
    st.subheader("📊 Summary Metrics")
    avg_health = data['Health_Score'].mean()
    good_count = sum(data['Health_Category'] == 'Good')
    moderate_count = sum(data['Health_Category'] == 'Moderate')
    poor_count = sum(data['Health_Category'] == 'Poor')
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Average Health", f"{avg_health:.1f}%")
    col2.metric("Good Readings", good_count)
    col3.metric("Moderate Readings", moderate_count)
    col4.metric("Poor Readings", poor_count)

    # Estimated Lifespan Metric
    st.subheader("🕒 Estimated Transformer Lifespan")
    avg_life = data['Estimated_Lifespan (years)'].mean()
    st.metric("Average Remaining Lifespan", f"{avg_life:.1f} years")

    # Optional bar chart for lifespan
    fig_life = px.bar(
        data, 
        x="Timestamp", 
        y="Estimated_Lifespan (years)", 
        title="Estimated Lifespan per Reading",
        text="Estimated_Lifespan (years)"
    )
    st.plotly_chart(fig_life, use_container_width=True)

    # Cost Impact
    cost_per_failure = 3500
    potential_savings = poor_count * cost_per_failure
    st.metric("💰 Potential Savings from Preventive Action", f"Rs {potential_savings:,}")
    
    # Latest reading gauge
    st.subheader("⚡ Latest Transformer Status")
    latest = data.iloc[-1]
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=latest['Health_Score'],
        title={'text': "Health Score"},
        gauge={
            'axis': {'range':[0,100]},
            'bar': {'color': severity_color(latest['Health_Category'])}
        }
    ))
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Trend chart
    st.subheader("📈 Health Trend")
    fig_trend = px.line(data, x="Timestamp", y='Health_Score', markers=True, title="Health Score Over Time")
    fig_trend.add_scatter(
        x=data["Timestamp"], y=data['Health_Score'],
        mode='markers',
        marker=dict(color=[severity_color(c) for c in data['Health_Category']], size=8),
        name="Health Category"
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    # Trend Slope (Predictive Insight)
    last_n = min(10, len(data))
    y = data['Health_Score'][-last_n:].values
    x = np.arange(last_n)
    coefs = Polynomial.fit(x, y, 1).convert().coef
    trend_slope = coefs[1]
    st.info(f"📉 Health Trend Slope: {trend_slope:.2f} per reading (negative → declining health)")

    # Parameter impact analysis
    data['Voltage_Impact'] = abs(data['Voltage (V)'] - 110) * 0.25
    data['Current_Impact'] = abs(data['Current (mA)'] - 150) * 0.20
    data['Temp_Impact'] = np.maximum(0, data['Temperature ( C )'] - 65) * 0.35
    data['Vibration_Impact'] = np.maximum(0, data['Vibration (mm/s2)'] - 1.5) * 0.20
    impact_avg = data[['Voltage_Impact','Current_Impact','Temp_Impact','Vibration_Impact']].mean()
    fig_impact = px.bar(
        impact_avg, x=impact_avg.index, y=impact_avg.values, 
        title="Average Parameter Impact on Health Score",
        labels={'x':'Parameter','y':'Impact Score'}
    )
    st.plotly_chart(fig_impact, use_container_width=True)

    # Recent readings table with maintenance advice
    st.subheader("📋 Recent Readings & Maintenance Advice")
    display_data = data.tail(10)[['Timestamp','Voltage (V)','Current (mA)','Temperature ( C )','Vibration (mm/s2)','Health_Score','Health_Category','Maintenance_Advice','Estimated_Lifespan (years)']]
    st.dataframe(display_data)

    # KPI summary tiles
    st.subheader("🏷️ Key KPIs")
    col1,col2,col3,col4 = st.columns(4)
    col1.metric("Good %", f"{good_count/len(data)*100:.1f}%")
    col2.metric("Moderate %", f"{moderate_count/len(data)*100:.1f}%")
    col3.metric("Poor %", f"{poor_count/len(data)*100:.1f}%")
    col4.metric("Max Temp (°C)", f"{data['Temperature ( C )'].max():.1f}")

    # Export CSV
    st.subheader("💾 Export Full Report")
    csv_data = data.to_csv(index=False)
    st.download_button(
        label="📥 Download Full Report (CSV)",
        data=csv_data,
        file_name=f"transformer_health_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()
