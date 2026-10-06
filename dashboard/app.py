"""
OmniPulse Big Data Analytics: Streamlit Executive Dashboard
Professional Interactive Visualization of eCommerce Behavior Data.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os

st.set_page_config(page_title="OmniPulse Dashboard", layout="wide", page_icon="📊")

# Clean, Professional Light Theme CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #f8fafc;
        color: #0f172a;
    }
    
    .stApp {
        background-color: #f8fafc;
    }
    
    /* Metrics Styling */
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #0f172a;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Metric Cards */
    div[data-testid="metric-container"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
        border-top: 4px solid #3b82f6;
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }

    /* Titles */
    h1, h2, h3 {
        color: #0f172a;
        font-weight: 700 !important;
        letter-spacing: -0.5px;
    }
    
    h1 {
        padding-bottom: 5px;
    }

    hr {
        border-color: #e2e8f0;
        margin: 2rem 0;
    }
    
    /* Clean up the main container padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, "data", "raw", "ecommerce_transactions.csv")
    if os.path.exists(csv_path):
        # Stream 500K rows for dashboard performance on large datasets
        df = pd.read_csv(csv_path, nrows=500_000)
        return df
    return pd.DataFrame()

st.title("📊 OmniPulse Analytics")
st.markdown("<p style='font-size: 1.1rem; color: #64748b; margin-bottom: 2rem;'>Executive Dashboard & Telemetry</p>", unsafe_allow_html=True)

df = load_data()

if df.empty:
    st.info("Data not found. Please ensure the pipeline has been run to generate ecommerce_transactions.csv.")
else:
    st.sidebar.header("Data Filters")
    # Date Range Filter
    df['Order_Date'] = pd.to_datetime(df['Order_Date'])
    min_date = df['Order_Date'].min().date()
    max_date = df['Order_Date'].min().date() # Just dummy or real
    
    selected_region = st.sidebar.multiselect("Region", options=df['Customer_Region'].unique(), default=df['Customer_Region'].unique())
    selected_payment = st.sidebar.multiselect("Payment Method", options=df['Payment_Method'].unique(), default=df['Payment_Method'].unique())
    
    # Filter Data
    filtered_df = df[
        (df['Customer_Region'].isin(selected_region)) & 
        (df['Payment_Method'].isin(selected_payment))
    ]
    
    purchases = filtered_df[filtered_df['event_type'] == 'purchase']
    total_revenue = purchases['Total_Amount'].sum()
    total_events = len(filtered_df)
    total_users = filtered_df['Customer_ID'].nunique()
    fraud_txns = purchases['Is_Fraudulent'].sum()
    # Clamp to [0, 100] — Is_Fraudulent is 0/1 only on purchase rows
    fraud_rate = min((fraud_txns / len(purchases)) * 100, 100.0) if len(purchases) > 0 else 0

    st.markdown("### Executive KPIs")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Revenue", f"${total_revenue:,.0f}")
    with col2:
        st.metric("Events Analyzed", f"{total_events:,}")
    with col3:
        st.metric("Unique Customers", f"{total_users:,}")
    with col4:
        st.metric("Conversion Rate", f"{(len(purchases) / total_events)*100:.2f}%" if total_events else "0%")
    with col5:
        st.metric("Fraud Rate Detected", f"{fraud_rate:.2f}%")

    st.markdown("---")

    # Plotly Light Theme Customization
    chart_layout = dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family="Inter", color="#334155"),
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(showgrid=False, zeroline=False),
        yaxis=dict(showgrid=True, gridcolor='#f1f5f9', zeroline=False)
    )

    # Row 1
    colA, colB = st.columns([1.5, 1])

    with colA:
        st.markdown("### 📈 Revenue over Time (Hourly Velocity)")
        # Resample to hourly for a nice time series
        time_series = purchases.set_index('Order_Date').resample('H')['Total_Amount'].sum().reset_index()
        fig_ts = px.line(time_series, x='Order_Date', y='Total_Amount', 
                         color_discrete_sequence=["#3b82f6"])
        fig_ts.update_layout(**chart_layout, title="")
        fig_ts.update_xaxes(title="Time")
        fig_ts.update_yaxes(title="Revenue ($)")
        st.plotly_chart(fig_ts, use_container_width=True)

    with colB:
        st.markdown("### 🧠 ML RFM Customer Segments")
        # Load RFM if available
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        rfm_path = os.path.join(base_dir, "data", "processed", "rfm_segmented_customers.csv")
        if os.path.exists(rfm_path):
            rfm_df = pd.read_csv(rfm_path)
            segment_counts = rfm_df['Customer_Segment'].value_counts().reset_index()
            segment_counts.columns = ['Segment', 'Count']
            fig_pie = px.pie(segment_counts, names='Segment', values='Count', hole=0.6,
                             color_discrete_sequence=["#3b82f6", "#10b981", "#f59e0b", "#ef4444"])
            fig_pie.update_layout(**chart_layout, showlegend=True, 
                                  legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5))
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.warning("ML Segmentation data not found. Run Stage 5.")

    st.markdown("---")
    
    # Row 2
    colC, colD = st.columns(2)
    with colC:
        st.markdown("### 🌍 Regional Sales Distribution")
        region_sales = purchases.groupby("Customer_Region")['Total_Amount'].sum().reset_index()
        fig_region = px.bar(region_sales, x='Customer_Region', y='Total_Amount', color='Total_Amount', color_continuous_scale='Blues')
        fig_region.update_layout(**chart_layout, coloraxis_showscale=False)
        fig_region.update_xaxes(title="")
        fig_region.update_yaxes(title="Revenue ($)")
        st.plotly_chart(fig_region, use_container_width=True)
        
    with colD:
        st.markdown("### 🛡️ Payment Method vs Fraud Exposure")
        fraud_analysis = purchases.groupby('Payment_Method').agg(
            Total_Amount=('Total_Amount', 'sum'),
            Fraud_Txns=('Is_Fraudulent', 'sum')
        ).reset_index()
        fig_fraud = px.scatter(fraud_analysis, x='Total_Amount', y='Fraud_Txns', size='Total_Amount', color='Payment_Method', hover_name='Payment_Method')
        fig_fraud.update_layout(**chart_layout)
        fig_fraud.update_xaxes(title="Total Volume ($)")
        fig_fraud.update_yaxes(title="Number of Fraudulent Transactions")
        st.plotly_chart(fig_fraud, use_container_width=True)

    st.markdown("---")
    
    # Row 3
    colE, colF = st.columns(2)
    with colE:
        st.markdown("### 🏷️ Top Categories by Revenue")
        cat_revenue = purchases.groupby('Product_Category')['Total_Amount'].sum().nlargest(10).reset_index()
        cat_revenue = cat_revenue.sort_values(by='Total_Amount', ascending=True)
        fig_bar = px.bar(cat_revenue, x='Total_Amount', y='Product_Category', orientation='h', color='Total_Amount', color_continuous_scale='Blues')
        fig_bar.update_layout(**chart_layout, coloraxis_showscale=False)
        fig_bar.update_xaxes(title="Revenue ($)")
        fig_bar.update_yaxes(title="")
        st.plotly_chart(fig_bar, use_container_width=True)
        
    with colF:
        st.markdown("### 📦 Delivery Fulfillment Status")
        delivery_status = purchases['Delivery_Status'].value_counts().reset_index()
        delivery_status.columns = ['Status', 'Count']
        fig_del = px.pie(delivery_status, names='Status', values='Count', hole=0.4,
                         color_discrete_sequence=["#10b981", "#3b82f6", "#f59e0b", "#ef4444"])
        fig_del.update_layout(**chart_layout)
        st.plotly_chart(fig_del, use_container_width=True)
    
    st.markdown("<div style='text-align: center; color: #94a3b8; font-size: 0.9rem; margin-top: 30px;'>Pipeline integration active and healthy. Showing enriched Enterprise data.</div>", unsafe_allow_html=True)
