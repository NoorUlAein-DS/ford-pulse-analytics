
import streamlit as st
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Page Configuration & Modern Light Theme Styles
st.set_page_config(page_title="Fraud Sentry Dashboard", layout="wide")

st.markdown("""
    <style>
    /* Main App Background - Soft Clean Off-White */
    .stApp { 
        background: radial-gradient(circle, #FCFBF9 0%, #F4F1EA 100%);
        color: #2D0202; 
    }
    
    /* Elegant Title */
    h1 { 
        color: #8B0000 !important; 
        font-family: 'Georgia', serif; 
        font-size: 42px; 
        text-align: center; 
        font-weight: 700;
        margin-bottom: 5px;
    }
    
    /* Input Labels */
    label, .stMarkdown p { 
        color: #4A4A4A !important; 
        font-size: 15px; 
        font-weight: 600;
    }
    
    /* Clean Minimal Input Boxes */
    .stNumberInput input, .stSelectbox div, .stSlider div { 
        background-color: #FAF9F6 !important; 
        color: #2D0202 !important; 
        border: 1px solid rgba(139, 0, 0, 0.2) !important; 
        border-radius: 8px !important;
    }
    
    /* Velvet Red Premium Action Button */
    .stButton>button { 
        background: linear-gradient(135deg, #A31D1D 0%, #6E0D0D 100%); 
        color: #FFFFFF !important; 
        border: none; 
        border-radius: 8px; 
        width: 100%; 
        height: 52px; 
        font-size: 18px; 
        font-weight: bold;
        box-shadow: 0 4px 15px rgba(163, 29, 29, 0.15);
        transition: all 0.3s ease;
    }
    .stButton>button:hover { 
        transform: translateY(-2px); 
        box-shadow: 0 6px 20px rgba(163, 29, 29, 0.3);
    }
    
    /* Soft Colored Result Alerts */
    .result-box-safe { 
        background-color: #E8F8F5;
        border: 2px solid #2ECC71; 
        padding: 25px; 
        border-radius: 12px; 
        text-align: center; 
    }
    .result-box-fraud { 
        background-color: #FFDAC1;
        border: 2px solid #E74C3C; 
        padding: 25px; 
        border-radius: 12px; 
        text-align: center; 
    }
    </style>
""", unsafe_allow_html=True)

# Top Title Header
st.markdown("<h1>FRAUD SENTRY DASHBOARD</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #6E0D0D; font-size: 18px; margin-bottom: 25px; font-weight: 600; font-family: Georgia, serif;'>Developed by Aein</p>", unsafe_allow_html=True)

# 2. Safe Asset Loader
@st.cache_resource
def load_assets():
    with open('knn_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return model, scaler

try:
    model, scaler = load_assets()
except Exception as e:
    st.error("⚠️ System Notice: 'knn_model.pkl' or 'scaler.pkl' not found. Please ensure both files are uploaded to your GitHub repository root.")

# 3. Secure & Clean Columns Layout (No HTML bugs!)
col1, col2 = st.columns([1.1, 1], gap="large")

with col1:
    # Streamlit ka apna safe container building blocks use kar rahe hain
    with st.container(border=True):
        st.markdown("<h3 style='color: #A31D1D; margin-top:0; font-family: Georgia, serif;'>Transaction Metrics</h3>", unsafe_allow_html=True)
        
        amount = st.number_input("Transaction Amount ($)", min_value=0.0, value=150.0)
        account_age_days = st.number_input("Account Age (Days)", min_value=0, value=365)
        shipping_distance_km = st.number_input("Shipping Distance (KM)", min_value=0.0, value=45.0)
        avg_amount_user = st.number_input("User Average Historical Amount ($)", min_value=0.0, value=120.0)
        total_transactions_user = st.number_input("Total Historical Transactions", min_value=0, value=25)
        transaction_hour = st.slider("Hour of Transaction (0-23)", 0, 23, 14)
        
        avs_match = st.selectbox("AVS Match Status (Address Verified)", [1, 0])
        cvv_result = st.selectbox("CVV Verification Result", [1, 0])
        three_ds_flag = st.selectbox("3D Secure Dynamic Flag", [1, 0])
        promo_used = st.selectbox("Promo Code Applied", [0, 1])
        
        input_data = np.zeros(33)
        input_data[0] = account_age_days
        input_data[1] = total_transactions_user
        input_data[2] = avg_amount_user
        input_data[3] = amount
        input_data[4] = promo_used
        input_data[5] = avs_match
        input_data[6] = cvv_result
        input_data[7] = three_ds_flag
        input_data[8] = shipping_distance_km
        input_data[9] = transaction_hour

        st.write("")
        predict_btn = st.button("Run Risk Evaluation")

with col2:
    with st.container(border=True):
        st.markdown("<h3 style='color: #A31D1D; margin-top:0; font-family: Georgia, serif;'>Operational Integrity Report</h3>", unsafe_allow_html=True)
        
        if predict_btn:
            try:
                scaled_input = scaler.transform([input_data])
                prediction = model.predict(scaled_input)[0]
                prediction_proba = model.predict_proba(scaled_input)[0]
                
                if prediction == 1:
                    st.markdown(f"""
                        <div class='result-box-fraud'>
                            <h2 style='color: #E74C3C; margin: 0; font-size: 24px;'>TRANSACTION BLOCKED</h2>
                            <p style='color: #2D0202; font-size: 16px; margin-top: 10px;'>High Probability Fraud Pattern Detected</p>
                            <h4 style='color: #A31D1D; margin: 5px;'>Confidence: {prediction_proba[1]*100:.1f}%</h4>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                        <div class='result-box-safe'>
                            <h2 style='color: #2ECC71; margin: 0; font-size: 24px;'>TRANSACTION CLEAN</h2>
                            <p style='color: #2D0202; font-size: 16px; margin-top: 10px;'>Authorized and Cleared for Settlement</p>
                            <h4 style='color: #A31D1D; margin: 5px;'>Confidence: {prediction_proba[0]*100:.1f}%</h4>
                        </div>
                    """, unsafe_allow_html=True)
            except NameError:
                st.warning("Cannot run evaluation because model/scaler assets are not loaded.")
        else:
            st.info("System Ready. Awaiting parameter submission from the left panel.")
            
    with st.container(border=True):
        st.markdown("<h3 style='color: #A31D1D; margin-top:0; font-family: Georgia, serif;'>Core Decision Drivers</h3>", unsafe_allow_html=True)
        st.caption("Operational feature impact weights computed by Mutual Information ranking:")
        
        # Light elegant clean chart
        fig, ax = plt.subplots(figsize=(6, 3.4))
        fig.patch.set_facecolor('#FFFFFF')
        ax.set_facecolor('#FAF9F6')
        
        features = ['Transaction Hour', 'Shipping Distance', 'Avg Amount', 'Account Age', 'Amount']
        importance = [0.008, 0.024, 0.026, 0.029, 0.033]
        
        ax.barh(features, importance, color='#A31D1D', edgecolor='#6E0D0D', height=0.55)
        ax.tick_params(colors='#2D0202', labelsize=10)
        ax.xaxis.grid(True, linestyle='--', alpha=0.3, color='#B0B0B0')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#6E0D0D')
        ax.spines['bottom'].set_color('#6E0D0D')
        
        st.pyplot(fig)
