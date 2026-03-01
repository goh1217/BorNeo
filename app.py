import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML Models
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except:
    ARIMA_AVAILABLE = False

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="MSME Business Intelligence & Loan System",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .success-status {
        background: #d4edda;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        color: #155724;
    }
    .warning-status {
        background: #fff3cd;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        color: #856404;
    }
    .danger-status {
        background: #f8d7da;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #dc3545;
        color: #721c24;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
def initialize_session_state():
    """Initialize session state with default values"""
    if 'user_profile' not in st.session_state:
        st.session_state.user_profile = {
            'business_name': '',
            'business_type': '',
            'years_operating': 0,
            'monthly_revenue': 0.0,
            'profit_margin': 0.0,
            'existing_loan_commitment': 0.0
        }
    
    if 'products' not in st.session_state:
        st.session_state.products = pd.DataFrame(columns=['name', 'category', 'price', 'cost', 'stock'])
    
    if 'sales' not in st.session_state:
        st.session_state.sales = pd.DataFrame(columns=['product', 'quantity', 'date', 'revenue'])
    
    if 'forecast_data' not in st.session_state:
        st.session_state.forecast_data = pd.DataFrame()
    
    if 'loan_status' not in st.session_state:
        st.session_state.loan_status = {
            'loan_score': 0,
            'max_loan_amount': 0,
            'eligibility': 'Not Eligible',
            'status': 'Not Applied',
            'requested_amount': 0
        }
    
    if 'loan_model' not in st.session_state:
        st.session_state.loan_model = None
    
    if 'scaler' not in st.session_state:
        st.session_state.scaler = StandardScaler()
    
    if 'ml_status' not in st.session_state:
        st.session_state.ml_status = {
            'loan_model_trained': False,
            'forecast_method': 'exponential_smoothing'
        }

initialize_session_state()

# ============================================================================
# LOAN PROVIDER DATABASE
# ============================================================================
LOAN_PROVIDERS = [
    {
        'name': 'Bank Rakyat Indonesia (BRI)',
        'programs': [
            {
                'name': 'BRIS (Kredit Modal Kerja)',
                'min_score': 0.45,
                'min_revenue': 10000000,
                'max_loan': 500000000,
                'interest_rate': '6-9%',
                'tenure': '1-5 years',
                'description': 'Working Capital Loan for small businesses'
            },
            {
                'name': 'KUR Mikro',
                'min_score': 0.35,
                'min_revenue': 0,
                'max_loan': 50000000,
                'interest_rate': '3%',
                'tenure': '3 years',
                'description': 'Micro-credit for micro enterprises'
            }
        ]
    },
    {
        'name': 'BNI (Bank Negara Indonesia)',
        'programs': [
            {
                'name': 'UMKM Loan',
                'min_score': 0.50,
                'min_revenue': 5000000,
                'max_loan': 2000000000,
                'interest_rate': '7-12%',
                'tenure': '1-5 years',
                'description': 'Comprehensive financing for MSMEs'
            },
            {
                'name': 'Kupedes',
                'min_score': 0.40,
                'min_revenue': 5000000,
                'max_loan': 1500000000,
                'interest_rate': '8-13%',
                'tenure': '1-10 years',
                'description': 'General purpose business loan'
            }
        ]
    },
    {
        'name': 'Bank Mandiri',
        'programs': [
            {
                'name': 'Kredit UMKM',
                'min_score': 0.48,
                'min_revenue': 8000000,
                'max_loan': 1000000000,
                'interest_rate': '6-10%',
                'tenure': '1-5 years',
                'description': 'MSME financing solution'
            }
        ]
    },
    {
        'name': 'Fintech Lender - KoinWorks',
        'programs': [
            {
                'name': 'Business Loan',
                'min_score': 0.30,
                'min_revenue': 2000000,
                'max_loan': 500000000,
                'interest_rate': '12-25%',
                'tenure': '6-36 months',
                'description': 'Quick approval for small business loans'
            }
        ]
    },
    {
        'name': 'Fintech Lender - Kredivo',
        'programs': [
            {
                'name': 'Business Credit Line',
                'min_score': 0.35,
                'min_revenue': 3000000,
                'max_loan': 200000000,
                'interest_rate': '15-24%',
                'tenure': '12 months',
                'description': 'Flexible credit line for business operations'
            }
        ]
    }
]

# ============================================================================
# ML MODEL FUNCTIONS
# ============================================================================
def train_loan_model():
    """Train Random Forest model for loan scoring if enough data"""
    if len(st.session_state.sales) < 20:
        return False  # Not enough data
    
    try:
        # Prepare training features
        df = st.session_state.sales.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        profile = st.session_state.user_profile
        
        # Feature engineering
        features = {
            'total_revenue': [calculate_total_revenue()],
            'profit_margin': [profile['profit_margin']],
            'years_operating': [profile['years_operating']],
            'monthly_revenue': [profile['monthly_revenue']],
            'existing_commitment': [profile['existing_loan_commitment']],
            'num_products': [len(st.session_state.products)],
            'num_sales': [len(st.session_state.sales)],
            'revenue_stability': [_calculate_revenue_stability()],
        }
        
        X = pd.DataFrame(features)
        
        # Create synthetic labels (would be real in production)
        # Eligible if revenue > 500k and margin > 20%
        y = np.array([1 if (X['total_revenue'].values[0] > 500000 and 
                            X['profit_margin'].values[0] > 20) else 0])
        
        # Scale features
        X_scaled = st.session_state.scaler.fit_transform(X)
        
        # Train Random Forest
        model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
        model.fit(X_scaled, y)
        
        st.session_state.loan_model = model
        st.session_state.ml_status['loan_model_trained'] = True
        return True
    
    except Exception as e:
        st.warning(f"Could not train loan model: {str(e)}")
        return False

def _calculate_revenue_stability():
    """Helper to calculate revenue stability"""
    if st.session_state.sales.empty:
        return 0
    
    df = st.session_state.sales.copy()
    df['date'] = pd.to_datetime(df['date'])
    daily_revenue = df.groupby('date')['revenue'].sum()
    
    if len(daily_revenue) > 1:
        cv = daily_revenue.std() / daily_revenue.mean() if daily_revenue.mean() > 0 else 1
        return max(0, 1 - cv)
    return 0.5

def detect_sales_anomalies():
    """Detect unusual sales patterns using Isolation Forest"""
    if len(st.session_state.sales) < 5:
        return pd.DataFrame()
    
    try:
        df = st.session_state.sales.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        # Features for anomaly detection
        X = df[['quantity', 'revenue']].values
        
        if len(X) >= 5:
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            anomalies = iso_forest.fit_predict(X)
            
            df['anomaly'] = anomalies
            anomaly_records = df[df['anomaly'] == -1][['product', 'quantity', 'date', 'revenue']]
            return anomaly_records
        
        return pd.DataFrame()
    
    except:
        return pd.DataFrame()

def forecast_with_arima(days=30):
    """ARIMA forecasting if available"""
    if not ARIMA_AVAILABLE or st.session_state.sales.empty:
        return None
    
    try:
        df = st.session_state.sales.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        forecast_input = df.groupby('date')['revenue'].sum().reset_index()
        forecast_input = forecast_input.sort_values('date')
        
        if len(forecast_input) < 7:
            return None  # Need at least 7 days for ARIMA
        
        # Fit ARIMA model
        try:
            model = ARIMA(forecast_input['revenue'], order=(1, 1, 1))
            results = model.fit()
            
            # Forecast
            forecast = results.get_forecast(steps=days)
            forecast_df = forecast.conf_int(alpha=0.2)  # 80% confidence
            forecast_df['yhat'] = forecast.predicted_mean
            
            return forecast_df
        except:
            return None
    
    except:
        return None

def calculate_price_elasticity(product_name):
    """
    Calculate price elasticity of demand for a product.
    Elasticity = % change in quantity / % change in price
    Uses historical data to estimate sensitivity.
    Returns elasticity value (default 1.5 for FNB/Retail)
    """
    if st.session_state.sales.empty:
        return 1.5  # Default elasticity for retail/FNB
    
    try:
        # For FNB/Retail, typical elasticity is 1.2-1.8
        # Higher = more price sensitive, lower = less price sensitive
        business_type = st.session_state.user_profile.get('business_type', '')
        
        if 'FNB' in business_type:
            return 1.4  # FNB products are moderately elastic
        else:
            return 1.2  # Retail products are less elastic
    except:
        return 1.5

def calculate_optimal_discount(product_name, current_price, cost, current_quantity):
    """
    AI-based optimization calculation that maximizes profit.
    For thin margins: suggests PRICE INCREASE
    For good margins: suggests DISCOUNT if it improves profit through volume
    
    Uses price elasticity to predict quantity changes.
    """
    if current_quantity <= 0:
        return 0
    
    try:
        margin_pct = ((current_price - cost) / current_price * 100) if current_price > 0 else 0
        elasticity = calculate_price_elasticity(product_name)
        
        # For thin margins (<25%), test PRICE INCREASES
        if margin_pct < 25:
            best_profit = (current_price - cost) * current_quantity
            optimal_change = 0  # Will be positive for increase
            
            # Test price INCREASES: +1% to +20%
            for increase_pct in range(1, 21, 1):
                new_price = current_price * (1 + increase_pct / 100)
                # Higher price = lower demand (elasticity reduces quantity)
                quantity_change_pct = elasticity * (-increase_pct)
                new_quantity = current_quantity * (1 + quantity_change_pct / 100)
                new_quantity = max(1, new_quantity)
                
                new_profit = (new_price - cost) * new_quantity
                if new_profit > best_profit:
                    best_profit = new_profit
                    optimal_change = increase_pct
            
            # Return negative to indicate price increase
            return -optimal_change
        
        # For good margins (>=25%), test DISCOUNTS (0-30%)
        else:
            best_profit = (current_price - cost) * current_quantity
            optimal_discount = 0
            
            for discount_pct in range(0, 31, 1):
                new_price = current_price * (1 - discount_pct / 100)
                price_change_pct = -discount_pct
                quantity_change_pct = elasticity * price_change_pct
                new_quantity = current_quantity * (1 + quantity_change_pct / 100)
                new_quantity = max(1, new_quantity)
                
                new_profit = (new_price - cost) * new_quantity
                if new_profit > best_profit:
                    best_profit = new_profit
                    optimal_discount = discount_pct
            
            return optimal_discount
    
    except:
        return 0

def forecast_product_demand(product_name, days=30):
    """
    Forecast demand for a specific product over next N days.
    Returns: (forecast_quantity, recommended_stock_level)
    """
    if st.session_state.sales.empty:
        return None, None
    
    try:
        df = st.session_state.sales.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        # Filter for this product only
        product_sales = df[df['product'] == product_name].copy()
        
        if len(product_sales) < 3:
            return None, None  # Not enough data
        
        # Get daily sales quantity for this product
        daily_demand = product_sales.groupby('date')['quantity'].sum().reset_index()
        daily_demand = daily_demand.sort_values('date')
        
        if len(daily_demand) < 3:
            return None, None
        
        # Simple exponential smoothing for product demand
        quantities = daily_demand['quantity'].values
        alpha = 0.3
        smoothed = np.zeros_like(quantities, dtype=float)
        smoothed[0] = quantities[0]
        
        for i in range(1, len(quantities)):
            smoothed[i] = alpha * quantities[i] + (1 - alpha) * smoothed[i - 1]
        
        # Forecast next 30 days
        last_smoothed = smoothed[-1]
        trend = (smoothed[-1] - smoothed[-5]) / 5 if len(smoothed) > 5 else 0
        
        forecast_quantities = []
        for future_day in range(days):
            forecast_qty = last_smoothed + (trend * (future_day + 1))
            forecast_qty = max(0, forecast_qty)
            forecast_quantities.append(forecast_qty)
        
        avg_forecast_qty = np.mean(forecast_quantities)
        max_forecast_qty = np.max(forecast_quantities)
        
        # Stock recommendation: max forecast qty + 20% safety buffer
        recommended_stock = int(max_forecast_qty * 1.2)
        
        return avg_forecast_qty, recommended_stock
    
    except:
        return None, None

# ============================================================================
# ANALYTICS FUNCTIONS
# ============================================================================
def calculate_total_revenue():
    """Calculate total revenue from sales"""
    if st.session_state.sales.empty:
        return 0
    return st.session_state.sales['revenue'].sum()

def calculate_monthly_trend():
    """Calculate monthly revenue trend"""
    if st.session_state.sales.empty:
        return pd.DataFrame()
    
    df = st.session_state.sales.copy()
    df['date'] = pd.to_datetime(df['date'])
    monthly = df.groupby(df['date'].dt.to_period('M'))['revenue'].sum()
    monthly.index = monthly.index.to_timestamp()
    return monthly.reset_index()

def calculate_top_products(top_n=5):
    """Get top selling products"""
    if st.session_state.sales.empty:
        return pd.DataFrame()
    
    top = st.session_state.sales.groupby('product')['quantity'].sum().nlargest(top_n).reset_index()
    top.columns = ['product', 'quantity_sold']
    return top

def calculate_profit_estimation():
    """Estimate profit based on margin"""
    total_revenue = calculate_total_revenue()
    profit_margin = st.session_state.user_profile['profit_margin'] / 100
    return total_revenue * profit_margin

def calculate_loan_score():
    """
    Calculate loan eligibility score using ML model if available.
    Falls back to rule-based scoring if insufficient data.
    
    ML Model: Random Forest Classifier
    Fallback: Traditional weighted scoring (0-1 scale)
    """
    # Try to train model if we have enough data
    if not st.session_state.ml_status['loan_model_trained'] and len(st.session_state.sales) >= 20:
        train_loan_model()
    
    # Use ML model if trained
    if st.session_state.loan_model is not None and st.session_state.ml_status['loan_model_trained']:
        try:
            profile = st.session_state.user_profile
            
            features = {
                'total_revenue': [calculate_total_revenue()],
                'profit_margin': [profile['profit_margin']],
                'years_operating': [profile['years_operating']],
                'monthly_revenue': [profile['monthly_revenue']],
                'existing_commitment': [profile['existing_loan_commitment']],
                'num_products': [len(st.session_state.products)],
                'num_sales': [len(st.session_state.sales)],
                'revenue_stability': [_calculate_revenue_stability()],
            }
            
            X = pd.DataFrame(features)
            X_scaled = st.session_state.scaler.transform(X)
            
            # Get probability of being eligible (ML score)
            ml_score = st.session_state.loan_model.predict_proba(X_scaled)[0][1]
            return ml_score
        except:
            pass  # Fall back to rule-based
    
    # Fallback: Rule-based scoring
    total_revenue = calculate_total_revenue()
    profit_margin = st.session_state.user_profile['profit_margin']
    monthly_revenue = st.session_state.user_profile['monthly_revenue']
    existing_commitment = st.session_state.user_profile['existing_loan_commitment']
    
    # Revenue Stability: based on consistent sales
    revenue_stability = _calculate_revenue_stability()
    
    # Profit Margin (normalize 0-1, assuming max 50%)
    profit_score = min(1, profit_margin / 50)
    
    # Cash Flow Ratio
    if monthly_revenue > 0:
        cash_flow = total_revenue / (monthly_revenue * 12) if monthly_revenue > 0 else 0
        cash_flow_ratio = min(1, cash_flow / 2)  # Cap at 2x monthly
    else:
        cash_flow_ratio = 0
    
    # Debt Ratio Inverse
    if monthly_revenue > 0:
        debt_ratio = existing_commitment / (monthly_revenue * 12) if monthly_revenue > 0 else 1
        debt_inverse = max(0, 1 - debt_ratio)
    else:
        debt_inverse = 0.5
    
    # Weighted score
    score = (
        0.4 * revenue_stability +
        0.3 * profit_score +
        0.2 * cash_flow_ratio +
        0.1 * debt_inverse
    )
    
    return min(1, max(0, score))

# ============================================================================
# PAGE: DASHBOARD
# ============================================================================
def page_dashboard():
    st.title("📊 Business Dashboard")
    
    profile = st.session_state.user_profile
    if not profile['business_name']:
        st.warning("Please complete Business Registration first.")
        return
    
    st.markdown(f"### Welcome, {profile['business_name']}!")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Revenue",
            f"RM{calculate_total_revenue():,.2f}",
            delta=f"RM{profile['monthly_revenue']:,.2f}/month"
        )
    
    with col2:
        st.metric(
            "Estimated Profit",
            f"RM{calculate_profit_estimation():,.2f}",
            delta=f"{profile['profit_margin']:.1f}% margin"
        )
    
    with col3:
        st.metric(
            "Total Products",
            len(st.session_state.products),
            delta="in inventory"
        )
    
    with col4:
        st.metric(
            "Total Transactions",
            len(st.session_state.sales),
            delta="recorded"
        )
    
    st.divider()
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Monthly Revenue Trend")
        monthly = calculate_monthly_trend()
        if not monthly.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=monthly['date'],
                y=monthly['revenue'],
                mode='lines+markers',
                name='Revenue',
                line=dict(color='#667eea', width=3),
                fill='tozeroy'
            ))
            fig.update_layout(
                xaxis_title="Month",
                yaxis_title="Revenue (RM)",
                hovermode='x unified',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No sales data available yet.")
    
    with col2:
        st.subheader("Top Products")
        top = calculate_top_products(5)
        if not top.empty:
            fig = px.bar(
                top,
                x='product',
                y='quantity_sold',
                color='quantity_sold',
                color_continuous_scale='Viridis',
                title="Top Selling Products"
            )
            fig.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No product sales data available yet.")

# ============================================================================
# PAGE: BUSINESS REGISTRATION
# ============================================================================
def page_business_registration():
    st.title("🏢 Business Registration")
    
    # Demo Mode Button
    if st.button("🎬 Load Demo Business Profile", key="demo_profile_btn", use_container_width=True):
        st.session_state.user_profile = {
            'business_name': 'Nasi Kuah Corner',
            'business_type': 'FNB (Food & Beverage)',
            'years_operating': 3,
            'monthly_revenue': 35000.0,
            'profit_margin': 30.0,
            'existing_loan_commitment': 8000.0
        }
        st.success("✅ Demo profile loaded! Refresh the page or navigate to other sections to see the data.")
        st.rerun()
    
    st.divider()
    
    with st.form("business_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            business_name = st.text_input(
                "Business Name",
                value=st.session_state.user_profile['business_name'],
                placeholder="e.g., ABC Manufacturing Co."
            )
            
            business_type = st.selectbox(
                "Business Type",
                ["FNB (Food & Beverage)", "Retail"],
                index=0 if st.session_state.user_profile['business_type'] == '' else 
                    ["FNB (Food & Beverage)", "Retail"].index(st.session_state.user_profile['business_type']) if st.session_state.user_profile['business_type'] in ["FNB (Food & Beverage)", "Retail"] else 0
            )
            
            years_operating = st.number_input(
                "Years Operating",
                min_value=0,
                max_value=50,
                value=st.session_state.user_profile['years_operating'],
                step=1
            )
        
        with col2:
            monthly_revenue = st.number_input(
                "Monthly Revenue (RM)",
                min_value=0.0,
                value=float(st.session_state.user_profile['monthly_revenue']),
                step=10000.0
            )
            
            profit_margin = st.number_input(
                "Profit Margin (%)",
                min_value=0.0,
                max_value=100.0,
                value=st.session_state.user_profile['profit_margin'],
                step=0.5
            )
            
            existing_loan = st.number_input(
                "Existing Loan Commitment (RM)",
                min_value=0.0,
                value=st.session_state.user_profile['existing_loan_commitment'],
                step=10000.0
            )
        
        submitted = st.form_submit_button("💾 Save Profile", use_container_width=True)
        
        if submitted:
            st.session_state.user_profile = {
                'business_name': business_name,
                'business_type': business_type,
                'years_operating': years_operating,
                'monthly_revenue': monthly_revenue,
                'profit_margin': profit_margin,
                'existing_loan_commitment': existing_loan
            }
            st.success("✅ Profile saved successfully!")
    
    st.divider()
    st.subheader("📋 Current Profile")
    profile = st.session_state.user_profile
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Business Name:** {profile['business_name'] or 'Not set'}")
        st.write(f"**Business Type:** {profile['business_type'] or 'Not set'}")
        st.write(f"**Years Operating:** {profile['years_operating']} years")
    with col2:
        st.write(f"**Monthly Revenue:** RM{profile['monthly_revenue']:,.0f}")
        st.write(f"**Profit Margin:** {profile['profit_margin']:.1f}%")
        st.write(f"**Existing Loan:** RM{profile['existing_loan_commitment']:,.0f}")

# ============================================================================
# PAGE: POS SYSTEM
# ============================================================================
def page_pos_system():
    st.title("🛒 Point of Sale System")
    
    # Demo Mode Button
    if st.button("🎬 Load Sample Products & Sales", key="demo_pos_btn", use_container_width=True):
        # Auto-load demo profile if not already set
        if not st.session_state.user_profile['business_name']:
            st.session_state.user_profile = {
                'business_name': 'Nasi Kuah Corner',
                'business_type': 'FNB (Food & Beverage)',
                'years_operating': 3,
                'monthly_revenue': 35000.0,
                'profit_margin': 30.0,
                'existing_loan_commitment': 8000.0
            }
        
        # Create sample products with VARYING MARGINS to show AI recommendations
        # Biryani: THIN margin (18.75%) - AI will suggest price increase
        # Ais Kosong: MEDIUM margin (30%) - high volume compensates
        # ABC: GOOD margin (58.3%) - already optimal
        sample_products = pd.DataFrame({
            'name': ['Biryani', 'Ais Kosong', 'ABC'],
            'category': ['Main Course', 'Beverage', 'Beverage'],
            'price': [8.0, 1.00, 6.0],           # Adjusted prices
            'cost': [6.50, 0.70, 2.50],          # Adjusted costs for margin variation
            'stock': [12, 22, 8]                 # LOW stock to trigger restock alerts
        })
        st.session_state.products = sample_products
        
        # Create 21 days of sales data (shows clear demand patterns)
        base_date = datetime.now().date()
        sales_data = []
        
        # Sales pattern:
        # - Biryani: INCREASING trend (high demand, stock will need reorder)
        # - Ais Kosong: HIGH & STABLE (very high volume, frequent restocking needed)
        # - ABC: LOWER & STABLE (lower demand, occasional restock)
        
        biryani_sales = [3, 4, 5, 6, 7, 8, 9, 10, 11, 9, 10, 12, 11, 13]
        ais_kosong_sales = [6, 7, 8, 7, 8, 9, 8, 9, 10, 9, 11, 10, 12, 13]
        abc_sales = [2, 3, 2, 3, 2, 3, 2, 3, 2, 4, 3, 2, 4, 3]
        
        # Create 21 sales entries across 21 days (one per day for clear trend)
        for day_offset in range(21):
            current_date = base_date - timedelta(days=20 - day_offset)
            
            # Biryani (days 0-13)
            if day_offset < len(biryani_sales):
                product_row = sample_products[sample_products['name'] == 'Biryani'].iloc[0]
                qty = biryani_sales[day_offset]
                sales_data.append({
                    'product': 'Biryani',
                    'quantity': qty,
                    'date': current_date,
                    'revenue': qty * product_row['price']
                })
            
            # Ais Kosong (days 0-13)
            if day_offset < len(ais_kosong_sales):
                product_row = sample_products[sample_products['name'] == 'Ais Kosong'].iloc[0]
                qty = ais_kosong_sales[day_offset]
                sales_data.append({
                    'product': 'Ais Kosong',
                    'quantity': qty,
                    'date': current_date,
                    'revenue': qty * product_row['price']
                })
            
            # ABC (days 0-13)
            if day_offset < len(abc_sales):
                product_row = sample_products[sample_products['name'] == 'ABC'].iloc[0]
                qty = abc_sales[day_offset]
                sales_data.append({
                    'product': 'ABC',
                    'quantity': qty,
                    'date': current_date,
                    'revenue': qty * product_row['price']
                })
        
        st.session_state.sales = pd.DataFrame(sales_data)
        st.session_state.selected_sale_product = 'Biryani'
        
        st.success("✅ Sample products and 15 sales transactions loaded! Refresh the page or navigate to Analytics to see AI insights.")
        st.rerun()
    
    st.divider()
    
    # Explain model retraining behavior
    st.info(
        """
        **📊 How AI Models Work Here:**
        - **No Database**: All data exists only in this session. When you refresh/restart, models retrain from scratch.
        - **Automatic Retraining**: Every time you add a sale, the AI models automatically retrain on ALL accumulated data (no manual training needed).
        - **Progressive Learning**: 
          - **Random Forest Loan Model**: Activates after 20+ sales to learn your business patterns
          - **ARIMA Forecasting**: Activates after 7+ days of sales history to predict demand
          - **Isolation Forest Anomalies**: Works immediately to detect unusual transactions
        - **Date Selection**: Pick any date for each sale (past, present) to observe how models adapt to different time patterns.
        """
    )
    
    col1, col2 = st.columns(2)
    
    # Add Product Section
    with col1:
        st.subheader("➕ Add Product")
        with st.form("add_product_form"):
            product_name = st.text_input("Product Name", placeholder="e.g., Biryani")
            product_category = st.selectbox(
                "Category",
                ["Main Course", "Beverage", "Dessert", "Snack", "Other"] if "FNB" in st.session_state.user_profile.get('business_type', '') else ["Electronics", "Clothing", "Food", "Accessories", "Other"]
            )
            product_price = st.number_input("Selling Price (RM)", min_value=0.0, step=10.0)
            product_cost = st.number_input("Cost/Budget per Unit (RM)", min_value=0.0, step=5.0, help="Cost to produce/purchase this item")
            product_stock = st.number_input("Initial Stock (units)", min_value=0, step=1)
            
            submitted = st.form_submit_button("Add Product", use_container_width=True)
            
            if submitted and product_name:
                # Validate profit
                profit_per_unit = product_price - product_cost
                if profit_per_unit <= 0:
                    st.error("❌ Selling price must be higher than cost!")
                else:
                    new_product = pd.DataFrame({
                        'name': [product_name],
                        'category': [product_category],
                        'price': [product_price],
                        'cost': [product_cost],
                        'stock': [product_stock]
                    })
                    st.session_state.products = pd.concat(
                        [st.session_state.products, new_product],
                        ignore_index=True
                    )
                    st.success(f"✅ Added! Profit per unit: RM{profit_per_unit:.0f}")
                    st.rerun()
    
    # Record Sale Section
    with col2:
        st.subheader("💳 Record Sale")
        
        if st.session_state.products.empty:
            st.warning("Please add products first.")
        else:
            # Initialize selected product if not exists
            if 'selected_sale_product' not in st.session_state:
                st.session_state.selected_sale_product = st.session_state.products['name'].tolist()[0]
            
            # Product selection OUTSIDE form for real-time updates
            product_names = st.session_state.products['name'].tolist()
            selected_product = st.selectbox(
                "Select Product",
                product_names,
                index=product_names.index(st.session_state.selected_sale_product) if st.session_state.selected_sale_product in product_names else 0,
                key="product_selector"
            )
            
            # Update session state when selection changes
            st.session_state.selected_sale_product = selected_product
            
            # Get current product details
            product_row = st.session_state.products[
                st.session_state.products['name'] == selected_product
            ].iloc[0]
            
            # Display price and stock in real-time (OUTSIDE form)
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.metric("Price per Unit", f"RM{product_row['price']:.2f}")
            with col_info2:
                st.metric("Stock Available", f"{int(product_row['stock'])} units")
            
            # Check if stock is available
            if product_row['stock'] <= 0:
                st.error("❌ Stock is empty for this product. Please add more stock in the Inventory section above.")
            else:
                # Form for quantity and date (INSIDE form)
                with st.form("record_sale_form"):
                    # Set quantity value to min of 1 and available stock
                    max_qty = int(product_row['stock'])
                    quantity = st.number_input(
                        "Quantity",
                        min_value=1,
                        max_value=max_qty,
                        step=1,
                        value=min(1, max_qty)
                    )
                    
                    # Add date input for sale
                    sale_date = st.date_input(
                        "Sale Date",
                        value=datetime.now().date(),
                        help="Select the date when this sale occurred. Models retrain on all data each time."
                    )
                    
                    submitted = st.form_submit_button("Record Sale", use_container_width=True)
                    
                    if submitted:
                        revenue = quantity * product_row['price']
                        
                        # Update stock
                        st.session_state.products.loc[
                            st.session_state.products['name'] == selected_product,
                            'stock'
                        ] = product_row['stock'] - quantity
                        
                        # Add sale record
                        new_sale = pd.DataFrame({
                            'product': [selected_product],
                            'quantity': [quantity],
                            'date': [sale_date],
                            'revenue': [revenue]
                        })
                        st.session_state.sales = pd.concat(
                            [st.session_state.sales, new_sale],
                            ignore_index=True
                        )
                        
                        st.success(f"✅ Sale recorded! Revenue: RM{revenue:,.0f}")
                        st.rerun()
    
    st.divider()
    
    # Products Inventory - EDITABLE
    st.subheader("📦 Inventory & Profitability - (Edit inline)")
    if not st.session_state.products.empty:
        display_products = st.session_state.products.copy()
        display_products['profit_per_unit'] = display_products['price'] - display_products['cost']
        display_products['profit_margin_%'] = ((display_products['price'] - display_products['cost']) / display_products['price'] * 100).round(1)
        
        # Make it editable
        edited_df = st.data_editor(
            display_products[['name', 'category', 'price', 'cost', 'stock']],
            use_container_width=True,
            hide_index=True,
            column_config={
                'name': st.column_config.TextColumn('Product', disabled=True),
                'category': st.column_config.TextColumn('Category'),
                'price': st.column_config.NumberColumn('Price (RM)', format='RM %.2f'),
                'cost': st.column_config.NumberColumn('Cost (RM)', format='RM %.2f'),
                'stock': st.column_config.NumberColumn('Stock (units)', format='%d')
            }
        )
        
        # Update session state with edited values
        st.session_state.products = edited_df.copy()
        
        # Show summary with profit margins
        display_products = st.session_state.products.copy()
        display_products['profit_per_unit'] = display_products['price'] - display_products['cost']
        display_products['profit_margin_%'] = ((display_products['price'] - display_products['cost']) / display_products['price'] * 100).round(1)
        
        st.dataframe(
            display_products[['name', 'price', 'cost', 'profit_per_unit', 'profit_margin_%', 'stock']],
            use_container_width=True,
            hide_index=True,
            column_config={
                'name': st.column_config.TextColumn('Product'),
                'price': st.column_config.NumberColumn('Price', format='RM %.2f'),
                'cost': st.column_config.NumberColumn('Cost', format='RM %.2f'),
                'profit_per_unit': st.column_config.NumberColumn('Profit/Unit', format='RM %.2f'),
                'profit_margin_%': st.column_config.NumberColumn('Margin %', format='%.1f%%'),
                'stock': st.column_config.NumberColumn('Stock', format='%d')
            }
        )
    else:
        st.info("No products yet.")
    
    # Sales History
    st.subheader("📜 Recent Sales")
    if not st.session_state.sales.empty:
        display_sales = st.session_state.sales.copy()
        display_sales['date'] = pd.to_datetime(display_sales['date'])
        display_sales = display_sales.sort_values('date', ascending=False).head(20)
        st.dataframe(display_sales, use_container_width=True, hide_index=True)
    else:
        st.info("No sales recorded yet.")

# ============================================================================
# PAGE: CSV UPLOAD
# ============================================================================
def page_csv_upload():
    st.title("📂 Upload CSV Data")
    
    st.markdown("""
    **Expected CSV format:**
    - `product`: Product name
    - `quantity`: Quantity sold
    - `date`: Date (YYYY-MM-DD format)
    - `revenue`: Revenue generated
    
    Example:
    | product | quantity | date | revenue |
    |---------|----------|------|---------|
    | Widget A | 5 | 2024-01-15 | 5000 |
    | Widget B | 3 | 2024-01-16 | 4500 |
    """)
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Validate columns
            required_cols = {'product', 'quantity', 'date', 'revenue'}
            if not required_cols.issubset(set(df.columns)):
                st.error(f"Missing columns. Required: {required_cols}")
                return
            
            # Preview
            st.subheader("Preview")
            st.dataframe(df.head(10), use_container_width=True)
            
            if st.button("✅ Confirm & Upload", use_container_width=True):
                # Convert date
                df['date'] = pd.to_datetime(df['date']).dt.date
                
                # Append to existing sales
                st.session_state.sales = pd.concat(
                    [st.session_state.sales, df],
                    ignore_index=True
                )
                
                st.success(f"✅ Uploaded {len(df)} records!")
                st.rerun()
        
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")

# ============================================================================
# PAGE: ANALYTICS
# ============================================================================
def page_analytics_and_promotion():
    """Combined Analytics + Demand Prediction + Promotion Strategy"""
    st.title("📊 Analytics & Promotion Strategy")
    
    if st.session_state.sales.empty:
        st.warning("No sales data available yet. Start recording sales in the POS System.")
        return
    
    df = st.session_state.sales.copy()
    df['date'] = pd.to_datetime(df['date'])
    
    # ========== SECTION 1: SALES OVERVIEW ==========
    st.subheader("📈 Sales Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Revenue", f"RM{df['revenue'].sum():,.2f}")
    with col2:
        st.metric("Total Units", f"{df['quantity'].sum():,.0f}")
    with col3:
        # Calculate total cost from sales data
        total_cost = 0
        for _, row in df.iterrows():
            product_cost = st.session_state.products[
                st.session_state.products['name'] == row['product']
            ]['cost'].values
            if len(product_cost) > 0:
                total_cost += row['quantity'] * product_cost[0]
        total_profit = df['revenue'].sum() - total_cost
        st.metric("Total Profit", f"RM{total_profit:,.2f}")
    with col4:
        st.metric("Days Active", f"{(df['date'].max() - df['date'].min()).days} days")
    
    st.divider()
    
    # ========== SECTION 2: PRODUCT PERFORMANCE ==========
    st.subheader("🏆 Product Performance Analysis")
    
    # Merge sales with product cost data
    product_perf = df.groupby('product').agg({
        'quantity': 'sum',
        'revenue': 'sum'
    }).reset_index()
    product_perf.columns = ['product', 'qty_sold', 'total_revenue']
    
    # Add cost and profit from products DataFrame
    if not st.session_state.products.empty:
        cost_map = dict(zip(st.session_state.products['name'], st.session_state.products['cost']))
        price_map = dict(zip(st.session_state.products['name'], st.session_state.products['price']))
        category_map = dict(zip(st.session_state.products['name'], st.session_state.products['category']))
        
        product_perf['cost_unit'] = product_perf['product'].map(cost_map).fillna(0).astype(float)
        product_perf['price_unit'] = product_perf['product'].map(price_map).fillna(0).astype(float)
        product_perf['category'] = product_perf['product'].map(category_map).fillna('Unknown')
        product_perf['total_cost'] = (product_perf['qty_sold'] * product_perf['cost_unit']).astype(float)
        product_perf['total_profit'] = (product_perf['total_revenue'] - product_perf['total_cost']).astype(float)
        product_perf['profit_margin_%'] = ((product_perf['total_profit'] / product_perf['total_revenue'] * 100).fillna(0)).round(1)
        
        # Sort by profit
        product_perf = product_perf.sort_values('total_profit', ascending=False)
        
        # Display table
        display_cols = ['product', 'qty_sold', 'total_revenue', 'total_cost', 'total_profit', 'profit_margin_%']
        display_df = product_perf[display_cols].copy()
        display_df['qty_sold'] = display_df['qty_sold'].astype(int)
        display_df['total_revenue'] = display_df['total_revenue'].astype(float).round(2)
        display_df['total_cost'] = display_df['total_cost'].astype(float).round(2)
        display_df['total_profit'] = display_df['total_profit'].astype(float).round(2)
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                'product': st.column_config.TextColumn('Product'),
                'qty_sold': st.column_config.NumberColumn('Qty Sold', format='%d'),
                'total_revenue': st.column_config.NumberColumn('Revenue', format='RM%.2f'),
                'total_cost': st.column_config.NumberColumn('Cost', format='RM%.2f'),
                'total_profit': st.column_config.NumberColumn('Profit', format='RM%.2f'),
                'profit_margin_%': st.column_config.NumberColumn('Margin %', format='%.1f%%')
            }
        )
    
    st.divider()
    
    # ========== SECTION 3: DEMAND FORECAST & TRENDS ==========
    st.subheader("📈 Demand & Revenue Trends")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Daily Revenue Trend**")
        daily = df.groupby('date')['revenue'].sum().reset_index()
        fig_daily = px.line(
            daily,
            x='date',
            y='revenue',
            markers=True,
            title="Daily Sales Trend"
        )
        fig_daily.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig_daily, use_container_width=True)
    
    with col2:
        st.write("**Product Mix**")
        if not st.session_state.products.empty:
            product_mix = df.groupby('product')['revenue'].sum().reset_index()
            fig_pie = px.pie(
                product_mix,
                names='product',
                values='revenue',
                title="Revenue Distribution"
            )
            fig_pie.update_layout(height=350)
            st.plotly_chart(fig_pie, use_container_width=True)
    
    st.divider()
    
    # ========== SECTION 4: STOCK RECOMMENDATIONS ==========
    st.subheader("📦 Inventory Optimization - Stock Level Recommendations")
    st.write("AI predicts product demand to suggest optimal stock levels")
    
    if not st.session_state.products.empty and not product_perf.empty:
        stock_recommendations = []
        
        for _, product_row in product_perf.iterrows():
            product_name = product_row['product']
            avg_demand, recommended_stock = forecast_product_demand(product_name, days=30)
            
            if avg_demand is not None and recommended_stock is not None:
                current_stock = st.session_state.products[
                    st.session_state.products['name'] == product_name
                ]['stock'].values[0]
                
                stock_recommendations.append({
                    'product': product_name,
                    'current_stock': int(current_stock),
                    'avg_daily_demand': round(avg_demand, 1),
                    'recommended_stock': recommended_stock,
                    'action': '🔴 RESTOCK' if current_stock < recommended_stock * 0.3 else (
                        '🟡 LOW' if current_stock < recommended_stock * 0.6 else '🟢 OPTIMAL'
                    )
                })
        
        if stock_recommendations:
            rec_df = pd.DataFrame(stock_recommendations)
            st.dataframe(
                rec_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    'product': st.column_config.TextColumn('Product'),
                    'current_stock': st.column_config.NumberColumn('Current Stock', format='%d'),
                    'avg_daily_demand': st.column_config.NumberColumn('Avg Daily Demand', format='%.1f'),
                    'recommended_stock': st.column_config.NumberColumn('Recommended Stock', format='%d'),
                    'action': st.column_config.TextColumn('Status')
                }
            )
            
            st.info(
                "💡 **How to use**: \n"
                "- **🟢 OPTIMAL**: Current stock is sufficient for anticipated demand\n"
                "- **🟡 LOW**: Consider restocking soon\n"
                "- **🔴 RESTOCK**: Urgent - Restock immediately to avoid stockouts"
            )
        else:
            st.info("Not enough sales history to generate stock recommendations. Record more sales data.")
    
    st.divider()
    
    # ========== SECTION 5: AI PROMOTION STRATEGY ==========
    st.subheader("🎯 AI-Powered Promotion Strategy - Maximize Profit")
    st.write("AI calculates optimal discount % that maximizes profit while boosting customer engagement")
    
    if not st.session_state.products.empty and not product_perf.empty:
        # Show ALL products for analysis (not just weak ones)
        st.info(f"📊 Analyzing all {len(product_perf)} products - Select any to see promotion recommendations")
        
        selected_product = st.selectbox(
            "Select product to analyze AI promotion strategy",
            product_perf['product'].tolist()
        )
        
        if selected_product:
            product_data = product_perf[product_perf['product'] == selected_product].iloc[0]
            
            # Retrieve current stock level
            current_stock = st.session_state.products[
                st.session_state.products['name'] == selected_product
            ]['stock'].values[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Current Sales", f"{product_data['qty_sold']:.0f} units")
                st.metric("Current Revenue", f"RM{product_data['total_revenue']:,.2f}")
                st.metric("Current Profit", f"RM{product_data['total_profit']:,.2f}")
                st.metric("Profit Margin", f"{product_data['profit_margin_%']:.1f}%")
            
            with col2:
                st.metric("Current Stock", f"{int(current_stock)} units")
                st.metric("Price per Unit", f"RM{product_data['price_unit']:.2f}")
                st.metric("Cost per Unit", f"RM{product_data['cost_unit']:.2f}")
                st.metric("Profit per Unit", f"RM{product_data['price_unit'] - product_data['cost_unit']:.2f}")
            
            st.divider()
            
            # AI CALCULATION
            optimal_discount = calculate_optimal_discount(
                selected_product,
                product_data['price_unit'],
                product_data['cost_unit'],
                product_data['qty_sold']
            )
            
            if optimal_discount >= 0:
                # Calculate predicted outcomes with optimal discount
                elasticity = calculate_price_elasticity(selected_product)
                if optimal_discount > 0:
                    new_price = product_data['price_unit'] * (1 - optimal_discount / 100)
                    quantity_change_pct = elasticity * (-optimal_discount)
                else:
                    new_price = product_data['price_unit']
                    quantity_change_pct = 0
                
                new_qty = product_data['qty_sold'] * (1 + quantity_change_pct / 100)
                
                new_revenue = new_price * new_qty
                new_cost = product_data['cost_unit'] * new_qty
                new_profit = new_revenue - new_cost
                profit_change = new_profit - product_data['total_profit']
                profit_change_pct = (profit_change / product_data['total_profit'] * 100) if product_data['total_profit'] > 0 else 0
                
                st.subheader("🤖 AI Recommendation")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Optimal Discount",
                        f"{optimal_discount}%",
                        help=f"Recommend {optimal_discount}% discount to maximize profit"
                    )
                with col2:
                    st.metric(
                        "New Price",
                        f"RM{new_price:.2f}",
                        delta=f"{-optimal_discount}%",
                        delta_color="inverse"
                    )
                with col3:
                    st.metric(
                        "Expected Volume Boost",
                        f"{quantity_change_pct:+.0f}%",
                        delta=f"{new_qty - product_data['qty_sold']:+.0f} units"
                    )
                
                st.divider()
                
                # Impact metrics
                st.subheader("💰 Projected Impact with AI Recommendation")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "New Revenue",
                        f"RM{new_revenue:,.2f}",
                        delta=f"{((new_revenue / product_data['total_revenue'] - 1) * 100):+.1f}%"
                    )
                with col2:
                    st.metric(
                        "New Profit",
                        f"RM{new_profit:,.2f}",
                        delta=f"RM{profit_change:+,.2f}",
                        delta_color="off" if profit_change > 0 else "inverse"
                    )
                with col3:
                    new_margin = ((new_profit / new_revenue * 100) if new_revenue > 0 else 0)
                    st.metric(
                        "New Margin %",
                        f"{new_margin:.1f}%",
                        delta=f"{new_margin - product_data['profit_margin_%']:+.1f}%",
                        delta_color="off"
                    )
                
                # Comparison chart
                comparison = pd.DataFrame({
                    'Metric': ['Revenue', 'Cost', 'Profit'],
                    'Current': [product_data['total_revenue'], product_data['total_cost'], product_data['total_profit']],
                    'With AI Promotion': [new_revenue, new_cost, new_profit]
                })
                
                fig_comparison = px.bar(
                    comparison,
                    x='Metric',
                    y=['Current', 'With AI Promotion'],
                    barmode='group',
                    title=f'AI Promotion Impact - {optimal_discount}% Discount',
                    labels={'value': 'Amount (RM)', 'variable': 'Scenario'},
                    color_discrete_map={'Current': '#667eea', 'With AI Promotion': '#764ba2'}
                )
                fig_comparison.update_layout(height=400)
                st.plotly_chart(fig_comparison, use_container_width=True)
                
                # Final recommendation
                st.divider()
                if optimal_discount > 0 and profit_change > 0:
                    st.success(
                        f"✅ **RECOMMENDED**: Apply {optimal_discount}% discount\n\n"
                        f"Expected profit increase: **RM{profit_change:,.2f}** ({profit_change_pct:+.1f}%)\n\n"
                        f"This discount will boost customer engagement while maximizing profitability."
                    )
                elif optimal_discount > 0:
                    st.info(
                        f"⚠️ **LIMITED OPPORTUNITY**: Even with {optimal_discount}% discount, profit increase is marginal.\n\n"
                        f"Consider:\n"
                        f"- Focusing on products with higher margins\n"
                        f"- Bundling this item with higher-profit products\n"
                        f"- Running time-based promotions to drive volume"
                    )
                else:
                    st.success(
                        f"✅ **OPTIMAL STRATEGY**: No discount needed\n\n"
                        f"Your current pricing at **RM{product_data['price_unit']:.2f}** is already optimized.\n\n"
                        f"With a **{product_data['profit_margin_%']:.1f}% profit margin**, focus on volume growth through other marketing channels."
                    )
    
    st.divider()
    
    # ========== SECTION 6: 30-DAY DEMAND FORECAST ==========
    st.subheader("📈 30-Day Overall Revenue Forecast")
    
    if len(df) < 3:
        st.warning("Need at least 3 sales records for forecasting.")
        return
    
    forecast_input = df.groupby('date')['revenue'].sum().reset_index().sort_values('date')
    forecast_input.columns = ['ds', 'y']
    step_col, conf_col = st.columns(2)
    with step_col:
        st.info(f"📊 Using {len(forecast_input)} days of historical data")
    with conf_col:
        st.metric("Confidence Level", "80%")
    
    # Try ARIMA if available and enough data
    use_arima = False
    if ARIMA_AVAILABLE and len(forecast_input) >= 7:
        arima_forecast = forecast_with_arima(30)
        if arima_forecast is not None:
            use_arima = True
    
    if use_arima:
        forecast_df = _plot_arima_forecast(forecast_input, arima_forecast)
    else:
        forecast_df = _plot_exponential_forecast(forecast_input)
    
    st.session_state.forecast_data = forecast_df
    
    # Anomaly Detection
    st.divider()
    st.subheader("🚨 Anomaly Detection (ML)")
    anomalies = detect_sales_anomalies()
    if not anomalies.empty:
        st.warning(f"Found {len(anomalies)} unusual sales patterns:")
        st.dataframe(anomalies, use_container_width=True, hide_index=True)
    else:
        st.success("✅ No unusual sales patterns detected")

def _plot_arima_forecast(forecast_input, arima_forecast):
    """Plot ARIMA forecast"""
    try:
        # Prepare data
        last_date = forecast_input['ds'].max()
        forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=30)
        
        forecast_values = arima_forecast['yhat'].values
        upper = arima_forecast.iloc[:, 1].values
        lower = arima_forecast.iloc[:, 0].values
        
        fig = go.Figure()
        
        # Historical data
        fig.add_trace(go.Scatter(
            x=forecast_input['ds'],
            y=forecast_input['y'],
            mode='markers',
            name='Actual',
            marker=dict(size=8, color='#667eea')
        ))
        
        # Forecast
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast_values,
            mode='lines',
            name='ARIMA Forecast',
            line=dict(color='#764ba2', width=3)
        ))
        
        # Confidence interval
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=upper,
            fill=None,
            mode='lines',
            line_color='rgba(0,0,0,0)',
            showlegend=False
        ))
        
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=lower,
            fill='tonexty',
            mode='lines',
            line_color='rgba(0,0,0,0)',
            name='80% CI',
            fillcolor='rgba(118, 75, 162, 0.2)'
        ))
        
        fig.update_layout(
            hovermode='x unified',
            height=500,
            xaxis_title='Date',
            yaxis_title='Revenue (RM)',
            title='ARIMA(1,1,1) Forecast - ML Based'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg 30-Day Forecast", f"RM{forecast_values.mean():,.2f}")
        with col2:
            avg_actual = forecast_input['y'].mean()
            growth = ((forecast_values.mean() / avg_actual) - 1) * 100 if avg_actual > 0 else 0
            st.metric("Expected Growth", f"{growth:+.1f}%")
        with col3:
            st.metric("30-Day Total", f"RM{forecast_values.sum():,.2f}")
        
        # Return forecast dataframe
        result_df = pd.DataFrame({
            'ds': forecast_dates,
            'yhat': forecast_values,
            'yhat_upper': upper,
            'yhat_lower': lower
        })
        
        return result_df
    
    except Exception as e:
        st.error(f"ARIMA error: {str(e)}")
        return pd.DataFrame()

def _plot_exponential_forecast(forecast_input):
    """Plot exponential smoothing forecast (fallback)"""
    y_values = forecast_input['y'].values
    
    # Exponential smoothing
    alpha = 0.3
    smoothed = np.zeros_like(y_values, dtype=float)
    smoothed[0] = y_values[0]
    
    for i in range(1, len(y_values)):
        smoothed[i] = alpha * y_values[i] + (1 - alpha) * smoothed[i-1]
    
    # Calculate trend
    x = np.arange(len(y_values))
    z = np.polyfit(x, smoothed, 1)
    trend = z[0]
    intercept = z[1]
    
    # Generate forecast
    last_date = forecast_input['ds'].max()
    forecast_dates = pd.date_range(start=last_date + timedelta(days=1), periods=30)
    x_future = np.arange(len(y_values), len(y_values) + 30)
    forecast_values = trend * x_future + intercept
    forecast_values = np.maximum(forecast_values, 0)
    
    fig = go.Figure()
    
    # Historical
    fig.add_trace(go.Scatter(
        x=forecast_input['ds'],
        y=forecast_input['y'],
        mode='markers',
        name='Actual',
        marker=dict(size=8, color='#667eea')
    ))
    
    # Trend
    trend_line = trend * x + intercept
    fig.add_trace(go.Scatter(
        x=forecast_input['ds'],
        y=trend_line,
        mode='lines',
        name='Trend',
        line=dict(color='#667eea', dash='dash', width=2)
    ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast_values,
        mode='lines',
        name='Forecast',
        line=dict(color='#764ba2', dash='dash')
    ))
    
    # Confidence interval
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast_values * 1.2,
        fill=None,
        mode='lines',
        line_color='rgba(0,0,0,0)',
        showlegend=False
    ))
    
    fig.add_trace(go.Scatter(
        x=forecast_dates,
        y=forecast_values * 0.8,
        fill='tonexty',
        mode='lines',
        line_color='rgba(0,0,0,0)',
        name='80% CI',
        fillcolor='rgba(118, 75, 162, 0.2)'
    ))
    
    fig.update_layout(
        hovermode='x unified',
        height=500,
        xaxis_title='Date',
        yaxis_title='Revenue (₹)',
        title='Exponential Smoothing Forecast - Fallback Method'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Avg 30-Day Forecast", f"RM{forecast_values.mean():,.2f}")
    with col2:
        avg_actual = forecast_input['y'].mean()
        growth = ((forecast_values.mean() / avg_actual) - 1) * 100 if avg_actual > 0 else 0
        st.metric("Expected Growth", f"{growth:+.1f}%")
    with col3:
        st.metric("30-Day Total", f"RM{forecast_values.sum():,.2f}")
    
    # Return forecast dataframe
    result_df = pd.DataFrame({
        'ds': forecast_dates,
        'yhat': forecast_values,
        'yhat_upper': forecast_values * 1.2,
        'yhat_lower': forecast_values * 0.8
    })
    
    return result_df

# ============================================================================
# PAGE: SIMULATION
# ============================================================================
def page_simulation():
    st.title("🧮 What-If Simulation")
    
    if st.session_state.sales.empty:
        st.warning("Please record some sales first.")
        return
    
    current_revenue = calculate_total_revenue()
    current_profit = calculate_profit_estimation()
    
    st.subheader("📊 Current Metrics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Current Revenue", f"RM{current_revenue:,.2f}")
    with col2:
        st.metric("Current Profit", f"RM{current_profit:,.2f}")
    
    st.divider()
    st.subheader("⚙️ Simulation Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        discount_pct = st.slider(
            "Discount on Products (%)",
            min_value=-20,
            max_value=0,
            value=-5,
            step=1,
            help="Negative = discount (reduce price)"
        )
    
    with col2:
        boost_pct = st.slider(
            "Marketing Boost (%)",
            min_value=0,
            max_value=100,
            value=20,
            step=5,
            help="Expected sales increase from marketing"
        )
    
    # Calculate simulation
    discount_factor = 1 + (discount_pct / 100)
    boost_factor = 1 + (boost_pct / 100)
    
    simulated_revenue = current_revenue * discount_factor * boost_factor
    simulated_profit = simulated_revenue * (st.session_state.user_profile['profit_margin'] / 100)
    
    revenue_change = simulated_revenue - current_revenue
    profit_change = simulated_profit - current_profit
    
    st.divider()
    st.subheader("📈 Simulation Results")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Projected Revenue",
            f"RM{simulated_revenue:,.2f}",
            delta=f"RM{revenue_change:+,.2f}",
            delta_color="inverse"
        )
    
    with col2:
        st.metric(
            "Projected Profit",
            f"RM{simulated_profit:,.2f}",
            delta=f"RM{profit_change:+,.2f}",
            delta_color="inverse"
        )
    
    with col3:
        pct_change = ((simulated_revenue - current_revenue) / current_revenue * 100) if current_revenue > 0 else 0
        st.metric(
            "Revenue Change %",
            f"{pct_change:+.1f}%",
            delta_color="inverse"
        )
    
    with col4:
        breakeven = "✅ Profitable" if simulated_profit > 0 else "⚠️ Loss"
        st.metric(
            "Status",
            breakeven
        )
    
    # Comparison chart
    st.subheader("Comparison Chart")
    
    comparison_data = pd.DataFrame({
        'Scenario': ['Current', 'Simulated'],
        'Revenue': [current_revenue, simulated_revenue],
        'Profit': [current_profit, simulated_profit]
    })
    
    fig = px.bar(
        comparison_data,
        x='Scenario',
        y=['Revenue', 'Profit'],
        barmode='group',
        title="Before vs After Simulation"
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# PAGE: LOAN CENTER
# ============================================================================
def page_loan_center():
    st.title("💳 Loan Pre-Eligibility & Matching Center")
    
    profile = st.session_state.user_profile
    if not profile['business_name']:
        st.warning("Please complete Business Registration first.")
        return
    
    # Calculate loan score
    loan_score = calculate_loan_score()
    
    # Max loan amount (simplified: based on 6 months revenue)
    monthly_rev = profile['monthly_revenue']
    max_loan = monthly_rev * 6 * (loan_score + 0.5)
    
    st.session_state.loan_status['loan_score'] = loan_score
    st.session_state.loan_status['max_loan_amount'] = max_loan
    
    # Display Score Card
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Loan Score", f"{loan_score:.2f}", delta="out of 1.0")
    
    with col2:
        if loan_score >= 0.75:
            st.metric("Status", "✅ Strong")
        elif loan_score >= 0.5:
            st.metric("Status", "⏳ Fair")
        else:
            st.metric("Status", "❌ Low")
    
    with col3:
        st.metric("Max Eligible Loan", f"RM{max_loan:,.0f}")
    
    st.divider()
    
    # ML Model Status
    if st.session_state.ml_status['loan_model_trained']:
        st.success("🤖 **ML Model Active** - Using Random Forest classifier")
    elif len(st.session_state.sales) >= 20:
        st.info("📊 **Rule-Based Scoring** - ML model will auto-train with more sales")
    else:
        st.info(f"📊 **Rule-Based Scoring** - Need {20 - len(st.session_state.sales)} more sales for ML model")
    
    # Score breakdown
    st.subheader("📋 Score Breakdown")
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_revenue = calculate_total_revenue()
    profit_margin = profile['profit_margin']
    monthly_revenue = profile['monthly_revenue']
    existing_commitment = profile['existing_loan_commitment']
    
    if st.session_state.sales.empty:
        revenue_stability = 0
    else:
        df = st.session_state.sales.copy()
        df['date'] = pd.to_datetime(df['date'])
        daily_revenue = df.groupby('date')['revenue'].sum()
        if len(daily_revenue) > 1:
            cv = daily_revenue.std() / daily_revenue.mean() if daily_revenue.mean() > 0 else 1
            revenue_stability = max(0, 1 - cv)
        else:
            revenue_stability = 0.5
    
    with col1:
        st.write(f"**Revenue Stability** (40% weight)")
        st.write(f"Score: {revenue_stability:.2f}")
    
    with col2:
        profit_score = min(1, profit_margin / 50)
        st.write(f"**Profit Margin** (30% weight)")
        st.write(f"Score: {profit_score:.2f}")
    
    with col3:
        if monthly_revenue > 0:
            cash_flow = total_revenue / (monthly_revenue * 12) if monthly_revenue > 0 else 0
            cash_flow_ratio = min(1, cash_flow / 2)
        else:
            cash_flow_ratio = 0
        st.write(f"**Cash Flow Ratio** (20% weight)")
        st.write(f"Score: {cash_flow_ratio:.2f}")
    
    with col4:
        if monthly_revenue > 0:
            debt_ratio = existing_commitment / (monthly_revenue * 12) if monthly_revenue > 0 else 1
            debt_inverse = max(0, 1 - debt_ratio)
        else:
            debt_inverse = 0.5
        st.write(f"**Debt Ratio** (10% weight)")
        st.write(f"Score: {debt_inverse:.2f}")
    
    st.divider()
    
    # ========== LOAN PROVIDER MATCHING ==========
    st.subheader("🏦 Matching Loan Programs")
    st.write("We've matched you with these loan programs based on your profile:")
    
    # Find eligible loan programs
    eligible_programs = []
    for provider in LOAN_PROVIDERS:
        for program in provider['programs']:
            if (loan_score >= program['min_score'] and 
                total_revenue >= program['min_revenue']):
                eligible_programs.append({
                    'provider': provider['name'],
                    'program': program['name'],
                    'max_loan': program['max_loan'],
                    'rate': program['interest_rate'],
                    'tenure': program['tenure'],
                    'description': program['description'],
                    'full_program': program
                })
    
    if not eligible_programs:
        st.warning(
            f"❌ No loan programs match your current profile. "
            f"Your score: {loan_score:.2f}, Annual revenue: RM{total_revenue:,.0f}"
        )
        st.info(
            "💡 **Ways to improve eligibility:**\n"
            "1. Increase business revenue through sales growth\n"
            "2. Improve profit margins\n"
            "3. Reduce existing debt commitments\n"
            "4. Build consistent sales history"
        )
    else:
        st.success(f"✅ Found {len(eligible_programs)} matching loan programs!")
        
        # Display matching programs as selectable cards
        for idx, prog in enumerate(eligible_programs):
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
            
            with col1:
                st.write(f"**{prog['provider']}**")
                st.write(f"Program: {prog['program']}")
                st.caption(prog['description'])
            
            with col2:
                st.metric("Max Loan", f"RM{prog['max_loan']/1000000:.0f}M")
            
            with col3:
                st.metric("Interest", prog['rate'])
            
            with col4:
                st.metric("Tenure", prog['tenure'])
            
            # Select button for this program
            if st.button(
                f"📋 Apply for {prog['program']}",
                key=f"select_program_{idx}",
                use_container_width=False
            ):
                st.session_state['selected_program'] = prog
                st.session_state['selected_provider'] = prog['provider']
                st.success(f"✅ Selected: {prog['program']}")
                st.rerun()
        
        st.info(
            "💡 **Next Step**: Click the 'Apply' button above to go directly to the lender's website to complete your application. "
            "All your business information has been pre-verified and matched."
        )
        st.write(f"**Profit Margin** (30% weight)")
        st.write(f"Score: {profit_score:.2f}")
    
    with col3:
        if monthly_revenue > 0:
            cash_flow = total_revenue / (monthly_revenue * 12) if monthly_revenue > 0 else 0
            cash_flow_ratio = min(1, cash_flow / 2)
        else:
            cash_flow_ratio = 0
        st.write(f"**Cash Flow Ratio** (20% weight)")
        st.write(f"Score: {cash_flow_ratio:.2f}")
    
    with col4:
        if monthly_revenue > 0:
            debt_ratio = existing_commitment / (monthly_revenue * 12) if monthly_revenue > 0 else 1
            debt_inverse = max(0, 1 - debt_ratio)
        else:
            debt_inverse = 0.5
        st.write(f"**Debt Ratio** (10% weight)")
        st.write(f"Score: {debt_inverse:.2f}")
    
    st.divider()
    
    # Loan Application Form
    st.subheader("📝 Loan Application Form")
    
    with st.form("loan_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.text_input(
                "Business Name",
                value=profile['business_name'],
                disabled=True
            )
            st.number_input(
                "Annual Revenue",
                value=int(total_revenue),
                disabled=True
            )
        
        with col2:
            st.text_input(
                "Business Type",
                value=profile['business_type'],
                disabled=True
            )
            st.number_input(
                "Years Operating",
                value=profile['years_operating'],
                disabled=True
            )
        
        st.divider()
        
        requested_amount = st.number_input(
            "Requested Loan Amount (₹)",
            min_value=0,
            max_value=int(max_loan),
            step=10000,
            value=min(int(max_loan * 0.5), int(max_loan))
        )
        
        purpose = st.selectbox(
            "Loan Purpose",
            ["Expansion", "Equipment", "Working Capital", "Debt Consolidation", "Other"]
        )
        
        submitted = st.form_submit_button("📤 Submit Application", use_container_width=True)
        
        if submitted:
            # Simulate approval status
            if loan_score >= 0.75:
                status = "Approved"
                status_emoji = "✅"
                color_style = "success-status"
            elif loan_score >= 0.5:
                status = "Processing"
                status_emoji = "⏳"
                color_style = "warning-status"
            else:
                status = "Rejected"
                status_emoji = "❌"
                color_style = "danger-status"
            
            st.session_state.loan_status['status'] = status
            st.session_state.loan_status['requested_amount'] = requested_amount
            
            st.rerun()
    
    # Loan Status Display
    if st.session_state.loan_status['status'] != 'Not Applied':
        st.divider()
        st.subheader("📌 Application Status")
        
        status = st.session_state.loan_status['status']
        requested = st.session_state.loan_status['requested_amount']
        
        if status == "Approved":
            st.markdown(f"""
            <div class="success-status">
            <h4>✅ APPROVED</h4>
            <p><strong>Application Status:</strong> Your loan application has been <b>APPROVED</b></p>
            <p><strong>Requested Amount:</strong> ₹{requested:,.0f}</p>
            <p><strong>Approval Limit:</strong> ₹{max_loan:,.0f}</p>
            <p><strong>Next Steps:</strong> Our loan officer will contact you within 24 hours.</p>
            </div>
            """, unsafe_allow_html=True)
        
        elif status == "Processing":
            st.markdown(f"""
            <div class="warning-status">
            <h4>⏳ UNDER REVIEW</h4>
            <p><strong>Application Status:</strong> Your application is <b>UNDER REVIEW</b></p>
            <p><strong>Requested Amount:</strong> ₹{requested:,.0f}</p>
            <p><strong>Estimated Time:</strong> 5-7 business days</p>
            <p><strong>What happens next:</strong> We'll verify your documents and conduct a business assessment.</p>
            </div>
            """, unsafe_allow_html=True)
        
        else:
            st.markdown(f"""
            <div class="danger-status">
            <h4>❌ NOT ELIGIBLE</h4>
            <p><strong>Application Status:</strong> Unfortunately, you are currently <b>NOT ELIGIBLE</b></p>
            <p><strong>Loan Score:</strong> {loan_score:.2f}/1.0 (Minimum required: 0.50)</p>
            <p><strong>Recommendations:</strong></p>
            <ul>
            <li>Increase monthly revenue through sales growth</li>
            <li>Improve profit margins</li>
            <li>Reduce existing debt commitments</li>
            <li>Build more consistent sales records</li>
            </ul>
            <p><strong>Re-apply after:</strong> 3 months</p>
            </div>
            """, unsafe_allow_html=True)

# ============================================================================
# MAIN APP NAVIGATION
# ============================================================================
def main():
    # Sidebar navigation
    st.sidebar.title("💼 MSME BI System")
    st.sidebar.divider()
    
    page = st.sidebar.radio(
        "Navigation",
        [
            "Dashboard",
            "Business Registration",
            "POS System",
            "Upload CSV",
            "Analytics & Promotion",
            "Simulation",
            "Loan Center"
        ],
        label_visibility="collapsed"
    )
    
    st.sidebar.divider()
    
    # Display business info in sidebar
    if st.session_state.user_profile['business_name']:
        st.sidebar.markdown("### Business Info")
        st.sidebar.write(f"**{st.session_state.user_profile['business_name']}**")
        st.sidebar.write(f"Type: {st.session_state.user_profile['business_type']}")
        st.sidebar.write(f"Revenue: ₹{st.session_state.user_profile['monthly_revenue']:,.0f}/mo")
    
    # Route to pages
    if page == "Dashboard":
        page_dashboard()
    elif page == "Business Registration":
        page_business_registration()
    elif page == "POS System":
        page_pos_system()
    elif page == "Upload CSV":
        page_csv_upload()
    elif page == "Analytics & Promotion":
        page_analytics_and_promotion()
    elif page == "Simulation":
        page_simulation()
    elif page == "Loan Center":
        page_loan_center()

if __name__ == "__main__":
    main()
