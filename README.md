# 💼 MSME Business Intelligence & Loan Pre-Eligibility System

A comprehensive Streamlit application designed to help Micro, Small & Medium Enterprises (MSMEs) manage their business intelligence, sales analytics, demand forecasting, and loan pre-eligibility assessment.

## 🎯 Features

### 🤖 ML-Powered Features
- **Random Forest Loan Scoring** - Auto-trained after 20 sales records
- **ARIMA Forecasting** - Statistical time-series forecasting (7+ days data)
- **Anomaly Detection** - Isolation Forest to detect unusual patterns
- **Smart Model Selection** - Auto-switches to ML when enough data available

### 📊 Dashboard
- Real-time KPI metrics (revenue, profit, products, transactions)
- Monthly revenue trend visualization
- Top products analysis

### 🏢 Business Registration
- Business profile setup and management
- Store business details (name, type, years operating, revenue, profit margin)
- Track existing loan commitments

### 🛒 POS System
- Add and manage products with pricing and inventory
- Record sales transactions in real-time
- Automatic stock management
- Sales history tracking

### 📂 CSV Upload
- Import historical sales data from CSV files
- Validate data format automatically
- Merge with existing sales records

### 📈 Analytics Engine
- Total revenue and profit calculation
- Daily and monthly trend analysis
- Product performance metrics
- Revenue composition analysis

### 📊 Demand Forecasting
- 30-day revenue forecasting using Prophet
- Confidence intervals visualization
- Growth projection analysis
- Data-driven insights with statistical models

### 🧮 What-If Simulation
- Test discount impact on revenue
- Model marketing boost scenarios
- Compare current vs. projected metrics
- Visual before/after comparison

### 💳 Loan Pre-Eligibility System
- Intelligent loan scoring algorithm
- Components: revenue stability, profit margin, cash flow, debt ratio
- Maximum eligible loan amount calculation
- Smart loan application form with auto-filled fields
- Application status simulation (Approved/Under Review/Rejected)
- Recommendations for improvement

## 🏗️ Architecture

```
Single Streamlit App (app.py)
├── Session State Management
│   ├── user_profile
│   ├── products
│   ├── sales
│   ├── forecast_data
│   └── loan_status
├── Analytics Functions
│   ├── calculate_total_revenue()
│   ├── calculate_monthly_trend()
│   ├── calculate_top_products()
│   ├── calculate_profit_estimation()
│   └── calculate_loan_score()
└── Page Functions
    ├── page_dashboard()
    ├── page_business_registration()
    ├── page_pos_system()
    ├── page_csv_upload()
    ├── page_analytics()
    ├── page_demand_forecast()
    ├── page_simulation()
    └── page_loan_center()
```

## 📦 Requirements

- **Python 3.11+**
- Streamlit 1.28+
- Pandas 2.0+
- NumPy 1.24+
- Scikit-learn 1.3+ (ML models: Random Forest, Isolation Forest)
- Statsmodels 0.14+ (ARIMA forecasting)
- Plotly 5.17+ (visualization)

## 🚀 Installation & Setup

### Step 1: Clone or Extract Project
```bash
cd c:\Study\Y3S1\BorNeo
```

### Step 2: Create Virtual Environment (Recommended)
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Application
```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

## 📖 Usage Guide

### 1. **Start with Business Registration**
   - Navigate to "Business Registration"
   - Fill in all details about your business
   - Save your profile

### 2. **Add Products (POS System)**
   - Go to "POS System"
   - Add products with name, price, and stock
   - Record sales transactions as they happen

### 3. **View Dashboard**
   - Monitor KPIs and trends
   - Check inventory and sales history
   - Get real-time business insights

### 4. **Import Historical Data (Optional)**
   - Use "Upload CSV" to add past sales data
   - Format: `product, quantity, date, revenue`
   - Dates should be in YYYY-MM-DD format

### 5. **Analyze Sales**
   - Visit "Analytics" page
   - View revenue trends, product mix, performance metrics
   - Understand your sales patterns

### 6. **Forecast Demand**
   - Go to "Demand Forecast"
   - View 30-day revenue projections
   - Analyze growth expectations
   - Plan inventory based on forecasts

### 7. **Run Simulations**
   - Use "Simulation" to test scenarios
   - Model discount and marketing boost impacts
   - Make data-driven business decisions

### 8. **Check Loan Eligibility**
   - Navigate to "Loan Center"
   - View your loan score and eligibility
   - Submit loan application
   - Review approval status and recommendations

## 📊 Data Models

### User Profile
```python
{
    'business_name': str,
    'business_type': str,
    'years_operating': int,
    'monthly_revenue': float,
    'profit_margin': float (0-100),
    'existing_loan_commitment': float
}
```

### Products DataFrame
| Column | Type | Description |
|--------|------|-------------|
| name | str | Product name |
| price | float | Unit price in ₹ |
| stock | int | Units in inventory |

### Sales DataFrame
| Column | Type | Description |
|--------|------|-------------|
| product | str | Product name |
| quantity | int | Quantity sold |
| date | date | Transaction date |
| revenue | float | Total revenue from sale |

## 🧠 Loan Scoring Algorithm

### Two Modes:
1. **Rule-Based** (Initial): 
   - Used when insufficient data
```
Score = (0.4 × Revenue Stability) + 
        (0.3 × Profit Margin) + 
        (0.2 × Cash Flow Ratio) + 
        (0.1 × Debt Ratio Inverse)
```

2. **ML Model** (Auto-activated):
   - Random Forest Classifier
   - Automatically trained after 20 sales records
   - Learns from 8 business features:
     - Total revenue
     - Profit margin
     - Years operating
     - Monthly revenue
     - Existing commitments
     - Number of products
     - Number of sales
     - Revenue stability

### Eligibility Tiers
| Score Range | Status |
|-------------|--------|
| ≥ 0.75 | ✅ Approved |
| 0.5 - 0.75 | ⏳ Under Review |
| < 0.50 | ❌ Not Eligible |

## 📊 Demand Forecasting

### Smart Method Selection:
1. **ARIMA** (7+ days of data):
   - ARIMA(1,1,1) statistical model
   - Better accuracy with temporal patterns
   - 80% confidence intervals

2. **Exponential Smoothing** (Fallback):
   - α = 0.3 smoothing factor
   - Linear trend extraction
   - Works with 3+ days data

## 🚨 Anomaly Detection

Uses **Isolation Forest** to detect unusual sales patterns:
- Identifies outliers in quantity & revenue
- Flags suspicious transactions
- 10% contamination threshold

## 🔄 Data Persistence

**Note:** This prototype uses Streamlit's session state, which means:
- Data persists during the session
- Data is lost when the browser is refreshed
- For production, integrate a database (PostgreSQL, Firebase, etc.)

To add database support in future:
```python
# Future enhancement
import sqlalchemy
engine = sqlalchemy.create_engine('postgresql://...')
```

## 📋 CSV Format Example

```csv
product,quantity,date,revenue
Widget A,5,2024-01-15,5000
Widget B,3,2024-01-16,4500
Widget A,2,2024-01-17,2000
```

## 🎨 UI Customization

The app includes custom CSS styling:
- Gradient metrics cards
- Status indicator colors (success/warning/danger)
- Clean business-style layout
- Responsive design for all screen sizes

## 🚨 Troubleshooting

### Prophet ImportError
```bash
pip install --upgrade pystan=3.8.0
```

### Port Already in Use
```bash
streamlit run app.py --server.port 8502
```

### Data Not Persisting
- Refresh clears session state by design
- Use CSV upload to reimport data

### Forecast Errors
- Ensure at least 5 days of sales data
- Check date format (YYYY-MM-DD)
- More data = better predictions

## 🔄 Workflow Example

1. Register business → 2 min
2. Add 5-10 products → 3 min
3. Record 10-15 sales → 5 min
4. View analytics → 2 min
5. Generate forecast → 1 min
6. Run simulations → 3 min
7. Check loan eligibility → 2 min

**Total:** ~16 minutes for complete demo

## 📈 Performance Notes

- Forecasting: Handles 1000+ sales records
- Dashboard: Real-time metrics update
- CSV: Can import 10,000+ rows
- Session state: Optimized for 100MB+ data

## 🤝 Contributing

This is a hackathon prototype. For production:
- Add Supabase/Firebase for data persistence
- Implement user authentication
- Add advanced machine learning models
- Create role-based access control
- Set up monitoring & logging

## 📝 License

MIT License - Free to use and modify

## 📞 Support

For issues or questions:
1. Check troubleshooting section above
2. Verify all dependencies are installed
3. Ensure Python 3.11+ is being used
4. Review CSV format for uploads

---

**Built with ❤️ for MSMEs | Made with Streamlit, Pandas, Prophet & Plotly**
