import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')
def display_model_status(models_dict):
    import streamlit as st
    if not models_dict:
        st.warning("⚠️ No models loaded yet.")
        return "No models"
    else:
        st.success(f"✅ {len(models_dict)} models loaded successfully.")
        return "Models loaded"
#calculat

# For importing existing models and analysis
try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    import joblib
    import pickle
    import nbformat
    from nbconvert import PythonExporter
    import importlib.util
    import sys
    import os
except ImportError as e:
    st.error(f"Required libraries not installed: {e}")

@st.cache_resource
def load_models_from_files():
    """Load pre-trained models from saved files"""
    models = {}
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Model file paths - you can modify these to match your setup
    saved_model_paths = {
        'arima_model': '../models/arima_model.pkl',
        'regression_model': '../models/regression_model.pkl', 
        'prophet_model': '../models/prophet_model.pkl',
    }
    
    # Try to load saved model files
    for model_name, path in saved_model_paths.items():
        if os.path.exists(path):
            try:
                # Use joblib for sklearn models, pickle for others
                if model_name in ['regression_model', 'scaler']:
                    try:
                        models[model_name] = joblib.load(path)
                    except:
                        with open(path, 'rb') as f:
                            models[model_name] = pickle.load(f)
                else:
                    with open(path, 'rb') as f:
                        models[model_name] = pickle.load(f)
                        
            except Exception as e:
                # Silently skip failed loads - we'll handle this in the UI
                continue
    
    return models

# Alternative function to upload models via Streamlit
def upload_models_interface():
    """Interface to upload pre-trained models"""
    st.sidebar.markdown("### 📤 Upload Your Models")
    
    uploaded_models = {}
    
    # ARIMA model upload
    arima_file = st.sidebar.file_uploader("Upload ARIMA Model", type=['pkl'], key="arima")
    if arima_file:
        try:
            uploaded_models['arima_model'] = pickle.load(arima_file)
            st.sidebar.success("✅ ARIMA model uploaded")
        except Exception as e:
            st.sidebar.error(f"Error loading ARIMA model: {e}")
    
    # Regression model upload
    reg_file = st.sidebar.file_uploader("Upload Regression Model", type=['pkl'], key="regression")
    if reg_file:
        try:
            uploaded_models['regression_model'] = pickle.load(reg_file)
            st.sidebar.success("✅ Regression model uploaded")
        except Exception as e:
            st.sidebar.error(f"Error loading regression model: {e}")
    
    # Scaler upload
    scaler_file = st.sidebar.file_uploader("Upload Scaler", type=['pkl'], key="scaler")
    if scaler_file:
        try:
            uploaded_models['scaler'] = pickle.load(scaler_file)
            st.sidebar.success("✅ Scaler uploaded")
        except Exception as e:
            st.sidebar.error(f"Error loading scaler: {e}")
    
    # Prophet model upload
    prophet_file = st.sidebar.file_uploader("Upload Prophet Model", type=['pkl'], key="prophet")
    if prophet_file:
        try:
            uploaded_models['prophet_model'] = pickle.load(prophet_file)
            st.sidebar.success("✅ Prophet model uploaded")
        except Exception as e:
            st.sidebar.error(f"Error loading Prophet model: {e}")
    
    return uploaded_models

# Function to execute notebook and extract models (if needed)
def extract_models_from_notebook(notebook_path):
    """Extract models from Jupyter notebook"""
    try:
        if not os.path.exists(notebook_path):
            st.warning(f"Notebook not found: {notebook_path}")
            return None
            
        # Read the notebook
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = nbformat.read(f, as_version=4)
        
        # Convert to Python script
        exporter = PythonExporter()
        script, _ = exporter.from_notebook_node(nb)
        
        # Execute the script in a controlled environment
        # This is a simplified approach - you might need to modify based on your notebook structure
        exec_globals = {'__name__': '__main__'}
        exec(script, exec_globals)
        
        # Extract models from the executed environment
        models = {}
        for key, value in exec_globals.items():
            if 'model' in key.lower() or 'scaler' in key.lower():
                models[key] = value
                
        return models
        
    except Exception as e:
        st.error(f"Error extracting from notebook {notebook_path}: {e}")
        return None

# Page configuration
st.set_page_config(
    page_title="Financial Forecasting Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .kpi-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .sidebar-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def calculate_technical_indicators(df):
    """Calculate various technical indicators"""
    df = df.copy()
    
    # Moving Averages
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['MA_200'] = df['Close'].rolling(window=200).mean()
    
    # Volatility (20-day rolling standard deviation)
    df['Volatility'] = df['Close'].rolling(window=20).std()
    
    # Daily Returns
    df['Daily_Return'] = df['Close'].pct_change()
    
    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['BB_Upper'] = df['MA_20'] + (df['Close'].rolling(window=20).std() * 2)
    df['BB_Lower'] = df['MA_20'] - (df['Close'].rolling(window=20).std() * 2)
    
    return df

def perform_time_series_analysis(df, target_col='Close'):
    """Perform comprehensive time series analysis"""
    
    # Stationarity Test
    result = adfuller(df[target_col].dropna())
    is_stationary = result[1] <= 0.05
    
    # Seasonal Decomposition
    if len(df) >= 24:  # Need at least 2 seasonal periods
        decomposition = seasonal_decompose(df.set_index('Date')[target_col], 
                                         model='multiplicative', period=12)
        
        return {
            'is_stationary': is_stationary,
            'adf_pvalue': result[1],
            'decomposition': decomposition
        }
    
    return {
        'is_stationary': is_stationary,
        'adf_pvalue': result[1],
        'decomposition': None
    }

def create_arima_forecast(data, periods=30, loaded_models=None):
    """Create ARIMA forecast using pre-trained model"""
    try:
        if loaded_models and 'arima_model' in loaded_models:
            # Use your pre-trained ARIMA model
            fitted_model = loaded_models['arima_model']
            st.success("✅ Using pre-trained ARIMA model")
        else:
            st.warning("⚠️ Pre-trained ARIMA model not found, creating new one...")
            # Fallback to creating new model
            from statsmodels.tsa.arima.model import ARIMA
            model = ARIMA(data, order=(1,1,1))
            fitted_model = model.fit()
        
        # Forecast
        forecast = fitted_model.forecast(steps=periods)
        forecast_dates = pd.date_range(start=data.index[-1] + timedelta(days=1), 
                                     periods=periods, freq='D')
        
        # Calculate confidence intervals
        forecast_ci = fitted_model.get_forecast(steps=periods).conf_int()
        
        return {
            'forecast': forecast,
            'dates': forecast_dates,
            'confidence_intervals': forecast_ci,
            'model_summary': fitted_model.summary(),
            'aic': getattr(fitted_model, 'aic', 'N/A'),
            'bic': getattr(fitted_model, 'bic', 'N/A'),
            'is_pretrained': 'arima_model' in (loaded_models or {})
        }
    except Exception as e:
        st.error(f"ARIMA modeling error: {e}")
        return None

def create_regression_model(df, loaded_models=None):
    """Use pre-trained regression model or create new one"""
    
    # Check if we have pre-trained models
    if loaded_models and 'regression_model' in loaded_models:
        model = loaded_models['regression_model']
        scaler = loaded_models.get('scaler', None)
        
        # Try to get feature names from the model (for sklearn models)
        if hasattr(model, 'feature_names_in_'):
            expected_features = model.feature_names_in_.tolist()
            st.info(f"Pre-trained model expects features: {expected_features}")
        else:
            # Fallback to common features - you may need to adjust this
            expected_features = ['Open', 'High', 'Low', 'Volume']
            st.warning("Could not determine expected features, using default: ['Open', 'High', 'Low', 'Volume']")
        
        # Check if all expected features exist in current dataset
        missing_features = [f for f in expected_features if f not in df.columns]
        if missing_features:
            st.error(f"Missing features required by pre-trained model: {missing_features}")
            st.info("Available columns in your dataset: " + ", ".join(df.columns.tolist()))
            return None
        
        # Use only the features the model was trained on, in the same order
        features = expected_features
        st.success("Using pre-trained regression model")
        
        # Prepare data with exact feature match
        df_clean = df[features + ['Close']].dropna()
        X = df_clean[features]
        y = df_clean['Close']
        
        # Scale features if scaler is available
        if scaler:
            try:
                X_scaled = scaler.transform(X)
            except Exception as e:
                st.error(f"Error scaling features: {e}")
                return None
        else:
            X_scaled = X
            
        # Make predictions on recent data
        try:
            y_pred = model.predict(X_scaled)
        except Exception as e:
            st.error(f"Error making predictions: {e}")
            st.info("This might be due to feature mismatch or data type issues.")
            return None
        
        # Calculate metrics on available data
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mape = np.mean(np.abs((y - y_pred) / y)) * 100
        
        # Get feature importance (assuming linear model)
        if hasattr(model, 'coef_'):
            feature_importance = dict(zip(features, model.coef_))
        else:
            feature_importance = {f: 0 for f in features}
        
        return {
            'model': model,
            'scaler': scaler,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'r2_score': model.score(X_scaled, y) if hasattr(model, 'score') else 0.0,
            'feature_importance': feature_importance,
            'is_pretrained': True,
            'features_used': features
        }
    
    else:
        # Fallback to creating new model
        st.warning("Pre-trained regression model not found, creating new one...")
        
        # Standard feature set for new model
        features = ['Open', 'High', 'Low', 'Volume']
        if not all(col in df.columns for col in features):
            missing = [col for col in features if col not in df.columns]
            st.error(f"Cannot create new model. Missing columns: {missing}")
            return None
        
        # Prepare data
        df_clean = df[features + ['Close']].dropna()
        X = df_clean[features]
        y = df_clean['Close']
        
        # Split data
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        feature_importance = dict(zip(features, model.coef_))
        return {
            'model': model,
            'scaler': scaler,
            'mae': mae,
            'rmse': rmse,
            'mape': mape,
            'r2_score': model.score(X_test_scaled, y_test),
            'feature_importance': feature_importance,
            'is_pretrained': False,
            'features_used': features
        }

def calculate_financial_metrics(df):
    
    current_price = df['Close'].iloc[-1]
    prev_price = df['Close'].iloc[-2] if len(df) > 1 else current_price
    
    daily_change = current_price - prev_price
    daily_change_pct = (daily_change / prev_price) * 100 if prev_price != 0 else 0
    
    volatility_30d = df['Close'].tail(30).std()
    volatility_annual = volatility_30d * np.sqrt(252) 
    avg_volume = df['Volume'].mean()
    current_volume = df['Volume'].iloc[-1]
    volume_ratio = current_volume / avg_volume if avg_volume != 0 else 0
    price_range = df['High'].max() - df['Low'].min()
    current_position = (current_price - df['Low'].min()) / price_range if price_range != 0 else 0
    return {
        'current_price': current_price,
        'daily_change': daily_change,
        'daily_change_pct': daily_change_pct,
        'volatility_30d': volatility_30d,
        'volatility_annual': volatility_annual,
        'avg_volume': avg_volume,
        'current_volume': current_volume,
        'volume_ratio': volume_ratio,
        'price_position': current_position
    }
    
def create_prophet_forecast(data, periods=30, loaded_models=None):
    try:
        if loaded_models and 'prophet_model' in loaded_models:
            prophet_model = loaded_models['prophet_model']
            st.success("✅ Using pre-trained Prophet model")
            future = prophet_model.make_future_dataframe(periods=periods)
            forecast = prophet_model.predict(future)
            forecast_future = forecast.tail(periods)
            
            return {
                'forecast': forecast_future['yhat'].values,
                'dates': pd.to_datetime(forecast_future['ds']),
                'upper_bound': forecast_future['yhat_upper'].values,
                'lower_bound': forecast_future['yhat_lower'].values,
                'full_forecast': forecast,
                'is_pretrained': True
            }
        else:
            st.warning("⚠️ Pre-trained Prophet model not found")
            return None
            
    except Exception as e:
        st.error(f"Prophet modeling error: {e}")
        return None

# Main Dashboard
def main():
    st.markdown('<h1 class="main-header">📈 Professional Financial Forecasting Dashboard</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    ### 🎯 Advanced Financial Analytics & Forecasting Platform
    This comprehensive dashboard provides deep insights into financial time series data with professional-grade forecasting models, 
    technical analysis, and key performance indicators for informed decision-making.
    """)
    st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.sidebar.header("📊 Dashboard Controls")
    loaded_models_from_files = load_models_from_files()
    uploaded_models = upload_models_interface()
    loaded_models = {**loaded_models_from_files, **uploaded_models}
    model_status = display_model_status(loaded_models)
    uploaded_file = st.sidebar.file_uploader("Upload your financial dataset", 
                                           type=["csv"], 
                                           help="Upload CSV file with Date, Open, Close, High, Low, Volume columns")
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date').reset_index(drop=True)
            required_columns = ['Date', 'Open', 'Close', 'High', 'Low', 'Volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.error(f"Missing required columns: {missing_columns}")
                st.stop()
            df = calculate_technical_indicators(df)
            st.sidebar.markdown("### 📅 Date Range Selection")
            date_range = st.sidebar.date_input(
                "Select date range",
                value=[df['Date'].min().date(), df['Date'].max().date()],
                min_value=df['Date'].min().date(),
                max_value=df['Date'].max().date()
            )
            
            if len(date_range) == 2:
                start_date, end_date = date_range
                mask = (df['Date'].dt.date >= start_date) & (df['Date'].dt.date <= end_date)
                df_filtered = df[mask].copy()
            else:
                df_filtered = df.copy()
            
            forecast_periods = st.sidebar.slider("Forecast Periods (days)", 7, 90, 30)
            
            st.sidebar.markdown('</div>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            metrics = calculate_financial_metrics(df_filtered)
            with col1:
                st.markdown(f"""
                <div class="kpi-card">
                    <h3>Current Price</h3>
                    <h2>${metrics['current_price']:.2f}</h2>
                    <p>Daily Change: {metrics['daily_change']:+.2f} ({metrics['daily_change_pct']:+.2f}%)</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="kpi-card">
                    <h3>30D Volatility</h3>
                    <h2>{metrics['volatility_30d']:.2f}</h2>
                    <p>Annualized: {metrics['volatility_annual']:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="kpi-card">
                    <h3>Volume Ratio</h3>
                    <h2>{metrics['volume_ratio']:.2f}x</h2>
                    <p>vs Average Volume</p>
                </div>
                """, unsafe_allow_html=True)
            
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Price Analysis", "🔮 Forecasting", "📊 Technical Analysis", "📋 Statistics", "📤 Export"])
            
            with tab1:
                st.subheader("📈 Comprehensive Price Analysis")
                
                fig = make_subplots(rows=2, cols=1, 
                                  shared_xaxes=True,
                                  vertical_spacing=0.1,
                                  subplot_titles=('Price Movement with Technical Indicators', 'Volume Analysis'),
                                  row_heights=[0.7, 0.3])
                
                fig.add_trace(go.Candlestick(
                    x=df_filtered['Date'],
                    open=df_filtered['Open'],
                    high=df_filtered['High'],
                    low=df_filtered['Low'],
                    close=df_filtered['Close'],
                    name="Price"
                ), row=1, col=1)
                
                fig.add_trace(go.Scatter(
                    x=df_filtered['Date'],
                    y=df_filtered['MA_20'],
                    name='MA 20',
                    line=dict(color='orange', width=1)
                ), row=1, col=1)
                
                fig.add_trace(go.Scatter(
                    x=df_filtered['Date'],
                    y=df_filtered['MA_50'],
                    name='MA 50',
                    line=dict(color='blue', width=1)
                ), row=1, col=1)
                
                # Volume
                fig.add_trace(go.Bar(
                    x=df_filtered['Date'],
                    y=df_filtered['Volume'],
                    name='Volume',
                    marker_color='lightblue'
                ), row=2, col=1)
                
                fig.update_layout(height=600, showlegend=True, xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # Price statistics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Price Statistics")
                    price_stats = df_filtered['Close'].describe()
                    st.dataframe(price_stats, use_container_width=True)
                
                with col2:
                    st.subheader("📈 Returns Distribution")
                    returns_fig = px.histogram(df_filtered, x='Daily_Return', 
                                             title='Daily Returns Distribution',
                                             nbins=30)
                    returns_fig.update_layout(height=300)
                    st.plotly_chart(returns_fig, use_container_width=True)
            
            with tab2:
                st.subheader("🔮 Advanced Forecasting Models")
                
                # Initialize variables to avoid UnboundLocalError
                arima_results = None
                reg_results = None
                prophet_results = None
                
                if st.button("🔮 Generate Forecasts", type="primary", help="Generate forecasts using available models"):
                    with st.spinner("Running forecasting models..."):
                        
                        # Time series analysis
                        ts_data = df_filtered.set_index('Date')['Close']
                        ts_analysis = perform_time_series_analysis(df_filtered)
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown("### 📈 ARIMA Forecast")
                            
                            # ARIMA forecast using pre-trained model
                            arima_results = create_arima_forecast(ts_data, forecast_periods, loaded_models)
                            
                            if arima_results:
                                # Plot forecast
                                fig_forecast = go.Figure()
                                
                                # Historical data
                                fig_forecast.add_trace(go.Scatter(
                                    x=ts_data.index[-60:],  # Last 60 days
                                    y=ts_data.values[-60:],
                                    name='Historical',
                                    line=dict(color='blue')
                                ))
                                
                                # Forecast
                                fig_forecast.add_trace(go.Scatter(
                                    x=arima_results['dates'],
                                    y=arima_results['forecast'],
                                    name='Forecast',
                                    line=dict(color='red', dash='dash')
                                ))
                                
                                # Confidence intervals
                                fig_forecast.add_trace(go.Scatter(
                                    x=arima_results['dates'],
                                    y=arima_results['confidence_intervals'].iloc[:, 0],
                                    fill=None,
                                    mode='lines',
                                    line_color='rgba(0,0,0,0)',
                                    showlegend=False
                                ))
                                
                                fig_forecast.add_trace(go.Scatter(
                                    x=arima_results['dates'],
                                    y=arima_results['confidence_intervals'].iloc[:, 1],
                                    fill='tonexty',
                                    mode='lines',
                                    line_color='rgba(0,0,0,0)',
                                    name='Confidence Interval',
                                    fillcolor='rgba(255,0,0,0.1)'
                                ))
                                
                                fig_forecast.update_layout(title='ARIMA Forecast', height=400)
                                st.plotly_chart(fig_forecast, use_container_width=True)
                                
                                # Model metrics
                                st.markdown("**Model Performance:**")
                                st.write(f"- AIC: {arima_results['aic']}")
                                st.write(f"- BIC: {arima_results['bic']}")
                                if arima_results['is_pretrained']:
                                    st.success("✅ Using your pre-trained ARIMA model")
                            else:
                                st.warning("ARIMA model could not be generated")
                        
                        with col2:
                            st.markdown("### 🤖 Regression Model")
                            
                            # Regression model using pre-trained model
                            reg_results = create_regression_model(df_filtered, loaded_models)
                            
                            if reg_results:
                                st.markdown("**Model Performance:**")
                                st.write(f"- MAE: {reg_results['mae']:.2f}")
                                st.write(f"- RMSE: {reg_results['rmse']:.2f}")
                                st.write(f"- MAPE: {reg_results['mape']:.2f}%")
                                st.write(f"- R² Score: {reg_results['r2_score']:.3f}")
                                if reg_results['is_pretrained']:
                                    st.success("✅ Using your pre-trained regression model")
                                
                                st.markdown("**Feature Importance:**")
                                importance_df = pd.DataFrame({
                                    'Feature': list(reg_results['feature_importance'].keys()),
                                    'Importance': list(reg_results['feature_importance'].values())
                                })
                                
                                fig_importance = px.bar(importance_df, x='Importance', y='Feature',
                                                    orientation='h', title='Feature Importance')
                                fig_importance.update_layout(height=300)
                                st.plotly_chart(fig_importance, use_container_width=True)
                            else:
                                st.warning("Regression model could not be generated")
                        
                        with col3:
                            st.markdown("### 🔮 Prophet Forecast")
                            
                            # Prophet forecast using pre-trained model
                            prophet_results = create_prophet_forecast(ts_data, forecast_periods, loaded_models)
                            
                            if prophet_results:
                                # Plot Prophet forecast
                                fig_prophet = go.Figure()
                                
                                # Historical data
                                fig_prophet.add_trace(go.Scatter(
                                    x=ts_data.index[-60:],  # Last 60 days
                                    y=ts_data.values[-60:],
                                    name='Historical',
                                    line=dict(color='blue')
                                ))
                                
                                # Forecast
                                fig_prophet.add_trace(go.Scatter(
                                    x=prophet_results['dates'],
                                    y=prophet_results['forecast'],
                                    name='Prophet Forecast',
                                    line=dict(color='green', dash='dash')
                                ))
                                
                                # Confidence intervals
                                fig_prophet.add_trace(go.Scatter(
                                    x=prophet_results['dates'],
                                    y=prophet_results['lower_bound'],
                                    fill=None,
                                    mode='lines',
                                    line_color='rgba(0,0,0,0)',
                                    showlegend=False
                                ))
                                
                                fig_prophet.add_trace(go.Scatter(
                                    x=prophet_results['dates'],
                                    y=prophet_results['upper_bound'],
                                    fill='tonexty',
                                    mode='lines',
                                    line_color='rgba(0,0,0,0)',
                                    name='Confidence Interval',
                                    fillcolor='rgba(0,255,0,0.1)'
                                ))
                                
                                fig_prophet.update_layout(title='Prophet Forecast', height=400)
                                st.plotly_chart(fig_prophet, use_container_width=True)
                                
                                st.success("✅ Using your pre-trained Prophet model")
                            else:
                                st.info("💡 Prophet model not available. Upload your model or use demo forecasting.")
                
                # Model comparison section - now inside the same scope
                st.markdown("### 📊 Model Comparison")
                
                # Create a comparison if multiple models are available
                comparison_data = []
                if arima_results:
                    comparison_data.append({
                        'Model': 'ARIMA',
                        'Status': '✅ Loaded' if arima_results['is_pretrained'] else '🔄 Generated',
                        'AIC': arima_results.get('aic', 'N/A'),
                        'BIC': arima_results.get('bic', 'N/A')
                    })
                
                if reg_results:
                    comparison_data.append({
                        'Model': 'Regression',
                        'Status': '✅ Loaded' if reg_results['is_pretrained'] else '🔄 Generated',
                        'MAE': f"{reg_results['mae']:.2f}",
                        'RMSE': f"{reg_results['rmse']:.2f}",
                        'R²': f"{reg_results['r2_score']:.3f}"
                    })
                
                if prophet_results:
                    comparison_data.append({
                        'Model': 'Prophet',
                        'Status': '✅ Loaded' if prophet_results['is_pretrained'] else '🔄 Generated'
                    })
                
                if comparison_data:
                    comparison_df = pd.DataFrame(comparison_data)
                    st.dataframe(comparison_df, use_container_width=True)
                else:
                    st.info("💡 Click 'Generate Forecasts' to see model comparisons.")
                
                # Time series decomposition - also moved inside button scope
                if st.button("Show Time Series Decomposition"):
                    ts_analysis = perform_time_series_analysis(df_filtered)
                    if ts_analysis['decomposition'] is not None:
                        st.markdown("### 📊 Time Series Decomposition")
                        
                        decomp = ts_analysis['decomposition']
                        
                        fig_decomp = make_subplots(rows=4, cols=1,
                                                subplot_titles=['Original', 'Trend', 'Seasonal', 'Residual'])
                        
                        fig_decomp.add_trace(go.Scatter(x=decomp.observed.index, y=decomp.observed.values,
                                                    name='Original'), row=1, col=1)
                        fig_decomp.add_trace(go.Scatter(x=decomp.trend.index, y=decomp.trend.values,
                                                    name='Trend'), row=2, col=1)
                        fig_decomp.add_trace(go.Scatter(x=decomp.seasonal.index, y=decomp.seasonal.values,
                                                    name='Seasonal'), row=3, col=1)
                        fig_decomp.add_trace(go.Scatter(x=decomp.resid.index, y=decomp.resid.values,
                                                    name='Residual'), row=4, col=1)
                        
                        fig_decomp.update_layout(height=800, showlegend=False)
                        st.plotly_chart(fig_decomp, use_container_width=True)
                    else:
                        st.warning("Time series decomposition requires at least 24 data points.")
            with tab3:
                st.subheader("📊 Technical Analysis Dashboard")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # RSI Chart
                    fig_rsi = go.Figure()
                    fig_rsi.add_trace(go.Scatter(x=df_filtered['Date'], y=df_filtered['RSI'],
                                               name='RSI', line=dict(color='purple')))
                    fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", 
                                     annotation_text="Overbought (70)")
                    fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", 
                                     annotation_text="Oversold (30)")
                    fig_rsi.update_layout(title='RSI (Relative Strength Index)', height=300)
                    st.plotly_chart(fig_rsi, use_container_width=True)
                
                with col2:
                    # Volatility Chart
                    fig_vol = px.line(df_filtered, x='Date', y='Volatility',
                                     title='20-Day Rolling Volatility')
                    fig_vol.update_layout(height=300)
                    st.plotly_chart(fig_vol, use_container_width=True)
                
                # Bollinger Bands
                fig_bb = go.Figure()
                fig_bb.add_trace(go.Scatter(x=df_filtered['Date'], y=df_filtered['Close'],
                                          name='Close Price', line=dict(color='blue')))
                fig_bb.add_trace(go.Scatter(x=df_filtered['Date'], y=df_filtered['BB_Upper'],
                                          name='Upper Band', line=dict(color='red', dash='dash')))
                fig_bb.add_trace(go.Scatter(x=df_filtered['Date'], y=df_filtered['MA_20'],
                                          name='Middle Band (MA20)', line=dict(color='orange')))
                fig_bb.add_trace(go.Scatter(x=df_filtered['Date'], y=df_filtered['BB_Lower'],
                                          name='Lower Band', line=dict(color='green', dash='dash')))
                
                fig_bb.update_layout(title='Bollinger Bands Analysis', height=400)
                st.plotly_chart(fig_bb, use_container_width=True)
            
            with tab4:
                st.subheader("📋 Comprehensive Statistics")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 📈 Price Metrics")
                    price_metrics = {
                        'Current Price': f"${metrics['current_price']:.2f}",
                        'Daily Change': f"{metrics['daily_change']:+.2f} ({metrics['daily_change_pct']:+.2f}%)",
                        '52-Week High': f"${df['High'].max():.2f}",
                        '52-Week Low': f"${df['Low'].min():.2f}",
                        'Price Position': f"{metrics['price_position']:.1%}",
                        '30-Day Volatility': f"{metrics['volatility_30d']:.2f}",
                        'Annualized Volatility': f"{metrics['volatility_annual']:.2f}"
                    }
                    
                    for metric, value in price_metrics.items():
                        st.markdown(f"**{metric}:** {value}")
                
                with col2:
                    st.markdown("### 📊 Volume Metrics")
                    volume_metrics = {
                        'Current Volume': f"{metrics['avg_volume']:,.0f}",
                        'Average Volume': f"{df_filtered['Volume'].iloc[-1]:,.0f}",
                        'Volume Ratio': f"{metrics['volume_ratio']:.2f}x",
                        'Max Volume': f"{df['Volume'].max():,.0f}",
                        'Min Volume': f"{df['Volume'].min():,.0f}"
                    }
                    
                    for metric, value in volume_metrics.items():
                        st.markdown(f"**{metric}:** {value}")
                
                # Correlation Matrix
                st.markdown("### 🔗 Correlation Analysis")
                corr_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Daily_Return', 'Volatility']
                corr_data = df_filtered[corr_cols].corr()
                
                fig_corr = px.imshow(corr_data, 
                                   title='Correlation Matrix',
                                   color_continuous_scale='RdBu',
                                   text_auto=True)
                fig_corr.update_layout(height=500)
                st.plotly_chart(fig_corr, use_container_width=True)
            
            with tab5:
                st.subheader("📤 Export Data & Reports")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 💾 Download Data")
                    
                    # Export enhanced dataset
                    csv_data = df_filtered.to_csv(index=False)
                    st.download_button(
                        label="📊 Download Enhanced Dataset (CSV)",
                        data=csv_data,
                        file_name=f"financial_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                    
                    # Export summary statistics
                    summary_stats = df_filtered[['Open', 'High', 'Low', 'Close', 'Volume']].describe()
                    summary_csv = summary_stats.to_csv()
                    st.download_button(
                        label="📈 Download Summary Statistics (CSV)",
                        data=summary_csv,
                        file_name=f"summary_stats_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    st.markdown("### 📋 Generate Report")
                    
                    if st.button("Generate Analysis Report", type="primary"):
                        # Create comprehensive report
                        report = f"""
# Financial Analysis Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
- **Current Price**: ${metrics['current_price']:.2f}
- **Daily Change**: {metrics['daily_change']:+.2f} ({metrics['daily_change_pct']:+.2f}%)
- **30-Day Volatility**: {metrics['volatility_30d']:.2f}
- **Volume Ratio**: {metrics['volume_ratio']:.2f}x average

## Key Metrics
- **52-Week High**: ${df['High'].max():.2f}
- **52-Week Low**: ${df['Low'].min():.2f}
- **Price Position**: {metrics['price_position']:.1%}
- **Annualized Volatility**: {metrics['volatility_annual']:.2f}

## Data Summary
- **Total Records**: {len(df_filtered):,}
- **Date Range**: {df_filtered['Date'].min().strftime('%Y-%m-%d')} to {df_filtered['Date'].max().strftime('%Y-%m-%d')}
- **Average Daily Volume**: {df_filtered['Volume'].mean():,.0f}

## Technical Indicators
- **Current RSI**: {df_filtered['RSI'].iloc[-1]:.2f}
- **20-Day MA**: ${df_filtered['MA_20'].iloc[-1]:.2f}
- **50-Day MA**: ${df_filtered['MA_50'].iloc[-1]:.2f}

## Risk Assessment
- **Volatility Level**: {"High" if metrics['volatility_30d'] > df['Close'].std() else "Normal"}
- **Price Trend**: {"Bullish" if metrics['daily_change'] > 0 else "Bearish"}

---
*This report was generated using advanced financial analytics and should be used for informational purposes only.*
                        """
                        
                        st.download_button(
                            label="📄 Download Analysis Report (TXT)",
                            data=report,
                            file_name=f"financial_report_{datetime.now().strftime('%Y%m%d')}.txt",
                            mime="text/plain"
                        )
                
                # Data preview
                st.markdown("### 👀 Data Preview")
                st.dataframe(df_filtered.tail(10), use_container_width=True)
        
        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.info("Please ensure your CSV file contains the required columns: Date, Open, Close, High, Low, Volume")
    
    else:
        st.info("👆 Please upload your financial dataset to begin analysis")
        
        # Sample data format
        st.markdown("### 📋 Expected Data Format")
        sample_data = {
            'Date': ['2024-01-01', '2024-01-02', '2024-01-03'],
            'Open': [150.00, 152.50, 151.20],
            'Close': [152.00, 151.80, 153.40],
            'High': [153.00, 153.20, 154.00],
            'Low': [149.50, 151.00, 150.80],
            'Volume': [1000000, 1200000, 980000]
        }
        
        sample_df = pd.DataFrame(sample_data)
        st.dataframe(sample_df, use_container_width=True)

if __name__ == "__main__":
    main()