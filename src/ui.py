import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.interpolate import RectBivariateSpline, interp1d, SmoothBivariateSpline, Rbf, griddata
import warnings
import io
from src.option_pricing import calculate_option_prices, enhanced_local_volatility_surface, implied_volatility
from src.greeks import calculate_greeks,  calculate_advanced_greeks
from src.strategy import calculate_strategy_pnl, calculate_strategy_greeks, calculate_strategy_performance, var_calculator, stress_test_portfolio, risk_scenario_analysis
from src.visualization import create_strategy_visualization,analyze_vol_surface

# Page Configuration
st.set_page_config(
    page_title="Options Pricing",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)
    
# Display functions for the UI
def display_option_prices(price_info):
    """Display option prices in a clean format"""
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
            <div style="background-color: #077B09; padding: 5px; border-radius: 8px; text-align: center;">
                <h4 style="color: white; margin: 3px;">CALL Value</h4>
                <h3 style="color: white; margin: 3px 0;">{price_info['call']}</h3>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div style="background-color: #F1592A; padding: 5px; border-radius: 8px; text-align: center;">
                <h4 style="color: white; margin: 3px;">PUT Value</h4>
                <h3 style="color: white; margin: 3px 0;">{price_info['put']}</h3>
            </div>
        """, unsafe_allow_html=True)

def display_greeks(calculated_greeks):
    """Display Greeks in a minimal grid layout"""
    st.markdown(f"""
        <div style="background-color: #1E1E1E; padding: 6px; border-radius: 10px;">
            <h4 style="color: white; margin-bottom: -1rem; margin-left: 1rem;">Position Greeks</h4>
            <div style="display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 8px;">
                <div class="greek-card">
                    <div class="greek-label">Delta</div>
                    <div class="greek-value">{round(calculated_greeks['delta'], 3)}</div>
                </div>
                <div class="greek-card">
                    <div class="greek-label">Gamma</div>
                    <div class="greek-value">{round(calculated_greeks['gamma'], 3)}</div>
                </div>
                <div class="greek-card">
                    <div class="greek-label">Theta</div>
                    <div class="greek-value">{round(calculated_greeks['theta'], 3)}</div>
                </div>
                <div class="greek-card">
                    <div class="greek-label">Vega</div>
                    <div class="greek-value">{round(calculated_greeks['vega'], 3)}</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

# Setup sidebar inputs
def setup_sidebar():
    # ✅ Inject CSS to increase font size and spacing in sidebar
    st.sidebar.markdown("""
        <style>
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] input,
        section[data-testid="stSidebar"] select,
        section[data-testid="stSidebar"] div[data-baseweb="select"],
        section[data-testid="stSidebar"] .stNumberInput,
        section[data-testid="stSidebar"] .stCheckbox,
        section[data-testid="stSidebar"] span {
            font-size: 1.2rem !important;
        }

        /* Increase font size for titles and subtitles */
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            font-size: 2rem !important;}
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] label {
            font-size: 1.15rem !important;
            //font-weight: 600 !important;
        }

        /* Increase spacing between widgets */
        section[data-testid="stSidebar"] .stElementContainer {
            margin-bottom: 0.15rem !important;
        </style>
    """, unsafe_allow_html=True)

    st.sidebar.markdown("# Model Construction")
    model_type = st.sidebar.selectbox(
        "Model Type",
        ["Black-Scholes", "Binomial", "Monte Carlo"],
        index=0
    )
    
    # Basic input parameters
    current_price = st.sidebar.number_input("Current Asset Price", value=100.00, step=0.01, format="%.2f")
    strike_price = st.sidebar.number_input("Strike Price", value=100.00, step=0.01, format="%.2f")
    time_to_maturity = st.sidebar.number_input("Time to Maturity (Years)", value=1.00, step=0.01, format="%.2f")
    volatility = st.sidebar.number_input("Volatility (σ)", value=0.20, step=0.01, format="%.2f")
    risk_free_rate = st.sidebar.number_input("Risk-Free Interest Rate", value=0.05, step=0.01, format="%.2f")
    
    # Model-specific parameters
    model_params = {}
    if model_type == "Binomial":
        model_params['steps'] = st.sidebar.slider("Number of Steps", 10, 1000, 100)
        model_params['option_style'] = st.sidebar.selectbox("Option Style", ["European", "American"])
    elif model_type == "Monte Carlo":
        model_params['n_simulations'] = st.sidebar.slider("Number of Simulations", 1000, 50000, 10000)
        model_params['n_steps'] = st.sidebar.slider("Time Steps", 50, 500, 100)
    
    # Advanced options
    if st.sidebar.checkbox("Advanced Market Parameters", False):
        st.sidebar.markdown("## Advanced Parameters")
        
        # Volatility term structure
        vol_term_structure = st.sidebar.checkbox("Use Volatility Term Structure", False)
        if vol_term_structure:
            vol_3m = st.sidebar.number_input("3-Month Volatility", value=volatility*0.9, step=0.01, format="%.2f")
            vol_6m = st.sidebar.number_input("6-Month Volatility", value=volatility, step=0.01, format="%.2f")
            vol_12m = st.sidebar.number_input("12-Month Volatility", value=volatility*1.1, step=0.01, format="%.2f")
            model_params['vol_term_structure'] = {
                0.25: vol_3m,
                0.5: vol_6m,
                1.0: vol_12m
            }
        
        # Interest rate term structure
        rate_term_structure = st.sidebar.checkbox("Use Rate Term Structure", False)
        if rate_term_structure:
            rate_3m = st.sidebar.number_input("3-Month Rate", value=risk_free_rate*0.8, step=0.001, format="%.3f")
            rate_6m = st.sidebar.number_input("6-Month Rate", value=risk_free_rate, step=0.001, format="%.3f")
            rate_12m = st.sidebar.number_input("12-Month Rate", value=risk_free_rate*1.2, step=0.001, format="%.3f")
            model_params['rate_term_structure'] = {
                0.25: rate_3m,
                0.5: rate_6m,
                1.0: rate_12m
            }
        
        # Dividend yield
        div_yield = st.sidebar.number_input("Dividend Yield", value=0.0, step=0.001, format="%.3f", key="div_yield_slider")
        if div_yield > 0:
            model_params['dividend_yield'] = div_yield
        
        # Market skew parameter
        skew = st.sidebar.slider("Volatility Skew", -0.2, 0.2, 0.0, 0.01, key="vol_skew_slider")
        if skew != 0:
            model_params['skew'] = skew
    
    return model_type, current_price, strike_price, time_to_maturity, volatility, risk_free_rate, model_params

def app():
    """Main application execution flow"""
    # Import all necessary libraries at the top level
    warnings.filterwarnings('ignore')
    
    # Custom Sidebar and Tabs CSS styles
    st.markdown("""
    <style>
    /* Main page upward, reduce the blanks */
    div[data-testid="stAppViewContainer"] {
    margin-top: -4rem !important;
    }
    /* Adjust Sidebar width */
    section[data-testid="stSidebar"] {
        width: 500px !important;
    }
    div[data-testid="stSidebarContent"] {
        padding: 8rem 1rem;
    }

    /* Adjust Tabs font size and padding */
    /* Tabs distance from the above */
    div[data-testid="stTabs"] {
    margin-top: 20px !important;
    }
    button[data-baseweb="tab"] {
    padding: 25px 20px !important;
    }
    button[data-baseweb="tab"] p {
    font-size: 20px !important;
    font-weight: 600 !important;
    margin: 5px !important;
    }
    </style>
    """, unsafe_allow_html=True)

    # CSS Styles
    st.markdown("""
        <style>
        .greek-card {
            background-color: #2E2E2E;
            padding: 1rem;
            border-radius: 1rem;
            margin: 0.8rem;
        }
        .greek-label { color: #9CA3AF; font-size: 1.15rem; font-weight: 500; }
        .greek-value { color: white; font-size: 1.3rem; font-weight: 600; }
        .main { background-color: #0E1117; }
        </style>
    """, unsafe_allow_html=True)

    # Author Section
    st.markdown("""
        <div style="background-color: #1E2124; padding: 6px; border-radius: 6px; width: 150px; margin-bottom:10px;">
            <div style="color: #9CA3AF; font-size: 15px; margin-bottom: 6px;">Created by</div>
            <div style="display: flex; align-items: center;">
                <div style="margin-right: 15px;">
                    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20">
                        <path fill="#0A66C2" d="M20.5 2h-17A1.5 1.5 0 002 3.5v17A1.5 1.5 0 003.5 22h17a1.5 1.5 0 001.5-1.5v-17A1.5 1.5 0 0020.5 2zM8 19H5v-9h3zM6.5 8.25A1.75 1.75 0 118.3 6.5a1.78 1.78 0 01-1.8 1.75zM19 19h-3v-4.74c0-1.42-.6-1.93-1.38-1.93A1.74 1.74 0 0013 14.19a.66.66 0 00.1.4V19h-3v-9h2.9v1.3a3.11 3.11 0 012.7-1.4c1.55 0 3.36.86 3.36 3.66z"></path>
                    </svg>
                </div>
                <a href="https://www.linkedin.com/in/yiyang-xu-wq" target="_blank" style="color: white; text-decoration: none; font-size: 20px; font-weight: 500;">Yiyang Xu</a>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    try:
        # Setup sidebar and get parameters
        model_type, current_price, strike_price, time_to_maturity, volatility, risk_free_rate, model_params = setup_sidebar()
        
        # Title and model selection display
        st.markdown(f"## {model_type} Option Pricing Model")
        
        # Calculate prices based on selected model
        with st.spinner("Calculating prices..."):
            price_info, call_value, put_value, paths_data = calculate_option_prices(
                model_type, current_price, strike_price, time_to_maturity,
                risk_free_rate, volatility, model_params
            )

        # Display option prices
        display_option_prices(price_info)
        
        # Add tabs for different functionalities
        main_tab, strategy_tab, quant_tab = st.tabs(["Basic Analysis", "Strategy Analysis", "Quant Research"])
        
        with main_tab:
            # Display Greeks
            calculated_greeks = calculate_greeks("Call", current_price, strike_price,
                                             time_to_maturity, risk_free_rate, volatility)
            display_greeks(calculated_greeks)
            
            # Display advanced Greeks if requested
            if st.checkbox("Show Advanced Greeks"):
                advanced_greeks = calculate_advanced_greeks("Call", current_price, strike_price,
                                                        time_to_maturity, risk_free_rate, volatility)
                st.markdown("### Advanced Greeks")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                        <div style="background-color: #1E1E1E; padding: 30px; border-radius: 10px;">
                            <h4 style="color: white;">Second-Order Greeks</h4>
                            <ul style="color: white; list-style-type: none; padding-left: 0; font-size: 1.15rem;">
                                <li>• Vanna: {advanced_greeks['vanna']:.4f} (Delta-Vega Sensitivity)</li>
                                <li>• Charm: {advanced_greeks['charm']:.4f} (Delta Decay)</li>
                                <li>• Volga: {advanced_greeks['volga']:.4f} (Vega Convexity)</li>
                                <li>• Veta: {advanced_greeks['veta']:.4f} (Vega Decay)</li>
                            </ul>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                        <div style="background-color: #1E1E1E; padding: 30px; border-radius: 10px;">
                            <h4 style="color: white;">Third-Order Greeks</h4>
                            <ul style="color: white; list-style-type: none; padding-left: 0; font-size: 1.15rem;">
                                <li>• Speed: {advanced_greeks['speed']:.4f} (Delta Acceleration)</li>
                                <li>• Zomma: {advanced_greeks['zomma']:.4f} (Gamma-Volga)</li>
                                <li>• Color: {advanced_greeks['color']:.4f} (Gamma Decay)</li>
                                <li>• Ultima: {advanced_greeks['ultima']:.4f} (Volga-Volga)</li>
                            </ul>
                        </div>
                    """, unsafe_allow_html=True)
            
            # Display Monte Carlo specific visualizations if selected
            if model_type == "Monte Carlo" and paths_data:
                call_paths, put_paths, n_steps = paths_data

                st.markdown("### Monte Carlo Simulation Paths")
                fig = plt.figure(figsize=(10, 6))

                path_shape = call_paths[:100].T.shape
                time_points = path_shape[0]

                time_array = np.linspace(0, time_to_maturity, time_points)

                plt.plot(time_array, call_paths[:100].T, alpha=0.1)
                mean_path = np.mean(call_paths, axis=0)
                plt.plot(time_array, mean_path, 'r', linewidth=2)

                plt.xlabel('Time (years)')
                plt.ylabel('Stock Price')
                plt.title('Monte Carlo Simulation Paths (first 100 paths)')
                st.pyplot(fig)

                # Added Value-at-Risk calculation option
                if st.checkbox("Calculate Value-at-Risk"):
                    confidence = st.slider("Confidence Level", 0.9, 0.99, 0.95, 0.01)
                    horizon = st.slider("Risk Horizon (days)", 1, 30, 1) / 252

                    # Add enhanced VaR options
                    use_t_dist = st.checkbox("Use Student's t-distribution (fat tails)", True,
                                           help="Better captures extreme market events than normal distribution")
                    if use_t_dist:
                        degrees_of_freedom = st.slider("Degrees of Freedom", 3, 10, 5, 1,
                                                     help="Lower values create fatter tails (3-5 for financial markets)")
                    else:
                        degrees_of_freedom = 5

                    use_garch = st.checkbox("Use GARCH volatility modeling", False,
                                          help="Models time-varying volatility instead of constant volatility")

                    # Add stress testing option
                    stress_test_tab = st.checkbox("Include Stress Testing", False,
                                               help="Test portfolio against historical crisis scenarios")

                    # Enhanced VaR calculation with new parameters
                    var_results = var_calculator(
                        strategies=["Long Call"],
                        quantities=[1],
                        spot_price=current_price,
                        strikes=[strike_price],
                        maturities=[time_to_maturity],
                        rates=risk_free_rate,
                        vols=volatility,
                        confidence=confidence,
                        horizon=horizon,
                        n_simulations=10000,
                        use_t_dist=use_t_dist,
                        degrees_of_freedom=degrees_of_freedom,
                        use_garch=use_garch
                    )

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"""
                            <div style="background-color: #1E1E1E; padding: 30px; border-radius: 10px;">
                                <h3 style="color: white;">Value-at-Risk Metrics</h3>
                                <ul style="color: white; list-style-type: none; padding-left: 0; font-size: 1.15rem;">
                                    <li>• VaR ({confidence*100:.1f}%): ${var_results['VaR']:.2f}</li>
                                    <li>• Expected Shortfall: ${var_results['Expected_Shortfall']:.2f}</li>
                                    <li>• Horizon: {var_results['Horizon_Days']:.0f} days</li>
                                    <li>• Distribution: {var_results['Distribution']}</li>
                                    <li>• Volatility Model: {var_results['Volatility_Model']}</li>
                                </ul>
                            </div>
                        """, unsafe_allow_html=True)

                    with col2:
                        st.markdown(f"""
                            <div style="background-color: #1E1E1E; padding: 30px; border-radius: 10px;">
                                <h3 style="color: white;">Risk Distribution</h3>
                                <ul style="color: white; list-style-type: none; padding-left: 0; font-size: 1.15rem;">
                                    <li>• Volatility: ${var_results['Volatility']:.2f}</li>
                                    <li>• Skewness: {var_results['Skewness']:.2f}</li>
                                    <li>• Kurtosis: {var_results['Kurtosis']:.2f}</li>
                                    <li>• Worst Case: ${var_results['Worst_Case']:.2f}</li>
                                    <li>• Best Case: ${var_results['Best_Case']:.2f}</li>
                                </ul>
                            </div>
                        """, unsafe_allow_html=True)
                        
                    # Add stress testing results if enabled
                    if stress_test_tab:
                        st.markdown("### Stress Test Results")
                        with st.spinner("Running stress tests..."):
                            # Correct input values for stress_test_portfolio function
                            stress_results = stress_test_portfolio(
                                strategies=["Long Call"],  # Must be a list, not a string
                                quantities=[1],             # Must be a list, not a single integer
                                spot_price=current_price,   # Single float value
                                strikes=[strike_price],     # Must be a list
                                maturities=[time_to_maturity], # Must be a list
                                rates=risk_free_rate,       # Single float value
                                vols=volatility             # Single float value
                            )
                            # Add the new error handling code here:
                            try:
                                # Create DataFrame from results
                                stress_data = []
                                for scenario, results in stress_results.items():
                                    try:
                                        # Handle if results is a dictionary with expected structure
                                        if isinstance(results, dict) and "P&L" in results:
                                            scenario_details = "N/A"
                                            if "Applied_Scenario" in results:
                                                scenario_details = f"{results['Applied_Scenario']['Price Change']} price, {results['Applied_Scenario']['Volatility Multiplier']} vol, {results['Applied_Scenario']['Rate Change']} rate"
                                            
                                            stress_data.append({
                                                "Scenario": scenario,
                                                "P&L": results["P&L"],
                                                "% Change": results.get("Portfolio_Change_Pct", 0.0),
                                                "Scenario Details": scenario_details
                                            })
                                        else:
                                            # Handle if results is just a float
                                            stress_data.append({
                                                "Scenario": scenario,
                                                "P&L": float(results) if isinstance(results, (int, float)) else 0.0,
                                                "% Change": 0.0,
                                                "Scenario Details": "N/A"
                                            })
                                    except Exception as e:
                                        st.warning(f"Error processing scenario {scenario}: {str(e)}")
                                
                                stress_df = pd.DataFrame(stress_data)
                                
                                # Display results as a table
                                st.dataframe(stress_df.style.format({
                                    "P&L": "${:.2f}",
                                    "% Change": "{:.2f}%"
                                }))
                                
                                # Create bar chart of stress test results
                                fig = plt.figure(figsize=(12, 7))
                                ax = plt.gca()
                                bars = plt.bar(stress_df["Scenario"], stress_df["P&L"],
                                       color=['#FF5555' if x < 0 else '#55CC55' for x in stress_df["P&L"]],
                                       width=0.7)
                                plt.axhline(y=0, color='white', linestyle='-', alpha=0.3)
                                plt.ylabel('P&L ($)', fontsize=12)
                                plt.xlabel('Scenario', fontsize=12)
                                plt.title('Stress Test Results', fontsize=14, fontweight='bold')
                                plt.xticks(rotation=30, ha='right', fontsize=10)
                                plt.grid(axis='y', alpha=0.3)
                                plt.tight_layout(pad=2)

                                # Add values on top of bars with better positioning
                                for bar in bars:
                                    height = bar.get_height()
                                    y_pos = min(-0.7, height - 0.5) if height < 0 else max(0.3, height + 0.5)
                                    plt.text(bar.get_x() + bar.get_width()/2, y_pos,
                                           f'${height:.2f}',
                                           ha='center',
                                           va='bottom' if height >= 0 else 'top',
                                           color='white',
                                           fontweight='bold',
                                           fontsize=11)
                                    
                                # Add more whitespace at the bottom for labels
                                plt.subplots_adjust(bottom=0.15)
                                    
                                st.pyplot(fig)
                            except Exception as e:
                                st.error(f"Error processing stress test results: {str(e)}")
        
        with strategy_tab:
            # Define strategy categories
            call_strategies = ["Covered Call Writing", "Long Call", "Bull Call Spread", "Bear Call Spread"]
            put_strategies = ["Long Put", "Protective Put", "Bull Put Spread", "Bear Put Spread"]
            combined_strategies = ["Long Straddle", "Short Straddle", "Iron Butterfly", "Iron Condor"]
            
            st.markdown("### Options Strategy Analysis")
            
            col1, col2 = st.columns([1, 2])
            with col1:
                strategy_category = st.selectbox("Select Strategy Category",
                                              ["Call Option Strategies", "Put Option Strategies", "Combined Strategies"],
                                              key="strategy_category")
            
            with col2:
                if strategy_category == "Call Option Strategies":
                    strategy = st.selectbox("Select Strategy", call_strategies, key="call_strategy")
                elif strategy_category == "Put Option Strategies":
                    strategy = st.selectbox("Select Strategy", put_strategies, key="put_strategy")
                else:
                    strategy = st.selectbox("Select Strategy", combined_strategies, key="combined_strategy")
            
            # Strategy explanation based on selection
            strategy_explanations = {
                "Covered Call Writing": "Own the stock and sell a call option. This strategy generates income but caps upside potential.",
                "Long Call": "Purchase a call option to profit from upward price movements with limited risk.",
                "Bull Call Spread": "Buy a lower strike call and sell a higher strike call. Reduces cost but caps profit potential.",
                "Bear Call Spread": "Sell a lower strike call and buy a higher strike call. Generates income with limited risk.",
                "Long Put": "Purchase a put option to profit from downward price movements with limited risk.",
                "Protective Put": "Own the stock and buy a put option as insurance against downside risk.",
                "Bull Put Spread": "Sell a higher strike put and buy a lower strike put. Generates income with limited risk.",
                "Bear Put Spread": "Buy a higher strike put and sell a lower strike put. Reduces cost but caps profit potential.",
                "Long Straddle": "Buy both a call and put at the same strike. Profit from large price movements in either direction.",
                "Short Straddle": "Sell both a call and put at the same strike. Profit from low volatility and small price movements.",
                "Iron Butterfly": "Combination of a bear call spread and bull put spread with the same middle strike.",
                "Iron Condor": "Combination of a bear call spread and bull put spread with a gap between middle strikes."
            }
            
            st.info(strategy_explanations.get(strategy, ""))
            
            # Visualize strategy
            spot_range = np.linspace(current_price * 0.5, current_price * 1.5, 200)
            fig_pnl = create_strategy_visualization(
                strategy, spot_range, current_price, strike_price,
                time_to_maturity, risk_free_rate, volatility, call_value, put_value
            )
            
            st.pyplot(fig_pnl)
            
            # Display strategy greeks
            st.markdown("### Strategy Risk Profile")
            calculated_greeks = calculate_strategy_greeks(
                strategy, current_price, strike_price, time_to_maturity, risk_free_rate, volatility
            )
            display_greeks(calculated_greeks)
            
            # Calculate and display strategy performance metrics
            if st.checkbox("Show Detailed Strategy Performance"):
                perf_metrics = calculate_strategy_performance(
                    strategy, current_price, strike_price, time_to_maturity,
                    risk_free_rate, volatility, call_value, put_value
                )
                
                st.markdown("### Strategy Performance Metrics")
                
                # Profitability metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                        <div style="background-color: #1E1E1E; padding: 30px; border-radius: 10px;">
                            <h4 style="color: white;">Profitability Metrics</h4>
                            <ul style="color: white; list-style-type: none; padding-left: 0; font-size: 1.15rem;">
                                <li>• Max Profit: ${perf_metrics['profitability']['max_profit']:.2f}</li>
                                <li>• Max Loss: ${perf_metrics['profitability']['max_loss']:.2f}</li>
                                <li>• Risk-Reward Ratio: {perf_metrics['profitability']['risk_reward_ratio']:.2f}</li>
                                <li>• Profit Probability: {perf_metrics['profitability']['profit_probability']:.1%}</li>
                            </ul>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                        <div style="background-color: #1E1E1E; padding: 30px; border-radius: 10px;">
                            <h4 style="color: white;">Risk Metrics</h4>
                            <ul style="color: white; list-style-type: none; padding-left: 0; font-size: 1.15rem;">
                                <li>• P&L Volatility: ${perf_metrics['risk_metrics']['pnl_volatility']:.2f}</li>
                                <li>• Sharpe Ratio: {perf_metrics['risk_metrics']['sharpe_ratio']:.2f}</li>
                                <li>• Kelly Criterion: {perf_metrics['risk_metrics']['kelly_criterion']:.1%}</li>
                            </ul>
                        </div>
                    """, unsafe_allow_html=True)
        
        with quant_tab:
            st.markdown("## Quantitative Analysis")
            st.info("Select a quant tool to perform advanced analysis")
            
            quant_tool = st.selectbox(
                "Select Quantitative Tool",
                ["Implied Volatility", "Local Volatility Surface", "Value at Risk (VaR)", "Risk Scenario Analysis"]
            )
            
            if quant_tool == "Implied Volatility":
                st.markdown("### Implied Volatility Calculator")
                
                col1, col2 = st.columns(2)
                with col1:
                    market_price = st.number_input("Market Option Price", value=5.0, step=0.1)
                    option_type = st.selectbox("Option Type", ["call", "put"])
                
                with col2:
                    initial_guess = st.slider("Initial Volatility Guess", 0.1, 1.0, 0.2, 0.05)
                
                if st.button("Calculate Implied Volatility"):
                    try:
                        iv = implied_volatility(
                            market_price, current_price, strike_price, time_to_maturity,
                            risk_free_rate, option_type, precision=1e-8, max_iterations=20
                        )
                        
                        st.success(f"Implied Volatility: {iv:.2%}")
                        
                        # Display implied vol vs current vol
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Implied Volatility", f"{iv:.2%}", f"{(iv-volatility)/volatility:.2%}")
                        
                        with col2:
                            st.metric("Model Volatility", f"{volatility:.2%}")
                    except Exception as iv_error:
                        st.error(f"Error calculating implied volatility: {str(iv_error)}")
            
            elif quant_tool == "Local Volatility Surface":
                st.subheader("Local Volatility Surface Generator")
                
                # Introduction with theory
                with st.expander("About Local Volatility Models"):
                    st.markdown("""
                    ### The Dupire Local Volatility Model
                    
                    The local volatility model, developed by Bruno Dupire and Emanuel Derman, extends the Black-Scholes framework by allowing volatility to vary with both strike price and time. This addresses key market features like volatility skew and smile that constant volatility models cannot capture.
                    
                    #### Mathematical Foundation:
                    
                    Dupire's formula relates the local volatility $\sigma_{loc}(K, T)$ to the implied volatility surface:
                    
                    $$\sigma_{loc}^2(K, T) = \\frac{\\frac{\partial C}{\partial T} + (r-q)K\\frac{\partial C}{\partial K} + qC}{\\frac{1}{2}K^2\\frac{\partial^2 C}{\partial K^2}}$$
                    
                    where:
                    - $C$ is the call option price
                    - $K$ is the strike price
                    - $T$ is time to maturity
                    - $r$ is the risk-free rate
                    - $q$ is the dividend yield
                    
                    This tool implements a numerically stable version of this formula with sophisticated smoothing techniques.
                    """)
                
                # Create interface with two columns
                col1, col2 = st.columns([2, 2])
                
                with col1:
                    # Data source selection
                    data_source = st.radio(
                        "Select Data Source",
                        ["Simulated Market", "Custom Parameters", "Upload Market Data"],
                        key="vol_data_source"
                    )
                    
                    if data_source == "Simulated Market":
                        # Parameters for simulated data
                        st.subheader("Simulated Market Parameters")
                        skew_factor = st.slider("Volatility Skew", -0.3, 0.3, -0.1, 0.05,
                                              help="Negative values create downward sloping skew (typical in equity markets)")
                        smile_factor = st.slider("Volatility Smile", 0.0, 0.3, 0.05, 0.01,
                                               help="Higher values create stronger smile (U-shape)")
                        term_structure = st.slider("Term Structure", -0.1, 0.2, 0.05, 0.01,
                                                 help="Positive values mean longer-dated options have higher volatility")
                    
                    elif data_source == "Custom Parameters":
                        # Advanced custom parameters
                        st.subheader("Custom Surface Parameters")
                        base_vol = st.number_input("Base Volatility", value=volatility, min_value=0.01, max_value=1.0, step=0.01)
                        num_strikes = st.slider("Number of Strikes", 5, 15, 9)
                        num_maturities = st.slider("Number of Maturities", 3, 10, 6)
                        moneyness_range = st.slider("Moneyness Range", 0.5, 2.0, (0.7, 1.3))
                        max_maturity = st.slider("Maximum Maturity (Years)", 0.5, 5.0, 2.0)
                    
                    elif data_source == "Upload Market Data":
                        # File upload option
                        st.subheader("Upload Market Data")
                        uploaded_file = st.file_uploader(
                            "Upload option chain CSV",
                            type=["csv"],
                            help="CSV should contain: Strike, Maturity, ImpliedVol columns"
                        )
                        
                        if uploaded_file is not None:
                            st.info("File uploaded successfully. Configure processing parameters below.")
                        
                        interpolation_method = st.selectbox(
                            "Interpolation Method",
                            ["Cubic Spline", "Thin Plate Spline", "Linear"],
                            help="Method used to fill gaps in market data"
                        )
                
                with col2:
                    # Common parameters for all modes
                    st.subheader("Calculation Parameters")
                    
                    dividend_yield = st.number_input(
                        "Dividend Yield (%)",
                        value=0.0, min_value=0.0, max_value=10.0, step=0.1
                    ) / 100.0
                    
                    smoothing_level = st.slider(
                        "Surface Smoothing",
                        0.0, 1.0, 0.2, 0.05,
                        help="Controls smoothness of the output (higher = smoother)"
                    )
                    
                    # Visualization options
                    st.subheader("Visualization Options")
                    
                    display_mode = st.radio(
                        "Display Mode",
                        ["3D Surface", "Contour Plot", "Term Structure", "Skew Analysis", "All Views"]
                    )
                    
                    color_scheme = st.selectbox(
                        "Color Scheme",
                        ["viridis", "plasma", "inferno", "magma", "cividis"]
                    )

                # Generate volatility surfaces based on selected method
                if st.button("Generate Local Volatility Surface"):
                    with st.spinner("Calculating local volatility surface..."):
                        try:
                            # Initialize parameters based on data source
                            if data_source == "Simulated Market":
                                # Create simulated data with user parameters
                                strike_pcts = np.array([0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3])
                                maturities = np.array([1/12, 2/12, 3/12, 6/12, 9/12, 1.0, 1.5, 2.0])
                                
                                strikes = current_price * strike_pcts
                                
                                # Create volatility surface with user-specified skew, smile, and term structure
                                implied_vols = np.zeros((len(maturities), len(strikes)))
                                for i, t in enumerate(maturities):
                                    for j, k in enumerate(strike_pcts):
                                        # Model volatility with skew, smile and term structure
                                        moneyness_effect = skew_factor * (1.0 - k)
                                        smile_effect = smile_factor * (k - 1.0)**2
                                        term_effect = term_structure * (t - maturities.mean())
                                        
                                        implied_vols[i, j] = max(0.05, volatility + moneyness_effect + smile_effect + term_effect)
                            
                            elif data_source == "Custom Parameters":
                                # Generate custom grid based on user specifications
                                strike_pcts = np.linspace(moneyness_range[0], moneyness_range[1], num_strikes)
                                maturities = np.linspace(1/12, max_maturity, num_maturities)
                                
                                strikes = current_price * strike_pcts
                                
                                # Generate implied volatility surface with custom parameters
                                implied_vols = np.zeros((len(maturities), len(strikes)))
                                for i, t in enumerate(maturities):
                                    for j, k in enumerate(strike_pcts):
                                        # Simple volatility model with term structure and skew
                                        term_factor = 0.05 * (t - maturities.mean())
                                        skew_factor = -0.1 * (k - 1.0)
                                        smile_factor = 0.05 * (k - 1.0)**2
                                        
                                        implied_vols[i, j] = max(0.05, base_vol + term_factor + skew_factor + smile_factor)
                            
                            elif data_source == "Upload Market Data" and uploaded_file is not None:
                                # Process uploaded data
                                market_data = pd.read_csv(uploaded_file)
                                
                                # Validate required columns
                                required_columns = ["Strike", "Maturity", "ImpliedVol"]
                                if not all(col in market_data.columns for col in required_columns):
                                    st.error(f"CSV must contain columns: {', '.join(required_columns)}")
                                    st.stop()
                                
                                # Extract unique strikes and maturities
                                unique_strikes = np.sort(market_data["Strike"].unique())
                                unique_maturities = np.sort(market_data["Maturity"].unique())
                                
                                # Create matrix for implied volatilities
                                strikes = unique_strikes
                                maturities = unique_maturities
                                implied_vols = np.zeros((len(maturities), len(strikes)))
                                
                                # Fill matrix with market data
                                for i, t in enumerate(maturities):
                                    for j, k in enumerate(strikes):
                                        mask = (market_data["Maturity"] == t) & (market_data["Strike"] == k)
                                        if mask.any():
                                            implied_vols[i, j] = market_data.loc[mask, "ImpliedVol"].values[0]
                                        else:
                                            # If using linear interpolation, we'll set missing values to NaN
                                            # and let the spline fill them
                                            implied_vols[i, j] = np.nan
                                
                                # Handle missing values in the grid
                                if np.isnan(implied_vols).any():
                                    if interpolation_method == "Linear":
                                        # Simple linear interpolation for missing values
                                        # Get coordinates of non-NaN values
                                        ii, jj = np.where(~np.isnan(implied_vols))
                                        known_points = np.column_stack([ii, jj])
                                        known_values = implied_vols[~np.isnan(implied_vols)]
                                        
                                        # Create grid for interpolation
                                        grid_i, grid_j = np.mgrid[0:len(maturities), 0:len(strikes)]
                                        
                                        # Interpolate
                                        implied_vols = griddata(known_points, known_values, (grid_i, grid_j), method='linear')
                                        
                                        # Fill any remaining NaNs with nearest value
                                        if np.isnan(implied_vols).any():
                                            mask = np.isnan(implied_vols)
                                            implied_vols[mask] = griddata(known_points, known_values, (grid_i[mask], grid_j[mask]), method='nearest')
                                    
                                    elif interpolation_method == "Cubic Spline":
                                        # More advanced spline interpolation
                                        # Get coordinates of non-NaN values
                                        ii, jj = np.where(~np.isnan(implied_vols))
                                        known_values = implied_vols[~np.isnan(implied_vols)]
                                        
                                        # Convert indices to actual maturity and strike values
                                        t_points = maturities[ii]
                                        k_points = strikes[jj]
                                        
                                        # Create spline
                                        spline = SmoothBivariateSpline(t_points, k_points, known_values, s=0.1)
                                        
                                        # Evaluate spline at all grid points
                                        T_mesh, K_mesh = np.meshgrid(maturities, strikes, indexing='ij')
                                        implied_vols = spline(maturities, strikes)
                                        
                                    else:  # Thin Plate Spline
                                        # Get coordinates of non-NaN values
                                        ii, jj = np.where(~np.isnan(implied_vols))
                                        known_values = implied_vols[~np.isnan(implied_vols)]
                                        
                                        # Convert indices to actual maturity and strike values
                                        t_points = maturities[ii]
                                        k_points = strikes[jj]
                                        
                                        # Create RBF interpolator
                                        rbf = Rbf(t_points, k_points, known_values, function='thin_plate')
                                        
                                        # Evaluate RBF at all grid points
                                        T_mesh, K_mesh = np.meshgrid(maturities, strikes, indexing='ij')
                                        implied_vols = rbf(T_mesh, K_mesh)
                            else:
                                st.error("Please upload a file when using 'Upload Market Data' option")
                                st.stop()
                            
                            # Generate local volatility surface
                            local_vols, smoothed_ivs, diagnostics = enhanced_local_volatility_surface(
                                strikes, maturities, implied_vols, current_price, risk_free_rate,
                                dividend_yield, smoothing_level
                            )
                            
                            # Analyze the volatility surfaces
                            analysis_results = analyze_vol_surface(
                                local_vols, smoothed_ivs, strikes, maturities, current_price
                            )
                            
                            # Create visualization based on selected display mode
                            # Create grid for visualization
                            K_grid, T_grid = np.meshgrid(strikes/current_price, maturities)
                            
                            if display_mode == "3D Surface" or display_mode == "All Views":
                                st.subheader("Volatility Surfaces")
                                
                                fig = plt.figure(figsize=(14, 10))
                                
                                # Plot two surfaces: implied and local volatility
                                ax1 = fig.add_subplot(221, projection='3d')
                                surf1 = ax1.plot_surface(K_grid, T_grid, smoothed_ivs, cmap=color_scheme, alpha=0.8)
                                ax1.set_xlabel('Moneyness (K/S)')
                                ax1.set_ylabel('Maturity (Years)')
                                ax1.set_zlabel('Volatility')
                                ax1.set_title('Implied Volatility Surface')
                                fig.colorbar(surf1, ax=ax1, shrink=0.5, aspect=5)
                                
                                ax2 = fig.add_subplot(222, projection='3d')
                                surf2 = ax2.plot_surface(K_grid, T_grid, local_vols, cmap=color_scheme, alpha=0.8)
                                ax2.set_xlabel('Moneyness (K/S)')
                                ax2.set_ylabel('Maturity (Years)')
                                ax2.set_zlabel('Volatility')
                                ax2.set_title('Local Volatility Surface')
                                fig.colorbar(surf2, ax=ax2, shrink=0.5, aspect=5)
                                
                                # Add contour plots below for clarity
                                ax3 = fig.add_subplot(223)
                                contour1 = ax3.contourf(K_grid, T_grid, smoothed_ivs, levels=20, cmap=color_scheme)
                                ax3.set_xlabel('Moneyness (K/S)')
                                ax3.set_ylabel('Maturity (Years)')
                                ax3.set_title('Implied Volatility Contour Map')
                                fig.colorbar(contour1, ax=ax3, shrink=0.5, aspect=5)
                                
                                ax4 = fig.add_subplot(224)
                                contour2 = ax4.contourf(K_grid, T_grid, local_vols, levels=20, cmap=color_scheme)
                                ax4.set_xlabel('Moneyness (K/S)')
                                ax4.set_ylabel('Maturity (Years)')
                                ax4.set_title('Local Volatility Contour Map')
                                fig.colorbar(contour2, ax=ax4, shrink=0.5, aspect=5)
                                
                                plt.tight_layout()
                                st.pyplot(fig)
                            
                            if display_mode == "Contour Plot" or display_mode == "All Views":
                                # High-resolution contour plots
                                st.subheader("Volatility Contour Maps")
                                
                                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
                                
                                # Implied volatility contour
                                im1 = ax1.contourf(K_grid, T_grid, smoothed_ivs, levels=30, cmap=color_scheme)
                                ax1.set_xlabel('Moneyness (K/S)')
                                ax1.set_ylabel('Maturity (Years)')
                                ax1.set_title('Implied Volatility')
                                plt.colorbar(im1, ax=ax1)
                                
                                # Add contour lines for key volatility levels
                                cs1 = ax1.contour(K_grid, T_grid, smoothed_ivs, levels=5, colors='white', alpha=0.6, linewidths=0.5)
                                ax1.clabel(cs1, inline=True, fontsize=8, fmt='%.2f')
                                
                                # Local volatility contour
                                im2 = ax2.contourf(K_grid, T_grid, local_vols, levels=30, cmap=color_scheme)
                                ax2.set_xlabel('Moneyness (K/S)')
                                ax2.set_ylabel('Maturity (Years)')
                                ax2.set_title('Local Volatility')
                                plt.colorbar(im2, ax=ax2)
                                
                                # Add contour lines for key volatility levels
                                cs2 = ax2.contour(K_grid, T_grid, local_vols, levels=5, colors='white', alpha=0.6, linewidths=0.5)
                                ax2.clabel(cs2, inline=True, fontsize=8, fmt='%.2f')
                                
                                plt.tight_layout()
                                st.pyplot(fig)
                            
                            if display_mode == "Term Structure" or display_mode == "All Views":
                                # Term structure visualization
                                st.subheader("Volatility Term Structure")
                                
                                # Get ATM volatilities
                                atm_index = np.argmin(np.abs(strikes/current_price - 1.0))
                                
                                fig, ax = plt.subplots(figsize=(10, 6))
                                
                                ax.plot(maturities, smoothed_ivs[:, atm_index], 'b-o', label='ATM Implied Vol')
                                ax.plot(maturities, local_vols[:, atm_index], 'r-^', label='ATM Local Vol')
                                
                                # Add OTM volatilities
                                otm_put_idx = max(0, np.argmin(np.abs(strikes/current_price - 0.9)))
                                otm_call_idx = min(len(strikes)-1, np.argmin(np.abs(strikes/current_price - 1.1)))
                                
                                ax.plot(maturities, smoothed_ivs[:, otm_put_idx], 'b--', alpha=0.6, label='90% OTM Put Implied Vol')
                                ax.plot(maturities, smoothed_ivs[:, otm_call_idx], 'b-.', alpha=0.6, label='110% OTM Call Implied Vol')
                                
                                ax.set_xlabel('Maturity (Years)')
                                ax.set_ylabel('Volatility')
                                ax.set_title('Volatility Term Structure')
                                ax.grid(True, alpha=0.3)
                                ax.legend()
                                
                                st.pyplot(fig)
                            
                            if display_mode == "Skew Analysis" or display_mode == "All Views":
                                # Volatility skew visualization
                                st.subheader("Volatility Skew Analysis")
                                
                                # Select maturities for skew analysis
                                if len(maturities) >= 3:
                                    maturity_indices = [0, len(maturities)//2, -1]  # Short, medium, long-term
                                    maturity_labels = ['Short-term', 'Medium-term', 'Long-term']
                                else:
                                    maturity_indices = list(range(len(maturities)))
                                    maturity_labels = [f"T={t:.2f}" for t in maturities]
                                
                                fig, ax = plt.subplots(figsize=(12, 6))
                                
                                for i, idx in enumerate(maturity_indices):
                                    if idx < 0:  # Handle negative index
                                        actual_idx = len(maturities) + idx
                                    else:
                                        actual_idx = idx
                                        
                                    ax.plot(strikes/current_price, smoothed_ivs[actual_idx, :],
                                           f'C{i}-o', label=f'{maturity_labels[i]} Implied Vol (T={maturities[actual_idx]:.2f})')
                                    ax.plot(strikes/current_price, local_vols[actual_idx, :],
                                           f'C{i}--', label=f'{maturity_labels[i]} Local Vol (T={maturities[actual_idx]:.2f})')
                                
                                ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, label='ATM')
                                ax.set_xlabel('Moneyness (K/S)')
                                ax.set_ylabel('Volatility')
                                ax.set_title('Volatility Skew Across Maturities')
                                ax.grid(True, alpha=0.3)
                                ax.legend()
                                
                                st.pyplot(fig)
                            
                            # Display analysis results
                            st.subheader("Quantitative Analysis")
                            
                            # Organize results into columns
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("### Surface Characteristics")
                                
                                # ATM Term Structure
                                atm_index = np.argmin(np.abs(strikes/current_price - 1.0))
                                
                                # Create DataFrame for ATM vols
                                atm_vol_df = pd.DataFrame({
                                    'Maturity': maturities,
                                    'ATM Implied Vol': smoothed_ivs[:, atm_index],
                                    'ATM Local Vol': local_vols[:, atm_index]
                                })
                                
                                st.markdown("#### ATM Volatility Term Structure")
                                st.dataframe(atm_vol_df.style.format({
                                    'Maturity': '{:.2f}',
                                    'ATM Implied Vol': '{:.2%}',
                                    'ATM Local Vol': '{:.2%}'
                                }))
                                
                                # Skew Analysis
                                st.markdown("#### Volatility Skew Analysis")
                                skew_df = pd.DataFrame(analysis_results['skew_analysis'])
                                st.dataframe(skew_df.style.format({
                                    'maturity': '{:.2f}',
                                    'local_skew': '{:.4f}',
                                    'implied_skew': '{:.4f}'
                                }))
                            
                            with col2:
                                st.markdown("### Numerical Diagnostics")
                                
                                # Surface stability assessment
                                st.markdown("#### Surface Stability")
                                st.info(f"Stability Assessment: {analysis_results['surface_stability']['stability_assessment']}")
                                st.metric("Smoothness Metric", f"{analysis_results['surface_stability']['smoothness_metric']:.4f}")
                                
                                # Implied vs Local Vol comparison
                                st.markdown("#### IV vs LV Comparison")
                                comp_df = pd.DataFrame([{
                                    'Metric': 'Mean Difference',
                                    'Value': analysis_results['vol_comparison']['mean_diff']
                                }, {
                                    'Metric': 'Max Difference',
                                    'Value': analysis_results['vol_comparison']['max_diff']
                                }, {
                                    'Metric': 'Min Difference',
                                    'Value': analysis_results['vol_comparison']['min_diff']
                                }, {
                                    'Metric': 'RMS Difference',
                                    'Value': analysis_results['vol_comparison']['rms_diff']
                                }])
                                
                                st.dataframe(comp_df.style.format({
                                    'Value': '{:.4f}'
                                }))
                                
                                # Calculation diagnostics
                                st.markdown("#### Calculation Diagnostics")
                                diag_df = pd.DataFrame([{
                                    'Metric': 'Boundary Points',
                                    'Count': diagnostics['boundary_points']
                                }, {
                                    'Metric': 'Denominator Fixes',
                                    'Count': diagnostics['denominator_fixes']
                                }, {
                                    'Metric': 'Extreme Values Clipped',
                                    'Count': diagnostics['extreme_values_clipped']
                                }])
                                
                                st.dataframe(diag_df)
                            
                                # Option to download data
                                st.subheader("Download Results")

                                try:
                                    # Try to import xlsxwriter
                                    import xlsxwriter
                                    excel_export_available = True
                                except ImportError:
                                    excel_export_available = False

                                # Create DataFrame collection for export
                                export_data = {
                                    'Parameters': pd.DataFrame([
                                        {'Parameter': 'Spot Price', 'Value': current_price},
                                        {'Parameter': 'Risk-Free Rate', 'Value': risk_free_rate},
                                        {'Parameter': 'Dividend Yield', 'Value': dividend_yield},
                                        {'Parameter': 'Smoothing Level', 'Value': smoothing_level}
                                    ]),
                                    'Strikes': pd.DataFrame({'Strike': strikes, 'Moneyness': strikes/current_price}),
                                    'Maturities': pd.DataFrame({'Maturity': maturities}),
                                    'ImpliedVolSurface': pd.DataFrame(
                                        smoothed_ivs,
                                        index=[f'T={t:.2f}' for t in maturities],
                                        columns=[f'K={k:.1f}' for k in strikes]
                                    ),
                                    'LocalVolSurface': pd.DataFrame(
                                        local_vols,
                                        index=[f'T={t:.2f}' for t in maturities],
                                        columns=[f'K={k:.1f}' for k in strikes]
                                    ),
                                    'ATM_TermStructure': atm_vol_df,
                                    'SkewAnalysis': skew_df
                                }

                                # CSV export option (always available)
                                for name, df in export_data.items():
                                    csv_buffer = io.StringIO()
                                    df.to_csv(csv_buffer)
                                    csv_buffer.seek(0)
                                    
                                    st.download_button(
                                        label=f"Download {name} as CSV",
                                        data=csv_buffer.getvalue().encode(),
                                        file_name=f"{name.lower()}.csv",
                                        mime="text/csv",
                                        key=f"csv_{name}"
                                    )

                                # Excel export option (if xlsxwriter is available)
                                if excel_export_available:
                                    buffer = io.BytesIO()
                                    
                                    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                                        for name, df in export_data.items():
                                            df.to_excel(writer, sheet_name=name, index=name in ['ImpliedVolSurface', 'LocalVolSurface'])
                                    
                                    buffer.seek(0)
                                    
                                    st.download_button(
                                        label="Download All Data as Excel Workbook",
                                        data=buffer,
                                        file_name="volatility_surfaces.xlsx",
                                        mime="application/vnd.ms-excel"
                                    )
                                else:
                                    st.info("Excel export requires the xlsxwriter package. Use the CSV downloads above or install xlsxwriter with 'pip install xlsxwriter'.")
                            
                        except Exception as e:
                            st.error(f"Error calculating volatility surface: {str(e)}")
                            st.exception(e)
            
            elif quant_tool == "Value at Risk (VaR)":
                st.subheader("Value at Risk Calculator")
                
                col1, col2 = st.columns(2)
                with col1:
                    confidence = st.slider("Confidence Level", 0.9, 0.99, 0.95, 0.01, key="confidence_level_slider")
                    horizon = st.slider("Risk Horizon (days)", 1, 30, 1, key="risk_horizon_slider") / 252  # Convert to years
                
                with col2:
                    n_simulations = st.slider("Number of Simulations", 1000, 100000, 10000, 1000, key="n_simulations_slider")
                    strategy_type = st.selectbox("Position Type", ["Long Call", "Long Put", "Covered Call Writing", "Protective Put"], key="strategy_type_checkbox")
                
                # Add enhanced VaR options
                use_t_dist = st.checkbox("Use Student's t-distribution (fat tails)", True,
                                      help="Better captures extreme market events than normal distribution", key="use_t_dist_checkbox")
                if use_t_dist:
                    degrees_of_freedom = st.slider("Degrees of Freedom", 3, 10, 5, 1,
                                                 help="Lower values create fatter tails (3-5 for financial markets)", key="degrees_of_freedom_slider")
                else:
                    degrees_of_freedom = 5

                use_garch = st.checkbox("Use GARCH volatility modeling", False,
                                      help="Models time-varying volatility instead of constant volatility", key="use_garch_checkbox")
                
                # Add stress testing option
                stress_test_tab = st.checkbox("Include Stress Testing", False,
                                           help="Test portfolio against historical crisis scenarios", key="stress_test_checkbox")
                
                if st.button("Calculate VaR"):
                    with st.spinner("Running VaR simulation..."):
                        var_results = var_calculator(
                            strategies=[strategy_type],
                            quantities=[1],
                            spot_price=current_price,
                            strikes=[strike_price],
                            maturities=[time_to_maturity],
                            rates=risk_free_rate,
                            vols=volatility,
                            confidence=confidence,
                            horizon=horizon,
                            n_simulations=n_simulations,
                            use_t_dist=use_t_dist,
                            degrees_of_freedom=degrees_of_freedom,
                            use_garch=use_garch
                        )
                        
                        # Display VaR results
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"""
                                <div style="background-color: #1E1E1E; padding: 20px; border-radius: 10px;">
                                    <h4 style="color: white;">Value-at-Risk</h4>
                                    <ul style="color: white; list-style-type: none; padding-left: 0;">
                                        <li>• VaR ({confidence*100:.1f}%): ${var_results['VaR']:.2f}</li>
                                        <li>• Expected Shortfall: ${var_results['Expected_Shortfall']:.2f}</li>
                                        <li>• Horizon: {var_results['Horizon_Days']:.0f} days</li>
                                        <li>• Distribution: {var_results['Distribution']}</li>
                                        <li>• Volatility Model: {var_results['Volatility_Model']}</li>
                                    </ul>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown(f"""
                                <div style="background-color: #1E1E1E; padding: 20px; border-radius: 10px;">
                                    <h4 style="color: white;">Risk Distribution</h4>
                                    <ul style="color: white; list-style-type: none; padding-left: 0;">
                                        <li>• Volatility: ${var_results['Volatility']:.2f}</li>
                                        <li>• Skewness: {var_results['Skewness']:.2f}</li>
                                        <li>• Kurtosis: {var_results['Kurtosis']:.2f}</li>
                                        <li>• Worst Case: ${var_results['Worst_Case']:.2f}</li>
                                        <li>• Best Case: ${var_results['Best_Case']:.2f}</li>
                                    </ul>
                                </div>
                            """, unsafe_allow_html=True)
                        
                        # Add stress testing results if enabled
                        if stress_test_tab:
                            st.subheader("Stress Test Results")
                            with st.spinner("Running stress tests..."):
                                stress_results = stress_test_portfolio(
                                    strategies=[strategy_type],
                                    quantities=[1],
                                    spot_price=current_price,
                                    strikes=[strike_price],
                                    maturities=[time_to_maturity],
                                    rates=risk_free_rate,
                                    vols=volatility
                                )
                                # Add the new error handling code here:
                                try:
                                    # Create DataFrame from results
                                    stress_data = []
                                    for scenario, results in stress_results.items():
                                        try:
                                            # Handle if results is a dictionary with expected structure
                                            if isinstance(results, dict) and "P&L" in results:
                                                scenario_details = "N/A"
                                                if "Applied_Scenario" in results:
                                                    scenario_details = f"{results['Applied_Scenario']['Price Change']} price, {results['Applied_Scenario']['Volatility Multiplier']} vol, {results['Applied_Scenario']['Rate Change']} rate"
                                                
                                                stress_data.append({
                                                    "Scenario": scenario,
                                                    "P&L": results["P&L"],
                                                    "% Change": results.get("Portfolio_Change_Pct", 0.0),
                                                    "Scenario Details": scenario_details
                                                })
                                            else:
                                                # Handle if results is just a float
                                                stress_data.append({
                                                    "Scenario": scenario,
                                                    "P&L": float(results) if isinstance(results, (int, float)) else 0.0,
                                                    "% Change": 0.0,
                                                    "Scenario Details": "N/A"
                                                })
                                        except Exception as e:
                                            st.warning(f"Error processing scenario {scenario}: {str(e)}")
                                    
                                    stress_df = pd.DataFrame(stress_data)
                                    
                                    # Display results as a table
                                    st.dataframe(stress_df.style.format({
                                        "P&L": "${:.2f}",
                                        "% Change": "{:.2f}%"
                                    }))
                                    
                                    # Create bar chart of stress test results
                                    # Create bar chart of stress test results
                                    fig = plt.figure(figsize=(12, 7))
                                    ax = plt.gca()
                                    bars = plt.bar(stress_df["Scenario"], stress_df["P&L"],
                                           color=['#FF5555' if x < 0 else '#55CC55' for x in stress_df["P&L"]],
                                           width=0.7)
                                    plt.axhline(y=0, color='white', linestyle='-', alpha=0.3)
                                    plt.ylabel('P&L ($)', fontsize=12)
                                    plt.xlabel('Scenario', fontsize=12)
                                    plt.title('Stress Test Results', fontsize=14, fontweight='bold')
                                    plt.xticks(rotation=30, ha='right', fontsize=10)
                                    plt.grid(axis='y', alpha=0.3)
                                    plt.tight_layout(pad=2)

                                    # Add values on top of bars with better positioning
                                    for bar in bars:
                                        height = bar.get_height()
                                        y_pos = min(-0.7, height - 0.5) if height < 0 else max(0.3, height + 0.5)
                                        plt.text(bar.get_x() + bar.get_width()/2, y_pos,
                                               f'${height:.2f}',
                                               ha='center',
                                               va='bottom' if height >= 0 else 'top',
                                               color='white',
                                               fontweight='bold',
                                               fontsize=11)
                                        
                                    # Add more whitespace at the bottom for labels
                                    plt.subplots_adjust(bottom=0.15)
                                        
                                    st.pyplot(fig)
                                except Exception as e:
                                    st.error(f"Error processing stress test results: {str(e)}")
            
            elif quant_tool == "Risk Scenario Analysis":
                st.subheader("Strategy Scenario Analysis")
                
                strategy_list = ["Long Call", "Long Put", "Covered Call Writing", "Protective Put",
                              "Bull Call Spread", "Bear Put Spread", "Long Straddle", "Iron Condor"]
                selected_strategy = st.selectbox("Select Strategy for Analysis", strategy_list)
                
                if st.button("Run Scenario Analysis"):
                    with st.spinner("Running scenario analysis..."):
                        risk_results = risk_scenario_analysis(
                            selected_strategy, current_price, strike_price, time_to_maturity,
                            risk_free_rate, volatility, calculate_strategy_pnl
                        )
                        
                        # Display price impact
                        st.subheader("Price Impact Scenarios")
                        price_fig = plt.figure(figsize=(10, 5))
                        bars = plt.bar(
                            [f"{p:.0f}" for p in risk_results['price_impact']['scenarios']],
                            risk_results['price_impact']['pnl'],
                            color=['red' if x < 0 else 'green' for x in risk_results['price_impact']['pnl']]
                        )
                        plt.axhline(y=0, color='white', linestyle='-', alpha=0.3)
                        plt.xlabel('Price Scenarios')
                        plt.ylabel('P&L')
                        plt.title('Price Impact on P&L')
                        
                        # Add values on top of bars
                        for bar in bars:
                            height = bar.get_height()
                            plt.text(bar.get_x() + bar.get_width()/2., height,
                                  f'${height:.2f}',
                                  ha='center', va='bottom' if height > 0 else 'top',
                                  color='white')
                        
                        st.pyplot(price_fig)
                        
                        # Display extreme scenarios
                        st.subheader("Extreme Market Scenarios")
                        st.markdown(f"""
                            <div style="background-color: #1E1E1E; padding: 20px; border-radius: 10px;">
                                <h4 style="color: white;">Extreme Scenario P&L Impact</h4>
                                <ul style="color: white; list-style-type: none; padding-left: 0;">
                                    <li>• Market Crash (-20%): ${risk_results['extreme_scenarios']['market_crash']:.2f}</li>
                                    <li>• Market Rally (+20%): ${risk_results['extreme_scenarios']['market_rally']:.2f}</li>
                                    <li>• Volatility Explosion (2x): ${risk_results['extreme_scenarios']['vol_explosion']:.2f}</li>
                                    <li>• Volatility Collapse (0.5x): ${risk_results['extreme_scenarios']['vol_collapse']:.2f}</li>
                                </ul>
                            </div>
                        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Please check your inputs and try again.")
        if st.checkbox("Show detailed error trace", key="show_error"):
            st.exception(e)