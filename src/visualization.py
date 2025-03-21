import numpy as np
import matplotlib.pyplot as plt
from src.strategy import calculate_strategy_pnl

# Strategy Visualization Function
def create_strategy_visualization(strategy, spot_range, current_price, strike_price, time_to_maturity, risk_free_rate, volatility, call_value, put_value):
    """
    Create visualization for option strategies with accurate payoffs and key levels
    
    Parameters:
    -----------
    strategy : str
        Strategy type
    spot_range : array
        Range of spot prices for calculation
    current_price, strike_price, time_to_maturity, risk_free_rate, volatility : float
        Market parameters
    call_value, put_value : float
        Current option values
    
    Returns:
    --------
    fig: matplotlib figure object
    """
    # Calculate P&L
    pnl = calculate_strategy_pnl(
        strategy, spot_range, current_price, strike_price,
        time_to_maturity, risk_free_rate, volatility, call_value, put_value
    )
    
    # Define common strikes for reference
    lower_strike = strike_price * 0.9
    upper_strike = strike_price * 1.1
    
    # Create figure with improved styling
    plt.style.use('dark_background')
    fig_pnl = plt.figure(figsize=(12, 6))
    
    # Plot P&L profile with better visibility
    plt.plot(spot_range, pnl, 'g-', linewidth=2.5, label='P&L Profile')
    plt.axhline(y=0, color='r', linestyle='--', alpha=0.6, label='Break-even Line')
    
    # Add reference lines at current price and strike price(s)
    plt.axvline(x=current_price, color='cyan', linestyle=':', alpha=0.5, label='Current Price')
    plt.axvline(x=strike_price, color='magenta', linestyle=':', alpha=0.5, label='Strike Price')
    
    # For spread strategies, add reference lines for upper/lower strikes
    if 'Spread' in strategy or 'Iron' in strategy:
        if 'Call Spread' in strategy or 'Iron' in strategy:
            plt.axvline(x=upper_strike, color='yellow', linestyle=':', alpha=0.3, label='Upper Strike')
        if 'Put Spread' in strategy or 'Iron' in strategy:
            plt.axvline(x=lower_strike, color='orange', linestyle=':', alpha=0.3, label='Lower Strike')
    
    # Find and mark break-even points
    break_even_indices = np.where(np.diff(np.signbit(pnl)))[0]
    break_even_points = [spot_range[i] for i in break_even_indices]
    
    # Mark key profit/loss points
    max_profit_idx = np.argmax(pnl)
    max_loss_idx = np.argmin(pnl)
    
    # Add annotations
    plt.annotate(f'Max Profit: ${pnl[max_profit_idx]:.2f}',
                xy=(spot_range[max_profit_idx], pnl[max_profit_idx]),
                xytext=(10, 15), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color='white', alpha=0.7))
    
    plt.annotate(f'Max Loss: ${pnl[max_loss_idx]:.2f}',
                xy=(spot_range[max_loss_idx], pnl[max_loss_idx]),
                xytext=(10, -15), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', color='white', alpha=0.7))
    
    # Mark break-even points
    for point in break_even_points:
        plt.axvline(x=point, color='white', linestyle='--', alpha=0.5)
        plt.text(point, plt.ylim()[0] * 0.9, f'BE: {point:.2f}', rotation=90,
               verticalalignment='bottom', color='white', fontweight='bold')
    
    # Improve chart aesthetics
    plt.grid(True, alpha=0.3)
    plt.xlabel('Stock Price ($)', fontsize=12)
    plt.ylabel('Profit/Loss ($)', fontsize=12)
    plt.title(f'{strategy} P&L Profile', fontsize=14, fontweight='bold')
    plt.legend(loc='best')
    
    # Add strategy description based on mathematical formulas
    if strategy == "Covered Call Writing":
        description = "Long stock + Short call\nMax Profit: (K - S₀) + C when S_T > K\nMax Loss: S₀ - C (if S_T → 0)"
    elif strategy == "Long Call":
        description = "Max Profit: Unlimited as S_T → ∞\nMax Loss: Premium paid (C)"
    elif strategy == "Bull Call Spread":
        description = "Long lower strike call + Short higher strike call\nMax Profit: (K₂ - K₁) - Net Debit\nMax Loss: Net debit paid"
    elif strategy == "Bear Call Spread":
        description = "Short lower strike call + Long higher strike call\nMax Profit: Net credit received\nMax Loss: (K₂ - K₁) - Net Credit"
    elif strategy == "Long Put":
        description = "Max Profit: K - P (if S_T → 0)\nMax Loss: Premium paid (P)"
    elif strategy == "Protective Put":
        description = "Long stock + Long put\nMax Profit: Unlimited as S_T → ∞\nMax Loss: S₀ + P - K"
    elif strategy == "Long Straddle":
        description = "Long call + Long put (same strike)\nMax Profit: Unlimited\nMax Loss: C + P"
    elif strategy == "Short Straddle":
        description = "Short call + Short put (same strike)\nMax Profit: C + P\nMax Loss: Unlimited"
    elif strategy == "Iron Butterfly":
        description = "Short ATM put + Short ATM call + Long OTM put + Long OTM call\nMax Profit: Net credit\nMax Loss: Δ - Net Credit"
    elif strategy == "Iron Condor":
        description = "Bull put spread + Bear call spread\nMax Profit: Net credit\nMax Loss: Δ - Net Credit"
    else:
        description = ""
    
    # Add description text box
    plt.figtext(0.02, 0.02, description, fontsize=10, bbox=dict(facecolor='black', alpha=0.7, edgecolor='white'))
    
    return fig_pnl

def analyze_vol_surface(local_vols, implied_vols, strikes, maturities, spot_price):
    """
    Analyze local volatility surface to extract key quantitative features
    
    Parameters:
    -----------
    local_vols : 2D array
        Local volatility surface
    implied_vols : 2D array
        Implied volatility surface
    strikes : array
        Strike prices
    maturities : array
        Maturity dates
    spot_price : float
        Current spot price
        
    Returns:
    --------
    dict: Analysis results
    """
    # Calculate moneyness for better comparisons
    moneyness = np.array([k/spot_price for k in strikes])
    
    # Initialize results
    results = {}
    
    # 1. Extract ATM volatility term structure
    atm_index = np.argmin(np.abs(moneyness - 1.0))
    results['atm_local_vol_term'] = local_vols[:, atm_index]
    results['atm_implied_vol_term'] = implied_vols[:, atm_index]
    
    # 2. Analyze volatility skew at various maturities
    skew_results = []
    
    # Select representative maturities for skew analysis
    maturity_indices = []
    if len(maturities) >= 3:
        # Short, medium and long term
        maturity_indices = [0, len(maturities)//2, len(maturities)-1]
    else:
        maturity_indices = list(range(len(maturities)))
    
    for idx in maturity_indices:
        # Calculate skew as slope of volatility curve near ATM
        if atm_index > 0 and atm_index < len(strikes) - 1:
            local_skew = (local_vols[idx, atm_index+1] - local_vols[idx, atm_index-1]) / (moneyness[atm_index+1] - moneyness[atm_index-1])
            implied_skew = (implied_vols[idx, atm_index+1] - implied_vols[idx, atm_index-1]) / (moneyness[atm_index+1] - moneyness[atm_index-1])
        else:
            # Handle boundary case
            if atm_index == 0:
                local_skew = (local_vols[idx, 1] - local_vols[idx, 0]) / (moneyness[1] - moneyness[0])
                implied_skew = (implied_vols[idx, 1] - implied_vols[idx, 0]) / (moneyness[1] - moneyness[0])
            else:
                local_skew = (local_vols[idx, -1] - local_vols[idx, -2]) / (moneyness[-1] - moneyness[-2])
                implied_skew = (implied_vols[idx, -1] - implied_vols[idx, -2]) / (moneyness[-1] - moneyness[-2])
        
        skew_results.append({
            'maturity': maturities[idx],
            'local_skew': local_skew,
            'implied_skew': implied_skew
        })
    
    results['skew_analysis'] = skew_results
    
    # 3. Volatility surface curvature (smile strength)
    # Calculate for a middle-term maturity
    mid_idx = len(maturities) // 2
    
    # Find OTM put, ATM, and OTM call volatilities
    otm_put_idx = max(0, np.argmin(np.abs(moneyness - 0.9)))
    otm_call_idx = min(len(moneyness)-1, np.argmin(np.abs(moneyness - 1.1)))
    
    # Calculate smile strength as average of deviations from ATM vol
    local_smile = 0.5 * ((local_vols[mid_idx, otm_put_idx] - local_vols[mid_idx, atm_index]) +
                         (local_vols[mid_idx, otm_call_idx] - local_vols[mid_idx, atm_index]))
    
    implied_smile = 0.5 * ((implied_vols[mid_idx, otm_put_idx] - implied_vols[mid_idx, atm_index]) +
                           (implied_vols[mid_idx, otm_call_idx] - implied_vols[mid_idx, atm_index]))
    
    results['smile_strength'] = {
        'local_vol_smile': local_smile,
        'implied_vol_smile': implied_smile
    }
    
    # 4. Implied vs local volatility differences
    results['vol_comparison'] = {
        'mean_diff': np.mean(local_vols - implied_vols),
        'max_diff': np.max(local_vols - implied_vols),
        'min_diff': np.min(local_vols - implied_vols),
        'rms_diff': np.sqrt(np.mean((local_vols - implied_vols)**2))
    }
    
    # 5. Surface stability assessment - higher values indicate potential numerical issues
    local_vol_gradients = np.gradient(local_vols)
    smoothness_metric = np.mean(np.abs(np.gradient(local_vol_gradients[0])) + np.abs(np.gradient(local_vol_gradients[1])))
    
    results['surface_stability'] = {
        'smoothness_metric': smoothness_metric,
        'stability_assessment': 'Good' if smoothness_metric < 0.05 else 'Moderate' if smoothness_metric < 0.1 else 'Poor'
    }
    
    return results