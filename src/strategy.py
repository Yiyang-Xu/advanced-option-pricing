import numpy as np
from src.option_pricing import black_scholes_calc
from src.greeks import calculate_greeks, calculate_advanced_greeks
from scipy.stats import norm

# Calculate Option Strategy Payoffs
def calculate_strategy_pnl(strategy_type, spot_range, current_price, strike_price, time_to_maturity, risk_free_rate, volatility, call_value=0, put_value=0):
    """Calculate P&L for option strategies based on precise mathematical formulas"""
    # Validate inputs
    if not isinstance(spot_range, (list, np.ndarray)):
        raise ValueError("spot_range must be a list or numpy array")
        
    # Add validation for numerical inputs
    for param in [current_price, strike_price, time_to_maturity, risk_free_rate, volatility]:
        if not isinstance(param, (int, float)) or param < 0:
            raise ValueError(f"Invalid parameter value: {param}")
    
    # Define common strikes for spreads
    lower_strike = strike_price * 0.9
    upper_strike = strike_price * 1.1
    
    # Calculate strikes for Iron Condor
    very_lower_strike = lower_strike * 0.95
    very_upper_strike = upper_strike * 1.05
    
    # Initialize PnL array - IMPORTANT: same length as spot_range
    pnl = np.zeros(len(spot_range))
    
    # Calculate option values at entry (for comparison in P&L)
    atm_call_value = call_value if call_value > 0 else black_scholes_calc(
        current_price, strike_price, time_to_maturity, risk_free_rate, volatility, 'call')
    atm_put_value = put_value if put_value > 0 else black_scholes_calc(
        current_price, strike_price, time_to_maturity, risk_free_rate, volatility, 'put')
    
    # For spreads - pre-calculate option values at other strikes
    lower_call_value = black_scholes_calc(
        current_price, lower_strike, time_to_maturity, risk_free_rate, volatility, 'call')
    upper_call_value = black_scholes_calc(
        current_price, upper_strike, time_to_maturity, risk_free_rate, volatility, 'call')
    lower_put_value = black_scholes_calc(
        current_price, lower_strike, time_to_maturity, risk_free_rate, volatility, 'put')
    upper_put_value = black_scholes_calc(
        current_price, upper_strike, time_to_maturity, risk_free_rate, volatility, 'put')
    
    # For Iron Condor
    very_lower_put_value = black_scholes_calc(
        current_price, very_lower_strike, time_to_maturity, risk_free_rate, volatility, 'put')
    very_upper_call_value = black_scholes_calc(
        current_price, very_upper_strike, time_to_maturity, risk_free_rate, volatility, 'call')
    
    for i, spot in enumerate(spot_range):
        # Calculate Call Option Strategies
        if strategy_type == "Covered Call Writing":
            # (ST - S0) + C if ST ≤ K, (K - S0) + C if ST > K
            if spot <= strike_price:
                pnl[i] = (spot - current_price) + atm_call_value
            else:
                pnl[i] = (strike_price - current_price) + atm_call_value
                
        elif strategy_type == "Long Call":
            # max(0, ST - K) - C
            pnl[i] = max(0, spot - strike_price) - atm_call_value
            
        elif strategy_type == "Bull Call Spread":
            # Long lower strike call, short higher strike call
            # max(0, ST - K1) - max(0, ST - K2) - Net Debit
            pnl[i] = max(0, spot - lower_strike) - max(0, spot - upper_strike) - (lower_call_value - upper_call_value)
            
        elif strategy_type == "Bear Call Spread":
            # Short lower strike call, long higher strike call
            # Net Credit - max(0, ST - K1) + max(0, ST - K2)
            pnl[i] = (lower_call_value - upper_call_value) - max(0, spot - lower_strike) + max(0, spot - upper_strike)
            
        # Calculate Put Option Strategies
        elif strategy_type == "Long Put":
            # max(0, K - ST) - P
            pnl[i] = max(0, strike_price - spot) - atm_put_value
            
        elif strategy_type == "Protective Put":
            # (ST - S0) + max(0, K - ST) - P
            pnl[i] = (spot - current_price) + max(0, strike_price - spot) - atm_put_value
            
        elif strategy_type == "Bull Put Spread":
            # Short higher strike put, long lower strike put
            # Net Credit - max(0, K1 - ST) + max(0, K2 - ST)
            pnl[i] = (upper_put_value - lower_put_value) - max(0, upper_strike - spot) + max(0, lower_strike - spot)
            
        elif strategy_type == "Bear Put Spread":
            # Long higher strike put, short lower strike put
            # max(0, K1 - ST) - max(0, K2 - ST) - Net Debit
            pnl[i] = max(0, upper_strike - spot) - max(0, lower_strike - spot) - (upper_put_value - lower_put_value)
            
        # Calculate Combined Strategies
        elif strategy_type == "Long Straddle":
            # max(0, ST - K) + max(0, K - ST) - (C + P)
            pnl[i] = max(0, spot - strike_price) + max(0, strike_price - spot) - (atm_call_value + atm_put_value)
            
        elif strategy_type == "Short Straddle":
            # (C + P) - max(0, ST - K) - max(0, K - ST)
            pnl[i] = (atm_call_value + atm_put_value) - max(0, spot - strike_price) - max(0, strike_price - spot)
            
        elif strategy_type == "Iron Butterfly":
            # Short ATM put, short ATM call, long OTM put, long OTM call
            # Net Credit - max(0, ST - K) + max(0, ST - (K + Δ)) - max(0, K - ST) + max(0, (K - Δ) - ST)
            net_credit = atm_call_value + atm_put_value - upper_call_value - lower_put_value
            pnl[i] = net_credit - max(0, spot - strike_price) + max(0, spot - upper_strike) - max(0, strike_price - spot) + max(0, lower_strike - spot)
            
        elif strategy_type == "Iron Condor":
            # Bull put spread + bear call spread
            # Net Credit - max(0, K1 - ST) + max(0, (K1 - Δ) - ST) - max(0, ST - K2) + max(0, ST - (K2 + Δ))
            net_credit = (lower_put_value - very_lower_put_value) + (upper_call_value - very_upper_call_value)
            pnl[i] = net_credit - max(0, lower_strike - spot) + max(0, very_lower_strike - spot) - max(0, spot - upper_strike) + max(0, spot - very_upper_strike)
        
        else:
            # Default for unknown strategies
            pnl[i] = max(0, spot - strike_price) - atm_call_value
    
    # Make sure we return an array with exactly the same length as spot_range
    return pnl

# Calculate Strategy Greeks
def calculate_strategy_greeks(strategy_type, spot_price, strike_price, time_to_maturity, risk_free_rate, volatility):
    """
    Calculate Greeks for option strategies
    
    Parameters:
    -----------
    strategy_type : str
        Type of option strategy
    spot_price : float
        Current price of underlying
    strike_price : float
        Strike price
    time_to_maturity : float
        Time to expiration in years
    risk_free_rate : float
        Risk-free interest rate (decimal)
    volatility : float
        Volatility (decimal)
        
    Returns:
    --------
    dict: Strategy Greeks
    """
    # Define strikes for spreads
    lower_strike = strike_price * 0.9
    upper_strike = strike_price * 1.1
    
    # Initialize Greeks
    greeks = {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0}
    
    # Calculate basic Greeks for standard options
    call_greeks = calculate_greeks("Call", spot_price, strike_price, time_to_maturity, risk_free_rate, volatility)
    put_greeks = calculate_greeks("Put", spot_price, strike_price, time_to_maturity, risk_free_rate, volatility)
    
    # Calculate Greeks for spread options
    upper_call_greeks = calculate_greeks("Call", spot_price, upper_strike, time_to_maturity, risk_free_rate, volatility)
    lower_call_greeks = calculate_greeks("Call", spot_price, lower_strike, time_to_maturity, risk_free_rate, volatility)
    upper_put_greeks = calculate_greeks("Put", spot_price, upper_strike, time_to_maturity, risk_free_rate, volatility)
    lower_put_greeks = calculate_greeks("Put", spot_price, lower_strike, time_to_maturity, risk_free_rate, volatility)
    
    # Calculate Greeks for Iron Condor
    very_lower_strike = lower_strike * 0.95
    very_upper_strike = upper_strike * 1.05
    very_lower_put_greeks = calculate_greeks("Put", spot_price, very_lower_strike, time_to_maturity, risk_free_rate, volatility)
    very_upper_call_greeks = calculate_greeks("Call", spot_price, very_upper_strike, time_to_maturity, risk_free_rate, volatility)
    
    # Calculate strategy-specific Greeks based on portfolio compositions
    if strategy_type == "Covered Call Writing":
        greeks['delta'] = 1 - call_greeks['delta']
        greeks['gamma'] = -call_greeks['gamma']
        greeks['theta'] = -call_greeks['theta']
        greeks['vega'] = -call_greeks['vega']
        
    elif strategy_type == "Long Call":
        greeks = call_greeks
        
    elif strategy_type == "Protected Short Sale":
        greeks['delta'] = -1 + call_greeks['delta']
        greeks['gamma'] = call_greeks['gamma']
        greeks['theta'] = call_greeks['theta']
        greeks['vega'] = call_greeks['vega']
        
    elif strategy_type == "Reverse Hedge":
        greeks['delta'] = call_greeks['delta'] + put_greeks['delta']
        greeks['gamma'] = call_greeks['gamma'] + put_greeks['gamma']
        greeks['theta'] = call_greeks['theta'] + put_greeks['theta']
        greeks['vega'] = call_greeks['vega'] + put_greeks['vega']
        
    elif strategy_type == "Naked Call Writing":
        greeks['delta'] = -call_greeks['delta']
        greeks['gamma'] = -call_greeks['gamma']
        greeks['theta'] = -call_greeks['theta']
        greeks['vega'] = -call_greeks['vega']
        
    elif strategy_type == "Bull Call Spread":
        greeks['delta'] = lower_call_greeks['delta'] - upper_call_greeks['delta']
        greeks['gamma'] = lower_call_greeks['gamma'] - upper_call_greeks['gamma']
        greeks['theta'] = lower_call_greeks['theta'] - upper_call_greeks['theta']
        greeks['vega'] = lower_call_greeks['vega'] - upper_call_greeks['vega']
        
    elif strategy_type == "Bear Call Spread":
        greeks['delta'] = -lower_call_greeks['delta'] + upper_call_greeks['delta']
        greeks['gamma'] = -lower_call_greeks['gamma'] + upper_call_greeks['gamma']
        greeks['theta'] = -lower_call_greeks['theta'] + upper_call_greeks['theta']
        greeks['vega'] = -lower_call_greeks['vega'] + upper_call_greeks['vega']
        
    elif strategy_type == "Long Put":
        greeks = put_greeks
        
    elif strategy_type == "Protective Put":
        greeks['delta'] = 1 + put_greeks['delta']
        greeks['gamma'] = put_greeks['gamma']
        greeks['theta'] = put_greeks['theta']
        greeks['vega'] = put_greeks['vega']
        
    elif strategy_type == "Bull Put Spread":
        # Short higher strike put, long lower strike put
        greeks['delta'] = -upper_put_greeks['delta'] + lower_put_greeks['delta']
        greeks['gamma'] = -upper_put_greeks['gamma'] + lower_put_greeks['gamma']
        greeks['theta'] = -upper_put_greeks['theta'] + lower_put_greeks['theta']
        greeks['vega'] = -upper_put_greeks['vega'] + lower_put_greeks['vega']
        
    elif strategy_type == "Bear Put Spread":
        # Long higher strike put, short lower strike put
        greeks['delta'] = upper_put_greeks['delta'] - lower_put_greeks['delta']
        greeks['gamma'] = upper_put_greeks['gamma'] - lower_put_greeks['gamma']
        greeks['theta'] = upper_put_greeks['theta'] - lower_put_greeks['theta']
        greeks['vega'] = upper_put_greeks['vega'] - lower_put_greeks['vega']
        
    elif strategy_type == "Long Straddle":
        greeks['delta'] = call_greeks['delta'] + put_greeks['delta']
        greeks['gamma'] = call_greeks['gamma'] + put_greeks['gamma']
        greeks['theta'] = call_greeks['theta'] + put_greeks['theta']
        greeks['vega'] = call_greeks['vega'] + put_greeks['vega']
        
    elif strategy_type == "Short Straddle":
        greeks['delta'] = -call_greeks['delta'] - put_greeks['delta']
        greeks['gamma'] = -call_greeks['gamma'] - put_greeks['gamma']
        greeks['theta'] = -call_greeks['theta'] - put_greeks['theta']
        greeks['vega'] = -call_greeks['vega'] - put_greeks['vega']
        
    elif strategy_type == "Iron Butterfly":
        greeks['delta'] = lower_put_greeks['delta'] - put_greeks['delta'] - call_greeks['delta'] + upper_call_greeks['delta']
        greeks['gamma'] = lower_put_greeks['gamma'] - put_greeks['gamma'] - call_greeks['gamma'] + upper_call_greeks['gamma']
        greeks['theta'] = lower_put_greeks['theta'] - put_greeks['theta'] - call_greeks['theta'] + upper_call_greeks['theta']
        greeks['vega'] = lower_put_greeks['vega'] - put_greeks['vega'] - call_greeks['vega'] + upper_call_greeks['vega']
        
    elif strategy_type == "Iron Condor":
        greeks['delta'] = very_lower_put_greeks['delta'] - lower_put_greeks['delta'] - upper_call_greeks['delta'] + very_upper_call_greeks['delta']
        greeks['gamma'] = very_lower_put_greeks['gamma'] - lower_put_greeks['gamma'] - upper_call_greeks['gamma'] + very_upper_call_greeks['gamma']
        greeks['theta'] = very_lower_put_greeks['theta'] - lower_put_greeks['theta'] - upper_call_greeks['theta'] + very_upper_call_greeks['theta']
        greeks['vega'] = very_lower_put_greeks['vega'] - lower_put_greeks['vega'] - upper_call_greeks['vega'] + very_upper_call_greeks['vega']
    
    return greeks

def var_calculator(strategies, quantities, spot_price, strikes, maturities, rates, vols,
                 confidence=0.95, horizon=1/252, n_simulations=10000,
                 use_t_dist=True, degrees_of_freedom=5, use_garch=False):
    """
    Calculate Value-at-Risk for an options portfolio using enhanced Monte Carlo simulation
    with Student's t-distribution and optional GARCH volatility modeling
    
    Parameters:
    -----------
    strategies : list
        List of strategy types
    quantities : list
        Number of positions for each strategy
    spot_price, strikes, maturities, rates, vols : lists
        Parameters for each position
    confidence : float
        Confidence level (default: 95%)
    horizon : float
        Risk horizon in years (default: 1 day)
    n_simulations : int
        Number of Monte Carlo simulations
    use_t_dist : bool
        Whether to use Student's t-distribution (True) or normal distribution (False)
    degrees_of_freedom : int
        Degrees of freedom for t-distribution (typically 3-10, lower = fatter tails)
    use_garch : bool
        Whether to use GARCH volatility modeling
    
    Returns:
    --------
    dict: VaR results and risk metrics
    """
    # Validate inputs
    if len(strategies) != len(quantities) or len(quantities) != len(strikes):
        raise ValueError("Input arrays must have the same length")
    
    n_positions = len(strategies)
    
    # Convert lists to arrays
    strikes = np.array(strikes)
    maturities = np.array(maturities)
    rates = np.array(rates) if isinstance(rates, (list, np.ndarray)) else np.ones(n_positions) * rates
    vols = np.array(vols) if isinstance(vols, (list, np.ndarray)) else np.ones(n_positions) * vols
    quantities = np.array(quantities)
    
    # Estimate portfolio volatility
    annual_vol = np.sqrt(np.mean(vols**2))  # Portfolio volatility estimate
    
    if use_garch:
        # Generate price paths with GARCH volatility
        n_steps = max(int(horizon * 252) + 1, 5)  # Ensure minimum steps for short horizons
        vol_paths = simulate_garch_volatility(annual_vol, n_steps, n_simulations)
        
        # Generate price paths with time-varying volatility
        price_paths = np.zeros((n_simulations, n_steps))
        price_paths[:, 0] = spot_price
        
        for i in range(n_simulations):
            for t in range(1, n_steps):
                # Daily drift and vol
                daily_horizon = 1/252
                drift = (rates.mean() - 0.5 * vol_paths[i, t-1]**2) * daily_horizon
                
                # Use t-distribution or normal distribution based on user choice
                if use_t_dist:
                    random_shock = np.random.standard_t(df=degrees_of_freedom)
                    # Scale to match volatility
                    diffusion = vol_paths[i, t-1] * np.sqrt(daily_horizon) * random_shock * np.sqrt((degrees_of_freedom-2)/degrees_of_freedom)
                else:
                    diffusion = vol_paths[i, t-1] * np.sqrt(daily_horizon) * np.random.normal()
                    
                price_paths[i, t] = price_paths[i, t-1] * np.exp(drift + diffusion)
        
        # Use final prices for VaR calculation
        simulated_prices = price_paths[:, -1]
    else:
        # Generate random price paths using t-distribution or normal distribution
        np.random.seed(42)  # For reproducibility
        
        if use_t_dist:
            # Student's t-distribution for fat tails
            # Generate t-distributed random variables
            t_random = np.random.standard_t(df=degrees_of_freedom, size=n_simulations)
            # Scale to match volatility and add drift
            price_changes = (rates.mean() - 0.5 * annual_vol**2) * horizon + \
                            annual_vol * np.sqrt(horizon) * t_random * np.sqrt((degrees_of_freedom-2)/degrees_of_freedom)
        else:
            # Regular normal distribution
            price_changes = np.random.normal(
                (rates.mean() - 0.5 * annual_vol**2) * horizon,
                annual_vol * np.sqrt(horizon),
                n_simulations
            )
        
        simulated_prices = spot_price * np.exp(price_changes)
    
    # Calculate current portfolio value
    current_portfolio_value = 0
    for i in range(n_positions):
        strategy = strategies[i]
        
        if strategy == "Protective Put":
            # For Protective Put: Current value = Stock value + Put value
            stock_value = spot_price
            put_value = black_scholes_calc(spot_price, strikes[i], maturities[i], rates[i], vols[i], 'put')
            position_value = stock_value + put_value
        elif strategy == "Covered Call Writing":
            # For Covered Call: Current value = Stock value - Call value
            stock_value = spot_price
            call_value = black_scholes_calc(spot_price, strikes[i], maturities[i], rates[i], vols[i], 'call')
            position_value = stock_value - call_value
        elif "Spread" in strategy or "Straddle" in strategy or "Iron" in strategy:
            # For spreads, straddles, and iron strategies, delegate to a specific calculator
            # This is a simplified approach - a real implementation would need separate logic for each
            # Just using a default option price for demonstration
            if "Call" in strategy:
                position_value = black_scholes_calc(spot_price, strikes[i], maturities[i], rates[i], vols[i], 'call')
            else:
                position_value = black_scholes_calc(spot_price, strikes[i], maturities[i], rates[i], vols[i], 'put')
        elif 'Call' in strategy:
            position_value = black_scholes_calc(spot_price, strikes[i], maturities[i], rates[i], vols[i], 'call')
        else:  # Put options
            position_value = black_scholes_calc(spot_price, strikes[i], maturities[i], rates[i], vols[i], 'put')
        
        current_portfolio_value += quantities[i] * position_value
    
    # Calculate simulated portfolio values
    simulated_portfolio_values = np.zeros(n_simulations)
    for j in range(n_simulations):
        sim_price = simulated_prices[j]
        portfolio_value = 0
        
        for i in range(n_positions):
            strategy = strategies[i]
            remaining_maturity = max(0, maturities[i] - horizon)
            
            if strategy == "Protective Put":
                # For Protective Put: Simulated value = Simulated stock value + Put value
                stock_value = sim_price
                put_value = black_scholes_calc(sim_price, strikes[i], remaining_maturity, rates[i], vols[i], 'put')
                position_value = stock_value + put_value
            elif strategy == "Covered Call Writing":
                # For Covered Call: Simulated value = Simulated stock value - Call value
                stock_value = sim_price
                call_value = black_scholes_calc(sim_price, strikes[i], remaining_maturity, rates[i], vols[i], 'call')
                position_value = stock_value - call_value
            elif "Bull Call Spread" in strategy:
                # Long lower strike call, short higher strike call
                lower_strike = strikes[i] * 0.9  # This is a simplification
                upper_strike = strikes[i] * 1.1  # This is a simplification
                long_call = black_scholes_calc(sim_price, lower_strike, remaining_maturity, rates[i], vols[i], 'call')
                short_call = black_scholes_calc(sim_price, upper_strike, remaining_maturity, rates[i], vols[i], 'call')
                position_value = long_call - short_call
            elif "Bear Call Spread" in strategy:
                # Short lower strike call, long higher strike call
                lower_strike = strikes[i] * 0.9  # This is a simplification
                upper_strike = strikes[i] * 1.1  # This is a simplification
                short_call = black_scholes_calc(sim_price, lower_strike, remaining_maturity, rates[i], vols[i], 'call')
                long_call = black_scholes_calc(sim_price, upper_strike, remaining_maturity, rates[i], vols[i], 'call')
                position_value = short_call - long_call
            elif "Bull Put Spread" in strategy:
                # Short higher strike put, long lower strike put
                lower_strike = strikes[i] * 0.9  # This is a simplification
                upper_strike = strikes[i] * 1.1  # This is a simplification
                short_put = black_scholes_calc(sim_price, upper_strike, remaining_maturity, rates[i], vols[i], 'put')
                long_put = black_scholes_calc(sim_price, lower_strike, remaining_maturity, rates[i], vols[i], 'put')
                position_value = short_put - long_put
            elif "Bear Put Spread" in strategy:
                # Long higher strike put, short lower strike put
                lower_strike = strikes[i] * 0.9  # This is a simplification
                upper_strike = strikes[i] * 1.1  # This is a simplification
                long_put = black_scholes_calc(sim_price, upper_strike, remaining_maturity, rates[i], vols[i], 'put')
                short_put = black_scholes_calc(sim_price, lower_strike, remaining_maturity, rates[i], vols[i], 'put')
                position_value = long_put - short_put
            elif "Long Straddle" in strategy:
                # Long call and long put
                call_value = black_scholes_calc(sim_price, strikes[i], remaining_maturity, rates[i], vols[i], 'call')
                put_value = black_scholes_calc(sim_price, strikes[i], remaining_maturity, rates[i], vols[i], 'put')
                position_value = call_value + put_value
            elif "Short Straddle" in strategy:
                # Short call and short put
                call_value = black_scholes_calc(sim_price, strikes[i], remaining_maturity, rates[i], vols[i], 'call')
                put_value = black_scholes_calc(sim_price, strikes[i], remaining_maturity, rates[i], vols[i], 'put')
                position_value = -(call_value + put_value)
            elif "Iron Butterfly" in strategy or "Iron Condor" in strategy:
                # These are complex strategies with 4 legs
                # Simplified approach here - a real implementation would need specific logic
                position_value = black_scholes_calc(sim_price, strikes[i], remaining_maturity, rates[i], vols[i], 'call')
            elif 'Call' in strategy:
                position_value = black_scholes_calc(sim_price, strikes[i], remaining_maturity, rates[i], vols[i], 'call')
            else:  # Put options
                position_value = black_scholes_calc(sim_price, strikes[i], remaining_maturity, rates[i], vols[i], 'put')
            
            portfolio_value += quantities[i] * position_value
            
        simulated_portfolio_values[j] = portfolio_value
    
    # Calculate P&L
    pnl = simulated_portfolio_values - current_portfolio_value
    
    # Sort P&L from worst to best
    sorted_pnl = np.sort(pnl)
    
    # Calculate VaR
    var_index = int(n_simulations * (1 - confidence))
    var = -sorted_pnl[var_index]
    
    # Calculate Expected Shortfall (Conditional VaR)
    es = -np.mean(sorted_pnl[:var_index])
    
    # Calculate additional risk metrics
    volatility = np.std(pnl)
    skewness = np.mean((pnl - np.mean(pnl))**3) / (volatility**3)
    kurtosis = np.mean((pnl - np.mean(pnl))**4) / (volatility**4) - 3
    
    return {
        'VaR': var,
        'Expected_Shortfall': es,
        'Volatility': volatility,
        'Skewness': skewness,
        'Kurtosis': kurtosis,
        'Worst_Case': -sorted_pnl[0],
        'Best_Case': sorted_pnl[-1],
        'Confidence_Level': confidence,
        'Horizon_Days': horizon * 252,
        'Distribution': 'Student-t' if use_t_dist else 'Normal',
        'DF': degrees_of_freedom if use_t_dist else None,
        'Volatility_Model': 'GARCH' if use_garch else 'Constant'
    }

def simulate_garch_volatility(initial_vol, n_steps, n_simulations, alpha=0.1, beta=0.8, omega=0.000001):
    """
    Simulate volatility paths using a GARCH(1,1) model
    
    Parameters:
    -----------
    initial_vol : float
        Initial volatility
    n_steps : int
        Number of time steps
    n_simulations : int
        Number of simulation paths
    alpha, beta, omega : float
        GARCH parameters
        
    Returns:
    --------
    numpy.ndarray: Matrix of volatility paths
    """
    # Initialize volatility array
    vol = np.zeros((n_simulations, n_steps+1))
    vol[:, 0] = initial_vol**2  # Variance
    
    # Simulate GARCH process
    for t in range(1, n_steps+1):
        # Generate random shocks
        z = np.random.standard_normal(n_simulations)
        
        # Calculate next period's variance using GARCH(1,1) formula
        # σ²(t) = ω + α * ε²(t-1) + β * σ²(t-1)
        # where ε(t-1) = σ(t-1) * z(t-1)
        epsilon_squared = vol[:, t-1] * z**2
        vol[:, t] = omega + alpha * epsilon_squared + beta * vol[:, t-1]
    
    # Convert variance to volatility
    return np.sqrt(vol)

def stress_test_portfolio(strategies, quantities, spot_price, strikes, maturities, rates, vols, stress_scenarios=None):
    """Perform stress testing on an options portfolio using historical scenarios"""
    # Convert string to list if needed
    if isinstance(strategies, str):
        strategies = [strategies]
    
    # Convert to lists if single values passed
    if not isinstance(quantities, (list, np.ndarray)):
        quantities = [quantities]
    if not isinstance(strikes, (list, np.ndarray)):
        strikes = [strikes]
    if not isinstance(maturities, (list, np.ndarray)):
        maturities = [maturities]
    
    # Ensure all lists have same length
    n = len(strategies)
    if len(quantities) == 1 and n > 1:
        quantities = quantities * n
    if len(strikes) == 1 and n > 1:
        strikes = strikes * n
    if len(maturities) == 1 and n > 1:
        maturities = maturities * n
    
    # Convert rates and vols to lists if they are single values
    if not isinstance(rates, (list, np.ndarray)):
        rates = [rates] * n
    if not isinstance(vols, (list, np.ndarray)):
        vols = [vols] * n
    
    # Ensure rates and vols have same length if they're lists
    if len(rates) == 1 and n > 1:
        rates = rates * n
    if len(vols) == 1 and n > 1:
        vols = vols * n
        
    # Default stress scenarios
    if stress_scenarios is None:
        stress_scenarios = {
            "2008_Crisis": {"price_change": -0.50, "vol_multiplier": 3.0, "rate_change": -0.02},
            "Covid_Crash": {"price_change": -0.30, "vol_multiplier": 2.5, "rate_change": -0.01},
            "Tech_Bubble": {"price_change": -0.40, "vol_multiplier": 2.0, "rate_change": 0.01},
            "2022_Inflation": {"price_change": -0.20, "vol_multiplier": 1.5, "rate_change": 0.03}
        }
    
    # Calculate current portfolio value
    current_portfolio_value = 0
    for i in range(len(strategies)):
        if 'Call' in strategies[i]:
            option_price = black_scholes_calc(spot_price, strikes[i], maturities[i], rates[i], vols[i], 'call')
        else:
            option_price = black_scholes_calc(spot_price, strikes[i], maturities[i], rates[i], vols[i], 'put')
        current_portfolio_value += quantities[i] * option_price
    
    # Apply stress scenarios
    results = {}
    for scenario_name, scenario in stress_scenarios.items():
        stressed_portfolio_value = 0
        stressed_spot = spot_price * (1 + scenario["price_change"])
        
        for i in range(len(strategies)):
            stressed_vol = vols[i] * scenario["vol_multiplier"]
            stressed_rate = max(0.001, rates[i] + scenario["rate_change"])
            
            if 'Call' in strategies[i]:
                option_price = black_scholes_calc(stressed_spot, strikes[i], maturities[i],
                                                stressed_rate, stressed_vol, 'call')
            else:
                option_price = black_scholes_calc(stressed_spot, strikes[i], maturities[i],
                                                stressed_rate, stressed_vol, 'put')
                
            stressed_portfolio_value += quantities[i] * option_price
        
        # Calculate P&L and percent change
        pnl = stressed_portfolio_value - current_portfolio_value
        pct_change = (pnl / current_portfolio_value) * 100 if current_portfolio_value != 0 else 0
        
        results[scenario_name] = {
            "P&L": pnl,
            "Portfolio_Change_Pct": pct_change,
            "Stressed_Value": stressed_portfolio_value,
            "Applied_Scenario": {
                "Price Change": f"{scenario['price_change']*100:.1f}%",
                "Volatility Multiplier": f"{scenario['vol_multiplier']:.1f}x",
                "Rate Change": f"{scenario['rate_change']*100:+.1f}%"
            }
        }
    
    return results

# Calculate Strategy Performance
def calculate_strategy_performance(strategy_type, spot_price, strike_price, time_to_maturity,
                                 risk_free_rate, volatility, call_value, put_value):
    """
    Calculate comprehensive performance metrics for option strategies
    
    Parameters:
    -----------
    strategy_type : str
        Type of option strategy
    spot_price, strike_price, time_to_maturity, risk_free_rate, volatility : float
        Market parameters
    call_value, put_value : float
        Current option values
    
    Returns:
    --------
    dict: Performance metrics
    """
    # Calculate strategy Greeks
    greeks = calculate_strategy_greeks(
        strategy_type, spot_price, strike_price, time_to_maturity, risk_free_rate, volatility
    )
    
    # Calculate advanced Greeks
    advanced_greeks = calculate_advanced_greeks("Call", spot_price, strike_price, time_to_maturity, risk_free_rate, volatility)
    
    # Define spot price range for P&L calculation
    spot_range = np.linspace(spot_price * 0.7, spot_price * 1.3, 100)
    
    # Calculate P&L
    pnl = calculate_strategy_pnl(
        strategy_type, spot_range, spot_price, strike_price,
        time_to_maturity, risk_free_rate, volatility, call_value, put_value
    )
    
    # Find break-even points
    be_indices = np.where(np.diff(np.signbit(pnl)))[0]
    break_even_points = [spot_range[i] for i in be_indices] if len(be_indices) > 0 else []
    
    # Profit probability estimation using lognormal distribution
    if len(break_even_points) > 0:
        # Sort break-even points
        break_even_points.sort()
        
        # Calculate probability below lower BE and above upper BE
        if len(break_even_points) == 1:
            # One break-even point
            be = break_even_points[0]
            if pnl[0] > 0:  # Profitable below BE
                profit_prob = norm.cdf(np.log(be/spot_price) / (volatility * np.sqrt(time_to_maturity)))
            else:  # Profitable above BE
                profit_prob = 1 - norm.cdf(np.log(be/spot_price) / (volatility * np.sqrt(time_to_maturity)))
        else:
            # Multiple break-even points, assume first and last define profitable region
            lower_be = break_even_points[0]
            upper_be = break_even_points[-1]
            
            if pnl[0] > 0:  # Profitable outside the range
                profit_prob = norm.cdf(np.log(lower_be/spot_price) / (volatility * np.sqrt(time_to_maturity))) + \
                             (1 - norm.cdf(np.log(upper_be/spot_price) / (volatility * np.sqrt(time_to_maturity))))
            else:  # Profitable inside the range
                profit_prob = norm.cdf(np.log(upper_be/spot_price) / (volatility * np.sqrt(time_to_maturity))) - \
                             norm.cdf(np.log(lower_be/spot_price) / (volatility * np.sqrt(time_to_maturity)))
    else:
        # No break-even points, strategy is always profitable or always unprofitable
        profit_prob = 1.0 if np.mean(pnl) > 0 else 0.0
    
    # Calculate maximum profit and loss
    max_profit = max(pnl)
    max_loss = abs(min(pnl))
    
    # Calculate risk-reward ratio
    risk_reward = max_profit / max_loss if max_loss > 0 else float('inf')
    
    # Calculate time value decay
    time_decay_rate = greeks['theta']
    
    # Calculate Sharpe-like ratio (expected return / volatility)
    expected_pnl = np.mean(pnl)
    pnl_volatility = np.std(pnl)
    sharpe = expected_pnl / pnl_volatility if pnl_volatility > 0 else 0
    
    # Kelly criterion - optimal position size
    if max_loss > 0:
        win_prob = profit_prob
        loss_prob = 1 - win_prob
        avg_win = max_profit
        avg_loss = max_loss
        kelly = (win_prob * avg_win - loss_prob * avg_loss) / (avg_win * avg_loss) if avg_win * avg_loss > 0 else 0
        kelly = max(0, min(1, kelly))  # Bound between 0 and 1
    else:
        kelly = 1  # No risk of loss
    
    # Return performance metrics
    return {
        'profitability': {
            'max_profit': max_profit,
            'max_loss': max_loss,
            'expected_pnl': expected_pnl,
            'break_even_points': break_even_points,
            'profit_probability': profit_prob,
            'risk_reward_ratio': risk_reward
        },
        'risk_metrics': {
            'delta': greeks['delta'],
            'gamma': greeks['gamma'],
            'theta': greeks['theta'],
            'vega': greeks['vega'],
            'vanna': advanced_greeks['vanna'],
            'volga': advanced_greeks['volga'],
            'pnl_volatility': pnl_volatility,
            'sharpe_ratio': sharpe,
            'kelly_criterion': kelly
        },
        'time_decay': {
            'daily_theta': greeks['theta'],
            'weekly_decay': greeks['theta'] * 5,
            'monthly_decay': greeks['theta'] * 21
        },
        'sensitivity': {
            'price_move_10pct_up': np.interp(spot_price * 1.1, spot_range, pnl) - np.interp(spot_price, spot_range, pnl),
            'price_move_10pct_down': np.interp(spot_price * 0.9, spot_range, pnl) - np.interp(spot_price, spot_range, pnl),
            'vol_move_up': calculate_strategy_pnl(strategy_type, [spot_price], spot_price, strike_price,
                                              time_to_maturity, risk_free_rate, volatility * 1.1, call_value, put_value)[0] -
                          calculate_strategy_pnl(strategy_type, [spot_price], spot_price, strike_price,
                                              time_to_maturity, risk_free_rate, volatility, call_value, put_value)[0]
        }
    }
    
# Risk Scenario Analysis
def risk_scenario_analysis(strategy, current_price, strike_price, time_to_maturity,
                         risk_free_rate, current_vol, pnl_function):
    """
    Perform stress testing and scenario analysis for an option strategy
    
    Parameters:
    -----------
    strategy : str
        Strategy type
    current_price, strike_price, time_to_maturity, risk_free_rate, current_vol : float
        Current market parameters
    pnl_function : function
        Function to calculate strategy P&L
    
    Returns:
    --------
    dict: Scenario analysis results
    """
    # Define scenarios
    price_scenarios = np.array([0.85, 0.90, 0.95, 1.0, 1.05, 1.10, 1.15]) * current_price
    vol_scenarios = np.array([0.7, 0.85, 1.0, 1.15, 1.3]) * current_vol
    time_scenarios = np.array([1/252, 5/252, 10/252, 21/252]) # 1, 5, 10, 21 days
    
    # Initialize results container
    results = {
        'price_impact': {},
        'vol_impact': {},
        'time_decay': {},
        'extreme_scenarios': {}
    }
    
    # Calculate P&L across price scenarios (keeping other factors constant)
    price_pnl = []
    for price in price_scenarios:
        spot_range = np.array([price])
        pnl = pnl_function(strategy, spot_range, current_price, strike_price,
                            time_to_maturity, risk_free_rate, current_vol)[0]
        price_pnl.append(pnl)
    
    results['price_impact'] = {
        'scenarios': price_scenarios,
        'pnl': price_pnl,
        'max_loss': min(price_pnl),
        'max_gain': max(price_pnl)
    }
    
    # Calculate P&L across volatility scenarios
    vol_pnl = []
    for vol in vol_scenarios:
        spot_range = np.array([current_price])
        pnl = pnl_function(strategy, spot_range, current_price, strike_price,
                            time_to_maturity, risk_free_rate, vol)[0]
        vol_pnl.append(pnl)
    
    results['vol_impact'] = {
        'scenarios': vol_scenarios,
        'pnl': vol_pnl,
        'max_loss': min(vol_pnl),
        'max_gain': max(vol_pnl)
    }
    
    # Calculate time decay impact
    time_pnl = []
    for time_left in time_scenarios:
        spot_range = np.array([current_price])
        pnl = pnl_function(strategy, spot_range, current_price, strike_price,
                            time_left, risk_free_rate, current_vol)[0]
        time_pnl.append(pnl)
    
    results['time_decay'] = {
        'scenarios': time_scenarios,
        'pnl': time_pnl,
        'effect': time_pnl[0] - time_pnl[-1]  # P&L difference between 1 day and 21 days
    }
    
    # Extreme scenarios
    extreme_scenarios = {
        'market_crash': pnl_function(strategy, np.array([current_price * 0.8]), current_price,
                                    strike_price, time_to_maturity, risk_free_rate, current_vol * 1.5)[0],
        'market_rally': pnl_function(strategy, np.array([current_price * 1.2]), current_price,
                                    strike_price, time_to_maturity, risk_free_rate, current_vol * 1.3)[0],
        'vol_explosion': pnl_function(strategy, np.array([current_price]), current_price,
                                     strike_price, time_to_maturity, risk_free_rate, current_vol * 2.0)[0],
        'vol_collapse': pnl_function(strategy, np.array([current_price]), current_price,
                                    strike_price, time_to_maturity, risk_free_rate, current_vol * 0.5)[0]
    }
    
    results['extreme_scenarios'] = extreme_scenarios
    
    return results