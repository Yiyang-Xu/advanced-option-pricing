import numpy as np
from scipy.stats import norm
from scipy.interpolate import RectBivariateSpline, interp1d

# Black-Scholes Option Pricing Model
def black_scholes_calc(S, K, T, r, sigma, option_type='call'):
    """
    Black-Scholes pricing model for European options
    
    Parameters:
    -----------
    S : float
        Current stock price
    K : float
        Strike price
    T : float
        Time to expiration in years
    r : float
        Risk-free interest rate (decimal)
    sigma : float
        Volatility (decimal)
    option_type : str
        'call' or 'put'
        
    Returns:
    --------
    float: Option price
    """
    if T <= 0 or sigma <= 0:
        # Handle edge cases for expiration or zero volatility
        if option_type == 'call':
            return max(0, S - K) if S > K else 0
        else:  # put
            return max(0, K - S) if K > S else 0
    
    # Calculate d1 and d2 parameters
    d1 = (np.log(S/K) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    # Calculate option price based on type
    if option_type == 'call':
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    else:  # put
        return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
    
# Cox-Ross-Rubinstein Binomial Option Pricing Model
def binomial_calc(S, K, T, r, sigma, n, option_type='call', style='european'):
    """
    Binomial option pricing model
    
    Parameters:
    -----------
    S : float
        Current stock price
    K : float
        Strike price
    T : float
        Time to expiration in years
    r : float
        Risk-free interest rate (decimal)
    sigma : float
        Volatility (decimal)
    n : int
        Number of time steps
    option_type : str
        'call' or 'put'
    style : str
        'european' or 'american'
        
    Returns:
    --------
    float: Option price
    """
    dt = T/n
    u = np.exp(sigma*np.sqrt(dt))  # Up factor
    d = 1/u                         # Down factor
    p = (np.exp(r*dt) - d)/(u - d)  # Risk-neutral probability
    
    # Initialize stock price tree
    stock = np.zeros((n+1, n+1))
    stock[0,0] = S
    
    # Generate stock price tree
    for i in range(1, n+1):
        stock[0:i+1,i] = S * u**np.arange(i,-1,-1) * d**np.arange(0,i+1)
    
    # Initialize option value tree
    option = np.zeros((n+1, n+1))
    
    # Set terminal option values (at expiration)
    if option_type == 'call':
        option[:,n] = np.maximum(stock[:,n] - K, 0)
    else:  # put
        option[:,n] = np.maximum(K - stock[:,n], 0)
    
    # Backward recursion for option values
    for j in range(n-1,-1,-1):
        for i in range(j+1):
            if style == 'european':
                # European option: simply use risk-neutral pricing
                option[i,j] = np.exp(-r*dt)*(p*option[i,j+1] + (1-p)*option[i+1,j+1])
            else:  # american
                # American option: consider early exercise
                hold = np.exp(-r*dt)*(p*option[i,j+1] + (1-p)*option[i+1,j+1])
                if option_type == 'call':
                    exercise = stock[i,j] - K
                else:
                    exercise = K - stock[i,j]
                option[i,j] = max(hold, exercise)
    
    return option[0,0]

# Monte Carlo Simulation for Option Pricing
def monte_carlo_calc(S, K, T, r, sigma, n_sim, n_steps, option_type='call'):
    """
    Monte Carlo simulation for option pricing
    
    Parameters:
    -----------
    S : float
        Current stock price
    K : float
        Strike price
    T : float
        Time to expiration in years
    r : float
        Risk-free interest rate (decimal)
    sigma : float
        Volatility (decimal)
    n_sim : int
        Number of price path simulations
    n_steps : int
        Number of time steps per path
    option_type : str
        'call' or 'put'
        
    Returns:
    --------
    tuple: (option_price, standard_error, price_paths)
    """
    dt = T/n_steps
    nudt = (r - 0.5*sigma**2)*dt
    sigsqrtdt = sigma*np.sqrt(dt)
    
    # Generate random standard normal samples
    Z = np.random.standard_normal((n_sim, n_steps))
    
    # Initialize price paths array
    S_path = np.zeros((n_sim, n_steps+1))
    S_path[:,0] = S
    
    # Simulate price paths using Geometric Brownian Motion
    for t in range(1, n_steps+1):
        S_path[:,t] = S_path[:,t-1] * np.exp(nudt + sigsqrtdt*Z[:,t-1])
    
    # Calculate payoffs at expiration
    if option_type == 'call':
        payoffs = np.maximum(S_path[:,-1] - K, 0)
    else:  # put
        payoffs = np.maximum(K - S_path[:,-1], 0)
    
    # Calculate option price (present value of expected payoff)
    price = np.exp(-r*T)*np.mean(payoffs)
    
    # Calculate standard error
    se = np.exp(-r*T)*np.std(payoffs)/np.sqrt(n_sim)
    
    return price, se, S_path

def implied_volatility(market_price, S, K, T, r, option_type='call', precision=1e-8, max_iterations=20):
    """
    Calculate implied volatility using enhanced analytical approximation and Newton-Raphson refinement
    
    Parameters:
    -----------
    market_price : float
        Observed market price of the option
    S, K, T, r : float
        Stock price, strike price, time to maturity (years), risk-free rate
    option_type : str
        'call' or 'put'
    precision : float
        Convergence threshold
    max_iterations : int
        Maximum number of Newton-Raphson iterations
        
    Returns:
    --------
    float: Implied volatility value
    """
    # Handle degenerate cases
    if T <= 0.01:
        return 0.3  # Default for very short-dated options
    
    # Set a more reasonable minimum volatility floor
    min_sigma = 0.05  # 5% minimum volatility
    
    # Calculate intrinsic value
    if option_type.lower() == 'call':
        intrinsic = max(0, S - K * np.exp(-r * T))
    else:  # put
        intrinsic = max(0, K * np.exp(-r * T) - S)
    
    # Time value
    time_value = max(0.01, market_price - intrinsic)
    
    # Moneyness
    moneyness = np.log(S / K) + r * T
    
    # Initial guess selection based on option characteristics
    if option_type.lower() == 'call':
        if K > S:  # OTM call
            # For OTM calls with low market prices, start with a reasonable volatility
            # instead of letting the algorithm drive it too low
            sigma = 0.15  # Start with 15% for OTM options
            
            # If market price is very low relative to spot, adjust initial guess
            if market_price < 0.01 * S:
                # Use bisection to find better initial guess
                low_vol, high_vol = 0.05, 0.5
                for _ in range(5):
                    mid_vol = (low_vol + high_vol) / 2
                    price = black_scholes_calc(S, K, T, r, mid_vol, option_type)
                    if price < market_price:
                        low_vol = mid_vol
                    else:
                        high_vol = mid_vol
                sigma = (low_vol + high_vol) / 2
        else:  # ITM or ATM call
            if abs(moneyness) < 0.1:  # Near ATM
                # Brenner-Subrahmanyam approximation for near-ATM options
                sigma = np.sqrt(2 * np.pi / T) * time_value / (S * np.exp(-r * T))
            else:  # ITM
                sigma = 0.2  # Start with 20% for ITM options
    else:  # put
        if K < S:  # OTM put
            # Similar approach for OTM puts
            sigma = 0.15  # Start with 15% for OTM options
            
            # Similar bisection for low-price OTM puts
            if market_price < 0.01 * S:
                low_vol, high_vol = 0.05, 0.5
                for _ in range(5):
                    mid_vol = (low_vol + high_vol) / 2
                    price = black_scholes_calc(S, K, T, r, mid_vol, option_type)
                    if price < market_price:
                        low_vol = mid_vol
                    else:
                        high_vol = mid_vol
                sigma = (low_vol + high_vol) / 2
        else:  # ITM or ATM put
            if abs(moneyness) < 0.1:  # Near ATM
                sigma = np.sqrt(2 * np.pi / T) * time_value / (S * np.exp(-r * T))
            else:  # ITM
                sigma = 0.2  # Start with 20% for ITM options
                
    # Ensure initial guess is reasonable
    sigma = max(min_sigma, min(sigma, 1.0))
    
    # Newton-Raphson refinement with robust safeguards
    for i in range(max_iterations):
        # Calculate option price
        price = black_scholes_calc(S, K, T, r, sigma, option_type)
        
        # Calculate difference
        diff = price - market_price
        
        # Check convergence
        if abs(diff) < precision:
            return sigma
        
        # Calculate vega (derivative of price with respect to volatility)
        d1 = (np.log(S/K) + (r + sigma**2/2)*T) / (sigma * np.sqrt(T))
        vega = S * np.sqrt(T) * norm.pdf(d1)
        
        # More robust handling for tiny vegas
        if abs(vega) < 1e-5:
            vega = 1e-5 * (1 if vega >= 0 else -1)
        
        # Newton-Raphson update with dampening for stability
        delta_sigma = diff / vega
        
        # Apply dampening for large steps
        if abs(delta_sigma) > 0.1:
            delta_sigma = 0.1 * (delta_sigma / abs(delta_sigma))
            
        sigma = sigma - delta_sigma
        
        # Ensure sigma stays within reasonable bounds
        sigma = max(min_sigma, min(sigma, 1.0))
    
    # If we didn't converge, try grid search as a last resort
    if abs(price - market_price) > precision:
        vol_range = np.linspace(min_sigma, 1.0, 20)
        best_vol = min_sigma
        min_diff = float('inf')
        
        for vol in vol_range:
            price = black_scholes_calc(S, K, T, r, vol, option_type)
            diff = abs(price - market_price)
            
            if diff < min_diff:
                min_diff = diff
                best_vol = vol
        
        return best_vol
    
    return sigma

# Calculate option prices
def calculate_option_prices(model_type, current_price, strike_price, time_to_maturity, risk_free_rate, volatility, model_params):
    """Calculate option prices based on the selected model"""
    if model_type == "Black-Scholes":
        call_value = black_scholes_calc(current_price, strike_price, time_to_maturity,
                                      risk_free_rate, volatility, 'call')
        put_value = black_scholes_calc(current_price, strike_price, time_to_maturity,
                                     risk_free_rate, volatility, 'put')
        return {"call": f"${call_value:.2f}", "put": f"${put_value:.2f}"}, call_value, put_value, None
        
    elif model_type == "Binomial":
        steps = model_params.get('steps', 100)
        option_style = model_params.get('option_style', 'European').lower()
        call_value = binomial_calc(current_price, strike_price, time_to_maturity,
                                 risk_free_rate, volatility, steps, 'call', option_style)
        put_value = binomial_calc(current_price, strike_price, time_to_maturity,
                                risk_free_rate, volatility, steps, 'put', option_style)
        return {"call": f"${call_value:.2f}", "put": f"${put_value:.2f}"}, call_value, put_value, None
        
    else:  # Monte Carlo
        n_simulations = model_params.get('n_simulations', 10000)
        n_steps = model_params.get('n_steps', 100)
        call_value, call_se, call_paths = monte_carlo_calc(current_price, strike_price,
                                                         time_to_maturity, risk_free_rate,
                                                         volatility, n_simulations, n_steps, 'call')
        put_value, put_se, put_paths = monte_carlo_calc(current_price, strike_price,
                                                      time_to_maturity, risk_free_rate,
                                                      volatility, n_simulations, n_steps, 'put')
        price_info = {
            "call": f"${call_value:.2f} ± ${call_se:.4f}",
            "put": f"${put_value:.2f} ± ${put_se:.4f}"
        }
        return price_info, call_value, put_value, (call_paths, put_paths, n_steps)
    
# Local Volatility Surface Calculation
def enhanced_local_volatility_surface(strikes, maturities, implied_vols, spot, rates,
                               dividend_yield=0.0, smoothing_level=0.1):
    """
    Generate enhanced local volatility surface using Dupire's formula with proper mathematical 
    formulation and numerical stability improvements
    
    Parameters:
    -----------
    strikes : array-like
        Array of strike prices
    maturities : array-like
        Array of maturities (in years)
    implied_vols : 2D array-like
        Matrix of implied volatilities for each strike/maturity pair
    spot : float
        Current spot price
    rates : array-like or float
        Risk-free rates for each maturity or a single rate
    dividend_yield : float or array-like
        Continuous dividend yield(s)
    smoothing_level : float
        Controls smoothing of input data (0.0-1.0)
        
    Returns:
    --------
    tuple: (local_vol_surface, smoothed_implied_vols, diagnostics)
        - local_vol_surface: Matrix of local volatilities
        - smoothed_implied_vols: Matrix of smoothed implied volatilities
        - diagnostics: Dictionary with diagnostic information
    """
    # Ensure inputs are numpy arrays
    strikes = np.array(strikes, dtype=float)
    maturities = np.array(maturities, dtype=float)
    implied_vols = np.array(implied_vols, dtype=float)
    
    # Convert rates and dividend yield to arrays if they're scalars
    if np.isscalar(rates):
        rates = np.full_like(maturities, rates)
    if np.isscalar(dividend_yield):
        dividend_yield = np.full_like(maturities, dividend_yield)
    
    # Diagnostics container
    diagnostics = {
        'boundary_points': 0,
        'denominator_fixes': 0,
        'extreme_values_clipped': 0
    }
    
    # 1. Smooth the implied volatility surface to reduce noise in derivatives
    # Adjust smoothing factor based on user input
    smoothing_factor = smoothing_level * (len(strikes) * len(maturities))
    
    # Create spline with appropriate smoothing
    iv_spline = RectBivariateSpline(
        maturities, strikes, implied_vols,
        kx=min(3, len(maturities)-1),
        ky=min(3, len(strikes)-1),
        s=smoothing_factor
    )
    
    # Create a denser grid for more accurate derivatives
    dense_maturities = np.linspace(maturities.min(), maturities.max(), max(50, len(maturities)*2))
    dense_strikes = np.linspace(strikes.min(), strikes.max(), max(50, len(strikes)*2))
    
    # Evaluate smoothed implied volatility on the dense grid
    K_dense, T_dense = np.meshgrid(dense_strikes, dense_maturities)
    smoothed_iv = iv_spline(dense_maturities, dense_strikes)
    
    # 2. Interpolate rates and dividends to match the dense grid
    rate_interp = interp1d(maturities, rates, kind='linear', fill_value='extrapolate')
    div_interp = interp1d(maturities, dividend_yield, kind='linear', fill_value='extrapolate')
    
    dense_rates = rate_interp(dense_maturities)
    dense_dividends = div_interp(dense_maturities)
    
    # 3. Calculate option prices from implied volatilities
    # This is critical - Dupire's formula applies to option prices, not volatilities directly
    option_prices = np.zeros_like(smoothed_iv)
    for i, t in enumerate(dense_maturities):
        for j, k in enumerate(dense_strikes):
            # Use Black-Scholes to get option prices from implied vols
            option_prices[i, j] = black_scholes_calc(
                spot, k, t, dense_rates[i] - dense_dividends[i], smoothed_iv[i, j], 'call'
            )
    
    # 4. Calculate derivatives with proper boundary handling
    # Initialize derivative arrays
    dC_dT = np.zeros_like(option_prices)
    dC_dK = np.zeros_like(option_prices)
    d2C_dK2 = np.zeros_like(option_prices)
    
    # Time steps for finite difference calculations
    dt_forward = np.diff(dense_maturities)
    dt_backward = np.diff(dense_maturities, prepend=dense_maturities[0])
    
    # Strike steps
    dk_forward = np.diff(dense_strikes)
    dk_backward = np.diff(dense_strikes, prepend=dense_strikes[0])
    
    # Calculate time derivatives (dC/dT)
    for j in range(len(dense_strikes)):
        # Forward difference for first point
        dC_dT[0, j] = (option_prices[1, j] - option_prices[0, j]) / dt_forward[0]
        
        # Central difference for interior points
        for i in range(1, len(dense_maturities)-1):
            dt_central = dense_maturities[i+1] - dense_maturities[i-1]
            dC_dT[i, j] = (option_prices[i+1, j] - option_prices[i-1, j]) / dt_central
        
        # Backward difference for last point
        last_idx = len(dense_maturities)-1
        dC_dT[last_idx, j] = (option_prices[last_idx, j] - option_prices[last_idx-1, j]) / dt_backward[last_idx]
    
    # Calculate strike derivatives (dC/dK and d2C/dK2)
    for i in range(len(dense_maturities)):
        # Forward differences for first point
        dC_dK[i, 0] = (option_prices[i, 1] - option_prices[i, 0]) / dk_forward[0]
        d2C_dK2[i, 0] = (option_prices[i, 2] - 2*option_prices[i, 1] + option_prices[i, 0]) / (dk_forward[0]**2)
        diagnostics['boundary_points'] += 1
        
        # Central differences for interior points
        for j in range(1, len(dense_strikes)-1):
            dk = dense_strikes[j+1] - dense_strikes[j-1]
            dC_dK[i, j] = (option_prices[i, j+1] - option_prices[i, j-1]) / dk
            d2C_dK2[i, j] = (option_prices[i, j+1] - 2*option_prices[i, j] + option_prices[i, j-1]) / ((dense_strikes[j+1] - dense_strikes[j-1])/2)**2
        
        # Backward differences for last point
        last_idx = len(dense_strikes)-1
        dC_dK[i, last_idx] = (option_prices[i, last_idx] - option_prices[i, last_idx-1]) / dk_backward[last_idx]
        d2C_dK2[i, last_idx] = (option_prices[i, last_idx] - 2*option_prices[i, last_idx-1] + option_prices[i, last_idx-2]) / (dk_backward[last_idx]**2)
        diagnostics['boundary_points'] += 1
    
    # 5. Apply Dupire's formula with robust numerical safeguards
    local_vol = np.zeros_like(option_prices)
    
    for i in range(len(dense_maturities)):
        for j in range(len(dense_strikes)):
            # Extract parameters at this grid point
            r = dense_rates[i]
            q = dense_dividends[i]
            K = dense_strikes[j]
            T = dense_maturities[i]
            
            # Proper Dupire formula
            # numerator = dC/dT + (r-q)*K*dC/dK + q*C
            numerator = dC_dT[i, j] + (r - q) * K * dC_dK[i, j] + q * option_prices[i, j]
            
            # denominator = 0.5 * K^2 * d2C/dK2
            denominator = 0.5 * K**2 * d2C_dK2[i, j]
            
            # Apply robust numerical safeguards
            if denominator > 1e-6 and numerator > 0:
                local_vol[i, j] = np.sqrt(numerator / denominator)
            else:
                # Fallback to implied vol if formula gives invalid result
                local_vol[i, j] = smoothed_iv[i, j]
                diagnostics['denominator_fixes'] += 1
            
            # Apply reasonable bounds for stability
            if not (0.01 <= local_vol[i, j] <= 2.0):
                local_vol[i, j] = np.clip(local_vol[i, j], 0.01, 2.0)
                diagnostics['extreme_values_clipped'] += 1
    
    # 6. Smooth the output local volatility surface to remove artifacts
    output_smoothing = smoothing_factor * 2  # More aggressive smoothing for output
    local_vol_spline = RectBivariateSpline(
        dense_maturities, dense_strikes, local_vol,
        kx=min(3, len(dense_maturities)-1),
        ky=min(3, len(dense_strikes)-1),
        s=output_smoothing
    )
    
    # 7. Interpolate back to the original grid points for consistency
    result_local_vol = local_vol_spline(maturities, strikes)
    result_implied_vol = iv_spline(maturities, strikes)
    
    # Add grid information to diagnostics
    diagnostics['grid_info'] = {
        'original_size': (len(maturities), len(strikes)),
        'dense_size': (len(dense_maturities), len(dense_strikes)),
        'smoothing_factor': smoothing_factor
    }
    
    return result_local_vol, result_implied_vol, diagnostics