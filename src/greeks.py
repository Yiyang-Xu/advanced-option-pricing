import numpy as np
from scipy.stats import norm

# Calculate First-Order Greeks
def calculate_greeks(option_type, S, K, T, r, sigma):
    """
    Calculate first-order option Greeks
    
    Parameters:
    -----------
    option_type : str
        "Call" or "Put"
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
        
    Returns:
    --------
    dict: Dictionary of first-order Greeks (delta, gamma, theta, vega)
    """
    if T <= 0 or sigma <= 0:
        # Handle edge cases
        if option_type == "Call":
            return {
                'delta': 1.0 if S > K else 0.0,
                'gamma': 0.0,
                'theta': 0.0,
                'vega': 0.0
            }
        else:  # Put
            return {
                'delta': -1.0 if S < K else 0.0,
                'gamma': 0.0,
                'theta': 0.0,
                'vega': 0.0
            }
    
    # Calculate d1 and d2 parameters
    d1 = (np.log(S/K) + (r + sigma**2/2)*T)/(sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    # Common calculations
    n_d1 = norm.pdf(d1)  # Standard normal probability density at d1
    
    # Gamma - second derivative of option price with respect to underlying price
    # Same for both call and put
    gamma = n_d1/(S * sigma * np.sqrt(T))
    
    # Vega - first derivative of option price with respect to volatility
    # Same for both call and put, typically expressed per 1% change in volatility
    vega = S * np.sqrt(T) * n_d1 * 0.01
    
    if option_type == "Call":
        # Delta - first derivative of option price with respect to underlying price
        delta = norm.cdf(d1)
        
        # Theta - first derivative of option price with respect to time
        # Typically expressed as daily decay (divided by 365)
        theta = (-S * n_d1 * sigma/(2 * np.sqrt(T)) -
                r * K * np.exp(-r * T) * norm.cdf(d2)) / 365.0
    else:  # Put
        delta = -norm.cdf(-d1)
        theta = (-S * n_d1 * sigma/(2 * np.sqrt(T)) +
                r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365.0
    
    return {
        'delta': delta,
        'gamma': gamma,
        'theta': theta,
        'vega': vega
    }

# Calculate Higher-Order Greeks
def calculate_advanced_greeks(option_type, S, K, T, r, sigma):
    """
    Calculate higher-order option Greeks
    
    Parameters:
    -----------
    option_type : str
        "call" or "put"
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
        
    Returns:
    --------
    dict: Dictionary of higher-order Greeks
    """
    if T <= 0 or sigma <= 0:
        return {
            'vanna': 0.0, 'charm': 0.0, 'volga': 0.0,
            'veta': 0.0, 'speed': 0.0, 'zomma': 0.0,
            'color': 0.0, 'ultima': 0.0
        }
    
    # Calculate parameters
    sqrt_t = np.sqrt(T)
    d1 = (np.log(S/K) + (r + sigma**2/2)*T)/(sigma*sqrt_t)
    d2 = d1 - sigma*sqrt_t
    
    # Standard normal PDF values
    nd1 = norm.pdf(d1)
    nd2 = norm.pdf(d2)
    
    # Higher-order Greeks (most are same for calls and puts)
    
    # Vanna/DdeltaDvol - sensitivity of delta to volatility changes
    vanna = -nd1 * d2 / sigma
    
    # Volga/Vomma - second derivative of option price with respect to volatility
    volga = S * sqrt_t * nd1 * d1 * d2 / sigma
    
    # Charm/DdeltaDtime - rate of change of delta with respect to time
    if option_type.lower() == 'call':
        charm = -nd1 * (r/(sigma*sqrt_t) - d2/(2*T))
    else:  # put
        charm = nd1 * (r/(sigma*sqrt_t) - d2/(2*T))
    
    # Veta/DvegaDtime - rate of change of vega with respect to time
    veta = -S * nd1 * sqrt_t * (r*d1/(sigma*sqrt_t) - (1+d1*d2)/(2*T))
    
    # Speed - third derivative of option price with respect to underlying price
    speed = -nd1 * d1/(S**2 * sigma * sqrt_t) * (1 + d1/(sigma * sqrt_t))
    
    # Zomma - sensitivity of gamma to volatility changes
    zomma = nd1 * (d1*d2 - 1)/(S * sigma)
    
    # Color/DgammaDtime - rate of change of gamma with respect to time
    color = -nd1 * (r*d2 + d1*d2/(2*T) - (1+d1*d2)/(2*T) + r*d1/(sigma*sqrt_t))/(S * sigma * sqrt_t)
    
    # Ultima - sensitivity of volga to volatility changes
    ultima = -S * sqrt_t * nd1 / (sigma**2) * (d1*d2*(1-d1*d2) + d1**2 + d2**2)
    
    return {
        'vanna': vanna, 'charm': charm, 'volga': volga,
        'veta': veta, 'speed': speed, 'zomma': zomma,
        'color': color, 'ultima': ultima
    }