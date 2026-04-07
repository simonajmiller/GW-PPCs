import numpy as np
import matplotlib.colors as mcolors

"""
Plot settings (colors, labels, etc.)
"""

run_labels = {
    'gaussian_likelihood_sigma_0.1':r'Gaussian $\mathcal{L}$, $\sigma_{\rm meas} = 0.1$',
    'gaussian_likelihood_sigma_0.3':r'Gaussian $\mathcal{L}$, $\sigma_{\rm meas} = 0.3$',
    'gaussian_likelihood_sigma_0.5':r'Gaussian $\mathcal{L}$, $\sigma_{\rm meas} = 0.5$',
    'bilby_likelihood':r'Realistic O3 noise $\mathcal{L}$'
}
run_labels_full = {
    'gaussian_likelihood_sigma_0.1':r'Gaussian likelihood, $\sigma_{\rm meas} = 0.1$',
    'gaussian_likelihood_sigma_0.3':r'Gaussian likelihood, $\sigma_{\rm meas} = 0.3$',
    'gaussian_likelihood_sigma_0.5':r'Gaussian likelihood, $\sigma_{\rm meas} = 0.5$',
    'bilby_likelihood':'Realistic O3 noise likelihood'
}
run_labels_short = {
    'gaussian_likelihood_sigma_0.1':r'$\sigma_{\rm meas} = 0.1$',
    'gaussian_likelihood_sigma_0.3':r'$\sigma_{\rm meas} = 0.3$',
    'gaussian_likelihood_sigma_0.5':r'$\sigma_{\rm meas} = 0.5$',
    'bilby_likelihood':'Realistic O3 noise'
}
run_colors = {
    'gaussian_likelihood_sigma_0.1':'#FF5733', 
    'gaussian_likelihood_sigma_0.3':'#6A6BAD',
    'gaussian_likelihood_sigma_0.5':'#0099FF',
    'bilby_likelihood':'#A1C935',
}
run_colors_darker = {
    'gaussian_likelihood_sigma_0.1':'#882E1B', 
    'gaussian_likelihood_sigma_0.3':'#41426D',
    'gaussian_likelihood_sigma_0.5':'#01609E',
    'bilby_likelihood':'#5E751E',
}

GWTC4_chi_color = '#ff9408'
GWTC4_tilt_color = '#6488ea'

GWTC4_labels_data_vs_event = {
    'a':{
        'event':{
            'label':r'$\chi_{\rm true}$',
            'label pred':r'$\chi_{\rm true}^{\rm pred}$',
            'label obs':r'$\chi_{\rm true}^{\rm obs}$',
        }, 
        'data':{
            'label':r'$\chi_{\mathrm{max}.\mathcal{L}}$',
            'label pred':r'$\chi_{\mathrm{max}.\mathcal{L}}^{\rm pred}$',
            'label obs':r'$\chi_{\mathrm{max}.\mathcal{L}}^{\rm obs}$',
        }
    },
    'tilt':{
        'event':{
            'label':r'$\cos\theta_{\rm true}$',
            'label pred':r'$\cos\theta_{\rm true}^{\rm pred}$',
            'label obs':r'$\cos\theta_{\rm true}^{\rm obs}$',
        }, 
        'data':{
            'label':r'$\cos\theta_{\mathrm{max}.\mathcal{L}}$',
            'label pred':r'$\cos\theta_{\mathrm{max}.\mathcal{L}}^{\rm pred}$',
            'label obs':r'$\cos\theta_{\mathrm{max}.\mathcal{L}}^{\rm obs}$',
        }
    }
}


"""
Functions
"""

def average_curve(pred_obs, param): 
    """
    Compute the average predicted and observed curves for a given parameter.

    Parameters
    ----------
    pred_obs : dict
        Dictionary containing 'predicted' and 'observed' arrays for each parameter.
        Each should be of shape (ncat, nevents).
    param : str
        Name of the parameter to compute averages for.

    Returns
    -------
    obs_avgs : np.ndarray
        Average of observed curves.
    pred_avgs : np.ndarray
        Average of predicted curves.
    """

    allpred = np.sort(pred_obs['predicted'][param])
    allobs = np.sort(pred_obs['observed'][param])
    
    pred_avgs = np.average(allpred, axis=0)
    obs_avgs = np.average(allobs, axis=0)

    return obs_avgs, pred_avgs

def calc_slope(x, y):
    """
    Compute the slope and intercept of y vs x using least squares.

    Parameters
    ----------
    x : np.ndarray
        Independent variable array.
    y : np.ndarray
        Dependent variable array.

    Returns
    -------
    slope : float
        Slope of the best-fit line.
    intercept : float
        Intercept of the best-fit line. Returns (np.inf, np.inf) 
        if matrix inversion fails.
    """
    
    X = np.zeros(shape=(x.size, 2))
    X[:, 0] = x
    X[:, 1] = np.zeros(x.size) + 1.
    Xt = np.transpose(X)

    try: 
        Xmat = np.matmul( np.linalg.inv ( np.matmul(Xt, X) ), Xt )
    except:
        return np.inf, np.inf

    slope, intercept = np.matmul(Xmat, y)

    return slope, intercept 

def fraction_underpredicted(pred_obs, params, ncut=4, nbins=50):
    """
    Compute the fraction of underpredicted points for each parameter over binned
    PPC slopes.

    Parameters
    ----------
    pred_obs : dict
        Dictionary with 'predicted' and 'observed' arrays for each parameter.
    params : list of str
        Parameters to evaluate.
    ncut : int, optional
        Sliding window size for slope calculation. Default is 4.
    nbins : int, optional
        Number of bins to divide the x-axis for slope fractions. Default is 50.

    Returns
    -------
    slopeDict : dict
        Dictionary of slope arrays for each parameter.
    percDict : dict
        Dictionary of fractions underpredicted for each bin for each parameter.
    """
    
    slopeDict, percDict = {}, {}

    for param in params: 
        ncat, nevents = pred_obs['predicted'][param].shape
        nxs = nevents-ncut-1

        # Get slopes of PPC traces
        slopes, xs = np.zeros((ncat, nxs)), np.zeros((ncat, nxs))
        for n in range(ncat):
            pred = pred_obs['predicted'][param][n]
            obs = pred_obs['observed'][param][n]

            pred_noendpoints = pred[int(ncut/2):-int(ncut/2)]
            xs[n,:] = np.array([0.5*(pred_noendpoints[i] + pred_noendpoints[i+1]) for i in range(len(pred_noendpoints)-1)])
            for i in range(nxs):
                slopes[n,i] = calc_slope(pred[i:i+ncut], obs[i:i+ncut])[0]

        slopeDict[param] = { 'slope_all': slopes , 'xs' : xs,}
                
        # Bin the x-axis and calculate fraction > 1 for slopes in each bin
        xarr = np.concatenate(xs)
        xmin = np.min(xarr)
        xmax = np.max(xarr)
        xbins = np.linspace(xmin, xmax, nbins)
        dx = xbins[1] - xbins[0]
        midpoints = xbins[:-1] + dx

        percs, Ns = [], []
        for k in range(len(midpoints)):
            low = xbins[k]
            high = xbins[k+1]

            inbound = (xs >= low) & (xs <= high)
            xin = xs[inbound]
            slopein = slopes[inbound]

            n = len(slopein)
            if n==0:
                p = np.nan
            else:
                p = sum(slopein < 1) / n 
                        
            percs.append(p)
            Ns.append(n)
            
        percDict[param] = {
            'xs' : midpoints, 
            'fraction' : np.array(percs), 
            'N' : np.array(Ns)
        }  

    return slopeDict, percDict

def zero_positive_spin(x):
    """
    Compute ratio of 'zero' spins to positive spins for cos(theta) values.

    Parameters
    ----------
    x : np.ndarray
        Array of cos(theta) values.

    Returns
    -------
    float
        Ratio of counts with |x| <= 0.33 to counts with x > 0.33.
    """
    positive_frac = np.sum(x > .33)
    zero_frac = np.sum( (x <= .33) & (x >= -.33) )
    return zero_frac / positive_frac
    
def fraction_tails(x): 
    """
    Compute the fraction of values exceeding |0.5|.

    Parameters
    ----------
    x : np.ndarray
        Array of values.

    Returns
    -------
    float
        Fraction of values with absolute magnitude > 0.5.
    """
    return sum(np.abs(x)>0.5)/len(x)

def get_test_statistics(pred_obs):
    """
    Compute test statistics for all events and parameters in a prediction-
    observation dataset.

    Parameters
    ----------
    pred_obs : dict
        Nested dictionary of predicted and observed values.

    Returns
    -------
    T_dict : dict
        Nested dictionary of test statistics for each category.
    """
    traces_dict = {
    k:{
        i:{
            'y_rep':np.concatenate([v['predicted']['costheta1'][i], v['predicted']['costheta2'][i]]), 
            'y':np.concatenate([v['observed']['costheta1'][i], v['observed']['costheta2'][i]])
        } for i in range(len(v['predicted']['costheta1']))
    }
        for k,v in pred_obs.items()  
    }
    T_dict = {k:get_Tdict(v) for k,v in traces_dict.items()}
    return T_dict

zp_title = r'$\frac{N(\cos\theta\in[-0.33, 0.33])}{N(\cos\theta\in[0.33, 1])}$'    

def get_Tdict(PPC_traces_event, verbose=False):
    """
    Compute p-values comparing observed and replicated test statistics.

    Parameters
    ----------
    PPC_traces_event : dict
        Dictionary of 'y' and 'y_rep' for each trace.
    verbose : bool, optional
        If True, print details of each statistic. Default is False.

    Returns
    -------
    T_dict_event : dict
        Dictionary containing p-values and observed/replicated statistics for each metric.
    """
    
    T_dict_event = {} # store test statistics
    for func_name, func in zip(
        ['mean', 'standard deviation',zp_title, r'fraction $|\cos\theta\,|>0.5$'],
        [np.average, np.std, zero_positive_spin, fraction_tails]
    ): 
        # Calculate different test statistics from 'y'  and 'y_rep' in each entry of PPC_traces_event
        Ntraces = len(PPC_traces_event)
        T_obs_arr = np.zeros(Ntraces)
        T_rep_arr = np.zeros(Ntraces)
        for i,p in enumerate(PPC_traces_event.values()):
            T_obs_arr[i] = func(p['y']) 
            T_rep_arr[i] = func(p['y_rep'])
        
        # Calculate p value by comparing them
        percentile = sum(T_obs_arr < T_rep_arr) / Ntraces
        pvalue = 1 - 2 * np.abs(percentile-0.5)
    
        # Print info
        if verbose:
            print(func_name, percentile, pvalue)
    
        # Save to ongoing dict
        T_dict_event[func_name] = dict(pvalue=np.round(pvalue,3), T=T_obs_arr, Trep=T_rep_arr)
    
    return T_dict_event


def darken_color(color, factor=0.7):
    """
    Darken a color by multiplying its lightness.

    Parameters
    ----------
    color : str or tuple
        Any matplotlib-compatible color.
    factor : float, optional
        Amount to darken (0–1).
        1.0 = no change
        0.0 = black
        Default is 0.7.

    Returns
    -------
    tuple
        Darkened RGB color tuple.
    """
    # Convert color to RGB
    rgb = mcolors.to_rgb(color)

    # Multiply each channel
    darker_rgb = tuple(max(0, c * factor) for c in rgb)

    return darker_rgb


def traces_and_underpred_plot(axes, DD, PD, param, color='C0', ntraces=1000, lbl=None, plot_avg=True): 
    """
    Plot PPC traces and fraction underpredicted for a given parameter.
    (Used for the GWTC-4.0 figure)

    Parameters
    ----------
    axes : list of matplotlib.axes.Axes
        Two axes: first for PPC traces, second for fraction underpredicted.
    DD : dict
        Dictionary of predicted and observed data arrays.
    PD : dict
        Dictionary of fraction-underpredicted results.
    param : str
        Parameter to plot ('a' or 'cos_tilt').
    color : str, optional
        Base color for plots. Default is 'C0'.
    ntraces : int, optional
        Number of traces to plot. Default is 1000.
    lbl : dict, optional
        Custom labels for the axes. Default is None.
    plot_avg : bool, optional
        Whether to plot the average curve. Default is True.
    """
    
    bounds = {
        'a':[0,1],
        'cos_tilt':[-1,1],
    }
    labels = {
        'a':r'$\chi$',
        'cos_tilt':r'$\cos\theta$',
    }
    color_dark = darken_color(color, factor=0.6)
    
    if lbl is None:
        label = labels[param]
        label_pred = 'predicted '+label
        label_obs = 'observed '+label
    else: 
        label = lbl['label']
        label_pred = lbl['label pred']
        label_obs = lbl['label obs']
        
    diag_kws = dict(color='k', ls='--', lw=0.5)
    
    # ======================
    # ===== PPC TRACES =====
    # ======================
    ax = axes[0]
    for obs_trace, pred_trace in zip(
        DD['observed'][param][:ntraces], 
        DD['predicted'][param][:ntraces]
    ):
        ax.plot(pred_trace, obs_trace, lw=0.3, alpha=0.03, color=color)
    
    b = bounds[param]
    ax.plot(b,b,**diag_kws)
    ax.set_xlim(*b)
    ax.set_ylim(*b)
    ax.grid()

    ax.set_xlabel(label_pred, fontsize=13)
    ax.set_ylabel(label_obs, fontsize=13)
    
    if plot_avg: 
        obs_avgs, pred_avgs = average_curve(DD, param)
        ax.plot(pred_avgs, obs_avgs, color=color_dark, lw=0.7)
    
    # ======================
    # = FRACTION UNDERPRED =
    # ======================
    ax = axes[1]
    
    x = PD[param]['xs']
    y = PD[param]['fraction']
    N = PD[param]['N']

    err = 1 / np.sqrt(N)

    ax.scatter(x, y, color=color_dark, s=2, marker='*')
    ax.fill_between(x, y+err, y-err, alpha=.3, color=color)

    b = bounds[param]
    ax.set_xlim(*b)
    ax.set_ylim(0,1)
    ax.axhline(0.5, **diag_kws)
    ax.grid()

    ax.set_xlabel(label, fontsize=13)
    ax.set_ylabel('Fraction underpredicted', fontsize=11)