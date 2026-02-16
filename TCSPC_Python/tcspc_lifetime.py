# -*- coding: utf-8 -*-
"""
Troy C. Messina
TCSPC Lifetime Fitting
February 12, 2026
"""

import numpy as np
from scipy.signal import fftconvolve
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import pandas as pd



# =============================================================================
# 1: DATA I/O AND FILE HANDLING
# =============================================================================

class DataLoader:
    """Load various TCSPC and spectroscopy file formats"""
    
    @staticmethod
    def load_becker_hickl_ascii(filepath: str):
        """Load Becker & Hickl .asc files with setup information"""
        cols = ['time', 'counts']
        data_array = pd.read_csv(filepath, sep='\\s+', dtype=np.float64, skiprows=10, skipfooter=1, names=cols, engine='python')
        
        return data_array
    
    
    @staticmethod
    def load_becker_hickl_sdt(filepath: str):
        """Load Becker & Hickl .sdt files with setup information. Requires https://github.com/cgohlke/sdtfile"""
        import sdtfile as sdt
        data = sdt.SdtFile(filepath)
        
        struct = {'time': data.times[0]*1e9, 
                'counts': data.data[0][0]
               }
        data_array = pd.DataFrame(struct)
        
        return data_array
    
# =============================================================================
# 2: DATA VISUALIZATION
# =============================================================================
class PlotRoutines:
    """Make graphs of data"""
    
    def quick_plot(irf, data):
        plt.semilogy(irf['time'], irf['counts'])
        plt.semilogy(data['time'], data['counts'])
        plt.xlabel('time (ns)')
        plt.ylabel('counts')
        plt.show()
        return 0
    
    def full_figure(df, params):
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.fillna(0)
        fit = DataFitting() #create an instance of the fitting class
        testy = fit.fit_multiexp(df['time'], params)
        # Create a figure and a single axes (subplot)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [1, 4]})

        ax1.plot(df['time'], (df['data']-testy)/np.sqrt(df['data']), 'r-')
        ax1.set_ylabel('Residuals')

        ax2.semilogy(df['time'], df['data'], label='Data', alpha=0.6)
        ax2.semilogy(df['time'], testy, 'k-', label='Fit')
        ax2.set_ylabel('counts')
        ax2.legend()

        plt.xlabel('time (ns)')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
        plt.show()


# =============================================================================
# 3: DATA PREPARATION AND FITTING
# =============================================================================
class DataFitting:
    """Prepare data and IRF for fitting multiexponential + convolution fitting"""
    
    def trunc(data, irf, thresh, left, right, pnt):
        """Cut the ends from the data where digitization effects show."""
        if pnt == 1: #start/stop are point numbers
            start = left
            stop = right
        else:
            start = (data['time'] - left).abs().idxmin()
            stop = (data['time'] - right).abs().idxmin()
        irf['clip'] = irf['counts']
        irf.loc[irf['counts'] < thresh, 'clip'] = 0 #get rid of noise in the IRF so it doesn't convolve
        irf['norm'] = irf['clip']/np.sum(irf['clip']) #normalize the IRF so it doesn't increase amplitude on convolution
        struct = {'time': data['time'].iloc[start:stop], 
                'irf': irf['norm'].iloc[start:stop], 
                'data': data['counts'].iloc[start:stop]
               }
        data4fit = pd.DataFrame(struct)
        global DELTA_T 
        DELTA_T = data4fit['time'].iloc[1] - data4fit['time'].iloc[0]
        global GLOBAL_IRF 
        GLOBAL_IRF = data4fit['irf']
        return data4fit
    
    def quick_fit(t, *args):
        """Multiexponential and convolution fitting algorithm."""
        #The fit parameters are
        # pw[0] = shift
        # pw [1] = baseline
        # pw[2] = amplitude 1
        # pw[3] = lifetime 1
        # pw[2n] = amplitude n
        # pw[2n+1] = lifetime n
        pw = np.array(args[0], dtype=np.float64)
        shift = int(pw[0])
        baseline = pw[1]
        N = len(pw)-2
        pw_cut = pw[-N:]
        numtau = int(len(pw_cut))
        #print(f"The list has {numtau} lifetimes.")
        y = np.zeros_like(t)
        for index in range(0, numtau, 2):
            y += pw_cut[index] * np.exp(-t/pw_cut[index+1])
        roll_irf = np.roll(GLOBAL_IRF, shift)#shift(GLOBAL_IRF, shift)
        #####
        #t_shifted = np.arange(len(GLOBAL_IRF)) * DELTA_T + shift
        #roll_irf = np.interp(t_shifted, np.arange(len(GLOBAL_IRF)) * DELTA_T, GLOBAL_IRF)
        #####
        dup_model = np.concatenate((y, y))
        #out = fftconvolve(roll_irf, dup_model) + baseline
        out = np.convolve(roll_irf, dup_model, mode='full') + baseline
        L = len(t)
        return out[L:2*L]
    
    def fit_multiexp(self, t, *args):
        """Multiexponential and convolution fitting algorithm."""
        #The fit parameters are
        # pw[0] = shift
        # pw [1] = baseline
        # pw[2] = amplitude 1
        # pw[3] = lifetime 1
        # pw[2n] = amplitude n
        # pw[2n+1] = lifetime n
        pw = np.array(args[0], dtype=np.float64)
        shift = int(pw[0])
        baseline = pw[1]
        N = len(pw)-2
        pw_cut = pw[-N:]
        numtau = int(len(pw_cut))
        #print(f"The list has {numtau} lifetimes.")
        y = np.zeros_like(t)
        for index in range(0, numtau, 2):
            y += pw_cut[index] * np.exp(-t/pw_cut[index+1])
        roll_irf = np.roll(GLOBAL_IRF, shift)#shift(GLOBAL_IRF, shift)
        #####
        #t_shifted = np.arange(len(GLOBAL_IRF)) * DELTA_T + shift
        #roll_irf = np.interp(t_shifted, np.arange(len(GLOBAL_IRF)) * DELTA_T, GLOBAL_IRF)
        #####
        dup_model = np.concatenate((y, y))
        #out = fftconvolve(roll_irf, dup_model) + baseline
        out = np.convolve(roll_irf, dup_model, mode='full') + baseline
        L = len(t)
        return out[L:2*L]

    def residual_function(self, params, x, y):
        """Computes the residuals (difference between observed and predicted y values)."""
        predicted_y = self.fit_multiexp(x, params)
        weights = np.sqrt(y)
        weights = weights.clip(lower=1)
        return (y - predicted_y)/weights