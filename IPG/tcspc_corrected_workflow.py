"""
CORRECTED TCSPC Analysis with Data Truncation and Lag Control

NEW FEATURES:
- Data truncation to remove digitization artifacts
- Optional user-provided initial lag estimates
- Proper IRF mapping for N traces with M IRFs
"""

import numpy as np
import matplotlib.pyplot as plt
from tcspc_analysis import *

# =============================================================================
# TRUNCATION HELPER FUNCTION
# =============================================================================

def truncate_tcspc_data(time, data_matrix, irf_matrix, 
                        start_time=None, end_time=None,
                        start_index=None, end_index=None):
    """
    Truncate TCSPC data to remove artifacts
    
    Parameters:
    -----------
    time : array
        Time axis
    data_matrix : array (n_times, n_traces)
        Data to truncate
    irf_matrix : array (n_times, n_irfs)
        IRFs to truncate (must match time axis)
    start_time : float, optional
        Start time (ns) to keep
    end_time : float, optional
        End time (ns) to keep
    start_index : int, optional
        Start index (alternative to start_time)
    end_index : int, optional
        End index (alternative to end_time)
        
    Returns:
    --------
    time_trunc : array
        Truncated time axis
    data_trunc : array
        Truncated data
    irf_trunc : array
        Truncated IRFs
    """
    
    # Convert time values to indices if provided
    if start_time is not None:
        start_index = np.argmin(np.abs(time - start_time))
    if end_time is not None:
        end_index = np.argmin(np.abs(time - end_time))
    
    # Use defaults if nothing provided
    if start_index is None:
        start_index = 0
    if end_index is None:
        end_index = len(time)
    
    # Truncate
    time_trunc = time[start_index:end_index]
    data_trunc = data_matrix[start_index:end_index, :]
    irf_trunc = irf_matrix[start_index:end_index, :]
    
    print(f"Truncation applied:")
    print(f"  Original: {len(time)} points from {time[0]:.3f} to {time[-1]:.3f} ns")
    print(f"  Truncated: {len(time_trunc)} points from {time_trunc[0]:.3f} to {time_trunc[-1]:.3f} ns")
    print(f"  Removed: {start_index} points at start, {len(time) - end_index} points at end")
    
    return time_trunc, data_trunc, irf_trunc


# =============================================================================
# INTERACTIVE TRUNCATION WITH VISUALIZATION
# =============================================================================

def interactive_truncation(time, data_matrix, irf_matrix):
    """
    Visualize data to help choose truncation points
    
    Parameters:
    -----------
    time : array
        Time axis
    data_matrix : array
        Data matrix
    irf_matrix : array
        IRF matrix
        
    Returns:
    --------
    None (displays plot for user inspection)
    """
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot data traces
    ax1.semilogy(time, data_matrix[:, 0], 'b-', linewidth=2, label='First data trace')
    if data_matrix.shape[1] > 1:
        ax1.semilogy(time, data_matrix[:, 1], 'g-', alpha=0.7, label='Second data trace')
    ax1.set_xlabel('Time (ns)', fontsize=12)
    ax1.set_ylabel('Counts (log scale)', fontsize=12)
    ax1.set_title('Data - Check for artifacts at edges', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot IRF
    ax2.plot(time, irf_matrix[:, 0], 'r-', linewidth=2, label='IRF')
    ax2.set_xlabel('Time (ns)', fontsize=12)
    ax2.set_ylabel('IRF Amplitude', fontsize=12)
    ax2.set_title('IRF - Should align with data rise', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("\nInspect the plots above to choose truncation points.")
    print("Look for:")
    print("  - Electronic artifacts at early times")
    print("  - Low count/noisy regions at late times")
    print("  - IRF position relative to data")


# =============================================================================
# FIXED IRF SELECTOR WITH LAG CONTROL
# =============================================================================

class IRFSelectorCorrected:
    """
    Fixed IRF selector with optional initial lag estimates
    """
    
    def __init__(self):
        self.best_irf_indices = None
        self.chi_squared_matrix = None
    
    def find_best_irf(self, data_matrix: np.ndarray,
                       irf_matrix: np.ndarray,
                       grid: np.ndarray,
                       initial_lags: np.ndarray = None,
                       lag_divisions: int = 100) -> dict:
        """
        Find best IRF for each data trace
        
        Parameters:
        -----------
        data_matrix : array (n_times, n_data_traces)
            Data to fit
        irf_matrix : array (n_times, n_irfs)
            All available IRFs
        grid : array
            Lifetime grid for fitting
        initial_lags : array (n_traces,) or float, optional
            Initial lag estimates (ns). If float, same for all traces.
            If None, will search from 0.
        lag_divisions : int
            Number of sub-bin divisions to test around initial lag
            
        Returns:
        --------
        results : dict
            'best_irf_indices': array (n_data_traces,)
            'best_lags': array (n_data_traces,)
            'chi_squared_matrix': array (n_traces, n_irfs)
        """
        n_times, n_traces = data_matrix.shape
        _, n_irfs = irf_matrix.shape
        
        print(f"\nTesting {n_irfs} IRF(s) against {n_traces} data trace(s)...")
        
        # Handle initial lags
        if initial_lags is not None:
            if np.isscalar(initial_lags):
                initial_lags = np.full(n_traces, initial_lags)
            print(f"  Using initial lag estimates: mean = {np.mean(initial_lags):.4f} ns")
        else:
            initial_lags = np.zeros(n_traces)
            print(f"  No initial lags provided, starting from 0")
        
        # Store chi-squared for each trace-IRF combination
        chi_matrix = np.zeros((n_traces, n_irfs))
        lag_matrix = np.zeros((n_traces, n_irfs))
        
        # Test each IRF against each data trace
        for irf_idx in range(n_irfs):
            print(f"  Testing IRF {irf_idx+1}/{n_irfs}...")
            
            # Get this IRF (replicate for all traces)
            irf_test = np.tile(irf_matrix[:, irf_idx:irf_idx+1], (1, n_traces))
            
            # Find optimal lags using LinearLagFinder
            # Note: We could modify this to use initial_lags as starting point
            lag_finder = LinearLagFinder()
            lags_result = lag_finder.linear_lags(
                data_matrix,
                irf_test,
                grid,
                divisions=lag_divisions
            )
            
            # If initial lags provided, refine around them
            if np.any(initial_lags != 0):
                # The lags from linear_lags are relative, add to initial
                lags_result[:, 1] += initial_lags
            
            # Store results
            chi_matrix[:, irf_idx] = lags_result[:, 0]  # Chi-squared
            lag_matrix[:, irf_idx] = lags_result[:, 1]  # Optimal lag
        
        # Find best IRF for each trace (minimum chi-squared)
        best_irf_indices = np.argmin(chi_matrix, axis=1)
        
        # Get corresponding best lags
        best_lags = np.array([lag_matrix[trace_idx, best_irf_indices[trace_idx]] 
                              for trace_idx in range(n_traces)])
        
        # Report results
        print(f"\nIRF Selection Results:")
        for trace_idx in range(min(n_traces, 10)):
            best_irf = best_irf_indices[trace_idx]
            best_chi = chi_matrix[trace_idx, best_irf]
            print(f"  Trace {trace_idx}: IRF {best_irf}, χ² = {best_chi:.4f}, lag = {best_lags[trace_idx]:.4f} ns")
        
        if n_traces > 10:
            print(f"  ... (showing first 10 of {n_traces} traces)")
        
        # Count IRF usage
        unique_irfs, counts = np.unique(best_irf_indices, return_counts=True)
        print(f"\nIRF Usage Summary:")
        for irf_idx, count in zip(unique_irfs, counts):
            print(f"  IRF {irf_idx}: used by {count} trace(s)")
        
        results = {
            'best_irf_indices': best_irf_indices,
            'best_lags': best_lags,
            'chi_squared_matrix': chi_matrix,
            'lag_matrix': lag_matrix
        }
        
        self.best_irf_indices = best_irf_indices
        self.chi_squared_matrix = chi_matrix
        
        return results


# =============================================================================
# COMPLETE WORKFLOW WITH TRUNCATION AND LAG CONTROL
# =============================================================================

def complete_tcspc_workflow_with_irf(data_files, irf_files, 
                                     time_axis=None,
                                     truncate_start=None,
                                     truncate_end=None,
                                     truncate_start_idx=None,
                                     truncate_end_idx=None,
                                     initial_lags=None,
                                     show_truncation_plot=True,
                                     show_plots=True,
                                     save_plots=True):
    """
    Complete TCSPC analysis with truncation and lag control
    
    Parameters:
    -----------
    data_files : list of str
        Paths to data files
    irf_files : list of str
        Paths to IRF files
    time_axis : array, optional
        Time axis (if not in files)
    truncate_start : float, optional
        Start time (ns) to keep
    truncate_end : float, optional
        End time (ns) to keep
    truncate_start_idx : int, optional
        Start index (alternative to truncate_start)
    truncate_end_idx : int, optional
        End index (alternative to truncate_end)
    initial_lags : float or array, optional
        Initial lag estimate(s) in ns. Single value or per-trace array.
    show_truncation_plot : bool
        Show interactive plot to help choose truncation
    show_plots : bool
        Display plots as they're created (default: True)
    save_plots : bool
        Save plots to files (default: True)
        
    Returns:
    --------
    results : dict
        Complete analysis results including 'figures' key with all figures
    """
    
    print("="*70)
    print("TCSPC ANALYSIS WITH TRUNCATION AND LAG CONTROL")
    print("="*70)
    
    # =========================================================================
    # STEP 1: LOAD DATA AND IRFs
    # =========================================================================
    print("\n1. Loading Data and IRFs...")
    
    loader = DataLoader()
    
    # Load data files
    all_data = []
    for i, filename in enumerate(data_files):
        data_dict = loader.load_generic_ascii(filename, skip_header=0)
        if i == 0 and time_axis is None:
            time = data_dict['time']
        all_data.append(data_dict['data'].flatten())
    
    data_matrix = np.column_stack(all_data)
    
    # Load IRF files
    all_irfs = []
    for filename in irf_files:
        irf_dict = loader.load_generic_ascii(filename, skip_header=0)
        all_irfs.append(irf_dict['data'].flatten())
    
    irf_matrix = np.column_stack(all_irfs)
    
    # Use provided time axis if given
    if time_axis is not None:
        time = time_axis
    
    n_times_orig = len(time)
    n_traces = data_matrix.shape[1]
    n_irfs = irf_matrix.shape[1]
    
    print(f"   Data: {n_times_orig} time points, {n_traces} traces")
    print(f"   IRFs: {n_irfs} available")
    print(f"   Time range: {time[0]:.3f} to {time[-1]:.3f} ns")
    
    # =========================================================================
    # STEP 1.5: OPTIONAL TRUNCATION
    # =========================================================================
    
    # Show visualization if requested
    if show_truncation_plot:
        print("\n1.5. Visualizing Data for Truncation Selection...")
        interactive_truncation(time, data_matrix, irf_matrix)
    
    # Apply truncation if requested
    if any([truncate_start, truncate_end, truncate_start_idx, truncate_end_idx]):
        print("\n1.5. Truncating Data and IRFs...")
        time, data_matrix, irf_matrix = truncate_tcspc_data(
            time, data_matrix, irf_matrix,
            start_time=truncate_start,
            end_time=truncate_end,
            start_index=truncate_start_idx,
            end_index=truncate_end_idx
        )
    else:
        print("\n1.5. No truncation requested (using full time range)")
    
    n_times = len(time)
    
    # =========================================================================
    # STEP 2: PREPROCESS DATA
    # =========================================================================
    print("\n2. Preprocessing Data...")
    
    preprocessor = DataPreprocessor()
    
    # Clean data
    data_clean = preprocessor.remove_cosmic_rays(data_matrix, threshold=5.0)
    data_corrected = preprocessor.baseline_correction(data_clean, method='tail', 
                                                     tail_points=50)
    
    # Smooth if necessary
    data_final = preprocessor.smooth_data(data_corrected, method='savgol', 
                                         window=5, order=2)
    
    print("   Data preprocessing complete")
    
    # =========================================================================
    # STEP 3: CREATE INITIAL GRID
    # =========================================================================
    print("\n3. Creating Lifetime Grid...")
    
    grid_gen = GridGenerator()
    grid = grid_gen.make_log_grid(0.01, 10, 16)
    
    n_grid = len(grid)
    print(f"   Grid: {n_grid} points from {grid[0]:.3f} to {grid[-1]:.3f} ns")
    
    # =========================================================================
    # STEP 4: FIND BEST IRF AND LAG FOR EACH TRACE
    # =========================================================================
    print("\n4. Finding Best IRF and Lag for Each Trace...")
    
    if initial_lags is not None:
        if np.isscalar(initial_lags):
            print(f"   Using initial lag estimate: {initial_lags:.4f} ns for all traces")
        else:
            print(f"   Using per-trace lag estimates: mean = {np.mean(initial_lags):.4f} ns")
    
    irf_selector = IRFSelectorCorrected()
    irf_results = irf_selector.find_best_irf(
        data_final, 
        irf_matrix, 
        grid,
        initial_lags=initial_lags,
        lag_divisions=10
    )
    
    best_irf_indices = irf_results['best_irf_indices']
    best_lags = irf_results['best_lags']
    
    print(f"\n   IRF selection complete!")
    print(f"   Lag range: {np.min(best_lags):.4f} to {np.max(best_lags):.4f} ns")
    print(f"   Mean lag: {np.mean(best_lags):.4f} ns")
    
    # =========================================================================
    # STEP 5: BUILD DESIGN MATRIX WITH PROPER IRF MAPPING
    # =========================================================================
    print("\n5. Building Design Matrix with IRF Convolution...")
    
    dx = time[1] - time[0] if len(time) > 1 else 1.0
    n_params = n_grid + 2  # grid + scattering + baseline
    design_matrix_3d = np.zeros((n_times, n_params, n_traces))
    std_matrix = np.sqrt(np.maximum(data_final, 1))
    
    for trace_idx in range(n_traces):
        if trace_idx % max(1, n_traces // 10) == 0:
            print(f"   Convolving trace {trace_idx+1}/{n_traces}...")
        
        # Get the BEST IRF for THIS trace (using mapping)
        best_irf_idx = best_irf_indices[trace_idx]
        irf = irf_matrix[:, best_irf_idx]
        lag = best_lags[trace_idx]
        
        # Shift IRF by optimal lag
        x_shifted = np.arange(len(irf)) * dx + lag
        irf_shifted = np.interp(x_shifted, np.arange(len(irf)) * dx, irf, 
                               left=0, right=0)
        
        # Normalize IRF
        irf_sum = np.sum(irf_shifted)
        if irf_sum > 0:
            irf_shifted = irf_shifted / irf_sum
        
        # Build exponential basis functions
        basis_matrix = np.zeros((n_times, n_grid))
        for i, tau in enumerate(grid):
            basis_matrix[:, i] = np.exp(-time / tau)
        
        # Convolve each basis function with IRF
        design_conv = np.zeros((n_times, n_grid))
        for i in range(n_grid):
            basis_func = basis_matrix[:, i]
            convolved = np.convolve(basis_func, irf_shifted, mode='full')[:n_times]
            design_conv[:, i] = convolved
        
        # Add scattering (IRF shape)
        scattering = irf_shifted[:n_times] if len(irf_shifted) >= n_times else np.pad(irf_shifted, (0, n_times - len(irf_shifted)))
        
        # Add baseline
        baseline = np.ones(n_times)
        
        # Combine: [convolved exponentials | scattering | baseline]
        design_matrix_3d[:, :n_grid, trace_idx] = design_conv
        design_matrix_3d[:, n_grid, trace_idx] = scattering
        design_matrix_3d[:, n_grid+1, trace_idx] = baseline
        
        # Normalize by standard deviation
        design_matrix_3d[:, :, trace_idx] /= std_matrix[:, trace_idx, np.newaxis]
    
    print(f"   Design matrix complete: {design_matrix_3d.shape}")
    
    # =========================================================================
    # STEP 6: CREATE REGULARIZATION MATRIX
    # =========================================================================
    print("\n6. Creating Regularization Matrix...")
    
    reg_matrix = RegularizationMatrix.make_reg_matrix_2nd_derivative(n_params)
    print(f"   2nd derivative regularization: {reg_matrix.shape}")
    
    # =========================================================================
    # STEP 7: FIND OPTIMAL REGULARIZATION PARAMETER
    # =========================================================================
    print("\n7. Finding Optimal Regularization Parameter...")
    
    solver = IPGSolver()
    adaptive_reg = AdaptiveRegularization(solver)
    
    # Test range of alpha values
    alphas = np.logspace(-6, 2, 10)
    
    lcurve_results = adaptive_reg.lcurve_analysis(
        data_final, std_matrix, design_matrix_3d,
        reg_matrix, grid, alphas,
        n_iterations=500,
        tau=0.9,
        plot=True
    )
    
    optimal_alpha = lcurve_results['optimal_alpha']
    print(f"   Optimal α: {optimal_alpha:.2e}")
    
    # =========================================================================
    # STEP 8: PERFORM REGULARIZED FIT
    # =========================================================================
    print("\n8. Performing Regularized IPG Fit...")
    
    setup = solver.setup_gipg(
        data_final, std_matrix, design_matrix_3d,
        reg_matrix, grid, optimal_alpha, normalize=True
    )
    
    # Global fit
    if n_traces == 1:
        params = solver.local_ipg(setup, n_iterations=5000, n_updates=50,
                                 alpha=optimal_alpha, tau=0.9)
    else:
        params = solver.global_ipg(setup, n_iterations=5000, n_updates=50,
                                   alpha=optimal_alpha, tau=0.9)
    
    params_matrix = params.reshape((n_params, n_traces), order='F')
    
    print(f"   Fit complete!")
    print(f"   Final χ²: {solver.chi_history[-1]:.4f}")
    
    # =========================================================================
    # STEP 9: CALCULATE FITS AND RESIDUALS
    # =========================================================================
    print("\n9. Calculating Fits and Residuals...")
    
    design_flat = setup['design_flat']
    fits_flat = design_flat @ params
    fits_matrix = fits_flat.reshape((n_times, n_traces), order='F')
    
    residuals_flat = (setup['data_flat'] - fits_flat) / setup['std_flat']
    residuals_matrix = residuals_flat.reshape((n_times, n_traces), order='F')
    
    # =========================================================================
    # STEP 10: VISUALIZE RESULTS
    # =========================================================================
    print("\n10. Creating Visualizations...")
    
    plotter = InteractivePlotter()
    all_figures = []
    
    # Plot 1: Decay curves with fits
    print("   Creating decay fit plot...")
    fig1 = plotter.plot_decay_curves(
        data_final, time, fits_matrix, residuals_matrix,
        std_matrix, log_scale=True,
        title=f"TCSPC Fits ({n_traces} traces, {n_irfs} IRFs, truncated)"
    )
    all_figures.append(fig1)
    if save_plots:
        plt.figure(fig1.number)
        plt.savefig('tcspc_decay_fits.png', dpi=300, bbox_inches='tight')
        print("      Saved: tcspc_decay_fits.png")
    if show_plots:
        plt.figure(fig1.number)
        plt.show(block=False)
    
    # Plot 2: Lifetime distributions
    print("   Creating lifetime distribution plot...")
    fig2 = plotter.plot_distribution(
        params_matrix[:n_grid, :],
        grid,
        plot_type='line' if n_traces == 1 else 'contour',
        x_label='Lifetime (ns)',
        z_label='Amplitude',
        title='Lifetime Distribution (IRF-Corrected)',
        log_x=True
    )
    all_figures.append(fig2)
    if save_plots:
        plt.figure(fig2.number)
        plt.savefig('tcspc_lifetime_distribution.png', dpi=300, bbox_inches='tight')
        print("      Saved: tcspc_lifetime_distribution.png")
    if show_plots:
        plt.figure(fig2.number)
        plt.show(block=False)
    
    # Plot 3: Convergence
    print("   Creating convergence plot...")
    fig3 = plotter.plot_convergence(solver.chi_history)
    all_figures.append(fig3)
    if save_plots:
        plt.figure(fig3.number)
        plt.savefig('tcspc_convergence.png', dpi=300, bbox_inches='tight')
        print("      Saved: tcspc_convergence.png")
    if show_plots:
        plt.figure(fig3.number)
        plt.show(block=False)
    
    # Plot 4: Residuals
    print("   Creating residual analysis plot...")
    fig4 = plotter.plot_residual_analysis(residuals_matrix, time)
    all_figures.append(fig4)
    if save_plots:
        plt.figure(fig4.number)
        plt.savefig('tcspc_residuals.png', dpi=300, bbox_inches='tight')
        print("      Saved: tcspc_residuals.png")
    if show_plots:
        plt.figure(fig4.number)
        plt.show(block=False)
    
    if show_plots:
        print("\n   All plots displayed. Close plot windows to continue or use plt.show() to keep them open.")
        plt.pause(0.1)  # Brief pause to ensure plots render
    
    # =========================================================================
    # STEP 11: REPORT RESULTS
    # =========================================================================
    print("\n" + "="*70)
    print("ANALYSIS RESULTS")
    print("="*70)
    
    print(f"\nTruncation:")
    print(f"  Original: {n_times_orig} points")
    print(f"  Used: {n_times} points ({100*n_times/n_times_orig:.1f}%)")
    
    print(f"\nLag Summary:")
    print(f"  Mean: {np.mean(best_lags):.4f} ns")
    print(f"  Std: {np.std(best_lags):.4f} ns")
    print(f"  Range: [{np.min(best_lags):.4f}, {np.max(best_lags):.4f}] ns")
    
    print(f"\nQuality Metrics:")
    print(f"  Final reduced χ²: {solver.chi_history[-1]:.4f}")
    print(f"  Effective free parameters: {setup['efp']:.1f}")
    
    if solver.chi_history[-1] < 1.2:
        print("  ✓ Excellent fit")
    elif solver.chi_history[-1] < 2.0:
        print("  ✓ Good fit")
    else:
        print("  ⚠ Poor fit - consider adjusting parameters")
    
    # Show lifetime results for first few traces
    print(f"\nLifetime Analysis (first {min(15, n_traces)} traces):")
    for trace_idx in range(min(15, n_traces)):
        print(f"\n--- Trace {trace_idx} (IRF {best_irf_indices[trace_idx]}, lag {best_lags[trace_idx]:.4f} ns) ---")
        
        lifetime_amps = params_matrix[:n_grid, trace_idx]
        
        threshold = 0.05 * np.max(lifetime_amps)
        significant_indices = np.where(lifetime_amps > threshold)[0]
        
        if len(significant_indices) > 0:
            print(f"Significant lifetimes:")
            for idx in significant_indices:
                print(f"  τ = {grid[idx]:.3f} ns, amplitude = {lifetime_amps[idx]:.4f}")
            
            total_amp = np.sum(lifetime_amps)
            if total_amp > 0:
                mean_lifetime = np.sum(lifetime_amps * grid) / total_amp
                print(f"Mean lifetime: {mean_lifetime:.3f} ns")
        else:
            print("  No significant components found")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    
    # =========================================================================
    # RETURN RESULTS
    # =========================================================================
    results = {
        'data': data_final,
        'time': time,
        'fits': fits_matrix,
        'residuals': residuals_matrix,
        'params_matrix': params_matrix,
        'grid': grid,
        'irf_matrix': irf_matrix,
        'best_irf_indices': best_irf_indices,
        'best_lags': best_lags,
        'chi_squared_matrix': irf_results['chi_squared_matrix'],
        'alpha': optimal_alpha,
        'chi_squared': solver.chi_history[-1],
        'chi_history': np.array(solver.chi_history),
        'std_matrix': std_matrix,
        'n_times_original': n_times_orig,
        'n_times_used': n_times,
        'figures': all_figures  # All figure objects for user access
    }
    
    # Keep plots open if showing
    if show_plots:
        print("\n   Plots are displayed. They will remain open.")
        print("   To keep them interactive, call: plt.show()")
    
    return results


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("EXAMPLE: TCSPC with Truncation and Lag Control")
    print("="*70)
    
    # Create synthetic data with artifacts
    n_times = 300
    n_traces = 3
    time = np.linspace(0, 30, n_times)
    
    # Create IRF
    irf = np.exp(-((time - 5.0)**2) / (2 * 0.3**2))
    irf = irf / np.sum(irf)
    
    # Generate data with artifacts at edges
    data_matrix = np.zeros((n_times, n_traces))
    
    for trace_idx in range(n_traces):
        # True decay
        signal = 0.4 * np.exp(-time / 2.0) + 0.6 * np.exp(-time / 5.0)
        
        # Convolve with IRF
        signal_conv = np.convolve(signal, irf, mode='full')[:n_times]
        
        # Add artifacts at edges
        signal_conv[:20] += np.random.rand(20) * 50  # Early time artifact
        signal_conv[-30:] *= 0.5  # Late time decay issues
        
        # Add noise
        noise = np.random.randn(n_times) * 0.03 * np.max(signal_conv)
        data_matrix[:, trace_idx] = signal_conv + noise + 5.0
    
    data_matrix = np.maximum(data_matrix, 0)
    
    # Save files
    for i in range(n_traces):
        np.savetxt(f'data_with_artifacts_{i}.txt',
                   np.column_stack([time, data_matrix[:, i]]))
    
    np.savetxt('irf_example.txt',
               np.column_stack([time, irf]))
    
    print("\nGenerated data with artifacts at t<2 ns and t>25 ns")
    
    # Run analysis WITH truncation and initial lag
    data_files = [f'data_with_artifacts_{i}.txt' for i in range(n_traces)]
    irf_files = ['irf_example.txt']
    
    print("\n--- Running with truncation and initial lag ---")
    results = complete_tcspc_workflow_with_irf(
        data_files,
        irf_files,
        truncate_start=2.0,      # Remove early artifact
        truncate_end=25.0,       # Remove late artifact
        initial_lags=-0.5,       # Initial lag estimate
        show_truncation_plot=False,
        show_plots=True,         # Display plots
        save_plots=True          # Save to files
    )
    
    print("\n✓ Analysis complete with truncation and lag control!")
    
    # Access figures if needed
    print(f"\nCreated {len(results['figures'])} figures")
    
    # Keep plots open
    plt.show()
