"""
Hidden Markov Model Time-Series Analysis

This module performs change point detection, segmentation, and HMM-based
reconstruction of time-series photon counting data.

Author: troy c. messina (refactored)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List, Optional
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import norm
import tkinter as tk
from tkinter import filedialog


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Constants
DEFAULT_BINNING = 0.1
MIN_POINTS = 4
CONFIDENCE_LEVEL = 0.99
LARGE_VALUE = 1e300
SMALL_VALUE = 1e-6
DEFAULT_TRANSITION_RATE_FACTOR = 0.05
MIN_STD_FACTOR = 0.25


@dataclass
class AnalysisConfig:
    """Configuration parameters for HMM analysis."""
    binning: float = DEFAULT_BINNING
    min_points: int = MIN_POINTS
    confidence_level: float = CONFIDENCE_LEVEL
    transition_rate_factor: float = DEFAULT_TRANSITION_RATE_FACTOR
    min_std_factor: float = MIN_STD_FACTOR
    output_dir: Path = Path('./figures')
    
    def __post_init__(self):
        """Ensure output directory exists."""
        self.output_dir.mkdir(parents=True, exist_ok=True)


class ChangePointDetector:
    """Detects change points in time-series data using recursive segmentation."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.change_points: List[int] = []
    
    def detect(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find segments with statistically different count rates.
        
        Args:
            data: Time-series data array
            
        Returns:
            Tuple of (Student-T values, change point indices)
        """
        self.change_points = []
        student_t = np.zeros(len(data))
        
        self._recursive_segment(data, 0, len(data), student_t)
        
        return student_t, np.array(self.change_points)
    
    def _recursive_segment(
        self,
        data: np.ndarray,
        mark1: int,
        mark2: int,
        student_t: np.ndarray
    ) -> None:
        """
        Recursively segment data to find change points.
        
        Args:
            data: Time-series data
            mark1: Start index of segment
            mark2: End index of segment
            student_t: Array to store Student-T values
        """
        data_std = np.std(data)
        
        # Calculate Student-T values for all potential split points
        for j in range(mark1 + 1, mark2 - 1):
            stats_left = self._calculate_segment_stats(data, mark1, j, data_std)
            stats_right = self._calculate_segment_stats(data, j + 1, mark2, data_std)
            
            t_value = self._calculate_t_statistic(stats_left, stats_right)
            
            if np.isfinite(t_value):
                student_t[j] = t_value
        
        # Find maximum T-value in current segment
        max_t = np.nanmax(student_t[mark1:mark2])
        split_point = np.nanargmax(student_t[mark1:mark2]) + mark1
        
        # Check if split is statistically significant
        degrees_of_freedom = abs(mark2 - mark1)
        critical_value = stats.t.ppf(
            self.config.confidence_level,
            degrees_of_freedom
        )
        
        is_significant = (
            abs(split_point - mark1) > self.config.min_points and
            abs(mark2 - mark1) > self.config.min_points and
            max_t > critical_value
        )
        
        if is_significant:
            self.change_points.append(split_point)
            self.change_points = sorted(list(set(self.change_points)))
            
            # Recurse on both sides of split
            self._recursive_segment(data, mark1, split_point, student_t)
            self._recursive_segment(data, split_point, mark2, student_t)
    
    @staticmethod
    def _calculate_segment_stats(
        data: np.ndarray,
        start: int,
        end: int,
        global_std: float
    ) -> Tuple[float, float, int]:
        """
        Calculate statistics for a data segment.
        
        Args:
            data: Time-series data
            start: Start index
            end: End index
            global_std: Global standard deviation
            
        Returns:
            Tuple of (mean, std, n_points)
        """
        segment = data[start:end]
        mean = np.mean(segment)
        std = max(np.std(segment), global_std * MIN_STD_FACTOR)
        n_points = max(len(segment), 1)
        
        return mean, std, n_points
    
    @staticmethod
    def _calculate_t_statistic(
        stats_left: Tuple[float, float, int],
        stats_right: Tuple[float, float, int]
    ) -> float:
        """
        Calculate Student-T statistic between two segments.
        
        Args:
            stats_left: (mean, std, n) for left segment
            stats_right: (mean, std, n) for right segment
            
        Returns:
            T-statistic value
        """
        mean_l, std_l, n_l = stats_left
        mean_r, std_r, n_r = stats_right
        
        pooled_std = (std_l * n_l + std_r * n_r) / (n_l + n_r)
        denominator = pooled_std * np.sqrt(1/n_l + 1/n_r)
        
        if denominator == 0:
            return 0.0
        
        return abs(mean_l - mean_r) / denominator


class RateMatrixBuilder:
    """Constructs rate matrices from change points."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
    
    def build(
        self,
        data: np.ndarray,
        change_points: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create rate matrix and standard deviations from change points.
        
        Args:
            data: Time-series data
            change_points: Indices of change points
            
        Returns:
            Tuple of (rate matrix, standard deviations)
        """
        # Add boundaries
        all_points = np.append(change_points, [0, len(data) - 1])
        all_points = np.sort(all_points)
        
        n_states = len(all_points) - 1
        rates = np.zeros(n_states * n_states)
        std_devs = np.zeros(n_states)
        
        # Calculate emission rates and standard deviations
        for i in range(n_states):
            cp1, cp2 = int(all_points[i]), int(all_points[i + 1])
            segment = data[cp1:cp2]
            
            rates[i] = np.mean(segment)
            std_devs[i] = max(
                np.std(segment),
                np.std(data) * self.config.min_std_factor
            )
        
        # Calculate transition rates
        self._fill_transition_rates(rates, n_states, len(data))
        
        return rates, std_devs
    
    def _fill_transition_rates(
        self,
        rates: np.ndarray,
        n_states: int,
        data_length: int
    ) -> None:
        """
        Fill transition rate portion of rate matrix.
        
        Args:
            rates: Rate matrix to fill
            n_states: Number of states
            data_length: Length of data for normalization
        """
        for i in range(n_states):
            if i == 0:
                continue
                
            for j in range(n_states):
                idx = i * n_states + j
                
                if j == 0:
                    # Reverse rates (decreasing as count rate increases)
                    rates[idx] = (n_states - i) / (
                        self.config.transition_rate_factor * data_length
                    )
                elif j == n_states - 1:
                    # Forward rates (increasing)
                    rates[idx] = i / (
                        self.config.transition_rate_factor * data_length
                    )
                else:
                    rates[idx] = SMALL_VALUE


class HMMAnalyzer:
    """Performs HMM-based analysis and reconstruction."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
    
    def calculate_log_likelihood(
        self,
        data: np.ndarray,
        rate_matrix: np.ndarray,
        std_devs: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """
        Calculate log-likelihood of model given data.
        
        Args:
            data: Time-series data
            rate_matrix: HMM rate matrix
            std_devs: Standard deviations for each state
            
        Returns:
            Tuple of (negative log-likelihood, HMM probability matrix)
        """
        n_states = len(std_devs)
        n_points = len(data)
        
        # Extract emission and transition rates
        emission_rates = rate_matrix[:n_states]
        
        # Validate rates
        if not self._validate_rates(emission_rates):
            logger.warning("Invalid emission rates detected")
            return LARGE_VALUE, np.zeros((n_points, n_states))
        
        # Build transition matrix
        transition_matrix = self._build_transition_matrix(
            rate_matrix,
            n_states
        )
        
        if transition_matrix is None:
            return LARGE_VALUE, np.zeros((n_points, n_states))
        
        # Initialize HMM matrix
        hmm = np.zeros((n_points, n_states), dtype=np.float64)
        hmm[0, :] = -1 / n_states
        
        # Forward pass
        self._forward_pass(data, hmm, emission_rates, std_devs, transition_matrix)
        
        # Backward pass
        self._backward_pass(data, hmm, emission_rates, std_devs, transition_matrix)
        
        # Calculate final log-likelihood
        max_likelihood = np.max(hmm[-1, :])
        
        return -max_likelihood, hmm
    
    def reconstruct_path(self, hmm_matrix: np.ndarray) -> np.ndarray:
        """
        Reconstruct most likely state path from HMM matrix.
        
        Args:
            hmm_matrix: HMM probability matrix
            
        Returns:
            Array of state indices representing the path
        """
        n_points, n_states = hmm_matrix.shape
        path = np.zeros(n_points, dtype=int)
        
        for i in range(n_points):
            max_state = np.argmax(hmm_matrix[i, :])
            max_prob = hmm_matrix[i, max_state]
            
            # Handle ties by using previous state
            if i > 0 and np.sum(hmm_matrix[i, :] == max_prob) > 1:
                path[i] = path[i - 1]
            else:
                path[i] = max_state
        
        return path
    
    @staticmethod
    def _validate_rates(rates: np.ndarray) -> bool:
        """Check if all rates are positive."""
        return np.all(rates > 0)
    
    def _build_transition_matrix(
        self,
        rate_matrix: np.ndarray,
        n_states: int
    ) -> Optional[np.ndarray]:
        """
        Build transition probability matrix from rates.
        
        Args:
            rate_matrix: Full rate matrix
            n_states: Number of states
            
        Returns:
            Transition matrix or None if invalid
        """
        transition_matrix = np.zeros((n_states, n_states))
        sum_rates = np.zeros(n_states)
        
        # Extract transition rates
        idx = n_states
        for i in range(n_states):
            for j in range(n_states):
                if i == j:
                    continue
                    
                rate = rate_matrix[idx]
                if rate <= 0:
                    logger.warning(f"Invalid transition rate: {rate}")
                    return None
                
                transition_matrix[i, j] = rate
                sum_rates[i] += rate
                idx += 1
        
        # Convert to log probabilities
        for i in range(n_states):
            trans_prob = 1.0 - np.exp(-sum_rates[i] * self.config.binning)
            
            for j in range(n_states):
                if i == j:
                    transition_matrix[i, j] = -sum_rates[i] * self.config.binning
                else:
                    if sum_rates[i] > 0:
                        transition_matrix[i, j] = np.log(
                            transition_matrix[i, j] * trans_prob / sum_rates[i]
                        )
        
        return transition_matrix
    
    def _forward_pass(
        self,
        data: np.ndarray,
        hmm: np.ndarray,
        emission_rates: np.ndarray,
        std_devs: np.ndarray,
        transition_matrix: np.ndarray
    ) -> None:
        """Forward recursion through data."""
        n_points, n_states = hmm.shape
        
        for i in range(1, n_points):
            for child in range(n_states):
                # Calculate emission probability
                emission_prob = norm.logpdf(
                    data[i],
                    loc=emission_rates[child],
                    scale=std_devs[child]
                )
                
                # Find best parent
                parent_probs = np.array([
                    hmm[i-1, parent] + transition_matrix[parent, child]
                    for parent in range(n_states)
                ])
                
                # Numerically stable log-sum-exp
                max_prob = np.max(parent_probs)
                log_sum = np.log(np.sum(np.exp(parent_probs - max_prob)))
                
                hmm[i, child] = log_sum + max_prob + emission_prob
    
    def _backward_pass(
        self,
        data: np.ndarray,
        hmm: np.ndarray,
        emission_rates: np.ndarray,
        std_devs: np.ndarray,
        transition_matrix: np.ndarray
    ) -> None:
        """Backward recursion through data."""
        n_points, n_states = hmm.shape
        
        for i in range(n_points - 2, -1, -1):
            for child in range(n_states):
                # Calculate emission probability
                emission_prob = norm.logpdf(
                    data[i],
                    loc=emission_rates[child],
                    scale=std_devs[child]
                )
                
                # Find best parent
                parent_probs = np.array([
                    hmm[i+1, parent] + transition_matrix[parent, child]
                    for parent in range(n_states)
                ])
                
                # Numerically stable log-sum-exp
                max_prob = np.max(parent_probs)
                log_sum = np.log(np.sum(np.exp(parent_probs - max_prob)))
                
                hmm[i, child] = log_sum + max_prob + emission_prob


class StateOptimizer:
    """Optimizes number of states using agglomerative clustering and BIC."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.hmm_analyzer = HMMAnalyzer(config)
        self.rate_builder = RateMatrixBuilder(config)
    
    def prune_unaccessed(
        self,
        path: np.ndarray,
        rates: np.ndarray,
        std_devs: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Remove states that are never accessed in reconstruction.
        
        Args:
            path: Reconstructed state path
            rates: Rate matrix
            std_devs: Standard deviations
            
        Returns:
            Tuple of (pruned rates, pruned std_devs)
        """
        n_states = len(std_devs)
        
        # Count state occurrences
        state_counts = np.array([
            np.sum(path == i) for i in range(n_states)
        ])
        
        # Keep only accessed states
        mask = state_counts > 0
        emission_rates = rates[:n_states][mask]
        new_std_devs = std_devs[mask]
        
        # Rebuild rate matrix with new number of states
        new_n_states = len(emission_rates)
        new_rates = np.zeros(new_n_states * new_n_states)
        new_rates[:new_n_states] = emission_rates
        
        self.rate_builder._fill_transition_rates(
            new_rates,
            new_n_states,
            len(path)
        )
        
        return new_rates, new_std_devs
    
    def agglomerate(
        self,
        data: np.ndarray,
        rates: np.ndarray,
        std_devs: np.ndarray,
        target_states: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reduce number of states through Ward's clustering.
        
        Args:
            data: Time-series data
            rates: Current rate matrix
            std_devs: Current standard deviations
            target_states: Target number of states
            
        Returns:
            Tuple of (new rates, new std_devs)
        """
        n_states = len(std_devs)
        
        # Sort by emission rate
        emission_rates = rates[:n_states]
        sorted_indices = np.argsort(emission_rates)
        emission_rates = emission_rates[sorted_indices]
        std_devs = std_devs[sorted_indices]
        
        while n_states > target_states:
            # Get current path
            temp_rates = np.zeros(n_states * n_states)
            temp_rates[:n_states] = emission_rates
            self.rate_builder._fill_transition_rates(
                temp_rates,
                n_states,
                len(data)
            )
            
            _, hmm = self.hmm_analyzer.calculate_log_likelihood(
                data,
                temp_rates,
                std_devs
            )
            path = self.hmm_analyzer.reconstruct_path(hmm)
            
            # Count state occurrences
            state_counts = np.array([
                max(np.sum(path == i), 1) for i in range(n_states)
            ])
            
            # Calculate Ward's distances
            wards = self._calculate_wards_distance(
                emission_rates,
                state_counts
            )
            
            # Find closest pair to merge
            min_idx = np.argmin(wards)
            i, j = np.unravel_index(min_idx, wards.shape)
            
            # Merge states
            emission_rates, std_devs = self._merge_states(
                i, j,
                emission_rates,
                std_devs,
                state_counts
            )
            
            n_states -= 1
        
        # Build final rate matrix
        final_rates = np.zeros(n_states * n_states)
        final_rates[:n_states] = emission_rates
        self.rate_builder._fill_transition_rates(
            final_rates,
            n_states,
            len(data)
        )
        
        return final_rates, std_devs
    
    @staticmethod
    def _calculate_wards_distance(
        rates: np.ndarray,
        counts: np.ndarray
    ) -> np.ndarray:
        """Calculate Ward's minimum variance distances."""
        n = len(rates)
        wards = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    wards[i, j] = LARGE_VALUE
                else:
                    factor = np.sqrt(
                        2 * counts[i] * counts[j] / (counts[i] + counts[j])
                    )
                    wards[i, j] = factor * (rates[i] - rates[j]) ** 2
        
        return wards
    
    @staticmethod
    def _merge_states(
        i: int,
        j: int,
        rates: np.ndarray,
        std_devs: np.ndarray,
        counts: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Merge two states using weighted averaging."""
        # Weighted average of rates
        new_rate = (
            (rates[i] * counts[i] + rates[j] * counts[j]) /
            (counts[i] + counts[j])
        )
        
        # Pooled standard deviation
        term1 = max(counts[i] - 1, 0.01) * std_devs[i] ** 2
        term2 = max(counts[j] - 1, 0.01) * std_devs[j] ** 2
        term3 = (
            counts[i] * counts[j] * (rates[i] - rates[j]) ** 2 /
            (counts[i] + counts[j])
        )
        term4 = counts[i] + counts[j] - 1
        
        new_std = np.sqrt((term1 + term2 + term3) / term4)
        
        # Update arrays
        new_rates = np.delete(rates, j)
        new_rates[i] = new_rate
        
        new_std_devs = np.delete(std_devs, j)
        new_std_devs[i] = new_std
        
        return new_rates, new_std_devs


class ResultsVisualizer:
    """Handles visualization and saving of results."""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
    
    def plot_bic_curve(
        self,
        n_states: np.ndarray,
        bic_values: np.ndarray,
        column_name: str
    ) -> None:
        """Plot BIC vs number of states."""
        plt.figure(figsize=(8, 6))
        plt.plot(n_states, bic_values, '-o', linewidth=2, markersize=8)
        plt.xlabel('Number of States', fontsize=12)
        plt.ylabel('BIC', fontsize=12)
        plt.title(f'BIC Curve - Column {column_name}', fontsize=14)
        plt.grid(True, alpha=0.3)
        
        output_path = self.config.output_dir / f'BIC_{column_name}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved BIC plot to {output_path}")
    
    def plot_reconstruction(
        self,
        data: np.ndarray,
        path: np.ndarray,
        column_name: str
    ) -> None:
        """Plot original data with reconstructed path."""
        time_axis = np.arange(0, len(data) * self.config.binning, self.config.binning)
        
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Plot intensity
        color1 = 'tab:red'
        ax1.set_xlabel('Time (s)', fontsize=12)
        ax1.set_ylabel('Intensity', color=color1, fontsize=12)
        ax1.plot(time_axis, data, color=color1, alpha=0.7, linewidth=1)
        ax1.tick_params(axis='y', labelcolor=color1)
        
        # Plot path
        ax2 = ax1.twinx()
        color2 = 'tab:blue'
        ax2.set_ylabel('State', color=color2, fontsize=12)
        ax2.plot(time_axis, path, color=color2, linewidth=2)
        ax2.tick_params(axis='y', labelcolor=color2)
        
        plt.title(f'Reconstruction - Column {column_name}', fontsize=14)
        fig.tight_layout()
        
        output_path = self.config.output_dir / f'Reconstruct_{column_name}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved reconstruction plot to {output_path}")
    
    def plot_stoichiometry_histogram(
        self,
        result_histogram: np.ndarray,
        filename: str = 'StoichTotal'
    ) -> None:
        """Plot histogram of stoichiometry results."""
        plt.figure(figsize=(8, 6))
        n_states = np.arange(1, len(result_histogram) + 1)
        plt.bar(n_states, result_histogram, color='steelblue', alpha=0.7)
        plt.xlabel('Number of GFP', fontsize=12)
        plt.ylabel('Number of Observations', fontsize=12)
        plt.title('Stoichiometry Distribution', fontsize=14)
        plt.grid(axis='y', alpha=0.3)
        
        output_path = self.config.output_dir / f'{filename}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved stoichiometry plot to {output_path}")


class DataProcessor:
    """Main processor for HMM analysis pipeline."""
    
    def __init__(self, config: Optional[AnalysisConfig] = None):
        self.config = config or AnalysisConfig()
        self.detector = ChangePointDetector(self.config)
        self.rate_builder = RateMatrixBuilder(self.config)
        self.hmm_analyzer = HMMAnalyzer(self.config)
        self.optimizer = StateOptimizer(self.config)
        self.visualizer = ResultsVisualizer(self.config)
    
    def process_file(self, file_path: Path) -> None:
        """
        Process a data file through the complete analysis pipeline.
        
        Args:
            file_path: Path to input CSV file
        """
        logger.info(f"Processing file: {file_path}")
        
        # Load data
        df = self._load_data(file_path)
        binaries = df.columns.values
        
        # Initialize result storage
        path_df = pd.DataFrame()
        rate_df = pd.DataFrame()
        sdev_df = pd.DataFrame()
        result_hist = np.zeros(3)
        
        # Process each column
        for idx, col in enumerate(df.columns):
            if binaries[idx] == 0:
                logger.info(f"Skipping column {col} (binary==0)")
                continue
            
            logger.info(f"Processing column {col}")
            
            try:
                results = self._process_column(df[col].values, col)
                
                # Store results
                path_df[f'{col}_path'] = results['path']
                rate_df[f'{col}_rmat'] = pd.Series(results['rates'])
                sdev_df[f'{col}_sdev'] = pd.Series(results['std_devs'])
                result_hist[results['n_states'] - 2] += 1
                
            except Exception as e:
                logger.error(f"Error processing column {col}: {e}", exc_info=True)
                continue
        
        # Save results
        self._save_results(path_df, rate_df, sdev_df, result_hist)
        
        logger.info("Processing complete")
    
    def _load_data(self, file_path: Path) -> pd.DataFrame:
        """Load and prepare data from CSV file."""
        df = pd.read_csv(
            file_path,
            header=None,
            index_col=0,
            delim_whitespace=True,
            dtype=np.float64
        ).T
        
        binaries = df.columns.values
        df.columns = list(range(len(binaries)))
        
        logger.info(f"Loaded data shape: {df.shape}")
        return df
    
    def _process_column(
        self,
        data: np.ndarray,
        column_name: str
    ) -> dict:
        """Process a single data column."""
        # Detect change points
        _, change_points = self.detector.detect(data)
        
        # Build initial rate matrix
        rates, std_devs = self.rate_builder.build(data, change_points)
        n_states = int(np.sqrt(len(rates)))
        
        logger.info(f"  Initial segments: {n_states}")
        
        # Optimize number of states using BIC
        best_rates, best_std_devs, best_n_states = self._optimize_states(
            data,
            rates,
            std_devs,
            column_name
        )
        
        # Final reconstruction
        log_p, hmm = self.hmm_analyzer.calculate_log_likelihood(
            data,
            best_rates,
            best_std_devs
        )
        path = self.hmm_analyzer.reconstruct_path(hmm)
        
        # Visualize results
        self.visualizer.plot_reconstruction(data, path, column_name)
        
        logger.info(f"  Final states: {best_n_states}")
        logger.info(f"  Emission rates: {best_rates[:best_n_states]}")
        
        return {
            'path': path,
            'rates': best_rates,
            'std_devs': best_std_devs,
            'n_states': best_n_states
        }
    
    def _optimize_states(
        self,
        data: np.ndarray,
        rates: np.ndarray,
        std_devs: np.ndarray,
        column_name: str
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        """Optimize number of states using BIC."""
        n_points = len(data)
        bic_values = []
        state_range = range(4, 1, -1)  # Try 4, 3, 2 states
        
        best_bic = LARGE_VALUE
        best_rates = rates
        best_std_devs = std_devs
        
        for n_states in state_range:
            # Agglomerate to target number of states
            test_rates, test_std_devs = self.optimizer.agglomerate(
                data,
                rates,
                std_devs,
                n_states
            )
            
            # Calculate likelihood
            log_p, hmm = self.hmm_analyzer.calculate_log_likelihood(
                data,
                test_rates,
                test_std_devs
            )
            
            if log_p == LARGE_VALUE:
                logger.warning(f"  Invalid model for {n_states} states")
                bic_values.append(LARGE_VALUE)
                continue
            
            # Prune unaccessed states
            path = self.hmm_analyzer.reconstruct_path(hmm)
            test_rates, test_std_devs = self.optimizer.prune_unaccessed(
                path,
                test_rates,
                test_std_devs
            )
            
            # Recalculate with pruned states
            log_p, hmm = self.hmm_analyzer.calculate_log_likelihood(
                data,
                test_rates,
                test_std_devs
            )
            
            if log_p == LARGE_VALUE:
                bic_values.append(LARGE_VALUE)
                continue
            
            # Calculate BIC
            actual_n_states = int(np.sqrt(len(test_rates)))
            bic = 2 * log_p + actual_n_states ** 2 * np.log(n_points)
            bic_values.append(bic)
            
            logger.info(f"  {n_states} states: BIC = {bic:.1f}")
            
            if bic < best_bic:
                best_bic = bic
                best_rates = test_rates
                best_std_devs = test_std_devs
        
        # Plot BIC curve
        self.visualizer.plot_bic_curve(
            np.array(list(state_range)),
            np.array(bic_values),
            column_name
        )
        
        best_n_states = int(np.sqrt(len(best_rates)))
        return best_rates, best_std_devs, best_n_states
    
    def _save_results(
        self,
        path_df: pd.DataFrame,
        rate_df: pd.DataFrame,
        sdev_df: pd.DataFrame,
        result_hist: np.ndarray
    ) -> None:
        """Save analysis results to files."""
        # Save paths
        path_df.to_excel('Reconstructed_paths.xlsx')
        logger.info("Saved reconstructed paths")
        
        # Save rates and standard deviations
        with pd.ExcelWriter('Optimized_Rates.xlsx', engine='xlsxwriter') as writer:
            rate_df.to_excel(writer, sheet_name='rates')
            sdev_df.to_excel(writer, sheet_name='stdevs')
            pd.DataFrame(
                result_hist,
                columns=['FinalHist']
            ).to_excel(writer, sheet_name='FinalHist')
        
        logger.info("Saved optimized rates and statistics")
        
        # Plot final histogram
        self.visualizer.plot_stoichiometry_histogram(result_hist)


def select_file() -> Optional[Path]:
    """Open file dialog to select input file."""
    root = tk.Tk()
    root.withdraw()
    
    file_path = filedialog.askopenfilename(
        parent=root,
        title="Select data file",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    
    root.destroy()
    
    return Path(file_path) if file_path else None


def main():
    """Main entry point for the analysis."""
    # Select input file
    file_path = select_file()
    
    if not file_path:
        logger.error("No file selected. Exiting.")
        return
    
    # Create configuration
    config = AnalysisConfig()
    
    # Process file
    processor = DataProcessor(config)
    processor.process_file(file_path)


if __name__ == "__main__":
    main()