#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Turbo Code Performance Comparison Simulation
Compares MAP (BCJR), Log-MAP, and Max-Log-MAP decoders

Features:
- Multiple SNR points
- Multiple iteration counts
- Operation counting for complexity analysis
- Parallel processing for speed
- Comprehensive plotting and analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import multiprocessing as mp
from itertools import product
import pickle
import os
import time

from turbo_decoder import MAPDecoder, LogMAPDecoder, MaxLogMAPDecoder
from utils import (TurboEncoder, QPSK_Modulator, AWGN_Channel, 
                  calculate_ber, interleave, deinterleave)

class TurboSimulation:
    """
    Turbo code simulation framework for decoder comparison
    """
    def __init__(self, config):
        """
        Initialize simulation with configuration
        
        Parameters:
        ----------
        config : dict
            Simulation configuration containing:
            - snr_range: list of Eb/N0 values in dB
            - iterations: list of iteration counts
            - num_bits: number of bits per run
            - num_runs: number of runs per configuration
            - seed: random seed for reproducibility
        """
        self.config = config
        self.decoders = {
            'MAP': MAPDecoder(),
            'Log-MAP': LogMAPDecoder(),
            'Max-Log-MAP': MaxLogMAPDecoder()
        }
        
        # Set random seed
        np.random.seed(config['seed'])
        
        # Initialize components
        self.encoder = TurboEncoder()
        self.modulator = QPSK_Modulator()
        self.channel = AWGN_Channel()
        
        # Create results directory
        os.makedirs(config['output_dir'], exist_ok=True)
    
    def _single_run(self, params):
        """
        Run a single simulation with given parameters
        
        Parameters:
        ----------
        params : tuple
            (decoder_type, snr_db, iteration_count, run_number)
        
        Returns:
        -------
        dict
            Results for this run
        """
        decoder_type, snr_db, iteration_count, run_number = params
        
        # Generate random bits
        info_bits = np.random.randint(0, 2, self.config['num_bits'])
        
        # Encode
        encoded_bits, systematic_bits, parity1_bits, parity2_bits, interleaver_indices = self.encoder.encode(info_bits)
        
        # Modulate
        symbols = self.modulator.modulate(encoded_bits)
        
        # Add noise
        noisy_symbols = self.channel.add_noise(symbols, snr_db)
        
        # Demodulate
        llrs = self.modulator.demodulate(noisy_symbols, self.channel.noise_variance)
        
        # Decode
        decoder = self.decoders[decoder_type]
        decoded_bits, total_operations = self._iterative_decode(decoder, llrs, info_bits.size, iteration_count, interleaver_indices)
        
        # Calculate BER
        ber = calculate_ber(info_bits, decoded_bits)
        
        return {
            'decoder': decoder_type,
            'snr': snr_db,
            'iterations': iteration_count,
            'run': run_number,
            'ber': ber,
            'operations': total_operations
        }
    
    def _iterative_decode(self, decoder, llrs, length, num_iterations, interleaver_indices):
        """
        Perform iterative decoding
        
        Parameters:
        ----------
        decoder : TurboDecoderBase
            Decoder instance to use
        llrs : ndarray
            Input LLRs
        length : int
            Number of information bits
        num_iterations : int
            Number of iterations to perform
        interleaver_indices : ndarray
            Interleaver indices used during encoding
            
        Returns:
        -------
        tuple
            (decoded_bits, total_operations)
        """
        # Extract systematic and parity LLRs
        systematic_llrs = llrs[0::3]
        parity1_llrs = llrs[1::3]
        parity2_llrs = llrs[2::3]
        
        # Initialize extrinsic information
        extrinsic_1 = np.zeros(length)
        extrinsic_2 = np.zeros(length)
        
        # Initialize total operation counter
        total_operations = {
            'additions': 0,
            'multiplications': 0,
            'comparisons': 0,
            'exponentiations': 0
        }
        
        # Iterative decoding
        for _ in range(num_iterations):
            # Decoder 1
            input_1 = np.zeros(2 * length)
            input_1[0::2] = systematic_llrs + extrinsic_2  # Add a priori info
            input_1[1::2] = parity1_llrs
            
            decoder.reset_counters()
            extrinsic_1 = decoder.decode(input_1, length)
            
            # Accumulate operations
            for op_type, count in decoder.op_counter.items():
                total_operations[op_type] += count
            
            # Interleave for decoder 2 using the same indices from encoding
            interleaved_systematic = systematic_llrs[interleaver_indices]
            interleaved_extrinsic = extrinsic_1[interleaver_indices]
            
            # Decoder 2
            input_2 = np.zeros(2 * length)
            input_2[0::2] = interleaved_systematic + interleaved_extrinsic
            input_2[1::2] = parity2_llrs
            
            decoder.reset_counters()
            temp_extrinsic = decoder.decode(input_2, length)
            
            # Accumulate operations
            for op_type, count in decoder.op_counter.items():
                total_operations[op_type] += count
            
            # Deinterleave for next iteration
            inverse_indices = np.argsort(interleaver_indices)
            extrinsic_2 = temp_extrinsic[inverse_indices]
        
        # Make final decisions
        final_llrs = systematic_llrs + extrinsic_1 + extrinsic_2
        decoded_bits = (final_llrs < 0).astype(int)
        
        return decoded_bits, total_operations
    
    def run_parallel(self):
        """
        Run simulation using parallel processing
        
        Returns:
        -------
        pd.DataFrame
            Results dataframe
        """
        print("Starting parallel simulation...")
        start_time = time.time()
        
        # Create parameter combinations
        params = list(product(
            self.decoders.keys(),
            self.config['snr_range'],
            self.config['iterations'],
            range(self.config['num_runs'])
        ))
        
        # Run simulations in parallel
        with mp.Pool() as pool:
            results = list(tqdm(
                pool.imap_unordered(self._single_run, params),
                total=len(params),
                desc="Progress"
            ))
        
        # Convert results to DataFrame
        df = pd.DataFrame(results)
        
        # Save results
        df.to_csv(f"{self.config['output_dir']}/simulation_results.csv", index=False)
        with open(f"{self.config['output_dir']}/simulation_results.pkl", 'wb') as f:
            pickle.dump(df, f)
        
        print(f"Simulation completed in {time.time() - start_time:.2f} seconds")
        return df
    
    def plot_results(self, df):
        """
        Create plots from simulation results
        
        Parameters:
        ----------
        df : pd.DataFrame
            Results dataframe
        """
        print("Generating plots...")
        
        # Set plot style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. BER vs SNR for each decoder at max iterations
        self._plot_ber_vs_snr_comparison(df)
        
        # 2. BER vs SNR for each iteration count (one plot per decoder)
        self._plot_ber_vs_snr_by_iterations(df)
        
        # 3. BER vs Iterations at fixed SNRs
        self._plot_ber_vs_iterations(df)
        
        # 4. Operation count comparison
        self._plot_operations_comparison(df)
        
        plt.close('all')
        print("Plots saved in output directory")
    
    def _plot_ber_vs_snr_comparison(self, df):
        """Plot BER vs SNR comparison of all decoders"""
        plt.figure(figsize=(10, 6))
        
        max_iter = max(self.config['iterations'])
        df_max_iter = df[df['iterations'] == max_iter]
        
        for decoder in self.decoders.keys():
            data = df_max_iter[df_max_iter['decoder'] == decoder]
            mean_ber = data.groupby('snr')['ber'].mean()
            std_ber = data.groupby('snr')['ber'].std()
            
            plt.semilogy(mean_ber.index, mean_ber.values, 'o-',
                        label=f'{decoder} ({max_iter} iterations)')
            plt.fill_between(mean_ber.index,
                           mean_ber - std_ber,
                           mean_ber + std_ber,
                           alpha=0.2)
        
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.xlabel('Eb/N0 (dB)')
        plt.ylabel('Bit Error Rate (BER)')
        plt.title('Decoder Performance Comparison')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.config['output_dir']}/ber_vs_snr_comparison.png", dpi=300)
    
    def _plot_ber_vs_snr_by_iterations(self, df):
        """Plot BER vs SNR for each iteration count"""
        for decoder in self.decoders.keys():
            plt.figure(figsize=(10, 6))
            
            for iter_count in self.config['iterations']:
                data = df[(df['decoder'] == decoder) & (df['iterations'] == iter_count)]
                mean_ber = data.groupby('snr')['ber'].mean()
                std_ber = data.groupby('snr')['ber'].std()
                
                plt.semilogy(mean_ber.index, mean_ber.values, 'o-',
                            label=f'{iter_count} iterations')
                plt.fill_between(mean_ber.index,
                               mean_ber - std_ber,
                               mean_ber + std_ber,
                               alpha=0.2)
            
            plt.grid(True, which="both", ls="-", alpha=0.2)
            plt.xlabel('Eb/N0 (dB)')
            plt.ylabel('Bit Error Rate (BER)')
            plt.title(f'{decoder} Performance at Different Iterations')
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"{self.config['output_dir']}/ber_vs_snr_{decoder.lower().replace('-', '_')}.png",
                       dpi=300)
    
    def _plot_ber_vs_iterations(self, df):
        """Plot BER vs Iterations at fixed SNRs"""
        selected_snrs = [1, 2, 3, 4]  # Selected SNR points to plot
        
        plt.figure(figsize=(10, 6))
        
        for decoder in self.decoders.keys():
            for snr in selected_snrs:
                data = df[(df['decoder'] == decoder) & (df['snr'] == snr)]
                mean_ber = data.groupby('iterations')['ber'].mean()
                std_ber = data.groupby('iterations')['ber'].std()
                
                plt.semilogy(mean_ber.index, mean_ber.values, 'o-',
                            label=f'{decoder}, {snr} dB')
                plt.fill_between(mean_ber.index,
                               mean_ber - std_ber,
                               mean_ber + std_ber,
                               alpha=0.2)
        
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.xlabel('Number of Iterations')
        plt.ylabel('Bit Error Rate (BER)')
        plt.title('BER vs Iterations at Different SNRs')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f"{self.config['output_dir']}/ber_vs_iterations.png",
                   dpi=300, bbox_inches='tight')
    
    def _plot_operations_comparison(self, df):
        """Plot operation count comparison"""
        plt.figure(figsize=(10, 6))
        
        max_iter = max(self.config['iterations'])
        df_max_iter = df[df['iterations'] == max_iter]
        
        for decoder in self.decoders.keys():
            data = df_max_iter[df_max_iter['decoder'] == decoder]
            total_ops = data.apply(lambda x: sum(x['operations'].values()), axis=1)
            mean_ops = total_ops.groupby(data['snr']).mean()
            
            plt.semilogy(mean_ops.index, mean_ops.values, 'o-',
                        label=f'{decoder} ({max_iter} iterations)')
        
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.xlabel('Eb/N0 (dB)')
        plt.ylabel('Total Operations')
        plt.title('Computational Complexity Comparison')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.config['output_dir']}/operations_comparison.png", dpi=300)


def main():
    """Main function to run the simulation"""
    # Simulation configuration
    config = {
        'snr_range': np.arange(0, 5.5, 0.5),  # Eb/N0 in dB
        'iterations': [1, 2, 4, 6, 8],        # Iteration counts
        'num_bits': 100000,                    # Bits per run
        'num_runs': 10,                        # Runs per configuration
        'seed': 42,                            # Random seed
        'output_dir': 'results'                # Output directory
    }
    
    # Create and run simulation
    sim = TurboSimulation(config)
    results_df = sim.run_parallel()
    
    # Generate plots
    sim.plot_results(results_df)
    
    # Print summary statistics
    print("\n=== Simulation Summary ===")
    
    # Compare decoders at highest SNR and max iterations
    max_snr = max(config['snr_range'])
    max_iter = max(config['iterations'])
    summary = results_df[
        (results_df['snr'] == max_snr) &
        (results_df['iterations'] == max_iter)
    ].groupby('decoder')['ber'].agg(['mean', 'std'])
    
    print(f"\nBER at {max_snr} dB, {max_iter} iterations:")
    print(summary)
    
    # Compare computational complexity
    ops_summary = results_df[
        (results_df['snr'] == max_snr) &
        (results_df['iterations'] == max_iter)
    ].groupby('decoder').apply(
        lambda x: pd.Series({
            'total_ops': x['operations'].apply(lambda ops: sum(ops.values())).mean()
        })
    )
    
    print("\nAverage operation counts:")
    print(ops_summary)

if __name__ == "__main__":
    main()
