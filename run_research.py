#!/usr/bin/env python3
"""
Script to run research-grade turbo code simulation
"""

import numpy as np
from simulation import TurboSimulation
import matplotlib.pyplot as plt
import os

def run_research_grade_simulation():
    """Run a research-grade simulation with smooth waterfall curves"""
    print("=== Research-Grade Turbo Code Simulation ===")
    print("This will take longer but produce smooth waterfall curves reaching 10^-5 BER")
    
    # Research-grade configuration
    config = {
        # Fine-grained SNR range for smooth curves
        'snr_range': np.arange(-0.5, 3.1, 0.25),  # -0.5 to 3.0 dB in 0.25 dB steps
        'iterations': [8],                         # Focus on best performance
        'num_bits': 50000,                        # More bits for better low-BER statistics  
        'num_runs': 10,                           # More runs for averaging
        'seed': 42,
        'output_dir': 'research_results'
    }
    
    print(f"SNR range: {config['snr_range'][0]} to {config['snr_range'][-1]} dB")
    print(f"SNR points: {len(config['snr_range'])}")
    print(f"Bits per run: {config['num_bits']:,}")
    print(f"Total bits per SNR point: {config['num_bits'] * config['num_runs']:,}")
    print(f"This will simulate {len(config['snr_range']) * 3 * config['num_runs']} runs")
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Create and run simulation
    sim = TurboSimulation(config)
    
    print("\nRunning research-grade simulation...")
    print("This may take 15-20 minutes to complete...")
    results_df = sim.run_parallel()
    
    print(f"\nResults shape: {results_df.shape}")
    
    # Generate research-quality plots
    print("\nGenerating research-quality plots...")
    sim.plot_results(results_df)
    
    # Create additional detailed analysis
    create_detailed_analysis(results_df, config)
    
    print("\nResearch-grade simulation completed!")
    return results_df

def create_detailed_analysis(df, config):
    """Create detailed analysis plots and statistics"""
    
    # Set publication-quality plot style
    plt.style.use('default')
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
        'lines.linewidth': 2,
        'lines.markersize': 6
    })
    
    # 1. High-quality waterfall plot
    plt.figure(figsize=(12, 8))
    
    colors = {'MAP': 'blue', 'Log-MAP': 'red', 'Max-Log-MAP': 'green'}
    markers = {'MAP': 'o', 'Log-MAP': 's', 'Max-Log-MAP': '^'}
    
    max_iter = max(config['iterations'])
    df_analysis = df[df['iterations'] == max_iter]
    
    for decoder in ['MAP', 'Log-MAP', 'Max-Log-MAP']:
        data = df_analysis[df_analysis['decoder'] == decoder]
        
        # Calculate statistics
        stats = data.groupby('snr')['ber'].agg(['mean', 'std', 'count']).reset_index()
        
        # Filter out zero BER points for log scale, but keep them for analysis
        non_zero_stats = stats[stats['mean'] > 0].copy()
        zero_ber_snrs = stats[stats['mean'] == 0]['snr'].values
        
        if len(non_zero_stats) > 0:
            # Plot non-zero BER points
            plt.semilogy(non_zero_stats['snr'], non_zero_stats['mean'], 
                        color=colors[decoder], marker=markers[decoder], 
                        label=f'{decoder}', linestyle='-', alpha=0.8)
            
            # Add error bars
            plt.fill_between(non_zero_stats['snr'],
                           np.maximum(non_zero_stats['mean'] - non_zero_stats['std'], 1e-6),
                           non_zero_stats['mean'] + non_zero_stats['std'],
                           color=colors[decoder], alpha=0.2)
        
        # Mark zero BER points with arrows
        if len(zero_ber_snrs) > 0:
            min_ber = 1e-6  # Lower bound for log scale
            for snr in zero_ber_snrs:
                plt.annotate(f'{decoder}\nBER < 10^-5', 
                           xy=(snr, min_ber), xytext=(snr, min_ber*10),
                           arrowprops=dict(arrowstyle='->', color=colors[decoder]),
                           ha='center', fontsize=10, color=colors[decoder])
    
    plt.grid(True, which="both", alpha=0.3)
    plt.xlabel('Eb/N0 (dB)')
    plt.ylabel('Bit Error Rate (BER)')
    plt.title('Turbo Code Performance - Waterfall Curves')
    plt.legend()
    plt.ylim(1e-6, 1e-1)
    plt.xlim(config['snr_range'][0] - 0.1, config['snr_range'][-1] + 0.1)
    plt.tight_layout()
    plt.savefig(f"{config['output_dir']}/waterfall_curves_detailed.png", dpi=300, bbox_inches='tight')
    
    # 2. Performance summary table
    print("\n=== Detailed Performance Analysis ===")
    print("\nBER Performance Summary (8 iterations):")
    print("-" * 60)
    
    for decoder in ['MAP', 'Log-MAP', 'Max-Log-MAP']:
        print(f"\n{decoder}:")
        data = df_analysis[df_analysis['decoder'] == decoder]
        stats = data.groupby('snr')['ber'].agg(['mean', 'std', 'min', 'max'])
        
        for snr, row in stats.iterrows():
            if row['mean'] > 0:
                print(f"  {snr:4.1f} dB: {row['mean']:.2e} ± {row['std']:.2e} "
                      f"(range: {row['min']:.2e} - {row['max']:.2e})")
            else:
                print(f"  {snr:4.1f} dB: < 10^-5 (no errors detected)")
    
    # 3. Find waterfall regions
    print("\n=== Waterfall Region Analysis ===")
    for decoder in ['MAP', 'Log-MAP', 'Max-Log-MAP']:
        data = df_analysis[df_analysis['decoder'] == decoder]
        stats = data.groupby('snr')['ber'].mean()
        
        # Find 10^-2 and 10^-4 crossover points
        high_ber_snrs = stats[stats > 1e-2].index
        low_ber_snrs = stats[stats < 1e-4].index
        
        if len(high_ber_snrs) > 0 and len(low_ber_snrs) > 0:
            waterfall_start = high_ber_snrs[-1] if len(high_ber_snrs) > 0 else "N/A"
            waterfall_end = low_ber_snrs[0] if len(low_ber_snrs) > 0 else "N/A"
            waterfall_width = waterfall_end - waterfall_start if waterfall_start != "N/A" and waterfall_end != "N/A" else "N/A"
            print(f"{decoder}: Waterfall from {waterfall_start:.1f} to {waterfall_end:.1f} dB "
                  f"(width: {waterfall_width:.1f} dB)")

if __name__ == "__main__":
    results = run_research_grade_simulation()
    print("\nResearch simulation completed! Check the research_results directory for plots.")
