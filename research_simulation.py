#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Research-grade turbo code simulation with smooth waterfall curves
"""

import numpy as np
from simulation import TurboSimulation
import matplotlib.pyplot as plt

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
    import matplotlib.pyplot as plt
    import seaborn as sns
    
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

def run_extended_snr_simulation():
    """Run simulation with extended SNR range to capture full performance"""
    print("\n=== Extended SNR Range Simulation ===")
    print("Running with wider SNR range to capture complete waterfall...")
    
    # Extended configuration
    config = {
        'snr_range': np.arange(-1.0, 4.1, 0.2),   # Wider range, finer steps
        'iterations': [4, 8],                      # Compare iteration effects
        'num_bits': 100000,                       # Even more bits
        'num_runs': 5,                            # Balanced runs vs time
        'seed': 42,
        'output_dir': 'extended_results'
    }
    
    print(f"SNR range: {config['snr_range'][0]} to {config['snr_range'][-1]} dB")
    print(f"Total SNR points: {len(config['snr_range'])}")
    
    sim = TurboSimulation(config)
    results_df = sim.run_parallel()
    
    # Custom plotting for extended range
    plot_extended_analysis(results_df, config)
    
    return results_df

def plot_extended_analysis(df, config):
    """Create plots for extended SNR range analysis"""
    plt.figure(figsize=(14, 10))
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    colors = {'MAP': 'blue', 'Log-MAP': 'red', 'Max-Log-MAP': 'green'}
    
    # Plot 1: 4 iterations
    for decoder in ['MAP', 'Log-MAP', 'Max-Log-MAP']:
        data = df[(df['decoder'] == decoder) & (df['iterations'] == 4)]
        stats = data.groupby('snr')['ber'].mean()
        non_zero = stats[stats > 0]
        
        ax1.semilogy(non_zero.index, non_zero.values, 'o-', 
                    color=colors[decoder], label=f'{decoder}', linewidth=2)
    
    ax1.grid(True, which="both", alpha=0.3)
    ax1.set_xlabel('Eb/N0 (dB)')
    ax1.set_ylabel('BER')
    ax1.set_title('4 Iterations')
    ax1.legend()
    ax1.set_ylim(1e-5, 1e-1)
    
    # Plot 2: 8 iterations  
    for decoder in ['MAP', 'Log-MAP', 'Max-Log-MAP']:
        data = df[(df['decoder'] == decoder) & (df['iterations'] == 8)]
        stats = data.groupby('snr')['ber'].mean()
        non_zero = stats[stats > 0]
        
        ax2.semilogy(non_zero.index, non_zero.values, 'o-',
                    color=colors[decoder], label=f'{decoder}', linewidth=2)
    
    ax2.grid(True, which="both", alpha=0.3)
    ax2.set_xlabel('Eb/N0 (dB)')
    ax2.set_ylabel('BER')
    ax2.set_title('8 Iterations')
    ax2.legend()
    ax2.set_ylim(1e-5, 1e-1)
    
    # Plot 3: Iteration comparison for MAP
    for iterations in [4, 8]:
        data = df[(df['decoder'] == 'MAP') & (df['iterations'] == iterations)]
        stats = data.groupby('snr')['ber'].mean()
        non_zero = stats[stats > 0]
        
        ax3.semilogy(non_zero.index, non_zero.values, 'o-',
                    label=f'{iterations} iterations', linewidth=2)
    
    ax3.grid(True, which="both", alpha=0.3)
    ax3.set_xlabel('Eb/N0 (dB)')
    ax3.set_ylabel('BER')
    ax3.set_title('MAP Decoder - Iteration Comparison')
    ax3.legend()
    ax3.set_ylim(1e-5, 1e-1)
    
    # Plot 4: Complexity vs Performance tradeoff
    max_iter_data = df[df['iterations'] == 8]
    complexity = {}
    performance = {}
    
    for decoder in ['MAP', 'Log-MAP', 'Max-Log-MAP']:
        data = max_iter_data[max_iter_data['decoder'] == decoder]
        # Average complexity
        complexity[decoder] = data['operations'].apply(lambda x: sum(x.values())).mean()
        # BER at 1.5 dB (middle of waterfall)
        ber_1_5 = data[data['snr'] == 1.5]['ber'].mean() if 1.5 in data['snr'].values else None
        performance[decoder] = ber_1_5
    
    # Filter valid performance data
    valid_decoders = [d for d in performance.keys() if performance[d] is not None and performance[d] > 0]
    
    if valid_decoders:
        x_vals = [complexity[d]/1e6 for d in valid_decoders]  # Millions of operations
        y_vals = [performance[d] for d in valid_decoders]
        
        ax4.semilogy(x_vals, y_vals, 'o', markersize=10)
        for i, decoder in enumerate(valid_decoders):
            ax4.annotate(decoder, (x_vals[i], y_vals[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        ax4.grid(True, which="both", alpha=0.3)
        ax4.set_xlabel('Complexity (Million Operations)')
        ax4.set_ylabel('BER at 1.5 dB')
        ax4.set_title('Performance vs Complexity Tradeoff')
    
    plt.tight_layout()
    plt.savefig(f"{config['output_dir']}/extended_analysis.png", dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    print("Choose simulation type:")
    print("1. Research-grade (fine SNR steps, high statistics)")
    print("2. Extended SNR range (wider range, good statistics)")
    print("3. Both (recommended)")
    
    choice = input("Enter choice (1, 2, or 3): ").strip()
    
    if choice in ['1', '3']:
        research_results = run_research_grade_simulation()
    
    if choice in ['2', '3']:
        extended_results = run_extended_snr_simulation()
    
    print("\n=== All simulations completed! ===")
    print("Check the generated plots for smooth waterfall curves reaching 10^-5 BER levels.")
