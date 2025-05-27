"""
Channel Model Implementation
"""
import numpy as np

class Channel:
    def __init__(self, channel_type="AWGN"):
        """
        Initialize a channel model.
        
        Args:
            channel_type: Type of channel ("AWGN", "Rayleigh", etc.)
        """
        self.channel_type = channel_type.upper()
    
    def add_noise(self, signal, snr_db, code_rate=1/3):
        """
        Add noise to the signal based on the specified SNR.
        
        Args:
            signal: Input signal (BPSK modulated: +1 -> bit 0, -1 -> bit 1)
            snr_db: Signal-to-Noise Ratio in dB (Eb/N0)
            code_rate: Code rate of the encoder
            
        Returns:
            Noisy signal
        """
        # Calculate noise power based on SNR
        signal_power = 1.0  # For BPSK, signal power is 1
        snr_linear = 10 ** (snr_db / 10.0)
        # For BPSK in AWGN with coding: σ² = 1/(2*R*SNR), where R is code rate
        noise_power = signal_power / (2 * code_rate * snr_linear)
        sigma = np.sqrt(noise_power)
        
        if self.channel_type == "AWGN":
            # Generate noise according to N(0, σ²)
            noise = sigma * np.random.randn(len(signal))
            return signal + noise
        else:
            raise ValueError(f"Channel type {self.channel_type} not supported")
    
    def modulate(self, bits):
        """
        BPSK modulation: 0 -> +1, 1 -> -1
        
        Args:
            bits: Binary input bits
            
        Returns:
            BPSK modulated signal
        """
        return 1 - 2 * bits  # Convert 0->+1, 1->-1
    
    def calculate_llr(self, received_signal, snr_db, code_rate=1/3):
        """
        Calculate Log-Likelihood Ratios (LLRs) from received signal.
        
        Args:
            received_signal: Received noisy signal
            snr_db: Signal-to-Noise Ratio in dB (Eb/N0)
            code_rate: Code rate of the encoder
            
        Returns:
            LLRs (positive -> likely bit 0, negative -> likely bit 1)
        """
        snr_linear = 10 ** (snr_db / 10.0)
        noise_variance = 1.0 / (2 * code_rate * snr_linear)
        
        # For BPSK in AWGN: LLR = 2y/σ² where y is received signal
        # Positive LLR means more likely bit 0 (+1)
        # Negative LLR means more likely bit 1 (-1)
        return 2.0 * received_signal / noise_variance
