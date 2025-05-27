#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility functions for turbo coding simulation
"""

import numpy as np


def random_interleaver(data, seed=42):
    """
    Create a random interleaver for the given data
    
    Parameters:
    ----------
    data : ndarray
        Input data to interleave
    seed : int
        Random seed for reproducibility
        
    Returns:
    -------
    interleaved_data : ndarray
        Interleaved data
    indices : ndarray
        Interleaver indices
    """
    np.random.seed(seed)
    indices = np.random.permutation(len(data))
    interleaved_data = data[indices]
    return interleaved_data, indices


def interleave(data, indices=None, seed=42):
    """
    Interleave data using given indices or create new ones
    
    Parameters:
    ----------
    data : ndarray
        Input data
    indices : ndarray, optional
        Interleaver indices
    seed : int
        Random seed if indices not provided
        
    Returns:
    -------
    ndarray
        Interleaved data
    """
    if indices is None:
        np.random.seed(seed)
        indices = np.random.permutation(len(data))
    return data[indices]


def deinterleave(data, indices=None, seed=42):
    """
    Deinterleave data using given indices
    
    Parameters:
    ----------
    data : ndarray
        Interleaved data
    indices : ndarray, optional
        Original interleaver indices
    seed : int
        Random seed if indices not provided
        
    Returns:
    -------
    ndarray
        Deinterleaved data
    """
    if indices is None:
        np.random.seed(seed)
        indices = np.random.permutation(len(data))
    
    # Create inverse mapping
    inverse_indices = np.argsort(indices)
    return data[inverse_indices]


def calculate_ber(transmitted_bits, received_bits):
    """
    Calculate Bit Error Rate (BER)
    
    Parameters:
    ----------
    transmitted_bits : ndarray
        Original transmitted bits
    received_bits : ndarray
        Received/decoded bits
        
    Returns:
    -------
    float
        Bit error rate
    """
    if len(transmitted_bits) != len(received_bits):
        raise ValueError("Transmitted and received bit arrays must have same length")
    
    errors = np.sum(transmitted_bits != received_bits)
    return errors / len(transmitted_bits)


class QPSK_Modulator:
    """
    QPSK Modulator for turbo coding simulation
    """
    
    def modulate(self, bits):
        """
        QPSK modulation: maps pairs of bits to complex symbols
        
        Parameters:
        ----------
        bits : ndarray
            Input bits
            
        Returns:
        -------
        ndarray
            Complex QPSK symbols
        """
        # For simplicity, use BPSK modulation (0 -> +1, 1 -> -1)
        return 1 - 2 * bits.astype(float)
    
    def demodulate(self, symbols, noise_variance):
        """
        QPSK demodulation: compute LLRs from received symbols
        
        Parameters:
        ----------
        symbols : ndarray
            Received noisy symbols
        noise_variance : float
            Noise variance
            
        Returns:
        -------
        ndarray
            Log-likelihood ratios
        """
        # For BPSK: LLR = 2*y/σ²
        return 2.0 * symbols / noise_variance


class AWGN_Channel:
    """
    Additive White Gaussian Noise channel
    """
    
    def __init__(self):
        self.noise_variance = None
    
    def add_noise(self, signal, snr_db, code_rate=1/3):
        """
        Add AWGN to the signal
        
        Parameters:
        ----------
        signal : ndarray
            Input signal
        snr_db : float
            SNR in dB
        code_rate : float
            Code rate
            
        Returns:
        -------
        ndarray
            Noisy signal
        """
        # Calculate noise variance
        snr_linear = 10 ** (snr_db / 10.0)
        self.noise_variance = 1.0 / (2 * code_rate * snr_linear)
        
        # Add noise
        noise = np.sqrt(self.noise_variance) * np.random.randn(len(signal))
        return signal + noise


class TurboEncoder:
    """
    Placeholder TurboEncoder that uses the one from turbo_encoder.py
    """
    
    def __init__(self, generator_matrix=None, interleaver_seed=42):
        # Import here to avoid circular imports
        from turbo_encoder import TurboEncoder as TE
        self._encoder = TE(generator_matrix, interleaver_seed)
    
    def encode(self, info_bits):
        """
        Encode information bits
        
        Parameters:
        ----------
        info_bits : ndarray
            Information bits
            
        Returns:
        -------
        tuple
            (encoded_bits, systematic_bits, parity1_bits, parity2_bits, interleaver_indices)
        """
        return self._encoder.encode(info_bits)
