#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Turbo Encoder implementation using two identical RSC encoders
"""

import numpy as np
from utils import random_interleaver

class RSCEncoder:
    """
    Rate 1/2 Recursive Systematic Convolutional (RSC) encoder
    """
    def __init__(self, generator_matrix=None):
        """
        Initialize RSC encoder
        
        Parameters:
        ----------
        generator_matrix : tuple
            Generator polynomials in octal (feedback, output)
            Default: (7, 5) in octal, which is (111, 101) in binary
        """
        # Default generator polynomials (7, 5) in octal = (111, 101) in binary
        # 7 is the feedback and 5 is output
        if generator_matrix is None:
            self.feedback = 0b111  # 7 in octal
            self.output = 0b101    # 5 in octal
        else:
            self.feedback = generator_matrix[0]
            self.output = generator_matrix[1]
        
        # Memory elements
        self.memory_length = 2  # Number of memory elements
        self.state = 0          # Initial state
        
    def reset(self):
        """Reset encoder state to zero"""
        self.state = 0
    
    def encode_bit(self, bit):
        """
        Encode a single bit
        
        Parameters:
        ----------
        bit : int
            Input bit (0 or 1)
            
        Returns:
        -------
        systematic_bit : int
            Systematic (input) bit
        parity_bit : int
            Parity bit
        """
        # Calculate feedback
        feedback_bit = bit
        temp_state = self.state
        
        # XOR with feedback connections
        for i in range(self.memory_length):
            if (self.feedback >> i) & 1:
                feedback_bit ^= (temp_state >> i) & 1
        
        # Calculate parity bit using output polynomial
        parity_bit = 0
        for i in range(self.memory_length + 1):  # +1 to include input bit
            if (self.output >> i) & 1:
                if i == 0:
                    parity_bit ^= bit
                else:
                    parity_bit ^= (temp_state >> (i-1)) & 1
        
        # Update state (shift in the feedback bit)
        self.state = ((temp_state << 1) | feedback_bit) & ((1 << self.memory_length) - 1)
        
        return bit, parity_bit
    
    def encode(self, bits):
        """
        Encode a sequence of bits
        
        Parameters:
        ----------
        bits : ndarray
            Input bits to encode
            
        Returns:
        -------
        encoded : ndarray
            Encoded bits (systematic and parity interleaved)
        """
        self.reset()
        n_bits = len(bits)
        encoded = np.zeros(2 * n_bits, dtype=int)
        
        for i in range(n_bits):
            systematic, parity = self.encode_bit(bits[i])
            encoded[2*i] = systematic     # Systematic bit
            encoded[2*i + 1] = parity     # Parity bit
            
        return encoded


class TurboEncoder:
    """
    Turbo Encoder using two identical RSC encoders
    """
    def __init__(self, generator_matrix=None, interleaver_seed=42):
        """
        Initialize Turbo Encoder
        
        Parameters:
        ----------
        generator_matrix : tuple
            Generator polynomials for RSC encoders
        interleaver_seed : int
            Seed for the random interleaver
        """
        self.encoder1 = RSCEncoder(generator_matrix)
        self.encoder2 = RSCEncoder(generator_matrix)
        self.interleaver_seed = interleaver_seed
    
    def encode(self, info_bits):
        """
        Encode information bits using Turbo coding
        
        Parameters:
        ----------
        info_bits : ndarray
            Information bits to encode
            
        Returns:
        -------
        encoded_bits : ndarray
            Full encoded stream
        systematic_bits : ndarray
            Systematic bits only
        parity1_bits : ndarray
            Parity bits from first encoder
        parity2_bits : ndarray
            Parity bits from second encoder
        interleaver_indices : ndarray
            Indices used for interleaving (needed for decoder)
        """
        n_bits = len(info_bits)
        
        # First encoder (RSC1) - encodes original bits
        # Extract systematic and parity bits
        encoded1 = self.encoder1.encode(info_bits)
        systematic_bits = encoded1[0::2]  # Even indices
        parity1_bits = encoded1[1::2]     # Odd indices
        
        # Interleave input bits
        interleaved_bits, interleaver_indices = random_interleaver(info_bits, 
                                                                   self.interleaver_seed)
        
        # Second encoder (RSC2) - encodes interleaved bits
        # Only need parity bits from second encoder
        encoded2 = self.encoder2.encode(interleaved_bits)
        parity2_bits = encoded2[1::2]     # Odd indices
        
        # Combine all bits (systematic, parity1, parity2)
        # Format: [s1, p1_1, p2_1, s2, p1_2, p2_2, ...]
        encoded_bits = np.zeros(3 * n_bits, dtype=int)
        encoded_bits[0::3] = systematic_bits  # Systematic bits
        encoded_bits[1::3] = parity1_bits     # Parity from encoder 1
        encoded_bits[2::3] = parity2_bits     # Parity from encoder 2
        
        return encoded_bits, systematic_bits, parity1_bits, parity2_bits, interleaver_indices