#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Turbo Decoder base class and implementations for MAP (BCJR), Log-MAP, and Max-Log-MAP.
Implements modular, reusable decoders with operation counting and consistent interfaces.
"""

import numpy as np
from abc import ABC, abstractmethod

# Constants
NEG_INF = -1e10  # Using a large negative number instead of -np.inf for better numerical stability
MIN_PROB = 1e-30  # Minimum probability value for numerical stability

class TurboDecoderBase(ABC):
    """
    Abstract base class for all turbo decoder implementations.
    Provides common functionality and enforces consistent interface.
    """
    def __init__(self, generator_matrix=None):
        """
        Initialize decoder with generator polynomials.
        
        Parameters:
        ----------
        generator_matrix : tuple, optional
            Generator polynomials in octal (feedback, output)
            Default is (7, 5) = (111, 101) in binary
        """
        # Default generator polynomials (7, 5) in octal = (111, 101) in binary
        if generator_matrix is None:
            self.feedback = 0b111  # 7 in octal
            self.output = 0b101    # 5 in octal
        else:
            self.feedback = generator_matrix[0]
            self.output = generator_matrix[1]
            
        self.memory_length = 2  # Number of memory elements
        self.num_states = 2 ** self.memory_length  # Number of possible states
        
        # Pre-compute state transitions for efficiency
        self.next_state = np.zeros((self.num_states, 2), dtype=int)
        self.output_parity = np.zeros((self.num_states, 2), dtype=int)
        self._precompute_transitions()
        
        # Operation counters
        self.reset_counters()
    
    def reset_counters(self):
        """Reset operation counters"""
        self.op_counter = {
            'additions': 0,
            'multiplications': 0,
            'comparisons': 0,
            'exponentiations': 0
        }
    
    def _precompute_transitions(self):
        """Pre-compute state transitions and outputs for each state and input"""
        for state in range(self.num_states):
            for input_bit in range(2):
                # Calculate feedback bit
                feedback_bit = input_bit
                
                # XOR with feedback connections
                for i in range(self.memory_length):
                    if (self.feedback >> i) & 1:
                        feedback_bit ^= (state >> i) & 1
                
                # Calculate next state
                next_state = ((state << 1) | feedback_bit) & ((1 << self.memory_length) - 1)
                self.next_state[state, input_bit] = next_state
                
                # Calculate parity bit
                parity_bit = 0
                for i in range(self.memory_length + 1):
                    if (self.output >> i) & 1:
                        if i == 0:
                            parity_bit ^= input_bit
                        else:
                            parity_bit ^= (state >> (i-1)) & 1
                
                self.output_parity[state, input_bit] = parity_bit
    
    @abstractmethod
    def _compute_branch_metrics(self, systematic_llr, parity_llr):
        """Compute branch metrics for all state transitions"""
        pass
    
    @abstractmethod
    def _forward_recursion(self, branch_metrics, num_symbols):
        """Compute forward metrics"""
        pass
    
    @abstractmethod
    def _backward_recursion(self, branch_metrics, num_symbols):
        """Compute backward metrics"""
        pass
    
    @abstractmethod
    def _compute_llr(self, alpha, beta, branch_metrics, num_symbols):
        """Compute LLRs from forward and backward metrics"""
        pass
    
    def decode(self, received_llrs, length, a_priori_llrs=None):
        """
        Decode a received sequence.
        
        Parameters:
        ----------
        received_llrs : ndarray
            Received LLRs (systematic and parity interleaved)
        length : int
            Length of the information sequence
        a_priori_llrs : ndarray, optional
            A priori LLRs from the other decoder
            
        Returns:
        -------
        extrinsic_llrs : ndarray
            Extrinsic information for each bit
        """
        self.reset_counters()
        
        # Extract systematic and parity LLRs
        num_symbols = length
        systematic_llrs = received_llrs[0::2]  # Even indices
        parity_llrs = received_llrs[1::2]      # Odd indices
        
        # Default a priori LLRs to zeros if not provided
        if a_priori_llrs is None:
            a_priori_llrs = np.zeros(num_symbols)
        
        # Compute branch metrics for each symbol
        branch_metrics = np.zeros((num_symbols, self.num_states, 2))
        for k in range(num_symbols):
            # Include a priori information in the branch metrics
            total_systematic_llr = systematic_llrs[k] + a_priori_llrs[k]
            branch_metrics[k] = self._compute_branch_metrics(total_systematic_llr, parity_llrs[k])
            self.op_counter['additions'] += 1  # For adding a_priori_llrs
        
        # Forward recursion (alpha)
        alpha = self._forward_recursion(branch_metrics, num_symbols)
        
        # Backward recursion (beta)
        beta = self._backward_recursion(branch_metrics, num_symbols)
        
        # Compute LLRs
        llrs = self._compute_llr(alpha, beta, branch_metrics, num_symbols)
        
        # Compute extrinsic information (subtract a priori and systematic)
        extrinsic_llrs = llrs - systematic_llrs - a_priori_llrs
        self.op_counter['additions'] += 2 * num_symbols  # For the subtraction operations
        
        return extrinsic_llrs


class MAPDecoder(TurboDecoderBase):
    """
    MAP (BCJR) Decoder Implementation using exact probabilities
    """
    def _compute_branch_metrics(self, systematic_llr, parity_llr):
        """
        Compute exact branch probabilities for all state transitions
        """
        # Convert LLRs to probabilities
        # P(bit=0) = exp(LLR/2) / (exp(LLR/2) + exp(-LLR/2))
        
        # Limit LLR values for numerical stability
        systematic_llr = np.clip(systematic_llr, -15.0, 15.0)
        parity_llr = np.clip(parity_llr, -15.0, 15.0)
        
        # Compute probabilities
        p_sys_0 = np.exp(systematic_llr / 2) / (np.exp(systematic_llr / 2) + np.exp(-systematic_llr / 2))
        p_sys_1 = 1 - p_sys_0
        
        p_par_0 = np.exp(parity_llr / 2) / (np.exp(parity_llr / 2) + np.exp(-parity_llr / 2))
        p_par_1 = 1 - p_par_0
        
        self.op_counter['exponentiations'] += 4  # For the exp() calls
        self.op_counter['additions'] += 2       # For the denominators
        self.op_counter['multiplications'] += 4  # For the divisions
        
        # Compute branch probabilities
        gamma = np.zeros((self.num_states, 2))
        
        for state in range(self.num_states):
            for input_bit in range(2):
                parity_bit = self.output_parity[state, input_bit]
                
                # Probability calculation:
                # P(y|s->s',x) = P(sys|x) * P(par|par_bit)
                if input_bit == 0:
                    p_sys = p_sys_0
                else:
                    p_sys = p_sys_1
                    
                if parity_bit == 0:
                    p_par = p_par_0
                else:
                    p_par = p_par_1
                
                gamma[state, input_bit] = max(p_sys * p_par, MIN_PROB)  # Prevent zero probabilities
                self.op_counter['multiplications'] += 1  # For p_sys * p_par
                
        return gamma
    
    def _forward_recursion(self, branch_probs, num_symbols):
        """
        Compute exact forward probabilities
        """
        alpha = np.zeros((num_symbols + 1, self.num_states))
        
        # Initialize alpha (start in state 0)
        alpha[0, 0] = 1.0
        
        # Forward recursion
        for k in range(num_symbols):
            for next_state in range(self.num_states):
                # Find all state transitions leading to next_state
                for current_state in range(self.num_states):
                    for input_bit in range(2):
                        if self.next_state[current_state, input_bit] == next_state:
                            alpha[k+1, next_state] += alpha[k, current_state] * branch_probs[k, current_state, input_bit]
                            self.op_counter['additions'] += 1
                            self.op_counter['multiplications'] += 1
                
            # Normalize to prevent underflow
            alpha_sum = np.sum(alpha[k+1, :])
            if alpha_sum > 0:
                alpha[k+1, :] /= alpha_sum
                self.op_counter['additions'] += self.num_states  # For the sum
                self.op_counter['multiplications'] += self.num_states  # For the division
        
        return alpha
    
    def _backward_recursion(self, branch_probs, num_symbols):
        """
        Compute exact backward probabilities
        """
        beta = np.zeros((num_symbols + 1, self.num_states))
        
        # Initialize beta (end in any state with equal probability)
        beta[num_symbols, :] = 1.0 / self.num_states
        
        # Backward recursion
        for k in range(num_symbols - 1, -1, -1):
            for current_state in range(self.num_states):
                for input_bit in range(2):
                    next_state = self.next_state[current_state, input_bit]
                    beta[k, current_state] += branch_probs[k, current_state, input_bit] * beta[k+1, next_state]
                    self.op_counter['additions'] += 1
                    self.op_counter['multiplications'] += 1
            
            # Normalize to prevent underflow
            beta_sum = np.sum(beta[k, :])
            if beta_sum > 0:
                beta[k, :] /= beta_sum
                self.op_counter['additions'] += self.num_states  # For the sum
                self.op_counter['multiplications'] += self.num_states  # For the division
        
        return beta
    
    def _compute_llr(self, alpha, beta, branch_probs, num_symbols):
        """
        Compute exact LLRs from probabilities
        """
        llr = np.zeros(num_symbols)
        
        for k in range(num_symbols):
            prob_0 = 0.0
            prob_1 = 0.0
            
            # Sum up probabilities for input bit = 0 and input bit = 1
            for current_state in range(self.num_states):
                for input_bit in range(2):
                    next_state = self.next_state[current_state, input_bit]
                    prob = alpha[k, current_state] * branch_probs[k, current_state, input_bit] * beta[k+1, next_state]
                    
                    if input_bit == 0:
                        prob_0 += prob
                    else:
                        prob_1 += prob
                    
                    self.op_counter['multiplications'] += 2  # For the multiplications in prob calculation
                    self.op_counter['additions'] += 1  # For the addition to prob_0/prob_1
            
            # Ensure non-zero probabilities for numerical stability
            prob_0 = max(prob_0, MIN_PROB)
            prob_1 = max(prob_1, MIN_PROB)
            
            # LLR = log(P(bit=0)/P(bit=1))
            llr[k] = np.log(prob_0 / prob_1)
            self.op_counter['multiplications'] += 1  # For the division
            self.op_counter['exponentiations'] += 1  # For the log
        
        return llr


class LogMAPDecoder(TurboDecoderBase):
    """
    Log-MAP Decoder Implementation using log domain computations
    """
    def _log_sum_exp(self, x, y):
        """
        Compute log(exp(x) + exp(y)) using the Jacobian logarithm:
        log(exp(x) + exp(y)) = max(x,y) + log(1 + exp(-|x-y|))
        """
        self.op_counter['comparisons'] += 1
        
        # Handle negative infinities
        if x <= NEG_INF or y <= NEG_INF:
            return max(x, y)
        
        max_val = max(x, y)
        min_val = min(x, y)
        
        # Calculate correction term
        abs_diff = max_val - min_val  # Will always be positive since max_val >= min_val
        
        # Early termination for large differences (for efficiency)
        if abs_diff > 15.0:
            return max_val
        
        # Compute the correction term using log1p (more stable than log(1+exp()))
        # log1p(x) = log(1+x)
        correction = np.log1p(np.exp(-abs_diff))
        self.op_counter['exponentiations'] += 1
        
        return max_val + correction
    
    def _compute_branch_metrics(self, systematic_llr, parity_llr):
        """
        Compute branch metrics in log domain
        """
        gamma = np.zeros((self.num_states, 2))
        
        # Limit LLR values for numerical stability
        systematic_llr = np.clip(systematic_llr, -15.0, 15.0)
        parity_llr = np.clip(parity_llr, -15.0, 15.0)
        
        for state in range(self.num_states):
            for input_bit in range(2):
                parity_bit = self.output_parity[state, input_bit]
                
                # Branch metric calculation in log domain
                branch_metric = 0.0
                
                # If input_bit = 0, add LLR/2; if input_bit = 1, subtract LLR/2
                if input_bit == 0:
                    branch_metric += systematic_llr / 2.0
                else:
                    branch_metric -= systematic_llr / 2.0
                    
                # Similarly for parity bit
                if parity_bit == 0:
                    branch_metric += parity_llr / 2.0
                else:
                    branch_metric -= parity_llr / 2.0
                
                gamma[state, input_bit] = branch_metric
                self.op_counter['additions'] += 2  # For each branch metric calculation
                
        return gamma
    
    def _forward_recursion(self, branch_metrics, num_symbols):
        """
        Compute forward metrics in log domain
        """
        alpha = np.full((num_symbols + 1, self.num_states), NEG_INF)
        
        # Initialize alpha (start in state 0)
        alpha[0, 0] = 0.0
        
        # Forward recursion
        for k in range(num_symbols):
            for next_state in range(self.num_states):
                temp_val = NEG_INF
                
                # Find all state transitions leading to next_state
                for current_state in range(self.num_states):
                    for input_bit in range(2):
                        if self.next_state[current_state, input_bit] == next_state:
                            val = alpha[k, current_state] + branch_metrics[k, current_state, input_bit]
                            temp_val = self._log_sum_exp(temp_val, val)
                            self.op_counter['additions'] += 1  # For addition in val calculation
                
                alpha[k+1, next_state] = temp_val
            
            # Normalize to prevent numerical issues
            if k % 10 == 0:  # Only normalize every few steps to save computation
                max_alpha = np.max(alpha[k+1,:])
                if max_alpha > 100:  # Only normalize if values are getting large
                    alpha[k+1,:] -= max_alpha
        
        return alpha
    
    def _backward_recursion(self, branch_metrics, num_symbols):
        """
        Compute backward metrics in log domain
        """
        beta = np.full((num_symbols + 1, self.num_states), NEG_INF)
        
        # Initialize beta (end in any state with equal probability)
        beta[num_symbols, :] = 0.0
        
        # Backward recursion
        for k in range(num_symbols - 1, -1, -1):
            for current_state in range(self.num_states):
                temp_val = NEG_INF
                
                # Find all transitions from current_state
                for input_bit in range(2):
                    next_state = self.next_state[current_state, input_bit]
                    val = beta[k+1, next_state] + branch_metrics[k, current_state, input_bit]
                    temp_val = self._log_sum_exp(temp_val, val)
                    self.op_counter['additions'] += 1  # For addition in val calculation
                
                beta[k, current_state] = temp_val
            
            # Normalize to prevent numerical issues
            if k % 10 == 0:  # Only normalize every few steps to save computation
                max_beta = np.max(beta[k,:])
                if max_beta > 100:  # Only normalize if values are getting large
                    beta[k,:] -= max_beta
        
        return beta
    
    def _compute_llr(self, alpha, beta, branch_metrics, num_symbols):
        """
        Compute LLRs in log domain
        """
        llr = np.zeros(num_symbols)
        
        for k in range(num_symbols):
            max_0 = NEG_INF
            max_1 = NEG_INF
            
            # Sum up probabilities for input bit = 0 and input bit = 1
            for current_state in range(self.num_states):
                for input_bit in range(2):
                    next_state = self.next_state[current_state, input_bit]
                    
                    val = alpha[k, current_state] + branch_metrics[k, current_state, input_bit] + beta[k+1, next_state]
                    
                    if input_bit == 0:
                        max_0 = self._log_sum_exp(max_0, val)
                    else:
                        max_1 = self._log_sum_exp(max_1, val)
                    
                    self.op_counter['additions'] += 2  # For the additions in val calculation
            
            # LLR = log(P(bit=0)) - log(P(bit=1))
            llr[k] = max_0 - max_1
            self.op_counter['additions'] += 1  # For the final subtraction
        
        return llr


class MaxLogMAPDecoder(TurboDecoderBase):
    """
    Max-Log-MAP Decoder Implementation using max approximation
    """
    def _max_operation(self, x, y):
        """
        Simple max operation for Max-Log-MAP:
        max*(x,y) ≈ max(x,y)
        """
        self.op_counter['comparisons'] += 1
        return max(x, y)
    
    def _compute_branch_metrics(self, systematic_llr, parity_llr):
        """
        Compute branch metrics using max approximation
        """
        gamma = np.zeros((self.num_states, 2))
        
        # Limit LLR values for numerical stability
        systematic_llr = np.clip(systematic_llr, -15.0, 15.0)
        parity_llr = np.clip(parity_llr, -15.0, 15.0)
        
        for state in range(self.num_states):
            for input_bit in range(2):
                parity_bit = self.output_parity[state, input_bit]
                
                # Branch metric calculation
                branch_metric = 0.0
                
                # If input_bit = 0, add LLR/2; if input_bit = 1, subtract LLR/2
                if input_bit == 0:
                    branch_metric += systematic_llr / 2.0
                else:
                    branch_metric -= systematic_llr / 2.0
                    
                # Similarly for parity bit
                if parity_bit == 0:
                    branch_metric += parity_llr / 2.0
                else:
                    branch_metric -= parity_llr / 2.0
                
                gamma[state, input_bit] = branch_metric
                self.op_counter['additions'] += 2  # For each branch metric calculation
                
        return gamma
    
    def _forward_recursion(self, branch_metrics, num_symbols):
        """
        Compute forward metrics using max approximation
        """
        alpha = np.full((num_symbols + 1, self.num_states), NEG_INF)
        
        # Initialize alpha (start in state 0)
        alpha[0, 0] = 0.0
        
        # Forward recursion
        for k in range(num_symbols):
            for next_state in range(self.num_states):
                max_val = NEG_INF
                
                # Find all state transitions leading to next_state
                for current_state in range(self.num_states):
                    for input_bit in range(2):
                        if self.next_state[current_state, input_bit] == next_state:
                            val = alpha[k, current_state] + branch_metrics[k, current_state, input_bit]
                            max_val = self._max_operation(max_val, val)
                            self.op_counter['additions'] += 1  # For addition in val calculation
                
                alpha[k+1, next_state] = max_val
            
            # Normalize to prevent numerical issues
            if k % 10 == 0:  # Only normalize every few steps to save computation
                max_alpha = np.max(alpha[k+1,:])
                if max_alpha > 100:  # Only normalize if values are getting large
                    alpha[k+1,:] -= max_alpha
        
        return alpha
    
    def _backward_recursion(self, branch_metrics, num_symbols):
        """
        Compute backward metrics using max approximation
        """
        beta = np.full((num_symbols + 1, self.num_states), NEG_INF)
        
        # Initialize beta (end in any state with equal probability)
        beta[num_symbols, :] = 0.0
        
        # Backward recursion
        for k in range(num_symbols - 1, -1, -1):
            for current_state in range(self.num_states):
                max_val = NEG_INF
                
                # Find all transitions from current_state
                for input_bit in range(2):
                    next_state = self.next_state[current_state, input_bit]
                    val = beta[k+1, next_state] + branch_metrics[k, current_state, input_bit]
                    max_val = self._max_operation(max_val, val)
                    self.op_counter['additions'] += 1  # For addition in val calculation
                
                beta[k, current_state] = max_val
            
            # Normalize to prevent numerical issues
            if k % 10 == 0:  # Only normalize every few steps to save computation
                max_beta = np.max(beta[k,:])
                if max_beta > 100:  # Only normalize if values are getting large
                    beta[k,:] -= max_beta
        
        return beta
    
    def _compute_llr(self, alpha, beta, branch_metrics, num_symbols):
        """
        Compute LLRs using max approximation
        """
        llr = np.zeros(num_symbols)
        
        for k in range(num_symbols):
            max_0 = NEG_INF
            max_1 = NEG_INF
            
            # Find maximum probabilities for input bit = 0 and input bit = 1
            for current_state in range(self.num_states):
                for input_bit in range(2):
                    next_state = self.next_state[current_state, input_bit]
                    
                    val = alpha[k, current_state] + branch_metrics[k, current_state, input_bit] + beta[k+1, next_state]
                    
                    if input_bit == 0:
                        max_0 = self._max_operation(max_0, val)
                    else:
                        max_1 = self._max_operation(max_1, val)
                    
                    self.op_counter['additions'] += 2  # For the additions in val calculation
            
            # LLR = log(P(bit=0)/P(bit=1)) ≈ max(paths with bit=0) - max(paths with bit=1)
            llr[k] = max_0 - max_1
            self.op_counter['additions'] += 1  # For the final subtraction
        
        return llr
