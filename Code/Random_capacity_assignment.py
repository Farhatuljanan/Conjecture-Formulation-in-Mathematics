# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 19:22:58 2024

@author: fjanan
"""

import random

def generate_random_sum_list(target_sum, num_elements):
    # Generate random values
    random_values = [random.random() for _ in range(num_elements)]
    # Scale the random values to sum up to the target sum
    scale_factor = target_sum / sum(random_values)
    scaled_values = [round(value * scale_factor, 2) for value in random_values]

    # Adjust any rounding error to ensure exact sum
    difference = target_sum - sum(scaled_values)
    scaled_values[0] += difference  # Adjust the first element to account for rounding error
    
    return scaled_values

# Example usage
target_sum = 10
num_elements = 16
random_list = generate_random_sum_list(target_sum, num_elements)
