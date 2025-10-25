import numpy as np


def calc_l1(ideal_amp, actual_amps):
    '''
    Calculate L1 distance: Sum of absolute value - Manhattan distance (edit distance)
    '''
    return np.sum(np.abs(ideal_amp - np.array(actual_amps)))


def calc_l2(ideal_amp, actual_amps):
    '''
    Calculate L2 distance: Sum of square - Square Euclidian distance
    '''
    return np.sqrt(np.sum((ideal_amp - np.array(actual_amps))**2))
