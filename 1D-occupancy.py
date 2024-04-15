'''
Replication of Figure 2 from "Occupancy Grids: A Stochastic Spatial Representation for Active Robot Perception" by Alberto Elfes
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve1d

def ideal_model(rng, measurement, sigma):
    '''
    Creates ideal occupancy model for a given measurement
    '''
    # Initialise p(OCC) = 0
    occupancy_profile = np.zeros(rng.size)

    # Set p(OCC) = 0.5 after the measurement range
    measurement_index = np.argmin(np.abs(rng-measurement))
    occupancy_profile[measurement_index:] = 0.5

    # Set p(OCC) = 1 within 1 std deviation of measurement range
    occupancy_profile[(rng>(measurement-sigma/2)) & (rng<(measurement+sigma/2))] = 1

    return occupancy_profile

def gaussian_model(x, sigma, mu):
    '''
    Creates gaussian model of measurement accuracy
    '''
    dx = x[1] - x[0]
    norm = 1/(np.sqrt(2*np.pi)*sigma)
    return norm * np.exp(-1/2*((x-mu)**2) / (sigma**2)) * dx

def state_probability(rng, measurement):
    '''
    Creates probability of given state for a noisy model
    '''
    p_sensor = gaussian_model(rng, sigma, measurement)
    profile = ideal_model(rng, measurement-2*sigma/3, sigma)
    p_state = convolve1d(profile, weights=p_sensor)
    
    return p_state

def bayesian_update(prior, measurement):
    '''
    Update bayesian estimator with new measurement
    '''
    p_occ = state_probability(rng, measurement)
    p_free = 1-p_occ
    posterior = p_occ*prior / (p_occ*prior + p_free*(1-prior))
    return posterior

sigma = 0.5
measurement = 2
n_points = 1001
rng = np.linspace(0, 4, n_points)
prior = np.ones(n_points)*0.5

for i in range(5):
    
    posterior = bayesian_update(prior, measurement)
    plt.plot(rng, prior)
    plt.plot(rng, posterior)

    prior = posterior

plt.xlabel('Range')
plt.ylabel('Probability of occupancy')
plt.ylim([0,1])
plt.grid(True)
plt.show()
