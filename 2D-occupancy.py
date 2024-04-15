'''
Replication of "Using occupancy grids for mobile robot perception and navigation" by Alberto Elfes
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from scipy.ndimage import map_coordinates as mp

def ideal_model(rng, theta, measurement_r, measurement_t, sigma_r, sigma_t):
    '''
    Creates ideal occupancy model for a given measurement
    '''
    # Initialise p(OCC) = 0.5
    occupancy_profile = np.ones((rng.size, theta.size)) * 0.5

    # Set p(OCC) = 0 befor the measurement range
    theta_indices = (theta>(measurement_t-sigma_t)) & (theta<(measurement_t+sigma_t))
    occupancy_profile[(rng<measurement_r)[:,None] & theta_indices] = 0

    # Set p(OCC) = 1 within 1 std deviation of measurement range
    rng_indices = (rng>(measurement_r-sigma_r)) & (rng<(measurement_r+sigma_r))
    occupancy_profile[rng_indices[:,None] & theta_indices] = 1

    return occupancy_profile

def gaussian(x, y, mu_x, mu_y, sigma_x, sigma_y):
    norm = 1/(2*np.pi*sigma_x*sigma_y)
    return norm * np.exp(-1/2*(((x-mu_x)**2)/(sigma_x**2) + ((y-mu_y)**2)/(sigma_y**2)))

def gaussian_model(x, y, mu_x, mu_y, sigma_x, sigma_y):
    '''
    Creates gaussian model of measurement accuracy
    '''
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    mesh = np.zeros((x.size, y.size))
    xi, yi = np.meshgrid(np.arange(x.size), np.arange(y.size), indexing='ij')
    xu, yv = np.meshgrid(rng, theta, indexing='ij')

    norm = 1/(2*np.pi*sigma_x*sigma_y)
    values = norm * np.exp(-1/2*(((xu-mu_x)**2)/(sigma_x**2) + ((yv-mu_y)**2)/(sigma_y**2)))

    mesh[xi, yi] = values*dx*dy

    return mesh

def state_probability(rng, theta, measurement_r, measurement_t, sigma_r, sigma_t):
    p_sensor = gaussian_model(rng, theta, measurement_r, measurement_t, sigma_r, sigma_t)
    profile = ideal_model(rng, theta, measurement_r-2*sigma_r/3, measurement_t, sigma_r, sigma_t)
    p_state = convolve(profile, p_sensor)
    
    return p_state

def bayesian_update(prior, measurement_r):
    '''
    Update bayesian estimator with new measurement
    '''
    p_polar = state_probability(rng, theta, measurement_r, measurement_t, sigma_r, sigma_t)
    p_occ = polar_to_cart(x, y, p_polar, rng, theta, order=3)
    p_free = 1-p_occ
    posterior = p_occ*prior / (p_occ*prior + p_free*(1-prior))
    return posterior

def create_gridmap(rng, theta, dxy):
    max_x = np.max(rng)
    x = np.arange(0, max_x, dxy)
    max_y = max_x*np.sin(np.max(theta))
    y = np.arange(-max_y, max_y, dxy)

    return x, y

def polar_to_cart(x, y, polar_data, rng, theta, order=3):

    rng_start = rng[0]
    theta_start = theta[0]
    rng_step = rng[1] - rng[0]
    theta_step = theta[1] - theta[0]

    X, Y = np.meshgrid(x, y)

    Tc = np.degrees(np.arctan2(Y, X)).ravel()
    Rc = (np.sqrt(X**2 + Y**2)).ravel()

    Ac = (Tc - theta_start) / theta_step
    Sc = (Rc - rng_start) / rng_step

    coords = np.vstack((Sc, Ac))

    cart_data = mp(polar_data, coords, order=order, mode='constant', cval=0.5)

    # The data is reshaped and returned
    return(cart_data.reshape(len(y), len(x)).T)


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

dxy = 0.1
sigma_r = 0.1
sigma_t = 10
measurement_r = 2
measurement_t = 0
n_points = 101
rng = np.linspace(0, 4, n_points)
theta = np.linspace(-45, 45, n_points)

x, y = create_gridmap(rng, theta, dxy)
xi, yi = np.meshgrid(np.arange(x.size), np.arange(y.size), indexing='ij')
xu, yv = np.meshgrid(x, y, indexing='ij')
prior = np.ones((x.size, y.size)) * 0.5

for i in range(5):
    
    posterior = bayesian_update(prior, measurement_r)

    prior = posterior

ax.plot_surface(xu, yv, posterior[xi,yi])
plt.figure()
plt.imshow(posterior)
plt.show()
