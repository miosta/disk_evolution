#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division
import numpy as np
import math
from numpy import sqrt, exp

try: input = raw_input
except NameError: pass

sim_time = float(input('simulation time [s]:'))
part_size = float(input('particle size [m]:'))
exponent = float(input('psi:'))
cell_count = int(input('number of cells:'))
#initial_gas = input('initial gas density profile (options: flare_density, pringle_density, flare_density, perturbed_density):')
#initial_dust = input('initial gas density profile (options: flare_density, pringle_density, flare_density, perturbed_density):')

GRAVITY_CONSTANT   = 6.674e-11          # [m³/kg/s²]
MOLECULAR_WEIGHT   = 2.3                # [proton mass]
BOLTZMANN_CONSTANT = 1.3806485279e-23   # [J/k]
PROTON_MASS        = 1.67262189821e-27  # [kg]
AU                 = 1.49597870700e11   # [m]
MASS_SUN           = 2e30
T_0                = 200                # [K]


#----------------------------------------
# numerics
#----------------------------------------

def D(y, x):
    return (y[1:]-y[:-1]) / (x[1:]-x[:-1])

def A(x):
    return (x[1:] + x[:-1])/2.

def euler_step(state, deriv, dt):
    return state + deriv(state) * dt

def rk4(state, deriv, dt):
    k1 = deriv(state)
    k2 = deriv(state + k1 * (dt/2.))
    k3 = deriv(state + k2 * (dt/2.))
    k4 = deriv(state + k3 * dt)
    return state + (k1 + k2*2. + k3*2. + k4) * (dt/6.)

#----------------------------------------
# variable functions
#----------------------------------------

def viscosity (radius, speed_of_sound_squard, a) :
    return a * speed_of_sound_squard / sun_kepler_vel(radius) # radius in [m]

def speed_of_sound_squard (temperature) :
    return (BOLTZMANN_CONSTANT / PROTON_MASS / MOLECULAR_WEIGHT) * temperature # [m²/s²]

def temperature (radius) :
    return T_0 / sqrt(radius / AU) # radius in [m]

def mradius (au_radius) :
    return au_radius * AU #transformes AU in meter

def alpha_viscosity(radius, alpha) :
    return viscosity(radius, speed_of_sound_squard(temperature(radius)), alpha)

def stokes (gas_density,
               particle_radius,
               particle_density):
    return particle_radius*particle_density/gas_density/2. * math.pi #radius [m], density [kg/m³]

def kepler_vel (mass, radius) :
    return sqrt(GRAVITY_CONSTANT * mass / radius**3) #mass [kg], radius [m]

def sun_kepler_vel (radius) :
    return kepler_vel(MASS_SUN, radius)

def alpha_factor (alpha, dust) :
    return 1./alpha /dust

#----------------------------------------
# Initial functions
#----------------------------------------

def flare_density(radius, density_maximum):
    return density_maximum / radius

def step_density(radius, density_maximum):
    boundary = 70.
    return np.where(radius < boundary,density_maximum, 1e-30)

def pringle_density(radius, density_maximum):
    outer_boundary = 10.
    return density_maximum /3./math.pi/alpha_viscosity(outer_boundary,1e-3)/(radius/outer_boundary)**(1.) * exp(-(radius/outer_boundary)**(1.))


def constant_value (radius, density_maximum):
    return density_maximum + 0.*radius


def constant_density (radius, total_density):
    return total_density * (2.*math.pi * sqrt(speed_of_sound_squard(temperature(radius))))/sun_kepler_vel(radius) * 1e-14 

def gauss_bump (radius):
    inner_boundary = 5.
    outer_boundary = 6.
    sigma2 = ((outer_boundary - inner_boundary)/5.)**2
    return ( 1. -  0.5 * (1./sqrt(2.* math.pi * sigma2)* exp(-(radius -(outer_boundary + inner_boundary)/2)**2/2./sigma2)))
    
def sine_perturbation (radius):
    amplitude = .75
    wavelength = .1 #in [AU]
    return 1. + amplitude* (np.sin(np.log(radius/wavelength)*2.*math.pi))

def bump_density (radius, mean_density):
    return mean_density / radius / gauss_bump(radius)
    
def perturbed_density (radius, mean_density):
    return mean_density / radius * sine_perturbation(radius)
    
def perturbed_alpha (radius, mean_alpha):
    return sine_perturbation(radius) * mean_alpha

def ice_bump(radius, mean_alpha):
    return gauss_bump(radius) * mean_alpha
#----------------------------------------
#Alpha Evolution
#----------------------------------------

def dependent_alpha(radius, alpha_maximum, dust_maximum, grid):
    return constant_value(radius, alpha_maximum) *(flare_density(radius, dust_maximum)/ grid.dust)**(.75)

def pinilla_alpha(density):
    alpha_dead = 1e-4
    alpha_active = 1e-2
    eps = .2
    x = 150
    return (1.- np.tanh((density-x)*eps))*alpha_active/2.+ alpha_dead

#----------------------------------------
# Fluxes and derivaties
#----------------------------------------

# Diffusionsequation for the Flux between neighbouring cells
#     Density(t+1)=density(t)+(dt)/V*(Flux(x+1/2)Fläche(x+1/2)-F(x-1/2)Fläche(x-1/2))
#     Flux(x+1/2)= -Diffusionskoofizient(density(x+1)-density(x))/((x+1) - x)
#     Fläche=dydz
#     V=dydz(x+1/2 - x-1/2)

def flux (diffusioncoeficient, grid): #simple advection
    return -diffusioncoeficient * D(grid.density, grid.radius)


# Calculations for a viscous flux between neighbouring cells
#     Viscous cylindric disk equation:
#     Flux(x+1/2)= -3/r(x+1/2) * (density(x)*viscousity(x)*r(x)^1/2 - density(x+1)*viscousity(x+1)*r(x+1)^1/2) / (r(x)-r(x+1))
#     Density(t+1) = density(t) + dt / r(x) (flux(x+1/2)*r(x+1/2) - flux(x-1/2)*r(x-1/2)) / (r(x+1/2) - r(x-1/2))

def viscous_velocity (viscosity, grid):
    vel = -3.0 / sqrt(A(grid.radius)) / A(grid.density) \
        * D(grid.density * viscosity * sqrt(grid.radius),
            grid.radius)
    vel_max = np.sqrt(speed_of_sound_squard(temperature(grid.radius[:-1]))*constant_value(grid.radius[:-1], 1e-3))
    cut_vel = np.where(vel_max > np.abs(vel), vel, np.sign(vel)*vel_max)
    return cut_vel

def viscous_flux (viscosity, grid):
    vel = viscous_velocity(viscosity, grid)
    gas = np.where(vel > 0, grid.density[:-1], grid.density[1:])
    return gas * vel

# from pinilla 2016
def dust_velocity (viscosity, grid):
    mid = A(grid.radius)
    mid_gas = A(grid.density)
    stoke = stokes(mid_gas, part_size, 1600.)
    vel = (1./(stoke**2+1.) *
    (viscous_flux(viscosity, grid)/mid_gas +
        stoke/mid_gas/sun_kepler_vel(mid)**2 *
        sqrt(speed_of_sound_squard(temperature(mid))) *
            D(grid.density * sun_kepler_vel(grid.radius) * sqrt(speed_of_sound_squard(temperature(grid.radius))), grid.radius)
    ))    
    vel_max = np.sqrt(speed_of_sound_squard(temperature(grid.radius[:-1]))*constant_value(grid.radius[:-1], 1e-3))
    cut_vel = np.where(vel_max > np.abs(vel), vel, np.sign(vel)*vel_max)
    return vel


def dust_flux(viscosity, grid):
    mid_gas = A(grid.density)
    stoke = stokes(mid_gas, part_size, 1600.)
    vel = dust_velocity(viscosity, grid)
    dust =  np.where(vel > 0, grid.dust[:-1], grid.dust[1:])
    return (dust * vel -  
        mid_gas * A(viscosity)/( stoke**2+1.) *
        D(grid.dust/grid.density, grid.radius))


# Forms the derivative of the surface density of one cell to scale it with a timestep and use for the iternation
def deriv_grid(grid):
    
    #outer_grid = Grid( np.hstack((grid.radius, grid.radius[-1]+1.)),
    #    np.hstack((grid.density, 1e-10)),
    #    np.hstack((grid.dust, grid.dust[-1]- abs(grid.dust[1]-grid.dust[2]) )),      
    #    np.hstack((grid.alpha, grid.alpha[-1]))
    #    )
    diffusion = alpha_viscosity
    disk_flux = viscous_flux

    diff         = diffusion(grid.radius, grid.alpha)
    density_flux = viscous_flux(diff, grid)
    dusty_flux   = dust_flux(diff, grid)

    r = A(grid.radius)
    densy_deriv = -2. * D(density_flux*r, r**2)
    dusty_deriv = -2. * D(  dusty_flux*r, r**2)

    densy_deriv = np.hstack((densy_deriv[0], densy_deriv, 1e-30))
    dusty_deriv = np.hstack((dusty_deriv[1], dusty_deriv, 1e-30))
    alpha_deriv = 0#1e-30#- dusty_deriv/grid.dust * grid.alpha

    return Grid(0., densy_deriv, dusty_deriv, alpha_deriv)

#----------------------------------------
# Main time evolution operation
#----------------------------------------

def write_grid_to_file(file_name, grid):
    inner = grid[:-1]
    np.savetxt(file_name, np.c_[inner.radius / AU, inner.density, inner.dust, inner.alpha,
        dust_velocity(alpha_viscosity(grid.radius, grid.alpha), grid), stokes(grid.density[:-1],part_size,1600.), 
	viscous_velocity(alpha_viscosity(grid.radius, grid.alpha), grid)], delimiter="\t")


class Grid(object):
    def __init__(self, radius, density, dust, alpha):
        self.radius  = radius
        self.density = density
        self.dust    = dust
        self.alpha   = alpha

    def __getitem__(self, slice):
        return Grid(self.radius[slice],
                    self.density[slice],
                    self.dust[slice],
                    self.alpha[slice])

    def __add__(self, other):
        return Grid(self.radius + other.radius,
                    self.density + other.density,
                    self.dust + other.dust,
                    self.alpha + other.alpha)

    def __mul__(self, scalar):
      	return Grid(self.radius * scalar,
                   self.density * scalar,
                   self.dust * scalar,
                   self.alpha * scalar)


def default_time_step():
    return 1e5

def advect_time_step(grid):
    courant_number = 0.01
    return np.min(np.abs(courant_number * (grid.radius[:-1]-grid.radius[1:]) /   
            dust_velocity(alpha_viscosity(grid.radius, grid.alpha), grid)))


class options:
    number_of_cells      = cell_count
    inner_boundary       = -1   #10^x
    outer_boundary       = +3   #10^x
    gas_density_maximum  = 1e3
    dust_density_maximum = 1e1
    alpha_maximum        = 1e-3


def main():
    radius  = np.logspace(options.inner_boundary,
                          options.outer_boundary,
                          options.number_of_cells)
    density = bump_density(radius, options.gas_density_maximum)
    dust    = flare_density(radius, options.dust_density_maximum)
    alpha   = constant_value(radius, options.alpha_maximum)
    grid    = Grid(radius * AU, density, dust, alpha)

    cur_step = 0
    cur_time, total_time  = 0, sim_time
    cur_plot, total_plots = 0, 50
    dt = 1.
    #x = 1/(pringle_density(radius, options.dust_density_maximum)*gauss_bump(radius))/options.alpha_maximum
    while cur_time < total_time:
        if cur_time >= cur_plot/total_plots * total_time:
            file_name = "output/cells_{}_{}".format(exponent,cur_plot)
            write_grid_to_file(file_name, grid)
            cur_plot += 1
            print(cur_plot, dt)

        old_dust = grid.dust
        dt        = advect_time_step(grid)
        grid      = rk4(grid, deriv_grid, dt)
        grid.alpha = abs(constant_value(radius, options.alpha_maximum) *(flare_density(radius, options.dust_density_maximum)/ grid.dust)**(exponent))
        #if cur_plot > 5:
        #    grid.alpha = pinilla_alpha(grid.density)#bump_density(radius, options.dust_density_maximum)
        cur_time += dt


if __name__ == '__main__':
    main()
