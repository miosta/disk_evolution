# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

psi = raw_input('psi:')

for i in range (0,50):
  plt.figure()
  filenumber = i
  arrays = np.loadtxt('output/cells_' + psi + '_' + str(filenumber)).T
  plt.plot(arrays[0], arrays[1], '-', label='gas surface density')
  plt.plot(arrays[0], arrays[2], '-', label='dust surface density')
  plt.plot(arrays[0], arrays[3], '-', label='alpha')
  plt.plot(arrays[0], -arrays[4], '-', label='dust velocity')
  plt.plot(arrays[0], -arrays[6], '-', label='gas velocity')
  plt.xlim(0.1, 1000)
  plt.ylim(1e-6, 1e5)
  plt.xlabel('Radius in [AU]')
  plt.xscale('log')
  plt.yscale('log')
  plt.ylabel('Surface density in [kg/m*m]')
  plt.legend(loc='lower right', fontsize='9')
  plt.savefig('output/cells_{}_{:02d}.png'.format(psi,filenumber))
  plt.close()
