import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
from matplotlib.patches import Circle
from matplotlib.collections import PatchCollection

direct = raw_input('directory:')
number = raw_input('plot number:')

arrays = np.loadtxt( direct + '/cells_' + number).T 
radius = arrays[0]
values = arrays[1]


cmap = matplotlib.cm.get_cmap('Spectral')

fig = plt.figure()
ax = fig.add_subplot(111)

ax.set_xlim(-100, 100)
ax.set_ylim(-100, 100)

norm = matplotlib.colors.LogNorm(
    vmin=min(values), vmax=3000)

center = (0, 0)
patches = [Circle(center, r) for r in reversed(radius)]

p = PatchCollection(patches)
p.set_norm(norm)
p.set_linewidth(0)
p.set_array(np.array(values[::-1]))
ax.add_collection(p)
ax.set_xlabel('Distance in [AU]')
ax.set_ylabel('Distance in [AU]')
cbar = fig.colorbar(p, ax=ax)
cbar.set_label('Surface density [kg/m^2]')

''' 
fig, ax = plt.subplots()
 
ax.set_xlim(-100, 100)
ax.set_ylim(-100, 100)
ax.set_xlabel('Radius in [AU]')
ax.set_ylabel('Radius in [AU]')

 
cmap = matplotlib.cm.get_cmap('Spectral')
norm = matplotlib.colors.LogNorm(
    vmin=min(values), vmax=max(values))
 
for r, v in reversed(list(zip(radius, values))):
    center = (0, 0)
    color = cmap(norm(v))
    ax.add_artist(plt.Circle(center, r, color=color))

ax1 = fig.add_axes([0.91, 0.02, 0.02, .97])
cb1 = matplotlib.colorbar.ColorbarBase(ax1, cmap=cmap,
                                norm=norm, 
                                #orientation='horizontal'
                                )
cb1.set_label('Surface density [kg/m^2]')
'''

plt.savefig(direct+'/cellmap'+number+'.png')
plt.close()
