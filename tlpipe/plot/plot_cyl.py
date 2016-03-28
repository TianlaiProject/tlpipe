import matplotlib.pyplot as plt


fig = plt.figure(figsize=(5,8))
ax  = fig.add_axes([0.1, 0.1, 0.85, 0.85])

for i in range(31):

    x = 6
    y = (30 - i) * 0.4133

    cir = plt.Circle((x, y), 
            radius = 0.16, fc='none', ec='k', lw=1)
    ax.text(x, y, '%02d'%i,
            horizontalalignment='center', verticalalignment='center', fontsize=8)
    ax.add_patch(cir)

for i in range(32):

    x = 3
    y = (31 - i) * 0.40

    cir = plt.Circle((x, y), 
            radius = 0.16, fc='none', ec='k', lw=1)
    #ax.text(x, y, '%02d'%(i+31),
    #        horizontalalignment='left', verticalalignment='center')
    ax.text(x, y, '%02d'%(i+31),
            horizontalalignment='center', verticalalignment='center', fontsize=8)
    ax.add_patch(cir)

for i in range(33):

    x = 0
    y = (32 - i) * 0.3875

    cir = plt.Circle((x, y), 
            radius = 0.16, fc='none', ec='k', lw=1)
    #ax.text(x, y, '%02d'%(i+63),
    #        horizontalalignment='left', verticalalignment='center')
    ax.text(x, y, '%02d'%(i+63),
            horizontalalignment='center', verticalalignment='center', fontsize=8)
    ax.add_patch(cir)

ax.autoscale_view()
ax.set_aspect('equal')
ax.set_ylim(ymin=-1, ymax=14)
ax.set_xlim(xmin=-1, xmax=7)
ax.set_xticklabels([])
ax.set_yticklabels([])
#ax.minorticks_on()
ax.tick_params(length=0, width=1., direction='out')
#ax.tick_params(which='minor', length=2, width=1., direction='out')
ax.set_xlabel('X (EW)')
ax.set_ylabel('Y (NS)')



plt.show()

