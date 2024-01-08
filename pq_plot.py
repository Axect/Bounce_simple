import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scienceplots

# Import parquet file
df = pd.read_parquet('sho.parquet')

# Prepare Data to Plot
omega = np.array(df['omega'][:])
a     = np.array(df['a'][:])
s     = np.array(df['s'][:])

vs    = []
for i in range(16):
    vs.append(np.array(df[f'v{i}'][:]))

vs = np.array(vs).transpose()

x = np.linspace(0, 1, 16)

# Plot params
pparam = dict(
    xlabel = r'$x$',
    ylabel = r'$V(x)$',
)

# Plot
with plt.style.context(["science", "nature"]):
    fig, ax = plt.subplots()
    ax.autoscale(tight=True)
    ax.set(**pparam)
    for i in range(vs.shape[0]):
        ax.plot(x, vs[i])
    fig.savefig('sho_potential_plot.png', dpi=600, bbox_inches='tight')

with plt.style.context(["science", "nature"]):
    fig, ax = plt.subplots()
    ax.autoscale(tight=True)
    ax.set(**pparam)
    plt.hist(s, bins=100)
    fig.savefig('sho_s_hist.png', dpi=600, bbox_inches='tight')
