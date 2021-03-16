import os

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

adaptive = pd.read_json("output/results_adaptive_thick.json")
uniform = pd.read_json("output/results_uniform_thick.json")
adaptive.sort_index(inplace=True)

width = 5.0
height = width / 1.648
size = (width, height)
fig = plt.figure(figsize=size)
ax = plt.gca()
line = np.linspace(5*10**2, 10**4, 10)
optimal = line**(-1.0/2.0) * 10**0
suboptimal = line**(-1.0/4.0) * 8*10**-1
ax.loglog(adaptive['dofs'], adaptive['eta'], '-o', label='Adaptive refinement')
ax.loglog(uniform['dofs'], uniform['eta'], '--o', label='Uniform refinement')
ax.loglog(line, optimal, '-', label='$\mathcal{O}(d^{-1/2})$')
ax.loglog(line, suboptimal, '-', label='$\mathcal{O}(d^{-1/4})$')
ax.set_ylim([10**-3, 3.0*10**-1])

ax.set_xlabel('Number of degrees of freedom $d$')
ax.set_ylabel('Error estimator $\eta$')
ax.legend()

plt.tight_layout()
plt.savefig("output/thick-convergence.pdf")

adaptive = pd.read_json("output/results_adaptive_thin.json")
uniform = pd.read_json("output/results_uniform_thin.json")
adaptive.sort_index(inplace=True)

fig = plt.figure(figsize=size)
ax = plt.gca()
line = np.linspace(5*10**2, 10**4, 10)
optimal = line**(-1.0/2.0) * 10**0
suboptimal = line**(-1.0/4.0) * 8*10**-1
ax.loglog(adaptive['dofs'], adaptive['eta'], '-o', label='Adaptive refinement')
ax.loglog(uniform['dofs'], uniform['eta'], '--o', label='Uniform refinement')
ax.loglog(line, optimal, '-', label='$\mathcal{O}(d^{-1/2})$')
ax.loglog(line, suboptimal, '-', label='$\mathcal{O}(d^{-1/4})$')
ax.set_ylim([10**-3, 3.0*10**-1])

ax.set_xlabel('Number of degrees of freedom $d$')
ax.set_ylabel('Error estimator $\eta$')
ax.legend()

plt.tight_layout()
plt.savefig("output/thin-convergence.pdf")
