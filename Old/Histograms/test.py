import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mplhep as hep
hep.style.use("CMS")

plt.rcParams.update({
    'font.size': 14,       # General font size
    'axes.titlesize': 16,  # Font size for axes titles
    'axes.labelsize': 14,  # Font size for axes labels
    'xtick.labelsize': 12, # Font size for x-axis tick labels
    'ytick.labelsize': 12, # Font size for y-axis tick labels
    'legend.fontsize': 12, # Font size for legend text
    'figure.titlesize': 18 # Font size for figure title
})


f_0_05 = pd.read_csv("./kinetic_sampling_0.8_0.05.dat", sep="\t")
f_100 = pd.read_csv("./kinetic_sampling_0.8_100.dat", sep="\t")

E_0_05 = f_0_05["energy"]
P_0_05 = f_0_05["P_decay"]

E_100 = f_100["energy"]
P_100 = f_100["P_decay"]


# f_0_1 = pd.read_csv("./kinetic_sampling_0.1.dat", sep="\t")
# f_1 = pd.read_csv("./kinetic_sampling_1.dat", sep="\t")
# f_100 = pd.read_csv("./kinetic_sampling_100.dat", sep="\t")
# f_1000 = pd.read_csv("./kinetic_sampling_1000.dat", sep="\t")

# E_0_1 = f_0_1["energy"]
# P_0_1 = f_0_1["P_decay"]

# E_1 = f_1["energy"]
# P_1 = f_1["P_decay"]

# E_100 = f_100["energy"]
# P_100 = f_100["P_decay"]

# E_1000 = f_1000["energy"]
# P_1000 = f_1000["P_decay"]

# Plot the weighted histograms normalized to probability density
fig = plt.figure(figsize=(8,6))

plt.hist(E_0_05, bins=30, weights=P_0_05, alpha=0.7, edgecolor='black', color = "deepskyblue",  density=True, label='cτ = 0.05 m') #Including weights show the production plot
plt.hist(E_100, bins=30, weights=P_100, alpha=0.7, edgecolor='black', color = "coral", density=True, label='cτ = 100 m')

# plt.hist(E_0_1, bins=30, weights=P_0_1, alpha=0.7, edgecolor='black', density=True, label='cτ = 0.1 m') #Including weights show the production plot
# plt.hist(E_1, bins=30, weights=P_1, alpha=0.7, edgecolor='black', density=True, label='cτ = 1 m')
# plt.hist(E_100, bins=30, weights=P_100, color = "limegreen", alpha=0.7, edgecolor='black', density=True, label='cτ = 100 m')

# plt.hist(E_1000, bins=30, weights=P_1000, alpha=0.7, edgecolor='red', density=True, label='Dataset 4')
# Add titles and labels
plt.title('Higgs like scalars, m = 0.8 GeV')
plt.xlabel('Energy [GeV]')
plt.ylabel('Probability Density')

# Add legend
plt.legend()
fig.savefig("Weighted_histograms.png", dpi=300)
# plt.show()

fig = plt.figure(figsize=(8,6))

plt.hist(E_0_05, bins=30, alpha=0.7, edgecolor='black', density=True) #Including weights show the production plot
# plt.hist(E_100, bins=30, alpha=0.7, edgecolor='black', density=True, label='cτ = 100 m')

# plt.hist(E_0_1, bins=30, alpha=0.7, edgecolor='black', density=True, label='cτ = 0.1 m') #Including weights show the production plot
# plt.hist(E_1, bins=30, alpha=0.7, edgecolor='black', density=True, label='cτ = 1 m')
# plt.hist(E_100, bins=30, alpha=0.7, color = "limegreen", edgecolor='black', density=True, label='cτ = 100 m')

# Add titles and labels
plt.xlabel('Energy [GeV]')
plt.ylabel('Probability Density')

plt.title('Higgs like scalars, m = 0.8 GeV')
# Add legend
# plt.legend()
fig.savefig("Unweighted_histograms.png", dpi=300)

# Show the plot
# plt.show()