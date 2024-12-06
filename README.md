# Decay Simulation Code

## Overview

This repository contains code for simulating particle decays of a Lightest Long-Lived Particle (LLP). The simulation includes the generation of decay products in the rest frame of the LLP, calculation of kinematic properties, and optimization for performance. The code utilizes Numba for Just-In-Time (JIT) compilation to enhance speed and efficiency.

## Installation

Ensure you have the required packages installed. You can use the following command to install the dependencies:

```bash
pip3 install numpy sympy numba scipy
```

## File Structure

- `funcs/`:
  - `initLLP.py`: Contains the `LLP` class which initializes the LLP object with attributes like mass, PDGs (Particle Data Group identifiers), and branching ratios.
  - `decayProducts.py`: Contains functions for simulating decays of various LLPs, as well as the routine to perform showering and hadronization of quark and gluon decays via interfacing with pythia8.
  - `HNLmerging.py`: Contains functions for handling HNL merging processes.
  - `PDG.py`: Contains functions or data related to Particle Data Group identifiers.
  - `rotateVectors.py`: Contains functions for rotating vectors.
  - `ThreeBodyDecay.py`: Contains functions for simulating three-body decays.
  - `TwoBodyDecay.py`: Contains functions for simulating two-body decays.
  - `boost.py`: Contains functions for boosting decay products to the lab frame.
  - `kinematics.py`: Contains functions for handling kinematic distributions and interpolations.

- Main code `simulate.py`: thr script to run the decay simulation. Produces two outputs in the folder `outputs/<LLP>`:
  - The information about the decay events of LLPs and decay products (the file `eventData/<LLP>_<mass>_<lifetime>_....txt`), with dots meaning the other parameters relevant for the simulation (such as the mixing pattern in the case of HNLs, etc.).
  - The information about total quantities from the simulation: mass, coupling, lifetime, number of events, etc. (the file `eventData/<LLP>/total/<LLP>-...-total.txt`).
  
- Post-processing:
  - `events_analysis.py`: the script computing various distributions with the decaying LLP and its decay products: position of the decay vertices, energy distributions, multiplicity, etc. The output is saved in the folder `plots/<LLP>/<GivenLLP>`.
  - `total-plots.py`: the script making the plots of some averaged quantites, such as the polar acceptance, total geometric acceptance, mean decay probability, etc. The output is a single plot saved in the folder `plots/<LLP>`.

## Usage

Running the main file `simulate.py` will first ask users about the number of LLPs sampled in the polar range of the SHiP experiment. Then, users will be asked about: 

 - Selecting the LLP.
 - Typing some LLP parameters such as the mixing pattern (optionally, depending on LLP).
 - Selecting the LLP's decay modes for which the simulation will be launched (their names sould be self-explanatory).
 - Range of LLP masses for which the simulation will be launched.
 - Range of LLP lifetimes.
 
 After that, the simulation will start. 

### Output files

- The detailed event record file: 
 - The first string is `Sampled ## events inside SHiP volume. Total number of produced LLPs: ##. Polar acceptance: ##. Azimuthal acceptance: ##. Averaged decay probability: ##. Visible Br Ratio: ##. Total number of events: ##`. The meanings of the numbers are: the total sample size; the total number of LLPs produced during 15 years of SHIP running; the amount of LLPs pointing to the polar range of the experiment; of those, the amount of LLPs that also pass the azimuthal acceptance cut; of those, the averaged probability to decay inside the SHiP volume; the visible branching ratio of selected decay channels; the total number of decay events inside the decay volume.
 - Then, the data is split into blocks. Each is started with `#<process=##; sample_points=##>`. The meanings of `##` are: the name of the LLP decay process; the number of samples per this process. After this string, there is the tabulated data with the decay information. The meaning of the elements is as follows: 
 
   `p_x,LLP p_y,LLP p_z,LLP E_LLP mass_LLP PDG_LLP P_decay,LLP x_decay,LLP y_decay,LLP z_decay,LLP p_x,prod1 p_y,prod1 p_z,prod1 E_prod1 mass_prod1 pdg_prod1 p_x,prod2 ...`
 
 - where `...` means the data for the other decay products. Some of the rows end with the strings `0. 0. 0. 0. 0. -999.`, to account for varying number of decay products in the same decay channel and maintain the flat array if merging all the datasets.
  
 -The file with the total information: contains the self-explanatory header describing the meaning of columns.


## Implemented LLPs

Currently, the following LLPs are implemented:

 - HNLs with arbitrary mixing pattern (`HNL`).
 - Higgs-like scalars produced by the mixing (`Scalar-mixing`).
 - Higgs-like scalars produced by the trilinear coupling (`Scalar-quartic`). If one wants to compute the number of events in the BC5 model, one needs to sum the event rate from the mixing model and the quartic model, with the appropriate rescaling.
 - ALPs coupled to photons (`ALP-photon`).
 
 