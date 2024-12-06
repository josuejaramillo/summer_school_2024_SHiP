# EventCalc

Sampler of decay events with hypothetical long-lived particles for the SHiP experiment. 

## Overview

The approach is based on the tabulated angle-energy distribution of the produced LLPs. Namely, the code
- Takes the tabuled distributions and the tabulated dependence of the maximum LLP energy on the mass and the polar angle, summed over all possible production channels, 
- Samples the 3-momentum of the LLPs in the direction of the decay volume of SHiP
- Samples the decay positions based on the exponential distribution in the LLP decay lengths.
- Simulates 2-, 3-, or 4-body decays using internal phase space simulator, and then passes the events to pythia8 for showering and hadronization. 
- Using the tabulated total yields, decay branching ratios, lifetime-mass dependence, and computed geometric acceptance and decay probability, calculates the total number of decay events.

Currently, the tabulated distributions are generated by [SensCalc](https://github.com/maksymovchynnikov/SensCalc). The matrix elements, branching ratios, lifetime dependence on mass, and production fluxes are also taken from it.

The code does not simulate decay products acceptance. Instead, its output is provided to [FairShip](https://github.com/ShipSoft/FairShip) for simulating the propagation and reconstruction of the decay products.

## Usage

Running the main file `simulate.py` will first ask users about the number of LLPs sampled in the polar range of the SHiP experiment. Then, users will be asked about: 

 - Selecting the LLP.
 - Typing some LLP parameters such as the mixing pattern, variation of the theoretical uncertainty, and others (optionally, depending on LLP).
 - Selecting the LLP's decay modes for which the simulation will be launched (their names sould be self-explanatory).
 - Range of LLP masses for which the simulation will be launched.
 - Range of LLP lifetimes.
 
 After that, the simulation will start. It produces two outputs in the folder `outputs/<LLP>` (see description below):
  - The information about the decay events of LLPs and decay products (the file `eventData/<LLP>_<mass>_<lifetime>_....txt`), with dots meaning the other parameters relevant for the simulation (such as the mixing pattern in the case of HNLs, etc.).
  - The information about total quantities from the simulation: mass, coupling, lifetime, number of events, etc. (the file `eventData/<LLP>/total/<LLP>-...-total.txt`).

## Installation

Ensure you have the required packages installed. You can use the following command to install the dependencies:

```bash
pip3 install numpy sympy numba scipy plotly
```

Also, [pythia8](https://pythia.org/) must be installed with python config. When configuring it, type

`./configure --with-python-config=python3-config`

After `make`, the lib folder has to contain the `pythia8.so` file.

Once this is done, the lib path has to be specified in the script `funcs/decayProducts.py`. Currently, it is

`sys.path.insert(0, '/home/name/Downloads/pythia8312/lib')`



## File Structure

- `funcs/`:
  - `initLLP.py`: Contains the `LLP` class which initializes the LLP object with attributes like mass, PDGs (Particle Data Group identifiers), and branching ratios.
  - `decayProducts.py`: Contains functions for simulating decays of various LLPs, as well as the routine to perform showering and hadronization of quark and gluon decays via interfacing with pythia8.
  - `HNLmerging.py`: Contains functions for handling HNLs with arbitrary mixing pattern given the tabulated distributions, branching ratios, lifetimes, total yields, and decay matrix elements for pure mixings.
  - `PDG.py`: Contains functions or data related to Particle Data Group identifiers.
  - `rotateVectors.py`: Contains functions for rotating vectors.
  - `FourBodyDecay.py`: Contains functions for simulating three-body decays.
  - `ThreeBodyDecay.py`: Contains functions for simulating three-body decays.
  - `TwoBodyDecay.py`: Contains functions for simulating two-body decays.
  - `boost.py`: Contains functions for boosting decay products to the lab frame.
  - `kinematics.py`: Contains functions for handling kinematic distributions and interpolations.
  - `ship_volume.py`: Contains geometry of the SHiP decay volume for making plots.

- Main code `simulate.py`: thr script to run the decay simulation. 
  
- Post-processing:
  - `events_analysis.py`: the script computing various distributions with the decaying LLP and its decay products: position of the decay vertices, energy distributions, multiplicity, etc. The output is saved in the folder `plots/<LLP>/<GivenLLP>`.
  - `total-plots.py`: the script making the plots of some averaged quantites, such as the polar acceptance, total geometric acceptance, mean decay probability, etc. The output is a single plot saved in the folder `plots/<LLP>`.
  - `event-display.py`: the script making .pdf and interactive .html plots showing the decay point of the LLP, the direction of its momentum, and the directions of its decay products.

## Output files

- The detailed event record file: 
  - The first string is `Sampled ## events inside SHiP volume. Total number of produced LLPs: ##. Polar acceptance: ##. Azimuthal acceptance: ##. Averaged decay probability: ##. Visible Br Ratio: ##. Total number of events: ##`. The meanings of the numbers are: the total sample size; the total number of LLPs produced during 15 years of SHIP running; the amount of LLPs pointing to the polar range of the experiment; of those, the amount of LLPs that also pass the azimuthal acceptance cut; of those, the averaged probability to decay inside the SHiP volume; the visible branching ratio of selected decay channels; the total number of decay events inside the decay volume.
  - After the first string, the data is split into blocks. Each is started with `#<process=##; sample_points=##>`. The meanings of `##` are: the name of the LLP decay process; the number of samples per this process. After this string, there is the tabulated data with the decay information. The meaning of the elements is as follows: 
 
   `p_x,LLP p_y,LLP p_z,LLP E_LLP mass_LLP PDG_LLP P_decay,LLP x_decay,LLP y_decay,LLP z_decay,LLP p_x,prod1 p_y,prod1 p_z,prod1 E_prod1 mass_prod1 pdg_prod1 p_x,prod2 ...`
 
  - where `...` means the data for the other decay products. Some of the rows end with the strings `0. 0. 0. 0. 0. -999.`, to account for varying number of decay products in the same decay channel and maintain the flat array if merging all the datasets.
  
- The file with the total information: contains the self-explanatory header describing the meaning of columns.


## Implemented LLPs

Currently, the following LLPs are implemented:

- HNLs with arbitrary mixing pattern (`HNL`).
- Higgs-like scalars produced by the mixing (`Scalar-mixing`).
- Higgs-like scalars produced by the trilinear coupling (`Scalar-quartic`). If one wants to compute the number of events in the BC5 model, one needs to sum the event rate from the mixing model and the quartic model, with the appropriate rescaling.
- ALPs coupled to photons (`ALP-photon`).
- Dark photons (`Dark-photons`). They have a large theoretical uncertainty in the production. Because of this, the users are asked to select the flux within the range of this uncertainty - `lower`, `central`, or `upper` (see 2409.11096 for details). 
 
## To be done

- Adding more LLPs (ALPs, B-L mediators, HNLs with dipole coupling, inelastic and elastic LDM, etc.).
- Adding theoretical uncertainty for Higgs-like scalars.
- Improving the performance of the code (parallelization of pythia8 run, etc.).
- Adjusting the SHiP setup with the up-to-date setup.
- Adding cascade production from kaons and light mesons for HNLs, Higgs-like scalars, dark photons.
- Adding more sophisticated simulation codes (such as the machinery to simulate HNL-anti-HNL oscillations).
- Introduce individual identifiers for various LLPs (currently, it is 12345678).