# Define a dictionary with particle properties
particle_db = {
    # Particle: [mass (GeV), charge, stability]
    11: [0.000511, -1, 1],  # Electron
    -11: [0.000511, +1, 1], # Positron (antiparticle of electron)
    13: [0.1057, -1, 1],    # Muon
    -13: [0.1057, +1, 1],   # Anti-muon
    211: [0.1396, +1, 1],   # Charged pion (π+)
    -211: [0.1396, -1, 1],  # Charged pion (π-)
    111: [0.134, 0, 1],     # Neutral pion (π0)
    321: [0.4937, +1, 1],   # Charged kaon (K+)
    -321: [0.4937, -1, 1],  # Charged kaon (K-)
    311: [0.4976, 0, 1],    # Neutral kaon (K0) short-lived
    -311: [0.4976, 0, 1],   # Anti-neutral kaon (K0-bar) short-lived
    15: [1.776, -1, 1],     # Tau
    -15: [1.776, +1, 1],    # Anti-tau
    1: [0.0022, +2/3, 1],   # Up quark
    -1: [0.0022, -2/3, 1],  # Down quark
    2: [0.0047, +2/3, 1],   # Charm quark
    -2: [0.0047, -2/3, 1],  # Strange quark
    3: [0.173, +2/3, 1],   # Top quark
    -3: [0.173, -2/3, 1],  # Bottom quark
    21: [0, 0, 1],          # Gluon
    113: [0.775, 0, 1],     # Rho meson (ρ0)
    213: [0.775, +1, 1],    # Rho meson (ρ+)
    -213: [0.775, -1, 1],   # Rho meson (ρ-)
    223: [0.782, 0, 1],     # Omega meson (ω)

    # Neutrinos
    12: [0, 0, 1],          # Electron neutrino (νe)
    -12: [0, 0, 1],         # Electron anti-neutrino (ν̅e)
    14: [0, 0, 1],          # Muon neutrino (νμ)
    -14: [0, 0, 1],         # Muon anti-neutrino (ν̅μ)
    16: [0, 0, 1],          # Tau neutrino (ντ)
    -16: [0, 0, 1]          # Tau anti-neutrino (ν̅τ)
}

def get_particle_properties(pdg_code):
    """Return the properties of a particle given its PDG code."""
    return particle_db.get(pdg_code, "Particle not found in the database.")

def get_mass(pdg_code):
    """Return the mass of a particle given its PDG code."""
    result = get_particle_properties(pdg_code)
    if isinstance(result, list):
        return result[0]
    return result

def get_charge(pdg_code):
    """Return the charge of a particle given its PDG code."""
    result = get_particle_properties(pdg_code)
    if isinstance(result, list):
        return result[1]
    return result

def get_stability(pdg_code):
    """Return the stability of a particle given its PDG code."""
    result = get_particle_properties(pdg_code)
    if isinstance(result, list):
        return result[2]
    return result
