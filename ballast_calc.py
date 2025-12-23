import numpy as np
import pandas as pd

true_le3_moi = {"xx": 857590.358, "yy": 857590.358, "zz": 1575.98}
sd_moi = {"xx": 18454.228, "yy": 18456.942, "zz": 228.432607}

material_densities = pd.read_csv("material_densities.csv").set_index("Material")


def sd_moi():
    return 

def calculate_cg(masses, positions):
    """
    Calculate the center of gravity for a system of objects.
    
    Parameters
    ----------
    masses : array-like
        List or array of masses (in any consistent units, e.g., pounds).
    positions : array-like
        List or array of positions, where each position is [x, y, z] coordinates
        relative to a shared reference point (in any consistent units, e.g., inches).
    
    Returns
    -------
    numpy.ndarray
        Center of gravity coordinates [x_cg, y_cg, z_cg] relative to the same
        reference point as the input positions.
    """
    masses = np.array(masses)
    positions = np.array(positions)
    
    # Ensure positions is 2D: (n_objects, 3)
    if positions.ndim == 1:
        positions = positions.reshape(1, -1)
    
    # Calculate total mass
    total_mass = np.sum(masses)
    
    if total_mass == 0:
        raise ValueError("Total mass cannot be zero")
    
    # Calculate weighted average position: CG = Σ(m_i * r_i) / Σ(m_i)
    cg = np.sum(masses[:, np.newaxis] * positions, axis=0) / total_mass
    
    return cg

def cylinder_moi(radius_in, length_in, material="all"):
    """
    Return mass and principal moments of inertia for a solid cylinder about its
    centroidal axes for every material listed in material_densities.csv, plus
    total material cost.

    Parameters
    ----------
    radius_in : float
        Cylinder radius (inches).
    length_in : float
        Cylinder length/height (inches).
    material : str, optional
        "all" (default) to compute for every material in the CSV, or a specific
        material name matching the "Material" column.

    Returns
    -------
    dict
        Mapping from material name to {"mass_lb": m, "Ixx": Ixx, "Iyy": Iyy,
        "Izz": Izz, "cost_usd": cost} with lb·in² units. Izz is about the
        cylinder's longitudinal axis; cost assumes uniform density and uses the
        per-pound cost from material_densities.csv. If a specific material is
        requested, the mapping contains only that material key.
    """
    volume = np.pi * radius_in**2 * length_in

    def _compute(material_name, row):
        rho = float(row["Density_lb_per_in3"])
        cost_per_lb = float(row["Approx_Cost_per_lb_USD"])
        mass = rho * volume  # pounds
        cost = mass * cost_per_lb

        # Centroidal MOI for a solid cylinder
        Izz = 0.5 * mass * radius_in**2
        Ixx = Iyy = (1.0 / 12.0) * mass * (3 * radius_in**2 + length_in**2)

        return {
            "mass_lb": mass,
            "Ixx": Ixx,
            "Iyy": Iyy,
            "Izz": Izz,
            "cost_usd": cost,
        }

    if material == "all":
        return {mat: _compute(mat, row) for mat, row in material_densities.iterrows()}

    if material not in material_densities.index:
        raise ValueError(f"Unknown material '{material}'")

    row = material_densities.loc[material]
    return {material: _compute(material, row)}

def parallel_axis_theorem(I_cm, m, d):
    """
    Return the moment of inertia about a new axis parallel to the centroidal axis.
    """
    return I_cm + m * d**2

if __name__ == "__main__":
    # Example: 4.5 in diameter aluminum cylinder, 10 in long.
    result = cylinder_moi(radius_in=4.5 / 2, length_in=10.0, material="Alloy steel (4140)")
    for material, values in result.items():
        print(material, values)