import numpy as np
import pandas as pd

booster_loc = 187.126706 # in in from tip, will change based on ballast location + length

# goal/le3 moi
true_le3_moi = {"xx": 857590.358, "yy": 857590.358, "zz": 1575.98} # in in^2 lb

# sd moi's
caldera_moi = {"xx": 10394.473, "yy": 10396.598, "zz": 242.361, "dist": 50.294, "mass": 23.3625} # in in^2 lb, dist in in from tip, mass in lb
keroTank_moi = {"xx": 2198.732, "yy": 2198.999, "zz": 177.349, "dist": 118.854674, "mass": 33.818} # in in^2 lb, dist in in from tip, mass in lb    
le3Top_moi = {"xx": 18454.228, "yy": 18456.942, "zz": 228.432607, "dist": booster_loc, "mass": 46.1461969} # in in^2 lb, dist in in from tip, mass in lb


material_densities = pd.read_csv("material_densities.csv").set_index("Material")

# goal: using the le3 moi, calculate the mass, position, and geometry of the ballast
# 1. give the program le3 moi, and known sd moi
# 2. using known sd mass, calculate the total moi using the parallel axis theorem
# 3. now the equation is le3 moi - sd moi = parallel axis theorem(ballast moi)
# 4. using the equation above, solve for the distance from cg using parallel axis and the shape, size, mass, etc
#    of the ballast using triple integral


def sd_moi(moi_dicts, masses=None, reference_distance=0.0):
    """
    Calculate the combined moment of inertia of multiple components using the 
    parallel axis theorem.
    
    Parameters
    ----------
    moi_dicts : list of dict
        List of MOI dictionaries, each containing "xx", "yy", "zz", and "dist" keys.
        MOI values are about the component's center of mass (in lb·in²).
        "dist" is the distance from a reference point (e.g., tip) to the component's 
        center of mass (in inches). Optionally may contain "mass" key.
    masses : list of float, optional
        List of masses for each component (in lb). Must match the order of moi_dicts.
        If None, masses will be taken from the "mass" key in each moi_dict if present.
        If a mass is provided in this list, it overrides any "mass" key in the dict.
    reference_distance : float, optional
        Distance from the reference point to the axis about which we want the 
        total MOI (default: 0.0, meaning MOI about the reference point itself).
    
    Returns
    -------
    dict
        Combined MOI with keys "xx", "yy", "zz" (in lb·in²) about the reference axis.
    """
    total_moi = {"xx": 0.0, "yy": 0.0, "zz": 0.0}
    
    # If masses not provided, extract from dicts or raise error
    if masses is None:
        masses = []
        for moi_dict in moi_dicts:
            if "mass" not in moi_dict:
                raise ValueError(f"MOI dict missing 'mass' key and no masses parameter provided")
            masses.append(moi_dict["mass"])
    elif len(masses) != len(moi_dicts):
        raise ValueError("masses list length must match moi_dicts length")
    
    for moi_dict, mass in zip(moi_dicts, masses):
        # Extract distance from dict
        if "dist" not in moi_dict:
            raise ValueError("MOI dict missing 'dist' key")
        
        dist = moi_dict["dist"]
        # Distance from component's center of mass to the reference axis
        d = dist - reference_distance
        
        # Apply parallel axis theorem for each axis
        for axis in ["xx", "yy", "zz"]:
            if axis not in moi_dict:
                raise ValueError(f"MOI dict missing '{axis}' key")
            total_moi[axis] += parallel_axis_theorem(moi_dict[axis], mass, d)
    
    return total_moi 

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

def tube_moi(length, dist_from_ref=0.0, density=0.06):
    """
    Calculate the mass, center of gravity, and moment of inertia for a hollow 
    cylindrical tube (hollow cylinder) about its centroidal axes.
    
    Returns a dictionary compatible with sd_moi function.
    
    Parameters
    ----------
    length : float
        Tube length (inches).
    dist_from_ref : float, optional
        Distance from reference point (e.g., tip) to the start of the tube (default: 0.0).
        The tube's center of mass will be at dist_from_ref + length/2.
    density : float, optional
        Material density (default: 0.06 lb/in³).
    
    Returns
    -------
    dict
        Dictionary compatible with sd_moi function containing:
        - "mass": mass in pounds
        - "xx": moment of inertia about x-axis (lb·in²) about tube's center of mass
        - "yy": moment of inertia about y-axis (lb·in²) about tube's center of mass
        - "zz": moment of inertia about z-axis (longitudinal, lb·in²) about tube's center of mass
        - "dist": distance from reference point to tube's center of mass (inches)
    """
    # Fixed tube dimensions
    inner_radius = 3.0  # inches
    outer_radius = 3.085  # inches
    
    if length <= 0:
        raise ValueError("length must be positive")
    
    # Calculate volume of hollow cylinder
    volume = np.pi * length * (outer_radius**2 - inner_radius**2)
    
    # Calculate mass
    mass = density * volume
    
    # Moment of inertia about longitudinal axis (z-axis)
    # For hollow cylinder: Izz = 0.5 * m * (R_outer² + R_inner²)
    Izz = 0.5 * mass * (outer_radius**2 + inner_radius**2)
    
    # Moment of inertia about transverse axes (x and y)
    # For hollow cylinder: Ixx = Iyy = (1/12) * m * (3*(R_outer² + R_inner²) + L²)
    Ixx = Iyy = (1.0 / 12.0) * mass * (3 * (outer_radius**2 + inner_radius**2) + length**2)
    
    # Center of gravity is at the geometric center of the tube
    # Distance from reference point to tube's center of mass
    dist = dist_from_ref + length / 2.0
    
    return {
        "mass": mass,
        "xx": Ixx,
        "yy": Iyy,
        "zz": Izz,
        "dist": dist,
    }

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

def calculate_current_SD_MOI():
    """
    Return the Current Moment of Inertia of SD by adding current known components: Caldera, keroTank, le3Top
    """
    #Mass of Each System
    caldera_mass = caldera_moi["mass"]
    kero_mass = keroTank_moi["mass"]
    le3Top_mass = le3Top_moi["mass"]

    #Center of Mass Distance
    caldera_dist = caldera_moi["dist"]
    kero_dist = keroTank_moi["dist"]
    le3Top_dist = le3Top_moi["dist"]

    #New Center of Mass of Combined System
    new_COM = (caldera_mass * caldera_dist + kero_mass * kero_dist + le3Top_mass * le3Top_dist) / (caldera_mass + kero_mass + le3Top_mass)

    #Calculations for New MOI
    caldera_xx = parallel_axis_theorem(caldera_moi["xx"], caldera_mass, caldera_dist - new_COM)
    kero_xx = parallel_axis_theorem(keroTank_moi["xx"], kero_mass, kero_dist - new_COM)
    le3Top_xx = parallel_axis_theorem(le3Top_moi["xx"], le3Top_mass, le3Top_dist - new_COM)
    
    caldera_yy = parallel_axis_theorem(caldera_moi["yy"], caldera_mass, caldera_dist - new_COM)
    kero_yy = parallel_axis_theorem(keroTank_moi["yy"], kero_mass, kero_dist - new_COM)
    le3Top_yy = parallel_axis_theorem(le3Top_moi["yy"], le3Top_mass, le3Top_dist - new_COM)

    caldera_zz = caldera_moi["zz"]
    kero_zz = keroTank_moi["zz"]
    le3Top_zz = le3Top_moi["zz"]

    #Assuming Symmetry ZZ components can just be added, if COM lies on the same centerline, meaning x and y values are same
    SD_xx = caldera_xx + kero_xx + le3Top_xx
    SD_yy = caldera_yy + kero_yy + le3Top_yy
    SD_zz = caldera_zz + kero_zz + le3Top_zz

    return {"xx": SD_xx, "yy": SD_yy, "zz": SD_zz}

def calculate_ballast_moi():
    SD_current_moi = calculate_current_SD_MOI()
    SD_xx = SD_current_moi["xx"]
    SD_yy = SD_current_moi["yy"]
    SD_zz = SD_current_moi["zz"]
    
    return {"xx": true_le3_moi["xx"] - SD_xx, "yy": true_le3_moi["yy"] - SD_yy, "zz": true_le3_moi["zz"] - SD_zz}
    

if __name__ == "__main__":


    '''
    # Example: 4.5 in diameter aluminum cylinder, 10 in long.
    result = cylinder_moi(radius_in=4.5 / 2, length_in=10.0, material="Alloy steel (4140)")
    for material, values in result.items():
        print(material, values)'''
