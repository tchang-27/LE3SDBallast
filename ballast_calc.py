import numpy as np
import pandas as pd

true_le3_moi = {"xx": 857590.358, "yy": 857590.358, "zz": 1575.98}
sd_moi = {"xx": 18454.228, "yy": 18456.942, "zz": 228.432607}

material_densities = pd.read_csv("material_densities.csv").set_index("Material")


def cylinder_moi(radius_in, length_in, material="Aluminum 6061-T6"):
    """
    Return mass and principal moments of inertia for a solid cylinder about its
    centroidal axes.

    Parameters
    ----------
    radius_in : float
        Cylinder radius (inches).
    length_in : float
        Cylinder length/height (inches).
    material : str, optional
        Material name matching the "Material" column in material_densities.csv.

    Returns
    -------
    dict
        {"mass_lb": m, "Ixx": Ixx, "Iyy": Iyy, "Izz": Izz} with lb·in² units.
        Izz is about the cylinder's longitudinal axis.
    """
    try:
        rho = float(material_densities.loc[material, "Density_lb_per_in3"])
    except KeyError as exc:
        raise ValueError(f"Unknown material '{material}'") from exc

    volume = np.pi * radius_in**2 * length_in
    mass = rho * volume  # pounds

    # Centroidal MOI for a solid cylinder
    Izz = 0.5 * mass * radius_in**2
    Ixx = Iyy = (1.0 / 12.0) * mass * (3 * radius_in**2 + length_in**2)

    return {"mass_lb": mass, "Ixx": Ixx, "Iyy": Iyy, "Izz": Izz}


if __name__ == "__main__":
    # Example: 4.5 in diameter aluminum cylinder, 10 in long.
    result = cylinder_moi(radius_in=4.5 / 2, length_in=10.0)
    print(result)