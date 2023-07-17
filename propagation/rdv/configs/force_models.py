from tudatpy.kernel.numerical_simulation import propagation_setup


def get_force_model(model_name):
    """
    Returns the force model corresponding to the given name.

    Parameters
    ----------
    model_name : str
        Name of the force model

    Returns
    -------
    force_model : dict
        Dictionary containing the force model

    """
    if model_name == "keplerian":
        acceleration_settings = dict(
            Earth=[propagation_setup.acceleration.point_mass_gravity()]
        )
    elif model_name == "J2":
        acceleration_settings = dict(
            Earth=[propagation_setup.acceleration.spherical_harmonic_gravity(2, 0)]
        )
    else:
        raise ValueError("Invalid force model name")

    return acceleration_settings
