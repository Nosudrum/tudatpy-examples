# Import statements
import numpy as np
import pandas as pd
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.interface import spice
from tudatpy.kernel.numerical_simulation import environment_setup, propagation_setup
from tudatpy.util import result2array

from rdv.configs.force_models import get_force_model
from rdv.utils.date_transformations import epoch_to_datetime
from rdv.utils.frame_transformations import inertial_to_lvlh

# Load spice kernels
spice.load_standard_kernels([])

# Create default body settings and bodies system
bodies_to_create = ["Earth"]
global_frame_origin = "Earth"
global_frame_orientation = "J2000"
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create, global_frame_origin, global_frame_orientation
)
bodies = environment_setup.create_system_of_bodies(body_settings)

# Add vehicle object to system of bodies
bodies.create_empty_body("Debris")
bodies.create_empty_body("Spacecraft")

# Define bodies that are propagated and their respective central bodies
bodies_to_propagate = ["Debris", "Spacecraft"]
central_bodies = ["Earth", "Earth"]

# Integrator constants
fixed_step_size = 1  # seconds

# Setup dependent variables to be save
keplerian_state_debris_dep_var = propagation_setup.dependent_variable.keplerian_state(
    "Debris", "Earth"
)
keplerian_state_spacecraft_dep_var = (
    propagation_setup.dependent_variable.keplerian_state("Spacecraft", "Earth")
)
dependent_variables_to_save = [
    keplerian_state_debris_dep_var,
    keplerian_state_spacecraft_dep_var,
]


def propagate(spacecraft_state, debris_state, start_epoch, leg_duration, force_model):
    """
     Propagates the debris and spacecraft for a given duration

    Parameters
     ----------
         spacecraft_state : np.array
             Initial state of the spacecraft in the inertial frame
         debris_state : np.array
             Initial state of the debris in the inertial frame
         start_epoch : float
             Start epoch of the propagation
         leg_duration : float
             Duration of the propagation in seconds
         force_model : str
             Name of the force model to use for the propagation

     Returns
     -------
         result : pd.DataFrame
             Propagation results
    """

    end_epoch = start_epoch + leg_duration

    # Define accelerations acting on the spacecraft
    acceleration_settings = get_force_model(force_model)

    acceleration_settings_dict = {
        "Debris": acceleration_settings,
        "Spacecraft": acceleration_settings,
    }

    # Create acceleration models
    acceleration_models = propagation_setup.create_acceleration_models(
        bodies, acceleration_settings_dict, bodies_to_propagate, central_bodies
    )

    # Create termination settings
    termination_condition = propagation_setup.propagator.time_termination(
        end_epoch, terminate_exactly_on_final_condition=True
    )

    # Create propagation settings
    propagator_settings = propagation_setup.propagator.translational(
        central_bodies,
        acceleration_models,
        bodies_to_propagate,
        np.concatenate((debris_state, spacecraft_state), axis=0),
        termination_condition,
        output_variables=dependent_variables_to_save,
    )

    # Create numerical integrator settings
    integrator_settings = propagation_setup.integrator.runge_kutta_4(
        start_epoch, fixed_step_size
    )

    print(
        f"Propagating from {epoch_to_datetime(start_epoch)} to {epoch_to_datetime(end_epoch)}..."
    )

    # Create simulation object and propagate the dynamics
    dynamics_simulator = numerical_simulation.SingleArcSimulator(
        bodies,
        integrator_settings,
        propagator_settings,
        print_state_data=False,
        print_dependent_variable_data=False,
    )

    # Extract the resulting state history and convert it to a ndarray
    cartesian_states = dynamics_simulator.state_history
    cartesian_states_array = result2array(cartesian_states)
    dependent_variables_history = dynamics_simulator.dependent_variable_history
    dependent_variables_history_array = result2array(dependent_variables_history)

    keplerian_states_debris = dependent_variables_history_array[:, 1:7]
    keplerian_states_spacecraft = dependent_variables_history_array[:, 7:13]

    time_df = pd.DataFrame(
        epoch_to_datetime(cartesian_states_array[:, 0]), columns=["time"]
    )

    cartesian_states_df = pd.DataFrame(
        cartesian_states_array[:, 1::],
        columns=[
            "x_debris",
            "y_debris",
            "z_debris",  # debris ECI position
            "vx_debris",
            "vy_debris",
            "vz_debris",  # debris ECI velocity
            "x_sat",
            "y_sat",
            "z_sat",  # s/c ECI position
            "vx_sat",
            "vy_sat",
            "vz_sat",
        ],
    )  # s/c ECI velocity

    relative_state_lvlh = np.empty((len(time_df), 6))
    for epoch_index in range(len(time_df)):
        relative_state_lvlh[epoch_index, :], _ = inertial_to_lvlh(
            cartesian_states_array[epoch_index, 1:7],
            cartesian_states_array[epoch_index, 7:13],
        )

    relative_state_df = pd.DataFrame(
        relative_state_lvlh,
        columns=["x_rel", "y_rel", "z_rel", "vx_rel", "vy_rel", "vz_rel"],
    )

    keplerian_states_debris_df = pd.DataFrame(
        keplerian_states_debris,
        columns=[
            "sma_debris",  # semi-major axis
            "ecc_debris",  # eccentricity
            "inc_debris",  # inclination
            "aop_debris",  # argument of periapsis
            "raan_debris",  # right ascension of ascending node
            "ta_debris",
        ],
    )  # true anomaly

    keplerian_states_spacecraft_df = pd.DataFrame(
        keplerian_states_spacecraft,
        columns=[
            "sma_spacecraft",  # semi-major axis
            "ecc_spacecraft",  # eccentricity
            "inc_spacecraft",  # inclination
            "aop_spacecraft",  # argument of periapsis
            "raan_spacecraft",  # right ascension of ascending node
            "ta_spacecraft",
        ],
    )  # true anomaly

    resultsDataFrame = pd.concat(
        [
            time_df,
            cartesian_states_df,
            relative_state_df,
            keplerian_states_debris_df,
            keplerian_states_spacecraft_df,
        ],
        axis=1,
    )

    return resultsDataFrame
