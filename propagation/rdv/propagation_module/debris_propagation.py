# Import statements
import pandas as pd
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.interface import spice
from tudatpy.kernel.numerical_simulation import environment_setup, propagation_setup
from tudatpy.util import result2array

from rdv.configs.force_models import get_force_model
from rdv.utils.date_transformations import epoch_to_datetime

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

# Define bodies that are propagated and their respective central bodies
bodies_to_propagate = ["Debris"]
central_bodies = ["Earth"]

# Integrator constants
fixed_step_size = 10  # seconds


def propagate_debris(
    debris_state, start_epoch, end_epoch, force_model, loop_mode=False
):
    """
     Propagates the debris and spacecraft for a given duration

    Parameters
     ----------
     debris_state : np.array
         Initial state of the debris in the inertial frame
     start_epoch : float
         Start epoch of the propagation
     end_epoch : float
         End epoch of the propagation
     force_model : str
         Name of the force model to use for the propagation
     loop_mode : bool
        If True, no text output is printed and the perturbed progress bar position is 1.

     Returns
     -------
         result : pd.DataFrame
             Propagation results
    """
    if not loop_mode:
        print(f"Propagating the debris trajectory with the {force_model} force model.")

    # Define accelerations acting on the spacecraft
    acceleration_settings = get_force_model(force_model)

    acceleration_settings_dict = {"Debris": acceleration_settings}

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
        debris_state,
        termination_condition,
    )

    # Create numerical integrator settings
    integrator_settings = propagation_setup.integrator.runge_kutta_4(
        start_epoch, fixed_step_size
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

    epoch_df = pd.DataFrame(cartesian_states_array[:, 0], columns=["epoch"])

    time_df = pd.DataFrame(
        epoch_to_datetime(cartesian_states_array[:, 0]), columns=["time"]
    )

    cartesian_states_df = pd.DataFrame(
        cartesian_states_array[:, 1::], columns=["x", "y", "z", "vx", "vy", "vz"]
    )

    resultsDataFrame = pd.concat([epoch_df, time_df, cartesian_states_df], axis=1)

    return resultsDataFrame
