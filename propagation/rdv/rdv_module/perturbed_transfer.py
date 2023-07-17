import numpy as np
import pandas as pd
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.interface import spice
from tudatpy.kernel.numerical_simulation import (
    environment_setup,
    propagation_setup,
    estimation_setup,
)

from rdv.configs.force_models import get_force_model
from rdv.utils.date_transformations import epoch_to_datetime

# Load spice kernels
spice.load_standard_kernels([])
bodies_to_create = ["Earth"]
global_frame_origin = "Earth"
global_frame_orientation = "J2000"
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create, global_frame_origin, global_frame_orientation
)
bodies = environment_setup.create_system_of_bodies(body_settings)

earth_gravitational_parameter = bodies.get("Earth").gravitational_parameter
bodies.create_empty_body("Spacecraft")

# Define bodies that are propagated and their respective central bodies
bodies_to_propagate = ["Spacecraft"]
central_bodies = ["Earth"]

# Integrator constants
fixed_step_size = 1  # seconds


def perturbed_transfer(
    initial_state_spacecraft,
    final_state_target,
    start_epoch,
    transfer_duration,
    lambert_maneuvers,
    force_model,
):
    """
    Computes the perturbed transfer trajectory between two states using the Lambert maneuver as a starting point and thr
    differential correction method.

    Parameters
    ----------
    initial_state_spacecraft : np.array
        Initial state of the spacecraft
    final_state_target : np.array
        Final state of the target
    start_epoch : float
        Start epoch of the transfer
    transfer_duration : float
        Duration of the transfer in seconds
    lambert_maneuvers : pd.DataFrame
        DataFrame containing the maneuvers computed by the Lambert targeter
    force_model : str
        Force model to use for the propagation

    Returns
    -------
    maneuvers : pd.DataFrame
        DataFrame containing the maneuvers computed by the differential correction method.
    deltav_tot : float
        Total delta-v of the transfer in m/s

    """
    initial_state_transfer = initial_state_spacecraft.copy()
    initial_state_transfer[3:6] = (
        initial_state_transfer[3:6] + lambert_maneuvers.iloc[0]["deltav_inertial"]
    )

    integrator_settings = propagation_setup.integrator.runge_kutta_4(
        start_epoch, fixed_step_size
    )
    leg_end_epoch = start_epoch + transfer_duration
    termination_settings = propagation_setup.propagator.time_termination(
        leg_end_epoch, terminate_exactly_on_final_condition=True
    )

    # Define accelerations acting on the spacecraft
    acceleration_settings = get_force_model(force_model)

    acceleration_settings_dict = {"Spacecraft": acceleration_settings}

    # Create acceleration models
    acceleration_models = propagation_setup.create_acceleration_models(
        bodies, acceleration_settings_dict, bodies_to_propagate, central_bodies
    )

    numerical_reaches_target = False
    final_state_spacecraft = None
    nb_iterations = 0
    while not numerical_reaches_target:
        nb_iterations += 1
        # Create propagation settings
        propagator_settings = propagation_setup.propagator.translational(
            central_bodies,
            acceleration_models,
            bodies_to_propagate,
            initial_state_transfer,
            termination_settings,
        )

        # Setup parameters settings to propagate the state transition matrix
        parameter_settings = estimation_setup.parameter.initial_states(
            propagator_settings, bodies, []
        )

        # Create the parameters that will be estimated
        parameters_to_estimate = estimation_setup.create_parameter_set(
            parameter_settings, bodies
        )
        # Create the variational equation solver and propagate the dynamics
        variational_equations_solver = (
            numerical_simulation.SingleArcVariationalSimulator(
                bodies,
                integrator_settings,
                propagator_settings,
                parameters_to_estimate,
                integrate_on_creation=True,
            )
        )

        # Extract the resulting state history, state transition matrix history, and sensitivity matrix history
        states = variational_equations_solver.state_history
        state_transition_matrices = (
            variational_equations_solver.state_transition_matrix_history
        )

        final_state_spacecraft = states[leg_end_epoch]

        final_state_deviation = final_state_spacecraft - final_state_target
        final_state_transition_matrix = state_transition_matrices[leg_end_epoch]

        initial_velocity_correction = np.linalg.inv(
            final_state_transition_matrix[:3, 3:]
        ) @ final_state_deviation[:3].reshape(-1, 1)
        initial_state_correction = np.concatenate(
            (np.zeros(3), initial_velocity_correction.reshape(1, -1)[0])
        )

        distance_to_target = np.linalg.norm(final_state_deviation[:3])
        if distance_to_target > 5e-3:
            if nb_iterations == 50:
                # If the correction does not converge after 50 iterations, we return None for the maneuvers and delta-v
                return None, None
            initial_state_transfer -= initial_state_correction
        else:
            numerical_reaches_target = True

    delta_v1_corrected = initial_state_transfer[3:6] - initial_state_spacecraft[3:6]
    delta_v2_corrected = final_state_target[3:6] - final_state_spacecraft[3:6]

    delta_v1_norm = np.linalg.norm(delta_v1_corrected)
    delta_v2_norm = np.linalg.norm(delta_v2_corrected)
    delta_v_total = delta_v1_norm + delta_v2_norm

    deltav_vec_dict = {
        "name": ["deltav_1", "deltav_2"],
        "time": [epoch_to_datetime(start_epoch), epoch_to_datetime(leg_end_epoch)],
        "epoch": [start_epoch, leg_end_epoch],
        "deltav_inertial": [delta_v1_corrected, delta_v2_corrected],
        "deltav_norm": [delta_v1_norm, delta_v2_norm],
    }

    maneuvers = pd.DataFrame(deltav_vec_dict)

    return maneuvers, delta_v_total
