# Orbital Rendezvous
"""

Copyright (c) 2010-2022, Delft University of Technology. All rights reserved. This file is part of the Tudat. Redistribution and use in source and binary forms, with or without modification, are permitted exclusively under the terms of the Modified BSD license. You should have received a copy of the license with this file. If not, please or visit: http://tudat.tudelft.nl/LICENSE.
"""

## Context
"""

???
"""

## Import statements
"""

The required import statements are made here, at the very beginning.

Some standard modules are first loaded. These are `numpy` and `matplotlib.pyplot`.

Then, the different modules of `tudatpy` that will be used are imported.
"""

# Load standard modules
import matplotlib.pyplot as plt
import numpy as np

# Load tudatpy modules
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.astro.element_conversion import (
    mean_motion_to_semi_major_axis,
    mean_to_true_anomaly,
    keplerian_to_cartesian,
)
from tudatpy.kernel.astro.two_body_dynamics import LambertTargeterIzzo
from tudatpy.kernel.interface import spice
from tudatpy.kernel.numerical_simulation import (
    environment_setup,
    propagation_setup,
    estimation_setup,
)
from tudatpy.util import result2array

# Load spice kernels
spice.load_standard_kernels([])

# Create default body settings and bodies system
bodies_to_create = ["Earth"]
global_frame_origin = "Earth"
global_frame_orientation = "J2000"
body_settings = environment_setup.get_default_body_settings(
    bodies_to_create, global_frame_origin, global_frame_orientation
)
mu_earth = (
    environment_setup.create_system_of_bodies(body_settings)
    .get_body("Earth")
    .gravitational_parameter
)

# Set the initial Keplerian state of the ISS
iss_sma = mean_motion_to_semi_major_axis(15.49773683 * 2 * np.pi / 86400, mu_earth)
iss_ecc = 0.0000399
iss_inc = np.deg2rad(51.6397)
iss_aop = np.deg2rad(80.2416)
iss_raan = np.deg2rad(185.0279)
iss_tan = mean_to_true_anomaly(iss_ecc, np.deg2rad(16.4271))
iss_keplerian = [iss_sma, iss_ecc, iss_inc, iss_aop, iss_raan, iss_tan]

# Define the initial keplerian orbit for the spacecraft with respect to the ISS: 10 km below and 50 km behind
spacecraft_keplerian = iss_keplerian.copy()
spacecraft_keplerian[0] -= 10000
spacecraft_keplerian[-1] -= 50000 / iss_sma
print(f"Spacecraft Keplerian state : \n{spacecraft_keplerian}")

# Convert the ISS and spacecraft Keplerian states to Cartesian coordinates
iss_state = keplerian_to_cartesian(iss_keplerian, mu_earth)
spacecraft_state = keplerian_to_cartesian(spacecraft_keplerian, mu_earth)
print(f"ISS Cartesian state : \n{iss_state}")
print(f"Spacecraft Cartesian state : \n{spacecraft_state}")

transfer_max_duration = 3 * 3600  # seconds

# Integrator constants
fixed_step_size = 10  # seconds

acceleration_settings = dict(
    Earth=[propagation_setup.acceleration.spherical_harmonic_gravity(2, 0)]
)


def propagate_target(
    initial_state: np.ndarray,
    start_epoch: float,
    end_epoch: float,
) -> np.ndarray:
    """
    Propagates the target object for a given duration

    Parameters
     ----------
     initial_state : np.ndarray
         Initial state of the target in the inertial frame
     start_epoch : float
         Start epoch of the propagation
     end_epoch : float
         End epoch of the propagation

     Returns
     -------
         np.ndarray
             Array containing the propagation results
    """
    bodies = environment_setup.create_system_of_bodies(body_settings)
    bodies.create_empty_body("Target")

    # Define bodies that are propagated and their respective central bodies
    bodies_to_propagate = ["Target"]
    central_bodies = ["Earth"]

    acceleration_settings_dict = {"Target": acceleration_settings}

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
        initial_state,
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

    return result2array(cartesian_states)


def lambert_transfer(
    initial_state_spacecraft: np.ndarray,
    final_state_target: np.ndarray,
    start_epoch: float,
    transfer_duration: float,
) -> (np.ndarray, float):
    """
    Computes the transfer from the initial state of the spacecraft to the final state of the target using the Lambert
    targeter.

    Parameters
    ----------
    initial_state_spacecraft : np.ndarray
        Initial state of the spacecraft in the inertial frame
    final_state_target : np.ndarray
        Final state of the target in the inertial frame
    start_epoch : float
        Start epoch of the transfer
    transfer_duration : float
        Duration of the transfer in seconds

    Returns
    -------
    maneuvers : np.ndarray
        Array containing the maneuvers
    dv_total : float
        Total delta-v of the transfer in m/s

    """
    lambert_targeter = LambertTargeterIzzo(
        departure_position=initial_state_spacecraft[0:3],
        arrival_position=final_state_target[0:3],
        time_of_flight=transfer_duration,
        gravitational_parameter=mu_earth,
        is_retrograde=False,
    )

    v1, v2 = lambert_targeter.get_velocity_vectors()

    delta_v1 = v1 - initial_state_spacecraft[3:6]
    delta_v2 = final_state_target[3:6] - v2

    delta_v1_norm = np.linalg.norm(delta_v1)
    delta_v2_norm = np.linalg.norm(delta_v2)

    dv_total = delta_v1_norm + delta_v2_norm

    if dv_total > np.linalg.norm(initial_state_spacecraft[3:6]):
        # Delta-v corresponds to an inversion of the orbit, recompute the transfer as retrograde
        lambert_targeter = LambertTargeterIzzo(
            departure_position=initial_state_spacecraft[0:3],
            arrival_position=final_state_target[0:3],
            time_of_flight=transfer_duration,
            gravitational_parameter=mu_earth,
            is_retrograde=True,
        )

        v1, v2 = lambert_targeter.get_velocity_vectors()

        delta_v1 = v1 - initial_state_spacecraft[3:6]
        delta_v2 = final_state_target[3:6] - v2

        delta_v1_norm = np.linalg.norm(delta_v1)
        delta_v2_norm = np.linalg.norm(delta_v2)

        dv_total = delta_v1_norm + delta_v2_norm

    maneuvers_epochs = [start_epoch, start_epoch + transfer_duration]
    maneuvers_dv = [delta_v1, delta_v2]
    maneuvers = np.array([maneuvers_epochs, maneuvers_dv])

    return maneuvers, dv_total


def perturbed_transfer(
    initial_state_spacecraft: np.ndarray,
    final_state_target: np.ndarray,
    start_epoch: float,
    transfer_duration: float,
    lambert_maneuvers: np.ndarray,
) -> (np.ndarray, float):
    """
    Computes the perturbed transfer trajectory between two states using the Lambert maneuver as a starting point and thr
    differential correction method.

    Parameters
    ----------
    initial_state_spacecraft : np.ndarray
        Initial state of the spacecraft
    final_state_target : np.ndarray
        Final state of the target
    start_epoch : float
        Start epoch of the transfer
    transfer_duration : float
        Duration of the transfer in seconds
    lambert_maneuvers : np.ndarray
        DataFrame containing the maneuvers computed by the Lambert targeter

    Returns
    -------
    maneuvers : np.ndarray
        Array containing the maneuvers computed by the differential correction method.
    dv_total : float
        Total delta-v of the transfer in m/s

    """
    # Create default body settings and bodies system
    bodies = environment_setup.create_system_of_bodies(body_settings)

    # Add vehicle object to system of bodies
    bodies.create_empty_body("Target")
    bodies.create_empty_body("Spacecraft")

    # Define bodies that are propagated and their respective central bodies
    bodies_to_propagate = ["Target", "Spacecraft"]
    central_bodies = ["Earth", "Earth"]

    initial_state_transfer = initial_state_spacecraft.copy()
    initial_state_transfer[3:6] += lambert_maneuvers[1, 0]

    integrator_settings = propagation_setup.integrator.runge_kutta_4(
        start_epoch, fixed_step_size
    )
    end_epoch = start_epoch + transfer_duration
    termination_settings = propagation_setup.propagator.time_termination(
        end_epoch, terminate_exactly_on_final_condition=True
    )

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

        final_state_spacecraft = states[-1]

        final_state_deviation = final_state_spacecraft - final_state_target
        final_state_transition_matrix = state_transition_matrices[-1]

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
    dv_total = delta_v1_norm + delta_v2_norm

    maneuvers_epochs = [start_epoch, end_epoch]
    maneuvers_dv = [delta_v1_corrected, delta_v2_corrected]
    maneuvers = np.array([maneuvers_epochs, maneuvers_dv])

    return maneuvers, dv_total


def propagate(
    initial_state_spacecraft: np.ndarray,
    initial_state_target: np.ndarray,
    start_epoch: float,
    duration: float,
) -> np.ndarray:
    """
    Propagates the target and spacecraft for a given duration and returns their cartesian, keplerian and relative states.

    Parameters
     ----------
         initial_state_spacecraft : np.array
             Initial state of the spacecraft in the inertial frame
         initial_state_target : np.array
             Initial state of the target in the inertial frame
         start_epoch : float
             Start epoch of the propagation
         duration : float
             Duration of the propagation in seconds

     Returns
     -------
         results : np.ndarray
             Array containing the propagation results
    """
    # Create default body settings and bodies system
    bodies = environment_setup.create_system_of_bodies(body_settings)

    # Add vehicle object to system of bodies
    bodies.create_empty_body("Target")
    bodies.create_empty_body("Spacecraft")

    # Define bodies that are propagated and their respective central bodies
    bodies_to_propagate = ["Target", "Spacecraft"]
    central_bodies = ["Earth", "Earth"]

    # Setup dependent variables to be save
    keplerian_state_target_dep_var = (
        propagation_setup.dependent_variable.keplerian_state("Target", "Earth")
    )
    keplerian_state_spacecraft_dep_var = (
        propagation_setup.dependent_variable.keplerian_state("Spacecraft", "Earth")
    )
    dependent_variables_to_save = [
        keplerian_state_target_dep_var,
        keplerian_state_spacecraft_dep_var,
    ]

    end_epoch = start_epoch + duration

    acceleration_settings_dict = {
        "Target": acceleration_settings,
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
        np.concatenate((initial_state_target, initial_state_spacecraft), axis=0),
        termination_condition,
        output_variables=dependent_variables_to_save,
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
    dependent_variables_history = dynamics_simulator.dependent_variable_history
    dependent_variables_history_array = result2array(dependent_variables_history)

    relative_state_lvlh = np.empty((cartesian_states_array.shape[0], 6))
    for ii in range(cartesian_states_array.shape[0]):
        relative_state_lvlh[ii, :] = inertial_to_lvlh(
            state_target=cartesian_states_array[ii, 1:7],
            state_spacecraft=cartesian_states_array[ii, 7:13],
        )

    results = np.concatenate(
        (
            cartesian_states_array,
            dependent_variables_history_array[:, 1:],
            relative_state_lvlh,
        ),
        axis=1,
    )

    return results


def full_propagation(
    initial_state_spacecraft: np.array,
    initial_state_target: np.array,
    start_epoch: float,
    end_epoch: float,
    maneuvers: np.ndarray,
):
    """
    Propagates the spacecraft and the target from the start epoch to the end epoch using the maneuvers provided.

    Parameters
    ----------
    initial_state_spacecraft : np.ndarray
        Initial cartesian state of the spacecraft.
    initial_state_target : np.ndarray
        Initial cartesian state of the target.
    start_epoch : float
        Start epoch of the propagation.
    end_epoch : float
        End epoch of the propagation.
    maneuvers : np.ndarray
        Array containing the maneuvers.

    Returns
    -------
    trajectory : np.ndarray
        Array containing the propagation results.

    """
    relative_state_lvlh = inertial_to_lvlh(
        state_target=initial_state_target, state_spacecraft=initial_state_spacecraft
    )

    trajectory = np.concatenate(
        (
            start_epoch,
            initial_state_target,
            initial_state_spacecraft,
            relative_state_lvlh,
        )
    )

    for leg in range(1, len(maneuvers) + 1):
        print("-----------------------------------")
        print(f"Starting leg {leg}...")
        if leg < len(maneuvers):
            # Leg lasts until the next maneuver
            leg_duration = maneuvers[leg, 0] - maneuvers[leg - 1, 0]
        else:
            # Last leg lasts until the end of the simulation
            leg_duration = end_epoch - maneuvers[leg - 1, 0]

        if leg > 1:
            # Update the initial state of the target and spacecraft
            initial_state_target = trajectory[-1, 1:7]
            initial_state_spacecraft = trajectory[-1, 7:13]
            maneuver_epoch = trajectory[-1, 0]
        else:
            maneuver_epoch = start_epoch

        print(f"Applying maneuver {leg} at epoch {maneuver_epoch}")

        # Apply the maneuver and propagate the leg
        initial_state_spacecraft[3:6] += maneuvers[leg - 1, 1:4]
        leg_array = propagate(
            initial_state_spacecraft=initial_state_spacecraft,
            initial_state_target=initial_state_target,
            start_epoch=maneuvers[leg - 1, 0],
            duration=leg_duration,
        )
        leg_array[:, -1] = leg

        # Add the leg trajectory to the total trajectory dataframe
        trajectory = np.concatenate((trajectory, leg_array), axis=0)

    print("-----------------------------------")
    print("Done with all legs.")

    return trajectory


def compute_trajectory(
    initial_state_spacecraft,
    initial_state_debris,
    start_time,
    end_time,
    mission_name,
    debris_name,
    rdv_strategy="direct",
    force_model="keplerian",
    optimization_method="lambert",
    skip_propagation=False,
    loop_mode=False,
):
    """Computes the trajectory of the spacecraft and the debris during the rendezvous.

    Parameters
    ----------
    initial_state_spacecraft : np.array
        inertial state of the spacecraft.
    initial_state_debris : np.array
        inertial state of the debris.
    start_time : datetime.datetime
        start time of the rendezvous.
    end_time : datetime.datetime
        End of the transfer.
    mission_name : str
        Name of the mission.
    debris_name : str
        Name of the debris.
    rdv_strategy : str
        Name of the rendezvous strategy to use. Default is "direct".
    force_model : str
        Name of the force model to use for the propagation. Default is "keplerian".
    optimization_method : str
        Method used optimize the deltaV (lambert or perturbed). Default is "lambert".
    skip_propagation : bool
        If True, the propagation is skipped and only the maneuvers are computed.
    loop_mode : bool
        If True, no text output is printed and the perturbed progress bar position is 1.

    Returns
    -------
    maneuvers : pd.DataFrame
        DataFrame containing the maneuvers.
    debris_trajectory : pd.DataFrame
        DataFrame containing the debris and spacecraft trajectory.

    """
    start_epoch = datetime_to_epoch(start_time)
    end_epoch = datetime_to_epoch(end_time)
    state_debris_df = propagate_debris(
        initial_state_debris, start_epoch, end_epoch, force_model, loop_mode=loop_mode
    )

    maneuvers = pd.DataFrame()

    makedirs(
        f"results/{mission_name}/{debris_name}/{rdv_strategy}/{force_model}/{optimization_method}",
        exist_ok=True,
    )

    if not loop_mode:
        print(
            f"Starting RDV with the {rdv_strategy} strategy, {force_model} force model, "
            f"and {optimization_method} optimization method."
        )
    if rdv_strategy == "direct":  # direct rendezvous --> no waypoints
        rdv_duration, maneuvers = transfer_maneuvers(
            initial_state_spacecraft,
            state_debris_df,
            start_epoch,
            end_epoch,
            mission_name,
            debris_name,
            force_model,
            optimization_method,
            rdv_strategy,
            1,
            loop_mode=loop_mode,
        )
    else:
        # Get waypoints and add the debris (relative state = 0) as final waypoint
        waypoints = rdv_waypoints(rdv_strategy)
        waypoints = np.vstack((waypoints, np.zeros(3)))

        # Start from the initial state of the spacecraft
        initial_state_waypoint = initial_state_spacecraft
        rdv_duration = 0
        leg_start = start_epoch

        for waypoint_id in range(waypoints.shape[0]):
            waypoint_relative_coordinates = waypoints[waypoint_id, :]
            state_waypoint_df = state_debris_df[
                state_debris_df["epoch"] >= leg_start
            ].copy()

            # Compute the state of the waypoint at each epoch (only position is changed compared to debris)
            for epoch_index in range(len(state_waypoint_df)):
                inertial_state_waypoint = lvlh_to_inertial(
                    state_debris_df[state_debris_df["epoch"] >= leg_start]
                    .iloc[epoch_index, 2:]
                    .to_numpy(dtype=np.float64),
                    np.concatenate((waypoint_relative_coordinates, np.zeros(3))),
                )
                # Update the state of the waypoint in the dataframe
                state_waypoint_df.iloc[epoch_index, 2:] = inertial_state_waypoint

            # Compute the optimal transfer for this leg
            leg_duration, leg_maneuvers = transfer_maneuvers(
                initial_state_waypoint,
                state_waypoint_df,
                leg_start,
                end_epoch,
                mission_name,
                debris_name,
                force_model,
                optimization_method,
                rdv_strategy,
                waypoint_id + 1,
                loop_mode=loop_mode,
            )

            # Update the initial info for the next leg
            rdv_duration += leg_duration
            leg_start = leg_start + leg_duration
            initial_state_waypoint = (
                state_waypoint_df[state_waypoint_df["epoch"] == leg_start]
                .iloc[0, 2:]
                .to_numpy(dtype=np.float64)
            )

            # Update the maneuvers
            if waypoint_id == 0:
                maneuvers = leg_maneuvers.copy()
            else:
                old_deltav_2 = maneuvers.loc[
                    maneuvers["name"] == f"deltav_{waypoint_id + 1}", "deltav_inertial"
                ].item()
                new_deltav_1 = leg_maneuvers.loc[
                    leg_maneuvers["name"] == "deltav_1", "deltav_inertial"
                ].item()
                combined_maneuver = old_deltav_2 + new_deltav_1

                row = {
                    "name": maneuvers.iloc[waypoint_id, 0],
                    "time": maneuvers.iloc[waypoint_id, 1],
                    "epoch": maneuvers.iloc[waypoint_id, 2],
                    "deltav_inertial": [combined_maneuver],
                    "deltav_norm": np.linalg.norm(combined_maneuver),
                }
                maneuvers.drop([waypoint_id], inplace=True)
                maneuvers = pd.concat((maneuvers, pd.DataFrame(row)), ignore_index=True)
                leg_maneuvers.iloc[-1, 0] = f"deltav_{waypoint_id + 2}"
                maneuvers = pd.concat(
                    [maneuvers, leg_maneuvers.tail(1)], ignore_index=True
                )
    if skip_propagation:
        return maneuvers, None
    trajectory_df = full_propagation(
        initial_state_spacecraft,
        initial_state_debris,
        start_epoch,
        end_epoch,
        maneuvers,
        force_model,
    )

    return maneuvers, trajectory_df


def optimal_transfer(
    initial_state_spacecraft,
    target_states_df,
    start_epoch,
    end_epoch,
    mission_name,
    debris_name,
    force_model,
    method,
    rdv_strategy,
    leg_number,
    loop_mode=False,
):
    """
    Computes the optimal transfer from the initial state of the spacecraft to the target for a range of transfer times.

    Parameters
    ----------
    initial_state_spacecraft : np.ndarray
        Initial state of the spacecraft in the inertial frame
    target_states_df : pandas.DataFrame
        DataFrame containing the debris states
    start_epoch : float
        Start epoch of the transfer
    end_epoch : float
        End epoch of the transfer
    mission_name : str
        Name of the mission
    debris_name : str
        Name of the debris
    force_model : str
        Name of the force model to use for the propagation
    method : str
        Method used to compute the transfer (lambert or perturbed)
    rdv_strategy : str
        Rendezvous strategy (only used to export optimization plots in the correct folder)
    leg_number : int
        Leg number (only used to export optimization plots in the correct folder)
    loop_mode : bool
        If True, no text output is printed and the perturbed progress bar position is 1

    Returns
    -------
    optimal_transfer : pandas.DataFrame
        DataFrame containing the optimal transfer maneuvers

    """
    semi_major_axis = cartesian_to_keplerian(initial_state_spacecraft, MU_EARTH)[0]
    orbit_period = 2 * np.pi * np.sqrt(semi_major_axis**3 / MU_EARTH)
    transfer_max_duration = min(1.05 * orbit_period, end_epoch - start_epoch)
    transfer_time_range = [10, transfer_max_duration]
    dt_vec = np.arange(transfer_time_range[0], transfer_time_range[1], 10)
    deltav_tot_vec = []

    if not loop_mode:
        print("-----------------------------------")
        print(
            f"Starting delta-v optimization of leg {leg_number} with the {method} method..."
        )
    dt_best = None
    deltav_best = None
    maneuvers_best = None
    state_target_best = None
    progress_bar_position = 1
    progress_bar_keep = False
    if method == "perturbed":
        disable_progress_bar = False
        if not loop_mode:
            print(
                "⚠️ Computation is expected to slow down as the transfer duration increases."
            )
            progress_bar_position = 0
            progress_bar_keep = True
    else:
        disable_progress_bar = True
    duration_scan = tqdm(
        dt_vec,
        ncols=80,
        disable=disable_progress_bar,
        position=progress_bar_position,
        leave=progress_bar_keep,
        desc=f"Leg {leg_number}",
    )
    for dt in duration_scan:
        state_target = (
            target_states_df[target_states_df["epoch"] == start_epoch + dt]
            .iloc[0, 2:]
            .to_numpy(dtype=np.float64)
        )
        maneuvers, dv_tot = lambert_transfer(
            initial_state_spacecraft, state_target, start_epoch, dt
        )
        if method == "perturbed":
            maneuvers, dv_tot = perturbed_transfer(
                initial_state_spacecraft,
                state_target,
                start_epoch,
                dt,
                maneuvers,
                force_model,
            )

        if dv_tot is not None:
            deltav_tot_vec.append(dv_tot)
            if deltav_best is None or dv_tot < deltav_best:
                dt_best = dt
                deltav_best = dv_tot
                maneuvers_best = maneuvers.copy()
                state_target_best = state_target.copy()
        else:
            deltav_tot_vec.append(np.nan)

    if not loop_mode:
        print(f"Best transfer duration: {dt_best} s")
        print(f"Best delta-v: {deltav_best} m/s")
    duration_scan.close()

    plt.figure()
    plt.plot(dt_vec / 60, deltav_tot_vec)
    plt.scatter(dt_best / 60, deltav_best, color="red")
    plt.xlabel("Transfer time [min]")
    plt.ylabel("Total delta-v [m/s]")
    plt.yscale("log")
    plt.title(f"Delta-v optimization of leg {leg_number} using the {method} method.")
    plt.tight_layout()
    plt.savefig(
        f"results/{mission_name}/{debris_name}/{rdv_strategy}/"
        f"{force_model}/{method}/deltav_optimization_{leg_number}.png",
        dpi=300,
        facecolor="white",
        transparent=False,
    )
    # For PDF export
    plt.savefig(
        f"results/{mission_name}/{debris_name}/{rdv_strategy}/"
        f"{force_model}/{method}/deltav_optimization_{leg_number}.pdf",
    )
    plt.show()

    return dt_best, maneuvers_best, state_target_best


def transfer_maneuvers(
    initial_state_spacecraft,
    state_target_df,
    start_epoch,
    end_epoch,
    mission_name,
    debris_name,
    force_model,
    optimization_method,
    rdv_strategy,
    leg_number,
    loop_mode=False,
):
    """
    Computes the transfer maneuvers with the given parameters.

    Parameters
    ----------
    initial_state_spacecraft : np.array
        Initial state of the spacecraft
    state_target_df : pd.DataFrame
        DataFrame containing the states of the target
    start_epoch : float
        Start epoch of the transfer
    end_epoch : float
        End epoch of the transfer
    mission_name : str
        Name of the mission
    debris_name : str
        Name of the debris
    force_model : str
        Force model to use for the propagation
    optimization_method : str
        Method to use for the delta-v optimization
    rdv_strategy : str
        Rendezvous strategy (only used to export optimization plots in the correct folder)
    leg_number : int
        Leg number (only used to export optimization plots in the correct folder)
    loop_mode : bool
        If True, no text output is printed and the perturbed progress bar position is 1.

    Returns
    -------
    transfer_duration : float
        Duration of the transfer in seconds
    maneuvers : pd.DataFrame
        DataFrame containing the maneuvers
    """
    transfer_duration, maneuvers, final_state_target = optimal_transfer(
        initial_state_spacecraft,
        state_target_df,
        start_epoch,
        end_epoch,
        mission_name,
        debris_name,
        force_model,
        optimization_method,
        rdv_strategy,
        leg_number,
        loop_mode,
    )

    if optimization_method == "lambert" and force_model != "keplerian":
        # The optimization was done assuming a Keplerian force model, but a perturbed force model is wanted for the
        # real trajectory. Therefore, we need to correct the maneuvers to account for the perturbations.
        maneuvers, _ = perturbed_transfer(
            initial_state_spacecraft,
            final_state_target,
            start_epoch,
            transfer_duration,
            maneuvers,
            force_model,
        )
        if maneuvers is None:
            raise ValueError(
                "The maneuvers could not be corrected for the perturbations."
            )
    return transfer_duration, maneuvers


def lvlh_transformation_matrix(state_debris):
    """
    Compute the transformation matrix from inertial to LVLH frame.
    Parameters
    ----------
    state_debris : np.array
        Inertial state of the debris.

    Returns
    -------
    Q_Xx : np.array
        Transformation matrix from inertial to LVLH frame.

    """
    i_vec = state_debris[0:3] / np.linalg.norm(
        state_debris[0:3]
    )  # unit vector in the direction of the debris
    h_vec = np.cross(state_debris[0:3], state_debris[3:6])  # angular momentum vector
    k_vec = h_vec / np.linalg.norm(
        h_vec
    )  # unit vector in the direction of the angular momentum
    j_vec = np.cross(k_vec, i_vec)  # unit vector to complete the basis
    Q_Xx = np.array([i_vec, j_vec, k_vec])  # rotation matrix from inertial to LVLH
    return Q_Xx


def inertial_to_lvlh(
    state_target: np.ndarray, state_spacecraft: np.ndarray
) -> np.ndarray:
    """
    Transform the inertial state of the target and the spacecraft to the relative state of the spacecraft with respect
    to the target in the LVLH frame.

    From section 7.2 of book "Orbital Mechanics for Engineering Students" by Howard Curtis.

    Parameters
    ----------
        state_target: np.array
            Inertial state of the target.
        state_spacecraft: np.array
            Inertial state of the spacecraft.

    Returns
    -------
        relative_state_lvlh: np.array
            Relative state of the spacecraft with respect to the target.
        Q_Xx: np.array
            Transformation matrix from inertial to LVLH frame.
    """

    Q_Xx = lvlh_transformation_matrix(state_target)
    relative_state_inertial = state_spacecraft - state_target

    relative_position_lvlh = np.matmul(Q_Xx, relative_state_inertial[0:3])
    relative_velocity_lvlh = np.matmul(Q_Xx, relative_state_inertial[3:6])
    relative_state_lvlh = np.concatenate(
        (relative_position_lvlh, relative_velocity_lvlh)
    )

    return relative_state_lvlh


def lvlh_to_inertial(state_debris, relative_state_lvlh):
    """
    Transform the relative state of the spacecraft with respect to the debris in the LVLH frame to the inertial state
    of the debris and the spacecraft.

    From section 7.2 of book "Orbital Mechanics for Engineering Students" by Howard Curtis.

    Parameters
    ----------
        state_debris: np.array
            Inertial state of the debris.
        relative_state_lvlh: np.array
            Relative state of the spacecraft with respect to the debris.

    Returns
    -------
        inertial_state_spacecraft: np.array
            Inertial state of the spacecraft.
    """
    Q_Xx = lvlh_transformation_matrix(state_debris)
    relative_position_inertial = np.matmul(
        np.linalg.inv(Q_Xx), relative_state_lvlh[0:3]
    )
    relative_velocity_inertial = np.matmul(
        np.linalg.inv(Q_Xx), relative_state_lvlh[3:6]
    )
    relative_state_inertial = np.concatenate(
        (relative_position_inertial, relative_velocity_inertial)
    )
    inertial_state_spacecraft = relative_state_inertial + state_debris
    return inertial_state_spacecraft


def export_results(
    trajectory_df,
    maneuvers,
    mission_name,
    debris_name,
    rdv_strategy,
    force_model,
    optimization_method,
    save_pdf=False,
):
    """
    Exports the results of the rendezvous in a csv file and plots the results.

    Parameters
    ----------
    trajectory_df : pd.DataFrame
        DataFrame containing the trajectory of the spacecraft and the debris.
    maneuvers : pd.DataFrame
        DataFrame containing the maneuvers.
    mission_name : str
        Name of the mission.
    debris_name : str
        Name of the debris.
    rdv_strategy : str
        Name of the rendezvous strategy used.
    force_model : str
        Name of the force model used.
    optimization_method : str
        Name of the optimization method used.
    save_pdf : bool
        If True, the plots are also saved as pdf files.
    """
    print("-----------------------------------")
    print("Exporting results...")

    results_path = f"results/{mission_name}/{debris_name}/{rdv_strategy}/{force_model}/{optimization_method}"

    trajectory_df.to_csv(f"{results_path}/trajectory.csv")
    maneuvers.to_csv(f"{results_path}/maneuvers.csv")

    plt.figure()
    plt.plot(
        trajectory_df["timedelta"].dt.total_seconds() / 3600,
        trajectory_df["x_rel"],
        label="x_rel",
    )
    plt.plot(
        trajectory_df["timedelta"].dt.total_seconds() / 3600,
        trajectory_df["y_rel"],
        label="y_rel",
    )
    plt.plot(
        trajectory_df["timedelta"].dt.total_seconds() / 3600,
        trajectory_df["z_rel"],
        label="z_rel",
    )
    plt.legend()
    plt.xlabel("Time [h]")
    plt.ylabel("Relative position [m]")
    plt.tight_layout()
    plt.savefig(
        f"{results_path}/relative_position.png",
        dpi=300,
        facecolor="white",
        transparent=False,
    )
    if save_pdf:
        plt.savefig(f"{results_path}/relative_position.pdf")
    plt.show()

    plt.figure()
    plt.plot(
        trajectory_df["timedelta"].dt.total_seconds() / 3600,
        trajectory_df["vx_rel"],
        label="vx_rel",
    )
    plt.plot(
        trajectory_df["timedelta"].dt.total_seconds() / 3600,
        trajectory_df["vy_rel"],
        label="vy_rel",
    )
    plt.plot(
        trajectory_df["timedelta"].dt.total_seconds() / 3600,
        trajectory_df["vz_rel"],
        label="vz_rel",
    )
    plt.legend()
    plt.xlabel("Time [h]")
    plt.ylabel("Relative velocity [m/s]")
    plt.tight_layout()
    plt.savefig(
        f"{results_path}/relative_velocity.png",
        dpi=300,
        facecolor="white",
        transparent=False,
    )
    if save_pdf:
        plt.savefig(f"{results_path}/relative_velocity.pdf")
    plt.show()

    plt.figure()
    plt.plot(
        trajectory_df["timedelta"].dt.total_seconds() / 3600,
        trajectory_df["x_sat"] - trajectory_df["x_debris"],
        label="x_rel_inertial",
    )
    plt.plot(
        trajectory_df["timedelta"].dt.total_seconds() / 3600,
        trajectory_df["y_sat"] - trajectory_df["y_debris"],
        label="y_rel_inertial",
    )
    plt.plot(
        trajectory_df["timedelta"].dt.total_seconds() / 3600,
        trajectory_df["z_sat"] - trajectory_df["z_debris"],
        label="z_rel_inertial",
    )
    plt.legend()
    plt.xlabel("Time [h]")
    plt.ylabel("Relative position [m]")
    plt.tight_layout()
    plt.savefig(
        f"{results_path}/relative_position_inertial.png",
        dpi=300,
        facecolor="white",
        transparent=False,
    )
    if save_pdf:
        plt.savefig(f"{results_path}/relative_position_inertial.pdf")
    plt.show()

    plt.figure()
    for leg in enumerate(trajectory_df.leg.dropna().unique()):
        plt.plot(
            trajectory_df.loc[trajectory_df["leg"] == leg[1], "y_rel"] / 1e3,
            trajectory_df.loc[trajectory_df["leg"] == leg[1], "x_rel"] / 1e3,
            color=colors[leg[0]],
            label=f"leg {int(leg[1])}",
        )
    plt.xlabel("Relative horizontal position [km]")
    plt.ylabel("Relative vertical position [km]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        f"{results_path}/relative_position_lvlh.png",
        dpi=300,
        facecolor="white",
        transparent=False,
    )
    if save_pdf:
        plt.savefig(f"{results_path}/relative_position_lvlh.pdf")
    plt.show()

    if len(trajectory_df.leg.dropna().unique()) > 2:
        plt.figure()
        for leg in enumerate(trajectory_df.leg.dropna().unique()):
            plt.plot(
                trajectory_df.loc[trajectory_df["leg"] == leg[1], "y_rel"] / 1e3,
                trajectory_df.loc[trajectory_df["leg"] == leg[1], "x_rel"] / 1e3,
                color=colors[leg[0]],
                label=f"leg {int(leg[1])}",
            )
        ax = plt.gca()
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        plt.xlabel("Relative horizontal position (RHP) [km]")
        plt.ylabel("Relative vertical position (RVP) [km]")
        plt.legend(loc="lower right")
        plt.tight_layout()

        # location for the zoomed portion
        axins = plt.axes([0.23, 0.7, 0.25, 0.25])

        for leg in enumerate(trajectory_df.leg.dropna().unique()):
            axins.plot(
                trajectory_df.loc[trajectory_df["leg"] == leg[1], "y_rel"],
                trajectory_df.loc[trajectory_df["leg"] == leg[1], "x_rel"],
                color=colors[leg[0]],
                label=f"leg {int(leg[1])}",
            )

        axins.set_xlabel("RHP [m]")
        axins.set_ylabel("RVP [m]")
        axins.set_xlim(-500, 500)
        axins.set_ylim(-200, 200)
        axins_fake = plt.axes([0.23, 0.7, 0.25, 0.25])
        axins_fake.set_xlim(axins.get_xlim()[0] / 1e3, axins.get_xlim()[1] / 1e3)
        axins_fake.set_ylim(axins.get_ylim()[0] / 1e3, axins.get_ylim()[1] / 1e3)
        axins_fake.patch.set_alpha(0)
        axins_fake.set_xticklabels([])
        axins_fake.set_yticklabels([])
        mark_inset(ax, axins_fake, loc1=1, loc2=4, fc="none", ec="0.5")
        plt.savefig(
            f"{results_path}/relative_position_lvlh_zoom.png",
            dpi=300,
            facecolor="white",
            transparent=False,
        )
        if save_pdf:
            plt.savefig(f"{results_path}/relative_position_lvlh_zoom.pdf")
        plt.show()

    plt.figure()
    plt.plot(
        trajectory_df["timedelta"].dt.total_seconds() / 3600,
        trajectory_df["vx_sat"] - trajectory_df["vx_debris"],
        label="vx_rel_inertial",
    )
    plt.plot(
        trajectory_df["timedelta"].dt.total_seconds() / 3600,
        trajectory_df["vy_sat"] - trajectory_df["vy_debris"],
        label="vy_rel_inertial",
    )
    plt.plot(
        trajectory_df["timedelta"].dt.total_seconds() / 3600,
        trajectory_df["vz_sat"] - trajectory_df["vz_debris"],
        label="vz_rel_inertial",
    )
    plt.legend()
    plt.xlabel("Time [h]")
    plt.ylabel("Relative velocity [m/s]")
    plt.tight_layout()
    plt.savefig(
        f"{results_path}/relative_velocity_inertial.png",
        dpi=300,
        facecolor="white",
        transparent=False,
    )
    if save_pdf:
        plt.savefig(f"{results_path}/relative_velocity_inertial.pdf")
    plt.show()

    plt.figure()
    plt.plot(
        trajectory_df["timedelta"].dt.total_seconds() / 3600,
        trajectory_df["sma_debris"],
        label="sma_debris",
    )
    plt.plot(
        trajectory_df["timedelta"].dt.total_seconds() / 3600,
        trajectory_df["sma_spacecraft"],
        label="sma_spacecraft",
    )
    plt.legend()
    plt.xlabel("Time [h]")
    plt.ylabel("Semi-major axis [m]")
    plt.tight_layout()
    plt.savefig(
        f"{results_path}/sma.png", dpi=300, facecolor="white", transparent=False
    )
    if save_pdf:
        plt.savefig(f"{results_path}/sma.pdf")
    plt.show()

    plt.figure()
    plt.plot(
        trajectory_df["timedelta"].dt.total_seconds() / 3600,
        trajectory_df["ecc_debris"],
        label="ecc_debris",
    )
    plt.plot(
        trajectory_df["timedelta"].dt.total_seconds() / 3600,
        trajectory_df["ecc_spacecraft"],
        label="ecc_spacecraft",
    )
    plt.legend()
    plt.xlabel("Time [h]")
    plt.ylabel("Eccentricity [-]")
    plt.tight_layout()
    plt.savefig(
        f"{results_path}/ecc.png", dpi=300, facecolor="white", transparent=False
    )
    if save_pdf:
        plt.savefig(f"{results_path}/ecc.pdf")
    plt.show()

    print("----------------------------------------")
