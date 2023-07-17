from os import makedirs

import pandas as pd

from rdv.configs.rdv_strategies import rdv_waypoints
from rdv.propagation_module.debris_propagation import propagate_debris
from rdv.propagation_module.full_propagation import full_propagation
from rdv.rdv_module.transfer_optimizer import transfer_maneuvers
from rdv.utils.date_transformations import datetime_to_epoch
from rdv.utils.frame_transformations import *


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
