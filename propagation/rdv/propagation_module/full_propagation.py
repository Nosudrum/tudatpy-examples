from rdv.utils.date_transformations import epoch_to_datetime
from rdv.utils.frame_transformations import inertial_to_lvlh
from rdv.propagation_module.leg_propagation import propagate
import pandas as pd


def full_propagation(
    initial_state_spacecraft,
    initial_state_debris,
    start_epoch,
    end_epoch,
    maneuvers,
    force_model="keplerian",
):
    """
    Propagates the spacecraft and the debris from the start epoch to the end epoch using the maneuvers DataFrame.

    Parameters
    ----------
    initial_state_spacecraft : np.array
        inertial state of the spacecraft.
    initial_state_debris : np.array
        inertial state of the debris.
    start_epoch : float
        start epoch of the propagation.
    end_epoch : float
        End of the propagation.
    force_model : str
        Name of the force model to use for the propagation. Default is "keplerian".
    maneuvers : pd.DataFrame
        DataFrame containing the maneuvers.

    Returns
    -------

    """
    relative_state_lvlh, _ = inertial_to_lvlh(
        initial_state_debris, initial_state_spacecraft
    )

    start_time = epoch_to_datetime(start_epoch)

    trajectory_df = pd.DataFrame(
        data=[
            [start_time]
            + initial_state_debris.tolist()
            + initial_state_spacecraft.tolist()
            + relative_state_lvlh.tolist()
        ],
        columns=[
            "time",
            "x_debris",
            "y_debris",
            "z_debris",
            "vx_debris",
            "vy_debris",
            "vz_debris",
            "x_sat",
            "y_sat",
            "z_sat",
            "vx_sat",
            "vy_sat",
            "vz_sat",
            "x_rel",
            "y_rel",
            "z_rel",
            "vx_rel",
            "vy_rel",
            "vz_rel",
        ],
    )

    for leg in range(1, len(maneuvers) + 1):
        print("-----------------------------------")
        print(f"Starting leg {leg}...")
        if leg < len(maneuvers):
            # Leg lasts until the next maneuver
            leg_duration = (
                maneuvers.iloc[leg]["epoch"] - maneuvers.iloc[leg - 1]["epoch"]
            )
        else:
            # Last leg lasts until the end of the simulation
            leg_duration = end_epoch - maneuvers.iloc[leg - 1]["epoch"]

        if leg > 1:
            # Update the initial state of the debris and spacecraft
            initial_state_debris = trajectory_df.tail(1)[
                [
                    "x_debris",
                    "y_debris",
                    "z_debris",
                    "vx_debris",
                    "vy_debris",
                    "vz_debris",
                ]
            ].values[0]
            initial_state_spacecraft = trajectory_df.tail(1)[
                ["x_sat", "y_sat", "z_sat", "vx_sat", "vy_sat", "vz_sat"]
            ].values[0]
            maneuver_time = trajectory_df.tail(1)["time"].values[0]
        else:
            maneuver_time = start_time

        print(f"Applying maneuver {leg} at {maneuver_time}")

        # Apply the maneuver and propagate the leg
        initial_state_spacecraft[3:6] = (
            initial_state_spacecraft[3:6] + maneuvers.iloc[leg - 1]["deltav_inertial"]
        )
        leg_df = propagate(
            initial_state_spacecraft,
            initial_state_debris,
            maneuvers.iloc[leg - 1]["epoch"],
            leg_duration,
            force_model,
        )
        leg_df["leg"] = leg

        # Add the leg trajectory to the total trajectory dataframe
        trajectory_df = pd.concat((trajectory_df, leg_df), ignore_index=True)

    print("-----------------------------------")
    print("Done with all legs.")
    trajectory_df.insert(
        1, "timedelta", trajectory_df["time"] - trajectory_df["time"][0]
    )

    return trajectory_df
