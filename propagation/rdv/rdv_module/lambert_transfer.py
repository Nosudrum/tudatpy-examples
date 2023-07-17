import numpy as np
import pandas as pd
from tudatpy.kernel.astro import two_body_dynamics

from rdv.utils.constants import MU_EARTH
from rdv.utils.date_transformations import epoch_to_datetime


def lambert_transfer(
    initial_state_spacecraft, final_state_debris, transfer_start, transfer_duration
):
    """
    Computes the transfer from the initial state of the spacecraft to the final state of the debris using the Lambert
    targeter.

    Parameters
    ----------
    initial_state_spacecraft : np.ndarray
        Initial state of the spacecraft in the inertial frame
    final_state_debris : np.ndarray
        Final state of the debris in the inertial frame
    transfer_start : float
        Start epoch of the transfer
    transfer_duration : float
        Duration of the transfer in seconds

    Returns
    -------
    maneuvers : pandas.DataFrame
        DataFrame containing the maneuvers
    deltav_tot : float
        Total delta-v of the transfer in m/s

    """

    # instantiate lambert targeter class using Dario Izzo's algorithm.
    lambert_targeter = two_body_dynamics.LambertTargeterIzzo(
        departure_position=initial_state_spacecraft[0:3],
        arrival_position=final_state_debris[0:3],
        time_of_flight=transfer_duration,
        gravitational_parameter=MU_EARTH,
    )

    v1, v2 = lambert_targeter.get_velocity_vectors()

    delta_v1 = v1 - initial_state_spacecraft[3:6]
    delta_v2 = final_state_debris[3:6] - v2

    delta_v1_norm = np.linalg.norm(delta_v1)
    delta_v2_norm = np.linalg.norm(delta_v2)

    delta_v_total = delta_v1_norm + delta_v2_norm

    if delta_v_total > np.linalg.norm(initial_state_spacecraft[3:6]):
        # instantiate lambert targeter class using Dario Izzo's algorithm.
        lambert_targeter = two_body_dynamics.LambertTargeterIzzo(
            departure_position=initial_state_spacecraft[0:3],
            arrival_position=final_state_debris[0:3],
            time_of_flight=transfer_duration,
            gravitational_parameter=MU_EARTH,
            is_retrograde=True,
        )

        v1, v2 = lambert_targeter.get_velocity_vectors()

        delta_v1 = v1 - initial_state_spacecraft[3:6]
        delta_v2 = final_state_debris[3:6] - v2

        delta_v1_norm = np.linalg.norm(delta_v1)
        delta_v2_norm = np.linalg.norm(delta_v2)

        delta_v_total = delta_v1_norm + delta_v2_norm

    deltav_vec_dict = {
        "name": ["deltav_1", "deltav_2"],
        "time": [
            epoch_to_datetime(transfer_start),
            epoch_to_datetime(transfer_start + transfer_duration),
        ],
        "epoch": [transfer_start, transfer_start + transfer_duration],
        "deltav_inertial": [delta_v1, delta_v2],
        "deltav_norm": [delta_v1_norm, delta_v2_norm],
    }

    return pd.DataFrame(deltav_vec_dict), delta_v_total
