import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from tudatpy.kernel.astro.element_conversion import cartesian_to_keplerian

from rdv.rdv_module.lambert_transfer import lambert_transfer
from rdv.rdv_module.perturbed_transfer import perturbed_transfer
from rdv.utils.constants import MU_EARTH


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
