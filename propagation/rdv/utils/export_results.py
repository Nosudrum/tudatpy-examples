import matplotlib.pyplot as plt
from rdv.utils.plot_functions import colors
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


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
