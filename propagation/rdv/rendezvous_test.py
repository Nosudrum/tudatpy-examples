from datetime import timedelta, datetime

import numpy as np

from rdv.rdv_module.rendezvous import compute_trajectory
from rdv.utils.export_results import export_results

from tudatpy.kernel.astro.element_conversion import (
    keplerian_to_cartesian,
    mean_motion_to_semi_major_axis,
    mean_to_true_anomaly,
)

earth_gravitational_parameter = 3.986004418e14

iss_sma = mean_motion_to_semi_major_axis(
    15.4990931 * 2 * np.pi / 86400, earth_gravitational_parameter
)
iss_ecc = 0.000498
iss_inc = np.deg2rad(51.6446)
iss_aop = np.deg2rad(225.5107)
iss_raan = np.deg2rad(60.0141)
iss_tan = mean_to_true_anomaly(iss_ecc, np.deg2rad(237.7172))
iss_keplerian = [iss_sma, iss_ecc, iss_inc, iss_aop, iss_raan, iss_tan]

atv_keplerian = iss_keplerian.copy()
atv_keplerian[0] -= 10000
atv_keplerian[-1] -= 50000 / iss_sma

iss_state = keplerian_to_cartesian(iss_keplerian, earth_gravitational_parameter)
atv_state = keplerian_to_cartesian(atv_keplerian, earth_gravitational_parameter)

# Loaded from files in interface.py
start_time = datetime(2020, 1, 1, 0, 0, 0)
state_debris = np.array(
    [
        -1.01723812e06,
        5.23735646e05,
        7.28439472e06,
        -6.60601811e03,
        -3.13522667e03,
        -6.82154831e02,
    ]
)  # [m, m/s]
state_spacecraft = np.array(
    [
        -9.70816528e05,
        5.44376035e05,
        7.27897347e06,
        -6.61726518e03,
        -3.13373143e03,
        -6.33150311e02,
    ]
)  # [m, m/s]

mission_name = "test"
rdv_strategy = "direct"  # 'test, 'test2', 'test3', 'ATV Kepler'
force_model = "J2"  # 'keplerian' or 'J2'
optimization_method = "perturbed"  # 'lambert' or 'perturbed'

transfer_max_duration = timedelta(minutes=100)  # Hard maximum = 7 days

maneuvers, trajectory_df = compute_trajectory(
    atv_state,
    iss_state,
    start_time,
    start_time + transfer_max_duration,
    mission_name,
    'test',
    rdv_strategy,
    force_model,
    optimization_method,
)

export_results(
    trajectory_df,
    maneuvers,
    mission_name,
    'test',
    rdv_strategy,
    force_model,
    optimization_method,
)
