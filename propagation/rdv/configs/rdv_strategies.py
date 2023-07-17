import numpy as np


def rdv_waypoints(strategy_name):
    """
    Returns the waypoints for the given rendezvous strategy.

    Parameters
    ----------
    strategy_name : str
        Name of the rendezvous strategy

    Returns
    -------
    waypoints : np.array
        Waypoints for the given strategy. Each row is an x,y,z coordinate in meters in the target LVLH frame.

    """
    if strategy_name == "test":
        return np.array(
            [[-5000, -20000, 0], [-1000, -5000, 0], [0, -1000, 0], [0, -100, 0]]
        )
    elif strategy_name == "test2":
        return np.array([[-5000, -20000, 0]])
    elif strategy_name == "test3":
        return np.array([[-100, -200, 0]])
    elif strategy_name == "ATV Kepler":
        # ⚠️ "Kepler" refers to the name of a real ATV spacecraft that followed this strategy, it is unrelated to
        # the Keplerian force model.
        return np.array(
            [
                [-5000, -39000, 0],
                [-5000, -15500, 0],
                [0, -3500, 0],
                [0, -249, 0],
                [0, -19, 0],
                [0, -11, 0],
            ]
        )
    else:
        raise ValueError(f"Unknown rendezvous strategy name: {strategy_name}")
