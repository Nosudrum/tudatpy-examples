import numpy as np


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
    i_vec = state_debris[0:3] / np.linalg.norm(state_debris[0:3])  # unit vector in the direction of the debris
    h_vec = np.cross(state_debris[0:3], state_debris[3:6])  # angular momentum vector
    k_vec = h_vec / np.linalg.norm(h_vec)  # unit vector in the direction of the angular momentum
    j_vec = np.cross(k_vec, i_vec)  # unit vector to complete the basis
    Q_Xx = np.array([i_vec, j_vec, k_vec])  # rotation matrix from inertial to LVLH
    return Q_Xx


def inertial_to_lvlh(state_debris, state_spacecraft):
    """
    Transform the inertial state of the debris and the spacecraft to the relative state of the spacecraft with respect
    to the debris in the LVLH frame.

    From section 7.2 of book "Orbital Mechanics for Engineering Students" by Howard Curtis.

    Parameters
    ----------
        state_debris: np.array
            Inertial state of the debris.
        state_spacecraft: np.array
            Inertial state of the spacecraft.

    Returns
    -------
        relative_state_lvlh: np.array
            Relative state of the spacecraft with respect to the debris.
        Q_Xx: np.array
            Transformation matrix from inertial to LVLH frame.
    """

    Q_Xx = lvlh_transformation_matrix(state_debris)
    relative_state_inertial = state_spacecraft - state_debris

    relative_position_lvlh = np.matmul(Q_Xx, relative_state_inertial[0:3])
    relative_velocity_lvlh = np.matmul(Q_Xx, relative_state_inertial[3:6])
    relative_state_lvlh = np.concatenate((relative_position_lvlh, relative_velocity_lvlh))

    return relative_state_lvlh, Q_Xx


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
    relative_position_inertial = np.matmul(np.linalg.inv(Q_Xx), relative_state_lvlh[0:3])
    relative_velocity_inertial = np.matmul(np.linalg.inv(Q_Xx), relative_state_lvlh[3:6])
    relative_state_inertial = np.concatenate((relative_position_inertial, relative_velocity_inertial))
    inertial_state_spacecraft = relative_state_inertial + state_debris
    return inertial_state_spacecraft
