import numpy as np

def fl_laws(v, yaw):
    """Feedback linearization laws to compute thrust and desired angles."""
    g = 9.81  # m/s^2
    m = 28e-3  # kg
    
    T = m * np.sqrt(v[0]**2 + v[1]**2 + (v[2] + g)**2)
    phi_d = np.arcsin(m * (v[0] * np.sin(yaw) - v[1] * np.cos(yaw)) / T)
    theta_d = np.arctan2(v[0] * np.cos(yaw) + v[1] * np.sin(yaw), v[2] + g)
    
    return T, phi_d, theta_d

def run_att_PD(state_eta, angle_refs, angular_vel_ref, ref_angle_ddot):
    """
    Computes the torque (tau) using a PD controller for attitude control.

    Parameters:
        state_eta (numpy.ndarray): Current state [angles (3,), angular velocities (3,)].
        angle_refs (numpy.ndarray): Desired angles [phi, theta, psi].
        angular_vel_ref (numpy.ndarray): Reference angular speed.
        ref_angle_ddot (numpy.ndarray): Second derivative of reference angles.

    Returns:
        numpy.ndarray: Computed torque (tau).
    """
    Kp_att = np.diag([40, 40, 40])
    Kd_att = np.diag([20, 20, 20])
    
    I_BF = np.diag([1.4, 1.4, 2.2]) * 1e-5  

    def W(eta):
        return np.array([
            [1, 0, -np.sin(eta[1])], 
            [0, np.cos(eta[0]), np.sin(eta[0]) * np.cos(eta[1])],
            [0, -np.sin(eta[0]), np.cos(eta[0]) * np.cos(eta[1])]
        ])

    def W_dot(eta, rates):
        return np.array([
            [1, 0, -np.cos(eta[1]) * rates[1]],
            [0, -np.sin(eta[0]) * rates[0], np.cos(eta[0]) * rates[0] * np.cos(eta[1]) - np.sin(eta[0]) * np.sin(eta[1]) * rates[1]],
            [0, -np.cos(eta[0]) * rates[0], -np.sin(eta[0]) * rates[0] * np.cos(eta[1]) + np.cos(eta[0]) * np.sin(eta[1]) * rates[1]]
        ])

    e_att = angle_refs - state_eta[:3]
    e_dot_att = angular_vel_ref - state_eta[3:6]

    sigma = ref_angle_ddot + Kp_att @ e_att + Kd_att @ e_dot_att

    angle = state_eta[:3]
    ang_vel = state_eta[3:6]

    tau = I_BF @ W(angle) @ sigma + I_BF @ W_dot(angle, ang_vel) @ ang_vel + np.cross(W(angle) @ ang_vel, I_BF @ W(angle) @ ang_vel)

    return tau, sigma

