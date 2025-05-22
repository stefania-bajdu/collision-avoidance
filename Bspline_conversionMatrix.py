"""
Created on July 2022
@author: Huu-Thinh DO
"""
import numpy as np
from scipy import interpolate
import warnings


def bsplineConversionMatrices(num_of_controlPoints: int, k: int, knot):
    # num_of_controlPoints = number of control points
    # polynomial order of spline: k
    # knot vector: knot
    n = num_of_controlPoints - 1
    d = k + 1
    tmp = np.eye(n + 1)
    M = []
    for r in range(d):
        M.append(np.zeros((n + r + 1, n + r + 1 + 1)))
        for i in range(n + r + 1):
            if knot[i + d - r - 1] == knot[i]:
                M[r][i][i] = 0
            else:
                M[r][i][i] = (d - r - 1) / (knot[i + d - r - 1] - knot[i])
            if knot[i + d - r] == knot[i + 1]:
                M[r][i][i + 1] = 0
            else:
                M[r][i][i + 1] = -(d - r - 1) / (knot[i + d - r] - knot[i + 1])
        tmp = tmp * np.matrix(M[r])
        M[r] = tmp
    return M


def b_spline_basis_functions(num_of_control_points: int, degree_k: int, knot: int) -> list:
    basis_list = []
    for i in range(degree_k + 1):
        list_tmp = []
        for j in range(i + num_of_control_points):
            ctr_point = [0] * (num_of_control_points + i)
            ctr_point[j] = 1.0
            basis_spln_tmp = interpolate.BSpline(knot, ctr_point, degree_k - i)
            list_tmp.append(basis_spln_tmp)
        basis_list.append(list_tmp)
    return basis_list


def knot_vector(polynomial_degree_k, number_of_control_points, *argv):
    # d = k + 1 (polynomial order +1)
    d = polynomial_degree_k + 1
    # n = number_of_control_points - 1
    n = number_of_control_points - 1
    if number_of_control_points < polynomial_degree_k + 1:
        print('Number of control points: {nc} \nDegree of B-spline: {deg_k}'.format(nc=number_of_control_points,
                                                                                    deg_k=polynomial_degree_k))
        raise Exception("The number of control points need to be higher than or at least equal "
                        "to B-spline degree + 1 ")
    if len(argv) == 0:
        t0 = 0
        tf = 1
    else:
        t0 = min(argv[0])
        tf = max(argv[0])
    knot_tmp = np.linspace(t0, tf, n - d + 3, endpoint=True)
    knot_tmp = np.append([t0] * (d - 1), knot_tmp)
    knot_tmp = np.append(knot_tmp, [tf] * (d - 1))
    return knot_tmp
