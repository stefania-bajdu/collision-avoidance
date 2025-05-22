from Bspline_conversionMatrix import *

spline_degree = 3  # therefore order of the spline = 4
n_ctrl_pts = 8
knot_vec = knot_vector(spline_degree, n_ctrl_pts, [0, 10])
basis_funcs = b_spline_basis_functions(n_ctrl_pts, spline_degree, knot_vec)

print(basis_funcs[0])
# print(len(knot_vec))
# print(len(basis_funcs)) # this is the same as the order
# print(len(basis_funcs[0])) # this is the number of control points
# print(len(basis_funcs[1]))
# print(len(basis_funcs[2]))
# print(len(basis_funcs[3]))
M = bsplineConversionMatrices(n_ctrl_pts, spline_degree, knot_vec)

# print(len(M))
# for i in range(len(M)):
#     print(M[i].shape)