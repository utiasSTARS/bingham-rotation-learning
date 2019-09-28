# QUaternion-based Sparse Sum Of Squares relAxation for Robust alignment
from ncpol2sdpa import generate_operators, generate_variables, SdpRelaxation


if __name__=='__main__':
    level = 1
    X = generate_variables('x', 3)

    obj = X[1] - 2 * X[0] * X[1] + X[1] * X[2]
    inequalities = [1 - X[0] ** 2 - X[1] ** 2, 1 - X[1] ** 2 - X[2] ** 2]

    sdp = SdpRelaxation(X)
    sdp.get_relaxation(level, objective=obj, inequalities=inequalities,
                       chordal_extension=True)
    sdp.solve(solver='mosek')
    print(sdp.primal, sdp.dual)
