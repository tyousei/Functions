

def bayesian(data, thresh, n):
    '''
    data: predicts of sub-models which is a 2-D list
    thresh: 
    n: number of the sub-models
    '''
    P_x_F = list(map(lambda i: np.exp(-thresh[i] / data[i]), range(n)))
    P_x_N = list(map(lambda i: np.exp(-data[i] / thresh[i]), range(n)))

    p_F_x = list(map(lambda i: P_x_F[i] * 0.01 / (P_x_N[i] * (1-0.01) + P_x_F[i] * 0.01), range(n)))

    sum_P_x_F = np.sum(list(map(lambda i: np.exp(P_x_F[i] * 0.5), range(n))), 0)
    BIC = np.sum(list(map(lambda i: p_F_x[i] * np.exp(P_x_F[i]*0.5) / sum_P_x_F, range(n))), 0)
    return BIC