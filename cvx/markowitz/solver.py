import numpy as np
# --A01--
def zerorows(x, j):
    y = x.copy()
    for k in j:
        y[k] = 0
    return y


# --A02--
def zerocols(x, j):
    y = x.copy()
    for k in j:
        y[:, k] = 0
    return y


# --A03--
def bar(x, j):
    y = x.copy()
    for k in j:
        y[k] = 0
        y[k, k] = 1
    return y


# --A04-- Portfolio Initialization.
def initport(mu, lb, ub):
    x = lb.copy()
    ii = np.argsort(-mu, axis=0)
    amtleft = 1 - np.sum(x)
    ix = 0
    ns = mu.shape[0]
    while ((amtleft > 0) and (ix < ns)):
        i = ii[ix]
        chg = min(ub[i] - lb[i], amtleft)
        x[i] = x[i] + chg
        amtleft = amtleft - chg
        ix += 1
    return (x)


#np.random.seed(0)
#ret = np.random.randn(100, 10)

def cla(mean, lower_bounds, upper_bounds, covariance):
    # --A06-- Set initial parameters.
    ns = mean.shape[0]
    #ns = ret.shape[1]
    lb = lower_bounds.reshape([ns, 1])
    ub = upper_bounds.reshape([ns, 1])

    A = np.ones([1, ns])
    b = 1.0
    m = 1
    tol = 1E-9

    # --A07-- Compute basic statistics of the data.
    #mu = np.mean(ret, axis=0).reshape([ns, 1])
    mu = mean.reshape([ns, 1])

    #C = np.cov(ret.T)
    C = covariance
    # --A08-- Initialize the portfolio.
    x = initport(mu, lb, ub)

    # --A09-- Set the state vector.
    up = 1 * (abs(x - ub) < tol)
    dn = 1 * (abs(x - lb) < tol)
    s = np.subtract(up, dn)

    # --A10-- Set the P matrix.
    P = np.concatenate((C, A.T), axis=1)

    # --A11 -- Initialize storage for quantities # to be computed in the main loop.
    LAM = np.array([np.inf])
    lam = np.inf
    X = x.copy()
    V = np.matmul(x.T, np.matmul(C, x))
    R = np.matmul(mu.T, x)
    S = s.copy()

    # --A12 -- The main CLA loop , which steps
    # from corner portfolio to corner portfolio.

    while lam > 0:

        # --A13-- Create the UP, DN, and IN
        # sets from the current state vector.
        UP = s > +0.9
        DN = s < -0.9
        IN = np.invert(np.logical_or(UP, DN))

        # --A14-- Create the Abar, Cbar, and Mbar matrices.
        io = np.where(np.logical_not(IN))[0]
        Abar = zerocols(A, io)
        Cbar = bar(C, io)
        toprow = np.concatenate((Cbar, Abar.T), axis=1)
        botrow = np.concatenate((Abar, np.zeros([m, m])), axis=1)
        Mbar = np.concatenate([toprow, botrow], axis=0)

        # --A15-- Create the right-hand sides for alpha and beta.
        up = np.multiply(1 * UP, ub)
        dn = np.multiply(1 * DN, lb)
        k = np.add(up, dn)
        bot = b - np.matmul(A, k)
        top = zerorows(mu, io)
        rhsa = np.concatenate([k, bot], axis=0)
        rhsb = np.concatenate([top, np.zeros([m, m])], axis=0)

        # --A16-- Compute alpha, beta, gamma, and delta.
        alpha = np.linalg.solve(Mbar, rhsa)
        beta = np.linalg.solve(Mbar, rhsb)
        gamma = np.matmul(P, alpha)
        delta = np.matmul(P, beta) - mu

        # -A17-- Prepare the ratio matrix.
        L = -np.inf*np.ones([ns, 4])

        # --A18-- IN security possibly going UP.
        for i in np.where(IN & (beta[range(ns)] < -tol))[0]:
            L[i, 0] = (ub[i] - alpha[i]) / beta[i]

        # --A19-- IN security possibly going DN.
        for i in np.where(IN & (beta[range(ns)] > +tol))[0]:
            L[i, 1] = (lb[i] - alpha[i]) / beta[i]

        # --A20-- DN security possibly going IN.
        for i in np.where(UP & (delta < -tol))[0]:
            L[i, 2] = -gamma[i] / delta[i]

        # --A21-- UP security possibly going IN.
        for i in np.where(DN & (delta > +tol))[0]:
            L[i, 3] = -gamma[i] / delta[i]

        # --A22--If all elements of ratio are negative,
        # we have reached the end of the efficient frontier.
        if np.max(L) < 0:
            lam = -np.inf
            break

        # --A23-- Find which security is changing state.
        secmax = np.max(L, axis=1)
        secchg = np.argmax(secmax)

        # --A24-- Find in which direction it is changing.
        dirmax = np.max(L, axis=0)
        dirchg = np.argmax(dirmax)

        # --A25-- Set the new value of lambda_E.
        lam = np.max(secmax)

        # --A26-- Set the state vector for the next segment.
        s[secchg] = (+1 if dirchg == 0 else -1 if dirchg == 1 else 0)

        # --A27-- Compute the portfolio at this corner.
        x = alpha[range(ns)] + lam * beta[range(ns)]
        v = np.matmul(x.T, np.matmul(C, x))
        r = np.matmul(mu.T, x)

        # --A28-- Save the data computed at this corner.
        X = np.concatenate([X, x], axis=1)
        S = np.concatenate([S, s], axis=1)
        V = np.append(V, v)
        R = np.append(R, r)
        LAM = np.append(LAM, np.array([lam]))
