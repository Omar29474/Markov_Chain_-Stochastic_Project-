
import numpy as np

def classify_markov_chain(P):

    n = P.shape[0]   # number of states
    d = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if P[i][j] > 0:
                d[i][j] = 1
    n_components, labels = connected_components(d, directed=True, return_labels=True)
    if n_components == 1:
        if is_aperiodic(P):
            return "Irreducible and aperiodic"
        else:
            return "Irreducible and periodic"
    else:
        return "Reducible"

def is_aperiodic(P):
    """
    This function checks if the given Markov chain is aperiodic.
    """
    n = P.shape[0]   # number of states
    for i in range(n):
        if gcd([j for j in range(1, P.shape[0]+1) if P[i,i] > 0 and j*P[i,i] % 1 == 0]) !=1:
            return False
    return True

def classify_state(P, state_id):
    """
    This function classifies the given state based on its properties:
    - Reachable
    - Absorbing
    """
    n = P.shape[0]   # number of states
    if np.all(P[state_id, :] == 0) and P[state_id, state_id] == 1:
        return "Absorbing"
    elif np.any(np.linalg.matrix_power(P, n)[state_id, :] > 0):
        return "Reachable"
    else:
        return "Transient"

def steady_state_prob(P):
    """
    This function computes the steady state probabilities of the given Markov chain.
    """
    n = P.shape[0]   # number of states
    Q = P[np.ix_(range(1, n), range(1, n))]
    I = np.identity(n-1)
    ones = np.ones((n-1, 1))
    A = np.transpose(Q-I)
    A = np.vstack((A, ones.T))
    b = np.zeros(n)
    b[-1] = 1
    pi = np.linalg.lstsq(A, b, rcond=None)[0]
    pi = np.insert(pi, n-1, 1-np.sum(pi))
    return pi

def state_prob_after_n_steps(P, pi0, n, state_id):
    """
    Thisfunction computes the state probabilities after n steps starting from the given initial state.
    """
    pi_n = np.linalg.matrix_power(P, n).dot(pi0)
    return pi_n[state_id]

# Example usage
P = np.array([[0.2, 0.3, 0.5], [0.1, 0.6, 0.3], [0.4, 0.1, 0.5]])
pi0 = np.array([0.3, 0.4, 0.3])
n = 10
state_id = 1

# Classify Markov chain
mc_class = classify_markov_chain(P)
print("Markov Chain Class: " + mc_class)

# Classify State
state_class = classify_state(P, state_id)
print("State Class: " + state_class)

# Compute steady state probabilities
pi = steady_state_prob(P)
print("Steady State Probabilities: " + str(pi))

# Compute state probabilities after n steps
pi_n = state_prob_after_n_steps(P, pi0, n, state_id)
print("State Probabilities after "+str(n)+" steps: " + str(pi_n))
