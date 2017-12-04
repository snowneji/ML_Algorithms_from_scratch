
'''
Refer to:
http://ai.stanford.edu/~chuongdo/papers/em_tutorial.pdf
https://people.duke.edu/~ccc14/sta-663/EMAlgorithm.html



Pseudocode for EM algorithm:


1. we guess/initialize the Thetas for each distribution,

Repeat the following until converge:

    2. E-step: Assuming thetas are correct, we use Thetas to re-estimate each data point using Thetas as the weights

    3. M-step: After re-estimate each data point using Thetas, now we re-calculate the parameter Thetas based on the re-estimated data from above step

'''






import numpy as np


xs = np.array([(5,5), (9,1), (8,2), (4,6), (7,3)])
thetas = np.array([[0.6, 0.4], [0.5, 0.5]])

tol = 0.01
max_iter = 100

ll_old = 0
for i in range(max_iter):
    ws_A = []
    ws_B = []

    vs_A = []
    vs_B = []

    ll_new = 0

    # E-step: calculate probability distributions over possible completions
    for x in xs:

        # multinomial (binomial) log likelihood
        ll_A = np.sum([x*np.log(thetas[0])])
        ll_B = np.sum([x*np.log(thetas[1])])

        # [EQN 1]
        w_A = np.exp(ll_A)/ (np.exp(ll_A) + np.exp(ll_B))
        w_B = np.exp(ll_B)/ (np.exp(ll_A) + np.exp(ll_B))

        ws_A.append(w_A)
        ws_B.append(w_B)

        # used for calculating theta
        vs_A.append(np.dot(w_A, x))
        vs_B.append(np.dot(w_B, x))

        # update complete log likelihood
        ll_new += w_A * ll_A + w_B * ll_B

    # M-step: update values for parameters given current distribution
    # [EQN 2]
    thetas[0] = np.sum(vs_A, 0)/np.sum(vs_A)
    thetas[1] = np.sum(vs_B, 0)/np.sum(vs_B)
    # print distribution of z for each x and current parameter estimate

    print "Iteration: %d" % (i+1)
    print "theta_A = %.2f, theta_B = %.2f, ll = %.2f" % (thetas[0,0], thetas[1,0], ll_new)

    if np.abs(ll_new - ll_old) < tol:
        break
    ll_old = ll_new

