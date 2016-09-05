# coding: utf-8

from __future__ import division
import numpy as np
cimport numpy as np
DTYPE = np.float32

ctypedef np.float32_t DTYPE_t

from scipy.special import gammaln
from scipy.stats import dirichlet, entropy


def same_segment(probabilities):
    """
    Compute a 2D-Array of probabilities, stating whether two degrees are in the
    same segment or not.

    :param probabilities The change point probabilities for each position
    :return: The probabilities that two joint states are in the same segment
             (I.e. no change points in between)

    """
    cdef int s, t
    pr = np.ones((360, 360))
    for s in range(360):
        for t in range(s+1, 360):
            pr[s, t] = pr[t, s] = pr[s, t-1] * (1-probabilities[t-1])
    return pr

    # p = np.zeros((360,360))
    # for s in range(360):
    #     for t in range(s+1, 360):
    #         p[s][t] = np.prod(1-probabilities[s:t])
    # for s in range(360):
    #     p[s][s] = 1
    #     for t in range(s):
    #         p[s][t] = p[t][s]
    # return p


def likelihood(experiences, np.ndarray[double, ndim=1] alpha_prior):
    """
    Compute the likelihood of a list of experiences for a given Dirichlet
    prior.

    :param experiences: Experiences made so far (dictionary)
    :param alpha_prior: The prior over the different joint states
    :return: The likelihood of the experiences. (float)
    """
    cdef double A, lnp
    cdef int N
    cdef np.ndarray[double, ndim=1] n
    A = np.sum(alpha_prior)
    n = np.zeros((alpha_prior.shape[0], ))
    for e in experiences:
        n[e['value']] += 1
    N = len(experiences)
    lnp = gammaln(A) - gammaln(N+A) + np.sum(gammaln(n + alpha_prior) -
                                             gammaln(alpha_prior))
    return np.exp(lnp)


def likelihood_dependent(experiences, int dependent_joint,
                         np.ndarray[double, ndim=3] p_same,
                         np.ndarray[double, ndim=1] alpha_prior):
    """
    Compute the likelihood of the experiences for a specific dependency model.

    :param experiences: Experiences made so far (dictionary)
    :param dependent_joint: The joint defining the dependency model (i.e.
                            condition on this joint being the (un-) locking
                            joint)
    :param alpha_prior: The prior over the different joint states
    :return: The likelihood of the experiences conditioned on the current joint
             joint being locked by `dependent_joint`
    """
    cdef double p
    cdef int pos, pos2
    cdef np.ndarray[double, ndim=1] buckets
    p = 1.
    for e in experiences:
        buckets = np.array(alpha_prior)
        pos = e['data'][dependent_joint]
        for e2 in experiences:
            pos2 = e2['data'][dependent_joint]
            buckets[e2['value']] += p_same[dependent_joint][pos][pos2]
        p *= likelihood([e], buckets)
    return p


def likelihood_independent(experiences,
                           np.ndarray[double, ndim=1] alpha_prior):
    """
    Compute the likelihood of the experiences for the dependency model, where
    no dependency to the current joint exists.

    :param experiences: Experiences made so far (dictionary)
    :param alpha_prior: The prior over the different joint states
    :return: The likelihood of the experiences conditioned on no joint
             dependency
    """
    cdef np.ndarray[double, ndim=1] buckets = np.array(alpha_prior)
    for e in experiences:
        buckets[e['value']] += 1
    return likelihood(experiences, buckets)


def model_posterior(experiences,
                    np.ndarray[double, ndim=3] p_same,
                    np.ndarray[double, ndim=1] alpha_prior,
                    np.ndarray[double, ndim=1] model_prior):
    """
    Compute the posterior over the different joint dependency models.

    :param experiences: The experiences made so far (dictionary)
    :param p_same: The probabilities of two joint positions being in the same
                   segment. (I.e. no change point in between)
    :param alpha_prior: The prior over the different joint states
    :param model_prior: The prior over the different dependency models
    :return: An array where the each entry gives the probability for the
             according model, where the $n$-th model is that where the $n$-th
             joint (un-) locks the observed joint. The last entry give the
             probability of an independent model.
    """
    cdef int num_models, dep_joint
    num_models = model_prior.shape[0]
    cdef np.ndarray[double, ndim=1] _likelihood = np.zeros((model_prior.shape[0],))
    _likelihood[-1] = model_prior[-1] * likelihood_independent(experiences,
                                                              alpha_prior)
    for dep_joint in range(num_models - 1):
        _likelihood[dep_joint] = model_prior[dep_joint] * \
                                likelihood_dependent(experiences, dep_joint,
                                                     p_same, alpha_prior)
    
    p = _likelihood/np.sum(_likelihood)
    return p


def create_alpha(int current_pos, experiences, int joint_idx,
                 np.ndarray[double, ndim=2] p_same):
    """
    Compute the hyperparameters for a Dirichlet distribution given the
    probabilities of the experiences being tin the same segment. (I.e. no
    change point between the current position and the experience)

    :param current_pos: The current position to compute the hyperparameters for
    :param experiences: The experiences made so far (dictionary)
    :param joint_idx: The joint to compute the hyperparameters for
    :param p_same: The probabilities of two joint states being in the same
                   segment. (I.e. no change point in between)
    :return: A vector holding the weighted counts for each value. To be used as
             hyperparameters of a Dirichlet distribution.
    """
    cdef np.ndarray[double, ndim=1] alpha = np.array([0., 0.])

    cdef int exp_pos
    cdef double exp_value

    for exp_pos, exp_value in [(e['data'][joint_idx], e['value'])
                               for e in experiences]:
        p = p_same[exp_pos][current_pos]
        alpha[exp_value] += p
    return alpha


def prob_locked(experiences, joint_pos, np.ndarray[double, ndim=3] p_same,
                np.ndarray[double, ndim=1] alpha_prior,
                np.ndarray[double, ndim=1] model_prior,
                np.ndarray[double, ndim=1] model_post=None):
    """
    Computes the Dirichlet distribution over the possible joint state
    distributions.

    :param experiences: The experiences so far
    :param joint_pos: The joint positions of all joints (array-like)
    :param p_same: The probability of two joint states being in the same
                   segment. (I.e. no change point in between)
    :param alpha_prior: The prior over the different joint locking states
    :param model_prior: The prior over the different joint dependency models
    :return: A Dirichlet distribution object giving the probability for the
             different locking state distributions
    """
    cdef int joint_idx
    cdef double pos
    cdef np.ndarray[double, ndim=1] alpha = np.array(alpha_prior)
    if model_post is None:
        model_post = model_posterior(experiences, p_same, alpha_prior,
                                     model_prior)
    for joint_idx, pos in enumerate(joint_pos):
        alpha += model_post[joint_idx] * create_alpha(pos,  experiences,
                                                      joint_idx,
                                                      p_same[joint_idx])
    d = dirichlet(alpha)
    return d


def random_objective(exp, joint_pos, p_same, alpha_prior, model_prior, idx_last_successes=[],idx_next_joint=None,idx_last_failures=[], world=None, use_joint_positions=False):
    return np.random.uniform()

def heuristic_proximity(exp, joint_pos, p_same, alpha_prior, model_prior, idx_last_successes=None,idx_next_joint=None,idx_last_failures=None, world=None, use_joint_positions=False):
    if not idx_last_successes:
        return np.random.uniform()

    if not idx_last_failures:
        idx_last_failures = []

    #TODO: Ask Johannes what are his objectives returning for the first action. If returns the same value for all the actions, the maximum will be just the first action of the list (?). Basically, how does he choose the first action?

    if use_joint_positions:
        distance = np.linalg.norm(world.joints[idx_last_successes[-1]].position- world.joints[idx_next_joint].position)
    else:
        distance = abs(idx_last_successes[-1] - idx_next_joint)
        
    
    if not idx_next_joint in idx_last_failures and not idx_next_joint in idx_last_successes:
        return -distance #Distance between next joint and last joint 
    else:
        return -np.inf


def exp_cross_entropy(experiences, joint_pos,
                      np.ndarray[double, ndim=3] p_same,
                      np.ndarray[double, ndim=1] alpha_prior,
                      np.ndarray[double, ndim=1] model_prior,
                      np.ndarray[double, ndim=1] model_post=None, idx_last_successes=[],idx_next_joint=None,idx_last_failures=[], world=None, use_joint_positions=False):
    """
    Compute the expected cross entropy between the current and the augmented
    model posterior, if we would make the next experience at joint_pos.

    :param experiences: The experiences made so far (dictionary)
    :param joint_pos: The positions of all joints (array-like), where the
                    expected cross entropy should be computed.
    :param p_same: The probability of two joint states being in the same
                   segment. (I.e. no change point in between)
    :param alpha_prior: The prior over the different joint locking states
    :param model_prior: The prior over the different joint dependency models
    :return: The expected cross entropy (float)
    """
    cdef int i
    cdef double ce, prob
    ce = 0.

    if model_post is None:
        model_post = model_posterior(experiences,
                                     p_same, alpha_prior,
                                     model_prior)

    output_likelihood = prob_locked(experiences, joint_pos, p_same,
                                    alpha_prior, model_prior,
                                    model_post=model_post)

    for i, prob in enumerate(output_likelihood.mean()):
        exp = {'data': joint_pos, 'value': i}
        
        augmented_exp = list(experiences)  # copy the list!
        augmented_exp.append(exp)  # add the 'new' experience

        augmented_post = model_posterior(augmented_exp, p_same, alpha_prior,
                                         model_prior)

        ce += prob * entropy(model_post, augmented_post)
    return ce


def exp_neg_entropy(experiences, joint_pos, p_same, alpha_prior, model_prior, idx_last_successes=[],idx_next_joint=None,idx_last_failures=[], world=None, use_joint_positions=False):
    ce = 0.

    output_likelihood = prob_locked(experiences, joint_pos, p_same,
                                    alpha_prior, model_prior)

    for i, prob in enumerate(output_likelihood.mean()):
        exp = {'data': joint_pos, 'value': i}

        augmented_exp = list(experiences)  # copy the list!
        augmented_exp.append(exp)  # add the 'new' experience

        augmented_post = model_posterior(augmented_exp, p_same, alpha_prior,
                                         model_prior)

        ce += prob * entropy(augmented_post)
    return -ce
