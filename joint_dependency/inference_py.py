# coding: utf-8

from __future__ import division
import numpy as np


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


def likelihood(experiences, alpha_prior):
    """
    Compute the likelihood of a list of experiences for a given Dirichlet
    prior.

    :param experiences: Experiences made so far (dictionary)
    :param alpha_prior: The prior over the different joint states
    :return: The likelihood of the experiences. (float)
    """
    A = np.sum(alpha_prior)
    n = np.zeros((alpha_prior.shape[0], ))
    for e in experiences:
        n[e['value']] += 1
    N = len(experiences)

    #Equation 5 in "Active Exploration of Joint Dependency Structures"
    lnp = gammaln(A) - gammaln(N+A) + np.sum(gammaln(n + alpha_prior) -
                                             gammaln(alpha_prior))
    return np.exp(lnp)


def likelihood_dependent(experiences, possible_master_joint, p_same, alpha_prior):
    """
    Compute the likelihood of the experiences for a specific dependency model.

    :param experiences: Experiences made so far (dictionary)
    :param possible_master_joint: The joint defining the dependency model (i.e.
                            condition on this joint being the (un-) locking
                            joint)
    :param alpha_prior: The prior over the different joint states
    :return: The likelihood of the experiences conditioned on the current joint
             joint being locked by `dependent_joint`
    """
    p = 1.
    for e in experiences:
        buckets = np.array(alpha_prior)
        pos = e['data'][possible_master_joint]
        for e2 in experiences:
            pos2 = e2['data'][possible_master_joint]
            buckets[e2['value']] += p_same[possible_master_joint][pos][pos2]

        print "buckets", buckets
        p *= likelihood([e], buckets)
    return p

def likelihood_dependent_pair(experiences, possible_master_joint_i, possible_master_joint_j, p_same, alpha_prior):
    """
    Compute the likelihood of the experiences for a specific dependency model.

    :param experiences: Experiences made so far (dictionary)
    :param possible_master_joint: The joint defining the dependency model (i.e.
                            condition on this joint being the (un-) locking
                            joint)
    :param alpha_prior: The prior over the different joint states
    :return: The likelihood of the experiences conditioned on the current joint
             joint being locked by `dependent_joint`
    """
    p = 1.
    for e in experiences:
        buckets = np.array(alpha_prior)
        pos_i = e['data'][possible_master_joint_i]
        pos_j = e['data'][possible_master_joint_j]
        for e2 in experiences:
            pos2_i = e2['data'][possible_master_joint_i]
            pos2_j = e2['data'][possible_master_joint_j]
            buckets[e2['value']] += p_same[possible_master_joint_i][pos_i][pos2_i]*\
                                    p_same[possible_master_joint_j][pos_j][pos2_j]
        p *= likelihood([e], buckets)
    return p


def likelihood_independent(experiences, alpha_prior):
    """
    Compute the likelihood of the experiences for the dependency model, where
    no dependency to the current joint exists.

    :param experiences: Experiences made so far (list of dictionaries, each dictionary has 2 entries: 'data'
    that contains the joint states of all the joints in this experience, and 'value' that is true or false
    depending on if this joint was locked or not)
    :param alpha_prior: The prior over the different joint states
    :return: The likelihood of the experiences conditioned on no joint
             dependency
    """
    buckets = np.array(alpha_prior)
    for e in experiences:
        buckets[e['value']] += 1
    return likelihood(experiences, buckets)


def model_posterior(experiences, p_same, alpha_prior, model_prior):
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

    #Note: 1 model is that a joint is locking this joint

    #Number of models is given by the size of the prior over models (it should be num joints + 1 for the independent)
    num_models = model_prior.shape[0]

    #Initialize the vector of likelihoods to have n=numJoints+1 elements
    _likelihood = np.zeros((model_prior.shape[0],))

    #Estimating the likelihood of the independent model (given exps and alpha)
    _likelihood[-1] = model_prior[-1] * likelihood_independent(experiences,
                                                              alpha_prior)

    #Estimating the likelihood that joint i locks this joint (given exps and alpha)
    for possible_master_joint in range(num_models - 1):
        print "likelihood dependent master ", possible_master_joint
        _likelihood[possible_master_joint] = model_prior[possible_master_joint] * \
                                likelihood_dependent(experiences, possible_master_joint,
                                                     p_same, alpha_prior)
    
    p = _likelihood/np.sum(_likelihood)
    return p


def model_posterior_pairs(experiences, p_same, alpha_prior, model_prior, num_joints):
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

    # Note: 1 model is that a joint is locking this joint

    # Number of models is given by the size of the prior over models (it should be num joints + 1 for the independent)
    num_models = model_prior.shape[0]

    # Initialize the vector of likelihoods to have n=numJoints+1 elements
    _model_posterior = np.zeros((model_prior.shape[0],))

    # Estimating the likelihood of the independent model (given exps and alpha)
    _model_posterior[-1] = model_prior[-1] * likelihood_independent(experiences,
                                                               alpha_prior)

    # Estimating the likelihood that joint i locks this joint (given exps and alpha)
    for possible_master_joint in range(num_joints):
        _model_posterior[possible_master_joint] = model_prior[possible_master_joint] * \
                                             likelihood_dependent(experiences, possible_master_joint,
                                                                  p_same, alpha_prior)
    index = num_joints
    for possible_master_joint_i in range(num_joints):
        for possible_master_joint_j in range(possible_master_joint_i+1, num_joints):
            _model_posterior[index] = model_prior[index] * \
                                    likelihood_dependent_pair(experiences, possible_master_joint_i, possible_master_joint_j,
                                                         p_same, alpha_prior)
            index += 1

    p = _model_posterior / np.sum(_model_posterior)
    return p


def create_alpha(current_pos, experiences, joint_idx, p_same):
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
    alpha = np.array([0., 0.])

    for exp_pos, exp_value in [(e['data'][joint_idx], e['value'])
                               for e in experiences]:
        p = p_same[exp_pos][current_pos]
        alpha[exp_value] += p
    return alpha


def prob_locked(experiences, joint_pos, p_same, alpha_prior, model_prior,
                model_post=None):
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
    alpha = np.array(alpha_prior)
    if model_post is None:
        model_post = model_posterior(experiences, p_same, alpha_prior,
                                     model_prior)
    for joint_idx, pos in enumerate(joint_pos):
        c = create_alpha(pos,  experiences,
                         joint_idx,
                         p_same[joint_idx])

        a = model_post[joint_idx] * c
        if np.min(a) < 0:
            print("c = {}".format(c))
            print("mp = {}".format(model_post[joint_idx]))
            print("a = c * mp = {}".format(a))
        alpha += a

    if np.min(alpha) <= 0:
        print("alpha_prior = {}".format(alpha_prior))
        print("alpha = {}".format(alpha))

    d = dirichlet(alpha)
    return d

def prob_locked_with_pairs(experiences, joint_pos, p_same, alpha_prior, model_prior,
                model_post=None):
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
    alpha = np.array(alpha_prior)

    num_joints = model_prior.shape[0]

    if model_post is None:
        model_post = model_posterior_pairs(experiences, p_same, alpha_prior,
                                     model_prior, num_joints)
    for joint_idx, pos in enumerate(joint_pos):
        c = create_alpha(pos,  experiences,
                         joint_idx,
                         p_same[joint_idx])

        a = model_post[joint_idx] * c
        if np.min(a) < 0:
            print("c = {}".format(c))
            print("mp = {}".format(model_post[joint_idx]))
            print("a = c * mp = {}".format(a))
        alpha += a

    if np.min(alpha) <= 0:
        print("alpha_prior = {}".format(alpha_prior))
        print("alpha = {}".format(alpha))

    d = dirichlet(alpha)
    return d


def random_objective(exp, joint_pos, p_same, alpha_prior, model_prior):
    return np.random.uniform()


def exp_cross_entropy(experiences, joint_pos, p_same, alpha_prior, model_prior,
                      model_post=None):
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


def exp_cross_entropy_with_pairs(experiences, joint_pos, p_same, alpha_prior, model_prior,
                      model_post=None):
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
    ce = 0.

    num_joints = model_prior.shape[0]
    if model_post is None:
        model_post = model_posterior_pairs(experiences,
                                     p_same, alpha_prior,
                                     model_prior, num_joints)

    output_likelihood = prob_locked_with_pairs(experiences, joint_pos, p_same,
                                    alpha_prior, model_prior,
                                    model_post=model_post)

    for i, prob in enumerate(output_likelihood.mean()):
        exp = {'data': joint_pos, 'value': i}

        augmented_exp = list(experiences)  # copy the list!
        augmented_exp.append(exp)  # add the 'new' experience

        augmented_post = model_posterior_pairs(augmented_exp, p_same, alpha_prior,
                                         model_prior, num_joints )

        ce += prob * entropy(model_post, augmented_post)
    return ce


def exp_neg_entropy(experiences, joint_pos, p_same, alpha_prior, model_prior):
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
