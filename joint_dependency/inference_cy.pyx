# coding: utf-8

from __future__ import division
import numpy as np
cimport numpy as np
DTYPE = np.float32

ctypedef np.float32_t DTYPE_t

from scipy.special import gammaln
from scipy.stats import dirichlet, entropy

from joint_dependency.utils import rand_max

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

    #print 'lkd %f' % p
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

    #print 'len %d' % len(experiences)
    for e in experiences:
        buckets[e['value']] += 1

    lk = likelihood(experiences, buckets)
    #print 'lki %f' % lk
    return lk


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
    #print("alpha: %f, %f" % (alpha[0],alpha[1]))
    #print("dirichlet mean: %f" % (d.mean()))
    return d


def random_objective(exp, joint_pos, p_same, alpha_prior, model_prior, model_post=None, idx_last_successes=[],idx_next_joint=None,idx_last_failures=[], world=None, use_joint_positions=False, check_joint=None,
                                   current_pos=None):
    return np.random.uniform()

def heuristic_proximity(exp, joint_pos, p_same, alpha_prior, model_prior, model_post=None, idx_last_successes=None,idx_next_joint=None,idx_last_failures=None, world=None, use_joint_positions=False, check_joint=None,current_pos=None):


    if not idx_last_failures:
        idx_last_failures = []

    if not idx_last_successes:
        if idx_next_joint in idx_last_failures:
            return -np.inf
        else:
            return np.random.uniform()



    #TODO: Ask Johannes what are his objectives returning for the first action. If returns the same value for all the actions, the maximum will be just the first action of the list (?). Basically, how does he choose the first action?

    if use_joint_positions:
        distance = np.linalg.norm(world.joints[idx_last_successes[-1]].position- world.joints[idx_next_joint].position)
    else:
        distance = abs(idx_last_successes[-1] - idx_next_joint)
        
    
    if not idx_next_joint in idx_last_failures and not idx_next_joint in idx_last_successes:
        return -distance #Distance between next joint and last joint
    else:
        return -np.inf

def exp_cross_entropy(experiences, desired_joint_pos,
                      np.ndarray[double, ndim=3] p_same,
                      np.ndarray[double, ndim=1] alpha_prior,
                      np.ndarray[double, ndim=2] model_prior,
                      np.ndarray[double, ndim=1] model_post=None, idx_last_successes=[],idx_next_joint=None,idx_last_failures=[], world=None, use_joint_positions=False,
                      check_joint=None,
                      current_pos=None):
    """
    Compute the expected cross entropy between the current and the augmented
    model posterior, if we would make the next experience at desired_joint_pos.
    :param experiences: The experiences made so far (dictionary)
    :param desired_joint_pos: The positions of all joints (array-like), where the
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

    n_joints = len(experiences)

#    if model_post is None:
#        model_post = model_posterior(experiences[check_joint],
#                                     p_same, alpha_prior,
#                                     model_prior[check_joint])
#
#    #print("alphas of %d" % (idx_next_joint))
#    locked = prob_locked(experiences[idx_next_joint], desired_joint_pos, p_same,
#                                 alpha_prior, model_prior[idx_next_joint],
#                                 model_post=model_post).mean()
#
#
#    #print("alphas of %d" % (check_joint))
#    output_likelihood = prob_locked(experiences[check_joint], desired_joint_pos, p_same,
#                                    alpha_prior, model_prior[check_joint],
#                                    model_post=model_post)
#
#    for i, prob in enumerate(output_likelihood.mean()):
#        exp = {'data': desired_joint_pos, 'value': i}
#
#        augmented_exp = list(experiences[check_joint])  # copy the list!
#        augmented_exp.append(exp)  # add the 'new' experience
#
#        augmented_post = model_posterior(augmented_exp, p_same, alpha_prior,
#                                         model_prior[check_joint])
#
#        # print("{} * H({}, {}) {}".format(prob, model_post, augmented_post,
#        #                                  entropy(model_post, augmented_post)))
#        ce += prob * entropy(model_post, augmented_post)
#    return ce


 #   if model_post is None:
 #       model_post = model_posterior(experiences[check_joint],
 #                                    p_same, alpha_prior,
 #                                    model_prior[check_joint])


    #print("alphas of %d" % (idx_next_joint))
    locked = prob_locked(experiences[idx_next_joint], current_pos, p_same,
                                 alpha_prior, model_prior[idx_next_joint],
                                 model_post=model_post).mean()


    #print("alphas of %d" % (check_joint))

    expected_cross_entropies_for_next_joints=np.zeros(n_joints)

    #Assuming that the joint we want to actuate is locked -> after the actuation the joint configuration is the same (current_pos)
    #it is now
    #Then we augment the experiences of each joint with this virtual configuration and compute the information of testing it

    for j in range(n_joints):

        model_post_j = model_posterior(experiences[j],
                                     p_same, alpha_prior,
                                     model_prior[j])

        p_locked_j = prob_locked(experiences[j], current_pos, p_same,
                                alpha_prior, model_prior[j],
                                model_post=model_post_j)

        for i, prob_j in enumerate(p_locked_j.mean()):
            exp = {'data': current_pos, 'value': i}

            augmented_exp = list(experiences[j])  # copy the list!
            augmented_exp.append(exp)  # add the 'new' experience

            augmented_post_j = model_posterior(augmented_exp, p_same, alpha_prior,
                                             model_prior[j])

            expected_cross_entropies_for_next_joints[j] += locked[0]*prob_j * entropy(model_post_j, augmented_post_j)

    #Assuming that the joint we want to actuate is unlocked -> after the actuation the joint configuration is different (desired_joint_pos)
    #to what it is now
    #Then we augment the experiences of each joint with this virtual configuration and compute the information of testing it
    for j in range(n_joints):

        model_post_j = model_posterior(experiences[j],
                                     p_same, alpha_prior,
                                     model_prior[j])

        p_locked_j = prob_locked(experiences[j], desired_joint_pos, p_same,
                                alpha_prior, model_prior[j],
                                model_post=model_post_j)

        for i, prob_j in enumerate(p_locked_j.mean()):
            exp = {'data': desired_joint_pos, 'value': i}

            augmented_exp = list(experiences[j])  # copy the list!
            augmented_exp.append(exp)  # add the 'new' experience

            augmented_post_j = model_posterior(augmented_exp, p_same, alpha_prior,
                                             model_prior[j])

            expected_cross_entropies_for_next_joints[j] += locked[1]*prob_j * entropy(model_post_j, augmented_post_j)


    value = np.sum(expected_cross_entropies_for_next_joints)

    return value

def one_step_look_ahead_ce(experiences, joint_pos,
                      np.ndarray[double, ndim=3] p_same,
                      np.ndarray[double, ndim=1] alpha_prior,
                      np.ndarray[double, ndim=2] model_prior,
                      np.ndarray[double, ndim=1] model_post=None, idx_last_successes=[],idx_next_joint=None,idx_last_failures=[], world=None, use_joint_positions=False,
                      check_joint=None,
                      current_pos=None):
    """,
    
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

    locked_now = prob_locked(experiences[idx_next_joint], current_pos, p_same,
                                     alpha_prior, model_prior[idx_next_joint],
                                     model_post=model_post).mean()

    ce_return = 0.
    #Compute cross entropy of checking a joint
    model_post_interacted = model_posterior(experiences[idx_next_joint],
                                 p_same, alpha_prior,
                                 model_prior[idx_next_joint])

    exp_locked = {'data': current_pos, 'value': 0}
    augmented_exp_if_locked = list(experiences[idx_next_joint])  # copy the list!
    augmented_exp_if_locked.append(exp_locked)  # add the 'new' experience

    #print 'here3'
    augmented_post_if_locked = model_posterior(augmented_exp_if_locked, p_same, alpha_prior,
                                         model_prior[idx_next_joint])

    ce_locked = locked_now[0] * entropy(model_post_interacted, augmented_post_if_locked)


    exp_locked = {'data': current_pos, 'value': 1}
    augmented_exp_if_unlocked = list(experiences[idx_next_joint])  # copy the list!
    augmented_exp_if_unlocked.append(exp_locked)  # add the 'new' experience

    #print 'here4'
    augmented_post_if_unlocked = model_posterior(augmented_exp_if_unlocked, p_same, alpha_prior,
                                         model_prior[idx_next_joint])
    ce_unlocked = locked_now[1] * entropy(model_post_interacted, augmented_post_if_unlocked)

    ce_return = ce_locked+ce_unlocked

    if ce_return < 1e-10:
        ce_return = np.inf
    #print 'ce_return: %f' % ce_return

    return ce_return


def two_step_look_ahead_ce(experiences, joint_pos,
                      np.ndarray[double, ndim=3] p_same,
                      np.ndarray[double, ndim=1] alpha_prior,
                      np.ndarray[double, ndim=2] model_prior,
                      np.ndarray[double, ndim=1] model_post=None, idx_last_successes=[],idx_next_joint=None,idx_last_failures=[], world=None, use_joint_positions=False,
                      check_joint=None,
                      current_pos=None):
    """,

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
    cdef int second_action
    cdef double ce, prob
    ce = 0.



    locked_now = prob_locked(experiences[idx_next_joint], current_pos, p_same,
                                     alpha_prior, model_prior[idx_next_joint],
                                     model_post=model_post).mean()

    ce_return = 0.
    #Compute cross entropy of checking a joint
    if check_joint is idx_next_joint:
        #print("alphas of %d" % (idx_next_joint))

        #print 'here2'
        model_post_interacted = model_posterior(experiences[idx_next_joint],
                                     p_same, alpha_prior,
                                     model_prior[idx_next_joint])

        exp_locked = {'data': current_pos, 'value': 0}
        augmented_exp_if_locked = list(experiences[idx_next_joint])  # copy the list!
        augmented_exp_if_locked.append(exp_locked)  # add the 'new' experience

        #print 'here3'
        augmented_post_if_locked = model_posterior(augmented_exp_if_locked, p_same, alpha_prior,
                                             model_prior[idx_next_joint])

        ce_all_locked =[]

        for second_action in range(5):

            ce_i = 0
            augmented_exp_if_locked_i = list(experiences[second_action])  # copy the list!
            exp_locked_i = {'data': current_pos, 'value': 0}
            augmented_exp_if_locked_i.append(exp_locked_i)  # add the 'new' experience

            model_post_i = model_posterior(augmented_exp_if_locked_i,
                                         p_same, alpha_prior,
                                         model_prior[second_action])

            #For the case it is unlocked, we would move it and we could check any other joint
            #We add the
            #print("alphas of %d" % (check_joint))
            output_likelihood_i = prob_locked(augmented_exp_if_locked_i, current_pos, p_same,
                                            alpha_prior, model_prior[second_action],
                                            model_post=model_post_i)

            for i, prob in enumerate(output_likelihood_i.mean()):


                exp = {'data': joint_pos, 'value': i}

                augmented_exp = list(augmented_exp_if_locked_i)  # copy the list!
                augmented_exp.append(exp)  # add the 'new' experience

                augmented_post = model_posterior(augmented_exp, p_same, alpha_prior,
                                                 model_prior[second_action])

                ce_i += prob * entropy(model_post_i, augmented_post)

            ce_all_locked.append(ce_i)


        max_second_after_locked = rand_max(ce_all_locked)

        ce_locked = locked_now[0] * (entropy(model_post_interacted, augmented_post_if_locked) + max_second_after_locked)


        exp_locked = {'data': current_pos, 'value': 1}
        augmented_exp_if_unlocked = list(experiences[idx_next_joint])  # copy the list!
        augmented_exp_if_unlocked.append(exp_locked)  # add the 'new' experience

        #print 'here4'
        augmented_post_if_unlocked = model_posterior(augmented_exp_if_unlocked, p_same, alpha_prior,
                                             model_prior[idx_next_joint])

        ce_all_unlocked =[]

        for second_action in range(5):

            ce_i = 0
            augmented_exp_if_unlocked_i = list(experiences[second_action])  # copy the list!
            exp_unlocked_i = {'data': current_pos, 'value': 1}
            augmented_exp_if_unlocked_i.append(exp_unlocked_i)  # add the 'new' experience

            model_post_i = model_posterior(augmented_exp_if_unlocked_i,
                                         p_same, alpha_prior,
                                         model_prior[second_action])

            #For the case it is unlocked, we would move it and we could check any other joint
            #We add the
            #print("alphas of %d" % (check_joint))
            output_likelihood_i = prob_locked(augmented_exp_if_unlocked_i, joint_pos, p_same,
                                            alpha_prior, model_prior[second_action],
                                            model_post=model_post_i)

            for i, prob in enumerate(output_likelihood_i.mean()):


                exp = {'data': joint_pos, 'value': i}

                augmented_exp = list(augmented_exp_if_unlocked_i)  # copy the list!
                augmented_exp.append(exp)  # add the 'new' experience

                augmented_post = model_posterior(augmented_exp, p_same, alpha_prior,
                                                 model_prior[second_action])

                ce_i += prob * entropy(model_post_i, augmented_post)

            ce_all_unlocked.append(ce_i)

        max_second_after_unlocked = rand_max(ce_all_unlocked)

        ce_unlocked = locked_now[1] * (entropy(model_post_interacted, augmented_post_if_unlocked) + max_second_after_unlocked)


        ce_return = ce_locked+ce_unlocked
        #print 'ce_return: %f' % ce_return

    #Compute the cross entropy of checking another joint after moving the joint (weighted by the probability of the joint being unlocked)
    #Should be added to the cross entropy of seing the joint move? yes!
    #print("alphas of %d" % (idx_next_joint))
    else:


        if model_post is None:
            model_post = model_posterior(experiences[check_joint],
                                         p_same, alpha_prior,
                                         model_prior[check_joint])


        exp_locked = {'data': current_pos, 'value': 1}
        augmented_exp_if_unlocked = list(experiences[idx_next_joint])  # copy the list!
        augmented_exp_if_unlocked.append(exp_locked)  # add the 'new' experience
        augmented_post_if_unlocked = model_posterior(experiences[idx_next_joint], p_same, alpha_prior,
                                             model_prior[idx_next_joint])

        model_post_interacted = model_posterior(experiences[idx_next_joint],
                                     p_same, alpha_prior,
                                     model_prior[idx_next_joint])

        ce_unlocked = locked_now[1] * entropy(model_post_interacted, augmented_post_if_unlocked)


        #For the case it is unlocked, we would move it and we could check any other joint
        #We add the
        #print("alphas of %d" % (check_joint))
        output_likelihood = prob_locked(experiences[check_joint], joint_pos, p_same,
                                        alpha_prior, model_prior[check_joint],
                                        model_post=model_post)

        for i, prob in enumerate(output_likelihood.mean()):
            exp = {'data': joint_pos, 'value': i}

            augmented_exp = list(experiences[check_joint])  # copy the list!
            augmented_exp.append(exp)  # add the 'new' experience

            augmented_post = model_posterior(augmented_exp, p_same, alpha_prior,
                                             model_prior[check_joint])

            ce += prob * entropy(model_post, augmented_post)
            #print 'ce: %f' % ce


        #print 'ce_unlocked: %f' % ce_unlocked
        #print 'locked_now[1]: %f' % locked_now[1]
        ce_return = (ce_unlocked + locked_now[1]*ce)
        #print 'ce_return: %f' % ce_return

    return ce_return

def exp_neg_entropy(experiences, 
  joint_pos, 
  p_same, 
  alpha_prior, 
  model_prior, 
  model_post=None, 
  idx_last_successes=[],
  idx_next_joint=None,
  idx_last_failures=[], 
  world=None, 
  use_joint_positions=False,
  check_joint=None, 
  current_pos=None):
    
    locked_now = prob_locked(experiences[idx_next_joint], current_pos, p_same,
                                     alpha_prior, model_prior[idx_next_joint],
                                     model_post=model_post).mean()

    ce_return = 0.
    #Compute cross entropy of checking a joint
    model_post_interacted = model_posterior(experiences[idx_next_joint],
                                 p_same, alpha_prior,
                                 model_prior[idx_next_joint])

    exp_locked = {'data': current_pos, 'value': 0}
    augmented_exp_if_locked = list(experiences[idx_next_joint])  # copy the list!
    augmented_exp_if_locked.append(exp_locked)  # add the 'new' experience

    #print 'here3'
    augmented_post_if_locked = model_posterior(augmented_exp_if_locked, p_same, alpha_prior,
                                         model_prior[idx_next_joint])

    ce_locked = locked_now[0] * entropy(augmented_post_if_locked)


    exp_locked = {'data': current_pos, 'value': 1}
    augmented_exp_if_unlocked = list(experiences[idx_next_joint])  # copy the list!
    augmented_exp_if_unlocked.append(exp_locked)  # add the 'new' experience

    #print 'here4'
    augmented_post_if_unlocked = model_posterior(augmented_exp_if_unlocked, p_same, alpha_prior,
                                         model_prior[idx_next_joint])
    ce_unlocked = locked_now[1] * entropy(augmented_post_if_unlocked)

    ce_return = ce_locked+ce_unlocked
    
    ce_return = -ce_return

    if ce_return < 1e-10:
        ce_return = np.inf

    return ce_return

def exp_neg_entropy_old(experiences, joint_pos, p_same, alpha_prior, model_prior, model_post=None, idx_last_successes=[],idx_next_joint=None,idx_last_failures=[], world=None, use_joint_positions=False, check_joint=None, current_pos=None):
    ce = 0.

    output_likelihood = prob_locked(experiences[check_joint], joint_pos, p_same,
                                    alpha_prior, model_prior[check_joint])

    for i, prob in enumerate(output_likelihood.mean()):
        exp = {'data': joint_pos, 'value': i}

        augmented_exp = list(experiences[check_joint])  # copy the list!
        augmented_exp.append(exp)  # add the 'new' experience

        augmented_post = model_posterior(augmented_exp, p_same, alpha_prior,
                                         model_prior[check_joint])

        ce += prob * entropy(augmented_post)
    return -ce
