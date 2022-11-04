import sys
import time
import numpy as np
from constants import *
from environment import *
from state import State

"""
solution.py

This file is a template you should use to implement your solution.

You should implement each section below which contains a TODO comment.

COMP3702 2022 Assignment 2 Support Code

Last updated by njc 08/09/22
"""


class Solver:

    def __init__(self, environment: Environment):

        self.environment = environment
        self.states = self.bfs()

        self.values = dict()
        self.policy = dict()
        self.values = {state: 0 for state in self.states}
        self.policy = {state: FORWARD for state in self.states}
        self.differences = []
        self.converged = False
        self.gamma = environment.gamma  # usikker om jeg trenger denne
        comb = set()
        for i in ACTION_BASE_COST.values():
            for j in ACTION_PUSH_COST.values():
                comb.add(-i-j)
                comb.add(-i)
                comb.add(-j)
        self.comb = comb

        self.previous_values = None
        self.previous_policy = None
        self.r = None
        self.counter = None
        self.rewards = None
        self.t_model = None
        self.r_model = None
        self.la_policy = None

    # === Value Iteration ==============================================================================================

    def bfs(self):
        states = []
        states.append(self.environment.get_init_state())
        frontier = [self.environment.get_init_state()]
        while len(frontier) > 0:
            current_state = frontier.pop()
            for action in ROBOT_ACTIONS:
                cost, new_state = self.environment.apply_dynamics(
                    current_state, action)
                if new_state not in states and new_state not in frontier:
                    frontier.append(new_state)
            if current_state not in states:
                states.append(current_state)
        return states

    def vi_initialise(self):
        return

    def vi_is_converged(self):
        """
        Check if Value Iteration has reached convergence.
        :return: True if converged, False otherwise
        """
        if len(self.differences) == 0:
            return False
        elif self.differences[-1] < self.environment.epsilon:

            self.converged = True
            return True
        # print(self.differences[-1])
        return False

    def stoch_action(self, action):
        probs = {}
        if action == FORWARD:
            probs[(FORWARD,)] = (1-(self.environment.drift_ccw_probs[FORWARD] +
                                    self.environment.drift_cw_probs[FORWARD]) - self.environment.double_move_probs[FORWARD])
            probs[(SPIN_LEFT, FORWARD)] = self.environment.drift_ccw_probs[FORWARD] * \
                (1-self.environment.double_move_probs[FORWARD])
            probs[(SPIN_RIGHT, FORWARD)] = self.environment.drift_cw_probs[FORWARD] * \
                (1-self.environment.double_move_probs[FORWARD])
            probs[(FORWARD, FORWARD)] = self.environment.double_move_probs[FORWARD]
            probs[(SPIN_LEFT, FORWARD, FORWARD)] = self.environment.drift_ccw_probs[FORWARD] * \
                self.environment.double_move_probs[FORWARD]
            probs[(SPIN_RIGHT, FORWARD, FORWARD)] = self.environment.drift_cw_probs[FORWARD] * \
                self.environment.double_move_probs[FORWARD]
        elif action == REVERSE:
            probs[(REVERSE,)] = (1-(self.environment.drift_ccw_probs[REVERSE] +
                                    self.environment.drift_cw_probs[REVERSE])) * (1-self.environment.double_move_probs[REVERSE])
            probs[(SPIN_LEFT, REVERSE)] = self.environment.drift_ccw_probs[REVERSE] * \
                (1-self.environment.double_move_probs[REVERSE])
            probs[(SPIN_RIGHT, REVERSE)] = self.environment.drift_cw_probs[REVERSE] * \
                (1-self.environment.double_move_probs[REVERSE])
            probs[(REVERSE, REVERSE)] = self.environment.double_move_probs[REVERSE]
            probs[(SPIN_LEFT, REVERSE, REVERSE)] = self.environment.drift_ccw_probs[REVERSE] * \
                self.environment.double_move_probs[REVERSE]
            probs[(SPIN_RIGHT, REVERSE, REVERSE)] = self.environment.drift_cw_probs[REVERSE] * \
                self.environment.double_move_probs[REVERSE]

        elif action == SPIN_LEFT:
            probs[(SPIN_LEFT,)] = 1
        elif action == SPIN_RIGHT:
            probs[(SPIN_RIGHT,)] = 1

        return probs

    def dict_argmax(self, d):
        max_value = max(d.values())
        for k, v in d.items():
            if v == max_value:
                return k

    def vi_iteration(self):
        """
        Perform a single iteration of Value Iteration (i.e. loop over the state space once).
        """
        new_values = dict()
        new_policy = dict()
        for s in self.states:
            action_values = dict()
            for a in ROBOT_ACTIONS:
                total = 0
                s_next = s
                for stoch_actions, p in self.stoch_action(a).items():
                    reward = 0
                    for action in stoch_actions:
                        rev, s_next = self.environment.apply_dynamics(
                            s_next, action)
                        if rev not in self.comb:
                            reward += rev
                    total += p * \
                        (reward + (self.environment.gamma *
                                   self.values[s_next]))

                if self.environment.is_solved(s):
                    action_values[a] = 100
                else:
                    action_values[a] = total
            new_values[s] = max(action_values.values())
            new_policy[s] = self.dict_argmax(action_values)

        differences = []
        differences = [abs(self.values[s] - new_values[s])
                       for s in self.states]
        max_diff = max(differences)
        self.differences.append(max_diff)
        self.values = new_values
        self.policy = new_policy

    def vi_plan_offline(self):
        """
        Plan using Value Iteration.
        """

        # !!! In order to ensure compatibility with tester, you should not modify this method !!!
        self.vi_initialise()
        while not self.vi_is_converged():
            self.vi_iteration()

    def vi_get_state_value(self, state: State):
        """
        Retrieve V(s) for the given state.
        :param state: the current state
        :return: V(s)
        """

        return self.values[state]

    def vi_select_action(self, state: State):
        """
        Retrieve the optimal action for the given state (based on values computed by Value Iteration).
        :param state: the current state
        :return: optimal action for the given state (element of ROBOT_ACTIONS)
        """
        return self.policy[state]

    # === Policy Iteration =============================================================================================

    def pi_initialise(self):
        """
        Initialise any variables required before the start of Policy Iteration.
        """
        start = time.time()

        self.USE_LIN_ALG = True
        self.converged = False

        self.t_model = np.zeros(
            [len(self.states), len(ROBOT_ACTIONS), len(self.states)])  # t_model
        self.r_model = np.zeros(
            [len(self.states), len(ROBOT_ACTIONS)])  # r_model

        for state_index, state in enumerate(self.states):
            for _, action in enumerate(ROBOT_ACTIONS):
                for stoch_actions, p in self.stoch_action(action).items():
                    prob = 0
                    rewardSum1 = 0
                    s_next = state
                    prob += p
                    rewards = []
                    for stoch_action in stoch_actions:
                        rev, s_next = self.environment.apply_dynamics(
                            s_next, stoch_action)
                        rewards.append(rev)

                    rewardSum1 += min(rewards) * p

                    self.t_model[state_index][action][
                        self.states.index(s_next)
                    ] += p
                    self.r_model[state_index][action] += rewardSum1
            if self.environment.is_solved(state):
                self.r_model[state_index][:] = 0
                self.t_model[state_index][:][:] = 0
        end = time.time()

        print(end - start)

    def pi_is_converged(self):
        return self.converged

    def pi_iteration(self):
        self.policy_evaluation()
        self.policy_improvement()

    def pi_plan_offline(self):
        """
        Plan using Policy Iteration.
        """
        # !!! In order to ensure compatibility with tester, you should not modify this method !!!
        self.pi_initialise()
        while not self.pi_is_converged():
            self.pi_iteration()

    def pi_select_action(self, state: State):
        # self.environment.render(state)
        return self.policy[state]

    # === Helper Methods ===============================================================================================

    def policy_evaluation(self):
        state_numbers = np.array(range(len(self.states)))  # state numbers
        t_pi = self.t_model[state_numbers,
                            list(self.policy.values())]  # t-pi
        r_model = self.r_model[state_numbers, list(
            self.policy.values())]
        values = np.linalg.solve(
            np.identity(len(self.states))
            - (self.environment.gamma * t_pi),
            r_model,
        )
        self.values = {s: values[i] for i, s in enumerate(self.states)}

    def policy_improvement(self):
        previous_policy = dict(self.policy)
        new_policy = dict()
        for state in self.values:
            action_values = dict()
            for action in ROBOT_ACTIONS:
                total = 0  # total
                for stoch_actions, p in self.stoch_action(action).items():
                    s_next = state
                    rewards = []
                    for stoch_action in stoch_actions:
                        rev, s_next = self.environment.apply_dynamics(
                            s_next, stoch_action)
                        rewards.append(rev)
                    if self.environment.is_solved(s_next):
                        rewards = [0]
                    total += p * (min(rewards) +
                                  self.environment.gamma * self.values[s_next])

                action_values[action] = total

            if self.environment.is_solved(state):
                action_values[state] = 0
                # self.values[state] = 0.0
            new_policy[state] = self.dict_argmax(action_values)

        if all(new_policy[s] == previous_policy[s] for s in self.states):
            self.converged = True
            state = self.environment.get_init_state()
            self.environment.render(state)
            while True:
                _, new_state = self.environment.apply_dynamics(
                    state, self.policy[state])
                self.environment.render(new_state)
                state = new_state
                if self.environment.is_solved(new_state):
                    break

        self.policy = new_policy
