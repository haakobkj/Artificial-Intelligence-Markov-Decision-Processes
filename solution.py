from select import select
from tracemalloc import reset_peak
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


# Some parts of this code were adapted from COMP3702 Tutorial 7
# solution code ("grid_world_solution.py", available on COMP3702
# Blackboard page, retrieved 17 Sep 2022)


class Solver:

    def __init__(self, environment: Environment):
        self.environment = environment

        #Used in both methods
        self.values = dict()
        self.policy = dict()
        self.converged = False
        self.states = None
        self.max_difference = None

        #Used in Policy Iteration
        self.new_policy = dict()


    # === Value Iteration ==============================================================================================

    def vi_initialise(self):
        """
        Initialise any variables required before the start of Value Iteration.
        """
        frontier = [self.environment.get_init_state()]
        explored = set()

        while frontier:
            current_state = frontier.pop()
            for action in ROBOT_ACTIONS:
                for movement in self.environment.apply_action_noise(action):
                    successor_state = self.environment.apply_dynamics(current_state, movement)[1]
                    if successor_state not in explored:
                        frontier.insert(0, successor_state)
                        explored.add(successor_state)
        self.states = list(explored)

        for state in self.states:
                self.values[state] = 0
                self.policy[state] = FORWARD

                
    def vi_is_converged(self):
        """
        Check if Value Iteration has reached convergence.
        :return: True if converged, False otherwise
        """
    
        if self.max_difference is None:
            return False
        elif self.max_difference < self.environment.epsilon:
            return True
        else:
            return False


    def probability_of_action(self, action):
        """
        Calculating the probabilities of each outcome of an action
        :param: action
        :return: dict with probabilities
        """
        prob_dict = dict()

        if action == FORWARD:
            prob_dict[(FORWARD, FORWARD)] = self.environment.double_move_probs[0] * (1- self.environment.drift_cw_probs[0] - self.environment.drift_ccw_probs[0])
            prob_dict[(SPIN_LEFT, FORWARD)] = self.environment.drift_ccw_probs[0] * (1 - self.environment.double_move_probs[0])
            prob_dict[(SPIN_RIGHT, FORWARD)] = self.environment.drift_cw_probs[0] * (1 - self.environment.double_move_probs[0])
            prob_dict[(SPIN_LEFT, FORWARD, FORWARD)] = self.environment.drift_ccw_probs[0] * self.environment.double_move_probs[0]
            prob_dict[(SPIN_RIGHT, FORWARD, FORWARD)] =  self.environment.drift_ccw_probs[0] * self.environment.double_move_probs[0]
            prob_dict[(FORWARD,)] = 1 - sum(prob_dict.values())
        elif action == REVERSE:
            prob_dict[(REVERSE, REVERSE)] = self.environment.double_move_probs[1] * (1- self.environment.drift_cw_probs[1] - self.environment.drift_ccw_probs[1])
            prob_dict[(SPIN_LEFT, REVERSE)] = self.environment.drift_ccw_probs[1] * (1 - self.environment.double_move_probs[1])
            prob_dict[(SPIN_RIGHT, REVERSE)] = self.environment.drift_cw_probs[1] * (1 - self.environment.double_move_probs[1])
            prob_dict[(SPIN_LEFT, REVERSE, REVERSE)] = self.environment.drift_ccw_probs[1] * self.environment.double_move_probs[1]
            prob_dict[(SPIN_RIGHT, REVERSE, REVERSE)] =  self.environment.drift_ccw_probs[1] * self.environment.double_move_probs[1]
            prob_dict[(REVERSE,)] = 1 - sum(prob_dict.values())
        elif action == SPIN_LEFT:
            prob_dict[(SPIN_LEFT,)] = 1
        elif action == SPIN_RIGHT:
            prob_dict[(SPIN_RIGHT,)] = 1
        
        return prob_dict

    
    def vi_iteration(self):
        """
        Perform a single iteration of Value Iteration (i.e. loop over the state space once).
        """
        updated_values = dict()
        updates_policy = dict()
        for state in self.states:
            value_of_actions = dict()
            for action in ROBOT_ACTIONS:
                value = 0
                for prob_actions, probability in self.probability_of_action(action).items():
                    reward = self.environment.apply_dynamics(state, action)[0]
                    for prob_action in prob_actions:
                        next_state = self.environment.apply_dynamics(state, prob_action)[1]
                    value += probability * (reward + (self.environment.gamma * self.values[next_state]))
                
                value_of_actions[action]  = value if not self.environment.is_solved(state) else 100

            updated_values[state] = max(value_of_actions.values())
            updates_policy[state] = max(value_of_actions, key=value_of_actions.get)

        diff = [abs(self.values[s] - updated_values[s]) for s in self.states]
        self.max_difference = (max(diff))

        self.values = updated_values
        self.policy = updates_policy
    

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
        frontier = [self.environment.get_init_state()]
        explored = set()

        while frontier:
            current_state = frontier.pop()
            for action in ROBOT_ACTIONS:
                for movement in self.environment.apply_action_noise(action):
                    successor_state = self.environment.apply_dynamics(current_state, movement)[1]
                    if successor_state not in explored:
                        frontier.insert(0, successor_state)
                        explored.add(successor_state)
        self.states = list(explored)

        for state in self.states:
            self.values[state] = 0
            self.policy[state] = FORWARD
            self.new_policy[state] = REVERSE


    def pi_is_converged(self):
        """
        Check if Policy Iteration has reached convergence.
        :return: True if converged, False otherwise
        """
        if self.new_policy != self.policy:
            self.policy = self.new_policy
            return False
        
        return True
    
    
    def policy_evaluation(self):
        value_converged = False

        while not value_converged:
            updated_values = dict()
            for state in self.states:
                value = 0
                action = self.policy[state]
                probabilities = self.probability_of_action(action).items()
                reward = self.environment.apply_dynamics(state, action)[0]
                for prob_actions, probability in probabilities:
                    for prob_action in prob_actions:          
                        next_state = self.environment.apply_dynamics(state, prob_action)[1]
                    value += probability * (reward + (self.environment.gamma * self.values[next_state]))

                updated_values[state] = value if not self.environment.is_solved(state) else 100
            diff = [abs(self.values[state] - updated_values[state]) for state in self.states]
            self.max_difference = max(diff)



            if self.max_difference < 10:
                value_converged = True
            
            self.values = updated_values
    

    def policy_improvement(self):
        updated_policy = dict()
        for state in self.states:
            value_of_actions = dict()
            for action in ROBOT_ACTIONS:
                value = 0
                for prob_actions, probability in self.probability_of_action(action).items():
                    reward = self.environment.apply_dynamics(state, action)[0]
                    for prob_action in prob_actions:          
                        next_state = self.environment.apply_dynamics(state, prob_action)[1]
                    value += probability * (reward + (self.environment.gamma * self.values[next_state]))
                
                value_of_actions[action]  = value if not self.environment.is_solved(state) else 100
            updated_policy[state] = max(value_of_actions, key=value_of_actions.get)

        return updated_policy


    def pi_iteration(self):
        """
        Perform a single iteration of Policy Iteration (i.e. perform one step of policy evaluation and one step of
        policy improvement).
        """
        self.policy_evaluation()
        self.new_policy = self.policy_improvement()
        

    def pi_plan_offline(self):
        """
        Plan using Policy Iteration.
        """
        # !!! In order to ensure compatibility with tester, you should not modify this method !!!
        self.pi_initialise()
        while not self.pi_is_converged():
            self.pi_iteration()


    def pi_select_action(self, state: State):
        """
        Retrieve the optimal action for the given state (based on values computed by Value Iteration).
        :param state: the current state
        :return: optimal action for the given state (element of ROBOT_ACTIONS)
        """
        return self.policy[state]
    
