# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for 
# educational purposes provided that (1) you do not distribute or publish 
# solutions, (2) you retain this notice, and (3) you provide clear 
# attribution to UC Berkeley, including a link to 
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero 
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and 
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        for i in range(iterations):
            # compute new set of values each iteration
            current_values = util.Counter()
            
            for state in mdp.getStates():
                # terminal states have no value
                if self.mdp.isTerminal(state):
                    current_values[state] = 0
                    continue

                # if there are no possible actions, the state is essentially terminal so treat same
                possible_actions = mdp.getPossibleActions(state)
                if not possible_actions:
                    current_values[state] = 0

                # find best action from this state based on q values
                best_action_value = float('-inf')
                for action in possible_actions:
                    q_value = self.getQValue(state, action)

                    if q_value > best_action_value:
                        best_action_value = q_value
                
                current_values[state] = best_action_value
            self.values = current_values

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        next_states = self.mdp.getTransitionStatesAndProbs(state, action)

        q_value = 0.0
        for next_state in next_states:
            transitional_prob = next_state[1]
            reward = self.mdp.getReward(state, action, next_state[0])
            
            q_value += transitional_prob*(reward + self.discount*self.values[next_state[0]])

        return q_value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        possible_actions = self.mdp.getPossibleActions(state)
        max_value = float('-inf')
        best_action = None
        
        if not possible_actions or self.mdp.isTerminal(state):
            return None

        for action in possible_actions:
            val = self.computeQValueFromValues(state, action)

            if val > max_value:
                max_value = val
                best_action = action

        return best_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
