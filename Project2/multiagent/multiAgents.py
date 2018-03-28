# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        currentFood = currentGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        score = 0.

        # find closest piece of food
        closest_food_distance = float("inf")
        for food in newFood.asList():
          dist = util.manhattanDistance(newPos, food)
          if dist < closest_food_distance:
            closest_food_distance = dist
            closest_food = food

          if dist == 0:
            break

        # find closest ghost
        closest_ghost_distance = float("inf")
        for ghost in newGhostStates:
          dist = util.manhattanDistance(newPos, ghost.getPosition())

          if dist < closest_ghost_distance:
            closest_ghost_distance = dist
            closest_ghost_state = ghost
        
        # default for when u get eaten
        if closest_ghost_distance == 0:
          return float("-inf")

        # add to score based on how close you are to food.
        score += 150*(1.0/closest_food_distance)

        # remove from score based on how close you are to ghosts.
        score -= 250*(1.0/closest_ghost_distance)

        # big reward for being 1 away from food - how to reward for actually eating the food??        
        if len(currentFood.asList()) > len(newFood.asList()):
          score += 500

        return score

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """

        max_list = list()

        def minimax(gameState, iteration):
            # print "iteration:", iteration

            # stopping conditions
            if iteration >= self.depth*gameState.getNumAgents():
              # print "reached depth"
              return self.evaluationFunction(gameState)
            if gameState.isWin() or gameState.isLose():
              # print "winning/losing state"
              return self.evaluationFunction(gameState)

            test_index = iteration%gameState.getNumAgents()
            
            # we are at max node (pacman)
            if test_index == 0:
              # print "Max Node"
              max_val = float("-inf")

              for action in gameState.getLegalActions(test_index):
                if action == Directions.STOP:
                  continue
                
                successor = gameState.generateSuccessor(test_index, action)
                max_val = max(max_val, minimax(successor, iteration+1))
                
                if iteration == 0:
                  max_list.append(max_val)
                
              return max_val 


            # we are at min node (ghost)
            else:
              # print "Min Node"
              min_val = float("inf")
              for action in gameState.getLegalActions(test_index):
                if action == Directions.STOP:
                  continue
                
                successor = gameState.generateSuccessor(test_index, action)
                
                min_val = min(min_val, minimax(successor, iteration+1))

              return min_val


        output = minimax(gameState, 0)
        # print max_list

        # get all legal actions besides stop
        # get the max value of the max_list (which represents best possible actions you can currently take)
        moves = [action for action in gameState.getLegalActions() if action != 'Stop']
        return moves[max_list.index(max(max_list))]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        max_list = list()

        def alphabeta(gameState, iteration, alpha, beta):
            # stopping conditions
            if iteration >= self.depth*gameState.getNumAgents(): # if reached depth
              return self.evaluationFunction(gameState)
            if gameState.isWin() or gameState.isLose(): 
              return self.evaluationFunction(gameState)

            # index to keep track of which agent we are on
            test_index = iteration%gameState.getNumAgents()
            
            # we are at max node (pacman)
            if test_index == 0:
              max_val = float("-inf")

              # get value of each valid successor
              for action in gameState.getLegalActions(test_index):
                if action == Directions.STOP:
                  continue
                
                successor = gameState.generateSuccessor(test_index, action)
                
                # if this successor has a greater value, it should replace the old max value
                max_val = max(max_val, alphabeta(successor, iteration+1, alpha, beta))
                
                # update alpha if the value returned by this successor is greater than alpha
                if max_val > alpha:
                  alpha = max_val

                # prune - doesn't matter what we return (alpha or beta)
                if alpha > beta:
                  # print "heheheh"
                  break
                
                # if we are back at the top of the recursive tree, append this max val to list
                if iteration == 0:
                  # print "LOL"
                  max_list.append(max_val)

              return max_val

            # we are at min node (ghost)
            else:
              min_val = float("inf")

              # get value of each valid successor
              for action in gameState.getLegalActions(test_index):
                # dont let ghost stop
                if action == Directions.STOP:
                  continue

                successor = gameState.generateSuccessor(test_index, action)
                
                # if this successor has smaller value than others, it should be min value
                min_val = min(min_val, alphabeta(successor, iteration+1, alpha, beta))

                # update beta if the minval is small enough
                if min_val < beta:
                  beta = min_val

                if alpha > beta:
                  break

              return min_val

        output = alphabeta(gameState, 0, float("-inf"), float("inf"))

        # get all legal actions besides stop
        # get the max value of the max_list (which represents best possible actions you can currently take)
        moves = [action for action in gameState.getLegalActions() if action != 'Stop']
        # print moves, len(max_list), max_list.index(max(max_list))
        return moves[max_list.index(max(max_list))]

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        max_list = list()

        def expectimax(gameState, iteration):
            # stopping conditions
            if iteration >= self.depth*gameState.getNumAgents():
              return self.evaluationFunction(gameState)
            if gameState.isWin() or gameState.isLose():
              return self.evaluationFunction(gameState)

            test_index = iteration%gameState.getNumAgents()
            
            # we are at max node (pacman)
            if test_index == 0:
              # print "Max Node"
              max_val = float("-inf")

              for action in gameState.getLegalActions(test_index):
                if action == Directions.STOP:
                  continue
                
                successor = gameState.generateSuccessor(test_index, action)
                max_val = max(max_val, expectimax(successor, iteration+1))
                
                if iteration == 0:
                  max_list.append(max_val)
                
              return max_val 
            # we are at chance node (ghost)
            else:
              min_vals = list()
              for action in gameState.getLegalActions(test_index):
                if action == Directions.STOP:
                  continue
                
                successor = gameState.generateSuccessor(test_index, action)
                
                min_vals.append(expectimax(successor, iteration+1))

              return sum([ float(val)/len(min_vals) for val in min_vals])


        output = expectimax(gameState, 0)
        # print max_list

        # get all legal actions besides stop
        # get the max value of the max_list (which represents best possible actions you can currently take)
        moves = [action for action in gameState.getLegalActions() if action != 'Stop']
        return moves[max_list.index(max(max_list))]


def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: The four features I chose to combine are distance to food, distance to ghosts, 
                    distance to scared ghosts, and distance to pellets. 
    """
    # Useful information you can extract from a GameState (pacman.py)
    currentPos = currentGameState.getPacmanPosition()
    currentFood = currentGameState.getFood()
    currentPellets = currentGameState.getCapsules()
    currentGhostStates = currentGameState.getGhostStates()

    score = currentGameState.getScore()

    # get distance to closest food
    distances = list()
    for food in currentFood.asList():
      dist = util.manhattanDistance(currentPos, food)
      distances.append(dist)

    if len(distances) != 0:
      closest_food_distance = min(distances)
    else:
      closest_food_distance = 0

    # get distance to each scared/regular ghost
    distances = list()
    scaredDistances = list()
    for ghost in currentGhostStates:
      dist = util.manhattanDistance(currentPos, ghost.getPosition())

      # prevent divide-by-zero error
      if dist == 0:
        continue

      if ghost.scaredTimer > 0:
        scaredDistances.append(dist)
      else:
        distances.append(dist)

    if len(scaredDistances) != 0:
      closest_scared_distance = min(scaredDistances)
    else:
      closest_scared_distance = 0
    if len(distances) != 0:
      closest_ghost_distance = min(distances)
    else:
      closest_ghost_distance = 0

    # get distance to closest pellet
    distances = list()
    for pellet in currentPellets:
      dist = util.manhattanDistance(currentPos, pellet)
      distances.append(dist)
    
    if len(distances) != 0:
      closest_pellet_distance = min(distances)
    else:
      closest_pellet_distance = 0

    # update score
    if closest_food_distance != 0:
      score += 1.0/closest_food_distance
    if closest_scared_distance != 0:
      score += 45*(1.0/closest_scared_distance)
    if closest_ghost_distance != 0:
      score -= 50*(1.0/closest_ghost_distance)
    if closest_pellet_distance != 0:
      score += 50*(1.0/closest_pellet_distance)

    return score

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
    """
      Your agent for the mini-contest
    """

    def getAction(self, gameState):
        """
          Returns an action.  You can use any method you want and search to any depth you want.
          Just remember that the mini-contest is timed, so you have to trade off speed and computation.

          Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
          just make a beeline straight towards Pacman (or away from him if they're scared!)
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

