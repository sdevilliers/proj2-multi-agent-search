# multiAgents.py
#
# Submitted by Rayyan u1298801 and Sebastien u1263428
#
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from cmath import inf
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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
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
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        remainingFood = newFood.asList()
        # Calculate the Manhattan distances from Pacman to each remaining food pellet
        foodDistances = [manhattanDistance(newPos, food) for food in remainingFood]
        # Determine the minimum distance to food, or 0 if no food is left
        minFoodDistance = min(foodDistances, default=0)

        # Check if Pacman will collect a power pellet
        if currentGameState.getNumFood() > successorGameState.getNumFood():
            minFoodDistance = -100  # Encourage Pacman to eat pellets

        # Calculate the distance to the closest ghost
        ghostDistances = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]

        # Determine the minimum distance to a ghost, or 0 if no ghosts are nearby
        minGhostDistance = min(ghostDistances, default=0)

        # Avoid getting too close to ghosts
        if minGhostDistance < 2:
            return -float('inf')

        # Calculate the evaluation score
        score = successorGameState.getScore() - minFoodDistance + minGhostDistance

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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        
        def minimax(state, depth, agentIndex):
            if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state), None
            
            # Pacman's turn (max layer)
            if agentIndex == 0:  
                bestValue = -float('inf')
                bestAction = None
                 # Iterate over legal actions for Pacman
                for action in state.getLegalActions(agentIndex):
                    result = minimax(state.generateSuccessor(agentIndex, action), depth, 1)
                    # Extract the value from the result
                    value = result[0] 
                    if value > bestValue:
                        bestValue = value
                        bestAction = action
                return bestValue, bestAction
            
            # Ghosts' turns (min layer)
            else:  
                bestValue = float('inf')
                bestAction = None
                # Determine the index of the next agent
                nextAgentIndex = (agentIndex + 1) % state.getNumAgents()
                
                # Iterate over legal actions for the current ghost
                for action in state.getLegalActions(agentIndex):
                    successorState = state.generateSuccessor(agentIndex, action)
                    # Recursively compute the value of successor states

                    if nextAgentIndex == 0:
                        # If the next agent is Pacman, reduce the depth by 1 (max layer)
                        newDepth = depth - 1
                    else:
                        # If the next agent is a ghost, keep the same depth (min layer)
                        newDepth = depth

                    # Calculate the value of the successor state using minimax
                    value, _ = minimax(successorState, newDepth, nextAgentIndex)

                    # Check if this value is better than the current best value
                    if value < bestValue:
                            bestValue = value
                            bestAction = action
                return bestValue, bestAction
        # Start with Pacman (agentIndex = 0)
        value, action = minimax(gameState, self.depth, 0)  
        return action

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """

        def value(state, depth, agentIndex, alpha = -float('inf'), beta = float('inf')):
            if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state), None
            
            # Pacman's turn (max layer)
            if agentIndex == 0:  
                return maxValue(state, depth, alpha, beta)
            
            # Ghosts' turns (min layer)
            else:  
                return minValue(state, depth, agentIndex, alpha, beta)
        
        def maxValue(state, depth, alpha, beta):
            # We know that the agentIndex is 0 here (PacMan is the only max layer)

            bestVal = -float('inf')
            bestAction = None

            for action in state.getLegalActions(0):
                successor = state.generateSuccessor(0, action)
                nextVal, _ = value(successor, depth, 1, alpha, beta)

                if nextVal > bestVal:
                    bestVal = nextVal
                    bestAction = action

                # check if we can prune
                if bestVal > beta:
                    return bestVal, bestAction
                
                # update minValue pruning constant for next iteration of for loop
                alpha = max(alpha, bestVal)

            return bestVal, bestAction
        
        def minValue(state, depth, agentIndex, alpha, beta):
            bestVal = float('inf')
            bestAction = None

            nextAgent = (agentIndex + 1) % state.getNumAgents()
            if nextAgent == 0:
                depth -= 1

            for action in state.getLegalActions(agentIndex):
                successor = state.generateSuccessor(agentIndex, action)
                nextVal, _ = value(successor, depth, nextAgent, alpha, beta)

                if nextVal < bestVal:
                    bestVal = nextVal
                    bestAction = action

                # check if we can prune
                if bestVal < alpha:
                    return bestVal, bestAction
                
                # update maxValue pruning constant for next iteration of for loop
                beta = min(beta, bestVal)

            return bestVal, bestAction
        
        v, action = value(gameState, self.depth, 0)

        return action

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
        
        def expectimax(state, depth, agentIndex):
            if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state), None
            
            # Pacman's turn (max layer)
            if agentIndex == 0:  
                bestValue = -float('inf')
                bestAction = None
                 # Iterate over legal actions for Pacman
                for action in state.getLegalActions(agentIndex):
                    value, _ = expectimax(state.generateSuccessor(agentIndex, action), depth, 1)

                    if value > bestValue:
                        bestValue = value
                        bestAction = action
                return bestValue, bestAction
            
            # Ghosts' turns (expectation layer)
            else:  
                valueSum = 0
                actionCount = 0

                newDepth = depth

                # Determine the index of the next agent
                nextAgentIndex = (agentIndex + 1) % state.getNumAgents()
                # If the next agent is Pacman, reduce the depth by 1 (max layer)
                if nextAgentIndex == 0:
                    newDepth -= 1
                
                # Iterate over legal actions for the current ghost
                for action in state.getLegalActions(agentIndex):
                    successorState = state.generateSuccessor(agentIndex, action)

                    # Recursively calculate the value of the successor state using expectimax
                    value, _ = expectimax(successorState, newDepth, nextAgentIndex)

                    # Update the action count
                    actionCount += 1
                    # Add value to the sum
                    valueSum += value
                return valueSum/actionCount, None
        
        # Start with Pacman (agentIndex = 0)
        value, action = expectimax(gameState, self.depth, 0)  
        return action

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: 
    Use the game score as a baseline. 
    Stay away from ghosts if a ghost is on top of you (0 dist) score dips by A LOT (100 points). 
    Eat scared ghosts - IMPORTANT
    Get closer to food - weighted slightly higher than staying away from ghosts to prevent stalling
    Reduce food count - weighted pretty high
    """

    pos = currentGameState.getPacmanPosition()
    foods = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()

    minGhostDist = min([manhattanDistance(pos, ghostState.getPosition()) if ghostState.scaredTimer == 0 else float('inf') for ghostState in ghostStates])
    minScaredGhostDist = min([manhattanDistance(pos, ghostState.getPosition()) if ghostState.scaredTimer > 0 else float('inf') for ghostState in ghostStates])
    minFoodDist = min([manhattanDistance(pos, food) for food in foods.asList()]) if foods.count() else float('inf')

    return currentGameState.getScore() + 15/(minFoodDist) - 10*foods.count() - 10/(minGhostDist + 0.1) + (100/minScaredGhostDist)

# Abbreviation
better = betterEvaluationFunction
