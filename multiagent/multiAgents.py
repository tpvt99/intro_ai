# multiAgents.py
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


from util import manhattanDistance
from game import Directions
import random, util
import math
from searchAgents import ClosestDotSearchAgent, mazeDistance

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
        newGhostPos = [ghostState.configuration.getPosition() for ghostState in newGhostStates]
        "*** YOUR CODE HERE ***"
        # cost = currentScore + 1.5 * mahDisToClosestFood + (-2) * mahDisToClosetGhost + (-1) * min(scareTimes)
        currentScore = successorGameState.getScore()
        mahDistToClosestFood = math.inf
        for i in range(len(currentGameState.getFood().data)):
            for j in range(len(currentGameState.getFood().data[0])):
                if currentGameState.getFood()[i][j] == True:
                    temp = manhattanDistance((i,j), newPos)
                    if temp < mahDistToClosestFood:
                        mahDistToClosestFood = temp

        mahDisToClosetGhost = min([manhattanDistance(pos, newPos) for pos in newGhostPos])
        return currentScore - 3 * 1/(mahDisToClosetGhost+1e-6) - 1 * mahDistToClosestFood  - 3 * sum([sum(i) for i in newFood.data])
        #return currentScore - 5 * mahDistToClosestFood - 5 * 1/(mahDisToClosetGhost+1e-6) - 1 * len(newScaredTimes)

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
        #self.evaluationFunction = util.lookup('better', globals())
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
        "*** YOUR CODE HERE ***"
        totalAgents = gameState.getNumAgents() # pacman + ghosts

        def max_value(state, depth, agentIndex, actions):
            v = -math.inf
            legalMoves = state.getLegalActions(agentIndex)
            action = actions
            if len(legalMoves) == 0:
                return self.evaluationFunction(state), actions
            for nextAction in legalMoves:
                successorState = state.generateSuccessor(agentIndex, nextAction)
                newV ,newAction = valueEval(successorState, depth, agentIndex+1, actions+[nextAction])
                if newV > v:
                    v = newV
                    action = newAction
            return v, action

        def min_value(state, depth, agentIndex, actions):
            v = math.inf
            legalMoves = state.getLegalActions(agentIndex)
            action = actions
            if len(legalMoves) == 0:
                return self.evaluationFunction(state), actions
            for nextAction in legalMoves:
                successorState = state.generateSuccessor(agentIndex, nextAction)
                if agentIndex < totalAgents-1:
                    newV, newAction = valueEval(successorState, depth, agentIndex+1, actions+[nextAction])
                    if newV < v:
                        v = newV
                        action = newAction
                else:
                    newV, newAction = valueEval(successorState, depth-1, self.index, actions+[nextAction])
                    if newV < v:
                        v = newV
                        action = newAction

            return v, action

        def valueEval(state, depth, agentIndex, actions):
            # if terminal state
            if (depth == 0 and agentIndex == 0) or gameState.isLose() or gameState.isWin():
                return self.evaluationFunction(state), actions
            if agentIndex == 0:
                return max_value(state, depth, self.index, actions)
            else:
                return min_value(state, depth, agentIndex, actions)

        _, actions = valueEval(gameState, self.depth, self.index, [])

        return actions[0]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        totalAgents = gameState.getNumAgents() # pacman + ghosts

        def max_value(state, depth, agentIndex, actions, alpha, beta):
            v = -math.inf
            legalMoves = state.getLegalActions(agentIndex)
            action = actions
            if len(legalMoves) == 0:
                return self.evaluationFunction(state), actions
            for nextAction in legalMoves:
                successorState = state.generateSuccessor(agentIndex, nextAction)
                newV ,newAction = valueEval(successorState, depth, agentIndex+1, actions+[nextAction], alpha, beta)
                if newV > v:
                    v = newV
                    action = newAction
                if v > beta:
                    return v, action
                alpha = max(alpha, v)
            return v, action

        def min_value(state, depth, agentIndex, actions, alpha, beta):
            v = math.inf
            legalMoves = state.getLegalActions(agentIndex)
            action = actions
            if len(legalMoves) == 0:
                return self.evaluationFunction(state), actions
            for nextAction in legalMoves:
                successorState = state.generateSuccessor(agentIndex, nextAction)
                if agentIndex < totalAgents-1:
                    newV, newAction = valueEval(successorState, depth, agentIndex+1, actions+[nextAction], alpha, beta)
                    if newV < v:
                        v = newV
                        action = newAction
                    if v < alpha:
                        return v, action
                    beta = min(beta, v)
                else:
                    newV, newAction = valueEval(successorState, depth-1, self.index, actions+[nextAction], alpha, beta)
                    if newV < v:
                        v = newV
                        action = newAction
                    if v < alpha:
                        return v, action
                    beta = min(beta, v)

            return v, action

        def valueEval(state, depth, agentIndex, actions, alpha, beta):
            # if terminal state
            if (depth == 0 and agentIndex == 0) or gameState.isLose() or gameState.isWin():
                return self.evaluationFunction(state), actions
            if agentIndex == 0:
                return max_value(state, depth, self.index, actions, alpha, beta)
            else:
                return min_value(state, depth, agentIndex, actions, alpha, beta)
        alpha = -math.inf
        beta = math.inf
        _, actions = valueEval(gameState, self.depth, self.index, [], alpha, beta)

        return actions[0]

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
        "*** YOUR CODE HERE ***"
        totalAgents = gameState.getNumAgents()  # pacman + ghosts

        def max_value(state, depth, agentIndex, actions):
            v = -math.inf
            legalMoves = state.getLegalActions(agentIndex)
            action = actions
            if len(legalMoves) == 0:
                return self.evaluationFunction(state), actions
            for nextAction in legalMoves:
                successorState = state.generateSuccessor(agentIndex, nextAction)
                newV, newAction = valueEval(successorState, depth, agentIndex + 1, actions + [nextAction])
                if newV > v:
                    v = newV
                    action = newAction
            return v, action

        def expect_value(state, depth, agentIndex, actions):
            v = 0
            legalMoves = state.getLegalActions(agentIndex)
            action = []
            if len(legalMoves) == 0:
                return self.evaluationFunction(state), actions
            for nextAction in legalMoves:
                successorState = state.generateSuccessor(agentIndex, nextAction)
                if agentIndex < totalAgents - 1:
                    newV, newAction = valueEval(successorState, depth, agentIndex + 1, actions + [nextAction])
                    v += newV
                    action.append(newAction)
                else:
                    newV, newAction = valueEval(successorState, depth - 1, self.index, actions + [nextAction])
                    v += newV
                    action.append(newAction)

            return v/len(legalMoves), random.choice(action)

        def valueEval(state, depth, agentIndex, actions):
            # if terminal state
            if (depth == 0 and agentIndex == 0) or gameState.isLose() or gameState.isWin():
                return self.evaluationFunction(state), actions
            if agentIndex == 0:
                return max_value(state, depth, self.index, actions)
            else:
                return expect_value(state, depth, agentIndex, actions)

        _, actions = valueEval(gameState, self.depth, self.index, [])

        return actions[0]

global minDistance
minDistance = {}


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pacmanPosition = currentGameState.data.agentStates[0].configuration.getPosition()
    ghostStates = currentGameState.getGhostStates()
    ghostPosition = [ghostState.configuration.getPosition() for ghostState in ghostStates]
    foodStates = currentGameState.getFood()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    #distances = ClosestDotSearchAgent().findPathToClosestDot(currentGameState)
    # cost = currentScore + 1.5 * mahDisToClosestFood + (-2) * mahDisToClosetGhost + (-1) * min(scareTimes)
    currentScore = currentGameState.getScore()
    global minDistance
    disToClosestFood = math.inf

    for posx in range(len(foodStates.data)):
        for posy in range(len(foodStates.data[0])):
            if foodStates.data[posx][posy] == True:
                a = (posx, posy, pacmanPosition[0], pacmanPosition[1])
                b = (pacmanPosition[0], pacmanPosition[1], posx, posy)
                if a in minDistance or b in minDistance:
                    temp = minDistance.get(a, 0) or minDistance.get(b, 0)
                else:
                    minDistance[a] = mazeDistance((int(posx), int(posy)),
                                                  (pacmanPosition[0], pacmanPosition[1]), currentGameState)
                    temp = minDistance[a]
                if temp < disToClosestFood:
                    disToClosestFood = temp

    disToClosetGhost = math.inf

    for posx, posy in ghostPosition:
        a = (int(posx), int(posy), pacmanPosition[0], pacmanPosition[1])
        b = (pacmanPosition[0], pacmanPosition[1], int(posx), int(posy))
        if a in minDistance or b in minDistance:
            temp = minDistance.get(a, 0) or minDistance.get(b, 0)
        else:
            minDistance[a] = mazeDistance((int(posx), int(posy)),
                                          (pacmanPosition[0], pacmanPosition[1]), currentGameState)
            temp = minDistance[a]

        if temp < disToClosetGhost:
            disToClosetGhost = temp

    if disToClosestFood == math.inf:
        disToClosestFood = 0

    return currentScore - 3.0 * 1.0 /(disToClosetGhost + 1e-6) \
           - 1.5 * disToClosestFood \
           - 5 * sum([sum(i) for i in foodStates.data])\
           - 2 * len(scaredTimes)


# Abbreviation
better = betterEvaluationFunction
