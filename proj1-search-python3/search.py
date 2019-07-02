# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.getStartState())
    print("Is the start a goal?", problem.isGoalState(problem.getStartState()))
    print("Start's successors:", problem.getSuccessors(problem.getStartState()))
    """
    "*** YOUR CODE HERE ***"

    searchTree = util.PriorityQueue() # Data structure to save all states
    startState = problem.getStartState() # Start State
    states = [(startState, "", 0)] #
    priority = 0
    closed = []
    searchTree.push(states, priority)
    while True:
        if searchTree.isEmpty():
            return []
        chosenNode = searchTree.pop() # chosenNode has format [ ((5,4), 'South', cost), ((5,6), 'West', cost) ]. Choose the last node to find successors
        state, direction, _ = chosenNode[-1]
        if problem.isGoalState(state):
            paths = []
            costs = 0
            for node, action, cost in chosenNode:
                if action == "":
                    continue
                paths.append(action)
                costs += cost

            return paths
        else:
            if state not in closed:
                closed.append(state)
                priority -= 1
                for state,direction,cost in problem.getSuccessors(state):
                    appendNode = chosenNode + [(state,direction, cost)]
                    searchTree.push(appendNode, priority)



    util.raiseNotDefined()

def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    searchTree = util.PriorityQueue()  # Data structure to save all states
    startState = problem.getStartState()  # Start State
    states = [(startState, "", 0)]  #
    priority = 0
    closed = []
    searchTree.push(states, priority)
    while True:
        if searchTree.isEmpty():
            return []
        chosenNode = searchTree.pop()  # chosenNode has format [ ((5,4), 'South', cost), ((5,6), 'West', cost) ]. Choose the last node to find successors
        state, direction, _ = chosenNode[-1]
        if problem.isGoalState(state):
            paths = []
            for node, action, _ in chosenNode:
                if action == "":
                    continue
                paths.append(action)
            return paths
        else:
            if state not in closed:
                closed.append(state)
                priority += 1
                for state, direction, cost in problem.getSuccessors(state):
                    appendNode = chosenNode + [(state, direction, cost)]
                    searchTree.push(appendNode, priority)

    util.raiseNotDefined()


    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    searchTree = util.PriorityQueue() # Data structure to save all states
    startState = problem.getStartState() # Start State
    states = [(startState, "", 0)] #
    closed = []
    searchTree.push(states, 0)
    while True:
        if searchTree.isEmpty():
            return []
        chosenNode = searchTree.pop() # chosenNode has format [ ((5,4), 'South', cost), ((5,6), 'West', cost) ]. Choose the last node to find successors
        state, direction, oldCost = chosenNode[-1]
        if problem.isGoalState(state):
            paths = []
            for node, action, cost in chosenNode:
                if action == "":
                    continue
                paths.append(action)
            return paths
        else:
            if state not in closed:
                closed.append(state)
                for state, direction, cost in problem.getSuccessors(state):
                    appendNode = chosenNode + [(state,direction, oldCost+cost)]
                    searchTree.push(appendNode, oldCost+cost)
    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    searchTree = util.PriorityQueue() # Data structure to save all states
    startState = problem.getStartState() # Start State
    states = [(startState, "", 0)] #
    closed = []
    searchTree.push(states, 0)
    while True:
        if searchTree.isEmpty():
            return []
        chosenNode = searchTree.pop() # chosenNode has format [ ((5,4), 'South', cost), ((5,6), 'West', cost) ]. Choose the last node to find successors
        oldState, direction, oldCost = chosenNode[-1]
        if problem.isGoalState(oldState):
            paths = []
            for node, action, _ in chosenNode:
                if action == "":
                    continue
                paths.append(action)
            return paths
        else:
            if oldState not in closed:
                closed.append(oldState)
                for state, direction, cost in problem.getSuccessors(oldState):
                    currentCost = oldCost + cost + heuristic(state, problem) - heuristic(oldState, problem)
                    appendNode = chosenNode + [(state,direction,currentCost)]
                    searchTree.push(appendNode, currentCost)
    util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
