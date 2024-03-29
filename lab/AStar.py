import numpy as np
from queue import PriorityQueue

class State:
    def __init__(self, state, parent):
        self.state=state
        self.parent=parent


    def __lt__(self, other):
        return False
    
class Puzzle:
    def __init__(self, initialState, goalState):
        self.initialState = initialState
        self.goalState = goalState

    def printState(self, state):
        print(state[:,:])

    def isGoalState(self,state):
        return np.array_equal(self.goalState, state)
    
    def heuristic(self, state):
        return np.count_nonzero(state!=self.goalState)
    
    def getPossibleMoves(self, state):
        possMoves=[]
        directions=[(-1,0),(1,0),(0,-1),(0,1)]
        zeroPos=np.where(state==0)

        for d in directions:
            newPos=(zeroPos[0]+d[0], zeroPos[1]+d[1])
            if 0<=newPos[0]<3 and 0<=newPos[1]<3 :
                newState = np.copy(state)
                newState[newPos], newState[zeroPos] = newState[zeroPos], newState[newPos]
                possMoves.append(newState)
        return possMoves
    
    def solve( self):
        pq=PriorityQueue()
        initialState=State(self.initialState, None)
        pq.put((0, initialState))
        vis=set()

        while not pq.empty():
            priority, curState = pq.get()

            if self.isGoalState(curState.state):
                return curState
            
            for move in self.getPossibleMoves(curState.state):
                moveState= State(move, curState)
                if str(move) not in vis:
                    vis.add(str(move))
                    priority = self.heuristic(move) 
                    pq.put((priority, moveState))
        return None
    

initial_state = np.array([[2, 8, 1], [0, 4, 3], [7, 6, 5]])
goal_state = np.array([[1, 2, 3], [8, 0, 4], [7, 6, 5]])
puzzle = Puzzle(initial_state, goal_state)

sol = puzzle.solve()

if sol is not None:
    move=[]
    while sol is not None:
        move.append(sol.state)
        sol = sol.parent
    for mv in reversed(move):
        puzzle.printState(mv)
else : 
    print("No solution found")