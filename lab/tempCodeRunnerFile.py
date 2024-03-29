
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