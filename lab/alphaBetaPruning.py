MAX, MIN = 1000, -1000

def minimax( depth, nodeIdx, maximizer, values, alpha, beta):
    if depth==3:
        return values[nodeIdx]
    
    if maximizer:
        best = MIN
        for i in range(2):
            val = minimax(depth+1, nodeIdx*2+i, False, values, alpha, beta)
            best = max( best, val)
            alpha = max( alpha, best)

            if alpha>= beta :
                break
        return best
    else: 
        best = MAX
        for i in range(2):
            val = minimax(depth+1, nodeIdx*2+i, True, values , alpha, beta)
            best = min( best, val)
            beta = min(beta, val)

            if alpha >= beta:
                break
        return best
    

if __name__ == "__main__":
    values = [30,1,6,5,1,2,10,20] 
    print("Maximizing Player : " , minimax(0,0,True, values, MIN, MAX))