class TicTacToe:
    def __init__(self):
        self.board=[[' ']*3 for i in range(3)]
        self.curPlayer='O'
        
    def printBoard(self):
        for i in self.board:
            print('|'.join(i))
            print('-'*5)
            
    def isWinner(self,player):
        for i in range(3):
            if all(self.board[i][j]==player for j in range(3)) or all(self.board[j][i]==player for j in range(3)): 
                return True
        if all(self.board[i][i]==player for i in range(3)) or all(self.board[i][2-i]==player for i in range(3)):
            return True
        return False
    def isBoardFull(self):
        return all(self.board[i][j]!=' ' for i in range(3) for j in range(3))
    
    def isGameOver(self):
        if self.isWinner('X'):
            print("AI wins")
            return True
        if self.isWinner('O'):
            print('Player Wins')
            return True
        if self.isBoardFull():
            print("Tied")
            return True
        return False
    
    def getEmptyCells(self):
        return [(i,j) for i in range(3) for j in range(3) if self.board[i][j]==' ']
    
    def minimax(self, d, maximize):
        if self.isWinner('X'):
            return 1
        elif self.isWinner('O'):
            return -1
        elif self.isBoardFull():
            return 0
        
        if maximize:
            max_val=float('-inf')
            for mov in self.getEmptyCells():
                self.board[mov[0]][mov[1]]='X'
                val= self.minimax(d+1,False)
                self.board[mov[0]][mov[1]]=' '
                max_val=max(max_val,val)
            return max_val
        else:
            min_val=float('inf')
            for mov in self.getEmptyCells():
                self.board[mov[0]][mov[1]]='O'
                val=self.minimax(d+1,True)
                self.board[mov[0]][mov[1]]=' '
                min_val=min(min_val, val)
                
            return min_val
        
    def dfs(self):
        bv=float('-inf')
        bm=None
        for mv in self.getEmptyCells():
            self.board[mv[0]][mv[1]]='X'
            val=self.minimax(0,False)
            self.board[mv[0]][mv[1]]=' '
            
            if val>bv:
                bv=val
                bm=mv
                
        self.board[bm[0]][bm[1]]='X'
        print('Ai has moved')
        self.printBoard()
        
        
        
if __name__=="__main__":
    game=TicTacToe()
    
    while not game.isGameOver():
        print("\nCurrent TicTacToe Board is ")
        game.printBoard()
        
        if game.curPlayer=='O':
            row = int(input('Enter the row (0.1,2) :  '))
            col = int(input("Enter the col (0,1,2) :  "))
            if row>=0 and row < 3 and col>=0 and col < 3 and game.board[row][col]==' ':
                game.board[row][col]='O'
            else:
                print('Invaled Entries')
                continue
        else:
            game.dfs()
        game.curPlayer= 'X' if game.curPlayer=='O' else 'O'