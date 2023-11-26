import numpy as np

class FirInRowGame:
    def __init__(self) -> None:
        self.BOARD_CHANNEL      = 0
        self.INDICATOR_CHANNEL  = 1
        self.INVALID_CHANNEL    = 2

        self.chess_board = np.zeros((3, 15, 15),dtype=np.uint8)
        self.chess_board[self.INDICATOR_CHANNEL] = 1

    def __str__(self):
        separator_line = "  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+"  
        s = f"   A B C D E F G H I J K L M N O \n{separator_line}\n"
        now_down = self.step_log[-1]
        for i in range(15):
            s += f"{i:>2}|"
            for j in range(15):
                if self.chess_board[self.BOARD_CHANNEL, i, j] == 1.0:
                    if i==now_down[0] and j==now_down[1]:
                        s += "\033[31m○\033[0m"
                    else:s+="○"
                elif self.chess_board[self.BOARD_CHANNEL, i, j] == 2.0:
                    if i==now_down[0] and j==now_down[1]:
                        s += "\033[31m●\033[0m"  
                    else:s+="●"
                else:
                    s += " "  
                s += "|"
            s += "\n" + separator_line + "\n"
        return s
    
    def __parser_y__(self,y):
        if type(y) == str:
            return ord(y)-ord("A")
        else:
            return y
        
    def check_chess_valid(self,x,y):
        y = self.__parser_y__(y)
        return self.chess_board[self.BOARD_CHANNEL,x,y]==0 and 15>x>=0 and 15>y>=0
    
    def get_now_player(self):
        return (self.chess_board[self.INDICATOR_CHANNEL] == 1).all()
    
    def down_black_chess(self, x, y):
        y = self.__parser_y__(y)
        self.step_log.append([x,y])
        if not self.check_chess_valid(x,y):
            return False
        self.now_down = [x,y]
        self.chess_board[self.BOARD_CHANNEL, x, y] = 1.0
        self.chess_board[self.INDICATOR_CHANNEL] = 2
        self.chess_board[self.INVALID_CHANNEL, x, y] = 1.0
        return True

    def down_white_chess(self, x, y):
        y = self.__parser_y__(y)
        self.step_log.append([x,y])
        if not self.check_chess_valid(x,y):
            return False
        self.now_down = [x,y]
        self.chess_board[self.BOARD_CHANNEL, x, y] = 2.0
        self.chess_board[self.INDICATOR_CHANNEL] = 1
        self.chess_board[self.INVALID_CHANNEL, x, y] = 1.0
        return True
    
    def check_win(self,x,y):
        if (self.chess_board[self.BOARD_CHANNEL,:,:]!=0).all():return "Draw"
        rowV = self.chess_board[self.BOARD_CHANNEL,x,:]
        colV = self.chess_board[self.BOARD_CHANNEL,:,y]
        diagMain = np.diag(self.chess_board[self.BOARD_CHANNEL],y-x)
        diagVice = np.diag(np.fliplr(self.chess_board[self.BOARD_CHANNEL]),14-x-y)
        vecs = [rowV,colV,diagMain,diagVice]
        for vec in vecs:
            for i in range(0,len(vec)-5):
                if all(vec[i:5+i]==[1,1,1,1,1]):
                    return "Black wins"
                if all(vec[i:5+i]==[2,2,2,2,2]):
                    return "White wins"
                
    def play(self,x,y): 
        if x==y==-1:return True
        is_valide = True
        if self.get_now_player():
            is_valide = self.down_black_chess(x,y)
        else:
            is_valide = self.down_white_chess(x,y)
        return is_valide