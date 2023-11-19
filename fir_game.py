import numpy as np

class FirInRowGame:
    def __init__(self) -> None:
        self.BOARD_CHANNEL      = 0
        self.INDICATOR_CHANNEL  = 1
        self.INVALID_CHANNEL    = 2

        self.chess_board = np.zeros((3, 15, 15),dtype=np.uint8)

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
        
    def down_black_chess(self, x, y):
        y = self.__parser_y__(y)
        if not self.check_chess_valid(x,y):
            return False
        self.step_log.append([x,y])
        self.now_down = [x,y]
        self.chess_board[self.BOARD_CHANNEL, x, y] = 1.0
        self.chess_board[self.INDICATOR_CHANNEL] = 2
        self.chess_board[self.INVALID_CHANNEL, x, y] = 1.0
        self.now_player = not self.now_player
        return True

    def down_white_chess(self, x, y):
        y = self.__parser_y__(y)
        if not self.check_chess_valid(x,y):
            return False
        self.step_log.append([x,y])
        self.now_down = [x,y]
        self.chess_board[self.BOARD_CHANNEL, x, y] = 2.0
        self.chess_board[self.INDICATOR_CHANNEL] = 1
        self.chess_board[self.INVALID_CHANNEL, x, y] = 1.0
        self.now_player = not self.now_player
        return True
    
    def check_win(self):
        if (self.chess_board[self.BOARD_CHANNEL,:,:]!=0).all():return "Draw"
        def consecutive_pieces(row):
            for i in range(len(row) - 4):
                if row[i:i + 5] == [1.0, 1.0, 1.0, 1.0, 1.0]:
                    return True
                elif row[i:i + 5] == [2.0, 2.0, 2.0, 2.0, 2.0]:
                    return True
            return False
        # Check rows
        for i in range(15):
            if consecutive_pieces(list(self.chess_board[self.BOARD_CHANNEL, i, :])):
                return "Black wins" if 1.0 in list(self.chess_board[self.BOARD_CHANNEL, i, :]) else "White wins"
        # Check columns
        for j in range(15):
            if consecutive_pieces(list(self.chess_board[self.BOARD_CHANNEL, :, j])):
                return "Black wins" if 1.0 in list(self.chess_board[self.BOARD_CHANNEL, :, j]) else "White wins"
        # Check diagonals
        for i in range(11):
            for j in range(11):
                if consecutive_pieces([self.chess_board[self.BOARD_CHANNEL, i + k, j + k] for k in range(5)]):
                    return "Black wins" if 1.0 in [self.chess_board[self.BOARD_CHANNEL, i + k, j + k] for k in range(5)] else "White wins"

                if consecutive_pieces([self.chess_board[self.BOARD_CHANNEL, i + 4 - k, j + k] for k in range(5)]):
                    return "Black wins" if 1.0 in [self.chess_board[self.BOARD_CHANNEL, i + 4 - k, j + k] for k in range(5)] else "White wins"
                
    def play(self,x,y): 
        is_valide = True
        if self.now_player:
            is_valide = self.down_black_chess(x,y)
        else:
            is_valide = self.down_white_chess(x,y)
        return is_valide