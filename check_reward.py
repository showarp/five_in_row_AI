import numpy as np
import copy

def check_equal_chess(vec,target_num,ct):
    """
    左右双指针中心扩散法判断连了多少子

    vec:棋子元组
    target_num:用于表示落子的数字
    ct:落子的颜色黑/白 可选值1或2
    """
    L,R = vec.index(target_num)-1,vec.index(target_num)+1
    LleftEqChessCnt,righttEqChessCnt = 0,0
    while (L>=0 and vec[L]==ct) or (R<len(vec) and vec[R]==ct):
        if L>=0 and vec[L]==ct:
            LleftEqChessCnt+=1
            L-=1
        if R<len(vec) and vec[R]==ct:
            righttEqChessCnt+=1
            R+=1
    return LleftEqChessCnt,righttEqChessCnt

def check_diff_chess(vec,target_num,ct):
    """
    左右双指针中心扩散法判截断了多少子

    vec:棋子元组
    target_num:用于表示落子的数字
    ct:落子的颜色黑/白 可选值1或2
    """
    L,R = vec.index(target_num)-1,vec.index(target_num)+1
    leftDiffChessCnt,rifghtDiffChessCnt = 0,0
    while (L>=0 and vec[L]!=ct) or (R<len(vec) and vec[R]!=ct):
        if L>=0 and vec[L]!=ct:
            leftDiffChessCnt+=1
            L-=1
        if R<len(vec) and vec[R]!=ct:
            rifghtDiffChessCnt+=1
            R+=1
    return leftDiffChessCnt,rifghtDiffChessCnt

def check_reward(board,row,col,ct):
    """
    x:落子横坐标
    y:落子纵坐标
    ct:落子的颜色黑/白 可选值1或2

    评分方法：
    所有包含本次落子在内的的最大连子，或者最大截断连子数
    rating_table:评分表，数组0-4代表同色连子数量1-5的得分，数组5-8代表阶段的异色连子数
    """
    row+=4
    col+=4
    board_b = copy.deepcopy(board)
    board_b = np.pad(board_b,4)
    TARGET_NUM = float("inf") #用于标识去除没有落子后的位置的落子坐标，数字大小无实际含义
    eq_rating_table = [7,35,800,15000,800000]#连子评分表
    diff_rating_table = [0,15,400,1800,100000]#截断连子评分表

    colTop = board_b[row-4:row+1,col]
    colbottom = board_b[row:row+5,col]
    rowLeft = board_b[row,col-4:col+1]
    rowRight = board_b[row,col:col+5]
    leftTop = np.diagonal(board_b[row-4:row+1,col-4:col+1])
    rightBottom = np.diagonal(board_b[row:row+5,col:col+5])
    rightTop = np.diagonal(board_b[row-4:row+1,col:col+5][::-1])
    leftBottom = np.diagonal(board_b[row:row+5,col-4:col+1][::-1])

    colVec = np.hstack((colTop[:-1],[TARGET_NUM],colbottom[1:]))#垂直方向
    rowVec = np.hstack((rowLeft[:-1],[TARGET_NUM],rowRight[1:]))#水平方向
    
    mdig =  np.hstack((leftTop[:-1],[TARGET_NUM],rightBottom[1:]))#主对角线
    vdig = np.hstack((leftBottom[:-1],[TARGET_NUM],rightTop[1:]))#反对角线

    colVec = [*filter(lambda x:x!=0,colVec)]
    rowVec = [*filter(lambda x:x!=0,rowVec)]
    mdig = [*filter(lambda x:x!=0,mdig)]
    vdig = [*filter(lambda x:x!=0,vdig)]

    #四个轴的回报
    col_reward = sum(map(lambda x:eq_rating_table[x],check_equal_chess(colVec,TARGET_NUM,ct)))
    col_reward += sum(map(lambda x:diff_rating_table[x],check_diff_chess(colVec,TARGET_NUM,ct)))

    row_reward = sum(map(lambda x:eq_rating_table[x],check_equal_chess(rowVec,TARGET_NUM,ct)))
    row_reward += sum(map(lambda x:diff_rating_table[x],check_diff_chess(rowVec,TARGET_NUM,ct)))

    mdig_reward = sum(map(lambda x:eq_rating_table[x],check_equal_chess(mdig,TARGET_NUM,ct)))
    mdig_reward += sum(map(lambda x:diff_rating_table[x],check_diff_chess(mdig,TARGET_NUM,ct)))
    
    vdig_reward = sum(map(lambda x:eq_rating_table[x],check_equal_chess(vdig,TARGET_NUM,ct)))
    vdig_reward += sum(map(lambda x:diff_rating_table[x],check_diff_chess(vdig,TARGET_NUM,ct)))

    return col_reward+row_reward+mdig_reward+vdig_reward


#测试样例

# test_board = np.zeros((15,15))
# test_board[0,0] = 1

# test_board[2,0] = 2
# test_board[1,3] = 2

# test_board[2,2] = 1

# separator_line = "  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+"  
# s = f"   A B C D E F G H I J K L M N O \n{separator_line}\n"
# for i in range(15):
#     s += f"{i:>2}|"
#     for j in range(15):
#         if test_board[i, j] == 1.0:
#             s += "○"  
#         elif test_board[i, j] == 2.0:
#             s += "●"  
#         else:
#             s += " "  
#         s += "|"
#     s += "\n" + separator_line + "\n"
# print(s)
# print("回报:",check_reward(test_board,2,2,1))

