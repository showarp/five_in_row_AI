import numpy as np
import copy

def get_scores(vecs,rating_table,ct):
    ad_scores = 0
    df_scores = 0
    pM,tM = ct,1 if ct==2 else 2
    for vec in vecs:
        admx = 0
        dfmx = 0
        for scores,mask in rating_table:
            pMask = [pM if i==1 else 0 for i in mask]
            tMask = [tM if i==1 else 0 for i in mask]
            if len(pMask)>len(vec):continue
            for i in range(0,len(vec)-len(pMask)):
                if all(vec[i:len(pMask)+i]==pMask):
                    admx = max(admx,scores)
                if all(vec[i:len(tMask)+i]==tMask):
                    dfmx = max(dfmx,scores)
        df_scores-=dfmx
        ad_scores+=admx
    return ad_scores+df_scores

def check_reward(board,ct):
    """
    ct:落子的颜色黑/白 可选值1或2

    评分方法：
    引用论文32页：董红安.(2006).计算机五子棋博奕系统的研究与实现(硕士学位论文,山东师范大学).https://kns.cnki.net/kcms2/article/abstract?v=j6HAoO1nZAwxrl0n5B2Hsq5cRk1uCBEoE6LYMJd1VRCx8HBHF8TUFz4zbvmDXAm2rjvy_z156pGzHkxVzeAUJZxY8Ml0QP_cKsQDjPWGlWq0M37dMcfvYxOSPcJL5vmt&uniplatform=NZKPT&language=CHS
    """
    board_b = copy.deepcopy(board)
    rating_table = [
        (4320,[0,1,1,1,1,0]),
        (720,[0,1,1,1,0,0]),
        (720,[0,0,1,1,1,0]),
        (720,[0,1,1,0,1,0]),
        (720,[0,1,0,1,1,0]),
        (120,[0,0,1,1,0,0]),
        (120,[0,0,1,0,1,0]),
        (120,[0,1,0,1,0,0]),
        (20,[0,0,0,1,0,0]),
        (20,[0,0,1,0,0,0]),
        (50000,[1,1,1,1,1]),
        (720,[1,1,1,1,0]),
        (720,[0,1,1,1,1]),
        (720,[1,1,0,1,1]),
        (720,[1,0,1,1,1]),
        (720,[1,1,1,0,1]),
    ]

    rowV = board_b
    colV = board_b.T
    diagMain = [np.diag(board_b,-14+i) for i in range(15*2)]
    diagVice = [np.diag(board_b[::,::-1],-14+i) for i in range(15*2)]
    rowScore = get_scores(rowV,rating_table,ct=ct)
    colScore = get_scores(colV,rating_table,ct=ct)
    digMScore = get_scores(diagMain,rating_table,ct=ct)
    digVScore = get_scores(diagVice,rating_table,ct=ct)
    return rowScore+colScore+digMScore+digVScore


#测试样例

# test_board = np.zeros((15,15))

# test_board[5,5] = 1
# test_board[5,6] = 1
# test_board[5,7] = 1

# test_board[5,10] = 2
# test_board[5,11] = 2



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

