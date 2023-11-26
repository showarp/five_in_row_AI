import numpy as np
import copy

def get_scores(vecs,rating_table,ct):
    pM,tM = ct,1 if ct==2 else 2
    tempVceP,tempVceT = vecs.copy(),vecs.copy()
    tempVceP[tempVceP==3]=pM
    tempVceT[tempVceT==3]=tM
    admx = 0
    dfmx = 0
    for scores,mask in rating_table:
        pMask = [pM if i==1 else 0 for i in mask]
        tMask = [tM if i==1 else 0 for i in mask]
        if len(pMask)>len(vecs):continue
        for i in range(0,len(vecs)-len(pMask)):
            if all(tempVceP[i:len(pMask)+i]==pMask):
                admx = max(admx,scores)
            if all(tempVceT[i:len(tMask)+i]==tMask):
                dfmx = max(dfmx,scores)
    return admx+dfmx

def check_reward(board,ct,x,y):
    """
    ct:落子的颜色黑/白 可选值1或2

    评分方法：
    引用论文32页：董红安.(2006).计算机五子棋博奕系统的研究与实现(硕士学位论文,山东师范大学).https://kns.cnki.net/kcms2/article/abstract?v=j6HAoO1nZAwxrl0n5B2Hsq5cRk1uCBEoE6LYMJd1VRCx8HBHF8TUFz4zbvmDXAm2rjvy_z156pGzHkxVzeAUJZxY8Ml0QP_cKsQDjPWGlWq0M37dMcfvYxOSPcJL5vmt&uniplatform=NZKPT&language=CHS
    """
    board_b = copy.deepcopy(board)
    centerX,centerY = 7,7
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
    board_b[x,y] = 3
    rowV = board_b[x,:]
    colV = board_b[:,y]
    diagMain = np.diag(board_b,y-x)
    diagVice = np.diag(np.fliplr(board_b),14-x-y)
    rowScore = get_scores(rowV,rating_table,ct=ct)
    colScore = get_scores(colV,rating_table,ct=ct)
    digMScore = get_scores(diagMain,rating_table,ct=ct)
    digVScore = get_scores(diagVice,rating_table,ct=ct)
    ToCenter = 10.5-((x-centerX)**2+(y-centerY)**2)**0.5
    return rowScore+colScore+digMScore+digVScore+ToCenter