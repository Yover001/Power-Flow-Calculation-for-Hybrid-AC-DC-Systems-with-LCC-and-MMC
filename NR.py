import math
import OutTxt
import numpy as np
import scipy.linalg
from OutTxt import Real
from math import pi,sqrt
from numpy import arctan,sin,cos,tan,arccos
def PolarNR(U,Angle,Vd,Id,Kt,W,fi,Udc,Idc,derta,M,Ydc,Yac,NodeData,LCC_NodeData,VSC_NodeData,AC_LCC,AC_VSC,Tol,**Option):
#------------------------------------------LCC不平衡量----------------------------------------------#
    Uc=np.append(Vd,Udc)                       # LCC 与 MMC 直流电压
    N_LCC = LCC_NodeData[:,12]                 # 换流器组数
    N_VSC = VSC_NodeData[:,12]                 
    N = np.append(N_LCC,N_VSC)
    Nc = LCC_NodeData.shape[0]                 
    Vds = LCC_NodeData[:,3]
    Ws = np.cos(LCC_NodeData[:,4])
    Pds = LCC_NodeData[:,5]
    Ids = LCC_NodeData[:,7]
    X = LCC_NodeData[:,8]
    control1 = LCC_NodeData[:,9].astype(int)   # 控制方式
    control2 = LCC_NodeData[:,10].astype(int)
    Kts = LCC_NodeData[:,11] 
    kr = 0.995
    Vt = np.zeros([Nc,1])                      # 交流电压
    for i in range(Nc):
        Vt[i] = U[AC_LCC[i]]
    Delta_D1 = np.zeros([Nc,1])                
    Delta_D2 = np.zeros([Nc,1])
    Delta_D3 = np.zeros([Nc,1])
    Delta_D4 = np.zeros([Nc,1])
    Delta_D5 = np.zeros([Nc,1])
    for i in range(Nc):                        # 计算Delta_
        Delta_D1[i] = Vd[i]-2.7*Kt[i]*Vt[i]*W[i]+1.9*X[i]*Id[i]  # LCC与VSC电压基准值不同
        Delta_D2[i] = Vd[i]-2.7*kr*Kt[i]*Vt[i]*cos(fi[i])
        if LCC_NodeData[i,2]==1:
            Delta_D3[i] = Id[i]-N_LCC[i]*np.sum(Ydc[i,:]*Uc)    
        else:
            Delta_D3[i] = -Id[i]-N_LCC[i]*np.sum(Ydc[i,:]*Uc)       
        if control1[i]==1:
            Delta_D4[i] = Id[i]-Ids[i]        # 定电流
        elif control1[i]==2:
            Delta_D4[i] = Vd[i]-Vds[i]        # 定电压
        elif control1[i]==3:
            Delta_D4[i] = N_LCC[i]*Vd[i]*Id[i]-Pds[i]  # 定功率
        elif control1[i]==4:
            Delta_D4[i] = W[i]-Ws[i]          # 定控制角
        elif control1[i]==5:
            Delta_D4[i] = Kt[i]-Kts[i]        # 定变比
        if control2[i]==1:
            Delta_D5[i] = Id[i]-Ids[i]
        elif control2[i]==2:
            Delta_D5[i] = Vd[i]-Vds[i]
        elif control2[i]==3:
            Delta_D5[i] = N_LCC[i]*Vd[i]*Id[i]-Pds[i]
        elif control2[i]==4:
            Delta_D5[i] = W[i]-Ws[i]
        elif control2[i]==5:
            Delta_D5[i]= Kt[i]-Kts[i]
#----------------------------------------------VSC不平衡量----------------------------------------#
    VSC_Num = VSC_NodeData.shape[0]
    Ps = VSC_NodeData[:,3]                 # 换流器控制值(迭代过程中不变)
    Qs = VSC_NodeData[:,4]
    R = VSC_NodeData[:,10]
    Xl = VSC_NodeData[:,11]
    a = arctan(R/Xl)
    Y = 1/np.sqrt(R*R+Xl*Xl)
    Usi = np.zeros([VSC_Num,1])
    for i in range(VSC_Num):
        Usi[i] = U[AC_VSC[i]]                 # 交流母线电压         
    Pv = np.zeros([VSC_Num,1])
    Qv = np.zeros([VSC_Num,1])
    Deltad1 = np.zeros([VSC_Num,1])
    Deltad2 = np.zeros([VSC_Num,1])
    Deltad3 = np.zeros([VSC_Num,1])
    Deltad4 = np.zeros([VSC_Num,1])
    iter = 0 
    for i in range(VSC_Num):                   # 求解功率不平衡量
        Pv[i] =  (sqrt(6)/4)*N_VSC[i]*M[i]*Usi[i]*Udc[i]*Y[i]*sin(derta[i]-a[i]) + N_VSC[i]*Usi[i]*Usi[i]*Y[i]*sin(a[i])
        Qv[i] = -(sqrt(6)/4)*N_VSC[i]*M[i]*Usi[i]*Udc[i]*Y[i]*cos(derta[i]-a[i]) + N_VSC[i]*Usi[i]*Usi[i]*Y[i]*cos(a[i])                     
        Deltad1[iter] = Ps[i]-Pv[i] 
        Deltad2[iter] = Qs[i]-Qv[i] 
        Deltad3[iter] = N_VSC[i]*Udc[i]*Idc[i]-(sqrt(6)/4)*N_VSC[i]*M[i]*Usi[i]*Udc[i]*Y[i]*sin(derta[i]+a[i]) + N_VSC[i]*Udc[i]*Udc[i]*(3/8*M[i]*M[i])*Y[i]*sin(a[i]) 
        Deltad4[iter] = Idc[i]-N_VSC[i]*np.sum((Ydc[i+Nc,:]*Uc))
        iter = iter+1
#-----------------------------------------------交流不平衡量--------------------------------------#
    PQNode = NodeData[np.where(NodeData[:,1]==1),0]-1                                    # PQ节点
    PVNode = NodeData[np.where(NodeData[:,1]==2),0]-1                                     # PV节点
    SlackNode = NodeData[np.where(NodeData[:,1]==3),0]-1                                  # 平衡节点
    SlackNode=SlackNode.squeeze(0) 
    # print(SlackNode)
    P_Real = -NodeData[:,2]+NodeData[:,4]                                                 # 节点输入有功功率
    Q_Real = -NodeData[:,3]+NodeData[:,5]                                                 # 节点输入无功功率
    NumNode = Yac.shape[0]                                                                # 节点数目
    NumPQ = max(PQNode.shape) 
    NumPV = max(PVNode.shape)                                                             # PQ节点数目
    G = Yac.real
    B = Yac.imag
    P = np.zeros([NumNode,1])
    Q = np.zeros([NumNode,1])
    DeltaP = np.zeros([NumNode-2,1])
    DeltaQ = np.zeros([NumPQ,1])
    P_iter = 0
    Q_iter = 0
    for i in range(NumNode):                 # 求解功率不平衡量
        if i in AC_LCC:
            jz = np.where(LCC_NodeData[:,0]==(i+1))
            if LCC_NodeData[jz,2]==1:
                P[i] = U[i]*np.sum(U*(G[i,:]*np.cos(Angle[i]-Angle) +  B[i,:]*np.sin(Angle[i]-Angle))) + N_LCC[jz]*Vd[jz]*Id[jz]
                Q[i] = U[i]*np.sum(U*(G[i,:]*np.sin(Angle[i]-Angle) -  B[i,:]*np.cos(Angle[i]-Angle))) + N_LCC[jz]*Vd[jz]*Id[jz]*tan(fi[jz])
            else:
                P[i] = U[i]*np.sum(U*(G[i,:]*np.cos(Angle[i]-Angle) +  B[i,:]*np.sin(Angle[i]-Angle))) - N_LCC[jz]*Vd[jz]*Id[jz]
                Q[i] = U[i]*np.sum(U*(G[i,:]*np.sin(Angle[i]-Angle) -  B[i,:]*np.cos(Angle[i]-Angle))) + N_LCC[jz]*Vd[jz]*Id[jz]*tan(fi[jz])   
        elif i in AC_VSC:
            jzz = np.where(VSC_NodeData[:,0]==(i+1))
            P[i] = U[i]*np.sum(U*(G[i,:]*np.cos(Angle[i]-Angle) +  B[i,:]*np.sin(Angle[i]-Angle))) + Pv[jzz]
            Q[i] = U[i]*np.sum(U*(G[i,:]*np.sin(Angle[i]-Angle) -  B[i,:]*np.cos(Angle[i]-Angle))) + Qv[jzz] 
        else:
            P[i] = U[i]*np.sum(U*(G[i,:]*np.cos(Angle[i]-Angle) +  B[i,:]*np.sin(Angle[i]-Angle)))
            Q[i] = U[i]*np.sum(U*(G[i,:]*np.sin(Angle[i]-Angle) -  B[i,:]*np.cos(Angle[i]-Angle)))
        if i not in SlackNode:                        # 不是平衡节点
            DeltaP[P_iter] = P_Real[i]-P[i]     # NumPQ+NumPV
            if i in PQNode:                     # PQ节点
                DeltaQ[Q_iter] = Q_Real[i]-Q[i] # NumPQ
                Q_iter = Q_iter+1
            P_iter = P_iter+1
#----------------------------------------------整合不平衡量------------------------------------------#
    DeltaD = np.vstack([DeltaP,DeltaQ,Delta_D1,Delta_D2,Delta_D3,Delta_D4,Delta_D5,Deltad1,Deltad2,Deltad3,Deltad4])  # 功率不平衡量
    Ng = 0
    for i in range(VSC_Num):
        if VSC_NodeData[i,2]==3:
            DeltaD = np.delete(DeltaD,NumPQ+NumNode-2+5*Nc+i-Ng,0)   #去掉MMC平衡换流器Udc
            Ng = Ng+1
    # Option['string'] = '功率不平衡量为：\n'
    # Real(DeltaD,**Option)
    MaxError = np.max(np.abs(DeltaD))
    if MaxError<Tol:
        print('交直流偏差：',MaxError)
        return(U,Angle,Vd,Id,Kt,W,fi,Udc,Idc,derta,M,MaxError,P,Q)
#----------------------------------------------雅克比矩阵-------------------------------------------#
#-----------------------------------------------交流-----------------------------------------------#
    # H = np.zeros([NumNode-2,NumNode-2])
    # Ns = np.zeros([NumNode-2,NumNode-2]) 
    # J = np.zeros([NumNode-2,NumNode-2])
    # L = np.zeros([NumNode-2,NumNode-2])
    # H_iter = -1                         # H代表行
    # for i in range(NumNode):
    #     N_iter = -1                     # N代表列
    #     if i not in SlackNode:                
    #         H_iter = H_iter+1
    #         for j in range(NumNode):
    #             if j not in SlackNode:          # 非平衡节点计算H矩阵
    #                 N_iter = N_iter+1
    #                 if i != j:
    #                     Angleij = Angle[i]-Angle[j]
    #                     H[H_iter,N_iter] = -U[i]*U[j]*(G[i,j]*np.sin(Angleij)-B[i,j]*np.cos(Angleij))
    #                     J[H_iter,N_iter] = U[i]*U[j]*(G[i,j]*np.cos(Angleij)+B[i,j]*np.sin(Angleij))
    #                     Ns[H_iter,N_iter] = -U[i]*U[j]*(G[i,j]*np.cos(Angleij)+B[i,j]*np.sin(Angleij))
    #                     L[H_iter,N_iter] = -U[i]*U[j]*(G[i,j]*np.sin(Angleij)-B[i,j]*np.cos(Angleij))
    #                 else:
    #                     H[H_iter,N_iter] = U[i]*np.sum(U*(G[i,:]*np.sin(Angle[i]-Angle) -  B[i,:]*np.cos(Angle[i]-Angle)))+U[i]**2*B[i,i]
    #                     J[H_iter,N_iter] = -U[i]*np.sum(U*(G[i,:]*np.cos(Angle[i]-Angle) +  B[i,:]*np.sin(Angle[i]-Angle)))+G[i,i]*U[i]**2
    #                     Ns[H_iter,N_iter] = -U[i]*np.sum(U*(G[i,:]*np.cos(Angle[i]-Angle) +  B[i,:]*np.sin(Angle[i]-Angle)))-G[i,i]*U[i]**2
    #                     L[H_iter,N_iter] = -U[i]*np.sum(U*(G[i,:]*np.sin(Angle[i]-Angle) -  B[i,:]*np.cos(Angle[i]-Angle)))+B[i,i]*U[i]**2
    # Jaccobi = np.vstack([np.hstack([H,Ns]),np.hstack([J,L])])
    
    ## H and N and J and L
    SlackNode_1level = [int(i) for i in SlackNode]
    # H
    H = 2 * np.ones([NumNode,NumNode])
    np.fill_diagonal(H,1)
    H = np.delete(H,SlackNode_1level,axis=0)
    H = np.delete(H,SlackNode_1level,axis=1)
    U_delete = np.delete(U,SlackNode_1level)
    Angle_delete = np.delete(Angle,SlackNode_1level)
    G_delete = np.delete(G,SlackNode_1level,axis=1) #axis=1代表列；axis=0代表行
    G_delete = np.delete(G_delete,SlackNode_1level,axis=0)
    B_delete = np.delete(B,SlackNode_1level,axis=1)
    B_delete = np.delete(B_delete,SlackNode_1level,axis=0)
    np.copyto(H, G_delete, where=((G_delete == 0) & (B_delete == 0) & (H != 1)))
    coordinate = np.where(H != 0)
    coordinate_1 = np.where(H == 1)
    H_ones_coordinate = list(zip(coordinate[0],coordinate[1]))
    H_ones_coordinate_1 = list(zip(coordinate_1[0],coordinate_1[1]))
    for element in H_ones_coordinate:
        if element in H_ones_coordinate_1:
            i = element[0]
            j = element[1]
            # 此处要增加i j的值
            if i < SlackNode_1level[0] and i < SlackNode_1level[1]-1:
                i_ = i
            elif i >= SlackNode_1level[0] and i < SlackNode_1level[1]-1:
                i_ = i + 1
            elif i < SlackNode_1level[0] and i >= SlackNode_1level[1]-1:
                i_ = i + 1
            elif i >= SlackNode_1level[0] and i >= SlackNode_1level[1]-1:
                i_ = i + 2
            if j < SlackNode_1level[0] and j < SlackNode_1level[1]-1:
                j_ = j
            elif j >= SlackNode_1level[0] and j < SlackNode_1level[1]-1:
                j_ = j + 1
            elif j < SlackNode_1level[0] and j >= SlackNode_1level[1]-1:
                j_ = j + 1
            elif j >= SlackNode_1level[0] and j >= SlackNode_1level[1]-1:
                j_ = j + 2
            Angleij = Angle[i_]-Angle[j_]
            Anglei_j = Angle[i_]-Angle
            H[i,j] = U[i_]*np.sum(U*(G[i_,:]*np.sin(Anglei_j) -  B[i_,:]*np.cos(Anglei_j)))+U[i_]**2*B[i_,i_]
        else:
            i = element[0]
            j = element[1]
            Angleij = Angle_delete[i]-Angle_delete[j]
            H[i,j] = -U_delete[i]*U_delete[j]*(G_delete[i,j]*np.sin(Angleij)-B_delete[i,j]*np.cos(Angleij))
    # N
    N = 2 * np.ones([NumNode,NumNode])
    np.fill_diagonal(N,1)
    N = np.delete(N,SlackNode_1level,axis=0)
    N = np.delete(N,SlackNode_1level,axis=1)
    U_delete_i = np.delete(U,SlackNode_1level)
    U_delete_j = np.delete(U,SlackNode_1level)
    Angle_delete_i = np.delete(Angle,SlackNode_1level)
    Angle_delete_j = np.delete(Angle,SlackNode_1level)
    G_delete = np.delete(G,SlackNode_1level,axis=0)
    G_delete = np.delete(G_delete,SlackNode_1level,axis=1)
    B_delete = np.delete(B,SlackNode_1level,axis=0)
    B_delete = np.delete(B_delete,SlackNode_1level,axis=1)
    np.copyto(N, G_delete, where=((G_delete == 0) & (B_delete == 0) & (N != 1)))
    coordinate = np.where(N != 0)
    coordinate_1 = np.where(N == 1)
    N_ones_coordinate = list(zip(coordinate[0],coordinate[1]))
    N_ones_coordinate_1 = list(zip(coordinate_1[0],coordinate_1[1]))
    for element in N_ones_coordinate:
        if element in N_ones_coordinate_1:
            i = element[0]
            j = element[1]
            if i < SlackNode_1level[0] and i < SlackNode_1level[1]-1:
                i_ = i
            elif i >= SlackNode_1level[0] and i < SlackNode_1level[1]-1:
                i_ = i + 1
            elif i < SlackNode_1level[0] and i >= SlackNode_1level[1]-1:
                i_ = i + 1
            elif i >= SlackNode_1level[0] and i >= SlackNode_1level[1]-1:
                i_ = i + 2
            if j < SlackNode_1level[0] and j < SlackNode_1level[1]-1:
                j_ = j
            elif j >= SlackNode_1level[0] and j < SlackNode_1level[1]-1:
                j_ = j + 1
            elif j < SlackNode_1level[0] and j >= SlackNode_1level[1]-1:
                j_ = j + 1
            elif j >= SlackNode_1level[0] and j >= SlackNode_1level[1]-1:
                j_ = j + 2
            Angleij = Angle[i_]-Angle[j_]
            Anglei_j = Angle[i_]-Angle
            N[i,j] = -U[i_]*np.sum(U*(G[i_,:]*np.cos(Anglei_j) +  B[i_,:]*np.sin(Anglei_j)))-G[i_,i_]*U[i_]**2
        else:
            i = element[0]
            j = element[1]
            Angleij = Angle_delete_i[i]-Angle_delete_j[j]
            N[i,j] = -U_delete_i[i]*U_delete_j[j]*(G_delete[i,j]*np.cos(Angleij)+B_delete[i,j]*np.sin(Angleij))
    # J
    J = 2 * np.ones([NumNode,NumNode])
    np.fill_diagonal(J,1)
    J = np.delete(J,SlackNode_1level,axis=0)
    J = np.delete(J,SlackNode_1level,axis=1)
    U_delete_i = np.delete(U,SlackNode_1level)
    U_delete_j = np.delete(U,SlackNode_1level)
    Angle_delete_i = np.delete(Angle,SlackNode_1level)
    Angle_delete_j = np.delete(Angle,SlackNode_1level)
    P_delete = np.delete(P,SlackNode_1level)
    G_delete = np.delete(G,SlackNode_1level,axis=0)
    G_delete = np.delete(G_delete,SlackNode_1level,axis=1)
    B_delete = np.delete(B,SlackNode_1level,axis=0)
    B_delete = np.delete(B_delete,SlackNode_1level,axis=1)
    np.copyto(J, G_delete, where=((G_delete == 0) & (B_delete == 0) & (J != 1)))
    coordinate = np.where(J != 0)
    coordinate_1 = np.where(J == 1)
    J_ones_coordinate = list(zip(coordinate[0],coordinate[1]))
    J_ones_coordinate_1 = list(zip(coordinate_1[0],coordinate_1[1]))
    for element in J_ones_coordinate:
        if element in J_ones_coordinate_1:
            i = element[0]
            j = element[1]
            if i < SlackNode_1level[0] and i < SlackNode_1level[1]-1:
                i_ = i
            elif i >= SlackNode_1level[0] and i < SlackNode_1level[1]-1:
                i_ = i + 1
            elif i < SlackNode_1level[0] and i >= SlackNode_1level[1]-1:
                i_ = i + 1
            elif i >= SlackNode_1level[0] and i >= SlackNode_1level[1]-1:
                i_ = i + 2
            if j < SlackNode_1level[0] and j < SlackNode_1level[1]-1:
                j_ = j
            elif j >= SlackNode_1level[0] and j < SlackNode_1level[1]-1:
                j_ = j + 1
            elif j < SlackNode_1level[0] and j >= SlackNode_1level[1]-1:
                j_ = j + 1
            elif j >= SlackNode_1level[0] and j >= SlackNode_1level[1]-1:
                j_ = j + 2
            Angleij = Angle[i_]-Angle[j_]
            Anglei_j = Angle[i_]-Angle
            J[i,j] = -U[i_]*np.sum(U*(G[i_,:]*np.cos(Anglei_j) +  B[i_,:]*np.sin(Anglei_j)))+G[i_,i_]*U[i_]**2
        else:
            i = element[0]
            j = element[1]
            Angleij = Angle_delete_i[i]-Angle_delete_j[j]
            J[i,j] = U_delete_i[i]*U_delete_j[j]*(G_delete[i,j]*np.cos(Angleij)+B_delete[i,j]*np.sin(Angleij))
    # L
    L = 2 * np.ones([NumNode,NumNode])
    np.fill_diagonal(L,1)
    L = np.delete(L,SlackNode_1level,axis=0)
    L = np.delete(L,SlackNode_1level,axis=1)
    U_delete = np.delete(U,SlackNode_1level)
    Angle_delete = np.delete(Angle,SlackNode_1level)
    Q_delete = np.delete(Q,SlackNode_1level)
    G_delete = np.delete(G,SlackNode_1level,axis=0)
    G_delete = np.delete(G_delete,SlackNode_1level,axis=1)
    B_delete = np.delete(B,SlackNode_1level,axis=0)
    B_delete = np.delete(B_delete,SlackNode_1level,axis=1)
    np.copyto(L, G_delete, where=((G_delete == 0) & (B_delete == 0) & (L != 1)))
    coordinate = np.where(L != 0)
    coordinate_1 = np.where(L == 1)
    L_ones_coordinate = list(zip(coordinate[0],coordinate[1]))
    L_ones_coordinate_1 = list(zip(coordinate_1[0],coordinate_1[1]))
    for element in L_ones_coordinate:
        if element in L_ones_coordinate_1:
            i = element[0]
            j = element[1]
            if i < SlackNode_1level[0] and i < SlackNode_1level[1]-1:
                i_ = i
            elif i >= SlackNode_1level[0] and i < SlackNode_1level[1]-1:
                i_ = i + 1
            elif i < SlackNode_1level[0] and i >= SlackNode_1level[1]-1:
                i_ = i + 1
            elif i >= SlackNode_1level[0] and i >= SlackNode_1level[1]-1:
                i_ = i + 2
            if j < SlackNode_1level[0] and j < SlackNode_1level[1]-1:
                j_ = j
            elif j >= SlackNode_1level[0] and j < SlackNode_1level[1]-1:
                j_ = j + 1
            elif j < SlackNode_1level[0] and j >= SlackNode_1level[1]-1:
                j_ = j + 1
            elif j >= SlackNode_1level[0] and j >= SlackNode_1level[1]-1:
                j_ = j + 2
            Angleij = Angle[i_]-Angle[j_]
            Anglei_j = Angle[i_]-Angle
            L[i,j] = -U[i_]*np.sum(U*(G[i_,:]*np.sin(Anglei_j) -  B[i_,:]*np.cos(Anglei_j)))+B[i_,i_]*U[i_]**2
        else:
            i = element[0]
            j = element[1]
            Angleij = Angle_delete[i]-Angle_delete[j]
            L[i,j] = -U_delete[i]*U_delete[j]*(G_delete[i,j]*np.sin(Angleij)-B_delete[i,j]*np.cos(Angleij))
    Jaccobi = np.vstack([np.hstack([H,N]),np.hstack([J,L])])

    
#  修正接MMC的交流雅克比
    for i in range(VSC_Num):                
        if AC_VSC[i]>SlackNode[1]:
            Jaccobi[AC_VSC[i]-2,NumNode-2+AC_VSC[i]-2] = Jaccobi[AC_VSC[i]-2,NumNode-2+AC_VSC[i]-2]-(sqrt(6)/4)*N_VSC[i]*M[i]*Udc[i]*Usi[i]*Y[i]*sin(derta[i]-a[i])-2*N_VSC[i]*Usi[i]*Usi[i]*Y[i]*sin(a[i])                     # P对U
            Jaccobi[NumNode-2+AC_VSC[i]-2,NumNode-2+AC_VSC[i]-2] = Jaccobi[NumNode-2+AC_VSC[i]-2,NumNode-2+AC_VSC[i]-2]+(sqrt(6)/4)*N_VSC[i]*M[i]*Udc[i]*Usi[i]*Y[i]*cos(derta[i]-a[i])-2*N_VSC[i]*Usi[i]*Usi[i]*Y[i]*cos(a[i]) # Q对U
        elif AC_VSC[i]<SlackNode[0]:
            Jaccobi[AC_VSC[i],NumNode-2+AC_VSC[i]] = Jaccobi[AC_VSC[i],NumNode-2+AC_VSC[i]]-(sqrt(6)/4)*N_VSC[i]*M[i]*Udc[i]*Usi[i]*Y[i]*sin(derta[i]-a[i])-2*N_VSC[i]*Usi[i]*Usi[i]*Y[i]*sin(a[i])                     # P对U
            Jaccobi[NumNode-2+AC_VSC[i],NumNode-2+AC_VSC[i]] = Jaccobi[NumNode-2+AC_VSC[i],NumNode-2+AC_VSC[i]]+(sqrt(6)/4)*N_VSC[i]*M[i]*Udc[i]*Usi[i]*Y[i]*cos(derta[i]-a[i])-2*N_VSC[i]*Usi[i]*Usi[i]*Y[i]*cos(a[i]) # Q对U
        else:
            Jaccobi[AC_VSC[i]-1,NumNode-2+AC_VSC[i]-1] = Jaccobi[AC_VSC[i]-1,NumNode-2+AC_VSC[i]-1]-(sqrt(6)/4)*N_VSC[i]*M[i]*Udc[i]*Usi[i]*Y[i]*sin(derta[i]-a[i])-2*N_VSC[i]*Usi[i]*Usi[i]*Y[i]*sin(a[i])                     # P对U
            Jaccobi[NumNode-2+AC_VSC[i]-1,NumNode-2+AC_VSC[i]-1] = Jaccobi[NumNode-2+AC_VSC[i]-1,NumNode-2+AC_VSC[i]-1]+(sqrt(6)/4)*N_VSC[i]*M[i]*Udc[i]*Usi[i]*Y[i]*cos(derta[i]-a[i])-2*N_VSC[i]*Usi[i]*Usi[i]*Y[i]*cos(a[i]) # Q对U
#------------------------------------------------LCC------------------------------------------------#   
    F11 = np.eye(Nc)
    F21 = np.eye(Nc)
    F31 = np.zeros([Nc,Nc])
    F32 = np.eye(Nc)
    for i in range(Nc):
        if LCC_NodeData[i,2]==2:
            F32[i,i]=F32[i,i]*(-1)
    F12 = np.diag(1.9*X)
    F13 = -np.diag(Vt.reshape(Nc)*W*2.7)
    F23 = -np.diag(kr*Vt.reshape(Nc)*cos(fi.reshape(Nc))*2.7)
    F14 = -np.diag(Kt*Vt.reshape(Nc)*2.7)
    F25 = np.diag(kr*Kt*Vt.reshape(Nc)*sin(fi.reshape(Nc))*2.7)
    F41 = np.zeros([Nc,Nc])
    F42 = np.zeros([Nc,Nc])
    F43 = np.zeros([Nc,Nc])
    F44 = np.zeros([Nc,Nc])
    F45 = np.zeros([Nc,Nc])
    F51 = np.zeros([Nc,Nc])
    F52 = np.zeros([Nc,Nc])
    F53 = np.zeros([Nc,Nc])
    F54 = np.zeros([Nc,Nc])
    F55 = np.zeros([Nc,Nc])
    A   = np.zeros([Nc,Nc])
    # F41 F42 F43 F44
    F4_iter = -1
    for i in range(Nc):
        F4_iter = F4_iter+1                     # 记录行数
        F4_iter_y = -1                          # 记录列数
        for j in range(4*Nc):
            F4_iter_y = F4_iter_y+1
            if j>=0 and j<Nc: # F41
                if j==i and control1[j]==2:
                    F41[F4_iter,F4_iter_y]=1
                elif j==i and control1[j]==3:
                    F41[F4_iter,F4_iter_y]=Id[i]
            elif j>=Nc and j<2*Nc: # F42
                if j-Nc==i and control1[j-Nc]==1:
                    F42[F4_iter,F4_iter_y-Nc]=1
                elif j-Nc==i and control1[j-Nc]==3:
                    F42[F4_iter,F4_iter_y-Nc]=Vd[i]
            elif j>=2*Nc and j<3*Nc: # F43
                if j-2*Nc==i and control1[j-2*Nc]==5:
                    F43[F4_iter,F4_iter_y-2*Nc]=1
            elif j>=3*Nc and j<4*Nc: # F44
                if j-3*Nc==i and control1[j-3*Nc]==4:
                    F44[F4_iter,F4_iter_y-3*Nc]=1
    # F51 F52 F53 F54
    F5_iter = -1
    for i in range(Nc):
        F5_iter = F5_iter+1           # 记录行数
        F5_iter_y = -1                # 记录列数
        for j in range(4*Nc):
            F5_iter_y = F5_iter_y+1
            if j>=0 and j<Nc: # F51
                if j==i and control2[j]==2:
                    F51[F5_iter,F5_iter_y]=1
                elif j==i and control2[j]==3:
                    F51[F5_iter,F5_iter_y]=Id[i]
            elif j>=Nc and j<2*Nc: # F52
                if j-Nc==i and control2[j-Nc]==1:
                    F52[F5_iter,F5_iter_y-Nc]=1
                elif j-Nc==i and control2[j-Nc]==3:
                    F52[F5_iter,F5_iter_y-Nc]=Vd[i]
            elif j>=2*Nc and j<3*Nc: # F53
                if j-2*Nc==i and control2[j-2*Nc]==5:
                    F53[F5_iter,F5_iter_y-2*Nc]=1
            elif j>=3*Nc and j<4*Nc: # F54
                if j-3*Nc==i and control2[j-3*Nc]==4:
                    F54[F5_iter,F5_iter_y-3*Nc]=1
    F1 = np.concatenate((F11,F12,F13,F14,A),axis=1)
    F2 = np.concatenate((F21,A,F23,A,F25),axis=1)
    F3 = np.concatenate((F31,F32,A,A,A),axis=1)
    F4 = np.concatenate((F41,F42,F43,F44,F45),axis=1)
    F5 = np.concatenate((F51,F52,F53,F54,F55),axis=1)
    F = np.concatenate((F1,F2,F3,F4,F5))  
#----------------------------------------------VSC-------------------------------------------------#
    D11=np.zeros([VSC_Num,VSC_Num])
    D12=np.zeros([VSC_Num,VSC_Num]) 
    D13=np.zeros([VSC_Num,VSC_Num])
    D14=np.zeros([VSC_Num,VSC_Num])
    D21=np.zeros([VSC_Num,VSC_Num])
    D22=np.zeros([VSC_Num,VSC_Num]) 
    D23=np.zeros([VSC_Num,VSC_Num])
    D24=np.zeros([VSC_Num,VSC_Num])
    D31=np.zeros([VSC_Num,VSC_Num])
    D32=np.zeros([VSC_Num,VSC_Num]) 
    D33=np.zeros([VSC_Num,VSC_Num])
    D34=np.zeros([VSC_Num,VSC_Num])
    D41=np.zeros([VSC_Num,VSC_Num])
    D42=np.eye(VSC_Num)              
    D43=np.zeros([VSC_Num,VSC_Num]) 
    D44=np.zeros([VSC_Num,VSC_Num]) 
    for i in range(VSC_Num):
        D11[i,i]=-(sqrt(6)/4)*N_VSC[i]*M[i]*Usi[i]*Y[i]*sin(derta[i]-a[i])
        D13[i,i]=-(sqrt(6)/4)*N_VSC[i]*M[i]*Usi[i]*Udc[i]*Y[i]*cos(derta[i]-a[i])
        D14[i,i]=-(sqrt(6)/4)*N_VSC[i]*Usi[i]*Udc[i]*Y[i]*sin(derta[i]-a[i])
        D21[i,i]=(sqrt(6)/4)*N_VSC[i]*M[i]*Usi[i]*Y[i]*cos(derta[i]-a[i])
        D23[i,i]=-(sqrt(6)/4)*N_VSC[i]*M[i]*Usi[i]*Udc[i]*Y[i]*sin(derta[i]-a[i])
        D24[i,i]=(sqrt(6)/4)*N_VSC[i]*Usi[i]*Udc[i]*Y[i]*cos(derta[i]-a[i])
        D31[i,i]=N_VSC[i]*Idc[i]-N_VSC[i]*(sqrt(6)/4)*M[i]*Usi[i]*Y[i]*sin(derta[i]+a[i])+N_VSC[i]*Udc[i]*(3/4)*M[i]*M[i]*Y[i]*sin(a[i]) 
        D32[i,i]=N_VSC[i]*Udc[i]
        D33[i,i]=-(sqrt(6)/4)*N_VSC[i]*M[i]*Usi[i]*Udc[i]*Y[i]*cos(derta[i]+a[i])
        D34[i,i]=-(sqrt(6)/4)*N_VSC[i]*Usi[i]*Udc[i]*Y[i]*sin(derta[i]+a[i])+N_VSC[i]*Udc[i]*Udc[i]*(3/4*M[i])*Y[i]*sin(a[i])
    D = np.vstack([np.hstack([D11,D12,D13,D14]),np.hstack([D21,D22,D23,D24]),np.hstack([D31,D32,D33,D34]),np.hstack([D41,D42,D43,D44])])      
#-----------------------------------------------整合雅克比矩阵---------------------------------------#
    J_J = scipy.linalg.block_diag(Jaccobi,F,D)
#-----------------------------------------------交流对LCC非对角--------------------------------------#
    for i in range(Nc):
        if AC_LCC[i]>SlackNode[1]:       # 对雅可比矩阵操作需要考虑平衡节点的问题
            o = 2
        elif AC_LCC[i]<SlackNode[0]:
            o = 0
        else:
            o = 1
        if LCC_NodeData[i,2]==1:
            J_J[AC_LCC[i]-o,2*(NumNode-2)+i]=-N_LCC[i]*Id[i]                                                      # P对Udc偏导,-1是减掉平衡节点
            J_J[AC_LCC[i]-o,2*(NumNode-2)+Nc+i]=-N_LCC[i]*Vd[i]                                                   # P对Idc偏导
            J_J[AC_LCC[i]-o+NumNode-2,2*(NumNode-2)+i]=-N_LCC[i]*Id[i]*tan(fi[i])                                 # Q对Udc偏导
            J_J[AC_LCC[i]-o+NumNode-2,2*(NumNode-2)+Nc+i]=-N_LCC[i]*Vd[i]*tan(fi[i])                              # Q对Idc偏导
            J_J[AC_LCC[i]-o+NumNode-2,2*(NumNode-2)+4*Nc+i]=-N_LCC[i]*Vd[i]*Id[i]*(1/cos(fi[i]))*(1/cos(fi[i]))   
        else:                                            
            J_J[AC_LCC[i]-o,2*(NumNode-2)+i]=N_LCC[i]*Id[i]                                                      
            J_J[AC_LCC[i]-o,2*(NumNode-2)+Nc+i]=N_LCC[i]*Vd[i]   
            J_J[AC_LCC[i]-o+NumNode-2,2*(NumNode-2)+i]=-N_LCC[i]*Id[i]*tan(fi[i])                               
            J_J[AC_LCC[i]-o+NumNode-2,2*(NumNode-2)+Nc+i]=-N_LCC[i]*Vd[i]*tan(fi[i])                              
            J_J[AC_LCC[i]-o+NumNode-2,2*(NumNode-2)+4*Nc+i]=-N_LCC[i]*Vd[i]*Id[i]*(1/cos(fi[i]))*(1/cos(fi[i]))           
        #-------------------------------------------------------------------------------------------#
        J_J[2*(NumNode-2)+i,NumNode-2+AC_LCC[i]-o]=-2.7*Kt[i]*Vt[i]*W[i]                                    
        J_J[2*(NumNode-2)+i+Nc,NumNode-2+AC_LCC[i]-o]=-2.7*kr*Kt[i]*Vt[i]*cos(fi[i])                        
#-----------------------------------------------交流对MMC非对角--------------------------------------#
    for i in range(VSC_Num):
        if AC_VSC[i]>SlackNode[1]:       # 对雅可比矩阵操作需要考虑平衡节点的问题
            o = 2
        elif AC_VSC[i]<SlackNode[0]:
            o = 0
        else:
            o = 1
        J_J[AC_VSC[i]-o,5*Nc+2*(NumNode-2)+i]=-(sqrt(6)/4)*N_VSC[i]*M[i]*Usi[i]*Y[i]*sin(derta[i]-a[i])                                            # P对Udc偏导
        J_J[AC_VSC[i]-o,5*Nc+2*(NumNode-2)+2*VSC_Num+i]=-(sqrt(6)/4)*N_VSC[i]*M[i]*Usi[i]*Udc[i]*Y[i]*cos(derta[i]-a[i])                           # P对a偏导
        J_J[AC_VSC[i]-o,5*Nc+2*(NumNode-2)+3*VSC_Num+i]=-(sqrt(6)/4)*N_VSC[i]*Usi[i]*Udc[i]*Y[i]*sin(derta[i]-a[i])                                # P对M偏导
        J_J[AC_VSC[i]-o+NumNode-2,5*Nc+2*(NumNode-2)+i]=(sqrt(6)/4)*N_VSC[i]*M[i]*Usi[i]*Y[i]*cos(derta[i]-a[i])                                   # Q对Udc偏导
        J_J[AC_VSC[i]-o+NumNode-2,5*Nc+2*(NumNode-2)+2*VSC_Num+i]=-(sqrt(6)/4)*N_VSC[i]*M[i]*Usi[i]*Udc[i]*Y[i]*sin(derta[i]-a[i])                 # Q对a偏导
        J_J[AC_VSC[i]-o+NumNode-2,5*Nc+2*(NumNode-2)+3*VSC_Num+i]=(sqrt(6)/4)*N_VSC[i]*Usi[i]*Udc[i]*Y[i]*cos(derta[i]-a[i])                       # Q对M偏导
        #-------------------------------------------------------------------------------------------#
        J_J[2*(NumNode-2)+5*Nc+i,NumNode-2+AC_VSC[i]-o]=-(sqrt(6)/4)*N_VSC[i]*M[i]*Udc[i]*Usi[i]*Y[i]*sin(derta[i]-a[i])-2*N_VSC[i]*Usi[i]*Usi[i]*Y[i]*sin(a[i])          # P对Us偏导
        J_J[2*(NumNode-2)+5*Nc+VSC_Num+i,NumNode-2+AC_VSC[i]-o]=(sqrt(6)/4)*N_VSC[i]*M[i]*Udc[i]*Usi[i]*Y[i]*cos(derta[i]-a[i])-2*N_VSC[i]*Usi[i]*Usi[i]*Y[i]*cos(a[i])   # Q对Us偏导
        J_J[2*(NumNode-2)+5*Nc+2*VSC_Num+i,NumNode-2+AC_VSC[i]-o]=-(sqrt(6)/4)*N_VSC[i]*M[i]*Udc[i]*Usi[i]*Y[i]*sin(derta[i]+a[i])                                 # Pdc对Us偏导
#---------------------------------------------LCC与MMC直流网络方程偏导-------------------------------# 
    for i in range(Nc):
        for j in range(Nc):
            J_J[2*(NumNode-2)+2*Nc+i,2*(NumNode-2)+j]=-N_LCC[j]*Ydc[i,j]                                    #注意：LCC标号靠前
        for j in range(VSC_Num):
            J_J[2*(NumNode-2)+2*Nc+i,2*(NumNode-2)+5*Nc+j]=-N_VSC[j]*Ydc[i,Nc+j]
    for i in range(VSC_Num):
        for j in range(Nc):
            J_J[2*(NumNode-2)+5*Nc+3*VSC_Num+i,2*(NumNode-2)+j]=-N_LCC[j]*Ydc[Nc+i,j]
        for j in range(VSC_Num):
            J_J[2*(NumNode-2)+5*Nc+3*VSC_Num+i,2*(NumNode-2)+5*Nc+j]=-N_VSC[j]*Ydc[Nc+i,Nc+j]
#-----------------------------------------去掉PV节点电压-对应J的列-----------------------------------#
    Ng = 0
    PVNode = PVNode.squeeze(-2)
    for i in PVNode:
        if i>SlackNode[1]:       # 对雅可比矩阵操作需要考虑平衡节点的问题
            o = 2
        elif i<SlackNode[0]:
            o = 0
        else:
            o = 1
        J_J = np.delete(J_J,NumNode-2+i.astype(int)-o-Ng,0)                                          # 注意：一切对于雅克比矩阵的处理，针对某行某列处理时需额外减1（平衡节点）
        J_J = np.delete(J_J,NumNode-2+i.astype(int)-o-Ng,1)                                          # ——1 取得平衡节点
        Ng = Ng+1
#-----------------------------------------去掉MMC平衡换流器P方程-------------------------------------#                                                                                     # MMC平衡换流器个数
    Ng = 0
    for i in range(VSC_Num):
        if VSC_NodeData[i,2]==3:
            J_J = np.delete(J_J,NumPQ+NumNode-2+5*Nc+i-Ng,0) #-NP 循环删减J矩阵维数发生变化
            J_J = np.delete(J_J,NumPQ+NumNode-2+5*Nc+i-Ng,1)
            Ng = Ng+1
#-------------------------------------------------修正----------------------------------------------#
    # Option['string'] = 'jacobi矩阵为：\n'
    # Real(J_J,**Option)
    Delta = np.linalg.solve(J_J,DeltaD)
    # print(DeltaD)
    # Option['string'] = '方程组求解结果：\n'
    # Real(Delta,**Option)
#-------------------------------------------------交流修正------------------------------------------#
    DeltaAngle = Delta[0:NumNode-2]                                                              #  注意下标
    DeltaU_U = Delta[NumNode-2:NumPQ+NumNode-2]
    DA_iter = -1
    U_U_iter = -1
    for i in range(NumNode):
        if i not in SlackNode:
            DA_iter = DA_iter+1
            Angle[i] = Angle[i]-DeltaAngle[DA_iter]
            if i in PQNode:
                U_U_iter = U_U_iter+1
                U[i] = U[i]-U[i]*DeltaU_U[U_U_iter] 
    # Option['string'] = '更新之后的电压幅值为：\n'
    # Real(U,**Option)
    # Option['string'] = '相角为：\n'
    # Real(Angle,**Option)
#---------------------------------------------------LCC修正-----------------------------------------#
    Vd = Vd-Delta[NumPQ+NumNode-2:NumPQ+NumNode-2+Nc].reshape(Nc)                                   
    Id = Id-Delta[NumPQ+NumNode-2+Nc:NumPQ+NumNode-2+2*Nc].reshape(Nc)
    Kt = Kt-Delta[NumPQ+NumNode-2+2*Nc:NumPQ+NumNode-2+3*Nc].reshape(Nc)
    W= W-Delta[NumPQ+NumNode-2+3*Nc:NumPQ+NumNode-2+4*Nc].reshape(Nc)
    fi = fi-Delta[NumPQ+NumNode-2+4*Nc:NumPQ+NumNode-2+5*Nc]
    # Option['string'] = '\nLCC更新之后的直流电压为：\n'
    # Real(Vd,**Option)
    # Option['string'] = '直流电流为：\n'
    # Real(Id,**Option)
    # Option['string'] = '换流变变比：\n'
    # Real(Kt,**Option)
    # Option['string'] = '控制角\n：'
    # Real(57.3*arccos(W),**Option)
    # Option['string'] = '功率因数：\n'
    # Real(cos(fi),**Option)
#---------------------------------------------------MMC修正-----------------------------------------#
    k = 0
    Ng = 0
    for i in range(VSC_Num):
        if VSC_NodeData[i,2]==3:
            Udc[i] = Udc[i]
            Ng = Ng+1
        else:
            Udc[i] = Udc[i]-Delta[NumPQ+NumNode-2+5*Nc+k]
            k=k+1            
    Idc = Idc-Delta[NumPQ+NumNode-2+5*Nc+VSC_Num-Ng:NumPQ+NumNode-2+5*Nc+VSC_Num-Ng+VSC_Num].reshape(VSC_Num)
    derta = derta-Delta[NumPQ+NumNode-2+5*Nc+VSC_Num-Ng+VSC_Num:NumPQ+NumNode-2+5*Nc+VSC_Num-Ng+2*VSC_Num].reshape(VSC_Num)
    M = M-Delta[NumPQ+NumNode-2+5*Nc+VSC_Num-Ng+2*VSC_Num:NumPQ+NumNode-2+5*Nc+VSC_Num-Ng+3*VSC_Num].reshape(VSC_Num)
    # Option['string'] = '\nVSC更新之后的直流电压为：\n'
    # Real(Udc,**Option)
    # Option['string'] = '直流电流为：\n'
    # Real(Idc,**Option)
    # Option['string'] = '功角：\n'
    # Real(57.3*derta,**Option)
    # Option['string'] = '调制比：\n'
    # Real(M,**Option)
    return(U,Angle,Vd,Id,Kt,W,fi,Udc,Idc,derta,M,MaxError,P,Q)