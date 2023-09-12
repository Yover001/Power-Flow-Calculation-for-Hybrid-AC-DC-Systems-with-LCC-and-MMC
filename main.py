import datetime
import TY_NetData
import Polar
import numpy as np
from math import pi,sqrt
from numpy import arctan,sin,cos,arccos
from NR import PolarNR
from OutTxt import SingleTxt,StringTxt,Real
Out_Path = 'Cal-Process.txt'
starttime = datetime.datetime.now()
#------------------------------------------网络信息读取------------------------------------------------#
# 获取直流系统信息
RootPath = 'C:\\Users\\lenovo\\Desktop\\统一迭代 - 双平衡节点\\Data\\' 
VSC_Node = RootPath+'VSC_NodeData.txt'
LCC_Node = RootPath+'LCC_NodeData.txt'    
DC_Line = RootPath+'DC_LineData.txt'
VSC_NodeData = TY_NetData.VSC_GetNodeData(VSC_Node,show=1)
LCC_NodeData = TY_NetData.LCC_GetNodeData(LCC_Node,show=1)
DC_LineData = TY_NetData.DC_GetLineData(DC_Line,show=1)
Ydc=TY_NetData.GetYdc(VSC_NodeData,LCC_NodeData,DC_LineData,path=Out_Path,width=6)
#获取交流系统信息
FilePath_Node = RootPath+'NodeData.txt'    
FilePath_Line = RootPath+'LineData.txt'
NodeData = TY_NetData.AC_GetNodeData(FilePath_Node,show=1)
LineData = TY_NetData.AC_GetLineData(FilePath_Line,show=1)
Y = TY_NetData.GetY(NodeData,LineData,path=Out_Path,width=6)                           
AC_LCC=LCC_NodeData[:,0]                            #相连的交流节点(LCC)
AC_LCC=AC_LCC.astype(int)-1
AC_VSC=VSC_NodeData[:,0]                            #相连的交流节点(VSC)
AC_VSC=AC_VSC.astype(int)-1
#----------------------------------------------交直流潮流计算----------------------------------------#
StringTxt(path=Out_Path,string='交直流混合系统潮流计算',fmt='w')
S0 = '-'
for i in range(130):
    S0 = S0+'-' 
StringTxt(path=Out_Path,string=S0)
U,Angle = Polar.AC_Polar(NodeData[:,8],NodeData[:,9],path=Out_Path,width=9)                         
Vd,Id,kt,W,fi = Polar.LCC_Polar(LCC_NodeData,path=Out_Path,width=9)                                 
Udc,Idc,derta,M = Polar.VSC_Polar(VSC_NodeData,path=Out_Path,width=9)                              
# 迭代
Iter = 0
Tol = 1e-5
MaxIter = 15
while True:
	Iter = Iter + 1
	U,Angle,Vd,Id,kt,W,fi,Udc,Idc,derta,M,MaxError,P,Q= PolarNR(U,Angle,Vd,Id,kt,W,fi,Udc,Idc,derta,M,Ydc,Y,NodeData,LCC_NodeData,VSC_NodeData,AC_LCC,AC_VSC,Tol,path=Out_Path,width=9)
	print(MaxError)
	if Iter>MaxIter or MaxError<Tol:
		break
	# 结束交流循环
if MaxError<Tol:
	SingleTxt(Iter-1,path=Out_Path,string='交直流迭代完成，更新次数为：')
	SingleTxt(MaxError,path=Out_Path,string='最大误差为：')
#-------------------------------------------AC计算结果---------------------------------------------#
	Real(U,path=Out_Path,string=S0+'\nAC电压：\n')
	Real(Angle,path=Out_Path,string='相角：\n')
#-------------------------------------------LCC计算结果---------------------------------------------#
	Real(Vd,path=Out_Path,string=S0+'\nLCC直流电压：\n')
	Real(Id,path=Out_Path,string='LCC直流电流：\n')
	Real(kt,path=Out_Path,string='换流变变比：\n')
	Real(57.3*arccos(W),path=Out_Path,string='控制角：\n')
	Real(cos(fi),path=Out_Path,string='功率因数：\n')
#-------------------------------------------VSC计算结果---------------------------------------------#
	Real(Udc,path=Out_Path,string=S0+'\nVSC直流电压：\n')
	Real(Idc,path=Out_Path,string='VSC直流电流：\n')
	Real(57.3*derta,path=Out_Path,string='功角：\n')
	Real(M,path=Out_Path,string='调制比：\n')
else:
	SingleTxt(MaxError,path=Out_Path,string='结果不收敛!')
	print ('潮流不收敛')
# TIME
endtime = datetime.datetime.now()
print (endtime - starttime)