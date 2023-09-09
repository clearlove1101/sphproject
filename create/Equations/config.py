"""
    所有相关参数
    非自由物理量的注释应包含物理量意义、解释、单位、数值出处
    [<单位>, <含义>, <数值>, <数值出处>, <解释>]
    单位<1>表示无单位
    <->表示无
    <?>表示待验证
    <!>表示忘记
    自由变量标注含义、单位、注释
    {<单位>, <含义>, <注释>}
"""


""" 1. 全局参数 """

""" 1.1. 有关SPH """
# [
#   <1>,
#   <平滑核长度 / 粒子间隔>,
#   <1.2~1.3>,
#   <《无网格弱可压缩SPH数值算法及应用扩展/董祥伟》>,
#   <与核函数有关，越大临近粒子搜索半径越大>
# ]
hdx = 1.2

# {<s>, <时间步间隔>, <->}
dt = 5e-4

# {<s>, <运行时间>, <->}
tf = 1.8

""" 1.2. Tillotson """
""" 注意单位 """
# [
#   <->,
#   <->,
#   <->,
#   <Christian R , Joachim S .Numerical aspects of giant impact simulations.2017[2023-07-05].DOI:10.1093/mnras/stx322.>,
#   <
#       下列没特殊说明的所有物理量
#       选取的花岗岩参数
#       下列注释为单位或注释
#       注意这里的单位大多不是国际单位制，需要转换
#   >
# ]

# <->
tillotson_a = 0.5

# <->
tillotson_b = 1.3

# <erg g-1>
tillotson_e0 = 1.6e7

# <m/s> <声速> <越大, 粒子约不容易分开> <?>
tillotson_c0 = 3.e3

# <g/cm3>
tillotson_rho0 = 2700

# <->
tillotson_A = 1.8e10

# <->
tillotson_B = 1.8e10

# <erg g-1>
tillotson_us = 3.5e6

# <erg g-1>
tillotson_us_ = 1.8e7

# <->
tillotson_alpha = 5.

# <->
tillotson_beta = 5.

# <-> <计算声速（用Tillotson的公式计算声速有点奇怪）>
tillotson_gamma = 7

# <erg / g K> <比热容>
tillotson_cv = 7.9e2

# 以下单位均为SI，采用miluphcuda框架内实验用的的多孔玄武岩参数
# 网址：https://github.com/christophmschaefer/miluphcuda/blob/main/examples/impact/analyze-results/plot_p_alpha_convergence.py
tillotson_p_e = 2.0e8    # 材料从弹性状态转到塑性状态时的压力
tillotson_p_t = 6.80e7   # 标明一个转换时的压力（压溃曲线斜率从n1->n2)
tillotson_p_s = 2.0e9   # 材料完全压实时的压力（alpha = 1）
tillotson_alpha_0 = 1.25    # 材料从弹性状态转到塑性状态时的膨胀率
tillotson_alpha_e = 1.25   # alpha_e = alpha(p=p_e)
tillotson_alpha_t = 1.90   # 标明一个转换时的膨胀率（压溃曲线斜率从n1->n2)
tillotson_n1 = 12.0    # 挤压曲线斜率1
tillotson_n2 = 3.0     # 挤压曲线斜率2
tillotson_crushcurve_style = 0   # 挤压曲线样式，0代表二次挤压曲线，1代表jutzi提出的实在的/陡峭的挤压曲线

""" 1.3. ROCK """
# [
#   <->,
#   <->,
#   <->,
#   <任健康,张庆明,刘文近,等.依兰陨石坑形成过程数值模拟研究[J].爆炸与冲击, 2023, 43(3):11.>,
#   <
#       下列没特殊说明的所有物理量
#       选取的花岗岩参数
#       下列注释为单位或注释
#   >
# ]

# <k> <零压缩点>
rock_Tm0 = 1673

# <pa>
rock_a = 6e9

# <->
rock_c = 3

# <->
rock_Xi = 1.2

# <-> <低压下最小失效应变>
rock_Epsilonfb = 1e-4

# <pa> <压缩失效应力>
rock_Pc = 3e8

# <pa-1>
rock_B = 1e-11

# <pa> <未损伤内聚力>
rock_Yi0 = 1e7

# <-> <未损伤内摩擦系数>
rock_miui = 2

# <pa> <未损伤高压强度极限>
rock_Yim = 2.5e9

# <pa> <0压强度>
rock_Yd0 = 1e4

# <-> <损伤内摩擦>
rock_miud = 0.6

# <pa> <极限强度>
rock_Ydm = 2.5e9

""" 2. 小行星(靶体)参数 """
# {<m>, <半径>, <->}
r_asteroid = 30

# {<kg/m2>, <初始密度>, <大理石体密度(百度)强转>}
rho_asteroid = 2700

# {<m>, <粒子间隔>, <->}
dx_asteroid = 3.

# {<m>, <质心坐标>, <->}
center_asteroid = [0, 0, 0]

# {<->, <粒子群名称>, <->}
name_asteroid = "asteroid"

omg_asteroid = [1., 1., 1.]

# {<m>, <平滑核长度>, <->}
h_asteroid = hdx * dx_asteroid

# {<pa/s>, <剪力张量变化率>, <有公式算，初始几都会被覆盖>}
as00_asteroid = 0
as01_asteroid = 0
as11_asteroid = 0

# {<pa>, <剪力张量>, <假设初始每个粒子都不受剪力>}
s00_asteroid = 0
s01_asteroid = 0
s11_asteroid = 0

# {<pa>, <旋转张量>, <被覆盖>}
r00_asteroid = 0
r01_asteroid = 0
r11_asteroid = 0

# [
#   <pa>,
#   <剪切模量>,
#   <~e9>,
#   <百度(某种岩石)>,
#   <计算剪切力张量变化率用，书(5-28)>
# ]
G_asteroid = 4.9e9

# {<K>, <初始温度>, <->}
T_asteroid = 0

# {<pa>, <屈服强度>, <ROCK会覆盖>}
Y_asteroid = 2.e8

# {<?>, <塑性形变>, <应力推回算法会计算累积量>}
dam_asteroid = 0

# {<->, <损伤>, <ROCK计算>}
D_asteroid = 0

# [
#   <J/g>,
#   <比内能>,
#   <!>,
#   <!>,
#   <好多方程用>
# ]
e_asteroid = T_asteroid * tillotson_cv  # 2维1.6e2比较好

# {<J/sg>, <比内能变化率>, <有公式覆盖，书(5-26)>}
ae_asteroid = 0

# {<1/s>, <速度对位置的梯度>, <->}
v00_asteroid = 0
v01_asteroid = 0
v10_asteroid = 0
v11_asteroid = 0

# {<?>, <?>, <不加报错>}
cwij_asteroid = 0


""" 3. 探测器(impactor)参数 """
# {<m>, <半径>, <->}
r_impactor = 4

# {<g/m2>, <初始密度>, <大理石体密度(百度)强转>}
rho_impactor = 2700

# {<m>, <粒子间隔>, <->}
dx_impactor = 3.

# {<m>, <质心坐标>, <->}
center_impactor = [0, 0, r_impactor + r_asteroid]

# {<->, <粒子群名称>, <->}
name_impactor = "impactor"

# {<m>, <平滑核长度>, <->}
h_impactor = hdx * dx_asteroid

# {<pa/s>, <剪力张量变化率>, <有公式算，初始几都会被覆盖>}
as00_impactor = 0
as01_impactor = 0
as11_impactor = 0

# {<pa>, <剪力张量>, <假设初始每个粒子都不受剪力>}
s00_impactor = 0
s01_impactor = 0
s11_impactor = 0

# {<pa>, <旋转张量>, <被覆盖>}
r00_impactor = 0
r01_impactor = 0
r11_impactor = 0
# [
#   <pa>,
#   <剪切模量>,
#   <~e9>,
#   <百度(某种岩石)>,
#   <计算剪切力张量变化率用，书(5-28)>
# ]
G_impactor = 4.9e9

# {<K>, <初始温度>, <->}
T_impactor = 0

# {<pa>, <屈服强度>, <ROCK会覆盖>}
Y_impactor = 2.e8

# {<?>, <塑性形变>, <应力推回算法会计算累积量>}
dam_impactor = 0

# {<->, <损伤>, <ROCK计算>}
D_impactor = 0

# [
#   <J/g>,
#   <比内能>,
#   <?>,
#   <?>,
#   <可以用温度算>
# ]
e_impactor = T_impactor * tillotson_cv # 2维1.6e2比较好

# {<J/sg>, <比内能变化率>, <有公式覆盖，书(5-26)>}
ae_impactor = 0

# {<1/s>, <速度对位置的梯度>, <->}
v00_impactor = 0
v01_impactor = 0
v10_impactor = 0
v11_impactor = 0

# {<?>, <?>, <不加报错>}
cwij_impactor = 0

""" 4. 撞击事件参数 """
# 需要转化的量 转化函数在tools里写

# {<m/s>, <撞击速度>, <要转化成正交速度uvw>}
V_impactor = 5000

# {矢量}
direction_impactor = [1., 1., -4.]

# {<度/s>, <探测器自转角速度，角速度矢量三个分量>, <3维为矢量>}
omg_impactor = [10., 10., 10.]

