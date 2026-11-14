"""PPT-aligned teaching scenes; synthetic diagrams, no experiment data."""
from math import pi, sin, cos
import numpy as np
from scenes_foundation import axes, phase, mix, pulse

BG='#0B101B'; BLUE='#58C4DD'; YELLOW='#FFFF00'; GREEN='#83C167'
RED='#FC6255'; PURPLE='#9A72AC'; WHITE='#F4F5F7'; MUTED='#9AA9BD'; GRID='#263348'


def box(c,x,y,w,h,title,color=BLUE,small=None):
    c.rect(x,y,x+w,y+h,fill='#101C2D',outline=color,width=2,radius=14)
    c.text(x+w/2,y+h/2-12 if small else y+h/2,title,size=30,color=color,anchor='mm')
    if small:c.text(x+w/2,y+h-27,small,size=22,color=MUTED,anchor='mm')


def scene_standard(c,t):
    c.text(125,200,'3GPP：通信标准合作计划',size=40,color=WHITE)
    box(c,130,360,340,155,'共同的技术规范',BLUE,'让不同设备能够互通')
    c.arrow(500,435,710,435,color=BLUE,width=4)
    box(c,750,285,420,130,'通信系统与接口',GREEN,'如何发送、接收与连接')
    box(c,750,530,420,150,'参考信道模型',YELLOW,'TR 38.901：0.5—100 GHz')
    u=phase(c,t,.12,.43)
    c.line([(650,435),(650,350),(750,350)],color=GREEN,width=4,progress=u)
    c.line([(650,435),(650,605),(750,605)],color=YELLOW,width=4,progress=u)
    if t>.35:
        box(c,1320,325,400,280,'传播的公共参考',PURPLE,'阵列、极化与传播统计')
        c.line([(1170,605),(1235,605),(1235,465),(1320,465)],color=PURPLE,width=4,progress=phase(c,t,.35,.65))
    c.text(170,775,'标准参考不是这个场景的真实信道',size=34,color=YELLOW)
    if t>.65:c.text(1040,775,'具体场景仍需要数据与地图约束',size=31,color=WHITE)
    c.circle(1280+45*sin(2*pi*t),240,7,fill=BLUE)


def scene_antenna(c,t):
    c.text(125,195,'代码实际采用：TR 38.901 阵元方向图与 V/H 极化',size=35,color=WHITE)
    cx,cy,R=450,515,205
    for rad in [R/3,2*R/3,R]:c.circle(cx,cy,rad,outline=GRID,width=1)
    c.arrow(cx-255,cy,cx+265,cy,color=MUTED,width=2)
    c.arrow(cx,cy+235,cx,cy-235,color=MUTED,width=2)
    phi=np.linspace(-pi,pi,361)
    db=-np.minimum(12*(np.degrees(phi)/65)**2,30)
    p=10**(db/10)
    pts=[(cx+R*r*cos(a),cy-R*r*sin(a)) for a,r in zip(phi,p)]
    c.line(pts,color=BLUE,width=4,progress=phase(c,t,0,.32))
    a=(-130+260*phase(c,t,.18,.90))*pi/180
    gain=10**(-min(12*(a*180/pi/65)**2,30)/10)
    c.arrow(cx,cy,cx+R*cos(a),cy-R*sin(a),color=MUTED,width=2)
    c.circle(cx+R*gain*cos(a),cy-R*gain*sin(a),8,fill=YELLOW)
    c.text(215,765,'半径表示相对功率',size=28,color=BLUE)
    c.text(690,390,'V',size=38,color=GREEN)
    c.arrow(720,545,720,445,color=GREEN,width=4)
    c.text(780,520,'H',size=38,color=PURPLE)
    c.arrow(750,490,850,490,color=PURPLE,width=4)
    x0,y0,w,h=1060,635,650,290
    axes(c,x0,y0,w,h,'方位角 φ','增益 / dB')
    c.line([(x0+(a+180)/360*w,y0-(v+30)/30*h) for a,v in zip(np.degrees(phi),db)],color=BLUE,width=4,progress=phase(c,t,.15,.52))
    for deg in [-180,-90,0,90,180]:c.text(x0+(deg+180)/360*w,y0+30,str(deg),size=21,color=MUTED,anchor='mt')
    for v in [-30,-15,0]:c.text(x0-18,y0-(v+30)/30*h,str(v),size=21,color=MUTED,anchor='rm')
    c.formula(1040,225,r'A_H(\phi)=-\min\left[12(\phi/65^\circ)^2,30\right]',size=35,color=BLUE)
    if t>.6:c.text(1010,755,'标准参考 + 地图候选 → 网络校准',size=31,color=YELLOW)
    c.text(1010,815,'这不等同于采用完整的 UMi / UMa 信道模型',size=25,color=MUTED)


def scene_problem(c,t):
    c.text(125,195,'已知有限位置，不等于知道所有位置',size=38,color=WHITE)
    x0,y0,w,h=160,650,850,340
    axes(c,x0,y0,w,h,'空间位置（示意）','某个信道分量')
    xs=np.linspace(-1,1,200);obs=np.array([-.90,-.40,.45,.95])
    base=.45+.15*np.sin(3*xs)
    prod=np.prod(xs[:,None]-obs,axis=1);prod=prod/max(abs(prod))*.22
    for k,(a,col) in enumerate([(-1,GREEN),(0,BLUE),(1,PURPLE)]):
        vals=base+a*prod
        c.line([(x0+(x+1)/2*w,y0-v*h/.85) for x,v in zip(xs,vals)],color=col,width=3,progress=phase(c,t,.02+k*.09,.35+k*.09))
    for x in obs:
        y=.45+.15*sin(3*x)
        c.circle(x0+(x+1)/2*w,y0-y*h/.85,8,fill=YELLOW)
    q=.05;xx=x0+(q+1)/2*w
    c.line([(xx,y0),(xx,y0-h)],color=MUTED,width=2)
    c.text(xx,y0+32,'未知位置',size=27,color=YELLOW,anchor='mt')
    c.text(170,785,'不同空间规律，可以吻合同一批已知点',size=30,color=WHITE)
    c.formula(1170,245,r'H(x)\in\mathbb{C}^{256\times4\times192}',size=40,color=BLUE)
    c.circle(1190,485,12,fill=YELLOW)
    for i,(col,dy) in enumerate([(GREEN,-125),(BLUE,0),(PURPLE,125)]):
        end=(1640,485+dy)
        c.line([(1190,485),(1370,485),(1500,485+dy),end],color=col,width=4,progress=phase(c,t,.35+i*.06,.65+i*.06))
        c.formula(1680,465+dy,rf'H_{i+1}',size=39,color=col)
    c.text(1150,720,'需要空间结构与物理约束',size=33,color=YELLOW)
    c.text(1150,785,'高维输出本身不是“数学欠定”的证明',size=26,color=MUTED)


def map_frame(c,x0,y0,w=670,h=400):
    for i in range(8):c.line([(x0+i*w/7,y0),(x0+i*w/7,y0+h)],color=GRID,width=1)
    for j in range(6):c.line([(x0,y0+j*h/5),(x0+w,y0+j*h/5)],color=GRID,width=1)
    c.rect(x0+200,y0+90,x0+420,y0+115,fill='#354359')
    q=(x0+470,y0+255)
    return q


def scene_weakpriors(c,t):
    c.text(140,195,'弱先验一：邻域信道基底',size=35,color=BLUE)
    c.text(1060,195,'弱先验二：几何传播候选',size=35,color=GREEN)
    q1=map_frame(c,145,315);q2=map_frame(c,1070,315)
    donors=[(65,90),(130,260),(300,320),(570,90),(590,330)]
    for x,y in donors:
        p=(145+x,315+y);c.circle(*p,8,fill=BLUE)
        c.line([p,q1],color=mix(BG,BLUE,.5),width=2,progress=phase(c,t,0,.35))
        xx=145+x+(q1[0]-(145+x))*((3*t)%1)
        yy=315+y+(q1[1]-(315+y))*((3*t)%1)
        c.circle(xx,yy,4,fill=BLUE)
    tx=(1120,580)
    c.circle(*tx,10,fill=GREEN)
    routes=[[tx,q2],[tx,(1390,430),q2],[tx,(1635,650),q2]]
    for i,pts in enumerate(routes):c.line(pts,color=[GREEN,PURPLE,BLUE][i],width=4,progress=phase(c,t,.20+i*.08,.48+i*.08))
    pulse(c,*q1,t,YELLOW);pulse(c,*q2,t,YELLOW)
    c.text(160,760,'数据锚定，但精细结构可能被平均',size=30,color=BLUE)
    c.text(1070,760,'提供结构，但贡献大小仍不确定',size=30,color=GREEN)
    if t>.60:c.text(600,825,'同一目标位置，两类互补而不完美的信息',size=30,color=YELLOW)


def peak_field(c,x0,y0,w,h,peaks,color,progress=1):
    for i in range(13):c.line([(x0+i*w/12,y0),(x0+i*w/12,y0+h)],color=GRID,width=1)
    for j in range(7):c.line([(x0,y0+j*h/6),(x0+w,y0+j*h/6)],color=GRID,width=1)
    for x,y,m in peaks:
        r=(8+22*m)*progress
        c.circle(x0+x*w,y0+y*h,r,fill=color)


def scene_common(c,t):
    c.text(140,195,'从互补弱先验中寻找共同可行结构',size=36,color=WHITE)
    left=[(.22,.32,.9),(.48,.60,.65),(.74,.28,.7),(.80,.78,.4)]
    right=[(.25,.34,.7),(.52,.58,.9),(.76,.30,.6),(.15,.82,.5)]
    u=phase(c,t,.27,.70)
    x1=150+210*u;x2=1070-710*u
    y=320;w=650;h=370
    peak_field(c,x1,y,w,h,left,mix(BLUE,BG,.35*u),phase(c,t,0,.24))
    peak_field(c,x2,y,w,h,right,mix(GREEN,BG,.35*u),phase(c,t,.08,.30))
    if t<.56:
        c.text(x1,y-55,'邻域信道的谱结构',size=29,color=BLUE)
        c.text(x2,y+h+40,'地图候选的谱结构',size=29,color=GREEN)
    if t>.59:
        grow=phase(c,t,.59,.87)
        for x,z,m in [(.235,.33,.9),(.50,.59,.9),(.75,.29,.8)]:
            c.circle(360+x*w,y+z*h,10+22*m*grow,fill=YELLOW)
        c.text(1120,370,'共同支持的角度',size=34,color=YELLOW)
        c.text(1120,445,'共同支持的时延',size=34,color=YELLOW)
        c.text(1120,520,'共同可行的功率分配',size=34,color=YELLOW)
    c.text(180,805,'教学示意：共同特征不等于已证明严格降维，也不保证候选是真实路径',size=28,color=MUTED)


def scene_pipeline(c,t):
    c.text(130,195,'三阶段：构造先验 → 预测目标双谱 → 重构完整信道',size=35,color=WHITE)
    blocks=[(150,'前处理',BLUE,'数据与地图构造弱先验'),(750,'双流预测',PURPLE,'预测 PAS / PDP 目标'),(1360,'后处理',GREEN,'约束合成完整复信道')]
    for i,(x,title,col,small) in enumerate(blocks):
        g=phase(c,t,.05+i*.12,.30+i*.12)
        box(c,x,350,410,195,title,mix(BG,col,g),small)
    c.arrow(580,445,725,445,color=BLUE,width=4)
    c.arrow(1180,445,1335,445,color=PURPLE,width=4)
    c.text(180,600,'邻域基底 + 几何候选',size=29,color=BLUE)
    c.text(790,600,'谱形网络 + 候选网络',size=29,color=PURPLE)
    c.text(1400,600,'交替投影 + 总功率恢复',size=28,color=GREEN)
    u=phase(c,t,.48,.86)
    c.line([(355,545),(355,745),(1560,745),(1560,545)],color=YELLOW,width=4,progress=u)
    if u>.98:c.arrow(1560,650,1560,548,color=YELLOW,width=4)
    c.formula(770,700,r'H_{\mathrm{base}}',size=38,color=YELLOW)
    c.text(700,795,'基础复信道作为独立支路进入重构',size=30,color=YELLOW)


def scene_spectral(c,t):
    c.text(140,195,'谱上的局部峰形可以复用',size=37,color=WHITE)
    xs=np.linspace(0,1,100)
    b=.12+.8*np.exp(-((xs-.27)/.065)**2)+.50*np.exp(-((xs-.69)/.09)**2)
    residual=.15*np.sin(2*pi*xs)*np.exp(-((xs-.5)/.4)**2)
    corrected=b*np.exp(residual*phase(c,t,.35,.90))
    for yoff,title,col in [(310,'PDP：一维局部邻域',BLUE),(640,'PAS：二维空域邻域',GREEN)]:
        c.text(150,yoff-35,title,size=30,color=col)
        if col==BLUE:
            for values,color in [(b,GRID),(corrected,BLUE)]:c.line([(170+x*700,yoff+220-v*170) for x,v in zip(xs,values)],color=color,width=4)
            xx=190+530*((t*1.6)%1)
            c.rect(xx,yoff+5,xx+110,yoff+240,outline=YELLOW,width=3,radius=8)
        else:
            for row in range(4):
                for k in range(14):
                    val=.10+.85*np.exp(-((k-4)/2)**2-(row-1.5)**2/2)+.45*np.exp(-((k-10)/2)**2-(row-2)**2/2)
                    x=180+k*47;y=yoff+35+row*36
                    c.rect(x,y,x+41,y+29,fill=mix(BG,GREEN,min(1,val)),radius=3)
            col=int(t*15)%10
            c.rect(175+col*47,yoff+25,175+(col+3)*47,yoff+185,outline=YELLOW,width=3)
    c.arrow(930,510,1080,510,color=WHITE,width=4)
    box(c,1120,325,585,170,'周期卷积 + 有界残差',PURPLE,'1D PDP / 2D PAS')
    c.formula(1090,570,r'\log P_{\mathrm{base}}+s\tanh(\Delta)',size=42,color=YELLOW)
    c.text(1110,665,'再归一化为目标功率分布',size=30,color=WHITE)
    c.text(1110,765,'局部修正，而非自由生成整块复信道',size=28,color=MUTED)


def scene_confidence(c,t):
    c.text(130,195,'两个预测来源，还需要学习何时相信谁',size=37,color=WHITE)
    box(c,140,320,470,150,'CNN 谱结构预测',BLUE,'PAS / PDP')
    box(c,140,600,470,150,'Attention 候选状态',GREEN,'结构化候选预测')
    u=phase(c,t,.15,.48)
    c.line([(610,395),(745,395),(745,535),(850,535)],color=BLUE,width=4,progress=u)
    c.line([(610,675),(745,675),(745,535),(850,535)],color=GREEN,width=4,progress=u)
    box(c,850,440,300,190,'MLP 置信门',PURPLE,'由训练数据校准')
    c.arrow(1175,535,1310,535,color=PURPLE,width=4)
    box(c,1340,325,400,135,'PAS 目标',YELLOW)
    box(c,1340,620,400,135,'PDP 目标',YELLOW)
    c.line([(1255,535),(1255,393),(1340,393)],color=YELLOW,width=3,progress=phase(c,t,.43,.67))
    c.line([(1255,535),(1255,688),(1340,688)],color=YELLOW,width=3,progress=phase(c,t,.43,.67))
    a=.5+.32*sin(2*pi*phase(c,t,.25,1))
    c.line([(860,740),(1130,740)],color=GRID,width=12)
    c.line([(860,740),(860+270*a,740)],color=PURPLE,width=12)
    c.circle(860+270*a,740,12,fill=YELLOW)
    c.text(860,772,'0',size=24,color=MUTED);c.text(1130,772,'1',size=24,color=MUTED,anchor='rt')
    c.text(840,825,'融合权重 ≠ 注意力自动满足物理守恒',size=26,color=MUTED)


SCENES={'standard':scene_standard,'antenna':scene_antenna,'problem':scene_problem,
        'weakpriors':scene_weakpriors,'common':scene_common,'pipeline':scene_pipeline,
        'spectral':scene_spectral,'confidence':scene_confidence}
