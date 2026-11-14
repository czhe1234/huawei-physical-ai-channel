"""Seven PPT-led scenes. Scores are PPT records; fields are teaching examples."""
import math
import numpy as np
from scenes_algorithm import (BG, BLUE, YELLOW, GREEN, RED, PURPLE, WHITE,
                              MUTED, COLORS, ease, mix, tint, panel, graph, _PROJ)


def _score_plot(c, scores, t, show_gains=False):
    x0,y0,w,h=255,690,1300,360
    low,high=.56,.70
    ys=lambda s:y0-h*(s-low)/(high-low)
    xs=np.linspace(x0,x0+w,len(scores))
    c.arrow(x0-45,y0,x0+w+100,y0,color=MUTED,width=2)
    c.arrow(x0-45,y0+10,x0-45,y0-h-30,color=MUTED,width=2)
    for val in [.56,.60,.64,.68,.70]:
        yy=ys(val)
        c.line([(x0-54,yy),(x0-44,yy)],color=MUTED,width=2)
        c.text(x0-75,yy,f'{val:.2f}',size=21,color=MUTED,anchor='rm')
    c.text(145,245,'线上评分：局部纵轴 0.56—0.70',size=27,color=MUTED)
    for i,(xx,s) in enumerate(zip(xs,scores)):
        u=ease((t-i*.13)*5)
        yy=ys(s)
        if i:
            c.line([(xs[i-1],ys(scores[i-1])),(xx,yy)],color=BLUE,width=5,progress=u)
        c.circle(xx,yy,10,fill=tint(COLORS[i],u))
        c.text(xx,yy-36,f'{s:.6f}',size=34,color=tint(COLORS[i],u),anchor='mm')
        c.text(xx,y0+46,f'PPT阶段{i+1}',size=28,color=tint(WHITE,u),anchor='mm')
        if show_gains and i:
            c.text((xx+xs[i-1])/2,(yy+ys(scores[i-1]))/2+45,
                   f'+{s-scores[i-1]:.6f}',size=24,color=tint(GREEN,u),anchor='mm')


def scene_evolution(c,t):
    c.text(145,190,'前三个阶段：从基础预测到多源物理信息',size=36,color=WHITE)
    _score_plot(c,[.594822,.636929,.662169],t)
    c.text(185,800,'分数来源：答辩PPT；阶段编号不是开发版本号',size=27,color=MUTED)


def pauli_example(theta):
    x=np.array([1,np.exp(1j*theta),-1,np.exp(1j*theta)],complex)
    f=np.fft.fft(x,norm='ortho')
    return x,f


# Check the complete parameterized counterexample numerically, not by labels.
for _theta in np.linspace(0,math.pi/2,37):
    _x,_f=pauli_example(_theta)
    assert np.allclose(abs(_x)**2,np.ones(4),atol=1e-12)
    assert np.allclose(abs(_f)**2,np.ones(4),atol=1e-12)
assert not np.allclose(pauli_example(0)[0],pauli_example(math.pi/2)[0])


def scene_pauli(c,t):
    theta=math.pi/2*ease((t-.10)/.72)
    x,f=pauli_example(theta)
    c.formula(140,190,r'$x_\theta=[1,e^{i\theta},-1,e^{i\theta}],\quad 0\leq\theta\leq\pi/2$',size=42,color=WHITE)
    c.text(145,285,f'θ = {theta/math.pi:.2f} π',size=32,color=YELLOW)
    c.text(145,340,'信号分量：相位在变化',size=28,color=BLUE)
    for i,v in enumerate(x):
        xx=250+i*390;yy=480
        c.circle(xx,yy,88,outline='#33435A',width=2)
        c.line([(xx-100,yy),(xx+100,yy)],color='#28374A',width=1)
        c.line([(xx,yy-100),(xx,yy+100)],color='#28374A',width=1)
        c.arrow(xx,yy,xx+77*v.real,yy-77*v.imag,color=COLORS[i],width=6)
        c.text(xx,yy+115,f'分量{i+1}',size=25,color=MUTED,anchor='mm')
    for row,(lab,vals,color) in enumerate([('位置域功率',abs(x)**2,BLUE),('FFT域功率',abs(f)**2,GREEN)]):
        yy=650+row*70
        c.text(145,yy+15,lab,size=27,color=color)
        for i,val in enumerate(vals):
            xx=445+i*315
            c.rect(xx,yy,xx+200*val,yy+32,fill=color,radius=3)
            c.text(xx+225,yy+15,f'{val:.1f}',size=25,color=color,anchor='mm')
    c.text(145,830,'Pauli问题：双域模平方恢复；不是泡利排斥原理。此例不充当PAS/PDP唯一性定理。',size=23,color=MUTED)


def scene_reference(c,t):
    c.text(150,195,'两个功率谱目标，仍可能对应多个复信道',size=35,color=WHITE)
    centers=[(400,420),(590,555),(820,390),(950,610)]
    for i,(xx,yy) in enumerate(centers):
        c.circle(xx,yy,62,outline=COLORS[i],width=3)
        ang=(i+.6)*.9+.30*math.sin(t*5+i)
        c.arrow(xx,yy,xx+46*math.cos(ang),yy-46*math.sin(ang),color=COLORS[i],width=4)
        c.text(xx,yy+90,f'候选H{i+1}',size=24,color=MUTED,anchor='mm')
    c.line([(350,310),(960,310)],color='#35445A',width=2,progress=ease(t*4))
    c.text(400,265,'谱约束给出候选范围',size=29,color=BLUE)
    u=ease((t-.25)*4)
    panel(c,1225,300,480,325,'引入基础复信道',GREEN)
    c.formula(1370,405,r'$H_{\rm base}$',size=56,color=tint(GREEN,u))
    c.text(1260,525,'额外的幅度与相位参考',size=28,color=tint(WHITE,u))
    for i,(xx,yy) in enumerate(centers):
        c.line([(xx+65,yy),(1225,465)],color=tint(COLORS[i],.45*u),width=2)
    c.text(160,755,'让重构既接近双谱目标，也不过度偏离已有参考',size=32,color=YELLOW)
    c.text(160,820,'基础参考也有误差；引入参考不等于保证真值恢复或唯一解。',size=25,color=MUTED)


def scene_variational(c,t):
    c.text(140,190,'PPT第9页：把三个要求写成一个目标',size=34,color=WHITE)
    c.formula(140,265,r'$J(H)=\frac{1}{2}d^2(H,\mathcal{C}_{\rm PAS})+\frac{1}{2}d^2(H,\mathcal{C}_{\rm PDP})+\frac{\beta}{2}\Vert H-H_{\rm base}\Vert_F^2$',size=40,color=WHITE)
    labels=[('接近PAS约束集合',BLUE),('接近PDP约束集合',GREEN),('不要过度偏离基础参考',YELLOW)]
    for i,(lab,col) in enumerate(labels):
        u=ease((t-i*.14)*5)
        x=175+i*555
        c.rect(x,380,x+500,442,outline=tint(col,u),radius=10,width=3)
        c.text(x+250,410,lab,size=28,color=tint(col,u),anchor='mm')
    beta=.15+3.85*ease((t-.35)/.55)
    xx=np.linspace(-1.5,2.8,150)
    a,b,base=-.7,1.5,2.1
    f1=.5*(xx-a)**2;f2=.5*(xx-b)**2;f3=beta/2*(xx-base)**2
    full=f1+f2+f3
    px=lambda z:290+(z+1.5)/4.3*1160
    py=lambda z:765-z/32*250
    c.arrow(260,765,1490,765,color=MUTED,width=2)
    c.arrow(275,775,275,480,color=MUTED,width=2)
    for vals,col in [(f1,BLUE),(f2,GREEN),(f3,PURPLE),(full,YELLOW)]:
        c.line([(px(x),py(y)) for x,y in zip(xx,vals)],color=col,width=3,progress=ease((t-.2)*3))
    xmin=(a+b+beta*base)/(2+beta);jmin=.5*(xmin-a)**2+.5*(xmin-b)**2+beta/2*(xmin-base)**2
    c.circle(px(xmin),py(jmin),8,fill=YELLOW)
    c.line([(px(base),765),(px(base),495)],color='#425169',width=2)
    c.text(1510,530,f'β = {beta:.2f}',size=34,color=YELLOW)
    c.text(1510,600,'参考权重',size=27,color=MUTED)
    c.text(140,815,'d是到约束集合的距离，不是原始谱差。下图仅为一维教学示意；原问题不保证唯一全局解。',size=23,color=MUTED)


def scene_solver(c,t):
    pas,pdp,pt,dt,powers=_PROJ
    z=min(8,max(0,t*9-.4));step=min(7,int(z));u=ease(z-step)
    p=(1-u)*pas[step]+u*pas[step+1];d=(1-u)*pdp[step]+u*pdp[step+1]
    c.text(145,190,'在多域目标与基础参考之间协调',size=35,color=WHITE)
    for xx,label,color,vals,target in [(180,'PAS目标',BLUE,p,pt),(1050,'PDP目标',GREEN,d,dt)]:
        c.text(xx,300,label,size=32,color=color)
        c.line([(xx,600),(xx+640,600)],color=MUTED,width=2)
        graph(c,target,xx,355,640,240,color='#52627A',maxv=.65)
        graph(c,vals,xx,355,640,240,color=color,maxv=.65)
    c.arrow(865,425,1000,425,color=YELLOW,width=3)
    c.arrow(1000,515,865,515,color=PURPLE,width=3)
    status='4轮完成' if z>=8 else f'第{min(4,int(z)//2+1)}轮'
    c.text(855,650,status,size=31,color=YELLOW)
    c.formula(170,705,r'$H_{\rm base}\ \longrightarrow\ H^{(1)}\ \longrightarrow\ \cdots\ \longrightarrow\ H^{(4)}$',size=38,color=WHITE)
    c.text(170,775,'保留当前变换域相位；恢复每个位置总功率。这里不声称精确求解J。',size=26,color=MUTED)
    c.text(170,828,'PPT示意5轮；现存V16默认4轮。曲线为合成教学示意。',size=24,color=MUTED)


def scene_results(c,t):
    c.text(145,190,'PPT记录的四阶段整体结果',size=36,color=WHITE)
    _score_plot(c,[.594822,.636929,.662169,.686754],t,show_gains=True)
    u=ease((t-.5)*4)
    c.text(175,790,'累计提升 +0.091932',size=32,color=tint(YELLOW,u))
    c.text(870,790,'答辩PPT记录：西北赛区第1',size=31,color=tint(GREEN,u))
    c.text(175,838,'这些是阶段整体分数，不是隔离单个模块后测得的因果增益。',size=23,color=MUTED)


def scene_recap(c,t):
    steps=[('前处理','物理与数据先验',BLUE),('双流建模','基础复信道 + 传播结构',GREEN),('重构','有参考的多域协调',YELLOW)]
    for i,(name,desc,col) in enumerate(steps):
        u=ease((t-i*.12)*4)
        x=165+i*565
        panel(c,x,230,485,220,name,col)
        c.text(x+30,335,desc,size=29,color=tint(WHITE,u))
        if i<2:c.arrow(x+495,335,x+550,335,color=tint(col,u),width=3)
    questions=['两类先验为何互补？','双谱为何不等于完整H？','基底参考补了什么？']
    for i,q in enumerate(questions):
        u=ease((t-.3-i*.12)*5)
        c.text(250,535+i*85,f'{i+1}   {q}',size=38,color=tint(COLORS[i],u))
    c.text(245,820,'从弱先验，到有基础参考的多域复信道重构。',size=33,color=tint(WHITE,ease((t-.7)*4)))


SCENES={'evolution':scene_evolution,'pauli':scene_pauli,'reference':scene_reference,
        'variational':scene_variational,'solver':scene_solver,
        'results':scene_results,'recap':scene_recap}
