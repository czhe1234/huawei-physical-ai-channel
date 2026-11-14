"""Animated V16 concepts. All numerical fields below are teaching examples."""
import math
import numpy as np

BG = '#0B101B'
BLUE = '#58C4DD'
YELLOW = '#FFFF00'
GREEN = '#83C167'
RED = '#FC6255'
PURPLE = '#9A72AC'
WHITE = '#F4F5F7'
MUTED = '#9AA9BD'
COLORS = [BLUE, GREEN, YELLOW, PURPLE, RED]


def clamp(x):
    return max(0.0, min(1.0, float(x)))


def ease(x):
    x = clamp(x)
    return x*x*(3-2*x)


def mix(a, b, u):
    return a + (b-a)*u


def tint(color, u):
    u = clamp(u)
    a = [int(BG[i:i+2], 16) for i in (1, 3, 5)]
    b = [int(color[i:i+2], 16) for i in (1, 3, 5)]
    return '#' + ''.join(f'{round(mix(x,y,u)):02x}' for x,y in zip(a,b))


def panel(c, x, y, w, h, label, color=BLUE):
    c.rect(x,y,x+w,y+h,outline='#26354B',radius=16,width=2)
    c.text(x+22,y+16,label,size=27,color=color)


def axes(c, ox, oy, r):
    c.arrow(ox-r,oy,ox+r,oy,color='#54647A',width=2)
    c.arrow(ox,oy+r,ox,oy-r,color='#54647A',width=2)
    c.text(ox+r-40,oy+12,'实部',size=22,color=MUTED)
    c.text(ox+14,oy-r+8,'虚部',size=22,color=MUTED)
    c.circle(ox,oy,r*.72,outline='#26354B',width=2)


def graph(c, values, x,y,w,h,color=BLUE, maxv=None, progress=1):
    values = np.asarray(values,float)
    mx = float(np.max(values)) if maxv is None else maxv
    mx = max(mx,1e-12)
    pts = [(x+i*w/(len(values)-1), y+h-h*v/mx) for i,v in enumerate(values)]
    c.line(pts,color=color,width=4,progress=progress)


def scene_align(c,t):
    axes(c,460,500,240)
    u=ease((t-.28)/.40)
    phi=math.pi*(1-u)
    a=np.array([160.,0.]);b=160*np.array([math.cos(phi),-math.sin(phi)])
    c.arrow(460,500,620,500,color=BLUE,width=7)
    c.arrow(460,500,460+b[0],500+b[1],color=RED,width=7)
    avg=(a+b)/2
    c.arrow(460,500,460+avg[0],500+avg[1],color=YELLOW,width=9)
    c.text(230,210,'同样的幅度，不同的方向',size=33,color=WHITE)
    c.text(245,775,'蓝：邻点一   红：邻点二   黄：平均',size=24,color=MUTED)
    panel(c,850,225,910,555,'先补偿主要相位变化，再组合')
    c.formula(910,310,r'$H_1=1,\quad H_2=e^{j\phi}$',size=44,color=WHITE)
    c.formula(910,400,r'$(H_1+H_2)/2$',size=44,color=YELLOW)
    c.text(1320,405,f'平均幅度  {np.linalg.norm(avg)/160:.2f}',size=31,color=YELLOW)
    c.formula(910,510,r'$H_i^{\rm aligned}=H_i e^{j\Delta r_i(\beta+\gamma f)}$',size=37,color=tint(GREEN,ease((t-.35)*5)))
    c.text(910,625,'从训练信道估计主要旋转量',size=31,color=tint(WHITE,ease((t-.45)*5)))
    c.text(910,685,'这是近似补偿，不是逐路径精确恢复',size=28,color=tint(MUTED,ease((t-.60)*5)))


def scene_interpolate(c,t):
    rng=np.random.default_rng(11)
    points=rng.uniform([180,295],[880,690],size=(15,2))
    q=np.array([525+28*math.sin(t*math.pi*2),485+18*math.sin(t*math.pi*3)])
    dist=np.linalg.norm(points-q,axis=1)
    ids=np.argsort(dist)[:5]
    weights=1/(dist[ids]**2+1);weights/=weights.sum()
    c.text(170,205,'附近的已知点，提供候选信道',size=32,color=WHITE)
    for i,p in enumerate(points):
        c.circle(*p,7,fill=BLUE)
        if i in ids:
            c.line([tuple(p),tuple(q)],color=tint(BLUE,.45),width=2,progress=ease(t*5))
    c.circle(*q,13,outline=YELLOW,width=4)
    c.text(q[0]+25,q[1]-10,'目标位置',size=27,color=YELLOW)
    for j,(idx,w) in enumerate(zip(ids,weights)):
        x=180+j*132
        c.rect(x,780-w*300,x+55,780,fill=COLORS[j],radius=4)
        c.text(x,792,f'{w:.2f}',size=22,color=MUTED)
    c.text(170,720,'近邻权重随目标位置变化',size=25,color=MUTED)
    panel(c,1030,220,720,590,'从加权平均到局部趋势')
    c.formula(1080,310,r'$\widehat H(q)=\sum_i w_i(q)H_i$',size=39,color=BLUE)
    u=ease((t-.30)*4)
    for j in range(7):
        yy=490+j*33
        c.line([(1110,yy+30*u),(1610,yy-55*u)],color=tint(PURPLE,.4+.08*j),width=2,progress=u)
    c.formula(1090,740,r'$H(x,y)\approx a+b_x\Delta x+b_y\Delta y$',size=32,color=tint(GREEN,ease((t-.45)*4)))
    c.text(1080,410,'局部仿射：允许信道随位置有斜率',size=27,color=tint(WHITE,u))


def scene_experts(c,t):
    center=(580,500)
    reveal=ease(t*2.7)
    active=int(t*34)%17
    for i in range(17):
        th=2*math.pi*i/17-math.pi/2
        x=center[0]+260*math.cos(th);y=center[1]+220*math.sin(th)
        col=COLORS[i%5]
        c.line([(x,y),center],color=tint(col,.25+.55*(i==active)),width=2,progress=reveal)
        c.circle(x,y,17,fill=tint(col,reveal))
        c.text(x,y,str(i+1),size=18,color=BG,anchor='mm')
    c.circle(*center,66,fill='#132637',outline=BLUE,width=3)
    c.text(*center,'17',size=47,color=WHITE,anchor='mm')
    c.text(340,785,'邻域尺度与空间假设不同',size=29,color=MUTED)
    c.arrow(930,500,1110,500,color=WHITE,width=3)
    c.text(1100,235,'学习复数融合权重',size=35,color=WHITE)
    for i in range(5):
        x=1180+i*115;y=390
        c.circle(x,y,39,outline='#33435A',width=2)
        a=.8*math.sin(3*t+i)+i
        c.arrow(x,y,x+32*math.cos(a),y-32*math.sin(a),color=COLORS[i],width=4)
    c.formula(1100,505,r'$H_{\rm mix}=\sum_{e=1}^{17}w_e H_e$',size=46,color=YELLOW)
    c.text(1100,625,'复数权重：可调整幅度和相位',size=29,color=tint(WHITE,ease((t-.25)*4)))
    c.text(1100,690,'岭正则：限制不稳定的大权重',size=29,color=tint(GREEN,ease((t-.45)*4)))
    c.text(1100,755,'这些权重不是概率',size=27,color=tint(MUTED,ease((t-.65)*5)))


def scene_geometry(c,t):
    c.text(165,205,'地图提出传播候选',size=34,color=WHITE)
    for b in [(300,285,405,490),(520,570,730,690),(780,280,870,500)]:
        c.rect(*b,fill='#172334',outline='#53627A',width=2)
    a=(195,610);b=(915,395)
    c.circle(*a,13,fill=BLUE);c.circle(*b,13,fill=YELLOW)
    c.text(150,655,'基站',size=27,color=BLUE);c.text(895,430,'目标',size=27,color=YELLOW)
    paths=[[a,b],[a,(480,250),b],[a,(475,760),(930,760),b],[a,(230,250),(720,250),b]]
    for i,path in enumerate(paths):
        pr=ease((t-i*.07)*3)
        c.line(path,color=tint(COLORS[i],.75),width=3,progress=pr)
        if pr>.9:
            lens=np.array([np.linalg.norm(np.subtract(path[j+1],path[j])) for j in range(len(path)-1)])
            walk=((t*2+i*.21)%1)*lens.sum()
            for j,L in enumerate(lens):
                if walk<=L:
                    q=np.asarray(path[j])+(np.asarray(path[j+1])-path[j])*walk/L
                    c.circle(*q,6,fill=COLORS[i]);break
                walk-=L
    panel(c,1080,225,670,575,'多次传播采样 → 稳定候选')
    for i in range(128):
        x=1130+(i%16)*35;y=360+(i//16)*35
        col=COLORS[(i*7)%5]
        pulse=.3+.7*(.5+.5*math.sin(t*24*math.pi+i))
        c.circle(x,y,6,fill=tint(col,pulse*ease((t-.15)*4)))
    c.text(1130,270,'每个位置保留128个候选',size=28,color=WHITE)
    c.text(1130,670,'24个随机种子：传播采样',size=29,color=tint(YELLOW,ease((t-.4)*4)))
    c.text(1130,720,'不是24次网络训练',size=29,color=tint(MUTED,ease((t-.55)*4)))
    c.text(165,800,'候选仍需训练信道监督与网络校准',size=30,color=GREEN)


def scene_calibrate(c,t):
    x=np.linspace(0,1,96)
    base=.08+np.exp(-((x-.3)/.10)**2)+.6*np.exp(-((x-.70)/.07)**2)
    delta=.6*np.sin(2*np.pi*x)-.3*np.cos(6*np.pi*x)
    adjusted=np.exp(np.log(base)+.4*np.tanh(delta))
    corrected=(1-ease((t-.18)*3))*base+ease((t-.18)*3)*adjusted
    c.text(160,205,'周期CNN：校准谱形',size=34,color=WHITE)
    c.line([(160,530),(930,530)],color=MUTED,width=2)
    graph(c,base,160,290,770,240,color='#45556A',maxv=1.4)
    graph(c,corrected,160,290,770,240,color=BLUE,maxv=1.4)
    win=180+((t*2)%1)*650
    c.rect(win,275,win+105,535,outline=YELLOW,width=2,radius=8)
    c.text(160,555,'窗口沿谱移动；首尾按周期连接',size=26,color=MUTED)
    c.formula(1070,310,r'$\log p_{\rm base}+s\tanh(\delta)$',size=41,color=GREEN)
    c.text(1070,405,'有界修正 → 归一化目标谱',size=30,color=WHITE)
    c.text(1070,485,'校准器不直接生成完整复信道',size=26,color=MUTED)
    u=ease((t-.4)*4)
    c.text(160,655,'LinkGate：为1024条收发链路分别选择融合强度',size=32,color=tint(WHITE,u))
    for i in range(32):
        val=.5+.45*math.sin(i*.62+t*3)
        xx=165+i*48
        c.rect(xx,730,xx+31,790,fill=tint(BLUE,.2+val*.8),radius=3)
        c.rect(xx,790-val*60,xx+31,790,fill=tint(GREEN,u),radius=3)
    c.text(165,815,'方格为链路缩略示意；门值在0到1之间',size=23,color=tint(MUTED,u))


def scene_transformer(c,t):
    c.text(150,205,'先理解一条候选的交互顺序',size=31,color=WHITE)
    nodes=[(185,420),(300,310),(410,500),(520,355)]
    for i in range(3):
        c.arrow(*nodes[i],*nodes[i+1],color=COLORS[i],width=3)
    for i,(x,y) in enumerate(nodes):
        hi=(int(t*12)%4)==i
        c.circle(x,y,22,fill=COLORS[i] if hi else '#22354A',outline=COLORS[i],width=2)
        c.text(x-30,y+42,['发射','反射','反射','到达'][i],size=25,color=MUTED)
    c.arrow(580,410,690,410,color=BLUE,width=3)
    panel(c,705,300,210,220,'序列编码',BLUE)
    c.text(740,390,'2层',size=42,color=BLUE)
    c.text(730,460,'Transformer',size=22,color=MUTED)
    u=ease((t-.20)*4)
    c.arrow(950,410,1040,410,color=tint(WHITE,u),width=3)
    for i in range(128):
        x=1090+(i%16)*35;y=300+(i//16)*35
        c.circle(x,y,5,fill=tint(COLORS[i%5],u))
    focus=int(t*70)%128
    fx=1090+(focus%16)*35;fy=300+(focus//16)*35
    for i in range(0,128,7):
        xx=1090+(i%16)*35;yy=300+(i//16)*35
        c.line([(fx,fy),(xx,yy)],color=tint(PURPLE,.55*u),width=1)
    c.text(1100,215,'再比较128个候选',size=31,color=tint(WHITE,u))
    c.text(1100,610,'2层集合Transformer',size=29,color=tint(PURPLE,u))
    v=ease((t-.48)*4)
    for i,(label,col) in enumerate([('能量分配',YELLOW),('角度—时延坐标',GREEN),('端口分布',BLUE)]):
        xx=400+i*450
        c.arrow(1320,650,xx+130,710,color=tint(col,.6*v),width=2)
        c.rect(xx,725,xx+300,800,outline=tint(col,v),radius=12,width=2)
        c.text(xx+150,762,label,size=27,color=tint(col,v),anchor='mm')


def projection_examples():
    """Exact toy transformations, not trained model or measured performance."""
    n,f=16,32
    aa=np.arange(n)[:,None];ff=np.arange(f)[None,:]
    def field(paths):
        return sum(am*np.exp(2j*np.pi*(ang*aa/n-delay*ff/f)) for ang,delay,am in paths)
    base=field([(-3,3,1.),(2,10,.7j),(5,20,.35)])
    truth=field([(-3,3,.5),(1,10,1.0j),(5,21,.55)])
    power=np.sum(abs(base)**2)
    def norm(p,axis):return p/np.maximum(p.sum(axis=axis,keepdims=True),1e-20)
    bp=norm(abs(np.fft.fft(base,axis=0,norm='ortho'))**2,0)
    tp=norm(abs(np.fft.fft(truth,axis=0,norm='ortho'))**2,0)
    bd=norm(abs(np.fft.ifft(base,axis=1,norm='ortho'))**2,1)
    td=norm(abs(np.fft.ifft(truth,axis=1,norm='ortho'))**2,1)
    target_p=norm((.8*np.sqrt(bp)+.2*np.sqrt(tp))**2,0)
    target_d=norm((.8*np.sqrt(bd)+.2*np.sqrt(td))**2,1)
    states=[base];h=base.copy()
    for _ in range(4):
        z=np.fft.fft(h,axis=0,norm='ortho')
        z=np.sqrt(target_p*np.sum(abs(z)**2,axis=0,keepdims=True))*np.exp(1j*np.angle(z))
        h=np.fft.ifft(z,axis=0,norm='ortho');h*=np.sqrt(power/np.sum(abs(h)**2));states.append(h.copy())
        z=np.fft.ifft(h,axis=1,norm='ortho')
        z=np.sqrt(target_d*np.sum(abs(z)**2,axis=1,keepdims=True))*np.exp(1j*np.angle(z))
        h=np.fft.fft(z,axis=1,norm='ortho');h*=np.sqrt(power/np.sum(abs(h)**2));states.append(h.copy())
    ps=[];ds=[]
    for h in states:
        p=np.fft.fftshift(np.sum(abs(np.fft.fft(h,axis=0,norm='ortho'))**2,axis=1));ps.append(p/p.sum())
        d=np.sum(abs(np.fft.ifft(h,axis=1,norm='ortho'))**2,axis=0);ds.append(d/d.sum())
    return np.array(ps),np.array(ds),np.fft.fftshift(target_p.mean(axis=1)),target_d.mean(axis=0),[np.sum(abs(h)**2)/power for h in states]


_PROJ = projection_examples()


def scene_projection(c,t):
    pas,pdp,pt,dt,powers=_PROJ
    z=min(8,max(0,t*9-.4));step=min(7,int(z));u=ease(z-step)
    p=(1-u)*pas[step]+u*pas[step+1]
    d=(1-u)*pdp[step]+u*pdp[step+1]
    c.text(150,205,'同一个复信道，在两个域中交替修正',size=34,color=WHITE)
    c.text(190,295,'角度功率谱 PAS',size=30,color=BLUE)
    c.text(1070,295,'时延功率谱 PDP',size=30,color=GREEN)
    c.line([(180,570),(820,570)],color=MUTED,width=2)
    c.line([(1050,570),(1690,570)],color=MUTED,width=2)
    graph(c,pt,180,340,640,230,color='#56657B',maxv=.65)
    graph(c,p,180,340,640,230,color=BLUE,maxv=.65)
    graph(c,dt,1050,340,640,230,color='#56657B',maxv=.65)
    graph(c,d,1050,340,640,230,color=GREEN,maxv=.65)
    c.arrow(865,415,1000,415,color=YELLOW,width=3)
    c.arrow(1000,510,865,510,color=PURPLE,width=3)
    phase='PAS' if (int(z)+1)%2 else 'PDP'
    status='4轮完成' if z>=8 else f'第{min(4,int(z)//2+1)}轮 · {phase}'
    c.text(850,610,status,size=30,color=YELLOW)
    c.formula(180,655,r'$\sqrt{p_{\rm target}}\ \leftarrow\ 0.8\sqrt{p_{\rm base}}+0.2\sqrt{p_{\rm candidate}}$',size=34,color=WHITE)
    c.text(180,715,'融合后归一化；每步保留当前变换域相位',size=27,color=MUTED)
    c.text(1070,715,'不是原始H的逐元素相位不变',size=26,color=MUTED)
    c.text(180,785,'每个位置的总功率',size=28,color=GREEN)
    c.rect(510,785,1090,813,outline='#33435A',width=2,radius=5)
    power=(1-u)*powers[step]+u*powers[step+1]
    c.rect(513,788,513+570*power,810,fill=GREEN,radius=3)
    c.text(1140,783,'归一化示意：1.00',size=27,color=GREEN)


SCENES = {
    'align':scene_align,'interpolate':scene_interpolate,'experts':scene_experts,
    'geometry':scene_geometry,'calibrate':scene_calibrate,
    'transformer':scene_transformer,'projection':scene_projection,
}
