"""Seven animated teaching scenes. All numerical examples are synthetic.

The parent renderer provides a Canvas with the documented drawing primitives.
No experiment data, model inference, or empirical performance is used here.
"""
from math import cos, sin, pi, hypot
import numpy as np

BG = '#0B101B'
BLUE = '#58C4DD'
YELLOW = '#FFFF00'
GREEN = '#83C167'
RED = '#FC6255'
PURPLE = '#9A72AC'
WHITE = '#F4F5F7'
MUTED = '#9AA9BD'
GRID = '#263348'


def phase(c, t, start=0, end=1):
    return c.smooth((t-start)/max(end-start, 1e-9))


def mix(a, b, t):
    t = min(1, max(0, t))
    aa = tuple(int(a[i:i+2], 16) for i in (1, 3, 5))
    bb = tuple(int(b[i:i+2], 16) for i in (1, 3, 5))
    return '#'+''.join(f'{round(x*(1-t)+y*t):02x}' for x, y in zip(aa, bb))


def axes(c, x, y, w, h, xlabel='', ylabel=''):
    c.arrow(x, y, x+w, y, color=MUTED, width=2)
    c.arrow(x, y, x, y-h, color=MUTED, width=2)
    if xlabel:
        c.text(x+w-10, y+18, xlabel, size=25, color=MUTED, anchor='rt')
    if ylabel:
        c.text(x-12, y-h-26, ylabel, size=25, color=MUTED)


def pulse(c, x, y, t, color=YELLOW, base=9):
    r = base+28*((t*4) % 1)
    c.circle(x, y, r, outline=mix(color, BG, (r-base)/30), width=2)
    c.circle(x, y, base, fill=color)


def on_path(points, u):
    lens = [hypot(b[0]-a[0], b[1]-a[1]) for a, b in zip(points[:-1], points[1:])]
    total = sum(lens)
    z = (u % 1)*total
    for a, b, length in zip(points[:-1], points[1:], lens):
        if z <= length:
            r = z/max(length, 1e-9)
            return (a[0]+r*(b[0]-a[0]), a[1]+r*(b[1]-a[1]))
        z -= length
    return points[-1]


def scene_task(c, t):
    # Spatial map is a conceptual sketch, not the official map or actual point IDs.
    c.text(125, 195, '地图 + 已知位置的信道', size=36, color=BLUE)
    for x in range(150, 950, 80):
        c.line([(x, 275), (x, 765)], color=GRID, width=1)
    for y in range(285, 800, 80):
        c.line([(130, y), (970, y)], color=GRID, width=1)
    for x, y, w, h in [(210,330,125,95),(490,290,135,100),(690,560,160,90),(340,625,140,110)]:
        c.rect(x,y,x+w,y+h,fill='#152337',outline=GRID,width=2,radius=5)
    bs=(165,570)
    c.line([(bs[0],bs[1]+32),(bs[0],bs[1]-42)],color=BLUE,width=5)
    c.line([(bs[0]-20,bs[1]+32),(bs[0],bs[1]-12),(bs[0]+20,bs[1]+32)],color=BLUE,width=3)
    c.text(122,620,'基站',size=27,color=BLUE)
    points=[(382,300),(405,367),(715,305),(800,345),(880,390),(470,450),(560,460),
            (760,460),(330,530),(470,565),(580,580),(625,675),(890,700),(250,725)]
    for i,(x,y) in enumerate(points):
        p=phase(c,t,.015+i*.007,.16+i*.007)
        c.circle(x,y,6*p,fill=GREEN)
    q=(660,490)
    pulse(c,*q,t,YELLOW)
    c.text(690,475,'未知位置',size=27,color=YELLOW)
    if t>.30:
        a=phase(c,t,.30,.58)
        for p in [(560,460),(760,460),(580,580)]:
            c.line([p,q],color=mix(BG,GREEN,.65),width=2,progress=a)
    c.text(175,805,'4000 个训练位置',size=29,color=GREEN)
    c.text(560,805,'500 个测试位置',size=29,color=YELLOW)
    c.arrow(1000,505,1130,505,color=WHITE,width=4)
    c.text(1280,200,'每个位置的完整复信道',size=34,color=WHITE)
    amount=phase(c,t,.45,.88)
    for k in reversed(range(4)):
        x0=1210+28*k; y0=300-18*k
        c.rect(x0,y0,x0+365,y0+345,fill='#111D2C',outline=mix(BG,BLUE,amount),width=2)
        for row in range(6):
            for col in range(8):
                a=.25+.65*(.5+.5*sin(row*.7+col*.6+k))
                c.circle(x0+28+col*44,y0+30+row*55,4,fill=mix(BG,BLUE,amount*a))
    c.formula(1180,700,r'H\in\mathbb{C}^{256\times4\times192}',size=43,color=YELLOW)
    c.text(1180,775,'天线端口 × 用户端口 × 频点',size=27,color=MUTED)
    c.text(1180,822,'地图与张量均为缩略示意',size=23,color=MUTED)


def scene_complex(c,t):
    cx,cy=505,500;radius=200
    for k in range(-2,3):
        c.line([(cx-290,cy+k*100),(cx+290,cy+k*100)],color=GRID,width=1)
        c.line([(cx+k*100,cy-260),(cx+k*100,cy+260)],color=GRID,width=1)
    c.arrow(cx-310,cy,cx+320,cy,color=MUTED,width=2)
    c.arrow(cx,cy+285,cx,cy-285,color=MUTED,width=2)
    c.text(cx+330,cy-10,'实部',size=27,color=MUTED)
    c.text(cx+13,cy-295,'虚部',size=27,color=MUTED)
    c.circle(cx,cy,radius,outline=GRID,width=2)
    theta=.28+1.6*pi*phase(c,t,.08,.86)
    amplitude=1+.22*sin(2*pi*phase(c,t,.48,1))
    rr=radius*amplitude
    tip=(cx+rr*cos(theta),cy-rr*sin(theta))
    c.arrow(cx,cy,*tip,color=BLUE,width=7)
    c.circle(*tip,7,fill=YELLOW)
    c.line([(tip[0],cy),tip,(cx,tip[1])],color=GRID,width=2)
    arc=[(cx+60*cos(a),cy-60*sin(a)) for a in np.linspace(0,theta%(2*pi),60)]
    c.line(arc,color=YELLOW,width=3)
    c.formula(cx+77,cy-65,r'\phi',size=40,color=YELLOW)
    c.text(250,815,'长度是幅度，方向是相位',size=33,color=WHITE)
    c.formula(1060,255,r'H=Ae^{j\phi}',size=72,color=WHITE)
    c.text(1080,405,'幅度 A',size=37,color=BLUE)
    c.line([(1080,485),(1080+300*amplitude/1.25,485)],color=BLUE,width=14)
    if t>.28:
        c.text(1080,560,'功率与幅度平方成正比',size=32,color=GREEN)
        c.formula(1080,628,r'P\propto |H|^2=A^2',size=50,color=GREEN)
    if t>.60:
        c.text(1080,775,'相位不同，复数也不同',size=31,color=YELLOW)


def scene_multipath(c,t):
    tx=(180,550);rx=(930,550)
    c.rect(405,240,760,260,fill='#263348')
    c.rect(440,760,790,780,fill='#263348')
    c.text(450,193,'反射表面',size=27,color=MUTED)
    paths=[([tx,rx],BLUE),([tx,(570,260),rx],GREEN),([tx,(610,760),rx],PURPLE)]
    for i,(pts,color) in enumerate(paths):
        u=phase(c,t,.03+i*.10,.28+i*.10)
        c.line(pts,color=color,width=4,progress=u)
        if u>.98:
            pos=on_path(pts,t*3-i*.19)
            c.circle(*pos,7,fill=color)
    pulse(c,*tx,t,BLUE)
    pulse(c,*rx,t,YELLOW)
    c.text(125,590,'发射端',size=30,color=BLUE)
    c.text(872,590,'接收端',size=30,color=YELLOW)
    c.text(300,815,'同一信号，走过不同路线',size=31,color=WHITE)
    c.line([(1040,225),(1040,850)],color=GRID,width=2)
    ox,oy=1210,530
    c.arrow(1120,oy,1790,oy,color=GRID,width=2)
    c.arrow(ox,760,ox,285,color=GRID,width=2)
    vals=[1+0j,.7*np.exp(1.75j),.48*np.exp(-.9j)]
    start=(ox,oy);s=0j
    for i,(v,color) in enumerate(zip(vals,[BLUE,GREEN,PURPLE])):
        u=phase(c,t,.30+i*.11,.49+i*.11)
        end=(start[0]+170*v.real*u,start[1]-170*v.imag*u)
        c.arrow(*start,*end,color=color,width=5)
        if u>.99:
            c.formula(end[0]+10,end[1]-35,rf'H_{i+1}',size=29,color=color)
        start=end;s+=v*u
    if t>.58:
        c.arrow(ox,oy,ox+170*s.real,oy-170*s.imag,color=YELLOW,width=7)
    c.formula(1150,220,r'H=H_1+H_2+H_3',size=48,color=WHITE)
    c.text(1120,700,'复数相加：可以增强，也可以抵消',size=29,color=YELLOW)
    c.text(1120,792,'路线不同 → 时延、角度和相位不同',size=27,color=MUTED)


def scene_mimo(c,t):
    c.text(140,195,'MIMO：多端口之间的传播关系',size=34,color=BLUE)
    tx=[]
    for row in range(4):
        for col in range(8):
            pos=(160+col*43,335+row*64);tx.append(pos)
            c.circle(*pos,8,fill=BLUE)
    rx=[(790+col*65,390+row*85) for row in range(2) for col in range(2)]
    k=int(t*8)%len(tx)
    for j,p in enumerate(rx):
        c.line([tx[k],p],color=mix(BG,GREEN,.4+.4*(j==int(t*12)%4)),width=2)
        c.circle(*p,12,fill=GREEN)
    c.text(160,650,'基站阵列：缩略示意',size=28,color=BLUE)
    c.text(720,650,'用户端口',size=28,color=GREEN)
    c.text(160,720,'实际端口：2 极化 × 16 水平 × 8 垂直',size=26,color=MUTED)
    c.text(160,775,'极化与阵元位置都影响响应',size=29,color=WHITE)
    c.line([(1005,220),(1005,840)],color=GRID,width=2)
    c.text(1080,195,'OFDM：在多个频点描述信道',size=33,color=YELLOW)
    x0,y0,w,h=1110,665,630,330
    axes(c,x0,y0,w,h,'频率','响应幅度')
    for j in range(13):
        x=x0+35+j*43
        value=.34+.30*sin(j*.6+.2)+.2*cos(j*1.2)
        value=max(.12,value)
        ht=h*value*phase(c,t,.20+j*.014,.45+j*.014)
        c.line([(x,y0),(x,y0-ht)],color=YELLOW,width=4)
        c.circle(x,y0-ht,5,fill=YELLOW)
    c.text(1120,740,'192 个频点，不是 192 个时刻',size=28,color=WHITE)
    if t>.52:
        c.formula(1140,798,r'H_{m,n}(f_k)',size=40,color=BLUE)


N_PDP=64
H_TIME=np.zeros(N_PDP,dtype=complex)
H_TIME[[3,11,19]]=np.array([1,.65*np.exp(.7j),.4*np.exp(-1.1j)])
H_FREQ=np.fft.fft(H_TIME)
PDP=np.abs(np.fft.ifft(H_FREQ))**2


def scene_pdp(c,t):
    c.text(130,195,'同一个合成多径信道',size=35,color=WHITE)
    c.text(130,250,'频率响应',size=31,color=BLUE)
    x0,y0,w,h=145,690,630,330
    axes(c,x0,y0,w,h,'频点 k','幅度 |H[k]|')
    vals=np.abs(H_FREQ); vmax=vals.max()*1.1
    pts=[(x0+i*w/63,y0-v*h/vmax) for i,v in enumerate(vals)]
    c.line(pts,color=BLUE,width=4,progress=phase(c,t,0,.30))
    kk=int((t*2%1)*63)
    c.circle(*pts[kk],6,fill=YELLOW)
    c.formula(815,435,r'\mathrm{IFFT}',size=41,color=WHITE)
    c.arrow(825,530,1035,530,color=WHITE,width=4)
    c.text(1090,250,'时延功率谱 PDP',size=31,color=GREEN)
    xx,yy,ww,hh=1100,690,625,330
    axes(c,xx,yy,ww,hh,'离散时延格 d','功率')
    grow=phase(c,t,.30,.67)
    for i in [3,11,19]:
        x=xx+i*ww/32; ht=PDP[i]*hh*.9*grow
        c.line([(x,yy),(x,yy-ht)],color=GREEN,width=9)
        c.circle(x,yy-ht,6,fill=GREEN)
        c.text(x,yy+33,str(i),size=23,color=MUTED,anchor='mt')
    c.formula(1110,760,r'P[d]=|\mathrm{IFFT}(H)[d]|^2',size=36,color=GREEN)
    c.text(145,797,'频率上的起伏，来自不同到达时延的叠加',size=29,color=WHITE)
    c.text(1100,822,'教学例：64 频点，3 / 11 / 19 格时延',size=24,color=MUTED)


N_PAS=16
SPATIAL_BINS=np.array([-4,1,5])
SPATIAL_COEF=np.array([.85,1.,.5*np.exp(.65j)])
ARRAY_RESPONSE=np.sum(SPATIAL_COEF[:,None]*np.exp(2j*pi*SPATIAL_BINS[:,None]*np.arange(N_PAS)/N_PAS),axis=0)
PAS=np.abs(np.fft.fftshift(np.fft.fft(ARRAY_RESPONSE)))**2


def scene_pas(c,t):
    c.text(130,195,'阵列位置上的复数响应',size=34,color=BLUE)
    c.text(1100,195,'空间 FFT → 角度功率谱 PAS',size=34,color=GREEN)
    for n,z in enumerate(ARRAY_RESPONSE):
        x=150+n*46;y=510
        c.circle(x,y,5,fill=MUTED)
        u=phase(c,t,.02+n*.008,.22+n*.008)
        scale=40
        c.arrow(x,y,x+z.real*scale*u,y-z.imag*scale*u,color=BLUE,width=3)
        c.text(x,610,str(n),size=21,color=MUTED,anchor='mt')
    c.line([(135,510),(900,510)],color=GRID,width=1)
    c.text(160,700,'相位沿阵列变化，包含方向信息',size=30,color=WHITE)
    c.text(160,755,'教学例：16 阵元的一维阵列',size=25,color=MUTED)
    c.arrow(930,500,1060,500,color=WHITE,width=4)
    xx,yy,ww,hh=1120,665,620,340
    axes(c,xx,yy,ww,hh,'空间频率格','功率')
    growth=phase(c,t,.28,.66)
    for i,value in enumerate(PAS):
        x=xx+(i+.5)*ww/16;ht=hh*.9*value/max(PAS)*growth
        c.line([(x,yy),(x,yy-ht)],color=GREEN,width=7)
    for b in [-4,1,5]:
        i=b+8;x=xx+(i+.5)*ww/16
        c.text(x,yy+35,str(b),size=23,color=MUTED,anchor='mt')
    if t>.60:
        i=SPATIAL_BINS[int(t*6)%3]+8
        x=xx+(i+.5)*ww/16
        c.circle(x,yy-hh*.9*PAS[i]/max(PAS),12,outline=YELLOW,width=3)
    c.text(1120,770,'方向格 ≠ 直接以度表示的角度',size=28,color=YELLOW)
    c.text(1120,822,'实际评分按二维阵列做 FFT',size=25,color=MUTED)


def scene_joint(c,t):
    c.text(130,195,'角度—时延：同时保留两种结构',size=34,color=WHITE)
    x0,y0,w,h=180,690,630,350
    for i in range(8):c.line([(x0+i*w/7,y0),(x0+i*w/7,y0-h)],color=GRID,width=1)
    for i in range(6):c.line([(x0,y0-i*h/5),(x0+w,y0-i*h/5)],color=GRID,width=1)
    c.arrow(x0,y0,x0+w+30,y0,color=MUTED,width=2)
    c.arrow(x0,y0,x0,y0-h-25,color=MUTED,width=2)
    c.text(450,730,'时延格',size=28,color=MUTED)
    c.text(130,275,'方向格',size=27,color=MUTED)
    spots=[(.18,.73,1,BLUE),(.48,.42,.65,GREEN),(.78,.67,.42,PURPLE)]
    u=phase(c,t,.02,.35)
    for a,b,v,color in spots:
        x=x0+a*w;y=y0-b*h
        c.circle(x,y,8+20*v*u,fill=color)
        c.line([(x,y0+4),(x,y0+42*v*u)],color=color,width=7)
        c.line([(x0-4,y),(x0-45*v*u,y)],color=color,width=7)
    c.text(225,820,'边缘分布分别对应 PDP 与 PAS',size=28,color=WHITE)
    c.line([(970,230),(970,850)],color=GRID,width=2)
    cx,cy=1390,475
    c.circle(cx,cy,140,outline=GRID,width=2)
    c.arrow(cx-210,cy,cx+220,cy,color=GRID,width=2)
    c.arrow(cx,cy+205,cx,cy-190,color=GRID,width=2)
    theta=2*pi*phase(c,t,.34,.96)
    c.arrow(cx,cy,cx+140*cos(theta),cy-140*sin(theta),color=YELLOW,width=6)
    c.formula(1120,220,r'H^\prime=e^{j\theta}H',size=50,color=YELLOW)
    c.text(1110,715,'整体相位旋转，PAS 与 PDP 不变',size=30,color=GREEN)
    c.text(1110,785,'功率谱相似，不等于完整复信道相同',size=29,color=WHITE)


SCENES={
    'task':scene_task,'complex':scene_complex,'multipath':scene_multipath,
    'mimo':scene_mimo,'pdp':scene_pdp,'pas':scene_pas,'joint':scene_joint,
}
