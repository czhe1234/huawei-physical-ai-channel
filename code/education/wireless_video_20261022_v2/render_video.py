"""Chinese mathematical explainer. All plots are explicitly pedagogical examples."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
import io
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback
import wave

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['mathtext.fontset']='cm'
from matplotlib import mathtext
from matplotlib.font_manager import FontProperties

# Maintained source resolves its workspace through the shared project resolver.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from project_paths import PATHS
from scenes_foundation import SCENES as FOUNDATION
from scenes_algorithm import SCENES as ALGORITHM

HERE = Path(__file__).resolve().parent
OUT = PATHS.analysis / 'wireless_video_20261022_v2'
W, H, FPS, SR = 1920, 1080, 30, 48000
BG = '#0B101B'
WHITE, MUTED = '#F4F5F7', '#9AA9BD'
BLUE, YELLOW, GREEN, RED, PURPLE = '#58C4DD', '#FFFF00', '#83C167', '#FC6255', '#9A72AC'
FONT_PATH = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
BOLD_PATH = Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc')


@lru_cache(maxsize=100)
def font(size, bold=False):
    return ImageFont.truetype(str(BOLD_PATH if bold else FONT_PATH), int(size), index=2)


@lru_cache(maxsize=2500)
def text_stamp(text, size, color, bold=False):
    f = font(size, bold)
    box = f.getbbox(text)
    im = Image.new('RGBA', (max(1,box[2]-box[0]+6), max(1,box[3]-box[1]+6)))
    ImageDraw.Draw(im).text((3-box[0],3-box[1]), text, font=f, fill=color)
    return im


@lru_cache(maxsize=180)
def math_alpha(tex, size):
    buf=io.BytesIO()
    mathtext.math_to_image('$'+tex.strip('$')+'$', buf, prop=FontProperties(size=size*.6),
                          dpi=120, format='png', color='black')
    im=Image.open(buf).convert('RGB')
    return Image.fromarray(255-np.asarray(im).min(axis=2))


@lru_cache(maxsize=500)
def math_stamp(tex, size, color):
    a=math_alpha(tex,size)
    im=Image.new('RGBA',a.size,color)
    im.putalpha(a)
    return im


class Canvas:
    BLUE, YELLOW, GREEN, RED, PURPLE = BLUE, YELLOW, GREEN, RED, PURPLE
    WHITE, MUTED, BG = WHITE, MUTED, BG
    def __init__(self):
        self.im = Image.new('RGB',(W,H),BG)
        self.draw = ImageDraw.Draw(self.im)
    @staticmethod
    def smooth(v):
        v=max(0.,min(1.,float(v)));return v*v*(3-2*v)
    def _place(self,im,x,y,anchor):
        if anchor[0]=='m':x-=im.width/2
        elif anchor[0]=='r':x-=im.width
        if anchor[1]=='m':y-=im.height/2
        elif anchor[1] in 'bs':y-=im.height
        self.im.paste(im,(int(x),int(y)),im)
    def text(self,x,y,s,size=32,color=WHITE,anchor='lt',bold=False):
        self._place(text_stamp(str(s),int(size),color,bold),x,y,anchor)
    def formula(self,x,y,s,size=42,color=WHITE,anchor='lt'):
        self._place(math_stamp(s,int(size),color),x,y,anchor)
    def line(self,points,color=WHITE,width=3,progress=1):
        pts=[(float(x),float(y)) for x,y in points]
        if len(pts)<2 or progress<=0:return
        if progress<1:
            ds=np.array([math.dist(a,b) for a,b in zip(pts[:-1],pts[1:])]);target=ds.sum()*progress
            ps=[pts[0]]
            for i,seg in enumerate(ds):
                if target>=seg:ps.append(pts[i+1]);target-=seg
                else:
                    if seg>0:ps.append(tuple(np.array(pts[i])+(np.array(pts[i+1])-pts[i])*target/seg))
                    break
            pts=ps
        if len(pts)>1:self.draw.line(pts,fill=color,width=max(1,int(width)),joint='curve')
    def arrow(self,x1,y1,x2,y2,color=WHITE,width=4):
        if math.hypot(x2-x1,y2-y1)<.2:return
        self.line([(x1,y1),(x2,y2)],color,width)
        a=math.atan2(y2-y1,x2-x1);r=13+width
        self.polygon([(x2,y2),(x2-r*math.cos(a-.38),y2-r*math.sin(a-.38)),
                     (x2-r*math.cos(a+.38),y2-r*math.sin(a+.38))],fill=color)
    def circle(self,x,y,r,fill=None,outline=None,width=2):
        self.draw.ellipse((x-r,y-r,x+r,y+r),fill=fill,outline=outline,width=max(1,int(width)))
    def rect(self,x0,y0,x1,y1,fill=None,outline=None,width=2,radius=0):
        if x1<x0:x0,x1=x1,x0
        if y1<y0:y0,y1=y1,y0
        box=(int(x0),int(y0),int(x1),int(y1))
        if radius:self.draw.rounded_rectangle(box,radius,fill=fill,outline=outline,width=max(1,int(width)))
        else:self.draw.rectangle(box,fill=fill,outline=outline,width=max(1,int(width)))
    def polygon(self,points,fill=None,outline=None,width=2):
        self.draw.polygon([(int(x),int(y)) for x,y in points],fill=fill,outline=outline,width=max(1,int(width)))


def scene_metrics(c,t):
    if t<.34:
        p=c.smooth(t/.27)
        for j,(name,eng,col) in enumerate([('角度分布','PAS',BLUE),('时延分布','PDP',GREEN),('完整复数误差','NMSE',YELLOW)]):
            x=130+j*595;c.rect(x,260,x+530,650,outline='#29384E',width=2,radius=16)
            c.text(x+265,300,name,36,col,'mt');c.text(x+265,355,eng,25,MUTED,'mt')
            xx=np.linspace(0,1,100)
            if j<2:
                yy=np.exp(-((xx-.30)/.12)**2)+.6*np.exp(-((xx-.74)/.09)**2)
                yp=np.exp(-((xx-(.42-.12*p))/.15)**2)+(.35+.25*p)*np.exp(-((xx-.74)/.11)**2)
                c.line(list(zip(x+55+xx*420,590-yy*145)),col,4)
                c.line(list(zip(x+55+xx*420,590-yp*145)),WHITE,2)
            else:
                o=(x+160,540);q=(x+160+155*math.cos(.9-1.4*p),540-155*math.sin(.9-1.4*p))
                c.arrow(*o,x+300,450,color=col,width=5);c.arrow(*o,*q,color=WHITE,width=4)
                c.line([(x+300,450),q],RED,3)
        c.formula(960,745,r'S=0.4\,PAS+0.4\,PDP+\frac{0.2}{1+NMSE}',52,WHITE,'mm')
        c.text(960,825,'谱形相似，不等于复相位准确',30,MUTED,'mm')
    elif t<.68:
        rng=np.random.default_rng(42);pts=rng.uniform([160,300],[1120,775],(160,2))
        c.rect(130,260,1160,800,outline='#29384E',width=2)
        c.rect(530,390,800,670,outline=YELLOW,width=3)
        for x,y in pts:
            inside=530<x<800 and 390<y<670
            c.circle(x,y,5,fill=YELLOW if inside else BLUE)
        c.text(1330,340,'空间留出',52,YELLOW)
        c.text(1260,435,'整片区域暂时隐藏',31,WHITE)
        c.text(1260,505,'只用区域外的训练信道',31,MUTED)
        c.text(1260,575,'预测区域内的未知位置',31,MUTED)
        c.text(645,710,'留出区域',27,YELLOW,'mm')
        c.text(960,850,'训练回代、本地留出、线上测试：三种结果分别报告',30,WHITE,'mm')
    else:
        names=['已知信道','基础预测','物理候选','网络校准','谱约束合成'];cols=[BLUE,BLUE,PURPLE,GREEN,YELLOW]
        for j,(name,col) in enumerate(zip(names,cols)):
            x=110+j*350;yy=390+35*math.sin(j*.7)
            c.rect(x,yy,x+290,yy+150,outline=col,width=3,radius=14)
            c.text(x+145,yy+75,name,35,col,'mm')
            if j<4:c.arrow(x+300,yy+75,x+340,yy+75,color=MUTED,width=3)
        p=(t-.68)/.32
        for j in range(5):
            if p>j/6:c.circle(255+j*350,610,7,fill=cols[j])
        c.text(960,710,'先理解复数叠加，再理解结构约束',48,WHITE,'mm')
        c.text(960,795,'比赛V16代码解读 · 所有概念图为教学示意',28,MUTED,'mm')


SCENES={**FOUNDATION,**ALGORITHM,'metrics':scene_metrics}


def load_story():return json.loads((HERE/'narration.json').read_text())


def audio_pcm(path):
    r=subprocess.run(['ffmpeg','-v','error','-i',str(path),'-ac','1','-ar',str(SR),
                      '-f','s16le','-'],check=True,capture_output=True)
    return r.stdout


def prepare():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'segments').mkdir(exist_ok=True)
    timeline=[];offset=0.;srt=[];cue=0
    for i,sc in enumerate(load_story(),1):
        pcm=bytearray(bytes(int(.8*SR)*2));events=[]
        for j,sentence in enumerate(sc['sentences']):
            start=len(pcm)/(SR*2);raw=audio_pcm(OUT/'voice'/f'{i:02d}_{j}.mp3');pcm.extend(raw)
            end=len(pcm)/(SR*2);events.append({'start':start,'end':end,'text':sentence})
            cue+=1;srt.append((cue,start+offset,end+offset,sentence))
            pcm.extend(bytes(int(.40*SR)*2))
        pcm.extend(bytes(int(.8*SR)*2))
        frames=math.ceil(len(pcm)/(SR*2)*FPS);target=round(frames/FPS*SR)*2
        pcm.extend(bytes(target-len(pcm)))
        with wave.open(str(OUT/'segments'/f'{i:02d}.wav'),'wb') as f:
            f.setparams((1,2,SR,0,'NONE','not compressed'));f.writeframes(pcm)
        item={**sc,'number':i,'duration':frames/FPS,'frames':frames,'events':events,'offset':offset}
        timeline.append(item);offset+=item['duration']
    (OUT/'timeline.json').write_text(json.dumps(timeline,ensure_ascii=False,indent=2))
    def stamp(t):
        ms=round(t*1000);return f'{ms//3600000:02d}:{ms//60000%60:02d}:{ms//1000%60:02d},{ms%1000:03d}'
    (OUT/'中文字幕.srt').write_text('\n\n'.join(f'{n}\n{stamp(a)} --> {stamp(b)}\n{s}' for n,a,b,s in srt)+'\n')
    print(json.dumps({'stage':'audio_ready','chapters':len(timeline),'duration_seconds':offset},ensure_ascii=False),flush=True)


def wrap_text(s,size=37,width=1730):
    f=font(size);lines=['']
    for ch in s:
        if f.getlength(lines[-1]+ch)>width:lines.append(ch)
        else:lines[-1]+=ch
    return lines


def draw_frame(sc,local_t,preview=False):
    c=Canvas();ev=sc['events'];duration=sc['duration']
    stage=0.
    for j,e in enumerate(ev):
        nextstart=ev[j+1]['start'] if j<len(ev)-1 else duration-.6
        if local_t>=e['start']:stage=(j+min(1,(local_t-e['start'])/max(.001,nextstart-e['start'])))/len(ev)
    SCENES[sc['id']](c,min(.9999,max(0,stage)))
    c.text(90,27,f'无线信道 · 从直觉到算法     {sc["number"]:02d} / 15',23,MUTED)
    c.text(90,76,sc['title'],size=51,color=WHITE,bold=True)
    c.text(1830,42,'概念动画 · 非实验曲线',22,MUTED,'rt')
    c.line([(90,151),(1830,151)],'#243248',2)
    c.rect(0,939,W,H,fill='#080D16')
    c.line([(90,923),(1830,923)],'#243248',2)
    c.line([(90,923),(90+1740*local_t/duration,923)],BLUE,3)
    current=''
    for e in ev:
        if e['start']-.05<=local_t<=e['end']+.25:current=e['text'];break
    if current:
        lines=wrap_text(current)
        assert len(lines)<=2,(sc['id'],current,lines)
        for j,line in enumerate(lines):c.text(W/2,984+(j-(len(lines)-1)/2)*49,line,37,WHITE,'mm')
    if not preview:
        fade=min(1.,local_t/.30,(duration-local_t)/.4)
        if fade<1:c.im=Image.blend(Image.new('RGB',(W,H),BG),c.im,max(0,fade))
    return c.im


def render_one(sc):
    n=sc['number'];out=OUT/'segments'/f'{n:02d}.mp4';meta=OUT/'segments'/f'{n:02d}.json'
    if out.exists() and meta.exists():return json.loads(meta.read_text())
    temp=out.with_suffix('.partial.mp4');log=OUT/'segments'/f'{n:02d}.log';tick=time.monotonic()
    cmd=['ffmpeg','-y','-hide_banner','-loglevel','warning','-f','rawvideo','-pix_fmt','rgb24',
         '-s',f'{W}x{H}','-r',str(FPS),'-i','-','-i',str(OUT/'segments'/f'{n:02d}.wav'),
         '-c:v','libx264','-preset','fast','-crf','18','-threads','2','-pix_fmt','yuv420p',
         '-c:a','aac','-b:a','160k','-movflags','+faststart','-shortest',str(temp)]
    try:
        with log.open('wb') as l:
            proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stderr=l,stdout=l)
            for frame in range(sc['frames']):proc.stdin.write(draw_frame(sc,frame/FPS).tobytes())
            proc.stdin.close();status=proc.wait()
        if status:raise RuntimeError(f'ffmpeg exit={status}: {log.read_text()[-1500:]}')
        temp.replace(out)
        result={'number':n,'id':sc['id'],'duration':sc['duration'],'frames':sc['frames'],'wall_seconds':time.monotonic()-tick}
        meta.write_text(json.dumps(result,ensure_ascii=False,indent=2));return result
    except Exception:
        (OUT/'segments'/f'{n:02d}.error.txt').write_text(traceback.format_exc());raise


def preview():
    timeline=json.loads((OUT/'timeline.json').read_text());p=OUT/'preview';p.mkdir(exist_ok=True)
    thumbs=[]
    for sc in timeline:
        for part in [0.19,.52,.85]:
            im=draw_frame(sc,sc['duration']*part,True)
            im.save(p/f'{sc["number"]:02d}_{int(part*100)}.png')
            # 3 contact sheets, one for each stage, preserve legible labels.
            if part==.52:thumbs.append(im.resize((640,360),Image.Resampling.LANCZOS))
    sheet=Image.new('RGB',(1920,1800),BG)
    for i,im in enumerate(thumbs):sheet.paste(im,((i%3)*640,(i//3)*360))
    sheet.save(OUT/'分镜总览.png')
    print('preview ready',flush=True)


def render(workers):
    timeline=json.loads((OUT/'timeline.json').read_text());results=[];tick=time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs={pool.submit(render_one,sc):sc for sc in timeline}
        for f in as_completed(jobs):
            r=f.result();results.append(r);print(json.dumps({'completed':len(results),'total':len(timeline),**r},ensure_ascii=False),flush=True)
    # Stage completion is atomic and resumable; no original asset is touched.
    listing=OUT/'segments'/'concat.txt'
    listing.write_text(''.join(f"file '{sc['number']:02d}.mp4'\nduration {sc['duration']:.9f}\n" for sc in timeline))
    dest=OUT/'无线信道比赛_中文讲解_v2.mp4';temp=dest.with_suffix('.partial.mp4')
    chapters=OUT/'chapters.ffmetadata'
    chapters.write_text(';FFMETADATA1\ntitle=无线信道比赛：从基础到V16\n'+''.join(
        f'[CHAPTER]\nTIMEBASE=1/1000\nSTART={round(sc["offset"]*1000)}\nEND={round((sc["offset"]+sc["duration"])*1000)}\ntitle={sc["number"]:02d} {sc["title"]}\n' for sc in timeline))
    joined=OUT/'narration.wav'
    with wave.open(str(joined),'wb') as output:
        output.setparams((1,2,SR,0,'NONE','not compressed'))
        for sc in timeline:
            with wave.open(str(OUT/'segments'/f'{sc["number"]:02d}.wav')) as source:
                output.writeframes(source.readframes(source.getnframes()))
    subprocess.run(['ffmpeg','-y','-hide_banner','-loglevel','warning','-f','concat','-safe','0',
                    '-i',str(listing),'-i',str(joined),'-f','ffmetadata','-i',str(chapters),
                    '-map','0:v:0','-map','1:a:0','-map_metadata','2','-map_chapters','2',
                    '-c:v','copy','-af','loudnorm=I=-18:TP=-1.5:LRA=8','-ar','48000','-c:a','aac','-b:a','160k','-shortest',
                    '-metadata:s:a:0','language=zho','-movflags','+faststart',str(temp)],check=True)
    temp.replace(dest)
    (OUT/'render_summary.json').write_text(json.dumps({'chapters':15,'width':W,'height':H,'fps':FPS,
        'duration':sum(s['duration'] for s in timeline),'render_wall_seconds':time.monotonic()-tick,
        'workers':workers,'gpu_used':False,'segments':sorted(results,key=lambda r:r['number'])},ensure_ascii=False,indent=2))
    print(json.dumps({'stage':'finished','file':str(dest),'wall_seconds':time.monotonic()-tick},ensure_ascii=False),flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('command',choices=['prepare','preview','render']);a.add_argument('--workers',type=int,default=4);args=a.parse_args()
    if args.command=='prepare':prepare()
    elif args.command=='preview':preview()
    else:render(args.workers)
