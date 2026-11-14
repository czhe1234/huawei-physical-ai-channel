"""Generate stock Mandarin male narration; no voice cloning or model training."""
import asyncio
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from project_paths import PATHS
sys.path.insert(0,str(PATHS.artifacts/'video_tools_20261022'))
import edge_tts

HERE=Path(__file__).resolve().parent
OUT=PATHS.analysis/'wireless_video_ppt_20261022'
VOICE='zh-CN-YunxiNeural'

async def main():
    story=json.loads((HERE/'narration.json').read_text())
    p=OUT/'voice';p.mkdir(parents=True,exist_ok=True)
    sem=asyncio.Semaphore(3);done=0;start=time.monotonic()
    async def one(i,j,text):
        nonlocal done
        dest=p/f'{i:02d}_{j}.mp3'
        if not dest.exists():
            async with sem:
                for attempt in range(3):
                    try:
                        spoken=text.replace('V16','第十六版').replace('LinkGate','链路门控网络').replace('Base与Scatter','基础传播与散射').replace('Transformer','注意力网络')
                        tmp=dest.with_suffix('.partial.mp3')
                        await asyncio.wait_for(edge_tts.Communicate(spoken,VOICE,rate='-5%',pitch='-2Hz').save(str(tmp)),timeout=45)
                        assert tmp.stat().st_size>2000
                        tmp.replace(dest);break
                    except Exception:
                        if attempt==2:raise
                        await asyncio.sleep(2*(attempt+1))
        done+=1
        print(json.dumps({'voice_segments':done,'total':sum(len(s['sentences']) for s in story),'chapter':i,'sentence':j,'wall_seconds':round(time.monotonic()-start,1)},ensure_ascii=False),flush=True)
    await asyncio.gather(*(one(i,j,text) for i,sc in enumerate(story,1) for j,text in enumerate(sc['sentences'])))
    (OUT/'voice_config.json').write_text(json.dumps({'voice':VOICE,'rate':'-5%','pitch':'-2Hz','type':'stock_neural_tts','segments':sum(len(s['sentences']) for s in story),'wall_seconds':time.monotonic()-start},ensure_ascii=False,indent=2))

if __name__=='__main__':asyncio.run(main())
