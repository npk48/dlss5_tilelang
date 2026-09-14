"""Two isolated Python implementations, same GPU, alternating measurement blocks."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np
class Worker:
    def __init__(self,python,script,root,directory):
        self.directory=directory;directory.mkdir(parents=True,exist_ok=True)
        self.log=(directory/'worker.log').open('w',encoding='utf8')
        self.p=subprocess.Popen([str(python),str(script),'--root',str(root),'--cache',str(directory/'cache'),'--output',str(directory)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf8',errors='replace',bufsize=1)
        self.ready=self.receive()
    def receive(self):
        for line in self.p.stdout:
            self.log.write(line);self.log.flush()
            if line.startswith('D5_RESULT '):
                value=json.loads(line[len('D5_RESULT '):])
                if 'error' in value:raise RuntimeError(value['error'])
                return value
        raise RuntimeError(f'worker exited {self.p.poll()}; see {self.directory}')
    def command(self,**value):
        self.p.stdin.write(json.dumps(value)+'\n');self.p.stdin.flush();return self.receive()
    def close(self):
        if self.p.poll() is None:
            self.p.stdin.write('{"op":"close"}\n');self.p.stdin.flush();self.p.wait(timeout=60)
        self.log.close()

def main(a):
    root=Path(__file__).resolve().parents[2];out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    shapes=[tuple(map(int,item.split('x'))) for item in a.shapes.split(',')]
    workers=[];reports=[]
    try:
        for name,path in [('static',a.baseline),('dynamic',root)]:
            workers.append(Worker(a.python,Path(__file__).with_name('bench_tilelang_dynamic.py'),path,out/name))
        for h,w in shapes:
            prep=[]
            for name,worker in zip(['static','dynamic'],workers):
                print('PREPARE',name,h,w,flush=True);prep.append(worker.command(op='prepare',h=h,w=w))
            numerical=[]
            for v in range(2):
                x=np.fromfile(out/'static'/f'{h}x{w}_{v}.head.f32',np.float32);y=np.fromfile(out/'dynamic'/f'{h}x{w}_{v}.head.f32',np.float32)
                assert x.size==y.size==h*w*4
                error=np.abs(x-y);row={'variant':v,'bit_exact':bool(np.array_equal(x,y)),'max_abs':float(error.max()),'pass':bool(np.allclose(x,y,rtol=1e-5,atol=1e-5))};numerical.append(row)
                if not row['pass']:raise AssertionError(row)
            samples=[[],[]];raw=[[],[]]
            for i in range(a.rounds):
                for k in ([0,1] if i%2==0 else [1,0]):
                    value=workers[k].command(op='measure',warm=30 if i==0 else 2,n=a.iterations)
                    samples[k].append(float(np.median(value['gpu_span_ms'])));raw[k].append(value)
            ratio=np.array(samples[1])/np.array(samples[0]);report={'h':h,'w':w,'prepare':prep,'numerical':numerical,'gpu_round_medians':samples,'paired_median_change_percent':float(100*(np.median(ratio)-1)),'measurements':raw}
            reports.append(report);(out/'comparison.json').write_text(json.dumps({'cases':reports},indent=2));print(json.dumps({'size':[h,w],'bit_exact':all(r['bit_exact'] for r in numerical),'dynamic_new_compiles':prep[1]['new_nvrtc_compiles'],'paired_change_percent':report['paired_median_change_percent']}),flush=True)
        final=[w.command(op='report') for w in workers];(out/'comparison.json').write_text(json.dumps({'cases':reports,'final':final},indent=2))
    finally:
        for worker in workers:
            try:worker.close()
            except Exception:worker.p.kill()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--python',type=Path,default=Path('C:/work/dlss5_remake/.venv/Scripts/python.exe'));p.add_argument('--baseline',type=Path,default=Path('C:/tmp/d5-tl-static-588548a'));p.add_argument('--output',type=Path,default=Path('C:/tmp/d5-tl-eval'));p.add_argument('--shapes',default='320x384,384x512,512x640,320x384');p.add_argument('--rounds',type=int,default=12);p.add_argument('--iterations',type=int,default=15);main(p.parse_args())
