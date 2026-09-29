import argparse,json
from ..pipeline import run_analysis,render_baseline

def main():
 p=argparse.ArgumentParser(prog='goalreel'); sp=p.add_subparsers(dest='cmd',required=True)
 a=sp.add_parser('analyze'); a.add_argument('video'); a.add_argument('--out',default='runs/source'); a.add_argument('--models',default='models'); a.add_argument('--every',type=int,default=15)
 r=sp.add_parser('render-vertical'); r.add_argument('video'); r.add_argument('--out',required=True)
 args=p.parse_args()
 if args.cmd=='analyze':print(json.dumps(run_analysis(args.video,args.out,args.models,args.every),indent=2,ensure_ascii=False))
 else: print(json.dumps(render_baseline(args.video,args.out),indent=2,ensure_ascii=False))
if __name__=='__main__':main()

if __name__=='__main__': main()
