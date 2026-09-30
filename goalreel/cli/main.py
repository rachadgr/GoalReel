import argparse
import json

from ..pipeline import run_analysis, render_baseline


def _cmd_verify(args):
    from ..models.manager import ModelManager
    from ..models.verify import verify_all

    report = verify_all(args.video, manager=ModelManager(), every=args.every)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    # Sortie non nulle uniquement si AUCUN modèle n'a d'inférence réelle vérifiée.
    any_verified = any(m.get("verified") for m in report["models"].values())
    return 0 if any_verified else 1


def main():
    p = argparse.ArgumentParser(prog='goalreel')
    sp = p.add_subparsers(dest='cmd', required=True)

    a = sp.add_parser('analyze')
    a.add_argument('video'); a.add_argument('--out', default='runs/source')
    a.add_argument('--models', default='models'); a.add_argument('--every', type=int, default=15)

    r = sp.add_parser('render-vertical')
    r.add_argument('video'); r.add_argument('--out', required=True)

    v = sp.add_parser('verify-models', help='Vérifie le runtime RÉEL de chaque modèle')
    v.add_argument('video', help='vidéo source pour l\'inférence réelle')
    v.add_argument('--every', type=int, default=60)

    args = p.parse_args()
    if args.cmd == 'analyze':
        print(json.dumps(run_analysis(args.video, args.out, args.models, args.every),
                         indent=2, ensure_ascii=False))
    elif args.cmd == 'render-vertical':
        print(json.dumps(render_baseline(args.video, args.out),
                         indent=2, ensure_ascii=False))
    else:
        raise SystemExit(_cmd_verify(args))


if __name__ == '__main__':
    main()
