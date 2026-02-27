"""Root entry point for the Image Colorization Ensemble web application.

Usage:
    python run.py           # production mode (serves Vue dist/)
    python run.py --dev     # dev mode (CORS enabled for Vite on :5173)
    python run.py --port 8000
"""

import argparse
import os
import sys

# Ensure project root is on sys.path so 'api' and 'ml' packages resolve
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

ML_PATH = os.path.join(ROOT, 'ml')
if ML_PATH not in sys.path:
    sys.path.insert(0, ML_PATH)


def main() -> None:
    parser = argparse.ArgumentParser(description='Image Colorization Ensemble — web server')
    parser.add_argument('--dev',  action='store_true', help='Enable CORS for Vite dev server on :5173')
    parser.add_argument('--host', default='0.0.0.0', help='Bind host (default: 0.0.0.0)')
    parser.add_argument('--port', type=int, default=5000, help='Bind port (default: 5000)')
    parser.add_argument('--debug', action='store_true', help='Enable Flask debug mode')
    args = parser.parse_args()

    from api.app import create_app

    app = create_app(dev=args.dev)

    print(f'Starting server on http://{args.host}:{args.port}')
    if args.dev:
        print('  [dev mode] CORS enabled for http://localhost:5173')
        print('  Run `cd frontend && npm run dev` separately for hot-reload.')
    else:
        print('  [production] Serving Vue from frontend/dist/')
        dist = os.path.join(ROOT, 'frontend', 'dist', 'index.html')
        if not os.path.exists(dist):
            print('  WARNING: frontend/dist/ not found — run `cd frontend && npm run build` first.')

    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()
