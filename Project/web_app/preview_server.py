"""Development-preview adapter. Normal Windows launch remains app.py."""
import argparse
from app import app

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4173)
    parser.add_argument("--strictPort", action="store_true")
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=False, use_reloader=False)
