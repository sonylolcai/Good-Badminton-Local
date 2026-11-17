#!/usr/bin/env python3
"""Local launcher for the standard fixed badminton GPU app, using private config."""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import signal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def supervise(command):
    """Honor the GPU worker's documented exit-75 recovery on local machines."""
    process = None
    def stop(_signum, _frame):
        if process is not None and process.poll() is None:
            process.terminate()
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired: process.kill()
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    while True:
        process = subprocess.Popen(command)
        code = process.wait()
        if code != 75:
            return code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env-file', default=str(ROOT / '.secrets/gpu-api.env'))
    parser.add_argument('--port', type=int, default=8082)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not Path(args.env_file).is_file():
        raise SystemExit('Missing private GPU environment file')
    if not args.worker:
        raise SystemExit(supervise([sys.executable, str(Path(__file__).resolve()), '--worker',
                                    '--env-file', args.env_file, '--port', str(args.port)]))
    os.environ['GOOD_BADMINTON_ENV_FILE'] = str(Path(args.env_file).resolve())
    from runtime_config import load_runtime_environment
    load_runtime_environment()
    digest = hashlib.sha256()
    paths = []
    for directory in ['api', 'apps', 'badminton_analysis', 'webui', 'good_badminton_contracts']:
        paths.extend((ROOT / directory).rglob('*.py'))
    paths.extend([ROOT / 'tools/run_lightweight_tracknet_raw.py', ROOT / 'runtime_config.py'])
    for path in sorted(paths):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    os.environ['GOOD_GPU_BUILD_SHA'] = revision + '+dirty:' + digest.hexdigest()
    os.chdir(ROOT)
    import uvicorn
    from apps.badminton_gpu.app import app
    uvicorn.run(app, host='127.0.0.1', port=args.port, log_level='warning', access_log=False)


if __name__ == '__main__':
    main()
