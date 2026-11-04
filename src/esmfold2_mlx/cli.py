"""Command line entry point: ``esmfold2-mlx fold ...`` and ``esmfold2-mlx serve ...``."""

import platform
import sys

USAGE = """usage: esmfold2-mlx <command> [options]

commands:
  fold    fold sequences (one-shot):   esmfold2-mlx fold --seq MKT... --name mine
  serve   keep a model loaded:         esmfold2-mlx serve start --model 6B
                                       esmfold2-mlx serve fold --model 6B --seq MKT...

Run `esmfold2-mlx <command> --help` for the options of each command."""


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(USAGE)
        return 0
    if argv[0] in ("-V", "--version"):
        from . import __version__

        print(f"esmfold2-mlx {__version__}")
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd not in ("fold", "serve"):
        print(f"esmfold2-mlx: unknown command {cmd!r}\n\n{USAGE}", file=sys.stderr)
        return 2
    if rest[:1] not in (["-h"], ["--help"]) and not (sys.platform == "darwin" and platform.machine() == "arm64"):
        print("esmfold2-mlx needs an Apple-silicon Mac (it runs on MLX and Metal).", file=sys.stderr)
        return 1
    if cmd == "fold":
        from .fold import main as run
    else:
        from .serve import main as run
    sys.argv = [f"esmfold2-mlx {cmd}"] + rest
    run()
    return 0
