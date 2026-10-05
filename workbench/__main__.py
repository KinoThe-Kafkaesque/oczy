"""One entry point for Oczy evidence, live comparisons and packaging."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    web = sub.add_parser("serve", help="Open the local evidence workbench")
    web.add_argument("--port", type=int, default=8765)
    web.add_argument("--runtime", type=Path)
    web.add_argument("--model", type=Path)
    sub.add_parser("verify", help="Check packaged source, evidence and state hashes")
    export = sub.add_parser("export", help="Refresh workbench evidence from audited results")
    export.add_argument("--study-run", type=Path)
    export.add_argument("--regression-run", type=Path)
    trial = sub.add_parser("trial", help="Run a live exploratory comparison offline")
    trial.add_argument("client", choices=("amber", "cobalt", "silver", "quartz"))
    trial.add_argument("word")
    trial.add_argument("--seed", type=int, choices=(0, 1, 2), default=0)
    trial.add_argument("--runtime", type=Path)
    trial.add_argument("--model", type=Path)
    pack = sub.add_parser("package", help="Build a portable source/evidence archive; weights excluded")
    pack.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "serve":
        from workbench.server import serve
        serve(args.port, args.runtime, args.model)
    elif args.command == "verify":
        from workbench.catalog import verify_data
        from workbench.pack import verify_package
        print(json.dumps({**verify_data(), **verify_package()}, indent=2))
    elif args.command == "export":
        from workbench.catalog import export
        print(export(args.study_run, args.regression_run)["study_state"])
    elif args.command == "package":
        from workbench.pack import build
        print(build(args.output))
    else:
        from workbench.live import LiveEngine
        engine = LiveEngine(args.runtime, args.model)
        try:
            print(json.dumps(engine.query({"client": args.client, "input": args.word, "seed": args.seed}), indent=2))
        finally:
            engine.close()


if __name__ == "__main__":
    main()
