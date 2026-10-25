#!/usr/bin/env python3
"""Export only qualified documents from a dedicated runner's evidence store."""
import json
import os
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.deploy_tested_release import deployment_plan
from scripts.promotion_gate import require, sha
from scripts.release_manifest import safe_path


def main():
    try:
        revision = os.environ['SN68_TESTED_SHA']
        require(sha(revision, 40), 'invalid_revision')
        bundle = Path(os.environ['SN68_QUALIFICATION_ROOT']) / revision
        output = Path(os.environ['SN68_EVIDENCE_OUTPUT'])
        require(not output.exists(), 'evidence_output_exists')
        _, variables = deployment_plan(Path(os.environ['SN68_RELEASE_DIR']) / 'build_stamp.json',
            revision, os.environ['SN68_DEPLOY_HOST'], True, bundle, os.environ['SN68_TARGET'])
        output.mkdir(parents=True, mode=0o700)
        for name in variables['sn68_promotion_files']:
            destination = safe_path(output, name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(safe_path(bundle, name), destination)
        # Recheck the copied bytes; a changing source cannot pass a TOCTOU gap.
        deployment_plan(Path(os.environ['SN68_RELEASE_DIR']) / 'build_stamp.json', revision,
                        os.environ['SN68_DEPLOY_HOST'], True, output, os.environ['SN68_TARGET'])
        print(json.dumps({'status': 'qualified', 'revision': revision}))
        return 0
    except (OSError, ValueError, KeyError, TypeError):
        print(json.dumps({'status': 'evidence_export_refused'}))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
