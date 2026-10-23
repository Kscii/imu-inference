"""Build a wheel and source distribution, then reproduce the wheel from that sdist."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import tomllib
import zipfile


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=all'], cwd=root):
        raise ValueError('Commit reviewed source before building release artifacts')
    if args.output.exists():
        raise FileExistsError('Refusing to overwrite a release build')
    version = tomllib.loads((root/'pyproject.toml').read_text())['project']['version']
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    with tempfile.TemporaryDirectory(prefix='imu-build-') as name:
        tmp = Path(name)
        subprocess.run(['uv', 'build', '--wheel', '--sdist', '--offline', '--out-dir', str(tmp/'dist')],
                       cwd=root, check=True)
        wheel = tmp/f'dist/imu_inference-{version}-py3-none-any.whl'
        sdist = tmp/f'dist/imu_inference-{version}.tar.gz'
        with tarfile.open(sdist) as archive:
            archive.extractall(tmp/'source', filter='data')
        subprocess.run(['uv', 'build', '--wheel', '--offline', '--out-dir', str(tmp/'rebuilt')],
                       cwd=tmp/f'source/imu_inference-{version}', check=True)
        rebuilt = tmp/'rebuilt'/wheel.name
        if digest(wheel) != digest(rebuilt):
            raise ValueError('Wheel rebuilt from source distribution differs')
        with zipfile.ZipFile(wheel) as archive:
            names = archive.namelist()
            for notice in ('LICENSE', 'THIRD_PARTY_NOTICES.md', 'MODEL_LICENSES.md'):
                if not any(n.endswith('/licenses/'+notice) for n in names):
                    raise ValueError(f'Missing packaged license file: {notice}')
        report = dict(schema='imu.delivery.build.v2', passed=True, source_commit=commit,
                      library_version=version, wheel_sha256=digest(wheel), sdist_sha256=digest(sdist),
                      sdist_rebuilt_wheel_sha256=digest(rebuilt),
                      public_release_allowed=json.loads((root/'provenance/source-license-review.json')
                                                        .read_text())['public_release_allowed'])
        args.output.mkdir(parents=True)
        for path in (wheel, sdist):
            shutil.copy2(path, args.output/path.name)
        (args.output/'build-validation.json').write_text(json.dumps(report, indent=2)+'\n')
        (args.output/'SHA256SUMS').write_text(''.join(f'{digest(p)}  {p.name}\n'
            for p in sorted(args.output.iterdir()) if p.is_file()))
        print(json.dumps(report))


if __name__ == '__main__':
    main()
