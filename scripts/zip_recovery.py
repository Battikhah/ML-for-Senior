"""Recover a complete experiment snapshot from one public Drive ZIP without login."""
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import stat
import zipfile


def extract_snapshot(archive, destination, file_id):
    """Validate every ZIP path before extracting, then commit a complete snapshot atomically."""
    destination = Path(destination)
    receipt = destination / '.zip_recovery.json'
    if receipt.exists():
        saved = json.loads(receipt.read_text(encoding='utf-8'))
        if saved['file_id'] != file_id:
            raise ValueError('Snapshot belongs to another ZIP; choose a different WORK folder.')
        root = destination / saved['root']
        if (root / 'notebook_settings.json').is_file():
            print('Reusing the extracted experiment ZIP.', flush=True)
            return root
    if destination.exists():
        raise ValueError(f'Unverified extraction exists: {destination}; preserve it and choose another WORK folder.')
    staging = destination.with_name(destination.name + '.extracting')
    # Only our uncommitted extraction is replaced; the ZIP and completed results stay intact.
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        with zipfile.ZipFile(archive) as bundle:
            members = bundle.infolist()
            seen = set()
            for member in members:
                name = member.filename
                path = PurePosixPath(name)
                windows = PureWindowsPath(name)
                key = name.rstrip('/').casefold()
                if (path.is_absolute() or windows.drive or '\\' in name or '..' in path.parts
                        or ':' in name or stat.S_ISLNK(member.external_attr >> 16)
                        or any(part.endswith((' ', '.')) or PureWindowsPath(part).is_reserved()
                               for part in path.parts)
                        or key in seen):
                    raise ValueError(f'Unsafe or duplicate ZIP entry: {name}')
                seen.add(key)
            if sum(m.file_size for m in members) > shutil.disk_usage(staging).free:
                raise OSError('Not enough free disk space to extract the experiment ZIP.')
            bundle.extractall(staging)  # zipfile verifies CRC while reading each member.
        roots = [p.parent for p in staging.rglob('notebook_settings.json')
                 if '__MACOSX' not in p.parts and (p.parent / 'runs').is_dir()
                 and (p.parent / 'cache').is_dir()]
        if len(roots) != 1:
            raise ValueError('ZIP must contain exactly one experiments folder with notebook_settings.json, runs and cache.')
        relative = roots[0].relative_to(staging)
        (staging / '.zip_recovery.json').write_text(
            json.dumps(dict(file_id=file_id, root=relative.as_posix())), encoding='utf-8')
        staging.replace(destination)
        return destination / relative
    except Exception:
        shutil.rmtree(staging)
        raise


def download_zip(work, file_id):
    """Reuse the local ZIP and extraction; downloading requires no Google authentication."""
    from scripts.public_drive_download import direct_download
    work = Path(work)
    destination = work / 'zip_snapshot'
    if (destination / '.zip_recovery.json').exists():
        return extract_snapshot(None, destination, file_id)
    downloads = work / 'downloads'
    downloads.mkdir(parents=True, exist_ok=True)
    archive = downloads / f'{file_id}.zip'
    if not archive.exists():
        print('Downloading the experiment ZIP from Google Drive. Keep this cell running.', flush=True)
        direct_download(file_id, archive)
    print(f'Extracting experiment ZIP ({archive.stat().st_size / 1024**3:.2f} GB).', flush=True)
    return extract_snapshot(archive, destination, file_id)
