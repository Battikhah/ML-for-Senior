"""Fetch recovery artifacts without spending Drive requests on disposable job logs."""
import json
from pathlib import Path
import sys


def list_recovery_entries(source, folder_id):
    """Prune disposable folders before recursion and cache successful folder listings."""
    import importlib
    import time
    from types import SimpleNamespace
    # This parser belongs to the isolated, pinned gdown 6.4.2 environment.
    drive = importlib.import_module('gdown.download_folder')
    session, _ = drive._get_session(proxy=None, use_cookies=False,
                                    user_agent='Mozilla/5.0', cookies_file=None)
    cache_path = source.parent / 'public_drive_listing.json'
    cache = json.loads(cache_path.read_text(encoding='utf-8')) if cache_path.exists() else {}
    def visit(identifier, relative):
        """Yield required files without entering job logs, tracking databases or exports."""
        if identifier not in cache:
            for attempt in range(4):
                try:
                    _, children = drive._parse_embedded_folder_view(
                        sess=session, folder_id=identifier, verify=True, timeout=(30, 120))
                    cache[identifier] = children
                    temporary = cache_path.with_suffix('.tmp')
                    temporary.write_text(json.dumps(cache), encoding='utf-8')
                    temporary.replace(cache_path)
                    break
                except Exception as error:
                    if attempt == 3:
                        raise RuntimeError(f'Drive folder listing blocked: {relative or Path(".")}. '
                                           f'Folder ID: {identifier}. Cached listings/downloads are retained.') from error
                    print(f'Retrying Drive listing: {relative or Path(".")} ({attempt+1}/3)', flush=True)
                    time.sleep(2 ** attempt)
        for child_id, name, kind in cache[identifier]:
            if name in {'.', '..'} or '/' in name or "\\" in name:
                raise ValueError(f'Unsafe Drive entry name: {name!r}')
            child = relative / name
            if kind == 'application/vnd.google-apps.folder':
                if child.parts[0] in {'runs', 'cache', 'campaigns'}:
                    yield from visit(child_id, child)
            elif (child.parts[0] in {'runs', 'cache'}
                  or child.as_posix() == 'notebook_settings.json'
                  or (child.parts[0] == 'campaigns' and name.startswith('finalists')
                      and child.suffix == '.json')):
                yield SimpleNamespace(id=child_id, local_path=str(source / child))
    try:
        yield from visit(folder_id, Path())
    finally:
        session.close()


def direct_download(identifier, target):
    """Use explicit public endpoints, rejecting HTML and incomplete transfers before commit."""
    from urllib.request import urlopen
    from urllib.parse import urlencode
    urls = [
        'https://drive.usercontent.google.com/download?' + urlencode(
            dict(id=identifier, export='download', confirm='t')),
        'https://drive.google.com/uc?' + urlencode(dict(export='download', id=identifier)),
    ]
    temporary = target.with_name(target.name + '.direct.part')
    for url in urls:
        try:
            with urlopen(url, timeout=120) as response:
                content_type = response.headers.get('Content-Type', '').lower()
                disposition = response.headers.get('Content-Disposition', '').lower()
                if response.status != 200 or 'attachment' not in disposition or 'text/html' in content_type:
                    raise RuntimeError('Drive returned a permission/confirmation page instead of a file')
                expected = response.headers.get('Content-Length')
                size = 0
                with temporary.open('wb') as stream:
                    while chunk := response.read(1024 * 1024):
                        stream.write(chunk)
                        size += len(chunk)
                        if size // (64 * 1024**2) != (size - len(chunk)) // (64 * 1024**2):
                            print(f'Downloaded {size / 1024**3:.2f} GB', flush=True)
                if expected is not None and size != int(expected):
                    raise RuntimeError('Incomplete public Drive download')
            temporary.replace(target)
            return str(target)
        except Exception as error:
            last_error = error
    raise RuntimeError('Both explicit public download endpoints failed') from last_error


def download_snapshot(source, folder_id, receipt):
    """Preserve partial downloads and stop on any missing non-disposable artifact."""
    import gdown
    source = Path(source)
    source.mkdir(parents=True, exist_ok=True)
    entries = list_recovery_entries(source, folder_id)
    files = []
    skipped = 0
    for entry in entries:
        target = Path(entry.local_path)
        relative = target.relative_to(source)
        # Recovery rebuilds reports/journals; preserve all scientific artifacts and final locks.
        required = (relative.parts[0] in {'runs', 'cache'}
                    or relative.as_posix() == 'notebook_settings.json'
                    or (relative.parts[0] == 'campaigns'
                        and relative.name.startswith('finalists') and relative.suffix == '.json'))
        if not required:
            skipped += 1
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_file():
            print(f'Downloading recovery artifact: {relative}', flush=True)
            try:
                result = gdown.download(id=entry.id, output=str(target), use_cookies=False,
                                        resume=True, quiet=False, timeout=(30, 120), retries=3)
            except Exception:
                print(f'Trying explicit public download endpoint: {relative}', flush=True)
                try:
                    result = direct_download(entry.id, target)
                except Exception as error:
                    raise RuntimeError(f'Required recovery artifact blocked: {relative}. '
                                       f'Drive file ID: {entry.id}. Existing downloads are retained.') from error
            if not result or not target.is_file():
                raise RuntimeError(f'Required recovery artifact could not be downloaded: {relative}')
        files.append(str(relative))
    if not files or not (source / 'notebook_settings.json').is_file():
        raise RuntimeError('Expected TreeVibesExperiments recovery artifacts; check folder sharing.')
    Path(receipt).write_text(json.dumps(dict(folder_id=folder_id, files=files,
                                            skipped_nonessential_files=skipped)), encoding='utf-8')
    print(f'Recovery download ready: {len(files)} files; skipped {skipped} nonessential files.')


if __name__ == '__main__':
    download_snapshot(*sys.argv[1:])
