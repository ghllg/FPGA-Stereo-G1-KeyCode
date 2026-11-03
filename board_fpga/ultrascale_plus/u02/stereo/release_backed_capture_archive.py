"""Release one verified, remotely backed-up board /tmp archive; retain raw frames."""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import time

U02 = Path(__file__).resolve().parents[1]
ROOT = U02.parents[1]
sys.path.insert(0, str(ROOT / 'bringup/axu5evb_e_boot'))
from uart_console import Console


def main():
    backup = U02 / 'milestones/stereo_720p_phone_sync_recorded'
    manifest = json.loads((backup / 'manifest.json').read_text())
    assert manifest['complete'] and manifest['all_jpegs_fully_decoded']
    head = subprocess.check_output([r'D:\Tools\Git\cmd\git.exe', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    remote_head = subprocess.check_output([r'D:\Tools\Git\cmd\git.exe', '-C', str(ROOT), 'ls-remote', 'origin', 'refs/heads/main'], text=True).split()[0]
    assert head == remote_head, 'Backup must be on remote main before removing duplicate'
    combined = hashlib.sha256()
    size = 0
    for entry in manifest['chunks']:
        path = backup / entry['name']
        assert path.parent == backup and path.is_file()
        relative = path.relative_to(ROOT).as_posix()
        tracked = subprocess.check_output([r'D:\Tools\Git\cmd\git.exe', '-C', str(ROOT), 'show', 'HEAD:' + relative])
        data = path.read_bytes()
        assert data == tracked and len(data) == entry['bytes']
        assert hashlib.sha256(data).hexdigest() == entry['sha256']
        combined.update(data)
        size += len(data)
    assert size == manifest['archive_bytes'] and combined.hexdigest() == manifest['archive_sha256']
    # Exact private capture directory from the accepted recovery evidence.
    directory = '/tmp/u02_ebc4a1bcc31a45d49f733300b5591999'
    target = directory + '/capture_outputs.tar.gz'
    output = U02 / ('reports/release_backed_capture_archive_' + str(time.time_ns()))
    output.mkdir()
    state = dict(complete=False, remote_backup_commit=head, target=target,
                 archive_sha256=combined.hexdigest(), bytes=size, raw_frames_deleted=False,
                 startup_files_modified=False, board_programmed=False)
    console = None
    try:
        with (output / 'uart.log').open('wb') as log:
            console = Console('COM9', log)
            console.sync()
            def checked(command):
                code, text = console.run(command, timeout=50)
                assert code == 0, (command, text)
                return text
            state['space_before'] = checked('df -k /tmp')
            checked('test ! -L ' + directory + ' && test ! -L ' + target + ' && test -f ' + target)
            assert re.search(r'(?m)^' + re.escape(target) + r'\r*$', checked('readlink -f ' + target))
            assert re.search(r'(?m)^0\r*$', checked('cat ' + directory + '/capture_exit'))
            assert 'U02_DUAL_CAPTURE_FUNCTIONAL_PASS frames_per_camera=450' in checked('cat ' + directory + '/capture.log')
            assert re.search(r'(?m)^' + combined.hexdigest() + r'\s', checked('sha256sum ' + target))
            checked('rm -- ' + target)
            checked('test ! -e ' + target)
            state['space_after'] = checked('df -k /tmp')
            state['complete'] = True
    finally:
        if console:
            console.port.close()
        state['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        state['uart_sha256'] = hashlib.sha256((output / 'uart.log').read_bytes()).hexdigest()
        (output / 'state.json').write_text(json.dumps(state, indent=2) + '\n')
        print(output, flush=True)


if __name__ == '__main__':
    main()

