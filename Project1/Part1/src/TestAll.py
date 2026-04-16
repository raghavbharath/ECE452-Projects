import os
import sys
import subprocess
from pathlib import Path

PtFiles = list(Path('./outputs/').rglob('*.pt'))
PtFiles = [str(i) for i in PtFiles]

try:
    os.makedirs(os.path.join('outputs', 'confusions'), exist_ok=True)
except Exception as e:
    pass

count = 1
total = len(PtFiles)
for file in PtFiles:
    result = subprocess.run([sys.executable, 'Test.py', file, '--conf', os.path.join('./outputs/confusions/', os.path.basename(file).split('.')[0] + '.png')], capture_output=True, text=True)
    last_line = result.stdout.strip().split('\n')[-1]
    print(f'{count}/{total} | {file} | {last_line}')
    if result.stderr:
        print(f'ERROR: {result.stderr.strip()}')
    count += 1
