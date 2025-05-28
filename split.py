import random
import json
import shutil
import pathlib
import tqdm

SRC = pathlib.Path('/path/to/hw4_realse_dataset/train').expanduser()
DST = pathlib.Path('/path/to/hw4_split').expanduser()
DST.mkdir(parents=True, exist_ok=True)

random.seed(2025)
pairs = {'rain': [], 'snow': []}
for f in (SRC/'degraded').glob('*.png'):
    pairs['rain' if f.name.startswith('rain') else 'snow'].append(f.stem)

split = {'train': [], 'val': []}
for tag, lst in pairs.items():
    random.shuffle(lst)
    n = int(0.2 * len(lst))
    split['val'] += lst[:n]
    split['train'] += lst[n:]


def cname(s):
    return s.replace('rain-', 'rain_clean-', 1) if s.startswith('rain-') \
           else s.replace('snow-', 'snow_clean-', 1)


for part in ['train', 'val']:
    for sub in ['degraded', 'clean']:
        (DST/part/sub).mkdir(parents=True, exist_ok=True)

for part, names in split.items():
    for stem in tqdm.tqdm(names, desc=part):
        shutil.copy(SRC/'degraded'/f'{stem}.png',
                    DST/part/'degraded'/f'{stem}.png')
        shutil.copy(SRC/'clean'/f'{cname(stem)}.png',
                    DST/part/'clean'/f'{cname(stem)}.png')

json.dump(split, open(DST/'split.json', 'w'), indent=2)
