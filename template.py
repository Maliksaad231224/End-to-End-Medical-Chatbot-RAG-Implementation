import os
from pathlib import Path
import logging

logging.basicConfig(level = logging.INFO, format = '[%(asctime)s]: %(message)s')

list_of_files= [
    'src/__init__.py',
    'src/helper.py',
    'src/prompt.py',
    'setup.py',
    'app.py',
    'research/trials.ipynb'
    ,'test.py'
]

for filepath in list_of_files:
    filepath=Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir!='':
        os.makedirs(filedir, exist_ok=True)
    
    if (not os.path.exists(filepath)) or (os.path.getsize(filename)):
        with open(filename, 'w') as f:
            pass
    else:
        logging.info(f"{filename} exists")
