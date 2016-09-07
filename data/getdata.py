import requests
from tqdm import tqdm
import os
from numpy import ceil

def download_file(url, filename, nested=False):
    # NOTE the stream=True parameter
    r = requests.get(url, stream=True)
    chunk_size = 8192
    file_size = int(r.headers.get('content-length'))
    total = int(ceil(file_size / chunk_size))
    with open(filename, 'wb') as f:
        for chunk in tqdm(r.iter_content(chunk_size=chunk_size), total=total, nested=nested): 
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)
                #f.flush() commented by recommendation from J.F.Sebastian


url_start =  "https://s3-us-west-2.amazonaws.com/1605-pk-efm-data/"
folders = ['tr-efm/', 'pk-efm/']

files = ['pk-efm/151217-201951-p1sun-phasekick.h5',
'pk-efm/151217-205912-p3sun-phasekick.h5',
'pk-efm/151217-214045-1sun-phasekick.h5',
'pk-efm/151217-220252-1sun-phasekick-shorter.h5',
'pk-efm/151217-233507-20sun-phasekick.h5',
'pk-efm/151218-001254-20sun-phasekick-short.h5',
'pk-efm/151218-002059-20sun-phasekick-short.h5',
'pk-efm/151218-004818-100sun-phasekick.h5',
'pk-efm/151218-011055-100sun-phasekick-768.h5',
'pk-efm/151218-012858-20sun-phasekick-768.h5',
'tr-efm/151217-200319-p1sun-df.h5',
'tr-efm/151217-205007-p3sun-df.h5',
'tr-efm/151217-211131-1sun-df.h5',
'tr-efm/151217-234238-20sun-df-384.h5',
'tr-efm/151218-003450-100sun-784.h5',]


for folder in folders:
    try:
        os.mkdir(folder)
    except:
        pass

# Download all the files
for file in tqdm(files, desc="Files", nested=True):
    if not os.path.exists(file):
        download_file(url_start+file, file, nested=True)
