import os
import tensorflow as tf
import urllib.request
import sys, tarfile, glob
import shutil

file_path = './cifar100'
batch_path = './cifar-100-python'
f = {}
if not os.path.exists(file_path):
    os.makedirs(file_path)

os.chdir(file_path)

url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
file_name = url.split('/')[-1]
u = urllib.request.urlopen(url)
file_meta = u.info()
file_size = int(file_meta["Content-Length"])

if not os.path.exists(file_name) or os.stat(file_name).st_size != file_size:
    f = open(file_name, 'wb')
    print ("Downloading: %s Bytes: %s" % (file_name, file_size))
    file_size_dl = 0
    file_block_sz = 1024 * 64
    while True:
        buffer = u.read(file_block_sz)
        if not buffer:
               break
        file_size_dl += len(buffer)
        f.write(buffer)
        status = r"%3.2f mb [%3.2f %%]" % (file_size_dl/(1024*1024),file_size_dl * 100. /file_size)
        status = status + chr(8)*(len(status)+1)
        sys.stdout.write(status)
        sys.stdout.flush()
    f.close()

f = tarfile.open(file_name)
f.extractall()
f.close()

for file in glob.glob('./*'):
	if os.path.isfile(file):
		os.remove(file)

for file in glob.glob(batch_path+'/*'):
	print(file)
	shutil.move(file, '.')
shutil.rmtree(batch_path)
