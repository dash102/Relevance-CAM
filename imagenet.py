import os
import urllib.request
from tqdm import tqdm

base_path = "/data/vision/torralba/datasets/imagenet_pytorch/"
output_path = 'dataset/imagenet'

url = "https://raw.githubusercontent.com/torch/tutorials/master/7_imagenet_classification/synset_words.txt"
# another resource: https://github.com/pytorch/vision/issues/484

content = urllib.request.urlopen(url)
wid_list = [line.decode('ascii').strip().split(' ', 1)[0] for line in content]

for split in ['train', 'val']:
    os.makedirs(os.path.join(output_path, split), exist_ok=True)
    for idx, wid in enumerate(tqdm(wid_list)):
        link_src = os.path.join(base_path, split, wid)
        link_tgt = os.path.join(output_path, split, '%04d' % idx)
        cmd = "ln -s %s %s" % (link_src, link_tgt)
        os.system(cmd)