import runpy
import sys
import re

# from https://github.com/vecto-ai/vecto
def run_module(name: str, args, run_name: str = '__main__') -> None:
    backup_sys_argv = sys.argv
    sys.argv = [name + '.py'] + list(args)
    runpy.run_module(name, run_name=run_name)
    sys.argv = backup_sys_argv

# run_module('Multi_CAM', [
#     '--models', 'resnet50',
#     '--target_layer', 'layer2',
#     '--seed', '0'
# ])

i = 0
cam = 'relevance_cam'

# run_module('eval3', [
#     '--models', 'resnet50',
#     '--cam', cam,
#     '--target_layer', 'layer2',
#     '--upsample', 'jbu-transformed',
#     '--seed', str(i)
# ])

run_module('eval3', [
    '--models', 'resnet50',
    '--cam', cam,
    '--target_layer', 'layer4',
    # '--upsample', 'jbu-sigma-transformed-guidance',
    '--inp_size', '224',
    '--seed', str(i)
])

# run_module('eval3', [
#     '--models', 'vgg16',
#     '--cam', cam,
#     '--target_layer', '23',
#     '--upsample', 'jbu-plain',
#     '--seed', str(i)
#     ])
#
# run_module('eval3', [
#     '--models', 'vgg16',
#     '--cam', cam,
#     '--target_layer', '43',
#     '--upsample', 'jbu-plain',
#     '--seed', str(i)
# ])
