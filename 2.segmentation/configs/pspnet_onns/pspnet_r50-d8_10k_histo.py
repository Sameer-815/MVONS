# _base_ = [
#     '../_base_/models/pspnet_r50_wsss.py',
#     '../_base_/datasets/wsss_bcss_224.py', '../_base_/default_runtime.py',
#     '../_base_/schedules/schedule_10k.py'
# ]       #BCSS
_base_ = [
    '../_base_/models/pspnet_r50_wsss.py',
    '../_base_/datasets/wsss_luad_224.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_10k.py'
]       #LUAD
model = dict(
    decode_head=dict(num_classes=4), auxiliary_head=dict(num_classes=4))
