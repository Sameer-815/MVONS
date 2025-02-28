import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class WSSSBCSS(CustomDataset):
    """STARE dataset.

    In segmentation map annotation for STARE, 0 stands for background, which is
    included in 2 categories. ``reduce_zero_label`` is fixed to False. The
    ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is fixed to
    '.ah.png'.
    """

    # CLASSES = ('Tumor', 'Stroma','Lymphocyte','Necrosis','Background')
    CLASSES = ('Tumor', 'Stroma', 'Lymphocyte', 'Necrosis')
    PALETTE = [[255, 0, 0], [0,255,0],[0,0,255], [153, 0, 255]]

    def __init__(self, **kwargs):
        super(WSSSBCSS, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
        assert osp.exists(self.img_dir)
