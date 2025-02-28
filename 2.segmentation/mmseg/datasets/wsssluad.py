import os.path as osp

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class WSSSLUAD(CustomDataset):
    """STARE dataset.

    In segmentation map annotation for STARE, 0 stands for background, which is
    included in 2 categories. ``reduce_zero_label`` is fixed to False. The
    ``img_suffix`` is fixed to '.png' and ``seg_map_suffix`` is fixed to
    '.ah.png'.
    """

    CLASSES = ('Tumor', 'Necrosis','Lymphocyte','Stroma',)

    PALETTE = [[205,51,51], [0,255,0],[65,105,225],[255,165,0]]

    def __init__(self, **kwargs):
        super(WSSSLUAD, self).__init__(
            img_suffix='.png',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)
        assert osp.exists(self.img_dir)
