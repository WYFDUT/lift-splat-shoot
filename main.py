"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

from fire import Fire

import src


if __name__ == '__main__':
    # Document for Fire Lib
    # https://blog.csdn.net/qq_17550379/article/details/79943740?ops_request_misc=&request_id=&biz_id=102&utm_term=python%20Fire%E5%BA%93&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-4-79943740.142^v99^control&spm=1018.2226.3001.4187
    Fire({
        'lidar_check': src.explore.lidar_check,
        'cumsum_check': src.explore.cumsum_check,

        'train': src.train.train,
        'eval_model_iou': src.explore.eval_model_iou,
        'viz_model_preds': src.explore.viz_model_preds,
    })