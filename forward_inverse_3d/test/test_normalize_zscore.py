'''
归一化函数的单元测试
'''

import numpy as np
from utils.visualize_tools import plot_standard_12_lead
from utils.signal_processing_tools import (
    normalize_ecg_zscore,
    transfer_bsp_to_standard12lead,
)

d_data = np.load('forward_inverse_3d/data/inverse/u_data_normal.npy')
standard_12 = transfer_bsp_to_standard12lead(d_data)
normalized_data = normalize_ecg_zscore(standard_12)

# plot standard 12 leads before and after normalization
import multiprocessing

p1 = multiprocessing.Process(
    target=plot_standard_12_lead,
    kwargs={
        'standard12Lead': standard_12,
        'filter_flag': False,
    },
)

p2 = multiprocessing.Process(
    target=plot_standard_12_lead,
    kwargs={
        'standard12Lead': normalized_data,
        'filter_flag': False,
    },
)
p1.start()
p2.start()

p1.join()
p2.join()
