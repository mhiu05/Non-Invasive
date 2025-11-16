import math

import numpy as np
from scipy import linalg
from scipy import signal
from rPPG_Toolbox.unsupervised_methods import utils


def LGI(frames):
    precessed_data = utils.process_video(frames)
    U, _, _ = np.linalg.svd(precessed_data)
    S = U[:, :, 0]
    S = np.expand_dims(S, 2)
    SST = np.matmul(S, np.swapaxes(S, 1, 2))
    p = np.tile(np.identity(3), (S.shape[0], 1, 1))
    P = p - SST
    Y = np.matmul(P, precessed_data)
    bvp = Y[:, 1, :]
    bvp = bvp.reshape(-1)
    return bvp