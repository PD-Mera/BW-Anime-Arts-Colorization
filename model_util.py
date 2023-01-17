import numpy as np
import time, datetime, os
import sklearn.neighbors as nn
import torch

def check_value(inds, val):
    # Check to see if an array is a single element equaling a particular value
    # Good for pre-processing inputs in a function
    if(np.array(inds).size == 1):
        if(inds == val):
            return True
    return False

def torch_setdiff1d(t1, t2, assume_unique=False):
    """
    Set difference of two 1D tensors.
    Returns the unique values in t1 that are not in t2.

    """
    if not assume_unique:
        t1 = torch.unique(t1)
        t2 = torch.unique(t2)
    return t1[(t1[:, None] != t2).all(dim=1)]

def flatten_nd_array(pts_nd, axis=1):
    # Flatten an nd array into a 2d array with a certain axis
    # INPUTS
    # 	pts_nd 		N0xN1x...xNd array
    # 	axis 		integer
    # OUTPUTS
    # 	pts_flt 	prod(N \ N_axis) x N_axis array
    NDIM = pts_nd.ndim
    SHP = np.array(pts_nd.shape)
    nax = np.setdiff1d(np.arange(0, NDIM), np.array((axis)))  # non axis indices
    NPTS = np.prod(SHP[nax])
    axorder = np.concatenate((nax, np.array(axis).flatten()), axis=0)
    pts_flt = pts_nd.permute([int(i) for i in axorder])
    # pts_flt = pts_nd.transpose((axorder))
    pts_flt = pts_flt.reshape(NPTS, SHP[axis])
    return pts_flt

def unflatten_2d_array(pts_flt, pts_nd, axis=1, squeeze=False):
    # Unflatten a 2d array with a certain axis
    # INPUTS
    # 	pts_flt 	prod(N \ N_axis) x M array
    # 	pts_nd 		N0xN1x...xNd array
    # 	axis 		integer
    # 	squeeze 	bool 	if true, M=1, squeeze it out
    # OUTPUTS
    # 	pts_out 	N0xN1x...xNd array
    NDIM = pts_nd.ndim
    SHP = np.array(pts_nd.shape)
    nax = np.setdiff1d(np.arange(0, NDIM), np.array((axis)))  # non axis indices

    if(squeeze):
        axorder = nax
        axorder_rev = np.argsort(axorder)
        M = pts_flt.shape[1]
        NEW_SHP = SHP[nax].tolist()
        pts_out = pts_flt.reshape(NEW_SHP)
        pts_out = pts_out.transpose(axorder_rev)
    else:
        axorder = np.concatenate((nax, np.array(axis).flatten()), axis=0)
        axorder_rev = np.argsort(axorder)
        M = pts_flt.shape[1]
        NEW_SHP = SHP[nax].tolist()
        NEW_SHP.append(M)
        pts_out = pts_flt.reshape(NEW_SHP)
        pts_out = pts_out.transpose(axorder_rev)

    return pts_out


def na():
    return np.newaxis


class Timer():
    def __init__(self):
        self.cur_t = time.time()

    def tic(self):
        self.cur_t = time.time()

    def toc(self):
        return time.time() - self.cur_t

    def tocStr(self, t=-1):
        if(t == -1):
            return str(datetime.timedelta(seconds=np.round(time.time() - self.cur_t, 3)))[:-4]
        else:
            return str(datetime.timedelta(seconds=np.round(t, 3)))[:-4]


class NNEncode():
    # Encode points as a linear combination of unordered points
    # using NN search and RBF kernel
    def __init__(self, NN, sigma, km_filepath='./data/color_bins/pts_in_hull.npy', cc=-1):
        if(check_value(cc, -1)):
            self.cc = np.load(km_filepath)
        else:
            self.cc = cc
        self.K = self.cc.shape[0]
        self.NN = int(NN)
        self.sigma = sigma
        self.nbrs = nn.NearestNeighbors(n_neighbors=self.NN, algorithm='auto').fit(self.cc)

    def encode_points_mtx_nd(self, pts_nd, axis=1, returnSparse=False):
        pts_flt = flatten_nd_array(pts_nd * 255, axis=axis)
        P = pts_flt.shape[0]
        pts_flt = pts_flt.cpu()
        (dists, inds) = self.nbrs.kneighbors(pts_flt)
        dists, inds = torch.tensor(dists), torch.tensor(inds)
        wts = torch.exp(-dists**2 / (2 * self.sigma**2))
        wts = wts / torch.sum(wts, dim=1)[:, na()]


        pts_enc_flt = np.zeros((P, self.K))
        pts_enc_flt[np.arange(0, P, dtype='int')[:, na()], inds] = wts
        pts_enc_nd = unflatten_2d_array(pts_enc_flt, pts_nd, axis=axis)
        return torch.tensor(pts_enc_nd)

    def decode_points_mtx_nd(self, pts_enc_nd, axis=1):
        pts_enc_flt = flatten_nd_array(pts_enc_nd, axis=axis)
        pts_dec_flt = np.dot(pts_enc_flt, self.cc)
        pts_dec_nd = unflatten_2d_array(pts_dec_flt, pts_enc_nd, axis=axis)
        return 

def encode_global(ab_global_tensor, NN = 1.0, sigma = 3.0, ENC_DIR = './color_bins'):
    nnenc = NNEncode(NN, sigma, km_filepath=os.path.join(ENC_DIR, 'pts_in_hull.npy'))
    hint_G = nnenc.encode_points_mtx_nd(ab_global_tensor[:, :, ::4, ::4], axis=1)
    hint_G = torch.nan_to_num(hint_G, nan=0.0, posinf=0.0, neginf=0.0)
    hint_G = hint_G.type(torch.FloatTensor)
    hint_G = torch.mean(hint_G, dim=3)
    hint_G = torch.mean(hint_G, dim=2)

    return hint_G