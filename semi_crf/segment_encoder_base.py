#!/usr/bin/env python
import torch


class SegmentEncoderBase(torch.nn.Module):
    def __init__(self, max_seg_len, use_cuda):
        super(SegmentEncoderBase, self).__init__()
        self.max_seg_len = max_seg_len
        self.use_cuda = use_cuda

    def encoding_dim(self) -> int:
        raise NotImplementedError()
