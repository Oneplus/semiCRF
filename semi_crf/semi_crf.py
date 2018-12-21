#!/usr/bin/env python
import torch


class ZeroOrderSemiCRFLayer(torch.nn.Module):
    def __init__(self, use_cuda):
        super(ZeroOrderSemiCRFLayer, self).__init__()
        self.use_cuda = use_cuda

    def forward(self, transitions: torch.Tensor,
                tags: torch.Tensor,
                lens: torch.Tensor):

        if self.training:
            numerator = self._compute_joint_llh(transitions, tags, lens)
            denominator = self._compute_log_partition_function(transitions, lens)
            llh = denominator - numerator

            loss = torch.sum(llh)
            return None, loss
        else:
            path, tags = self._viterbi_decode(transitions)
            pred = self.get_labels(path, tags, lens)
            return pred, None

    def _compute_joint_llh(self, transitions: torch.Tensor,
                           tags: torch.Tensor,
                           lens: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, max_seg_len, n_tags = transitions.size()
        # (batch_size, seq_len + 1)
        alpha = torch.zeros(batch_size, seq_len + 1)
        if self.use_cuda:
            alpha = alpha.cuda()

        for ending_pos in range(1, seq_len + 1):
            length = min(max_seg_len, ending_pos)
            indices = list(reversed(range(length)))
            indices = torch.LongTensor(indices).cuda() if self.use_cuda else torch.LongTensor(indices)
            # indices: (length)

            alpha_rev = alpha[:, ending_pos - length: ending_pos].index_select(1, indices)
            # alpha_rev: (batch_size, length)

            transition = transitions[:, ending_pos - 1, :length, :]
            # transition: (batch_size, length, n_tags)

            f = alpha_rev.unsqueeze(2) + transition
            f = f.view(batch_size, -1)
            # f: (batch_size, length * n_tags)
            indices = tags[:, ending_pos - 1].view(-1, 1)
            # indices: (batch_size, 1)

            alpha[:, ending_pos] = f.gather(1, indices).squeeze(-1)
        llh = alpha.gather(1, lens.view(-1, 1))
        return llh

    def _compute_log_partition_function(self, transitions: torch.Tensor,
                                        lens: torch.Tensor):
        # transitions: (batch_size, seq_len,
        batch_size, seq_len, max_seg_len, n_tags = transitions.size()

        alpha = torch.zeros(batch_size, seq_len + 1)

        if self.use_cuda:
            alpha = alpha.cuda()

        for ending_pos in range(1, seq_len + 1):
            length = min(max_seg_len, ending_pos)
            # (batch_size, l)
            indices = list(reversed(range(length)))
            indices = torch.LongTensor(indices).cuda() if self.use_cuda else torch.LongTensor(indices)

            alpha_rev = alpha[:, ending_pos - length: ending_pos].index_select(1, indices)
            # (batch_size, l)
            transition = transitions[:, ending_pos - 1, :length, :]

            f = alpha_rev.unsqueeze(2) + transition
            f = f.view(batch_size, -1)
            alpha[:, ending_pos] = self._log_sum_exp(f, 1)

        llh = torch.gather(alpha, 1, lens.view(-1, 1))
        return llh

    def _viterbi_decode(self, transitions: torch.Tensor):
        # transitions: (batch_size, seq_len, max_seg_len, n_tags)
        batch_size, seq_len, max_seg_len, n_tags = transitions.size()

        alpha = torch.FloatTensor(batch_size, seq_len + 1).fill_(-1e9)
        path = torch.LongTensor(batch_size, seq_len + 1).fill_(-1)
        tags = torch.LongTensor(batch_size, seq_len + 1).fill_(-1)

        # set all tags in the first place to zero
        alpha[:, 0] = 0
        if self.use_cuda:
            alpha, path, tags = alpha.cuda(), path.cuda(), tags.cuda()

        for ending_pos in range(1, seq_len + 1):
            length = min(max_seg_len, ending_pos)
            indices = list(reversed(range(length)))
            indices = torch.LongTensor(indices).cuda() if self.use_cuda else torch.LongTensor(indices)

            # alpha_rev: (batch_size, length)
            alpha_rev = alpha[:, ending_pos - length: ending_pos].index_select(1, indices)

            # transition: (batch_size, length, n_tags)
            transition = transitions[:, ending_pos - 1, :length, :]

            # f: (batch_size, length, n_tags)
            f = alpha_rev.unsqueeze(2) + transition
            f = f.view(batch_size, -1)
            alpha[:, ending_pos], tmp = f.max(-1)

            tags[:, ending_pos] = tmp.fmod(n_tags)
            path[:, ending_pos] = tmp // n_tags

        # add one to length
        return path[:, 1:] + 1, tags[:, 1:]

    @staticmethod
    def _log_sum_exp(tensor, dim):
        # Find the max value along `dim`
        offset, _ = tensor.max(dim)
        # Make offset broadcastable
        broadcast_offset = offset.unsqueeze(dim)
        # Perform log-sum-exp safely
        safe_log_sum_exp = torch.log(torch.sum(torch.exp(tensor - broadcast_offset), dim))
        # Add offset back
        return offset + safe_log_sum_exp

    @staticmethod
    def get_labels(paths: torch.Tensor, tags: torch.Tensor, lens: torch.Tensor):
        batch_size = paths.size(0)
        paths_list = paths.data.tolist()
        tags_list = tags.data.tolist()
        lens_list = lens.data.tolist()
        outputs = []
        for i in range(batch_size):
            paths = paths_list[i]
            tags = tags_list[i]
            length = lens_list[i]
            segments = []
            while length > 0:
                segments.append((paths[length - 1], tags[length - 1]))
                length = length - paths[length - 1]
            segments.reverse()
            start_id = 0
            output = []
            for length, tag in segments:
                output.append((start_id, length, tag))
                start_id += length
            outputs.append(output)
        return outputs


if __name__ == "__main__":
    semi_crf = ZeroOrderSemiCRFLayer(False)

    seq_len = 5
    max_seg_len = 3
    n_tags = 4
    batch_size = 2

    transitions = torch.FloatTensor(batch_size, seq_len, max_seg_len, n_tags).fill_(0.)
    transitions[0][1][1][1] = 1.
    transitions[0][3][1][0] = 1.
    transitions[0][4][0][1] = 1.
    transitions[1][1][1][0] = 1.
    transitions[1][3][1][1] = 1.
    transitions[1][4][0][0] = 1.

    tags = torch.LongTensor(batch_size, seq_len).fill_(0)
    tags[0][1] = (2 - 1) * n_tags + 1
    tags[0][3] = (2 - 1) * n_tags + 0
    tags[0][4] = (1 - 1) * n_tags + 1
    tags[1][1] = (2 - 1) * n_tags + 1
    tags[1][3] = (2 - 1) * n_tags + 0
    tags[1][4] = (1 - 1) * n_tags + 1

    lens = torch.LongTensor(batch_size).fill_(0)
    lens[0] = seq_len
    lens[1] = seq_len

    #semi_crf.train()
    #print(semi_crf.forward(transitions, tags, lens))

    semi_crf.eval()
    print(semi_crf.forward(transitions, None, lens))
