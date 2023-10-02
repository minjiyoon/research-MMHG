from enum import Enum
from transformers import AutoFeatureExtractor
from PIL import Image
import shutil
import torch
import torch.distributed as dist
#import torch_scatter
import nltk
try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    nltk.download("punkt", quiet=True)

from scipy.sparse.linalg import eigsh
import numpy as np

def scatter_sum(src, index, dim=-1, out_size=None):
    if out_size is None:
        out_size = int(index.max().item()) + 1

    # Create an output tensor filled with zeros
    size = list(src.size())
    size[dim] = out_size
    out = torch.zeros(*size, device=src.device, dtype=src.dtype)

    # Expand index tensor to have the same size as src
    expanded_idx = index.unsqueeze(dim).expand_as(src)

    # Scatter the source values
    out.scatter_add_(dim, expanded_idx, src)

    return out


def to_scipy_sparse_matrix(edge_index, edge_attr = None, num_nodes = None):
    row, col = edge_index
    edge_attr = edge_attr.view(-1)
    N = num_nodes
    out = scipy.sparse.coo_matrix((edge_attr.numpy(), (row.numpy(), col.numpy())), (N, N))
    return out

def remove_self_loops(edge_index, edge_attr):
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]
    return edge_index, None #edge_attr[mask]

def add_self_loops(
    edge_index,
    edge_attr,
    fill_value,
    num_nodes,
):
    N = num_nodes
    size = (N, N)
    loop_index = torch.arange(0, N, dtype=torch.long, device=edge_index.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)
    loop_attr = edge_attr.new_full((N, ) + edge_attr.size()[1:], fill_value)
    edge_attr = torch.cat([edge_attr, loop_attr], dim=0)
    edge_index = torch.cat([edge_index, loop_index], dim=1)
    return edge_index, edge_attr


def get_laplacian(
    edge_index,
    edge_weight= None,
    normalization= None,
    num_nodes= None,
):
    edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

    edge_weight = torch.ones(edge_index.size(1), dtype=torch.float, device=edge_index.device)

    row, col = edge_index[0], edge_index[1]
    deg = scatter_sum(edge_weight, row, 0, out_size=num_nodes)

    # Compute A_norm = -D^{-1/2} A D^{-1/2}.
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    edge_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    # L = I - A_norm.
    edge_index, tmp = add_self_loops(edge_index, -edge_weight,
                                        fill_value=1., num_nodes=num_nodes)
    assert tmp is not None
    edge_weight = tmp
    return edge_index, edge_weight


def normalize_graph(graph):
    graph = torch.where(graph + graph.t() > 0, 1, 0)
    graph = graph * (1 - torch.eye(graph.size(0)))
    row_sum = torch.sum(graph, dim=1)
    row_sum = row_sum.masked_fill_(row_sum == 0, 1.)
    row_sum = torch.diag(1/row_sum)
    graph = torch.mm(row_sum, graph)
    return graph


def compute_LPE(data):
    num_nodes = data.num_nodes
    k = num_nodes - 5
    edge_index, edge_weight = get_laplacian(
        data.edge_index,
        data.edge_weight,
        normalization='sym',
        num_nodes=num_nodes,
    )

    L = to_scipy_sparse_matrix(edge_index, edge_weight, num_nodes)

    eig_vals, eig_vecs = eigsh(
        L,
        k=k+1,
        which='SA',
        return_eigenvectors=True,
    )

    eig_vecs = np.real(eig_vecs[:, eig_vals.argsort()])
    pe = torch.from_numpy(eig_vecs[:, 1:(k+1)])
    sign = -1 + 2 * torch.randint(0, 2, (k, ))
    pe *= sign
    return pe


def get_feature_extractor_for_model(model_name: str):
    print(f'Using HuggingFace AutoFeatureExtractor for {model_name}.')
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
    return feature_extractor


def get_pixel_values_for_model(feature_extractor, img: Image.Image):
    pixel_values = feature_extractor(img.convert('RGB'), return_tensors="pt").pixel_values[0, ...]  # (3, H, W)
    return pixel_values


def get_params_count(model, max_name_len: int = 60):
    params = [(name[:max_name_len], p.numel(), str(tuple(p.shape)), p.requires_grad) for name, p in model.named_parameters()]
    total_trainable_params = sum([x[1] for x in params if x[-1]])
    total_nontrainable_params = sum([x[1] for x in params if not x[-1]])
    return params, total_trainable_params, total_nontrainable_params


def get_params_count_str(model, max_name_len: int = 60):
    padding = 70  # Hardcoded depending on desired amount of padding and separators.
    params, total_trainable_params, total_nontrainable_params = get_params_count(model, max_name_len)
    param_counts_text = ''
    param_counts_text += '=' * (max_name_len + padding) + '\n'
    param_counts_text += f'| {"Module":<{max_name_len}} | {"Trainable":<10} | {"Shape":>15} | {"Param Count":>12} |\n'
    param_counts_text += '-' * (max_name_len + padding) + '\n'
    for name, param_count, shape, trainable in params:
        param_counts_text += f'| {name:<{max_name_len}} | {"True" if trainable else "False":<10} | {shape:>15} | {param_count:>12,} |\n'
    param_counts_text += '-' * (max_name_len + padding) + '\n'
    param_counts_text += f'| {"Total trainable params":<{max_name_len}} | {"":<10} | {"":<15} | {total_trainable_params:>12,} |\n'
    param_counts_text += f'| {"Total non-trainable params":<{max_name_len}} | {"":<10} | {"":<15} | {total_nontrainable_params:>12,} |\n'
    param_counts_text += '=' * (max_name_len + padding) + '\n'
    return param_counts_text


def save_checkpoint(state, is_best, filename='checkpoint'):
    torch.save(state, filename + '.pth.tar')
    if is_best:
        shutil.copyfile(filename + '.pth.tar', filename + '_best.pth.tar')


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def all_reduce(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        total = torch.tensor([self.sum, self.count], dtype=torch.float32, device=device)
        dist.all_reduce(total, dist.ReduceOp.SUM, async_op=False)
        self.sum, self.count = total.tolist()
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ''
        if self.summary_type is Summary.NONE:
            fmtstr = ''
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = '{name} {avg:.3f}'
        elif self.summary_type is Summary.SUM:
            fmtstr = '{name} {sum:.3f}'
        elif self.summary_type is Summary.COUNT:
            fmtstr = '{name} {count:.3f}'
        else:
            raise ValueError('invalid summary type %r' % self.summary_type)

        return fmtstr.format(**self.__dict__)

