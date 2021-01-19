import heapq
import torch


def tensor_find(tensor, tensorSize, index):
    remaining = index
    loc = []
    for i in range(-1, -len(tensorSize) - 1, -1):
        size = tensorSize[i]
        loc.insert(0, remaining % size)
        remaining = int(remaining / size)

    val = tensor
    for l in loc:
        val = val[l]

    return val, loc


def plus_minus(x):
  return 1 if x > 0 else -1


class Gradients:
    def __init__(self, params1, params2):
        self.grads = {key: params1[key] - params2[key] for key in params1.keys()}

    def topN(self, N):
        abs_top_N = []

        for key in self.grads.keys():
            size = self.grads[key].size()
            total_size = 1
            for num in size:
                total_size *= num

            for i in range(total_size):
                val, loc = tensor_find(self.grads[key], size, i)
                if len(abs_top_N) < N:
                    heapq.heappush(abs_top_N, [abs(val), key, loc, plus_minus(val)])

                elif abs_top_N[0][0] > abs(val):
                    heapq.heapreplace(abs_top_N, [abs(val), key, loc, plus_minus(val)])

        return abs_top_N


class Parameters:
    def __init__(self, paramTensor=None):
        self.paramTensor = paramTensor # flattened tensor

    def getAsDict(self, dic):
      start = 0
      for key in dic.keys():
        shape = dic[key].shape
        length = torch.prod(torch.tensor(shape))
        dic[key] = self.paramTensor[start:start + length].reshape(shape)
        start += length
      return dic
    
    def setByDict(self, dic):
      self.paramTensor = torch.cat([torch.flatten(dic[key]) for key in dic.keys()])

    def update_params(self, update_info, DSSGD=False, fedAvg=False):
        if DSSGD:
            for grad in update_info:
                val = self.params[grad[1]]
                for l in grad[2]:
                    val = val[l]
                val += grad[0]*grad[3]
        elif fedAvg:
            average = {key: torch.sum(torch.stack([info[0][key]*info[1] for info in update_info], dim=1), 1) for key in update_info[0][0].keys()}
            self.params = average