import torch

import parameters as pm

def start_ps(server):
    ps = server
    for i in range(10):
        ps.update()

class parameterServer():
    def __init__(self, paramTensor, n_chunks, p_level):
        self.chunks = []
        for i in range(n_chunks):
          self.chunks.append paramTensor[int(i/n_chunks*len(paramTensor))) : int((i+1)/n_chunks*len(paramTensor)))])
        self.n_chunks = n_chunks
        self.p_level = p_level
        self.update_infos = []


    def download_params(self):
      return torch.cat(self.chunks)


    def upload(self, update_info):
      if len(self.update_infos) > self.p_level:
        return
      self.update_infos.append(update_info)

    
    def push(self, chunk_num, newChunk):
      self.chunks[chunk_num] = newChunk


    def update(self):
        print("------------------server updating------------------")
        print(self.update_infos)
        accepted_bids = {}
        for update_info in self.update_infos:
          for key in update_info.keys():
            if key in accepted_bids.keys():
              if accepted_bids[key][0] < update_info[key][0]:
                accepted_bids[key] = update_info[key]
            else:
              accepted_bids[key] = update_info[key]
        
        for key in accepted_bids.keys():
          accepted_bids[key][1].push(key)

        self.update_infos = []