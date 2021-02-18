import json
from web3 import Web3
import torch 
import numpy as np

def smartContractMaker(clientIDs):
  ganache_url = "http://127.0.0.1:8545"
  web3 = Web3(Web3.HTTPProvider(ganache_url))
  accounts = web3.eth.accounts
  truffleFile = json.load(open('../truffle/build/contracts/Rank.json', encoding="utf-8"))
  abi = truffleFile['abi']
  bytecode = truffleFile['bytecode']
  contract = web3.eth.contract(bytecode=bytecode, abi=abi)
  tx_hash = contract.constructor(len(clientIDs)).transact({'from':accounts[0]})
  tx_receipt = web3.eth.waitForTransactionReceipt(tx_hash)
  print("contract made")
  return tx_receipt.contractAddress, abi

class SmartContract:
  def __init__(self, clientIDs, contractAddress, abi):
    self.numclient = len(clientIDs)
    ganache_url = "http://127.0.0.1:8545"
    self.web3 = Web3(Web3.HTTPProvider(ganache_url, request_kwargs={'timeout': 120}))
    accounts = self.web3.eth.accounts
    self.accounts = {clientIDs[i]: accounts[i] for i in range(len(clientIDs))}
    self.contract = self.web3.eth.contract(address=contractAddress, abi=abi)

  def upload_tx(self, clientID, points):
    distPoints, answerOnNthPoints = points
    print(clientID, distPoints, answerOnNthPoints)
    tx_hash = self.contract.functions.upload(distPoints, answerOnNthPoints).transact({'from':self.accounts[clientID]})
    self.web3.eth.waitForTransactionReceipt(tx_hash)
    print("upload_contract by client : {}".format(clientID))

  def seeRank1_call(self):
    return self.contract.functions.seeRank1().call()
  
  def seeRank2_call(self):
    return self.contract.functions.seeRank2().call()