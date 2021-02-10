import json
from web3 import Web3
import torch 
import numpy as np

def smartContractMaker(clientIDs):
  ganache_url = "http://127.0.0.1:7545"
  web3 = Web3(Web3.HTTPProvider(ganache_url))
  accounts = web3.eth.accounts
  truffleFile = json.load(open('../truffle/build/contracts/Rank.json', encoding="utf-8"))
  abi = truffleFile['abi']
  bytecode = truffleFile['bytecode']
  contract = web3.eth.contract(bytecode=bytecode, abi=abi)
  tx_hash = contract.constructor().transact({'from':accounts[0]})
  tx_receipt = web3.eth.waitForTransactionReceipt(tx_hash)
  print("contract made")
  return tx_receipt.contractAddress, abi

class SmartContract:
  def __init__(self, clientIDs, contractAddress, abi):
    self.numclient = len(clientIDs)
    ganache_url = "http://127.0.0.1:7545"
    self.web3 = Web3(Web3.HTTPProvider(ganache_url))
    accounts = self.web3.eth.accounts
    self.accounts = {clientIDs[i]: accounts[i] for i in range(len(clientIDs))}
    self.contract = self.web3.eth.contract(address=contractAddress, abi=abi)
    print("contract made")

  def set_contract(self,clientID,numclient):
    print("number of client {}".format(self.numclient))
    tx_hash = self.contract.functions.setting(numclient).transact({'from':self.accounts[clientID]})
    self.web3.eth.waitForTransactionReceipt(tx_hash)

  def upload_contract(self, clientID, avgdist):
    tx_hash = self.contract.functions.upload(avgdist).transact({'from':self.accounts[clientID]})
    self.web3.eth.waitForTransactionReceipt(tx_hash)
    print("upload_contract by client : {}".format(clientID))


  # def rank_contract(self, clientID):
  #   print("rank_contract by client : {}".format(clientID))
  #   tx_hash = self.contract.functions.ranking({from:self.accounts[clientID]}).transact()
  #   web3.eth.waitForTransactionReceipt(tx_hash)

  def seerank_contract(self):
    return self.contract.functions.see_rank().call()