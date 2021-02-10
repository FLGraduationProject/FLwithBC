import json
from web3 import Web3
import torch 
import numpy as np

class SmartContract:
  def __init__(self):
    self.numclient
    ganache_url = "http://127.0.0.1:7545"
    web3 = Web3(Web3.HTTPProvider(ganache_url))
    web3.eth.getAccounts().then(function(result){accounts = result})
    #web3.eth.defaultAccount = web3.eth.accounts[0]
    truffleFile = json.load(open('./build/contracts/Rank.json'))
    abi = truffleFile['abi']
    bytecode = truffleFile['bytecode']
    contract = web3.eth.contract(bytecode=bytecode, abi=abi)
    tx_hash = contract.constructor().transact()
    tx_receipt = web3.eth.waitForTransactionReceipt(tx_hash)
    self.contract = web3.eth.contract(address=tx_receipt.contractAddress, abi=abi)
    print("contract made")

  def set_contract(self,clientID,numclient):
    print("number of client {}".format(self.numclient))
    tx_hash = self.contract.functions.setting(numclient, {from:accounts[clientID]}).transact()
    web3.eth.waitForTransactionReceipt(tx_hash)

  def upload_contract(self, clientID, avgdist): 
    print("upload_contract by client : {}".format(clientID))
    tx_hash = self.contract.functions.upload(avgdist, {from:accounts[clientID]}).transact()
    web3.eth.waitForTransactionReceipt(tx_hash)

  def rank_contract(self, clientID):
    print("rank_contract by client : {}".format(clientID))
    tx_hash = self.contract.functions.ranking({from:accounts[clientID]}).transact()
    web3.eth.waitForTransactionReceipt(tx_hash)

  def seerank_contract(self):
    return self.contract.functions.see_rank().call()



