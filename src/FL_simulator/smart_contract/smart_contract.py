import json
from web3 import Web3
import torch
import numpy as np
import statistics

ganache_url = "http://127.0.0.1:8545"


def smartContractMaker(clientIDs, maxHeapSize, n_teachers):
	web3 = Web3(Web3.HTTPProvider(ganache_url))
	accounts = web3.eth.accounts
	truffleFile = json.load(
		open('../truffle/build/contracts/PointsBoard.json', encoding="utf-8"))
	abi = truffleFile['abi']
	bytecode = truffleFile['bytecode']
	contract = web3.eth.contract(bytecode=bytecode, abi=abi)
	tx_hash = contract.constructor(
		maxHeapSize, n_teachers).transact({'from': accounts[0]})
	tx_receipt = web3.eth.waitForTransactionReceipt(tx_hash)
	print("contract made")
	return tx_receipt.contractAddress, abi


class SmartContract:
	def __init__(self, clientIDs, contractAddress, abi):
		self.web3 = Web3(Web3.HTTPProvider(ganache_url, request_kwargs={'timeout': 120}))
		self.clientIDs = clientIDs
		accounts = self.web3.eth.accounts
		self.accounts = {clientIDs[i]: accounts[i] for i in range(len(clientIDs))}
		self.accountToID = {self.accounts[clientID]: clientID for clientID in clientIDs}
		self.contract = self.web3.eth.contract(address=contractAddress, abi=abi)

	def uploadPoints(self, clientID, uploadData):
		teacherIDs, points = uploadData['teacherIDs'], uploadData['points']
		teacherAddrs = [self.accounts[teacherID] for teacherID in teacherIDs]
		tx_hash = self.contract.functions.uploadPoints(
			teacherAddrs, points).transact({'from': self.accounts[clientID]})
		self.web3.eth.waitForTransactionReceipt(tx_hash)

	def getTeachersInRank(self, clientID):
		tx_hash = self.contract.functions.assignTeachers().transact({'from': self.accounts[clientID]})
		self.web3.eth.waitForTransactionReceipt(tx_hash)
		
		teacherAddrs = self.contract.functions.seeTeachers().call({'from': self.accounts[clientID]})
		teachersPoints = {}
		print(teacherAddrs)
		for teacherAddr in teacherAddrs:
			points = self.contract.functions.seePoints(teacherAddr).call({'from': self.accounts[clientID]})
			if len(points) == 0:
				med = 0
			else:
				med = statistics.median(points)
			teachersPoints[self.accountToID[teacherAddr]] = med
		
		return sorted(list(teachersPoints.keys()), key=lambda k: teachersPoints[k])