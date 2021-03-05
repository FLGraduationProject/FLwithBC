import json
from web3 import Web3
import torch
import numpy as np

ganache_url = "http://127.0.0.1:8545"


def smartContractMaker(clientIDs, maxHeapSize):
	web3 = Web3(Web3.HTTPProvider(ganache_url))
	accounts = web3.eth.accounts
	truffleFile = json.load(
		open('../truffle/build/contracts/PointsBoard.json', encoding="utf-8"))
	abi = truffleFile['abi']
	bytecode = truffleFile['bytecode']
	contract = web3.eth.contract(bytecode=bytecode, abi=abi)
	tx_hash = contract.constructor(
		maxHeapSize).transact({'from': accounts[0]})
	tx_receipt = web3.eth.waitForTransactionReceipt(tx_hash)
	print("contract made")
	return tx_receipt.contractAddress, abi


class SmartContract:
	def __init__(self, clientIDs, contractAddress, abi):
		self.web3 = Web3(Web3.HTTPProvider(ganache_url, request_kwargs={'timeout': 120}))
		self.clientIDs = clientIDs
		accounts = self.web3.eth.accounts
		self.accounts = {clientIDs[i]: accounts[i] for i in range(len(clientIDs))}
		self.contract = self.web3.eth.contract(address=contractAddress, abi=abi)

	def upload_tx(self, clientID, uploadData):
		print(uploadData)
		teacherIDs, points = uploadData['teacherIDs'], uploadData['points']
		teacherAddrs = [self.accounts[teacherID] for teacherID in teacherIDs]
		tx_hash = self.contract.functions.upload(
			teacherAddrs, points).transact({'from': self.accounts[clientID]})
		self.web3.eth.waitForTransactionReceipt(tx_hash)

	def seeTeachersRank(self, teacherIDs):
		medianPoints = {}

		for teacherID in teacherIDs:
			medianPoint = self.contract.functions.seeMedianPoint(self.accounts[teacherID]).call()
			if medianPoint != 0:
				medianPoints[teacherID] = medianPoint

		sortedTeachersList = sorted(
			medianPoints.keys(), key=lambda teacherID: medianPoints[teacherID])

		return {sortedTeachersList[i]: i+1 for i in range(len(sortedTeachersList))}

	def getTeachersHeap(self):
		medianPoints = {clientID: self.contract.functions.seeMedianPoint(
			self.accounts[clientID]).call() for clientID in self.clientIDs}

		rankFromTop = sorted(
			medianPoints.keys(), key=lambda clientID: medianPoints[clientID])
		return rankFromTop
