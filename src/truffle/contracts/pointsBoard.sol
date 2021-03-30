// SPDX-License-Identifier: MIT
pragma solidity >=0.5.1 <0.9.0;

import {PointsQ, PointsQueue} from "./pointsQ.sol";

import {Q, Queue} from "./queue.sol";

contract PointsBoard {
    // if not use rounds, we have to save the sender
    // or erase the previous points uploaded
    mapping(address => PointsQ) pointsQueue;
    Q recentUpload;
    mapping(address => address[]) assignedTeachers;
    uint256 n_points;
    uint256 n_teachers;
    uint256 qSize;

    constructor(uint256 nPoints, uint256 nTeachers) {
        n_points = nPoints;
        n_teachers = nTeachers;
        qSize = n_teachers * 4;
        Queue.initialize(recentUpload, qSize);
    }

    function randomNumber() internal view returns (uint256) {
        return uint256(keccak256(abi.encodePacked(block.timestamp)));
    }

    function assignTeachers() public {
        // recently uploaded n_teachers will be teachers
        delete assignedTeachers[msg.sender];
        uint256 randIdx = randomNumber() % qSize;
        address teacher = Queue.byIndex(recentUpload, randIdx);
        for (uint256 i = 0; i < n_teachers && teacher != address(0); i++) {
            assignedTeachers[msg.sender].push(teacher);
            randIdx = randIdx + 47;
            teacher = Queue.byIndex(recentUpload, randIdx);
        }
    }

    function seeTeachers() public view returns (address[] memory) {
        return assignedTeachers[msg.sender];
    }

    function seePoints(address teacher) public view returns (uint256[] memory) {
        uint256 size = pointsQueue[teacher].size;
        uint256[] memory points = new uint256[](size);
        address curr = pointsQueue[teacher].next[address(0)];
        for (uint256 i = 0; i < size; i++) {
            uint256 point = pointsQueue[teacher].value[curr];
            points[i] = point;
            curr = pointsQueue[teacher].next[curr];
        }
        return points;
    }

    // instead of rounds, how about getting the median of the last n votes?
    function uploadPoints(address[] memory teachers, uint256[] memory points)
        public
    {
        address senderAddr = msg.sender;

        for (uint256 i = 0; i < teachers.length; i++) {
            if (PointsQueue.is_in(pointsQueue[teachers[i]], senderAddr)) {
                PointsQueue.remove(pointsQueue[teachers[i]], senderAddr);
            } else {
                if (pointsQueue[teachers[i]].size == n_points)
                    PointsQueue.pop(pointsQueue[teachers[i]]);
            }
            PointsQueue.insert(pointsQueue[teachers[i]], senderAddr, points[i]);
        }

        if (!recentUpload.isIn[senderAddr]) {
            if (Queue.isFull(recentUpload)) Queue.pop(recentUpload);
            Queue.insert(recentUpload, senderAddr);
        }
    }
}
