// SPDX-License-Identifier: MIT
pragma solidity >=0.5.1 <0.9.0;

import "./heap.sol";
import "./median_heap.sol";

contract PointsBoard {
    // if not use rounds, we have to save the sender
    // or erase the previous points uploaded
    using MinHeap for uint256;
    using MaxHeap for uint256;
    using MedianHeap for uint256;

    uint256 maxHeapSize;

    mapping(address => MedHeap) pointsHeap;
    mapping(address => uint256) medianPoints;
    mapping(address => bool) heapMade;
    
    constructor(uint256 maxSize) public {
        maxHeapSize = maxSize;
    }

    // instead of rounds, how about getting the median of the last n votes?
    function upload(address[] memory teachers, uint256[] memory points) public {
        address senderAddr = msg.sender;
        
        for (uint256 i = 0; i < teachers.length; i++) {
            if (!heapMade[teachers[i]]){
                heapMade[teachers[i]] = true;
                pointsHeap[teachers[i]].maxHeap.data = new address[](maxHeapSize);
                pointsHeap[teachers[i]].minHeap.data = new address[](maxHeapSize);
                pointsHeap[teachers[i]].maxHeapSize = maxHeapSize;
            }
            MedianHeap.insert(pointsHeap[teachers[i]], senderAddr, points[i]);
        }
    }

    function seeMedianPoint(address teacherAddr) public view returns (uint256) {
        if (heapMade[teacherAddr]) {
            return MedianHeap.get_median(pointsHeap[teacherAddr]);
        } else {
            return 0;
        }
    }

    function seeMaxHeap(address teacherAddr) public view returns (address[] memory) {
        if (heapMade[teacherAddr]) {
            return pointsHeap[teacherAddr].maxHeap.data;
        } else {
            return new address[](1);
        }
    }
    
    function seeMinHeap(address teacherAddr) public view returns (address[] memory) {
        if (heapMade[teacherAddr]) {
            return pointsHeap[teacherAddr].minHeap.data;
        } else {
            return new address[](1);
        }
    }
    
    function seeQueue(address teacherAddr) public view returns (address[] memory) {
        if (heapMade[teacherAddr]) {
            address[] memory queue = new address[](maxHeapSize);
            address addr = pointsHeap[teacherAddr].addrQueue.next[address(0)];
            uint256 index = 0;
            while (addr != address(0)){
                queue[index] = addr;
                addr = pointsHeap[teacherAddr].addrQueue.next[addr];
                index ++;
            }
            return queue;
        } else {
            return new address[](1);
        }
    }
}