// SPDX-License-Identifier: MIT
pragma solidity >=0.5.1 <0.9.0;

import "./heap.sol";
import "./queue.sol";

struct MedHeap {
    Heap maxHeap;
    Heap minHeap;
    // ex) 0: not on Heap, 1: maxHeap, 2: minHeap
    mapping(address => uint256) whichHeap;
    mapping(address => uint256) values;
    Q addrQueue;
    uint256 maxHeapSize;
}

library MedianHeap {
    using MinHeap for uint256;
    using MaxHeap for uint256;
    using Queue for uint256;

    function remove(MedHeap storage medHeap, address clientAddr) internal {
        uint256 heapLoc = medHeap.whichHeap[clientAddr];
        if (heapLoc == 1) {
            MaxHeap.remove(
                medHeap.maxHeap,
                clientAddr,
                medHeap.values
            );
        } else if (heapLoc == 2) {
            MinHeap.remove(
                medHeap.minHeap,
                clientAddr,
                medHeap.values
            );
        }
        
        if (medHeap.maxHeap.size > medHeap.minHeap.size + 1) {
            address topAddr = MaxHeap.top(medHeap.maxHeap);
            MaxHeap.remove(
                medHeap.maxHeap,
                topAddr,
                medHeap.values
            );
            MinHeap.insert(
                medHeap.minHeap,
                topAddr,
                medHeap.values
            );
            medHeap.whichHeap[topAddr] = 2;
        } else if (medHeap.maxHeap.size < medHeap.minHeap.size) {
            address topAddr = MinHeap.top(medHeap.minHeap);
            MinHeap.remove(
                medHeap.minHeap,
                topAddr,
                medHeap.values
            );
            MaxHeap.insert(
                medHeap.maxHeap,
                topAddr,
                medHeap.values
            );
            medHeap.whichHeap[topAddr] = 1;
        }
        delete medHeap.values[clientAddr];
        delete medHeap.whichHeap[clientAddr];
    }

    function insert(MedHeap storage medHeap, address clientAddr, uint256 value) internal {
        // if already on heap remove first
        if (medHeap.whichHeap[clientAddr] != 0) {
            Queue.remove(medHeap.addrQueue, clientAddr);
            remove(medHeap, clientAddr);
        }
        // if heap is full remove oldest
        else if (medHeap.addrQueue.size == medHeap.maxHeapSize) {
            address removedAddr = Queue.oldest(medHeap.addrQueue);
            Queue.remove(medHeap.addrQueue, removedAddr);
            remove(medHeap, removedAddr);
        }
        
        Queue.insert(medHeap.addrQueue, clientAddr);
        medHeap.values[clientAddr] = value;
        if (medHeap.maxHeap.size == 0) {
            MaxHeap.insert(
                medHeap.maxHeap,
                clientAddr,
                medHeap.values
            );
            medHeap.whichHeap[clientAddr] = 1;
        } else {
            if (medHeap.values[MaxHeap.top(medHeap.maxHeap)] > value) {
                MaxHeap.insert(
                    medHeap.maxHeap,
                    clientAddr,
                    medHeap.values
                );
                medHeap.whichHeap[clientAddr] = 1;
            } else {
                MinHeap.insert(
                    medHeap.minHeap,
                    clientAddr,
                    medHeap.values
                );
                medHeap.whichHeap[clientAddr] = 2;
            }

            if (medHeap.maxHeap.size > medHeap.minHeap.size + 1) {
                address topAddr = MaxHeap.top(medHeap.maxHeap);
                MaxHeap.remove(
                    medHeap.maxHeap,
                    topAddr,
                    medHeap.values
                );
                MinHeap.insert(
                    medHeap.minHeap,
                    topAddr,
                    medHeap.values
                );
                medHeap.whichHeap[topAddr] = 2;
            } else if (medHeap.maxHeap.size < medHeap.minHeap.size) {
                address topAddr = MinHeap.top(medHeap.minHeap);
                MinHeap.remove(
                    medHeap.minHeap,
                    topAddr,
                    medHeap.values
                );
                MaxHeap.insert(
                    medHeap.maxHeap,
                    topAddr,
                    medHeap.values
                );
                medHeap.whichHeap[topAddr] = 1;
            }
        }
    }

    function get_median(MedHeap storage heap) internal view returns (uint256) {
        return heap.values[MaxHeap.top(heap.maxHeap)];
    }
}