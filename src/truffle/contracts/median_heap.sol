// SPDX-License-Identifier: MIT
pragma solidity >=0.5.1 <0.9.0;

import "./heap.sol";
import "./queue.sol";

struct MedHeap {
    Heap maxHeap;
    Heap minHeap;
    // if 2n+1 is minHeap, if 2n+2 is maxHeap
    // ex) 0: no place on Heap, 1: minHeap[0], 2: maxHeap[0], 3: minHeap[1]
    mapping(address => uint256) locOnHeap;
    mapping(address => uint256) values;
    Q addrQueue;
    uint256 maxHeapSize;
}

library MedianHeap {
    using MinHeap for uint256;
    using MaxHeap for uint256;
    using Queue for uint256;

    function remove(MedHeap storage idxHeap, address clientAddr) internal {
        uint256 rmIdx = idxHeap.locOnHeap[clientAddr];
        idxHeap.locOnHeap[clientAddr] = 0;
        
        if (rmIdx % 2 == 1) {
            uint256 idx = rmIdx / 2;
            MinHeap.remove(
                idxHeap.minHeap,
                idx,
                idxHeap.values,
                idxHeap.locOnHeap
            );
        } else {
            uint256 idx = (rmIdx / 2) - 1;
            MaxHeap.remove(
                idxHeap.maxHeap,
                idx,
                idxHeap.values,
                idxHeap.locOnHeap
            );
        }
        
        if (idxHeap.maxHeap.size > idxHeap.minHeap.size + 1) {
            address topAddr = MaxHeap.top(idxHeap.maxHeap);
            MaxHeap.remove(
                idxHeap.maxHeap,
                0,
                idxHeap.values,
                idxHeap.locOnHeap
            );
            MinHeap.insert(
                idxHeap.minHeap,
                topAddr,
                idxHeap.values,
                idxHeap.locOnHeap
            );
        } else if (idxHeap.maxHeap.size < idxHeap.minHeap.size) {
            address topAddr = MinHeap.top(idxHeap.minHeap);
            MinHeap.remove(
                idxHeap.minHeap,
                0,
                idxHeap.values,
                idxHeap.locOnHeap
            );
            MaxHeap.insert(
                idxHeap.maxHeap,
                topAddr,
                idxHeap.values,
                idxHeap.locOnHeap
            );
        }
    }

    function insert(MedHeap storage idxHeap, address clientAddr) internal {
        if (idxHeap.locOnHeap[clientAddr] != 0) {
            Queue.remove(idxHeap.addrQueue, clientAddr);
            remove(idxHeap, clientAddr);
        }
        else if (idxHeap.addrQueue.size == idxHeap.maxHeapSize) {
            address removedAddr = Queue.pop(idxHeap.addrQueue);
            remove(idxHeap, removedAddr);
        }
        
        Queue.insert(idxHeap.addrQueue, clientAddr);
        uint256 value = idxHeap.values[clientAddr];
        if (idxHeap.maxHeap.size == 0) {
            MaxHeap.insert(
                idxHeap.maxHeap,
                clientAddr,
                idxHeap.values,
                idxHeap.locOnHeap
            );
        } else {
            if (idxHeap.values[MaxHeap.top(idxHeap.maxHeap)] > value) {
                MaxHeap.insert(
                    idxHeap.maxHeap,
                    clientAddr,
                    idxHeap.values,
                    idxHeap.locOnHeap
                );
            } else {
                MinHeap.insert(
                    idxHeap.minHeap,
                    clientAddr,
                    idxHeap.values,
                    idxHeap.locOnHeap
                );
            }

            if (idxHeap.maxHeap.size > idxHeap.minHeap.size + 1) {
                address topAddr = MaxHeap.top(idxHeap.maxHeap);
                MaxHeap.remove(
                    idxHeap.maxHeap,
                    0,
                    idxHeap.values,
                    idxHeap.locOnHeap
                );
                MinHeap.insert(
                    idxHeap.minHeap,
                    topAddr,
                    idxHeap.values,
                    idxHeap.locOnHeap
                );
            } else if (idxHeap.maxHeap.size < idxHeap.minHeap.size) {
                address topAddr = MinHeap.top(idxHeap.minHeap);
                MinHeap.remove(
                    idxHeap.minHeap,
                    0,
                    idxHeap.values,
                    idxHeap.locOnHeap
                );
                MaxHeap.insert(
                    idxHeap.maxHeap,
                    topAddr,
                    idxHeap.values,
                    idxHeap.locOnHeap
                );
            }
        }
    }

    function get_median(MedHeap storage heap) internal view returns (uint256) {
        return heap.values[MaxHeap.top(heap.maxHeap)];
    }
}

