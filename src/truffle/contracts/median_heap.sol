// SPDX-License-Identifier: MIT
pragma solidity >=0.5.1 <0.9.0;

import "./heap.sol";

struct MedHeap {
    Heap maxHeap;
    Heap minHeap;
}

library MedianHeap {
    using MinHeap for uint256;
    using MaxHeap for uint256;

    function reset(MedHeap storage heap) external {
        heap.maxHeap.size = 0;
        heap.minHeap.size = 0;
    }

    function insert(MedHeap storage heap, uint256 value) external {
        if (heap.maxHeap.size == 0) {
            MaxHeap.insert(heap.maxHeap, value);
        } else {
            if (MaxHeap.top(heap.maxHeap) > value) {
                MaxHeap.insert(heap.maxHeap, value);
            } else {
                MinHeap.insert(heap.minHeap, value);
            }

            if (heap.maxHeap.size > heap.minHeap.size + 1) {
                uint256 temp = MaxHeap.top(heap.maxHeap);
                MaxHeap.pop(heap.maxHeap);
                MinHeap.insert(heap.minHeap, temp);
            } else if (heap.maxHeap.size < heap.minHeap.size) {
                uint256 temp = MinHeap.top(heap.minHeap);
                MinHeap.pop(heap.minHeap);
                MaxHeap.insert(heap.maxHeap, temp);
            }
        }
    }

    function get_median(MedHeap storage heap) external view returns (uint256) {
        return MaxHeap.top(heap.maxHeap);
    }
}
