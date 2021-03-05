// SPDX-License-Identifier: MIT
pragma solidity >=0.5.1 <0.9.0;

struct Heap {
    address[] data;
    uint256 size;
    mapping(address=>uint256) locOnHeap;
}

library MinHeap {
    function insert(
        Heap storage minHeap,
        address insertAddr,
        mapping(address=>uint256) storage values
    ) internal {
        uint256 value = values[insertAddr];
        minHeap.size++;
        uint256 index = minHeap.size - 1;
        uint256 parent = (index-1)/2;
        while (index > 0){
            if (value < values[minHeap.data[parent]]) {
                minHeap.data[index] = minHeap.data[parent];
                minHeap.locOnHeap[minHeap.data[parent]] = index;
                index = parent;
                parent = (index-1)/2;
            } else {
                break;
            }
        }
        minHeap.data[index] = insertAddr;
        minHeap.locOnHeap[insertAddr] = index;
    }

    function top(Heap storage minHeap) internal view returns (address) {
        return minHeap.data[0];
    }

    function remove(
        Heap storage minHeap,
        address removeAddr,
        mapping(address=>uint256) storage values
    ) internal {
        // get last element and replace with removeAddr
        address lastAddr = minHeap.data[minHeap.size - 1];
        uint256 lastVal = values[lastAddr];
        minHeap.size--;

        // move from replace place until both childs are smaller
        uint256 index = minHeap.locOnHeap[removeAddr];
        uint256 lChild = 2 * index + 1;
        uint256 rChild = 2 * index + 2;
        uint256 minChild;
        while (rChild < minHeap.size) {
            if (values[minHeap.data[lChild]] < values[minHeap.data[rChild]]) {
                minChild = lChild;
            } else {
                minChild = rChild;
            }

            if (lastVal > values[minHeap.data[minChild]]) {
                minHeap.data[index] = minHeap.data[minChild];
                minHeap.locOnHeap[minHeap.data[minChild]] = index;
                index = minChild;
                lChild = 2 * index + 1;
                rChild = 2 * index + 2;
            } else {
                break;
            }
        }

        // if has only left child
        if (lChild < minHeap.size) {
            if (lastVal > values[minHeap.data[lChild]]) {
                minHeap.data[index] = minHeap.data[lChild];
                minHeap.locOnHeap[minHeap.data[lChild]] = index;
                index = lChild;
            }
        }

        // set last element index in place
        minHeap.data[index] = lastAddr;
        minHeap.locOnHeap[lastAddr] = index;
        
        // remove removeAddr
        delete minHeap.locOnHeap[removeAddr];
    }
}

library MaxHeap {
    function insert(
        Heap storage maxHeap,
        address insertAddr,
        mapping(address=>uint256) storage values
    ) internal {
        uint256 value = values[insertAddr];
        maxHeap.size++;
        uint256 index = maxHeap.size - 1;
        uint256 parent = (index-1)/2;
        while (index > 0){
            if (value > values[maxHeap.data[parent]]) {
                maxHeap.data[index] = maxHeap.data[parent];
                maxHeap.locOnHeap[maxHeap.data[parent]] = index;
                index = parent;
                parent = (index-1)/2;
            } else {
                break;
            }
        }
        maxHeap.data[index] = insertAddr;
        maxHeap.locOnHeap[insertAddr] = index;
    }

    function top(Heap storage maxHeap) internal view returns (address) {
        return maxHeap.data[0];
    }

    function remove(
        Heap storage maxHeap,
        address removeAddr,
        mapping(address=>uint256) storage values
    ) internal {
        // get last element and replace with removeAddr
        address lastAddr = maxHeap.data[maxHeap.size - 1];
        uint256 lastVal = values[lastAddr];
        maxHeap.size--;

        // move from replace place until both childs are smaller
        uint256 index = maxHeap.locOnHeap[removeAddr];
        uint256 lChild = 2 * index + 1;
        uint256 rChild = 2 * index + 2;
        uint256 maxChild;
        while (rChild < maxHeap.size) {
            if (values[maxHeap.data[lChild]] > values[maxHeap.data[rChild]]) {
                maxChild = lChild;
            } else {
                maxChild = rChild;
            }

            if (lastVal < values[maxHeap.data[maxChild]]) {
                maxHeap.data[index] = maxHeap.data[maxChild];
                maxHeap.locOnHeap[maxHeap.data[maxChild]] = index;
                index = maxChild;
                lChild = 2 * index + 1;
                rChild = 2 * index + 2;
            } else {
                break;
            }
        }

        // if has only left child
        if (lChild < maxHeap.size) {
            if (lastVal < values[maxHeap.data[lChild]]) {
                maxHeap.data[index] = maxHeap.data[lChild];
                maxHeap.locOnHeap[maxHeap.data[lChild]] = index;
                index = lChild;
            }
        }

        // set last element index in place
        maxHeap.data[index] = lastAddr;
        maxHeap.locOnHeap[lastAddr] = index;
        
        // remove removeAddr
        delete maxHeap.locOnHeap[removeAddr];
    }
}