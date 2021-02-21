// SPDX-License-Identifier: MIT
pragma solidity >=0.5.1 <0.9.0;

struct Heap {
    address[] data;
    uint256 size;
}

library MinHeap {
    function insert(
        Heap storage idxHeap,
        address clientAddr,
        mapping(address=>uint256) storage values,
        mapping(address=>uint256) storage locOnHeap
    ) internal {
        uint256 value = values[clientAddr];
        idxHeap.size++;
        uint256 index;
        for (
            index = idxHeap.size - 1;
            index > 0 && value < values[idxHeap.data[index / 2]];
            index /= 2
        ) {
            idxHeap.data[index] = idxHeap.data[index / 2];
            locOnHeap[idxHeap.data[index / 2]] = 2 * index + 1;
        }
        idxHeap.data[index] = clientAddr;
        locOnHeap[clientAddr] = 2 * index + 1;
    }

    function top(Heap storage idxHeap) internal view returns (address) {
        return idxHeap.data[0];
    }

    function remove(
        Heap storage idxHeap,
        uint256 removeIdx,
        mapping(address=>uint256) storage values,
        mapping(address=>uint256) storage locOnHeap
    ) internal {
        // move last element to head
        address lastAddr = idxHeap.data[idxHeap.size - 1];
        uint256 lastVal = values[lastAddr];
        idxHeap.size--;

        // move head until both childs are smaller
        uint256 index = removeIdx;
        uint256 lChild = 2 * removeIdx + 1;
        uint256 rChild = 2 * removeIdx + 2;
        uint256 minChild;
        while (rChild < idxHeap.size) {
            if (values[idxHeap.data[lChild]] < values[idxHeap.data[rChild]]) {
                minChild = lChild;
            } else {
                minChild = rChild;
            }

            if (lastVal > values[idxHeap.data[minChild]]) {
                idxHeap.data[index] = idxHeap.data[minChild];
                locOnHeap[idxHeap.data[minChild]] = 2 * index + 1;
                index = minChild;
                lChild = 2 * index + 1;
                rChild = 2 * index + 2;
            } else {
                break;
            }
        }

        // if has only left child
        if (lChild < idxHeap.size) {
            if (lastVal > values[idxHeap.data[lChild]]) {
                idxHeap.data[index] = idxHeap.data[lChild];
                locOnHeap[idxHeap.data[lChild]] = 2 * index + 1;
                index = lChild;
            }
        }

        // set last element index in place
        idxHeap.data[index] = lastAddr;
        locOnHeap[lastAddr] = 2 * index + 1;
    }
}

library MaxHeap {
    function insert(
        Heap storage idxHeap,
        address clientAddr,
        mapping(address=>uint256) storage values,
        mapping(address=>uint256) storage locOnHeap
    ) internal {
        uint256 value = values[clientAddr];
        idxHeap.size++;
        uint256 index;
        for (
            index = idxHeap.size - 1;
            index > 0 && value > values[idxHeap.data[index / 2]];
            index /= 2
        ) {
            idxHeap.data[index] = idxHeap.data[index / 2];
            locOnHeap[idxHeap.data[index / 2]] = 2 * index + 2;
        }
        idxHeap.data[index] = clientAddr;
        locOnHeap[clientAddr] = 2 * index + 2;
    }

    function top(Heap storage idxHeap) internal view returns (address) {
        return idxHeap.data[0];
    }

    function remove(
        Heap storage idxHeap,
        uint256 removeIdx,
        mapping(address=>uint256) storage values,
        mapping(address=>uint256) storage locOnHeap
    ) internal {
        // move last element to remove index
        address lastAddr = idxHeap.data[idxHeap.size - 1];
        uint256 lastVal = values[lastAddr];
        idxHeap.size--;

        // move from remove index until both childs are smaller
        uint256 index = removeIdx;
        uint256 lChild = 2 * removeIdx + 1;
        uint256 rChild = 2 * removeIdx + 2;
        uint256 maxChild;
        while (rChild < idxHeap.size) {
            if (values[idxHeap.data[lChild]] > values[idxHeap.data[rChild]]) {
                maxChild = lChild;
            } else {
                maxChild = rChild;
            }

            if (lastVal < values[idxHeap.data[maxChild]]) {
                idxHeap.data[index] = idxHeap.data[maxChild];
                locOnHeap[idxHeap.data[maxChild]] = 2 * index + 2;
                index = maxChild;
                lChild = 2 * index + 1;
                rChild = 2 * index + 2;
            } else {
                break;
            }
        }

        // if has only left child
        if (lChild < idxHeap.size) {
            if (lastVal < values[idxHeap.data[lChild]]) {
                idxHeap.data[index] = idxHeap.data[lChild];
                locOnHeap[idxHeap.data[lChild]] = 2 * index + 2;
                index = lChild;
            }
        }

        // set last element index in place
        idxHeap.data[index] = lastAddr;
        locOnHeap[lastAddr] = 2 * index + 2;
    }
}