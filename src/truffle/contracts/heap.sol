// SPDX-License-Identifier: MIT
pragma solidity >=0.5.1 <0.9.0;

struct Heap {
    uint256[] data;
    uint256 size;
}

library MinHeap {
    function insert(Heap storage _heap, uint256 _value) external {
        _heap.size++;
        uint256 _index;
        for (
            _index = _heap.size - 1;
            _index > 0 && _value < _heap.data[_index / 2];
            _index /= 2
        ) {
            _heap.data[_index] = _heap.data[_index / 2];
        }
        _heap.data[_index] = _value;
    }

    function top(Heap storage _heap) external view returns (uint256) {
        return _heap.data[0];
    }

    function pop(Heap storage _heap) external {
        uint256 last = _heap.data[_heap.size - 1];
        uint256 index;
        for (index = 0; 2 * index < _heap.size; ) {
            uint256 nextIndex = 2 * index;
            if (
                2 * index + 1 < _heap.size &&
                _heap.data[2 * index + 1] < _heap.data[2 * index]
            ) nextIndex = 2 * index + 1;
            if (_heap.data[nextIndex] < last)
                _heap.data[index] = _heap.data[nextIndex];
            else break;
            index = nextIndex;
        }
        _heap.data[index] = last;
        _heap.size--;
    }
}

library MaxHeap {
    function insert(Heap storage _heap, uint256 _value) external {
        _heap.size++;
        uint256 _index;
        for (
            _index = _heap.size - 1;
            _index > 0 && _value > _heap.data[_index / 2];
            _index /= 2
        ) {
            _heap.data[_index] = _heap.data[_index / 2];
        }
        _heap.data[_index] = _value;
    }

    function top(Heap storage _heap) external view returns (uint256) {
        return _heap.data[0];
    }

    function pop(Heap storage _heap) external {
        uint256 last = _heap.data[_heap.size - 1];
        uint256 index;
        for (index = 0; 2 * index < _heap.size; ) {
            uint256 nextIndex = 2 * index;
            if (
                2 * index + 1 < _heap.size &&
                _heap.data[2 * index + 1] > _heap.data[2 * index]
            ) nextIndex = 2 * index + 1;
            if (_heap.data[nextIndex] > last)
                _heap.data[index] = _heap.data[nextIndex];
            else break;
            index = nextIndex;
        }
        _heap.data[index] = last;
        _heap.size--;
    }
}
