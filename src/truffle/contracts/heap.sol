// SPDX-License-Identifier: MIT
pragma solidity >=0.5.1 <0.9.0;

struct Heap {
    uint256[] data;
    uint256 size;
}

library MinHeap {
    function insert(Heap storage _heap, uint256 _value) internal {
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

    function top(Heap storage _heap) internal view returns (uint256) {
        return _heap.data[0];
    }

    function pop(Heap storage _heap) internal returns (uint256){
        uint256 result = _heap.data[0];
        
        // move last element to head
        uint256 last = _heap.data[_heap.size-1];
        _heap.data[0] = last;
        _heap.size--;
        
        // move head until both childs are smaller
        uint256 index = 0;
        uint256 lChild = 1;
        uint256 rChild = 2;
        uint256 minChild;
        while ( rChild < _heap.size) {
            if (_heap.data[lChild] < _heap.data[rChild]) {
                minChild = lChild;
            } else {
                minChild = rChild;
            }
            
            if (last > _heap.data[minChild]){
                _heap.data[index] = _heap.data[minChild];
                index = minChild;
                lChild = 2*index + 1;
                rChild = 2*index + 2;
            } else{
                break;
            }
        }
        
        // if has only left child
        if (lChild < _heap.size) {
            if (last > _heap.data[lChild]){
                _heap.data[index] = _heap.data[lChild];
                index = lChild;
            }
        }
        
        _heap.data[index] = last;
        return result;
    }
}

library MaxHeap {
    function insert(Heap storage _heap, uint256 _value) internal {
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

    function top(Heap storage _heap) internal view returns (uint256) {
        return _heap.data[0];
    }

    function pop(Heap storage _heap) internal returns (uint256) {
        uint256 result = _heap.data[0];
        
        // move last element to head
        uint256 last = _heap.data[_heap.size-1];
        _heap.data[0] = last;
        _heap.size--;
        
        // move head until both childs are smaller
        uint256 index = 0;
        uint256 lChild = 1;
        uint256 rChild = 2;
        uint256 maxChild;
        while ( rChild < _heap.size) {
            if (_heap.data[lChild] > _heap.data[rChild]) {
                maxChild = lChild;
            } else {
                maxChild = rChild;
            }
            
            if (last < _heap.data[maxChild]){
                _heap.data[index] = _heap.data[maxChild];
                index = maxChild;
                lChild = 2*index + 1;
                rChild = 2*index + 2;
            } else{
                break;
            }
        }
        
        // if has only left child
        if (lChild < _heap.size) {
            if (last < _heap.data[lChild]){
                _heap.data[index] = _heap.data[lChild];
                index = lChild;
            }
        }
        
        _heap.data[index] = last;
        return result;
    }
}
