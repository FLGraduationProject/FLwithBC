// SPDX-License-Identifier: MIT
pragma solidity >=0.5.1 <0.9.0;

struct Q {
    mapping(address => bool) isIn;
    address[] list;
    uint256 start;
    uint256 end;
    uint256 maxSize;
}

// first in first out queue
// 0 -> oldest -> -> -> newest -> 0
library Queue {
    function initialize(Q storage q, uint256 qSize) internal {
        q.maxSize = qSize;
        q.end = 1;
        for (uint256 i = 0; i < qSize; i++) {
            q.list.push(address(0));
        }
    }

    function insert(Q storage q, address insertAddr) internal {
        q.end = (q.end + 1) % q.maxSize;
        q.list[q.end] = insertAddr;
        q.isIn[insertAddr] = true;
    }

    function pop(Q storage q) internal {
        address oldest = q.list[q.start];
        q.isIn[oldest] = false;
        q.start = (q.start + 1) % q.maxSize;
    }

    function byIndex(Q storage q, uint256 index) internal view returns (address){
        return q.list[(q.start + index) % q.maxSize];
    }

    function isFull(Q storage q) internal view returns (bool) {
        return q.start == q.end;
    }
}
