// SPDX-License-Identifier: MIT
pragma solidity >=0.5.1 <0.9.0;

struct PointsQ {
    mapping(address => address) next;
    mapping(address => address) before;
    mapping(address => uint256) value;
    uint256 size;
}

// first in first out queue
// 0 -> oldest -> -> -> newest -> 0
library PointsQueue {
    function insert(PointsQ storage q, address insertAddr, uint256 value) internal {
        q.value[insertAddr] = value;
        q.next[q.before[address(0)]] = insertAddr;
        q.next[insertAddr] = address(0);
        q.before[insertAddr] = q.before[address(0)];
        q.before[address(0)] = insertAddr;
        q.size++;
    }

    function pop(PointsQ storage q) internal {
        address oldest = q.next[address(0)];
        q.next[address(0)] = q.next[oldest];
        q.before[q.next[address(0)]] = address(0);
        delete q.next[oldest];
        delete q.before[oldest];
        delete q.value[oldest];
        q.size--;
    }

    function remove(PointsQ storage q, address removeAddr) internal {
        q.next[q.before[removeAddr]] = q.next[removeAddr];
        q.before[q.next[removeAddr]] = q.before[removeAddr];
        delete q.next[removeAddr];
        delete q.before[removeAddr];
        delete q.value[removeAddr];
        q.size--;
    }

    function is_in(PointsQ storage q, address checkAddr) internal view returns (bool) {
        if (q.next[checkAddr] != address(0)) {
            return true;
        } else if (q.before[checkAddr] != address(0)) {
            return true;
        }

        return false;
    }
}