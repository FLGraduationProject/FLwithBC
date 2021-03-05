// SPDX-License-Identifier: MIT
pragma solidity >=0.5.1 <0.9.0;

struct Q {
    mapping(address => address) next;
    mapping(address => address) before;
    uint256 size;
}

// first in first out queue
// 0 -> oldest -> -> -> newest -> 0
library Queue {
    function insert(Q storage q, address clientAddr) internal {
        q.next[q.before[address(0)]] = clientAddr;
        q.next[clientAddr] = address(0);
        q.before[clientAddr] = q.before[address(0)];
        q.before[address(0)] = clientAddr;
        q.size++;
    }

    function remove(Q storage q, address clientAddr) internal {
        q.next[q.before[clientAddr]] = q.next[clientAddr];
        q.before[q.next[clientAddr]] = q.before[clientAddr];
        delete q.next[clientAddr];
        delete q.before[clientAddr];
        q.size--;
    }

    function oldest(Q storage q) internal view returns (address){
        return q.next[address(0)];
    }
}