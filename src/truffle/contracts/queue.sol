// SPDX-License-Identifier: MIT
pragma solidity >=0.5.1 <0.9.0;

struct Q {
    mapping(address => address) next;
    mapping(address => address) before;
    uint256 size;
}

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
        q.size--;
    }

    function pop(Q storage q) internal returns (address){
        address popResult = q.next[address(0)];
        q.next[address(0)] = q.next[q.next[address(0)]];
        q.before[q.next[address(0)]] = address(0);
        q.size--;
        return popResult;
    }
}
