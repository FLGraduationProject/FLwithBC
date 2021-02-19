// SPDX-License-Identifier: MIT
pragma solidity >=0.5.1 <0.9.0;

import "./heap.sol";
import "./median_heap.sol";

contract Rank {
    // if not use rounds, we have to save the sender
    // or erase the previous points uploaded
    // mapping(address => uint256[]) uploadPoints;
    using MinHeap for uint256;
    using MaxHeap for uint256;
    using MedianHeap for uint256;

    uint256 n_clients;
    uint256 n_clientsVoted;

    MedHeap[] points1;
    
    uint256[] medianPoints1;

    constructor(uint256 num_clients) public {
        n_clients = num_clients;
        n_clientsVoted = 0;

        for (uint256 i = 0; i < n_clients; i++) {
            points1.push(
                MedHeap(
                    Heap(new uint256[](n_clients), 0),
                    Heap(new uint256[](n_clients), 0)
                )
            );
        }

        medianPoints1 = new uint256[](n_clients);
    }

    function reset() public {
        n_clientsVoted = 0;
        for (uint256 i = 0; i < n_clients; i++) {
            MedianHeap.reset(points1[i]);
        }
    }

    // instead of rounds, how about getting the median of the last n votes?
    function upload(uint256[] memory _points1) public {
        require(n_clientsVoted < n_clients, "n_clientsVoted is over n_clients");
        n_clientsVoted++;

        for (uint256 i = 0; i < n_clients; i++) {
            if (_points1[i] != 0) {
                MedianHeap.insert(points1[i], _points1[i]);
                medianPoints1[i] = MedianHeap.get_median(points1[i]);
            }
        }

        if (n_clientsVoted == n_clients) {
            reset();
        }
    }

    function seeMedianPoints1() public view returns (uint256[] memory) {
        return medianPoints1;
    }
}