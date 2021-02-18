// SPDX-License-Identifier: MIT
pragma solidity >=0.5.1 <0.9.0;

import "./heap.sol";
import "./median_heap.sol";

contract Rank {
    // mapping(uint => uint) public IdtoIndex;
    // mapping(uint => uint) public IndextoId;
    using MinHeap for uint256;
    using MaxHeap for uint256;
    using MedianHeap for uint256;

    uint256 n_clients;
    uint256 n_clientsVoted;

    MedHeap[] points1;
    MedHeap[] points2;

    uint256[] medianPoints1;
    uint256[] medianPoints2;

    uint256[] points1Rank;
    uint256[] points2Rank;

    uint256[] rankArr1;
    uint256[] rankArr2;

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
            points2.push(
                MedHeap(
                    Heap(new uint256[](n_clients), 0),
                    Heap(new uint256[](n_clients), 0)
                )
            );
        }

        medianPoints1 = new uint256[](n_clients);
        medianPoints2 = new uint256[](n_clients);

        points1Rank = new uint256[](n_clients);
        points2Rank = new uint256[](n_clients);


        for (uint256 i = 0; i < n_clients; i++) {
            rankArr1.push(i);
        }
        rankArr2 = new uint256[](n_clients);
    }

    function reset() public {
        n_clientsVoted = 0;
        for (uint256 i = 0; i < n_clients; i++) {
            MedianHeap.reset(points1[i]);
            MedianHeap.reset(points2[i]);
        }
    }

    function ranking() private {
        for (uint256 i = 0; i < n_clients; i++) {
            medianPoints1[i] = MedianHeap.get_median(points1[i]);
            medianPoints2[i] = MedianHeap.get_median(points2[i]);
        }
        merge_sort(medianPoints1, 0, n_clients - 1);
        for (uint256 i = 0; i < n_clients; i++) {
            points1Rank[rankArr1[i]] = i + 1;
        }

        merge_sort(medianPoints2, 0, n_clients - 1);
        for (uint256 i = 0; i < n_clients; i++) {
            points2Rank[rankArr2[i]] = i + 1;
        }
    }

    function upload(uint256[] memory _points1, uint256[] memory _points2) public {
        require(n_clientsVoted < n_clients, "n_clientsVoted is over n_clients");
        n_clientsVoted++;

        for (uint256 i = 0; i < n_clients; i++) {
            if (_points1[i] != 0) {
                MedianHeap.insert(points1[i], _points1[i]);
                MedianHeap.insert(points2[i], _points2[i]);
            }
        }

        if (n_clientsVoted == n_clients) {
            ranking();
            reset();
        }
    }

    function seeRank1() public view returns (uint256[] memory) {
        return points1Rank;
    }

    function seeRank2() public view returns (uint256[] memory) {
        return points2Rank;
    }

    function merge(
        uint256[] memory list,
        uint256 left,
        uint256 mid,
        uint256 right
    ) private {
        uint256 i = left;
        uint256 j = mid + 1;
        uint256 k = left;

        while (i <= mid && j <= right) {
            if (list[rankArr1[i]] <= list[rankArr1[j]]) {
                rankArr2[k++] = rankArr1[i++];
            } else {
                rankArr2[k++] = rankArr1[j++];
            }
        }

        if (i > mid) {
            for (uint256 l = j; l <= right; l++) {
                rankArr2[k++] = rankArr1[l];
            }
        } else {
            for (uint256 l = i; l <= mid; l++) {
                rankArr2[k++] = rankArr1[l];
            }
        }

        for (uint256 l = left; l <= right; l++) {
            rankArr1[l] = rankArr2[l];
        }
    }

    // sorts in increasing order
    function merge_sort(
        uint256[] memory list,
        uint256 left,
        uint256 right
    ) private {
        if (left < right) {
            uint256 mid = (left + right) / 2;
            merge_sort(list, left, mid);
            merge_sort(list, mid + 1, right);
            merge(list, left, mid, right);
        }
    }
}
