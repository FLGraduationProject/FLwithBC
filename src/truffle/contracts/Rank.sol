// SPDX-License-Identifier: MIT
pragma solidity >=0.4.22 <0.9.0;

contract Rank {
    // mapping(uint16 => uint16) public IdtoIndex;
    // mapping(uint16 => uint16) public IndextoId;

    uint8 n_clients;
    uint8 n_clientsVoted;

    uint16[][] uploadedPoints;
    uint16[] evalPoints;

    uint16[] justAvgPoints;
    uint16[] weightAvgPoints;
    uint16[] n_points;
    uint16[] n_teachers;
    uint16[] rankArr;

    constructor(uint8 num_clients) public {
        n_clients = num_clients;
        n_clientsVoted = 0;
        rankArr = new uint16[](n_clients);
        uploadedPoints = new uint16[][](n_clients);
        evalPoints = new uint16[](n_clients);
        justAvgPoints = new uint16[](n_clients);
        weightAvgPoints = new uint16[](n_clients);
        n_points = new uint16[](n_clients);
        n_teachers = new uint16[](n_clients);
    }

    function reset() public {
        n_clientsVoted = 0;
        for (uint16 i = 0; i < n_clients; i++) {
            evalPoints[i] = 0;
            justAvgPoints[i] = 0;
            weightAvgPoints[i] = 0;
            n_points[i] = 0;
            n_teachers[i] = 0;
        }
    }

    function dist(uint16 x, uint16 y) private pure returns (uint16) {
        if (x >= y) {
            return x - y;
        } else {
            return y - x;
        }
    }

    function ranking() public {
        // 1. 최초 단순평균 구하기
        uint8 i = 0;
        uint8 j = 0;

        for (i = 0; i < n_clients; i++) {
            justAvgPoints[i] /= n_points[i]; //소수점 !!!!
        }

        // 2. weight 구하기
        // weight = (evalPointSum-evalPoint) / evalPointSum
        // since solidity has no float type and we need only ranking, instead of using weight, use evalPointSum-evalPoint
        // uint16 evalPointSum = 0;
        for (i = 0; i < n_clients; i++) {
            for (j = 0; j < n_clients; j++) {
                if (uploadedPoints[i][j] != 0) {
                    evalPoints[i] += dist(
                        uploadedPoints[i][j],
                        justAvgPoints[j]
                    );
                }
            }
            evalPoints[i] /= n_teachers[i];
            // evalPointSum += evalPoints[i];
        }

        // 3. 가중평균에 따른 ranking 구하기
        for (i = 0; i < n_clients; i++) {
            for (j = 0; j < n_clients; j++) {
                if (uploadedPoints[i][j] != 0) {
                    weightAvgPoints[j] += (uploadedPoints[i][j] /
                        evalPoints[j]);
                }
            }
        }

        for (i = 0; i < n_clients; i++) {
            rankArr[i] = 1;
            for (j = 0; j < n_clients; j++) {
                if (weightAvgPoints[j] < weightAvgPoints[i]) {
                    rankArr[i]++;
                }
            }
        }
    }

    function upload(uint16[] memory points) public {
        require(n_clientsVoted < n_clients, "n_clientsVoted is over n_clients");
        uploadedPoints[n_clientsVoted] = points;

        for (uint8 i = 0; i < n_clients; i++) {
            if (points[i] != 0) {
                n_teachers[n_clientsVoted]++;
                n_points[i]++;
                justAvgPoints[i] += points[i];
            }
        }
        n_clientsVoted++;
        if (n_clientsVoted == n_clients) {
            ranking();
            reset();
        }
    }

    function see_rank() public view returns (uint16[] memory) {
        return rankArr;
    }
}
