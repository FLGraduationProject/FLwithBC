// SPDX-License-Identifier: MIT
pragma solidity >=0.4.22 <0.9.0;

contract Rank {
	// mapping(uint8 => uint8) public IdtoIndex;
	// mapping(uint8 => uint8) public IndextoId;

	uint8 n_clientsVoted ;
	uint[] avgdistarr ; 
	uint[] votearr ; 
	uint[] rankarr ; 

	uint8 numclient ; 
	
	function setting(uint8 numclients) public{
		n_clientsVoted = 0 ;
		numclient = numclients ; 
		avgdistarr = new uint[](numclient) ; 
		votearr = new uint[](numclient) ; 
		rankarr = new uint[](numclient) ; 
	}

	function ranking() public {
		for(uint i = 0 ; i<votearr.length ; i++){
			avgdistarr[i]/=votearr[i] ; 
		}

		for(uint i = 0 ; i<avgdistarr.length ; i++){
			uint8 rank = 1 ;
			for(uint j = 0 ; j<avgdistarr.length ; j++){
				if(avgdistarr[j]!=0){
					if(avgdistarr[j]<avgdistarr[i]){
						rank ++ ; 
					}
				} 
			}
			rankarr[i] = rank ; 
		}
		setting(numclient) ;
	}

	function upload(uint16[] memory avgdist) public {
		n_clientsVoted ++ ;
		for (uint i = 0 ; i < avgdist.length ; i++){
			if(avgdist[i]!=0){
				votearr[i]++ ;
			}
			avgdistarr[i] += avgdist[i] ; 
		}
		if (n_clientsVoted == numclient){
			ranking();
		}
	}

	function see_rank() public view returns(uint[] memory){
		return rankarr ; 
	}


    //개별 ranking 산정 version 
    // function ranking(uint8 clientIndex) returns(uint8) public{
    //     uint8 rank = 1 ; 
    //     for(uint i = 0 ; i<avgdistarr.length ; i++){
    //         if(avgdistarr[i]!=0){
    //             if(avgdistarr[i]<avgdistarr[clientIndex]){
    //                 rank ++ ; 
    //             }
    //         }
    //     }
    //     return rank ; 
    // }

}