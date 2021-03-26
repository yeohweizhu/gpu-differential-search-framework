
#include <iostream>

using namespace std;

unsigned char sbox_gift[16] ={0x1,0xa,0x4,0xc,0x6,0xf,0x3,0x9,0x2,0xd,0xb,0x7,0x5,0x0,0x8,0xe};

//Sample for generating diff_table
//NOTE That the prob is actually frequency. To obtain prob just divide it by the total frequency.

int main(){
	//Differential Table
	unsigned int diff_table_raw[16][16] ={0};
	for (int x1=0;x1<16;x1++){
		for (int x2=0;x2<16;x2++){
			int x_diff=  x1 ^ x2;
			int y_diff = sbox_gift[x1] ^ sbox_gift[x2];
			diff_table_raw[x_diff][y_diff] +=1;
		}
	}
	// Display
	for (int i=0;i<16;i++){
		for (int j=0;j<16;j++){
			std::cerr<< diff_table_raw[i][j] << ", ";
		}
		std::cerr<< std::endl;
	}
	std::cerr<< std::endl;


	//Sorted Diff
	unsigned int diff_table_sorted_dy[16][16] ={0};
	unsigned int diff_table_sorted_prob[16][16] ={0};
	for (int i=0;i<16;i++){ //Init 
		//Row repeat
		for (int j=0;j<16;j++){
			diff_table_sorted_dy[i][j]=0;
			diff_table_sorted_prob[i][j] = 0;
		}
	}
	for (int i=0;i<16;i++){ //Do
		//Row repeat
		int how_many = 0; 
		for (int j=0;j<16;j++){
			//Selection Sort
			int largest = diff_table_raw[i][j];
			int largest_index = j;
			for (int m=j+1;m<16;m++){
				if (diff_table_raw[i][m] > largest){
					largest = diff_table_raw[i][m];
					largest_index = m;
				}
			}

			if (largest ==0){
				break;
			}
			else{
				int temp = largest;
				int temp_index = largest_index;
				diff_table_raw[i][largest_index]= 0;

				diff_table_sorted_dy[i][how_many] = temp_index;
				diff_table_sorted_prob[i][how_many] = temp;
				how_many +=1;
			}
		}
	}

	//Reverse Diff
	unsigned int diff_table_sorted_dy_reversed[16][16] ={0};
	unsigned int diff_table_sorted_prob_reversed[16][16] ={0};
	for (int outer_row=1;outer_row<16;outer_row++){ //This is for column
		int number = 0;
		for (int i=0;i<8;i++){ //This is for column
			for (int row=0;row<16;row++){
				if (diff_table_sorted_dy[row][i] == outer_row){
					diff_table_sorted_dy_reversed[outer_row][number] = row;
					diff_table_sorted_prob_reversed[outer_row][number] = diff_table_sorted_prob[row][i]; 

					number+= 1 ;
				}
			}
		}
	}
	// Display
	std::cout << "\nReversed DY\n";
	for (int i=0;i<16;i++){
		for (int j=0;j<16;j++){
			std::cerr<< diff_table_sorted_dy_reversed[i][j] << ", ";
		}
		std::cerr<< std::endl;
	}
	std::cout << "\nReversed Prob\n";
	for (int i=0;i<16;i++){ 
		for (int j=0;j<16;j++){
			std::cerr<< diff_table_sorted_prob_reversed[i][j] << ", ";
		}
		std::cerr<< std::endl;
	}

	// Display
	std::cout << "\n DY\n";
	for (int i=0;i<16;i++){
		for (int j=0;j<16;j++){
			std::cerr<< diff_table_sorted_dy[i][j] << ", ";
		}
		std::cerr<< std::endl;
	}
	std::cout << "\n Prob\n";
	for (int i=0;i<16;i++){ 
		for (int j=0;j<16;j++){
			std::cerr<< diff_table_sorted_prob[i][j] << ", ";
		}
		std::cerr<< std::endl;
	}

	// for (int i=0;i<16;i++){ //Copy
	// 	for (int j=0;j<16;j++){
	// 		= diff_table_sorted_dy[i][j] ;
	// }
	// 	for (int i=0;i<16;i++){
	// 	for (int j=0;j<16;j++){
	// 		= diff_table_sorted_prob[i][j];
	// 	}
	// 	std::cerr<< std::endl;
	// }
}
