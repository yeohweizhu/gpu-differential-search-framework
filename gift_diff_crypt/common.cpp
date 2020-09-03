#include "common.h"

#include <vector>
#include <algorithm>
#include <iostream>
#include <string>
#include <sstream>

std::string differential_128b_4s_t::dx_str() {
	std::ostringstream ss;
	ss << std::hex;
	for (int i = 0; i < 32; i++) {
		if ( (i % 4 == 0) && (i != 0) ) {
			ss << " ";
		}
		ss << (int)this->dx[i];
	}
	return ss.str();
}
std::string differential_128b_4s_t::dy_str() {
	std::ostringstream ss;
	ss << std::hex;
	for (int i = 0; i < 32; i++) {
		if ( (i % 4 == 0) && (i != 0) ) {
			ss << " ";
		}
		ss << (int)this->dy[i];
	}
	return ss.str();
}
std::string differential_128b_4s_t::dy_b4p_str() {
	std::ostringstream ss;
	ss << std::hex;
	for (int i = 0; i < 32; i++) {
		if ((i % 4 == 0) && (i != 0)) {
			ss << " ";
		}
		ss << (int)this->dy_b4p[i];
	}
	return ss.str();
}

void wz::diff_table_4bit(const unsigned char* sbox, unsigned int(*table)[16]) {
	for (int i = 0; i < 16; i++) {
		for (int j = 0; j < 16; j++) {
			table[i][j] = 0;
		}
	}

	//Loop thorugh all x1
	for (int x1 = 0; x1 < 16; x1++) {
		for (int x2 = 0; x2 < 16; x2++) {  //X2
			int dx = x1 ^ x2;              //DeltaX

			int y1 = sbox[x1];  //Run x1 and x2 thorugh s-box
			int y2 = sbox[x2];

			int dy = y1 ^ y2;  //DeltaY

			table[dx][dy] += 1;  //DeltaX produce DeltaY frequency counter
		}
	}
}

void wz::diff_table_4bit_compact(const unsigned char* sbox, unsigned int(*diff_table)[8], unsigned int(*freq_table)[8], unsigned int* table_dy_length, int hw
	, bool to_print) {
	for (int i = 0; i < 16; i++) { //Zero initialzed for safety.
		table_dy_length[i] = 0;

		for (int j = 0; j < 8; j++) { 
			diff_table[i][j] = 0;
			freq_table[i][j] = 0;
		}
	}

	unsigned int full_table[16][16] = { 0 };
	wz::diff_table_4bit(sbox, full_table);

	//Sorting Table
	std::vector<std::vector<std::pair<unsigned int, unsigned int>>>full_vec;
	for (int i = 0; i < 16; i++) {
		std::vector<std::pair<unsigned int, unsigned int>>new_row;
		for (int j = 0; j < 16; j++) {
			std::pair<unsigned int, unsigned int> new_pair;
			new_pair.first = j; //Dy
			new_pair.second = full_table[i][j]; // Freq
			new_row.push_back(new_pair);
		}
		full_vec.push_back(new_row);
	}
	for (int i = 0; i < 16; i++) {
		std::sort(full_vec[i].begin(), full_vec[i].end(),
			[](std::pair<unsigned int, unsigned int> a, std::pair<unsigned int, unsigned int> b) { return a.second > b.second; });
	}

	//Converting Table
	int weight = 0;

	for (int i = 0; i < 16; i++) {
		int j_offset = 0;
		for (int j = 0; j < 16; j++) {
			//Guaranteed to be stop at maximum j=8, do not change to j<8 without refactoring because of assignment opr
			if (full_vec[i][j].second == 0) { // If reached frequency zero, thus all other follow will be zero becuase sort
				table_dy_length[i] = j - j_offset;
				break;
			}

			unsigned char dx_char = static_cast<unsigned char>((full_vec[i][j].first & 0xf));
			wz::hw_bit_u4(&dx_char, 1, weight);
			if (weight > hw) {
				j_offset++;
				continue;
			}
			else {
				diff_table[i][j - j_offset] = full_vec[i][j].first;
				freq_table[i][j - j_offset] = full_vec[i][j].second;
			}
		}
	}

	//Printing Table
	if (to_print){
		std::cout << std::hex;
		std::cout << "\n\nSbox: [";
		for (int i=0;i<16;i++){
			std::cout << sbox[i] <<" , ";
		}
		std::cout << "]";
		std::cout << "\n\nDifferential Table";
		for (int i=0;i<16;i++){
			std::cout << "\n[ ";
			for (int j=0;j<8;j++){
				std:: cout << "0x" << diff_table[i][j] << ", ";
			}
			std::cout << "]";
		}
		std::cout << std::dec;
		std::cout << "\n\nFreq Table";
		for (int i=0;i<16;i++){
			std::cout << "\n[";
			for (int j=0;j<8;j++){
				std:: cout << freq_table[i][j] << " , ";
			}
			std::cout << "]";
		}

		std::cout.flush();
		std::cout << "\n\nPRESS ENTER KEY TO CONTINUE\n";
		getchar();
	}
}


void wz::diff_table_4bit_reversed(const unsigned char* sbox, unsigned int(*table)[16]) {
	unsigned char sbox_reversed[16] = {0};
	for (int i=0;i<16;i++){
		sbox_reversed[sbox[i]] = i;
	}

	for (int i = 0; i < 16; i++) {
		for (int j = 0; j < 16; j++) {
			table[i][j] = 0;
		}
	}

	//Loop thorugh all x1
	for (int x1 = 0; x1 < 16; x1++) {
		for (int x2 = 0; x2 < 16; x2++) {  //X2
			int dx = x1 ^ x2;              //DeltaX

			int y1 = sbox_reversed[x1];  //Run x1 and x2 thorugh s-box
			int y2 = sbox_reversed[x2];

			int dy = y1 ^ y2;  //DeltaY

			table[dx][dy] += 1;  //DeltaX produce DeltaY frequency counter
		}
	}
}

void wz::diff_table_4bit_compact_reversed(const unsigned char* sbox, unsigned int(*diff_table)[8], unsigned int(*freq_table)[8], unsigned int* table_dy_length, int hw,
	bool to_print) {

	for (int i = 0; i < 16; i++) { //Zero initialzed for safety.
		table_dy_length[i] = 0;
		for (int j = 0; j < 8; j++) {
			diff_table[i][j] = 0;
			freq_table[i][j] = 0;
		}
	}

	unsigned int full_table[16][16] = { 0 };
	wz::diff_table_4bit_reversed(sbox, full_table);

	//Sorting Table
	std::vector<std::vector<std::pair<unsigned int, unsigned int>>>full_vec;
	for (int i = 0; i < 16; i++) {
		std::vector<std::pair<unsigned int, unsigned int>>new_row;
		for (int j = 0; j < 16; j++) {
			std::pair<unsigned int, unsigned int> new_pair;
			new_pair.first = j; //Dy
			new_pair.second = full_table[i][j]; // Freq
			new_row.push_back(new_pair);
		}
		full_vec.push_back(new_row);
	}
	for (int i = 0; i < 16; i++) {
		std::sort(full_vec[i].begin(), full_vec[i].end(),
			[](std::pair<unsigned int, unsigned int> a, std::pair<unsigned int, unsigned int> b) { return a.second > b.second; });
	}

	//Converting Table
	int weight = 0;

	for (int i = 0; i < 16; i++) {
		int j_offset = 0;
		for (int j = 0; j < 16; j++) {
			//Guaranteed to be stop at maximum j=8, do not change to j<8 without refactoring because of assignment opr
			if (full_vec[i][j].second == 0) { // If reached frequency zero, thus all other follow will be zero becuase sort
				table_dy_length[i] = j - j_offset;
				break;
			}

			unsigned char dx_char = static_cast<unsigned char>((full_vec[i][j].first & 0xf));
			wz::hw_bit_u4(&dx_char, 1, weight);
			if (weight > hw) {
				j_offset++;
				continue;
			}
			else {
				diff_table[i][j - j_offset] = full_vec[i][j].first;
				freq_table[i][j - j_offset] = full_vec[i][j].second;
			}
		}
	}

	//Printing Table
	if (to_print){
		std::cout << std::hex;
		std::cout << "\n\nSbox: [";
		for (int i=0;i<16;i++){
			std::cout << sbox[i] <<" , ";
		}
		std::cout << "]";
		std::cout << "\n\nDifferential Table REVERSED";
		for (int i=0;i<16;i++){
			std::cout << "\n[ ";
			for (int j=0;j<8;j++){
				std:: cout << "0x" << diff_table[i][j] << ", ";
			}
			std::cout << "]";
		}
		std::cout << std::dec;
		std::cout << "\n\nFreq Table REVERSED";
		for (int i=0;i<16;i++){
			std::cout << "\n[";
			for (int j=0;j<8;j++){
				std:: cout << freq_table[i][j] << ", ";
			}
			std::cout << "]";
		}

		std::cout.flush();
		std::cout << "\n\nPRESS ENTER KEY TO CONTINUE\n";
		getchar();
	}

}

