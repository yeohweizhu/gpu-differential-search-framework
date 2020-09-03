#ifndef COMMON_GUARD
#define COMMON_GUARD 

#include <iostream>
#include <string>
// #include <cstdint> 

/*
* "common.h"
* Contains Common Macro (prefixed with WZ_) or Inline Functions (namespace wz::)
* Contains Common Function such as Differential Distribution Table, DDT - 4 bit, DDT - 8 bit
* Macro/Inline Functions Patterns: (Input, Input Length*, Output, Ouput Length**(share the same as input if required and not specified else omitted))
*/

/** List of Function
 * Hamming Weight - hw_byte, hw_word, hw_word_u4
 * Differential - diff_xor,
 *
 */

 //128-bit (4 * 32) Differential Structure that contains dx,dy, p and as
struct differential_128b_4s_t {
	//Input Difference
	unsigned char dx[32] = { 0 };
	//Output Difference
	unsigned char dy[32] = { 0 };
	//Probability of Dx->Dy
	unsigned char dy_b4p[32] = { 0 }; // Before mix column and shift row, debug puporses for MDS structure inside differential_t

	double p;
	//Active AS from Dx->Dy (basically AS HW of DX)
	unsigned long long as;

	differential_128b_4s_t() :p(0.0), as(0) {

	}
	differential_128b_4s_t(unsigned char* dx){
		for (int i=0;i<32;i++){
			this->dx[i] = dx[i];
		}
	}

	//Copy assignment
	differential_128b_4s_t& operator=(const differential_128b_4s_t& other){
		for (int i=0;i<32;i++){
			this->dx[i] = other.dx[i];
			this->dy[i] = other.dy[i];
			this->dy_b4p[i] = other.dy_b4p[i];
		}
		this->p = other.p;
		this->as = other.as;

		// by convention, always return *this
		return *this;
	}

	//Condense for string (in hexadicmal sepearted by a space for every 4 hex digits)
	std::string dx_str();
	//Condense for string (in hexadicmal sepearted by a space for every 4 hex digits)
	std::string dy_str();
	//Condense for string (in hexadicmal sepearted by a space for every 4 hex digits)
	std::string dy_b4p_str();
};

//INLINE FUNCTION
namespace wz {

	//ROTL_64ULL


	//ROTR_64ULL


	//Get Hamming Weight of 8bit-grouped
	inline void hw_byte(unsigned char* array1, const int& length, int& weight) {
		weight = 0;
		for (int i = 0; i < length; i++) {
			//*reinterpret_cast<int*>(&array1[i]); reinterpret_cast<int&>(array1[i]) > 0; DINT WORK Because reinterpret will take in extra memory space
			if ((array1[i]) > 0) {
				weight++;
			}
		}
	}


	//Hamming Weight of a nibble (4bit) on byte(8bit), MAX=2 and MIN=0
	inline void hw_word(unsigned char* array1, const int& length, int& weight) {
		weight = 0;
		for (int i = 0; i < length; i++) {
			unsigned char word = array1[i] & 0xf;

			//*reinterpret_cast<int*>(&array1[i]); reinterpret_cast<int&>(array1[i]) > 0; DINT WORK Because reinterpret will take in extra memory space
			if ((word) > 0) {
				weight++;
			}

			word = (array1[i] >> 4) & 0xf;
			if ((word) > 0) {
				weight++;
			}
		}
	}

	//Hamming Weight of the lower nibble on a byte (0x01 = 1, 0x11 = 1), MAX = 1 and MIN = 0 
	inline void hw_word_u4(unsigned char* array1, const int& length, int& weight) {
		weight = 0;
		for (int i = 0; i < length; i++) {
			// unsigned char word = array1[i]&0xf;
			if ((array1[i] & 0xf) > 0) {
				weight++;
			}
		}
	}

	//Hamming Weight of the Bit (MIN = 0 and MAX = 4)
	inline void hw_bit_u4(unsigned char* array1, const int& length, int& weight) {
		weight = 0;
		for (int i = 0; i < length; i++) {
			for (int j = 0; j < 4; j++) {
				if ((array1[i] & (0x1 << j)) > 0) {
					weight++;
				}
			}
		}
	}

	//Differential Computation between two unsigned char array - For tracking differential purpose
	inline void diff_xor(unsigned char* array1, unsigned char* array2, const int& length, unsigned char* diff_array) {
		for (int i = 0; i < length; i++) {
			diff_array[i] = array1[i] ^ array2[i];
		}
	}

	//Start of non inline function

	//Differential Distribution Table (4bit Sbox), Table is assumed to be initialized to be 2D table of at least 16 * 16.
	void diff_table_4bit(const unsigned char* sbox, unsigned int(*table)[16]);
	/*
	*Differential Distribution Table (4bit Sbox), Diff_table [16][table_dy_length[16]] provide Dy, Freq_table is [16][table_dy_length[16]] provide Dy Frequency
	* Note: [8] is the theoritcal maximum and is used instead of table_dy_length[16]
	*/
	void diff_table_4bit_compact(const unsigned char* sbox, unsigned int(*diff_table)[8], unsigned int(*freq_table)[8], unsigned int* table_dy_length, int hw = 4,
	bool to_print = true);

	void diff_table_4bit_reversed(const unsigned char* sbox, unsigned int(*table)[16]);
	void diff_table_4bit_compact_reversed(const unsigned char* sbox, unsigned int(*diff_table)[8], unsigned int(*freq_table)[8], unsigned int* table_dy_length, int hw = 4,
	bool to_print = true);

	//Differential Distribution Table (8bit Sbox), Table is assumed to be initialized to be 2D table of at least 2^8 * 2^8. 
	void diff_table_8(const unsigned char* sbox, unsigned int** table);

	//Compact version of diff_table_8 whereby entry of dx->dy with zero frequency is omitted (sorted by freq descending), Second table is used to record the dy.
	void diff_table_8_compact();

}; //END of Namespace WZ

//Get Hamming Weight of 4bit-grouped
#define WZ_HW_WORD() ()

//Get Hamming Weight of 1bit
#define WZ_HW_BIT() ());

//Differential Computation between two unsigned char array - For tracking differential purpose
// #define WZ_DIFFERENTIAL_XOR(array1, array2, length, diff_array) (for(int i=0;i<length;i++){diff_array[i]= array1[i] ^ array2[i];})

// /*
// * Common Specific Not exposed
// *
// */
// namespace wz_common_internal{

// }

#endif
