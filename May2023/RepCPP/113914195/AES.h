#pragma once





#include <vector>
#include <iostream>

#include "Table.h"

#define AES_BITS 128
#define NUM_ROUNDS 10
#define SUB_KEYS (NUM_ROUNDS + 1)
#define KEY_BLOCK 16

using std::vector;






__device__ unsigned char *d_keySchedule;

__device__ void byte_sub(unsigned char *block_biffer, unsigned char *sharedSbox);
__device__ void shift_rows(unsigned char *block_biffer);
__device__ void aes_MixColumns(unsigned char *block_biffer);
__device__ void mix_columns(unsigned char *column);
__device__ void key_addition(unsigned char *block_biffer, unsigned char *key, const unsigned int &round);





__global__ void aes_encryption_shared(unsigned char *message, unsigned char *result, unsigned char *sbox, unsigned char *keys, unsigned int width);
__global__ void aes_encryption(unsigned char *message, unsigned char *result, unsigned char *keys, unsigned int width);





unsigned char* key_schedule(unsigned char *key);
unsigned char* sub_key128(unsigned char *prev_subkey, const int &r);