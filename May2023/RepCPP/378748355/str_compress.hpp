
#pragma once

#include "bitstream.hpp"
#include "huffman_tree.hpp"

namespace Encoding {

class StringCompressor {
public:
static StringCompressor* Instance(void);

void EncodeString(const char* input, int maxCharsToWrite, NetworkBitStream* output);

bool DecodeString(char* output, int maxCharsToWrite, NetworkBitStream* input);

bool DecodeString(char* output, int maxCharsToWrite, NetworkBitStream* input, unsigned& stringBitLength, bool skip);

private:
StringCompressor();

DataStructures::HuffmanEncodingTree huffmanEncodingTree;

static StringCompressor* instance;
};
}

#define stringCompressor Encoding::StringCompressor::Instance()
