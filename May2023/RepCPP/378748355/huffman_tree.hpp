
#pragma once

#include "bitstream.hpp"
#include "huffman_tree_node.hpp"
#include <list>

namespace Encoding {
namespace DataStructures {
class HuffmanEncodingTree {

public:
HuffmanEncodingTree();
~HuffmanEncodingTree();

void EncodeArray(unsigned char* input, unsigned sizeInBytes, NetworkBitStream* output);

unsigned DecodeArray(NetworkBitStream* input, unsigned& sizeInBits, unsigned maxCharsToWrite, unsigned char* output, bool skip = true);
void DecodeArray(unsigned char* input, unsigned sizeInBits, NetworkBitStream* output);

void GenerateFromFrequencyTable(unsigned int frequencyTable[256]);

void FreeMemory(void);

private:

HuffmanEncodingTreeNode* root;


struct CharacterEncoding {
unsigned char* encoding;
unsigned short bitLength;
};

CharacterEncoding encodingTable[256];

void InsertNodeIntoSortedList(HuffmanEncodingTreeNode* node, std::list<HuffmanEncodingTreeNode*>& huffmanEncodingTreeNodeList) const;
};
}
}
