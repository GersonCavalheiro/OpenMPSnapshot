
#pragma once

namespace Encoding {
namespace DataStructures {
struct HuffmanEncodingTreeNode {
unsigned char value;
unsigned weight;
HuffmanEncodingTreeNode* left;
HuffmanEncodingTreeNode* right;
HuffmanEncodingTreeNode* parent;
};
}
}
