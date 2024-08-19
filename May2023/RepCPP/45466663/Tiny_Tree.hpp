#pragma once

#include <memory>
#include <unordered_map>

#include "core/pll/pllhead.hpp"
#include "seq/Sequence.hpp"
#include "util/constants.hpp"
#include "util/Options.hpp"
#include "sample/Placement.hpp"
#include "tree/Tree.hpp"
#include "core/pll/pll_util.hpp"
#include "core/Lookup_Store.hpp"


class Tiny_Tree
{
public:
Tiny_Tree(pll_unode_t * edge_node, 
const unsigned int branch_id, 
Tree& reference_tree,
const bool opt_branches,
const Options& options,
std::shared_ptr<Lookup_Store>& lookup);

Tiny_Tree()   = delete;
~Tiny_Tree()  = default;

Tiny_Tree(Tiny_Tree const& other) = delete;
Tiny_Tree(Tiny_Tree&& other)      = default;

Tiny_Tree& operator= (Tiny_Tree const& other) = delete;
Tiny_Tree& operator= (Tiny_Tree && other)     = default;

Placement place(const Sequence& s);

private:
std::unique_ptr<pll_partition_t, partition_deleter> partition_;
std::unique_ptr<pll_utree_t, utree_deleter> tree_;

bool opt_branches_;
double original_branch_length_;
bool premasking_ = true;
bool sliding_blo_;
unsigned int branch_id_;

std::shared_ptr<Lookup_Store> lookup_;

};
