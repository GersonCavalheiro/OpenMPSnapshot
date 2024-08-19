#pragma once

#include <string>
#include <tuple>
#include <limits>

#include "core/raxml/Model.hpp"
#include "core/pll/pllhead.hpp"
#include "core/pll/rtree_mapper.hpp"
#include "tree/Tree_Numbers.hpp"
#include "tree/Tree.hpp"
#include "seq/MSA.hpp"
#include "seq/MSA_Info.hpp"
#include "util/Options.hpp"

class MSA;

MSA build_MSA_from_file(const std::string& msa_file, const MSA_Info& info, const bool premasking = false);
pll_utree_s * build_tree_from_file( const std::string& tree_file,
Tree_Numbers& nums,
rtree_mapper& mapper,
const bool preserve_rooting = true);
pll_partition_t * make_partition( const raxml::Model& model,
Tree_Numbers& nums,
const int num_sites,
const Options options);
void file_check(const std::string& file_path);
std::vector<size_t> get_offsets(const std::string& file, MSA& msa);
int pll_fasta_fseek(pll_fasta_t* fd, const long int offset, const int whence);
