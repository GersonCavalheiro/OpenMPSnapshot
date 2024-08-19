#pragma once

#include <gtest/gtest.h>

#include "core/raxml/Model.hpp"
#include "util/Options.hpp"

class Epatest : public ::testing::Environment {
public:

std::string data_dir;
std::string out_dir;
std::string tree_file;
std::string tree_file_rooted;
std::string tree_file_rooted_2;
std::string tree_file_rooted_3;
std::string reference_file;
std::string combined_file;
std::string query_file;
std::string binary_file;
std::string info_file;
raxml::Model model = raxml::Model("GTR+G");
Options options;

};

extern Epatest* env;

#include <cmath>

#define COMPL_REPEATS       (1 << 0)
#define COMPL_OPTIMIZE      (1 << 1)
#define COMPL_SLIDING_BLO   (1 << 2)
#define COMPL_PRESCORING    (1 << 3)
#define COMPL_MASKING       (1 << 4)

static inline Options get_options_config(const unsigned int d)
{
Options o;
if (d & COMPL_REPEATS) {
o.repeats = not o.repeats;
}
if (d & COMPL_OPTIMIZE) {
o.opt_branches = not o.opt_branches;
o.opt_model = not o.opt_model;
}
if (d & COMPL_SLIDING_BLO) {
o.sliding_blo = not o.sliding_blo;
}
if (d & COMPL_PRESCORING) {
o.prescoring = not o.prescoring;
}
if (d & COMPL_MASKING) {
o.premasking = not o.premasking;
}
return o;
}

template <class Func>
void all_combinations(Func f, bool verbose=false)
{
for (size_t i = 0; i < pow(2, 4); ++i) {
auto o = get_options_config(i);
if (verbose) {
printf("\nrepeats\toptim\tsliding\tprescore\tmasking\n");
printf( "%d\t%d\t%d\t%d\t%d\n",
o.repeats,
o.opt_model,
o.sliding_blo,
o.prescoring,
o.premasking);
}
f(o);
}
}
