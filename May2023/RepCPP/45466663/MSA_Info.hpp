#pragma once

#include "genesis/sequence/functions/functions.hpp"
#include "genesis/utils/math/bitvector.hpp"
#include "genesis/utils/math/bitvector/operators.hpp"
#include "genesis/sequence/formats/fasta_input_iterator.hpp"

#include <string>


class MSA_Info
{
public:
using mask_type = genesis::utils::Bitvector;


MSA_Info( const std::string& file_path,
std::function<void(const genesis::sequence::Sequence&)> fn = nullptr )
: path_(file_path)
{
auto it = genesis::sequence::FastaInputIterator( genesis::utils::from_file(file_path) );

if (it) {
sites_ = it->length();
gap_mask_ = mask_type(sites_, true);
}

while ( it ) {
++sequences_;

const auto& seq = *it;

if (fn) {
fn(seq);
}

if (sites_ and (sites_ != seq.length())) {
throw std::runtime_error{path_
+ " does not contain equal size sequences! First offending sequence: "
+ seq.label()};
}

auto cur_mask = genesis::sequence::gap_sites(seq);
gap_mask_ &= cur_mask;

++it;
}
}

MSA_Info( const std::string& file_path,
const size_t sequences,
const mask_type& mask,
const size_t sites = 0)
: path_(file_path)
, sites_(sites)
, sequences_(sequences)
, gap_mask_(mask)
{ }

MSA_Info() = default;
~MSA_Info() = default;

const std::string& path() const {return path_;}
size_t sites() const {return sites_;}
size_t sequences() const {return sequences_;}
const mask_type& gap_mask() const {return gap_mask_;}
size_t gap_count() const {return gap_mask_.count();}


static void or_mask(MSA_Info& lhs, MSA_Info& rhs)
{
if (lhs.sites() != rhs.sites()) {
throw std::runtime_error{std::string("")
+ "MSA_Infos are unequal site width: "
+ std::to_string(lhs.sites()) + " vs. " + std::to_string(rhs.sites())};
}

lhs.gap_mask_ = rhs.gap_mask_ = lhs.gap_mask() | rhs.gap_mask();

}

private:
std::string path_ = "";
size_t sites_ = 0;
size_t sequences_ = 0;
mask_type gap_mask_;

};

inline std::string subset_sequence( const std::string& seq,
const MSA_Info::mask_type& mask)
{
const size_t nongap_count = mask.size() - mask.count();
std::string result(nongap_count, '$');

if (seq.length() != mask.size()) {
throw std::runtime_error{"In subset_sequence: mask and seq incompatible"};
}

size_t k = 0;
for (size_t i = 0; i < seq.length(); ++i) {
if (not mask[i]) {
result[k++] = seq[i];
}
}

assert(nongap_count == k);

return result;
}

inline std::ostream& operator << (std::ostream& out, MSA_Info const& rhs)
{
out << "Path: " << rhs.path();
out << "\nSequences: " << rhs.sequences();
out << "\nSites: " << rhs.sites();
out << "\nGaps: " << rhs.gap_count();
out << "\nFraction of gaps: " << rhs.gap_count() / static_cast<double>(rhs.sites());
out << "\n";

return out;
}

MSA_Info make_msa_info(const std::string& file_path);
