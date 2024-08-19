#pragma once

#include <string>
#include <limits>
#include <memory>
#include <sstream>

#include "seq/MSA.hpp"
#include "seq/MSA_Info.hpp"
#include "io/encoding.hpp"
#include "util/template_magic.hpp"
#include "util/stringify.hpp"
#include "util/logging.hpp"

#include "genesis/utils/io/serializer.hpp"
#include "genesis/utils/io/deserializer.hpp"
#include "genesis/sequence/formats/fasta_input_iterator.hpp"
#include "genesis/sequence/functions/functions.hpp"
#include "genesis/sequence/sequence.hpp"

constexpr char MAGIC[] = "BFAST\0";
constexpr size_t MAGIC_SIZE = array_size(MAGIC);

using mask_type = MSA_Info::mask_type;

using namespace genesis;

static FourBit& code_()
{
static FourBit obj;
return obj;
}

static inline size_t data_section_offset(const size_t num_sequences, const size_t mask_size)
{
return MAGIC_SIZE
+ sizeof(num_sequences)
+ (num_sequences *
sizeof(uint64_t) * 2 )
+ mask_size + sizeof(uint64_t);
}

static inline void write_header(utils::Serializer& ser,
const std::vector<size_t>& entry_sizes,
const mask_type& mask)
{
const uint64_t num_sequences = entry_sizes.size();

ser.put_raw(MAGIC, MAGIC_SIZE);
ser.put_int(num_sequences);

std::stringstream ss;
ss << mask;
ser.put_string(ss.str());

uint64_t offset = data_section_offset(num_sequences, mask.size());
for (uint64_t i = 0; i < num_sequences; ++i) {
ser.put_int(i);
ser.put_int(offset);

const uint64_t sizeof_entry = sizeof(uint64_t)*2 
+ static_cast<uint64_t>(entry_sizes[i]);
offset += sizeof_entry;
}
}

static inline std::vector<size_t> get_entry_sizes(const MSA& msa)
{
std::vector<size_t> res;
for (const auto& s : msa) {
res.push_back(
code_().packed_size(s.sequence().size()) +
s.header().size()
);
}
return res;
}

static inline void put_encoded(utils::Serializer& ser,
const std::string& seq)
{
const auto encoded_seq = code_().to_fourbit(seq);
ser.put_int<uint64_t>(seq.size());
ser.put_raw_string(encoded_seq);
}

static std::string get_decoded(utils::Deserializer& des)
{
const auto decoded_size = des.get_int<uint64_t>();

const auto coded_size = code_().packed_size(decoded_size);

auto coded_str = des.get_raw_string(coded_size);

return code_().from_fourbit(coded_str, decoded_size);
}

static void read_header(utils::Deserializer& des,
std::vector<uint64_t>& offset,
mask_type& mask)
{
char magic[MAGIC_SIZE];
des.get_raw(magic, MAGIC_SIZE);

if (strcmp(magic, MAGIC)) {
throw std::runtime_error{std::string("File is not an epa::Binary_Fasta file")};
}

const uint64_t num_sequences = des.get_int<uint64_t>();

mask = mask_type();
std::stringstream mask_str( des.get_string() );
mask_str >> mask;

offset = std::vector<uint64_t>(num_sequences);
for (size_t i = 0; i < num_sequences; ++i) {
const auto idx = des.get_int<uint64_t>();
offset[idx] = des.get_int<uint64_t>();
}
}

static MSA read_sequences(utils::Deserializer& des,
const mask_type& mask,
const size_t number,
const size_t sites = 0)
{
MSA msa(sites);

for (size_t i = 0; i < number; ++i) {
auto label    = des.get_string();
auto sequence = get_decoded(des);

if ( mask.count() ) {
sequence = subset_sequence(sequence, mask);
}

msa.append( label, sequence );
}

return msa;
}

static void ensure_dna(const std::string& seq)
{
for (const auto& s : seq) {
bool found = false;
for (size_t i = 0; i < NT_MAP_SIZE and not found; ++i) {
found = (s == NT_MAP[i]);
}
if (not found) {
throw std::runtime_error{std::string("AA DATA NOT SUPPORTED for conversion to bfast!")
+ " Sorry! Offending char: " + s
};
}
}
}

class Binary_Fasta
{
private:
Binary_Fasta() = delete;
~Binary_Fasta() = delete;

public:

static MSA_Info get_info(const std::string& file)
{
utils::Deserializer des(file);

std::vector<uint64_t> offset;
mask_type mask;
read_header(des, offset, mask);

const auto sequences = offset.size();

return MSA_Info(file, sequences, mask, mask.size());
}

static void save(const MSA& msa, const std::string& file_name)
{
mask_type gap_mask(msa.num_sites(), true);
for (const auto& s : msa) {
const genesis::sequence::Sequence seq("", s.sequence());
auto cur_mask = genesis::sequence::gap_sites(seq);
gap_mask &= cur_mask;
}

utils::Serializer ser(file_name);

auto sizes = get_entry_sizes(msa);
write_header(ser, sizes, gap_mask);

for (const auto& s : msa) {
ser.put_string(s.header());
put_encoded(ser, s.sequence());
}

}

static MSA load(const std::string& file_name,
const bool premasking = false)
{
utils::Deserializer des(file_name);

std::vector<uint64_t> offset;
mask_type mask;
read_header(des, offset, mask);

if (not premasking) {
mask = mask_type();
}

return read_sequences( des, mask, offset.size() );
}

static std::string fasta_to_bfast( const std::string& fasta_file,
std::string out_dir)
{
auto parts = split_by_delimiter(fasta_file, "/");

out_dir += parts.back() + ".bfast";

std::vector<size_t> entry_sizes;
auto get_sizes = [&](const genesis::sequence::Sequence& s)
{
entry_sizes.push_back(
s.label().size() +
code_().packed_size(s.sites().size())
);
};

MSA_Info info(fasta_file, get_sizes);

LOG_DBG << info;

utils::Serializer ser(out_dir);
write_header(ser, entry_sizes, info.gap_mask());

auto it = sequence::FastaInputIterator( utils::from_file(fasta_file) );

ensure_dna( it->sites() );

while ( it ) {
ser.put_string(it->label());
put_encoded(ser, it->sites());
++it;
}
return out_dir;
}

};

#include "io/msa_reader_interface.hpp"
#include "net/epa_mpi_util.hpp"

class Binary_Fasta_Reader : public msa_reader
{
public:
Binary_Fasta_Reader(std::string const& file_name,
MSA_Info const& info,
bool const premasking = false,
bool const split = false)
: istream_(file_name)
, des_(istream_)
, mask_(info.gap_mask())
{
mask_type dummy_mask;
read_header(des_, seq_offsets_, dummy_mask);

assert(seq_offsets_.size() == info.sequences());


#ifdef __MPI
if ( split ) {
std::tie( local_seq_offset_, max_read_ ) = local_seq_package( info.sequences() );

istream_ = std::ifstream( file_name );
istream_.seekg( seq_offsets_[ local_seq_offset_ ], istream_.beg );

des_ = utils::Deserializer( istream_ );
}
#else
static_cast<void>(split);
#endif

max_read_ = std::min( seq_offsets_.size(), max_read_);

if (not premasking) {
mask_ = mask_type();
}
}

~Binary_Fasta_Reader() = default;

virtual size_t read_next(MSA& result, const size_t number) override
{
const auto to_read =
std::min( number, max_read_ - num_read_ );

result = read_sequences( des_, mask_, to_read );

num_read_ += result.size();

return result.size();
}

virtual size_t num_sequences() const override
{
return seq_offsets_.size();
}

virtual size_t local_seq_offset() const override
{
return local_seq_offset_;
}


private:
std::ifstream istream_;
utils::Deserializer des_;
mask_type mask_;
std::vector<uint64_t> seq_offsets_;
size_t num_read_  = 0;
size_t max_read_  = std::numeric_limits<size_t>::max();
size_t local_seq_offset_ = 0;
};
