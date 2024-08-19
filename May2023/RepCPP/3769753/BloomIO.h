#ifndef BLOOM_IO_H
#define BLOOM_IO_H 1

#include "BloomDBG/RollingHash.h"
#include "BloomDBG/RollingHashIterator.h"
#include "DataLayer/FastaReader.h"
#include "vendor/btl_bloomfilter/BloomFilter.hpp"

namespace BloomDBG {


template<typename T>
inline static T
roundUpToMultiple(T num, T base)
{
if (base == 0)
return num;
T remainder = num % base;
if (remainder == 0)
return num;
return num + base - remainder;
}


template<typename BF>
inline static void
loadSeq(BF& bloom, const std::string& seq)
{
const unsigned k = bloom.getKmerSize();
const unsigned numHashes = bloom.getHashNum();
for (RollingHashIterator it(seq, numHashes, k); it != RollingHashIterator::end(); ++it) {
bloom.insert(*it);
}
}


template<typename BF>
inline static void
loadFile(BF& bloom, const std::string& path, bool verbose = false)
{
const size_t BUFFER_SIZE = 1000000;
const size_t LOAD_PROGRESS_STEP = 10000;

assert(!path.empty());
if (verbose)
std::cerr << "Reading `" << path << "'..." << std::endl;

FastaReader in(path.c_str(), FastaReader::FOLD_CASE);
uint64_t readCount = 0;
#pragma omp parallel
for (std::vector<std::string> buffer(BUFFER_SIZE);;) {
buffer.clear();
size_t bufferSize = 0;
bool good = true;
#pragma omp critical(in)
for (; good && bufferSize < BUFFER_SIZE;) {
std::string seq;
good = in >> seq;
if (good) {
buffer.push_back(seq);
bufferSize += seq.length();
}
}
if (buffer.size() == 0)
break;
for (size_t j = 0; j < buffer.size(); j++) {
loadSeq(bloom, buffer.at(j));
if (verbose)
#pragma omp critical(cerr)
{
readCount++;
if (readCount % LOAD_PROGRESS_STEP == 0)
std::cerr << "Loaded " << readCount << " reads into Bloom filter\n";
}
}
}
assert(in.eof());
if (verbose) {
std::cerr << "Loaded " << readCount << " reads from `" << path << "` into Bloom filter\n";
}
}


template<typename BloomFilterT>
static inline void
loadBloomFilter(int argc, char** argv, BloomFilterT& bloom, bool verbose = false)
{

for (int i = optind; i < argc; ++i) {

if (strcmp(argv[i], ":") == 0) {
optind = i + 1;
break;
}
BloomDBG::loadFile(bloom, argv[i], verbose);
}
if (verbose)
cerr << "Bloom filter FPR: " << setprecision(3) << bloom.FPR() * 100 << "%" << endl;
}

} 

#endif
