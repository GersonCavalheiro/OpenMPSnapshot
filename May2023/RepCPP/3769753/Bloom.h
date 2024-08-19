

#ifndef BLOOM_H_
#define BLOOM_H_

#include "Common/Kmer.h"
#include "Common/HashFunction.h"
#include "Common/Uncompress.h"
#include "Common/IOUtil.h"
#include "DataLayer/FastaReader.h"
#include <iostream>
#include <vector>

#if _OPENMP
# include <omp.h>
#endif

namespace Bloom {

typedef Kmer key_type;


struct FileHeader {
unsigned bloomVersion;
unsigned k;
size_t fullBloomSize;
size_t startBitPos;
size_t endBitPos;
size_t hashSeed;
};


static const unsigned LOAD_PROGRESS_STEP = 100000;

static const unsigned BLOOM_VERSION = 5;


inline static size_t hash(const key_type& key)
{
if (key.isCanonical())
return hashmem(&key, key.bytes());

key_type copy(key);
copy.reverseComplement();
return hashmem(&copy, copy.bytes(), 0);
}


inline static size_t hash(const key_type& key, size_t seed)
{
if (key.isCanonical())
return hashmem(&key, key.bytes(), seed);

key_type copy(key);
copy.reverseComplement();
return hashmem(&copy, copy.bytes(), seed);
}

template <typename BF>
inline static void loadSeq(BF& bloomFilter, unsigned k, const std::string& seq);


template <typename BF>
inline static void loadFile(BF& bloomFilter, unsigned k, const std::string& path,
bool verbose = false, size_t taskIOBufferSize = 100000)
{
assert(!path.empty());
if (verbose)
std::cerr << "Reading `" << path << "'...\n";
FastaReader in(path.c_str(), FastaReader::FOLD_CASE);
uint64_t count = 0;
#pragma omp parallel
for (std::vector<std::string> buffer(taskIOBufferSize);;) {
buffer.clear();
size_t bufferSize = 0;
bool good = true;
#pragma omp critical(in)
for (; good && bufferSize < taskIOBufferSize;) {
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
loadSeq(bloomFilter, k, buffer.at(j));
if (verbose)
#pragma omp critical(cerr)
{
count++;
if (count % LOAD_PROGRESS_STEP == 0)
std::cerr << "Loaded " << count << " reads into bloom filter\n";
}
}
}
assert(in.eof());
if (verbose) {
std::cerr << "Loaded " << count << " reads from `"
<< path << "` into bloom filter\n";
}
}


template <typename BF>
inline static void loadSeq(BF& bloomFilter, unsigned k, const std::string& seq)
{
if (seq.size() < k)
return;
for (size_t i = 0; i < seq.size() - k + 1; ++i) {
std::string kmer = seq.substr(i, k);
size_t pos = kmer.find_last_not_of("ACGTacgt");
if (pos == std::string::npos) {
bloomFilter.insert(Kmer(kmer));
} else
i += pos;
}
}

inline static void writeHeader(std::ostream& out, const FileHeader& header)
{
(void)writeHeader;

out << BLOOM_VERSION << '\n';
out << Kmer::length() << '\n';
out << header.fullBloomSize
<< '\t' << header.startBitPos
<< '\t' << header.endBitPos
<< '\n';
out << header.hashSeed << '\n';
assert(out);
}

FileHeader readHeader(std::istream& in)
{
FileHeader header;


in >> header.bloomVersion >> expect("\n");
assert(in);
if (header.bloomVersion != BLOOM_VERSION) {
std::cerr << "error: bloom filter version (`"
<< header.bloomVersion << "'), does not match version required "
"by this program (`" << BLOOM_VERSION << "').\n";
exit(EXIT_FAILURE);
}


in >> header.k >> expect("\n");
assert(in);
if (header.k != Kmer::length()) {
std::cerr << "error: this program must be run with the same kmer "
"size as the bloom filter being loaded (k="
<< header.k << ").\n";
exit(EXIT_FAILURE);
}


in >> header.fullBloomSize
>> expect("\t") >> header.startBitPos
>> expect("\t") >> header.endBitPos
>> expect("\n");

in >> header.hashSeed >> expect("\n");

assert(in);
assert(header.startBitPos < header.fullBloomSize);
assert(header.endBitPos < header.fullBloomSize);
assert(header.startBitPos <= header.endBitPos);

return header;
}

};

#endif 
