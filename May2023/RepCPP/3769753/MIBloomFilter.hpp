

#ifndef MIBLOOMFILTER_HPP_
#define MIBLOOMFILTER_HPP_

#include <algorithm> 
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <math.h>
#include <omp.h>
#include <sdsl/bit_vector_il.hpp>
#include <sdsl/rank_support.hpp>
#include <stdint.h>
#include <stdio.h>
#include <string>
#include <sys/stat.h>
#include <vector>

using namespace std;

template<typename T>
class MIBloomFilter
{
public:
static const T s_mask = 1 << (sizeof(T) * 8 - 1);
static const T s_antiMask = (T)~s_mask;

static const T s_strand = 1 << (sizeof(T) * 8 - 2);
static const T s_antiStrand = (T)~s_strand;

static const T s_idMask = s_antiStrand & s_antiMask;

static const unsigned BLOCKSIZE = 512;

static inline vector<vector<unsigned>> parseSeedString(const vector<string>& spacedSeeds)
{
vector<vector<unsigned>> seeds(spacedSeeds.size(), vector<unsigned>());
for (unsigned i = 0; i < spacedSeeds.size(); ++i) {
const string ss = spacedSeeds.at(i);
for (unsigned j = 0; j < ss.size(); ++j) {
if (ss.at(j) == '0') {
seeds[i].push_back(j);
}
}
}
return seeds;
}

static inline double
calcProbSingleFrame(double occupancy, unsigned hashNum, double freq, unsigned allowedMisses)
{
double probTotal = 0.0;
for (unsigned i = hashNum - allowedMisses; i <= hashNum; i++) {
double prob = nChoosek(hashNum, i);
prob *= pow(occupancy, i);
prob *= pow(1.0 - occupancy, hashNum - i);
prob *= (1.0 - pow(1.0 - freq, i));
probTotal += prob;
}
return probTotal;
}

static inline double calcProbSingle(double occupancy, double freq) { return occupancy * freq; }


static size_t calcOptimalSize(size_t entries, unsigned hashNum, double occupancy)
{
size_t non64ApproxVal = size_t(-double(entries) * double(hashNum) / log(1.0 - occupancy));
return non64ApproxVal + (64 - non64ApproxVal % 64);
}


static unsigned insert(sdsl::bit_vector& bv, uint64_t* hashValues, unsigned hashNum)
{
unsigned colliCount = 0;
for (unsigned i = 0; i < hashNum; ++i) {
uint64_t pos = hashValues[i] % bv.size();
uint64_t* dataIndex = bv.data() + (pos >> 6);
uint64_t bitMaskValue = (uint64_t)1 << (pos & 0x3F);
colliCount += __sync_fetch_and_or(dataIndex, bitMaskValue) >> (pos & 0x3F) & 1;
}
return colliCount;
}

#pragma pack(1) 
struct FileHeader
{
char magic[8];
uint32_t hlen; 
uint64_t size;
uint32_t nhash;
uint32_t kmer;
uint32_t version;
};


MIBloomFilter<T>(
unsigned hashNum,
unsigned kmerSize,
sdsl::bit_vector& bv,
const vector<string> seeds = vector<string>(0))
: m_dSize(0)
, m_hashNum(hashNum)
, m_kmerSize(kmerSize)
, m_sseeds(seeds)
, m_probSaturated(0)
{
m_bv = sdsl::bit_vector_il<BLOCKSIZE>(bv);
bv = sdsl::bit_vector();
if (!seeds.empty()) {
m_ssVal = parseSeedString(m_sseeds);
assert(m_sseeds[0].size() == kmerSize);
for (vector<string>::const_iterator itr = m_sseeds.begin(); itr != m_sseeds.end();
++itr) {
assert(m_kmerSize == itr->size());
}
}
m_rankSupport = sdsl::rank_support_il<1>(&m_bv);
m_dSize = getPop();
m_data = new T[m_dSize]();
}

MIBloomFilter<T>(const string& filterFilePath)
{
#pragma omp parallel for
for (unsigned i = 0; i < 2; ++i) {
if (i == 0) {
FILE* file = fopen(filterFilePath.c_str(), "rb");
if (file == NULL) {
#pragma omp critical(stderr)
cerr << "file \"" << filterFilePath << "\" could not be read." << endl;
exit(1);
}

FileHeader header;
if (fread(&header, sizeof(struct FileHeader), 1, file) == 1) {
#pragma omp critical(stderr)
cerr << "Loading header..." << endl;
} else {
#pragma omp critical(stderr)
cerr << "Failed to Load header" << endl;
exit(1);
}

char magic[9];
memcpy(magic, header.magic, 8);
magic[8] = '\0';

#pragma omp critical(stderr)
cerr << "Loaded header... magic: " << magic << " hlen: " << header.hlen
<< " size: " << header.size << " nhash: " << header.nhash
<< " kmer: " << header.kmer << endl;

m_hashNum = header.nhash;
m_kmerSize = header.kmer;
m_dSize = header.size;
m_data = new T[m_dSize]();

if (header.hlen > sizeof(struct FileHeader)) {
for (unsigned i = 0; i < header.nhash; ++i) {
char temp[header.kmer];

if (fread(temp, header.kmer, 1, file) != 1) {
cerr << "Failed to load spaced seed string" << endl;
exit(1);
} else {
cerr << "Spaced Seed " << i << ": " << string(temp, header.kmer)
<< endl;
}
m_sseeds.push_back(string(temp, header.kmer));
}

m_ssVal = parseSeedString(m_sseeds);
assert(m_sseeds[0].size() == m_kmerSize);
for (vector<string>::const_iterator itr = m_sseeds.begin();
itr != m_sseeds.end();
++itr) {
assert(m_kmerSize == itr->size());
}
}

if (header.hlen != (sizeof(FileHeader) + m_kmerSize * m_sseeds.size())) {
cerr << "Multi Index Bloom Filter header length: " << header.hlen
<< " does not match expected length: "
<< (sizeof(FileHeader) + m_kmerSize * m_sseeds.size())
<< " (likely version mismatch)" << endl;
exit(1);
}

if (strcmp(magic, "MIBLOOMF")) {
cerr << "Bloom Filter type does not match " << endl;
exit(1);
}

if (header.version != MIBloomFilter_VERSION) {
cerr << "Multi Index Bloom Filter version does not match: " << header.version
<< " expected: " << MIBloomFilter_VERSION << endl;
exit(1);
}

#pragma omp critical(stderr)
cerr << "Loading data vector" << endl;

long int lCurPos = ftell(file);
fseek(file, 0, 2);
size_t fileSize = ftell(file) - header.hlen;
fseek(file, lCurPos, 0);
if (fileSize != m_dSize * sizeof(T)) {
cerr << "Error: " << filterFilePath
<< " does not match size given by its header. Size: " << fileSize << " vs "
<< m_dSize * sizeof(T) << " bytes." << endl;
exit(1);
}

size_t countRead = fread(m_data, fileSize, 1, file);
if (countRead != 1 && fclose(file) != 0) {
cerr << "file \"" << filterFilePath << "\" could not be read." << endl;
exit(1);
}
} else {
string bvFilename = filterFilePath + ".sdsl";
#pragma omp critical(stderr)
cerr << "Loading sdsl interleaved bit vector from: " << bvFilename << endl;
load_from_file(m_bv, bvFilename);
m_rankSupport = sdsl::rank_support_il<1>(&m_bv);
}
}

cerr << "Bit Vector Size: " << m_bv.size() << endl;
cerr << "Popcount: " << getPop() << endl;
m_probSaturated = pow(double(getPopSaturated()) / double(getPop()), m_hashNum);
}


void store(string const& filterFilePath) const
{

#pragma omp parallel for
for (unsigned i = 0; i < 2; ++i) {
if (i == 0) {
ofstream myFile(filterFilePath.c_str(), ios::out | ios::binary);

assert(myFile);
writeHeader(myFile);


myFile.write(reinterpret_cast<char*>(m_data), m_dSize * sizeof(T));

myFile.close();
assert(myFile);

FILE* file = fopen(filterFilePath.c_str(), "rb");
if (file == NULL) {
cerr << "file \"" << filterFilePath << "\" could not be read." << endl;
exit(1);
}
} else {
string bvFilename = filterFilePath + ".sdsl";
store_to_file(m_bv, bvFilename);
}
}
}


bool insert(const uint64_t* hashes, const bool* strand, T val, unsigned max)
{
unsigned count = 0;
std::vector<unsigned> hashOrder;
bool saturated = true;
uint64_t randValue = val;
bool strandDir = max % 2;

for (unsigned i = 0; i < m_hashNum; ++i) {
uint64_t pos = m_rankSupport(hashes[i] % m_bv.size());
T value = strandDir ^ strand[i] ? val | s_strand : val;
T oldVal = m_data[pos];
if (oldVal > s_mask) {
oldVal = oldVal & s_antiMask;
} else {
saturated = false;
}
if (oldVal == value) {
++count;
} else {
hashOrder.push_back(i);
}
if (count >= max) {
return true;
}
randValue ^= hashes[i];
}
std::minstd_rand g(randValue);
std::shuffle(hashOrder.begin(), hashOrder.end(), g);

for (std::vector<unsigned>::iterator itr = hashOrder.begin(); itr != hashOrder.end();
++itr) {
uint64_t pos = m_rankSupport(hashes[*itr] % m_bv.size());
T value = strandDir ^ strand[*itr] ? val | s_strand : val;
T oldVal = setVal(&m_data[pos], value);
if (oldVal > s_mask) {
oldVal = oldVal & s_antiMask;
} else {
saturated = false;
}
if (oldVal == 0) {
++count;
}
if (count >= max) {
return true;
}
}
if (count == 0) {
if (!saturated) {
assert(max == 1); 
saturate(hashes);
}
return false;
}
return true;
}


bool insert(const uint64_t* hashes, T value, unsigned max)
{
unsigned count = 0;
std::vector<unsigned> hashOrder;
uint64_t randValue = value;

bool saturated = true;

for (unsigned i = 0; i < m_hashNum; ++i) {
uint64_t pos = m_rankSupport(hashes[i] % m_bv.size());
T oldVal = m_data[pos];
if (oldVal > s_mask) {
oldVal = oldVal & s_antiMask;
} else {
saturated = false;
}
if (oldVal == value) {
++count;
} else {
hashOrder.push_back(i);
}
if (count >= max) {
return true;
}
randValue ^= hashes[i];
}
std::minstd_rand g(randValue);
std::shuffle(hashOrder.begin(), hashOrder.end(), g);

for (std::vector<unsigned>::iterator itr = hashOrder.begin(); itr != hashOrder.end();
++itr) {
uint64_t pos = m_rankSupport(hashes[*itr] % m_bv.size());
T oldVal = setVal(&m_data[pos], value);
if (oldVal > s_mask) {
oldVal = oldVal & s_antiMask;
} else {
saturated = false;
}
if (oldVal == 0) {
++count;
}
if (count >= max) {
return true;
}
}
if (count == 0) {
if (!saturated) {
assert(max == 1); 
saturate(hashes);
}
return false;
}
return true;
}

void saturate(const uint64_t* hashes)
{
for (unsigned i = 0; i < m_hashNum; ++i) {
uint64_t pos = m_rankSupport(hashes[i] % m_bv.size());
__sync_or_and_fetch(&m_data[pos], s_mask);
}
}

inline vector<T> at(const uint64_t* hashes, bool& saturated, unsigned maxMiss = 0)
{
vector<T> results(m_hashNum);
unsigned misses = 0;
for (unsigned i = 0; i < m_hashNum; ++i) {
uint64_t pos = hashes[i] % m_bv.size();
if (m_bv[pos] == 0) {
++misses;
saturated = false;
if (misses > maxMiss) {
return vector<T>();
}
} else {
uint64_t rankPos = m_rankSupport(pos);
T tempResult = m_data[rankPos];
if (tempResult > s_mask) {
results[i] = m_data[rankPos] & s_antiMask;
} else {
results[i] = m_data[rankPos];
saturated = false;
}
}
}
return results;
}


unsigned atRank(
const uint64_t* hashes,
vector<uint64_t>& rankPos,
vector<bool>& hits,
unsigned maxMiss) const
{
unsigned misses = 0;
for (unsigned i = 0; i < m_hashNum; ++i) {
uint64_t pos = hashes[i] % m_bv.size();
if (m_bv[pos]) {
rankPos[i] = m_rankSupport(pos);
hits[i] = true;
} else {
if (++misses > maxMiss) {
return misses;
}
hits[i] = false;
}
}
return misses;
}


bool atRank(const uint64_t* hashes, vector<uint64_t>& rankPos) const
{
for (unsigned i = 0; i < m_hashNum; ++i) {
uint64_t pos = hashes[i] % m_bv.size();
if (m_bv[pos]) {
rankPos[i] = m_rankSupport(pos);
} else {
return false;
}
}
return true;
}

vector<uint64_t> getRankPos(const uint64_t* hashes) const
{
vector<uint64_t> rankPos(m_hashNum);
for (unsigned i = 0; i < m_hashNum; ++i) {
uint64_t pos = hashes[i] % m_bv.size();
rankPos[i] = m_rankSupport(pos);
}
return rankPos;
}

uint64_t getRankPos(const uint64_t hash) const { return m_rankSupport(hash % m_bv.size()); }

const vector<vector<unsigned>>& getSeedValues() const { return m_ssVal; }

unsigned getKmerSize() const { return m_kmerSize; }

unsigned getHashNum() const { return m_hashNum; }


size_t getIDCounts(vector<size_t>& counts) const
{
size_t saturatedCounts = 0;
for (size_t i = 0; i < m_dSize; ++i) {
if (m_data[i] > s_mask) {
++counts[m_data[i] & s_antiMask];
++saturatedCounts;
} else {
++counts[m_data[i]];
}
}
return saturatedCounts;
}


size_t getIDCountsStrand(vector<size_t>& counts) const
{
size_t saturatedCounts = 0;
for (size_t i = 0; i < m_dSize; ++i) {
if (m_data[i] > s_mask) {
++counts[m_data[i] & s_idMask];
++saturatedCounts;
} else {
++counts[m_data[i] & s_antiStrand];
}
}
return saturatedCounts;
}

size_t getPop() const
{
size_t index = m_bv.size() - 1;
while (m_bv[index] == 0) {
--index;
}
return m_rankSupport(index) + 1;
}


size_t getPopNonZero() const
{
size_t count = 0;
for (size_t i = 0; i < m_dSize; ++i) {
if (m_data[i] != 0) {
++count;
}
}
return count;
}


T checkValues(T maxVal) const
{
for (size_t i = 0; i < m_dSize; ++i) {
if ((m_data[i] & s_antiMask) > maxVal) {
return m_data[i];
}
}
return maxVal;
}

size_t getPopSaturated() const
{
size_t count = 0;
for (size_t i = 0; i < m_dSize; ++i) {
if (m_data[i] > s_mask) {
++count;
}
}
return count;
}

size_t size() const { return m_bv.size(); }

void setData(uint64_t pos, T id)
{
T oldValue;
do {
oldValue = m_data[pos];
if (oldValue > s_mask) {
id |= s_mask;
}
} while (!__sync_bool_compare_and_swap(&m_data[pos], oldValue, id));
}

void saturateData(uint64_t pos)
{
#pragma omp critical
m_data[pos] |= s_mask;
}

void setDataIfEmpty(uint64_t pos, T id) { setVal(&m_data[pos], id); }

vector<T> getData(const vector<uint64_t>& rankPos) const
{
vector<T> results(rankPos.size());
for (unsigned i = 0; i < m_hashNum; ++i) {
results[i] = m_data[rankPos[i]];
}
return results;
}

T getData(uint64_t rank) const { return m_data[rank]; }


double calcFrameProbs(vector<double>& frameProbs, unsigned allowedMiss)
{
double occupancy = double(getPop()) / double(size());
vector<size_t> countTable = vector<size_t>(frameProbs.size(), 0);
double satProp = double(getIDCounts(countTable));
size_t sum = 0;
for (size_t i = 1; i < countTable.size(); ++i) {
sum += countTable[i];
}
satProp /= double(sum);
for (size_t i = 1; i < countTable.size(); ++i) {
frameProbs[i] = calcProbSingleFrame(
occupancy, m_hashNum, double(countTable[i]) / double(sum), allowedMiss);
}
return satProp;
}


double calcFrameProbsStrand(vector<double>& frameProbs, unsigned allowedMiss)
{
double occupancy = double(getPop()) / double(size());
vector<size_t> countTable = vector<size_t>(frameProbs.size(), 0);
double satProp = double(getIDCountsStrand(countTable));
size_t sum = 0;
for (vector<size_t>::const_iterator itr = countTable.begin(); itr != countTable.end();
++itr) {
sum += *itr;
}
satProp /= double(sum);
#pragma omp parallel for
for (size_t i = 1; i < countTable.size(); ++i) {
frameProbs[i] = calcProbSingleFrame(
occupancy, m_hashNum, double(countTable[i]) / double(sum), allowedMiss);
}
return satProp;
}

~MIBloomFilter() { delete[] m_data; }

private:
static bool sortbysec(const pair<int, int>& a, const pair<int, int>& b)
{
return (a.second < b.second);
}


void writeHeader(ofstream& out) const
{
FileHeader header;
memcpy(header.magic, "MIBLOOMF", 8);

header.hlen = sizeof(struct FileHeader) + m_kmerSize * m_sseeds.size();
header.kmer = m_kmerSize;
header.size = m_dSize;
header.nhash = m_hashNum;
header.version = MIBloomFilter_VERSION;


out.write(reinterpret_cast<char*>(&header), sizeof(struct FileHeader));

for (vector<string>::const_iterator itr = m_sseeds.begin(); itr != m_sseeds.end(); ++itr) {
out.write(itr->c_str(), m_kmerSize);
}
}


inline static unsigned calcOptiHashNum(double fpr) { return unsigned(-log(fpr) / log(2)); }


double calcFPR_numInserted(size_t numEntr) const
{
return pow(
1.0 - pow(1.0 - 1.0 / double(m_bv.size()), double(numEntr) * double(m_hashNum)),
double(m_hashNum));
}


double calcFPR_hashNum(int hashFunctNum) const { return pow(2.0, -hashFunctNum); }


T setVal(T* val, T newVal)
{
T oldValue;
do {
oldValue = *val;
if (oldValue != 0)
break;
} while (!__sync_bool_compare_and_swap(val, oldValue, newVal));
return oldValue;
}

static inline unsigned nChoosek(unsigned n, unsigned k)
{
if (k > n)
return 0;
if (k * 2 > n)
k = n - k;
if (k == 0)
return 1;

int result = n;
for (unsigned i = 2; i <= k; ++i) {
result *= (n - i + 1);
result /= i;
}
return result;
}

size_t m_dSize;

sdsl::bit_vector_il<BLOCKSIZE> m_bv;
T* m_data;
sdsl::rank_support_il<1> m_rankSupport;

unsigned m_hashNum;
unsigned m_kmerSize;

typedef vector<vector<unsigned>> SeedVal;
vector<string> m_sseeds;

double m_probSaturated;
SeedVal m_ssVal;

static const uint32_t MIBloomFilter_VERSION = 1;
};

#endif 
