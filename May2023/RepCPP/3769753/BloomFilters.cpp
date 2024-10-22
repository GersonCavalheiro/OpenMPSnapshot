#include "BloomFilters.h"
#include "RAlgorithmsShort.h"
#include "RUtils.h"
#include "Common/IOUtil.h"
#include "Common/Options.h"
#include "DataLayer/FastaReader.h"

#include "btllib/seq_reader.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <random>
#include <iomanip>

btllib::KmerBloomFilter *g_vanillaBloom = nullptr;
btllib::SeedBloomFilter *g_spacedSeedsBloom = nullptr;

static std::vector<std::string>
generateSpacedSeedsPatterns(const int count, const int size, const int misses)
{
assert(count < size);
assert(misses < count);
assert(misses > 0);

std::vector<std::string> seeds;
auto randomEngine = std::mt19937();
std::vector<int> seedsPermutation;

for (int i = 0; i < count; i++) {
seeds.push_back(std::string(size, '1'));
seedsPermutation.push_back(i);
}

for (int i = 0; i < (size + 1) / 2; i++) {
std::shuffle(seedsPermutation.begin(), seedsPermutation.end(), randomEngine);
for (int j = 0; j < count; j++) {
char c = j < misses ? '0' : '1';
seeds[seedsPermutation[j]][i] = c;
}
if (i < size / 2) {
for (int j = 0; j < count; j++) {
seeds[count - j - 1][size - i - 1] = seeds[j][i];
}
}
}

return seeds;
}

void
QCSpacedSeedsPatterns(const std::vector<std::string>& patterns)
{
int k = patterns[0].size();

for (const auto& pattern : patterns) {
int patternBasesCovered = k;
bool hasZero = false;
for (const char c : pattern) {
if (c == '0') {
hasZero = true;
patternBasesCovered--;
break;
}
}
if (opt::verbose && !hasZero) {
std::cerr << "A spaced seed has no zeros!\n";
}
if (opt::verbose && (patternBasesCovered < k * SPACED_SEEDS_QC_MIN_BASES_PATTERN)) {
std::cerr << "A spaced seed pattern does not cover enough bases!\n";
}
}

const auto combinations = genCombinations(SPACED_SEEDS_COUNT, SPACED_SEEDS_MIN_HITS);
std::string overallErrorCoverage(k, '1');
int overallErrorsCovered = 0;
std::string overall_base_coverage(k, '0');
int overallBasesCovered = 0;
std::string worstCombinationCoverage(k, '1');
int worstCombinationBasesCovered = k;

for (auto combination : combinations) {
std::string combinationCoverage(k, '0');
int combinationBasesCovered = 0;
for (auto index : combination) {
const auto& pattern = patterns[index];
for (unsigned i = 0; i < pattern.size(); i++) {
if (pattern[i] == '1' && combinationCoverage[i] != '1') {
combinationCoverage[i] = '1';
combinationBasesCovered++;
}
}
}
if (combinationBasesCovered < worstCombinationBasesCovered) {
worstCombinationCoverage = combinationCoverage;
worstCombinationBasesCovered = combinationBasesCovered;
}
if (opt::verbose && (combinationBasesCovered <
combinationCoverage.size() * SPACED_SEEDS_QC_MIN_BASES_COMBINATION)) {
std::cerr << "A spaced seed combination does not cover enough bases!\n";
}

for (unsigned i = 0; i < combinationCoverage.size(); i++) {
if (combinationCoverage[i] == '0') {
if (overallErrorCoverage[i] != '0') {
overallErrorCoverage[i] = '0';
overallErrorsCovered++;
}
} else {
if (overall_base_coverage[i] != '1') {
overall_base_coverage[i] = '1';
overallBasesCovered++;
}
}
}
}

if (opt::verbose) {
std::cerr << std::fixed;
std::cerr << "Worst combination coverage: " << worstCombinationCoverage << '\n';
std::cerr << "Worst combination bases covered: "
<< worstCombinationBasesCovered / double(k) * 100.0 << "%\n";
std::cerr << "Overall base coverage:\n" << overall_base_coverage << '\n';
std::cerr << "Bases covered: " << overallBasesCovered / double(k) * 100.0 << "%\n";
std::cerr << "Overall error coverage:\n" << overallErrorCoverage << '\n';
std::cerr << "Errors covered: " << overallErrorsCovered / double(k) * 100.0 << "%\n";
std::cerr << std::defaultfloat << std::endl;
}

if (opt::verbose && (overallErrorsCovered < k * SPACED_SEEDS_QC_MIN_ERRORS_OVERALL)) {
std::cerr << "Spaced seeds do not cover enough error positions!\n\n";
}
if (opt::verbose && (overallBasesCovered < k * SPACED_SEEDS_QC_MIN_BASES_OVERALL)) {
std::cerr << "Spaced seeds do not cover enough base positions!\n\n";
}
std::cerr << std::flush;
}

static void
loadReads(const std::vector<std::string>& readFilepaths, int r)
{
ReadSize::readsSampleSize = 0;
ReadSize::current.sampleCount = 0;
int i = 0;
for (auto& b : ReadSize::readSizes) {
b.sampleCount = 0;
i++;
}

for (const auto& path : readFilepaths) {
uint64_t currentReadCount = 0;
uint64_t totalReadCount = 0;
size_t readSizesNum = ReadSize::readSizes.size();
uint64_t *readSizeCounts = new uint64_t[readSizesNum];
for (size_t i = 0; i < readSizesNum; i++) {
readSizeCounts[i] = 0;
}

assert(!path.empty());
if (opt::verbose) {
std::cerr << "Loading reads from `" << path << "'..." << std::endl;;
}

btllib::SeqReader reader(path, btllib::SeqReader::Flag::SHORT_MODE);
#pragma omp parallel reduction(+:currentReadCount,totalReadCount,readSizeCounts[:readSizesNum])
for (btllib::SeqReader::Record record; (record = reader.read());) {
totalReadCount++;
int i = 0;
for (const auto& b : ReadSize::readSizes) {
if (b.sizeAndMergedSizes.find(int(record.seq.size())) != b.sizeAndMergedSizes.end()) {
readSizeCounts[i]++;
break;
}
i++;
}
if (ReadSize::current.sizeAndMergedSizes.find(int(record.seq.size())) == ReadSize::current.sizeAndMergedSizes.end()) {
continue;
}
currentReadCount++;

std::string seq = record.seq.substr(0, r + opt::extract - 1);
if (seq.size() >= g_vanillaBloom->get_k()) {
g_vanillaBloom->insert(seq);
if (opt::errorCorrection) {
g_spacedSeedsBloom->insert(seq);
}
}
}

ReadSize::readsSampleSize += totalReadCount;
ReadSize::current.sampleCount += currentReadCount;
int i = 0;
for (auto& b : ReadSize::readSizes) {
b.sampleCount += readSizeCounts[i];
i++;
}
}

if (opt::verbose) {
std::cerr << "\nUpdated read lengths' fractions determined to be: " << std::fixed;
std::cerr << ReadSize::readSizes[0].size << " ("
<< (ReadSize::readSizes[0].getFractionOfTotal() * 100.0) << "%)";
for (size_t i = 1; i < ReadSize::readSizes.size(); i++) {
std::cerr << ", " << ReadSize::readSizes[i].size << " ("
<< (ReadSize::readSizes[i].getFractionOfTotal() * 100.0) << "%)";
}
std::cerr << std::defaultfloat << std::endl;
}
}

void
buildFilters(
const std::vector<std::string>& readFilepaths,
const int r,
const size_t bloomBytesTotal)
{
assert(bloomBytesTotal > 0);
try {
if (opt::verbose) {
std::cerr << "Building Bloom filter(s) for r value " << r << '\n';
}

delete g_vanillaBloom;
delete g_spacedSeedsBloom;

size_t bloomBytesVanilla = size_t(bloomBytesTotal);
size_t bloomBytesSpacedSeeds = 0;

if (opt::errorCorrection) {
double vanillaRatio =
VANILLA_TO_SEEDS_MEM_RATIO * double(HASH_NUM) /
(double(HASH_NUM) + double(SPACED_SEEDS_COUNT * SPACED_SEEDS_HASHES_PER_SEED));
bloomBytesVanilla = size_t(vanillaRatio * bloomBytesTotal);
bloomBytesSpacedSeeds = bloomBytesTotal - bloomBytesVanilla;
}

if (opt::verbose > 1) {
if (opt::errorCorrection) {
std::cerr << "Total Bloom filter memory = " << bytesToSI(bloomBytesTotal) << '\n';
std::cerr << "Vanilla Bloom filter memory = " << bytesToSI(bloomBytesVanilla)
<< '\n';
std::cerr << "Spaced seeds Bloom filter memory = "
<< bytesToSI(bloomBytesSpacedSeeds) << '\n';
} else {
std::cerr << "Vanilla Bloom filter memory = " << bytesToSI(bloomBytesVanilla)
<< '\n';
}
}

g_vanillaBloom = new btllib::KmerBloomFilter(bloomBytesVanilla, HASH_NUM, r);
if (opt::errorCorrection) {
const auto patterns =
generateSpacedSeedsPatterns(SPACED_SEEDS_COUNT, r, SPACED_SEEDS_MISSES);
for (const auto& pattern : patterns) {
(void)pattern;
assert(pattern.size() == size_t(r));
}
if (SPACED_SEEDS_QC) {
QCSpacedSeedsPatterns(patterns);
}
g_spacedSeedsBloom = new btllib::SeedBloomFilter(bloomBytesSpacedSeeds, r, patterns, SPACED_SEEDS_HASHES_PER_SEED);
}

loadReads(readFilepaths, r);

if (opt::verbose > 1) {
const auto vanillaFPR = g_vanillaBloom->get_fpr();

std::cerr << "Vanilla Bloom filter (k = "
<< std::to_string(g_vanillaBloom->get_k()) << std::setprecision(3)
<< ") occupancy = "
<< g_vanillaBloom->get_occupancy() * 100.0
<< "%"
<< ", FPR = " << vanillaFPR * 100.0 << "%" << std::endl;

if (opt::errorCorrection) {
std::cerr << "FPR for base substitution = "
<< (1 - std::pow(
1 - vanillaFPR,
3 * g_vanillaBloom->get_k() / SPACED_SEEDS_COUNT *
SPACED_SEEDS_SNP_FRACTION)) *
100.0
<< "%" << std::endl;

std::cerr << "Spaced seeds Bloom filter (k = "
<< std::to_string(g_spacedSeedsBloom->get_k()) << std::setprecision(3)
<< ") occupancy = "
<< g_spacedSeedsBloom->get_occupancy() * 100.0
<< "%"
<< ", FPR = " << g_spacedSeedsBloom->get_fpr() * 100.0 << "%" << std::endl;
}
}
} catch (const std::bad_alloc& e) {
std::cerr << "Bloom filter allocation failed: " << e.what() << '\n';
exit(EXIT_FAILURE);
}
}
