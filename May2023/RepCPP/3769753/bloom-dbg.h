#ifndef BLOOM_DBG_H
#define BLOOM_DBG_H 1

#include "config.h"

#include "BloomDBG/AssemblyCounters.h"
#include "BloomDBG/AssemblyParams.h"
#include "BloomDBG/BloomIO.h"
#include "BloomDBG/Checkpoint.h"
#include "BloomDBG/MaskedKmer.h"
#include "BloomDBG/RollingBloomDBG.h"
#include "BloomDBG/RollingHash.h"
#include "BloomDBG/RollingHashIterator.h"
#include "Common/Hash.h"
#include "Common/IOUtil.h"
#include "Common/Sequence.h"
#include "Common/Uncompress.h"
#include "Common/UnorderedSet.h"
#include "DataLayer/FastaConcat.h"
#include "DataLayer/FastaReader.h"
#include "Graph/BreadthFirstSearch.h"
#include "Graph/ExtendPath.h"
#include "Graph/Path.h"
#include "vendor/btl_bloomfilter/BloomFilter.hpp"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>

#if _OPENMP
#include <omp.h>
#endif

#if HAVE_GOOGLE_SPARSE_HASH_MAP

#include <google/sparse_hash_set>
typedef google::sparse_hash_set<RollingBloomDBGVertex, hash<RollingBloomDBGVertex>> KmerHash;

#else

#include "Common/UnorderedSet.h"
typedef unordered_set<RollingBloomDBGVertex, hash<RollingBloomDBGVertex>> KmerHash;

#endif

namespace BloomDBG {


typedef RollingBloomDBGVertex Vertex;


template<typename BloomT>
inline static bool
allKmersInBloom(const Sequence& seq, const BloomT& bloom)
{
const unsigned k = bloom.getKmerSize();
const unsigned numHashes = bloom.getHashNum();
assert(seq.length() >= k);
unsigned validKmers = 0;
for (RollingHashIterator it(seq, numHashes, k); it != RollingHashIterator::end();
++it, ++validKmers) {
if (!bloom.contains(*it))
return false;
}

if (validKmers < seq.length() - k + 1)
return false;
return true;
}


template<typename BloomT>
inline static void
addKmersToBloom(const Sequence& seq, BloomT& bloom)
{
const unsigned k = bloom.getKmerSize();
const unsigned numHashes = bloom.getHashNum();
for (RollingHashIterator it(seq, numHashes, k); it != RollingHashIterator::end(); ++it) {
bloom.insert(*it);
}
}


template<typename CountingBloomT>
inline static unsigned
getSeqAbsoluteKmerCoverage(const Sequence& seq, const CountingBloomT& bloom)
{
const unsigned k = bloom.getKmerSize();
const unsigned numHashes = bloom.getHashNum();
assert(seq.length() >= k);
unsigned coverage = 0;
for (RollingHashIterator it(seq, numHashes, k); it != RollingHashIterator::end();
++it) {
coverage += bloom.minCount(*it);
}
return coverage;
}


inline static Path<Vertex>
seqToPath(const Sequence& seq, unsigned k, unsigned numHashes)
{
Path<Vertex> path;
assert(seq.length() >= k);
for (RollingHashIterator it(seq, numHashes, k); it != RollingHashIterator::end(); ++it) {
path.push_back(Vertex(it.kmer().c_str(), it.rollingHash()));
}
return path;
}


inline static Sequence
pathToSeq(const Path<Vertex>& path, unsigned k)
{
assert(path.size() > 0);
assert(k > 0);

const std::string& spacedSeed = MaskedKmer::mask();
assert(spacedSeed.empty() || spacedSeed.length() == k);
Sequence seq;
seq.resize(path.size() + k - 1, 'N');

for (size_t i = 0; i < path.size(); ++i) {
std::string kmer(path.at(i).kmer().c_str());
for (size_t j = 0; j < k; ++j) {
if (spacedSeed.empty() || spacedSeed.at(j) == '1') {
if (seq.at(i + j) != 'N' && seq.at(i + j) != kmer.at(j)) {
std::cerr << "warning: inconsistent DBG path detected "
"at position "
<< i + j << ": " << seq.substr(0, i + j) << " (orig base: '"
<< seq.at(i + j) << "'"
<< ", new base: '" << kmer.at(j) << "')" << std::endl;
}
seq.at(i + j) = kmer.at(j);
}
}
}

return seq;
}

enum SeedType
{
ST_BRANCH_KMER = 0,
ST_READ
};

static inline std::string
seedTypeStr(const SeedType& type)
{
switch (type) {
case ST_BRANCH_KMER:
return "BRANCH_KMER";
case ST_READ:
return "READ";
default:
break;
}
assert(false);
return "";
}


struct ContigRecord
{

size_t contigID;

unsigned length;

unsigned coverage;

std::string readID;

SeedType seedType;

Sequence seed;

PathExtensionResult leftExtensionResult;

PathExtensionResult rightExtensionResult;


bool redundant;

ContigRecord()
: contigID(std::numeric_limits<size_t>::max())
, readID()
, leftExtensionResult(std::make_pair(0, ER_DEAD_END))
, rightExtensionResult(std::make_pair(0, ER_DEAD_END))
, redundant(false)
{}

static std::ostream& printHeaders(std::ostream& out)
{
out << "contig_id" << '\t' << "length" << '\t' << "redundant" << '\t' << "read_id" << '\t'
<< "left_result" << '\t' << "left_extension" << '\t' << "right_result" << '\t'
<< "right_extension" << '\t' << "seed_type" << '\t' << "seed_length" << '\t' << "seed"
<< '\n';
return out;
}

friend std::ostream& operator<<(std::ostream& out, const ContigRecord& o)
{
if (o.redundant)
out << "NA" << '\t';
else
out << o.contigID << '\t';

out << o.length << '\t' << o.redundant << '\t' << o.readID << '\t';

if (o.leftExtensionResult.first > 0)
out << pathExtensionResultStr(o.leftExtensionResult.second) << '\t'
<< o.leftExtensionResult.first << '\t';
else
out << "NA" << '\t' << "NA" << '\t';

if (o.rightExtensionResult.first > 0)
out << pathExtensionResultStr(o.rightExtensionResult.second) << '\t'
<< o.rightExtensionResult.first << '\t';
else
out << "NA" << '\t' << "NA" << '\t';

out << seedTypeStr(o.seedType) << '\t' << o.seed.length() << '\t' << o.seed << '\n';

return out;
}
};

enum ReadResult
{
RR_UNINITIALIZED = 0,
RR_SHORTER_THAN_K,
RR_NON_ACGT,
RR_BLUNT_END,
RR_NOT_SOLID,
RR_ALL_KMERS_VISITED,
RR_ALL_BRANCH_KMERS_VISITED,
RR_GENERATED_CONTIGS
};

static inline std::string
readResultStr(const ReadResult& result)
{
switch (result) {
case RR_UNINITIALIZED:
return "NA";
case RR_SHORTER_THAN_K:
return "SHORTER_THAN_K";
case RR_NON_ACGT:
return "NON_ACGT";
case RR_BLUNT_END:
return "BLUNT_END";
case RR_NOT_SOLID:
return "NOT_SOLID";
case RR_ALL_KMERS_VISITED:
return "ALL_KMERS_VISITED";
case RR_ALL_BRANCH_KMERS_VISITED:
return "ALL_BRANCH_KMERS_VISITED";
case RR_GENERATED_CONTIGS:
return "GENERATED_CONTIGS";
default:
break;
}
assert(false);
return "";
}


struct ReadRecord
{

std::string readID;

ReadResult result;

ReadRecord()
: readID()
, result(RR_UNINITIALIZED)
{}

ReadRecord(const std::string& readID)
: readID(readID)
, result(RR_UNINITIALIZED)
{}

ReadRecord(const std::string& readID, ReadResult result)
: readID(readID)
, result(result)
{}

static std::ostream& printHeaders(std::ostream& out)
{
out << "read_id" << '\t' << "result" << '\n';
return out;
}

friend std::ostream& operator<<(std::ostream& out, const ReadRecord& o)
{
out << o.readID << '\t' << readResultStr(o.result) << '\n';

return out;
}
};


void
readsProgressMessage(AssemblyCounters counters)
{
std::cerr << "Processed " << counters.readsProcessed << " reads"
<< ", solid reads: " << counters.solidReads << " (" << std::setprecision(3)
<< (float)100 * counters.solidReads / counters.readsProcessed << "%)"
<< ", visited reads: " << counters.visitedReads << " (" << std::setprecision(3)
<< (float)100 * counters.visitedReads / counters.readsProcessed << "%)" << std::endl;
}


void
basesProgressMessage(AssemblyCounters counters)
{
std::cerr << "Assembled " << counters.basesAssembled << " bp in " << counters.contigID + 1
<< " contigs" << std::endl;
}


template<typename GraphT>
inline static std::vector<Path<typename boost::graph_traits<GraphT>::vertex_descriptor>>
splitPath(
const Path<typename boost::graph_traits<GraphT>::vertex_descriptor>& path,
const GraphT& dbg,
unsigned minBranchLen)
{
assert(path.size() > 0);

typedef typename boost::graph_traits<GraphT>::vertex_descriptor V;
typedef typename Path<V>::const_iterator PathIt;

std::vector<Path<V>> splitPaths;
Path<V> currentPath;
for (PathIt it = path.begin(); it != path.end(); ++it) {
currentPath.push_back(*it);
unsigned inDegree = trueBranches(*it, REVERSE, dbg, minBranchLen).size();
unsigned outDegree = trueBranches(*it, FORWARD, dbg, minBranchLen).size();
if (inDegree > 1 || outDegree > 1) {

splitPaths.push_back(currentPath);
currentPath.clear();
currentPath.push_back(*it);
}
}
if (currentPath.size() > 1 || splitPaths.empty())
splitPaths.push_back(currentPath);

assert(splitPaths.size() >= 1);
return splitPaths;
}


template<typename BloomT>
static inline void
trimSeq(Sequence& seq, const BloomT& goodKmerSet)
{
const unsigned k = goodKmerSet.getKmerSize();
const unsigned numHashes = goodKmerSet.getHashNum();

if (seq.length() < k) {
seq.clear();
return;
}

const unsigned UNSET = UINT_MAX;
unsigned prevPos = UNSET;
unsigned matchStart = UNSET;
unsigned matchLen = 0;
unsigned maxMatchStart = UNSET;
unsigned maxMatchLen = 0;


for (RollingHashIterator it(seq, numHashes, k); it != RollingHashIterator::end();
prevPos = it.pos(), ++it) {
if (!goodKmerSet.contains(*it) || (prevPos != UNSET && it.pos() - prevPos > 1)) {

if (matchStart != UNSET && matchLen > maxMatchLen) {
maxMatchLen = matchLen;
maxMatchStart = matchStart;
}
matchStart = UNSET;
matchLen = 0;
}
if (goodKmerSet.contains(*it)) {

if (matchStart == UNSET)
matchStart = it.pos();
matchLen++;
}
}

if (matchStart != UNSET && matchLen > maxMatchLen) {
maxMatchLen = matchLen;
maxMatchStart = matchStart;
}

if (maxMatchLen == 0) {
seq.clear();
return;
}

seq = seq.substr(maxMatchStart, maxMatchLen + k - 1);
}


inline static void
printContig(
const Sequence& seq,
unsigned length,
unsigned coverage,
size_t contigID,
const std::string& readID,
unsigned k,
std::ostream& out)
{
assert(seq.length() >= k); (void)k;

FastaRecord contig;


std::ostringstream id;
id << contigID;


std::ostringstream comment;
comment << length << ' ' << coverage << ' ';
comment << "read:" << readID;
assert(id.good());
contig.id = id.str();
contig.comment = comment.str();


contig.seq = seq;


out << contig;
assert(out);
}


template<typename GraphT>
inline static unsigned
leftIsBluntEnd(const Sequence& seq, const GraphT& graph, const AssemblyParams& params)
{
unsigned k = params.k;
unsigned numHashes = params.numHashes;
unsigned fpLookAhead = 5;

if (seq.length() < k)
return false;

Sequence firstKmerStr(seq, 0, k);
Path<Vertex> path = seqToPath(firstKmerStr, k, numHashes);
Vertex& firstKmer = path.front();

return !lookAhead(firstKmer, REVERSE, fpLookAhead, graph);
}


template<typename GraphT>
inline static bool
hasBluntEnd(const Sequence& seq, const GraphT& graph, const AssemblyParams& params)
{
if (leftIsBluntEnd(seq, graph, params))
return true;

Sequence rc = reverseComplement(seq);
if (leftIsBluntEnd(rc, graph, params))
return true;

return false;
}


template<typename SolidKmerSetT, typename AssembledKmerSetT, typename AssemblyStreamsT>
inline static void
outputContig(
const Path<Vertex>& contigPath,
ContigRecord& rec,
SolidKmerSetT& solidKmerSet,
AssembledKmerSetT& assembledKmerSet,
KmerHash& contigEndKmers,
const AssemblyParams& params,
AssemblyCounters& counters,
AssemblyStreamsT& streams)
{
const unsigned fpLookAhead = 5;

Sequence seq = pathToSeq(contigPath, params.k);



Sequence kmer1 = seq.substr(0, params.k);
canonicalize(kmer1);
RollingHash hash1(kmer1.c_str(), params.numHashes, params.k);
Vertex v1(kmer1.c_str(), hash1);

Sequence kmer2 = seq.substr(seq.length() - params.k);
canonicalize(kmer2);
RollingHash hash2(kmer2.c_str(), params.numHashes, params.k);
Vertex v2(kmer2.c_str(), hash2);

bool redundant = false;
#pragma omp critical(redundancyCheck)
{

if (seq.length() < params.k + fpLookAhead - 1) {

if (contigEndKmers.find(v1) != contigEndKmers.end() &&
contigEndKmers.find(v2) != contigEndKmers.end()) {
redundant = true;
} else {
contigEndKmers.insert(v1);
contigEndKmers.insert(v2);
}

} else if (allKmersInBloom(seq, assembledKmerSet)) {
redundant = true;
}

if (!redundant) {


addKmersToBloom(seq, assembledKmerSet);
}
}
rec.redundant = redundant;

if (!redundant) {
#pragma omp critical(fasta)
{
rec.length = seq.length();
rec.coverage = getSeqAbsoluteKmerCoverage(seq, solidKmerSet);


printContig(seq, rec.length, rec.coverage, counters.contigID, rec.readID, params.k, streams.out);


if (params.checkpointsEnabled())
printContig(seq, rec.length, rec.coverage, counters.contigID, rec.readID, params.k, streams.checkpointOut);

rec.contigID = counters.contigID;

counters.contigID++;
counters.basesAssembled += seq.length();
}
}

#pragma omp critical(trace)
streams.traceOut << rec;
}

enum ContigType
{
CT_LINEAR,
CT_CIRCULAR,
CT_HAIRPIN
};

template<typename GraphT>
static inline ContigType
getContigType(const Path<Vertex>& contigPath, const GraphT& dbg)
{
if (edge(contigPath.back(), contigPath.front(), dbg).second) {

Vertex v = contigPath.front().clone();
v.shift(ANTISENSE, contigPath.back().kmer().getBase(0));

if (v.kmer() == contigPath.back().kmer())
return CT_CIRCULAR;
else
return CT_HAIRPIN;
}

return CT_LINEAR;
}


template<typename GraphT>
static inline void
preprocessCircularContig(Path<Vertex>& contigPath, const GraphT& dbg, unsigned trim)
{
assert(!contigPath.empty());

ContigType contigType = getContigType(contigPath, dbg);
assert(contigType != CT_LINEAR);

if (contigPath.size() <= 2)
return;


const unsigned fpTrim = 5;



bool branchStart = ambiguous(contigPath.front(), FORWARD, dbg, trim, fpTrim) ||
ambiguous(contigPath.front(), REVERSE, dbg, trim, fpTrim);

bool branchEnd = ambiguous(contigPath.back(), FORWARD, dbg, trim, fpTrim) ||
ambiguous(contigPath.back(), REVERSE, dbg, trim, fpTrim);

if (branchStart && !branchEnd) {

if (contigType == CT_CIRCULAR) {
contigPath.push_back(contigPath.front());
} else {
assert(contigType == CT_HAIRPIN);
Vertex rc = contigPath.front().clone();
rc.reverseComplement();
contigPath.push_back(rc);
}

} else if (!branchStart && branchEnd) {

if (contigType == CT_CIRCULAR) {
contigPath.push_front(contigPath.back());
} else {
assert(contigType == CT_HAIRPIN);
Vertex rc = contigPath.back().clone();
rc.reverseComplement();
contigPath.push_front(rc);
}
}
}


template<typename GraphT>
inline static void
trimBranchKmers(Path<Vertex>& contigPath, const GraphT& dbg, unsigned trim)
{
assert(!contigPath.empty());

if (contigPath.size() == 1)
return;



ContigType contigType = getContigType(contigPath, dbg);
if (contigType == CT_CIRCULAR || contigType == CT_HAIRPIN)
preprocessCircularContig(contigPath, dbg, trim);

unsigned l = contigPath.size();


const unsigned fpTrim = 5;

bool ambiguous1 = ambiguous(contigPath.at(0), contigPath.at(1), FORWARD, dbg, trim, fpTrim);

bool ambiguous2 =
ambiguous(contigPath.at(l - 1), contigPath.at(l - 2), REVERSE, dbg, trim, fpTrim);

if (ambiguous1)
contigPath.pop_front();

assert(!contigPath.empty());

if (ambiguous2)
contigPath.pop_back();

assert(!contigPath.empty());
}

bool
isTip(
unsigned length,
PathExtensionResultCode leftResult,
PathExtensionResultCode rightResult,
unsigned trim)
{
if (length > trim)
return false;

if (leftResult == ER_DEAD_END && (rightResult == ER_DEAD_END || rightResult == ER_AMBI_IN))
return true;

if (rightResult == ER_DEAD_END && (leftResult == ER_DEAD_END || leftResult == ER_AMBI_IN))
return true;

return false;
}


template<typename SolidKmerSetT, typename AssembledKmerSetT, typename AssemblyStreamsT>
static inline ReadRecord
processRead(
const FastaRecord& rec,
const SolidKmerSetT& solidKmerSet,
AssembledKmerSetT& assembledKmerSet,
KmerHash& contigEndKmers,
KmerHash& visitedBranchKmers,
const AssemblyParams& params,
AssemblyCounters& counters,
AssemblyStreamsT& streams)
{
(void)visitedBranchKmers;

typedef typename Path<Vertex>::iterator PathIt;


RollingBloomDBG<SolidKmerSetT> dbg(solidKmerSet);

unsigned k = params.k;
const Sequence& seq = rec.seq;


if (seq.length() < k)
return ReadRecord(rec.id, RR_SHORTER_THAN_K);


if (!allACGT(seq))
return ReadRecord(rec.id, RR_NON_ACGT);


if (hasBluntEnd(seq, dbg, params))
return ReadRecord(rec.id, RR_BLUNT_END);


if (!allKmersInBloom(seq, solidKmerSet))
return ReadRecord(rec.id, RR_NOT_SOLID);

#pragma omp atomic
counters.solidReads++;


if (allKmersInBloom(seq, assembledKmerSet)) {
#pragma omp atomic
counters.visitedReads++;
return ReadRecord(rec.id, RR_ALL_KMERS_VISITED);
}


unordered_set<Vertex> assembledKmers;

Path<Vertex> path = seqToPath(rec.seq, params.k, params.numHashes);
for (PathIt it = path.begin(); it != path.end(); ++it) {

if (assembledKmers.find(*it) != assembledKmers.end())
continue;

ExtendPathParams extendParams;
extendParams.trimLen = params.trim;
extendParams.fpTrim = 5;
extendParams.maxLen = NO_LIMIT;
extendParams.lookBehind = true;
extendParams.lookBehindStartVertex = false;

ContigRecord contigRec;
contigRec.readID = rec.id;
contigRec.seedType = ST_READ;
contigRec.seed = it->kmer().c_str();

Path<Vertex> contigPath;
contigPath.push_back(*it);

contigRec.leftExtensionResult = extendPath(contigPath, REVERSE, dbg, extendParams);

contigRec.rightExtensionResult = extendPath(contigPath, FORWARD, dbg, extendParams);

PathExtensionResultCode leftResult = contigRec.leftExtensionResult.second;
PathExtensionResultCode rightResult = contigRec.rightExtensionResult.second;

if (!isTip(contigPath.size(), leftResult, rightResult, params.trim)) {

trimBranchKmers(contigPath, dbg, params.trim);


outputContig(
contigPath, contigRec, solidKmerSet, assembledKmerSet, contigEndKmers, params, counters, streams);
}


for (PathIt it2 = contigPath.begin(); it2 != contigPath.end(); ++it2)
assembledKmers.insert(*it2);
}

return ReadRecord(rec.id, RR_GENERATED_CONTIGS);
}


template<typename SolidKmerSetT>
inline static void
assemble(
int argc,
char** argv,
SolidKmerSetT& solidKmerSet,
const AssemblyParams& params,
std::ostream& out)
{

BloomFilter assembledKmerSet(
solidKmerSet.size(), solidKmerSet.getHashNum(), solidKmerSet.getKmerSize());


AssemblyCounters counters;


FastaConcat in(argv, argv + argc, FastaReader::FOLD_CASE);


std::ofstream checkpointOut;
if (params.checkpointsEnabled()) {
assert(!params.checkpointPathPrefix.empty());
std::string path = params.checkpointPathPrefix + CHECKPOINT_FASTA_EXT + CHECKPOINT_TMP_EXT;
checkpointOut.open(path.c_str());
assert_good(checkpointOut, path);
}


std::ofstream traceOut;
if (!params.tracePath.empty()) {
traceOut.open(params.tracePath.c_str());
assert_good(traceOut, params.tracePath);
ContigRecord::printHeaders(traceOut);
assert_good(traceOut, params.tracePath);
}


std::ofstream readLogOut;
if (!params.readLogPath.empty()) {
readLogOut.open(params.readLogPath.c_str());
assert_good(readLogOut, params.readLogPath);
ReadRecord::printHeaders(readLogOut);
assert_good(readLogOut, params.readLogPath);
}


AssemblyStreams<FastaConcat> streams(in, out, checkpointOut, traceOut, readLogOut);


assemble(solidKmerSet, assembledKmerSet, counters, params, streams);
}


template<typename SolidKmerSetT, typename AssembledKmerSetT, typename InputReadStreamT>
inline static void
assemble(
const SolidKmerSetT& goodKmerSet,
AssembledKmerSetT& assembledKmerSet,
AssemblyCounters& counters,
const AssemblyParams& params,
AssemblyStreams<InputReadStreamT>& streams)
{
assert(params.initialized());


const size_t SEQ_BUFFER_SIZE = 1000000;

if (params.verbose)
std::cerr << "Trimming branches " << params.trim << " k-mers or shorter" << std::endl;

InputReadStreamT& in = streams.in;
std::ostream& checkpointOut = streams.checkpointOut;

KmerHash contigEndKmers;
contigEndKmers.rehash((size_t)pow(2, 28));

KmerHash visitedBranchKmers;



const size_t READS_PROGRESS_STEP = 100000;
const size_t BASES_PROGRESS_STEP = 1000000;
size_t basesProgressLine = BASES_PROGRESS_STEP;

while (true) {
size_t readsUntilCheckpoint = params.readsPerCheckpoint;

#pragma omp parallel
for (std::vector<FastaRecord> buffer;;) {

buffer.clear();
size_t bufferSize;
bool good = true;
#pragma omp critical(in)
for (bufferSize = 0; bufferSize < SEQ_BUFFER_SIZE && readsUntilCheckpoint > 0;) {
FastaRecord rec;
good = in >> rec;
if (!good)
break;
#pragma omp atomic
readsUntilCheckpoint--;
buffer.push_back(rec);
bufferSize += rec.seq.length();
}
if (buffer.size() == 0)
break;

for (std::vector<FastaRecord>::iterator it = buffer.begin(); it != buffer.end(); ++it) {
ReadRecord result = processRead(
*it,
goodKmerSet,
assembledKmerSet,
contigEndKmers,
visitedBranchKmers,
params,
counters,
streams);

#pragma omp critical(readsProgress)
{
++counters.readsProcessed;
if (params.verbose && counters.readsProcessed % READS_PROGRESS_STEP == 0)
readsProgressMessage(counters);
}

if (params.verbose)
#pragma omp critical(basesProgress)
{
if (counters.basesAssembled >= basesProgressLine) {
basesProgressMessage(counters);
while (counters.basesAssembled >= basesProgressLine)
basesProgressLine += BASES_PROGRESS_STEP;
}
}

if (!params.readLogPath.empty()) {
#pragma omp critical(readLog)
streams.readLogOut << result;
}
}

} 

if (readsUntilCheckpoint > 0) {
assert(in.eof());
break;
}


checkpointOut.flush();
createCheckpoint(goodKmerSet, assembledKmerSet, counters, params);

} 

assert(in.eof());

if (params.verbose) {
readsProgressMessage(counters);
basesProgressMessage(counters);
std::cerr << "Assembly complete" << std::endl;
}

if (params.checkpointsEnabled() && !params.keepCheckpoint)
removeCheckpointData(params);
}


template<typename GraphT>
class GraphvizBFSVisitor : public DefaultBFSVisitor<GraphT>
{
typedef typename boost::graph_traits<GraphT>::vertex_descriptor VertexT;
typedef typename boost::graph_traits<GraphT>::edge_descriptor EdgeT;

public:

GraphvizBFSVisitor(std::ostream& out)
: m_out(out)
, m_nodesVisited(0)
, m_edgesVisited(0)
{

m_out << "digraph g {\n";
}


~GraphvizBFSVisitor()
{

m_out << "}\n";
}


BFSVisitorResult discover_vertex(const VertexT& v, const GraphT&)
{
++m_nodesVisited;

m_out << '\t' << v.kmer().c_str() << ";\n";

return BFS_SUCCESS;
}


BFSVisitorResult examine_edge(const EdgeT& e, const GraphT& g)
{
++m_edgesVisited;
const VertexT& u = source(e, g);
const VertexT& v = target(e, g);


m_out << '\t' << u.kmer().c_str() << " -> " << v.kmer().c_str() << ";\n";

return BFS_SUCCESS;
}


size_t getNumNodesVisited() const { return m_nodesVisited; }


size_t getNumEdgesVisited() const { return m_edgesVisited; }

protected:

std::ostream& m_out;

size_t m_nodesVisited;

size_t m_edgesVisited;
};


template<typename BloomT>
static inline void
outputGraph(
int argc,
char** argv,
const BloomT& kmerSet,
const AssemblyParams& params,
std::ostream& out)
{
assert(params.initialized());

typedef RollingBloomDBG<BloomT> GraphT;


const unsigned progressStep = 1000;
const unsigned k = kmerSet.getKmerSize();
const unsigned numHashes = kmerSet.getHashNum();


size_t readsProcessed = 0;


GraphT dbg(kmerSet);


DefaultColorMap<GraphT> colorMap;


GraphvizBFSVisitor<GraphT> visitor(out);

if (params.verbose)
std::cerr << "Generating GraphViz output..." << std::endl;

FastaConcat in(argv, argv + argc, FastaReader::FOLD_CASE);
for (FastaRecord rec;;) {
bool good;
good = in >> rec;
if (!good)
break;
Sequence& seq = rec.seq;


trimSeq(seq, kmerSet);
if (seq.length() > 0) {


std::string startKmer = seq.substr(0, k);
Vertex start(startKmer.c_str(), RollingHash(startKmer, numHashes, k));
breadthFirstSearch(start, dbg, colorMap, visitor);


Sequence rcSeq = reverseComplement(seq);
std::string rcStartKmer = rcSeq.substr(0, k);
Vertex rcStart(rcStartKmer.c_str(), RollingHash(rcStartKmer, numHashes, k));
breadthFirstSearch(rcStart, dbg, colorMap, visitor);
}

if (++readsProcessed % progressStep == 0 && params.verbose) {
std::cerr << "processed " << readsProcessed
<< " (k-mers visited: " << visitor.getNumNodesVisited()
<< ", edges visited: " << visitor.getNumEdgesVisited() << ")" << std::endl;
}
}
assert(in.eof());
if (params.verbose) {
std::cerr << "processed " << readsProcessed
<< " reads (k-mers visited: " << visitor.getNumNodesVisited()
<< ", edges visited: " << visitor.getNumEdgesVisited() << ")" << std::endl;
std::cerr << "GraphViz generation complete" << std::endl;
}
}


static inline void
outputWigBlock(
const std::string& chr,
size_t start,
size_t length,
unsigned val,
ostream& out,
const std::string& outPath)
{
assert(length > 0);
out << "variableStep chrom=" << chr << " span=" << length << "\n";
out << start << ' ' << val << '\n';
assert_good(out, outPath);
}


template<class BloomT>
static inline void
writeCovTrack(const BloomT& goodKmerSet, const AssemblyParams& params)
{
assert(!params.covTrackPath.empty());
assert(!params.refPath.empty());

const unsigned k = goodKmerSet.getKmerSize();
const unsigned numHashes = goodKmerSet.getHashNum();

std::ofstream covTrack(params.covTrackPath.c_str());
assert_good(covTrack, params.covTrackPath);

if (params.verbose)
std::cerr << "Writing 0/1 k-mer coverage track for `" << params.refPath << "` to `"
<< params.covTrackPath << "`" << std::endl;

FastaReader ref(params.refPath.c_str(), FastaReader::FOLD_CASE);
for (FastaRecord rec; ref >> rec;) {
std::string chr = rec.id;
bool firstVal = true;
size_t blockStart = 1;
size_t blockLength = 0;
uint8_t blockVal = 0;
for (RollingHashIterator it(rec.seq, numHashes, k); it != RollingHashIterator::end();
++it) {
uint8_t val = goodKmerSet.contains(*it) ? 1 : 0;
if (firstVal) {
firstVal = false;

blockStart = it.pos() + 1;
blockLength = 1;
blockVal = val;
} else if (val != blockVal) {
assert(firstVal == false);
outputWigBlock(
chr, blockStart, blockLength, blockVal, covTrack, params.covTrackPath);

blockStart = it.pos() + 1;
blockLength = 1;
blockVal = val;
} else {
blockLength++;
}
}

if (blockLength > 0) {
outputWigBlock(chr, blockStart, blockLength, blockVal, covTrack, params.covTrackPath);
}
}
assert(ref.eof());

assert_good(covTrack, params.covTrackPath);
covTrack.close();
}

} 

#endif
