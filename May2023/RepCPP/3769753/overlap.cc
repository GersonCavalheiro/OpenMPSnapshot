#if __APPLE_CC__

#define NDEBUG 1
#endif

#include "BitUtil.h"
#include "ContigProperties.h"
#include "DataLayer/Options.h"
#include "FMIndex.h"
#include "FastaIndex.h"
#include "FastaReader.h"
#include "IOUtil.h"
#include "MemoryUtil.h"
#include "SAM.h"
#include "StringUtil.h"
#include "Uncompress.h"
#include "Graph/ContigGraph.h"
#include "Graph/DirectedGraph.h"
#include "Graph/GraphAlgorithms.h"
#include "Graph/GraphIO.h"
#include "Graph/GraphUtil.h"
#include <algorithm>
#include <cassert>
#include <cctype> 
#include <cstdlib>
#include <getopt.h>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <utility>
#if _OPENMP
# include <omp.h>
#endif

using namespace std;

#define PROGRAM "abyss-overlap"

static const char VERSION_MESSAGE[] =
PROGRAM " (" PACKAGE_NAME ") " VERSION "\n"
"Written by Shaun Jackman.\n"
"\n"
"Copyright 2014 Canada's Michael Smith Genome Sciences Centre\n";

static const char USAGE_MESSAGE[] =
"Usage: " PROGRAM " [OPTION]... FILE\n"
"Find overlaps of [m,k) bases. Sequences are read from FILE.\n"
"Output is written to standard output. The index files FILE.fai\n"
"and FILE.fm will be used if present.\n"
"\n"
" Options:\n"
"\n"
"  -m, --min=N             find matches at least N bp [50]\n"
"  -k, --max=N             find matches less than N bp [inf]\n"
"  -j, --threads=N         use N parallel threads [1]\n"
"  -s, --sample=N          sample the suffix array [1]\n"
"      --tred              remove transitive edges [default]\n"
"      --no-tred           do not remove transitive edges\n"
"      --adj             output the results in adj format\n"
"      --dot             output the results in dot format [default]\n"
"      --sam             output the results in SAM format\n"
"      --SS                expect contigs to be oriented correctly\n"
"      --no-SS             no assumption about contig orientation\n"
"  -v, --verbose           display verbose output\n"
"      --help              display this help and exit\n"
"      --version           output version information and exit\n"
"\n"
"Report bugs to <" PACKAGE_BUGREPORT ">.\n";

namespace opt {

static unsigned minOverlap = 50;


static unsigned maxOverlap = UINT_MAX;


static unsigned sampleSA;


static int ss;


static int tred = true;


static unsigned threads = 1;


static int verbose;

unsigned k; 
int format = DOT; 
}

static const char shortopts[] = "j:k:m:s:v";

enum { OPT_HELP = 1, OPT_VERSION };

static const struct option longopts[] = {
{ "adj", no_argument, &opt::format, ADJ },
{ "dot", no_argument, &opt::format, DOT },
{ "sam", no_argument, &opt::format, SAM },
{ "help", no_argument, NULL, OPT_HELP },
{ "max", required_argument, NULL, 'k' },
{ "min", required_argument, NULL, 'm' },
{ "sample", required_argument, NULL, 's' },
{ "threads", required_argument, NULL, 'j' },
{ "SS", no_argument, &opt::ss, 1 },
{ "no-SS", no_argument, &opt::ss, 0 },
{ "tred", no_argument, &opt::tred, true },
{ "no-tred", no_argument, &opt::tred, false },
{ "verbose", no_argument, NULL, 'v' },
{ "version", no_argument, NULL, OPT_VERSION },
{ NULL, 0, NULL, 0 }
};


typedef DirectedGraph<ContigProperties, Distance> DG;
typedef ContigGraph<DG> Graph;

typedef FMIndex::Match Match;


static void addSuffixOverlaps(Graph &g,
const FastaIndex& faIndex, const FMIndex& fmIndex,
const ContigNode& u, const Match& fmi)
{
typedef edge_property<Graph>::type EP;
typedef graph_traits<Graph>::edge_descriptor E;
typedef graph_traits<Graph>::vertex_descriptor V;

Distance ep(-fmi.qspan());
assert(ep.distance < 0);
for (unsigned i = fmi.l; i < fmi.u; ++i) {
size_t toffset = fmIndex[i] + 1;
FastaIndex::SeqPos seqPos = faIndex[toffset];
const FAIRecord& tseq = seqPos.get<0>();
size_t tstart = seqPos.get<1>();
size_t tend = tstart + fmi.qspan();
assert(tend <= tseq.size);
(void)tend;
if (tstart > 0) {
continue;
}
V v = find_vertex(tseq.id, false, g);
#pragma omp critical(g)
{
pair<E, bool> e = edge(u, v, g);
if (e.second) {
const EP& ep0 = g[e.first];
if (opt::verbose > 1)
cerr << "duplicate edge: "
<< get(edge_name, g, e.first) << ' '
<< ep0 << ' ' << ep << '\n';
assert(ep0.distance < ep.distance);
} else if(u.sense()) {
add_edge(u, v, ep, static_cast<DG&>(g));
} else {
add_edge(u, v, ep, g);
}
}
}
}


static void addPrefixOverlaps(Graph &g,
const FastaIndex& faIndex, const FMIndex& fmIndex,
const ContigNode& v, const Match& fmi)
{
typedef edge_property<Graph>::type EP;
typedef graph_traits<Graph>::edge_descriptor E;
typedef graph_traits<Graph>::vertex_descriptor V;

assert(v.sense());
assert(fmi.qstart == 0);
Distance ep(-fmi.qspan());
assert(ep.distance < 0);
for (unsigned i = fmi.l; i < fmi.u; ++i) {
size_t toffset = fmIndex[i];
FastaIndex::SeqPos seqPos = faIndex[toffset];
const FAIRecord& tseq = seqPos.get<0>();
size_t tstart = seqPos.get<1>();
size_t tend = tstart + fmi.qspan();
assert(tend <= tseq.size);
if (tend < tseq.size) {
continue;
}
V u = find_vertex(tseq.id, false, g);
#pragma omp critical(g)
{
pair<E, bool> e = edge(u, v, g);
if (e.second) {
const EP& ep0 = g[e.first];
if (opt::verbose > 1)
cerr << "duplicate edge: "
<< get(edge_name, g, e.first) << ' '
<< ep0 << ' ' << ep << '\n';
assert(ep0.distance < ep.distance);
} else {
add_edge(u, v, ep, static_cast<DG&>(g));
}
}
}
}


static void findOverlapsSuffix(Graph &g,
const FastaIndex& faIndex, const FMIndex& fmIndex,
const ContigNode& u, const string& seq)
{
size_t pos = seq.size() > opt::maxOverlap
? seq.size() - opt::maxOverlap + 1 : 1;
string suffix(seq, pos);
typedef vector<Match> Matches;
Matches matches;
fmIndex.findOverlapSuffix(suffix, back_inserter(matches),
opt::minOverlap);

for (Matches::reverse_iterator it = matches.rbegin();
!(it == matches.rend()); ++it)
addSuffixOverlaps(g, faIndex, fmIndex, u, *it);
}


static void findOverlapsPrefix(Graph &g,
const FastaIndex& faIndex, const FMIndex& fmIndex,
const ContigNode& v, const string& seq)
{
assert(v.sense());
string prefix(seq, 0,
min((size_t)opt::maxOverlap, seq.size()) - 1);
typedef vector<Match> Matches;
Matches matches;
fmIndex.findOverlapPrefix(prefix, back_inserter(matches),
opt::minOverlap);

for (Matches::reverse_iterator it = matches.rbegin();
!(it == matches.rend()); ++it)
addPrefixOverlaps(g, faIndex, fmIndex, v, *it);
}

static void findOverlaps(Graph& g,
const FastaIndex& faIndex, const FMIndex& fmIndex,
const FastqRecord& rec)
{
typedef graph_traits<Graph>::vertex_descriptor V;
V u = find_vertex(rec.id, false, g);
V uc = get(vertex_complement, g, u);
findOverlapsSuffix(g, faIndex, fmIndex, u, rec.seq);
if (!opt::ss) {
string rcseq = reverseComplement(rec.seq);
findOverlapsSuffix(g, faIndex, fmIndex, uc, rcseq);
findOverlapsPrefix(g, faIndex, fmIndex, uc, rcseq);
}
}


static void findOverlaps(Graph& g,
const FastaIndex& faIndex, const FMIndex& fmIndex,
FastaReader& in)
{
#pragma omp parallel
for (FastqRecord rec;;) {
bool good;
#pragma omp critical(in)
good = in >> rec;
if (good)
findOverlaps(g, faIndex, fmIndex, rec);
else
break;
}
assert(in.eof());
}


static void buildFMIndex(FMIndex& fm, const char* path)
{
if (opt::verbose > 0)
std::cerr << "Reading `" << path << "'...\n";
std::vector<FMIndex::value_type> s;
readFile(path, s);

uint64_t MAX_SIZE = numeric_limits<FMIndex::sais_size_type>::max();
if (s.size() > MAX_SIZE) {
std::cerr << PROGRAM << ": `" << path << "', "
<< toSI(s.size())
<< "B, must be smaller than "
<< toSI(MAX_SIZE) << "B\n";
exit(EXIT_FAILURE);
}

transform(s.begin(), s.end(), s.begin(), ::toupper);
fm.setAlphabet("-ACGT");
fm.assign(s.begin(), s.end());
}


static void addVertices(const string& path, Graph& g)
{
if (opt::verbose > 0)
cerr << "Reading `" << path << "'...\n";
ifstream in(path.c_str());
assert_good(in, path);
in >> g;
g_contigNames.lock();
assert(in.eof());
}


static streampos fileSize(const string& path)
{
std::ifstream in(path.c_str());
assert_good(in, path);
in.seekg(0, std::ios::end);
assert_good(in, path);
return in.tellg();
}


static void checkIndexes(const string& path,
const FMIndex& fmIndex, const FastaIndex& faIndex)
{
size_t fastaFileSize = fileSize(path);
if (fmIndex.size() != fastaFileSize) {
cerr << PROGRAM ": `" << path << "': "
"The size of the FM-index, "
<< fmIndex.size()
<< " B, does not match the size of the FASTA file, "
<< fastaFileSize << " B. The index is likely stale.\n";
exit(EXIT_FAILURE);
}
if (faIndex.fileSize() != fastaFileSize) {
cerr << PROGRAM ": `" << path << "': "
"The size of the FASTA index, "
<< faIndex.fileSize()
<< " B, does not match the size of the FASTA file, "
<< fastaFileSize << " B. The index is likely stale.\n";
exit(EXIT_FAILURE);
}
}

int main(int argc, char** argv)
{
string commandLine;
{
ostringstream ss;
char** last = argv + argc - 1;
copy(argv, last, ostream_iterator<const char *>(ss, " "));
ss << *last;
commandLine = ss.str();
}

bool die = false;
for (int c; (c = getopt_long(argc, argv,
shortopts, longopts, NULL)) != -1;) {
istringstream arg(optarg != NULL ? optarg : "");
switch (c) {
case '?':
die = true;
break;
case 'j':
arg >> opt::threads;
break;
case 'm':
arg >> opt::minOverlap;
break;
case 'k':
arg >> opt::maxOverlap;
break;
case 's':
arg >> opt::sampleSA;
break;
case 'v':
opt::verbose++;
break;
case OPT_HELP:
cout << USAGE_MESSAGE;
exit(EXIT_SUCCESS);
case OPT_VERSION:
cout << VERSION_MESSAGE;
exit(EXIT_SUCCESS);
}
if (optarg != NULL && !arg.eof()) {
cerr << PROGRAM ": invalid option: `-"
<< (char)c << optarg << "'\n";
exit(EXIT_FAILURE);
}
}

if (argc - optind < 1) {
cerr << PROGRAM ": missing arguments\n";
die = true;
} else if (argc - optind > 1) {
cerr << PROGRAM ": too many arguments\n";
die = true;
}

if (die) {
cerr << "Try `" << PROGRAM
<< " --help' for more information.\n";
exit(EXIT_FAILURE);
}

#if _OPENMP
if (opt::threads > 0)
omp_set_num_threads(opt::threads);
#endif

if (opt::minOverlap == 0)
opt::minOverlap = opt::maxOverlap - 1;
opt::minOverlap = min(opt::minOverlap, opt::maxOverlap - 1);
if (opt::maxOverlap != UINT_MAX)
opt::k = opt::maxOverlap;

const char* fastaFile(argv[--argc]);
ostringstream ss;
ss << fastaFile << ".fm";
string fmPath(ss.str());
ss.str("");
ss << fastaFile << ".fai";
string faiPath(ss.str());

ifstream in;

FastaIndex faIndex;
in.open(faiPath.c_str());
if (in) {
if (opt::verbose > 0)
cerr << "Reading `" << faiPath << "'...\n";
in >> faIndex;
assert(in.eof());
in.close();
} else {
if (opt::verbose > 0)
cerr << "Reading `" << fastaFile << "'...\n";
faIndex.index(fastaFile);
}

FMIndex fmIndex;
in.open(fmPath.c_str());
if (in) {
if (opt::verbose > 0)
cerr << "Reading `" << fmPath << "'...\n";
assert_good(in, fmPath);
in >> fmIndex;
assert_good(in, fmPath);
in.close();
} else
buildFMIndex(fmIndex, fastaFile);
if (opt::sampleSA > 1)
fmIndex.sampleSA(opt::sampleSA);

if (opt::verbose > 0) {
size_t bp = fmIndex.size();
cerr << "Read " << toSI(bp) << "B in "
<< faIndex.size() << " contigs.\n";
ssize_t bytes = getMemoryUsage();
if (bytes > 0)
cerr << "Using " << toSI(bytes) << "B of memory and "
<< setprecision(3) << (float)bytes / bp << " B/bp.\n";
}

checkIndexes(fastaFile, fmIndex, faIndex);

opt::chastityFilter = false;
opt::trimMasked = false;

Graph g;
addVertices(fastaFile, g);
FastaReader fa(fastaFile, FastaReader::FOLD_CASE);
findOverlaps(g, faIndex, fmIndex, fa);

if (opt::verbose > 0)
printGraphStats(cerr, g);

if (opt::tred) {
unsigned numTransitive = remove_transitive_edges(g);
if (opt::verbose > 0) {
cerr << "Removed " << numTransitive
<< " transitive edges.\n";
printGraphStats(cerr, g);
}
}

write_graph(cout, g, PROGRAM, commandLine);
cout.flush();
assert_good(cout, "stdout");
return 0;
}
