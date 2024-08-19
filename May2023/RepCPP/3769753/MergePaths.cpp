#include "Common/Options.h"
#include "ContigID.h"
#include "ContigPath.h"
#include "Functional.h" 
#include "Graph/Assemble.h"
#include "Graph/ContigGraph.h"
#include "Graph/DirectedGraph.h"
#include "Graph/DotIO.h"
#include "Graph/GraphAlgorithms.h"
#include "Graph/GraphUtil.h"
#include "IOUtil.h"
#include "Uncompress.h"
#include "config.h"
#include <algorithm>
#include <boost/tuple/tuple.hpp>
#include <cassert>
#include <climits> 
#include <cstdlib>
#include <deque>
#include <fstream>
#include <functional>
#include <getopt.h>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <vector>
#if _OPENMP
#include <omp.h>
#endif
#include "DataBase/DB.h"
#include "DataBase/Options.h"

using namespace std;
using boost::tie;

#define PROGRAM "MergePaths"

DB db;

static const char VERSION_MESSAGE[] =
PROGRAM " (" PACKAGE_NAME ") " VERSION "\n"
"Written by Jared Simpson and Shaun Jackman.\n"
"\n"
"Copyright 2014 Canada's Michael Smith Genome Sciences Centre\n";

static const char USAGE_MESSAGE[] =
"Usage: " PROGRAM " -k<kmer> [OPTION]... LEN PATH\n"
"Merge sequences of contigs IDs.\n"
"\n"
" Arguments:\n"
"\n"
"  LEN   lengths of the contigs\n"
"  PATH  sequences of contig IDs\n"
"\n"
" Options:\n"
"\n"
"  -k, --kmer=KMER_SIZE  k-mer size\n"
"  -s, --seed-length=L   minimum length of a seed contig [0]\n"
"  -G, --genome-size=N   expected genome size. Used to calculate NG50\n"
"                        and associated stats [disabled]\n"
"  -o, --out=FILE        write result to FILE\n"
"      --no-greedy       use the non-greedy algorithm [default]\n"
"      --greedy          use the greedy algorithm\n"
"  -g, --graph=FILE      write the path overlap graph to FILE\n"
"  -j, --threads=N       use N parallel threads [1]\n"
"  -v, --verbose         display verbose output\n"
"      --help            display this help and exit\n"
"      --version         output version information and exit\n"
"      --db=FILE         specify path of database repository in FILE\n"
"      --library=NAME    specify library NAME for database\n"
"      --strain=NAME     specify strain NAME for database\n"
"      --species=NAME    specify species NAME for database\n"
"\n"
"Report bugs to <" PACKAGE_BUGREPORT ">.\n";

namespace opt {
string db;
dbVars metaVars;
unsigned k; 
static string out;
static int threads = 1;


static unsigned seedLen;


static int greedy;


static long long unsigned genomeSize;


static string graphPath;
}

static const char shortopts[] = "G:g:j:k:o:s:v";

enum
{
OPT_HELP = 1,
OPT_VERSION,
OPT_DB,
OPT_LIBRARY,
OPT_STRAIN,
OPT_SPECIES
};

static const struct option longopts[] = { { "genome-size", required_argument, NULL, 'G' },
{ "graph", no_argument, NULL, 'g' },
{ "greedy", no_argument, &opt::greedy, true },
{ "no-greedy", no_argument, &opt::greedy, false },
{ "kmer", required_argument, NULL, 'k' },
{ "out", required_argument, NULL, 'o' },
{ "seed-length", required_argument, NULL, 's' },
{ "threads", required_argument, NULL, 'j' },
{ "verbose", no_argument, NULL, 'v' },
{ "help", no_argument, NULL, OPT_HELP },
{ "version", no_argument, NULL, OPT_VERSION },
{ "db", required_argument, NULL, OPT_DB },
{ "library", required_argument, NULL, OPT_LIBRARY },
{ "strain", required_argument, NULL, OPT_STRAIN },
{ "species", required_argument, NULL, OPT_SPECIES },
{ NULL, 0, NULL, 0 } };

typedef map<ContigID, ContigPath> ContigPathMap;


enum dir_type
{
DIR_X, 
DIR_F, 
DIR_R, 
DIR_B, 
};


typedef vector<unsigned> Lengths;

static ContigPath
align(const Lengths& lengths, const ContigPath& p1, const ContigPath& p2, ContigNode pivot);
static ContigPath
align(
const Lengths& lengths,
const ContigPath& p1,
const ContigPath& p2,
ContigNode pivot,
dir_type& orientation);

static bool gDebugPrint;


static Histogram
buildAssembledLengthHistogram(const Lengths& lengths, const ContigPaths& paths)
{
Histogram h;

vector<bool> used(lengths.size());
for (ContigPaths::const_iterator pathIt = paths.begin(); pathIt != paths.end(); ++pathIt) {
const ContigPath& path = *pathIt;
size_t totalLength = 0;
for (ContigPath::const_iterator it = path.begin(); it != path.end(); ++it) {
if (it->ambiguous())
continue;
unsigned id = it->id();
assert(id < lengths.size());
totalLength += lengths[id];
used[id] = true;
}
h.insert(totalLength);
}

for (unsigned i = 0; i < lengths.size(); ++i) {
if (!used[i])
h.insert(lengths[i]);
}

return h;
}


static void
reportAssemblyMetrics(const Lengths& lengths, const ContigPaths& paths)
{
Histogram h = buildAssembledLengthHistogram(lengths, paths);
const unsigned STATS_MIN_LENGTH = opt::seedLen;
printContiguityStats(cerr, h, STATS_MIN_LENGTH, true, "\t", opt::genomeSize)
<< '\t' << opt::out << '\n';
}


static set<ContigID>
findRepeats(const ContigPathMap& paths)
{
set<ContigID> repeats;
for (ContigPathMap::const_iterator pathIt = paths.begin(); pathIt != paths.end(); ++pathIt) {
const ContigPath& path = pathIt->second;
map<ContigID, unsigned> count;
for (ContigPath::const_iterator it = path.begin(); it != path.end(); ++it)
if (!it->ambiguous())
count[it->contigIndex()]++;
for (map<ContigID, unsigned>::const_iterator it = count.begin(); it != count.end(); ++it)
if (it->second > 1)
repeats.insert(it->first);
}
return repeats;
}


static set<ContigID>
removeRepeats(ContigPathMap& paths)
{
set<ContigID> repeats = findRepeats(paths);
if (gDebugPrint) {
cout << "Repeats:";
if (!repeats.empty()) {
for (set<ContigID>::const_iterator it = repeats.begin(); it != repeats.end(); ++it)
cout << ' ' << get(g_contigNames, *it);
} else
cout << " none";
cout << '\n';
}

unsigned removed = 0;
for (set<ContigID>::const_iterator it = repeats.begin(); it != repeats.end(); ++it)
if (paths.count(*it) > 0)
removed++;
if (removed == paths.size()) {
repeats.clear();
return repeats;
}

ostringstream ss;
for (set<ContigID>::const_iterator it = repeats.begin(); it != repeats.end(); ++it)
if (paths.erase(*it) > 0)
ss << ' ' << get(g_contigNames, *it);

if (opt::verbose > 0 && removed > 0)
cout << "Removing paths in repeats:" << ss.str() << '\n';
return repeats;
}

static void
appendToMergeQ(deque<ContigNode>& mergeQ, set<ContigNode>& seen, const ContigPath& path)
{
for (ContigPath::const_iterator it = path.begin(); it != path.end(); ++it)
if (!it->ambiguous() && seen.insert(*it).second)
mergeQ.push_back(*it);
}


typedef ContigGraph<DirectedGraph<>> PathGraph;


static bool
addOverlapEdge(
const Lengths& lengths,
PathGraph& gout,
ContigNode pivot,
ContigNode seed1,
const ContigPath& path1,
ContigNode seed2,
const ContigPath& path2)
{
assert(seed1 != seed2);

dir_type orientation = DIR_X;
ContigPath consensus = align(lengths, path1, path2, pivot, orientation);
if (consensus.empty())
return false;
assert(orientation != DIR_X);
if (orientation == DIR_B) {
orientation = find(consensus.begin(), consensus.end(), seed1) <
find(consensus.begin(), consensus.end(), seed2)
? DIR_F
: DIR_R;
}
assert(orientation == DIR_F || orientation == DIR_R);

ContigNode u = orientation == DIR_F ? seed1 : seed2;
ContigNode v = orientation == DIR_F ? seed2 : seed1;
bool added = false;
#pragma omp critical(gout)
if (!edge(u, v, gout).second) {
add_edge(u, v, gout);
added = true;
}
return added;
}


static ContigPath
getPath(const ContigPathMap& paths, ContigNode u)
{
ContigPathMap::const_iterator it = paths.find(u.contigIndex());
assert(it != paths.end());
ContigPath path = it->second;
if (u.sense())
reverseComplement(path.begin(), path.end());
return path;
}


static void
findPathOverlaps(
const Lengths& lengths,
const ContigPathMap& paths,
const ContigNode& seed1,
const ContigPath& path1,
PathGraph& gout)
{
for (ContigPath::const_iterator it = path1.begin(); it != path1.end(); ++it) {
ContigNode seed2 = *it;
if (seed1 == seed2)
continue;
if (seed2.ambiguous())
continue;
ContigPathMap::const_iterator path2It = paths.find(seed2.contigIndex());
if (path2It == paths.end())
continue;

ContigPath path2 = path2It->second;
if (seed2.sense())
reverseComplement(path2.begin(), path2.end());
addOverlapEdge(lengths, gout, seed2, seed1, path1, seed2, path2);
}
}


static unsigned
mergePaths(
const Lengths& lengths,
ContigPath& path,
deque<ContigNode>& mergeQ,
set<ContigNode>& seen,
const ContigPathMap& paths)
{
unsigned merged = 0;
deque<ContigNode> invalid;
for (ContigNode pivot; !mergeQ.empty(); mergeQ.pop_front()) {
pivot = mergeQ.front();
ContigPathMap::const_iterator path2It = paths.find(pivot.contigIndex());
if (path2It == paths.end())
continue;

ContigPath path2 = path2It->second;
if (pivot.sense())
reverseComplement(path2.begin(), path2.end());
ContigPath consensus = align(lengths, path, path2, pivot);
if (consensus.empty()) {
invalid.push_back(pivot);
continue;
}

appendToMergeQ(mergeQ, seen, path2);
path.swap(consensus);
if (gDebugPrint)
#pragma omp critical(cout)
cout << get(g_contigNames, pivot) << '\t' << path2 << '\n' << '\t' << path << '\n';
merged++;
}
mergeQ.swap(invalid);
return merged;
}


static ContigPath
mergePath(const Lengths& lengths, const ContigPathMap& paths, const ContigPath& seedPath)
{
assert(!seedPath.empty());
ContigNode seed1 = seedPath.front();
ContigPathMap::const_iterator path1It = paths.find(seed1.contigIndex());
assert(path1It != paths.end());
ContigPath path(path1It->second);
if (seedPath.front().sense())
reverseComplement(path.begin(), path.end());
if (opt::verbose > 1)
#pragma omp critical(cout)
cout << "\n* " << seedPath << '\n'
<< get(g_contigNames, seedPath.front()) << '\t' << path << '\n';
for (ContigPath::const_iterator it = seedPath.begin() + 1; it != seedPath.end(); ++it) {
ContigNode seed2 = *it;
ContigPathMap::const_iterator path2It = paths.find(seed2.contigIndex());
assert(path2It != paths.end());
ContigPath path2 = path2It->second;
if (seed2.sense())
reverseComplement(path2.begin(), path2.end());

ContigNode pivot = find(path.begin(), path.end(), seed2) != path.end() ? seed2 : seed1;
ContigPath consensus = align(lengths, path, path2, pivot);
if (consensus.empty()) {
if (opt::verbose > 1)
#pragma omp critical(cout)
cout << get(g_contigNames, seed2) << '\t' << path2 << '\n' << "\tinvalid\n";
} else {
path.swap(consensus);
if (opt::verbose > 1)
#pragma omp critical(cout)
cout << get(g_contigNames, seed2) << '\t' << path2 << '\n' << '\t' << path << '\n';
}
seed1 = seed2;
}
return path;
}


typedef vector<ContigPath> ContigPaths;


static ContigPaths
mergeSeedPaths(const Lengths& lengths, const ContigPathMap& paths, const ContigPaths& seedPaths)
{
if (opt::verbose > 0)
cout << "\nMerging paths\n";

ContigPaths out;
out.reserve(seedPaths.size());
for (ContigPaths::const_iterator it = seedPaths.begin(); it != seedPaths.end(); ++it)
out.push_back(mergePath(lengths, paths, *it));
return out;
}


static void
extendPaths(const Lengths& lengths, ContigID id, const ContigPathMap& paths, ContigPathMap& out)
{
ContigPathMap::const_iterator pathIt = paths.find(id);
assert(pathIt != paths.end());

pair<ContigPathMap::iterator, bool> inserted;
#pragma omp critical(out)
inserted = out.insert(*pathIt);
assert(inserted.second);
ContigPath& path = inserted.first->second;

if (gDebugPrint)
#pragma omp critical(cout)
cout << "\n* " << get(g_contigNames, id) << "+\n" << '\t' << path << '\n';

set<ContigNode> seen;
seen.insert(ContigNode(id, false));
deque<ContigNode> mergeQ;
appendToMergeQ(mergeQ, seen, path);
while (mergePaths(lengths, path, mergeQ, seen, paths) > 0)
;

if (!mergeQ.empty() && gDebugPrint) {
#pragma omp critical(cout)
{
cout << "invalid\n";
for (deque<ContigNode>::const_iterator it = mergeQ.begin(); it != mergeQ.end(); ++it)
cout << get(g_contigNames, *it) << '\t' << paths.find(it->contigIndex())->second
<< '\n';
}
}
}


static bool
equalOrBothAmbiguos(const ContigNode& a, const ContigNode& b)
{
return a == b || (a.ambiguous() && b.ambiguous());
}


static bool
equalIgnoreAmbiguos(const ContigPath& a, const ContigPath& b)
{
return a.size() == b.size() && equal(a.begin(), a.end(), b.begin(), equalOrBothAmbiguos);
}


static bool
isCycle(const Lengths& lengths, const ContigPath& path)
{
return !align(lengths, path, path, path.front()).empty();
}


static ContigID
identifySubsumedPaths(
const Lengths& lengths,
ContigPathMap::const_iterator path1It,
ContigPathMap& paths,
set<ContigID>& out,
set<ContigID>& overlaps)
{
ostringstream vout;
out.clear();
ContigID id(path1It->first);
const ContigPath& path = path1It->second;
if (gDebugPrint)
vout << get(g_contigNames, ContigNode(id, false)) << '\t' << path << '\n';

for (ContigPath::const_iterator it = path.begin(); it != path.end(); ++it) {
ContigNode pivot = *it;
if (pivot.ambiguous() || pivot.id() == id)
continue;
ContigPathMap::iterator path2It = paths.find(pivot.contigIndex());
if (path2It == paths.end())
continue;
ContigPath path2 = path2It->second;
if (pivot.sense())
reverseComplement(path2.begin(), path2.end());
ContigPath consensus = align(lengths, path, path2, pivot);
if (consensus.empty())
continue;
if (equalIgnoreAmbiguos(consensus, path)) {
if (gDebugPrint)
vout << get(g_contigNames, pivot) << '\t' << path2 << '\n';
out.insert(path2It->first);
} else if (equalIgnoreAmbiguos(consensus, path2)) {
return identifySubsumedPaths(lengths, path2It, paths, out, overlaps);
} else if (isCycle(lengths, consensus)) {
bool isCyclePath1 = isCycle(lengths, path);
bool isCyclePath2 = isCycle(lengths, path2);
if (!isCyclePath1 && !isCyclePath2) {
if (gDebugPrint)
vout << get(g_contigNames, pivot) << '\t' << path2 << '\n'
<< "ignored\t" << consensus << '\n';
overlaps.insert(id);
overlaps.insert(path2It->first);
} else {
if (gDebugPrint)
vout << get(g_contigNames, pivot) << '\t' << path2 << '\n'
<< "cycle\t" << consensus << '\n';
if (isCyclePath1 && isCyclePath2)
out.insert(path2It->first);
else if (!isCyclePath1)
overlaps.insert(id);
else if (!isCyclePath2)
overlaps.insert(path2It->first);
}
} else {
if (gDebugPrint)
vout << get(g_contigNames, pivot) << '\t' << path2 << '\n'
<< "ignored\t" << consensus << '\n';
overlaps.insert(id);
overlaps.insert(path2It->first);
}
}
cout << vout.str();
return id;
}


static ContigPathMap::const_iterator
removeSubsumedPaths(
const Lengths& lengths,
ContigPathMap::const_iterator path1It,
ContigPathMap& paths,
ContigID& seed,
set<ContigID>& overlaps)
{
if (gDebugPrint)
cout << '\n';
set<ContigID> eq;
seed = identifySubsumedPaths(lengths, path1It, paths, eq, overlaps);
++path1It;
for (set<ContigID>::const_iterator it = eq.begin(); it != eq.end(); ++it) {
if (*it == path1It->first)
++path1It;
paths.erase(*it);
}
return path1It;
}


static set<ContigID>
removeSubsumedPaths(const Lengths& lengths, ContigPathMap& paths)
{
set<ContigID> overlaps, seen;
for (ContigPathMap::const_iterator iter = paths.begin(); iter != paths.end();) {
if (seen.count(iter->first) == 0) {
ContigID seed;
iter = removeSubsumedPaths(lengths, iter, paths, seed, overlaps);
seen.insert(seed);
} else
++iter;
}
return overlaps;
}


static void
addMissingEdges(const Lengths& lengths, PathGraph& g, const ContigPathMap& paths)
{
typedef graph_traits<PathGraph>::adjacency_iterator Vit;
typedef graph_traits<PathGraph>::vertex_iterator Uit;
typedef graph_traits<PathGraph>::vertex_descriptor V;

unsigned numAdded = 0;
pair<Uit, Uit> urange = vertices(g);
for (Uit uit = urange.first; uit != urange.second; ++uit) {
V u = *uit;
if (out_degree(u, g) < 2)
continue;
pair<Vit, Vit> vrange = adjacent_vertices(u, g);
for (Vit vit1 = vrange.first; vit1 != vrange.second;) {
V v1 = *vit1;
++vit1;
assert(v1 != u);
ContigPath path1 = getPath(paths, v1);
if (find(path1.begin(), path1.end(), ContigPath::value_type(u)) == path1.end())
continue;
for (Vit vit2 = vit1; vit2 != vrange.second; ++vit2) {
V v2 = *vit2;
assert(v2 != u);
assert(v1 != v2);
if (edge(v1, v2, g).second || edge(v2, v1, g).second)
continue;
ContigPath path2 = getPath(paths, v2);
if (find(path2.begin(), path2.end(), ContigPath::value_type(u)) == path2.end())
continue;
numAdded += addOverlapEdge(lengths, g, u, v1, path1, v2, path2);
}
}
}
if (opt::verbose > 0)
cout << "Added " << numAdded << " missing edges.\n";
if (!opt::db.empty())
addToDb(db, "addedMissingEdges", numAdded);
}


static void
removeTransitiveEdges(PathGraph& pathGraph)
{
unsigned nbefore = num_edges(pathGraph);
unsigned nremoved = remove_transitive_edges(pathGraph);
unsigned nafter = num_edges(pathGraph);
if (opt::verbose > 0)
cout << "Removed " << nremoved << " transitive edges of " << nbefore << " edges leaving "
<< nafter << " edges.\n";
assert(nbefore - nremoved == nafter);
if (!opt::db.empty()) {
addToDb(db, "Edges_init", nbefore);
addToDb(db, "Edges_removed_transitive", nremoved);
}
}


static void
removeSmallOverlaps(PathGraph& g, const ContigPathMap& paths)
{
typedef graph_traits<PathGraph>::edge_descriptor E;
typedef graph_traits<PathGraph>::out_edge_iterator Eit;
typedef graph_traits<PathGraph>::vertex_descriptor V;
typedef graph_traits<PathGraph>::vertex_iterator Vit;

vector<E> edges;
pair<Vit, Vit> urange = vertices(g);
for (Vit uit = urange.first; uit != urange.second; ++uit) {
V u = *uit;
if (out_degree(u, g) < 2)
continue;
ContigPath pathu = getPath(paths, u);
pair<Eit, Eit> uvits = out_edges(u, g);
for (Eit uvit = uvits.first; uvit != uvits.second; ++uvit) {
E uv = *uvit;
V v = target(uv, g);
assert(v != u);
if (in_degree(v, g) < 2)
continue;
ContigPath pathv = getPath(paths, v);
if (pathu.back() == pathv.front() && paths.count(pathu.back().contigIndex()) > 0)
edges.push_back(uv);
}
}
remove_edges(g, edges.begin(), edges.end());
if (opt::verbose > 0)
cout << "Removed " << edges.size() << " small overlap edges.\n";
if (!opt::db.empty())
addToDb(db, "Edges_removed_small_overlap", edges.size());
}


static void
outputPathGraph(PathGraph& pathGraph)
{
if (opt::graphPath.empty())
return;
ofstream out(opt::graphPath.c_str());
assert_good(out, opt::graphPath);
write_dot(out, pathGraph);
assert_good(out, opt::graphPath);
}


static void
outputSortedPaths(const Lengths& lengths, const ContigPathMap& paths)
{
vector<ContigPath> sortedPaths(paths.size());
transform(
paths.begin(),
paths.end(),
sortedPaths.begin(),
mem_var(&ContigPathMap::value_type::second));
sort(sortedPaths.begin(), sortedPaths.end());

ofstream fout(opt::out.c_str());
ostream& out = opt::out.empty() ? cout : fout;
assert_good(out, opt::out);
for (vector<ContigPath>::const_iterator it = sortedPaths.begin(); it != sortedPaths.end(); ++it)
out << createContigName() << '\t' << *it << '\n';
assert_good(out, opt::out);

reportAssemblyMetrics(lengths, sortedPaths);
}


static void
assemblePathGraph(const Lengths& lengths, PathGraph& pathGraph, ContigPathMap& paths)
{
ContigPaths seedPaths;
assembleDFS(pathGraph, back_inserter(seedPaths));
ContigPaths mergedPaths = mergeSeedPaths(lengths, paths, seedPaths);
if (opt::verbose > 1)
cout << '\n';

for (ContigPaths::const_iterator it1 = seedPaths.begin(); it1 != seedPaths.end(); ++it1) {
const ContigPath& path(mergedPaths[it1 - seedPaths.begin()]);
ContigPath pathrc(path);
reverseComplement(pathrc.begin(), pathrc.end());
for (ContigPath::const_iterator it2 = it1->begin(); it2 != it1->end(); ++it2) {
ContigNode seed(*it2);
if (find(path.begin(), path.end(), seed) != path.end()) {
paths[seed.contigIndex()] = seed.sense() ? pathrc : path;
} else {
}
}
}

removeRepeats(paths);

if (opt::verbose > 0)
cout << "Removing redundant contigs\n";
removeSubsumedPaths(lengths, paths);

outputSortedPaths(lengths, paths);
}


static ContigPathMap
readPaths(const Lengths& lengths, const string& filePath)
{
if (opt::verbose > 0)
cerr << "Reading `" << filePath << "'..." << endl;
ifstream in(filePath.c_str());
assert_good(in, filePath);

unsigned tooSmall = 0;
ContigPathMap paths;
std::string name;
ContigPath path;
while (in >> name >> path) {
ContigID id(get(g_contigNames, name));
unsigned len = lengths[id] + opt::k - 1;
if (len < opt::seedLen) {
tooSmall++;
continue;
}

bool inserted = paths.insert(make_pair(id, path)).second;
assert(inserted);
(void)inserted;
}
assert(in.eof());

if (opt::seedLen > 0)
cout << "Ignored " << tooSmall << " paths whose seeds are shorter than " << opt::seedLen
<< " bp.\n";
return paths;
}


template<class T1, class T2, class T3>
bool
atomicInc(T1& it, T2 last, T3& out)
{
#pragma omp critical(atomicInc)
out = it == last ? it : it++;
return out != last;
}


static void
buildPathGraph(const Lengths& lengths, PathGraph& g, const ContigPathMap& paths)
{
PathGraph(lengths.size()).swap(g);

typedef graph_traits<PathGraph>::vertex_iterator vertex_iterator;
pair<vertex_iterator, vertex_iterator> vit = g.vertices();
for (vertex_iterator u = vit.first; u != vit.second; ++u)
if (paths.count(get(vertex_contig_index, g, *u)) == 0)
remove_vertex(*u, g);

ContigPathMap::const_iterator sharedIt = paths.begin();
#pragma omp parallel
for (ContigPathMap::const_iterator it; atomicInc(sharedIt, paths.end(), it);)
findPathOverlaps(lengths, paths, ContigNode(it->first, false), it->second, g);
if (gDebugPrint)
cout << '\n';

addMissingEdges(lengths, g, paths);
removeTransitiveEdges(g);
removeSmallOverlaps(g, paths);
if (opt::verbose > 0)
printGraphStats(cout, g);

vector<int> vals = passGraphStatsVal(g);
vector<string> keys = make_vector<string>() << "V"
<< "E"
<< "degree0pctg"
<< "degree1pctg"
<< "degree234pctg"
<< "degree5pctg"
<< "degree_max";
if (!opt::db.empty()) {
for (unsigned i = 0; i < vals.size(); i++)
addToDb(db, keys[i], vals[i]);
}
outputPathGraph(g);
}


static Lengths
readContigLengths(istream& in)
{
assert(in);
assert(g_contigNames.empty());
Lengths lengths;
string s;
unsigned len;
while (in >> s >> len) {
in.ignore(numeric_limits<streamsize>::max(), '\n');
put(g_contigNames, lengths.size(), s);
assert(len >= opt::k);
lengths.push_back(len - opt::k + 1);
}
assert(in.eof());
assert(!lengths.empty());
g_contigNames.lock();
return lengths;
}


static Lengths
readContigLengths(const string& path)
{
ifstream fin(path.c_str());
if (path != "-")
assert_good(fin, path);
istream& in = path == "-" ? cin : fin;
return readContigLengths(in);
}

int
main(int argc, char** argv)
{
if (!opt::db.empty())
opt::metaVars.resize(3);

bool die = false;
for (int c; (c = getopt_long(argc, argv, shortopts, longopts, NULL)) != -1;) {
istringstream arg(optarg != NULL ? optarg : "");
switch (c) {
case '?':
die = true;
break;
case 'G': {
double x;
arg >> x;
opt::genomeSize = x;
break;
}
case 'g':
arg >> opt::graphPath;
break;
case 'j':
arg >> opt::threads;
break;
case 'k':
arg >> opt::k;
break;
case 'o':
arg >> opt::out;
break;
case 's':
arg >> opt::seedLen;
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
case OPT_DB:
arg >> opt::db;
break;
case OPT_LIBRARY:
arg >> opt::metaVars[0];
break;
case OPT_STRAIN:
arg >> opt::metaVars[1];
break;
case OPT_SPECIES:
arg >> opt::metaVars[2];
break;
}
if (optarg != NULL && !arg.eof()) {
cerr << PROGRAM ": invalid option: `-" << (char)c << optarg << "'\n";
exit(EXIT_FAILURE);
}
}

if (opt::k <= 0) {
cerr << PROGRAM ": missing -k,--kmer option\n";
die = true;
}

if (argc - optind < 2) {
cerr << PROGRAM ": missing arguments\n";
die = true;
} else if (argc - optind > 2) {
cerr << PROGRAM ": too many arguments\n";
die = true;
}

if (die) {
cerr << "Try `" << PROGRAM << " --help' for more information.\n";
exit(EXIT_FAILURE);
}

if (!opt::graphPath.empty())
opt::greedy = false;

gDebugPrint = opt::verbose > 1;

#if _OPENMP
if (opt::threads > 0)
omp_set_num_threads(opt::threads);
#endif

if (opt::verbose > 0)
cerr << "Reading `" << argv[optind] << "'..." << endl;

if (!opt::db.empty()) {
init(db, opt::db, opt::verbose, PROGRAM, opt::getCommand(argc, argv), opt::metaVars);
}

Lengths lengths = readContigLengths(argv[optind++]);
ContigPathMap originalPathMap = readPaths(lengths, argv[optind++]);

removeRepeats(originalPathMap);

if (!opt::db.empty())
addToDb(db, "K", opt::k);
if (!opt::greedy) {
PathGraph pathGraph;
buildPathGraph(lengths, pathGraph, originalPathMap);
if (!opt::out.empty())
assemblePathGraph(lengths, pathGraph, originalPathMap);
exit(EXIT_SUCCESS);
}

ContigPathMap resultsPathMap;
#if _OPENMP
ContigPathMap::iterator sharedIt = originalPathMap.begin();
#pragma omp parallel
for (ContigPathMap::iterator it; atomicInc(sharedIt, originalPathMap.end(), it);)
extendPaths(lengths, it->first, originalPathMap, resultsPathMap);
#else
for (ContigPathMap::const_iterator it = originalPathMap.begin(); it != originalPathMap.end();
++it)
extendPaths(lengths, it->first, originalPathMap, resultsPathMap);
#endif
if (gDebugPrint)
cout << '\n';

set<ContigID> repeats = removeRepeats(resultsPathMap);

if (gDebugPrint)
cout << "\nRemoving redundant contigs\n";
set<ContigID> overlaps = removeSubsumedPaths(lengths, resultsPathMap);

if (!overlaps.empty() && !repeats.empty()) {
for (set<ContigID>::const_iterator it = repeats.begin(); it != repeats.end(); ++it)
originalPathMap.erase(*it);

if (gDebugPrint) {
cout << "\nReassembling overlapping contigs:";
for (set<ContigID>::const_iterator it = overlaps.begin(); it != overlaps.end(); ++it)
cout << ' ' << get(g_contigNames, *it);
cout << '\n';
}

for (set<ContigID>::const_iterator it = overlaps.begin(); it != overlaps.end(); ++it) {
if (originalPathMap.count(*it) == 0)
continue; 
ContigPathMap::iterator oldIt = resultsPathMap.find(*it);
if (oldIt == resultsPathMap.end())
continue; 
ContigPath old = oldIt->second;
resultsPathMap.erase(oldIt);
extendPaths(lengths, *it, originalPathMap, resultsPathMap);
if (gDebugPrint) {
if (resultsPathMap[*it] == old)
cout << "no change\n";
else
cout << "was\t" << old << '\n';
}
}
if (gDebugPrint)
cout << '\n';

removeRepeats(resultsPathMap);
overlaps = removeSubsumedPaths(lengths, resultsPathMap);
if (!overlaps.empty() && gDebugPrint) {
cout << "\nOverlapping contigs:";
for (set<ContigID>::const_iterator it = overlaps.begin(); it != overlaps.end(); ++it)
cout << ' ' << get(g_contigNames, *it);
cout << '\n';
}
}
originalPathMap.clear();

outputSortedPaths(lengths, resultsPathMap);
return 0;
}


static unsigned
getLength(const Lengths& lengths, const ContigNode& u)
{
return u.ambiguous() ? u.length() : lengths.at(u.id());
}


struct AddLength
{
AddLength(const Lengths& lengths)
: m_lengths(lengths)
{}
unsigned operator()(unsigned addend, const ContigNode& u) const
{
return addend + getLength(m_lengths, u);
}

private:
const Lengths& m_lengths;
};


template<class iterator, class oiterator>
static bool
alignCoordinates(
const Lengths& lengths,
iterator& first1,
iterator last1,
iterator& first2,
iterator last2,
oiterator& result)
{
oiterator out = result;

int ambiguous1 = 0, ambiguous2 = 0;
iterator it1 = first1, it2 = first2;
while (it1 != last1 && it2 != last2) {
if (it1->ambiguous()) {
ambiguous1 += it1->length();
++it1;
assert(it1 != last1);
assert(!it1->ambiguous());
}
if (it2->ambiguous()) {
ambiguous2 += it2->length();
++it2;
assert(it2 != last2);
assert(!it2->ambiguous());
}

if (ambiguous1 > 0 && ambiguous2 > 0) {
if (ambiguous1 > ambiguous2) {
*out++ = ContigNode(ambiguous2, 'N');
ambiguous1 -= ambiguous2;
ambiguous2 = 0;
} else {
*out++ = ContigNode(ambiguous1, 'N');
ambiguous2 -= ambiguous1;
ambiguous1 = 0;
}
} else if (ambiguous1 > 0) {
ambiguous1 -= getLength(lengths, *it2);
*out++ = *it2++;
} else if (ambiguous2 > 0) {
ambiguous2 -= getLength(lengths, *it1);
*out++ = *it1++;
} else
assert(false);

if (ambiguous1 == 0 && ambiguous2 == 0)
break;
if (ambiguous1 < 0 || ambiguous2 < 0)
return false;
}

assert(ambiguous1 == 0 || ambiguous2 == 0);
int ambiguous = ambiguous1 + ambiguous2;
assert(out > result);
if (out[-1].ambiguous())
assert(ambiguous == 0);
else
*out++ = ContigNode(max(1, ambiguous), 'N');
first1 = it1;
first2 = it2;
result = out;
return true;
}


template<class iterator, class oiterator>
static bool
buildConsensus(
const Lengths& lengths,
iterator it1,
iterator it1e,
iterator it2,
iterator it2e,
oiterator& out)
{
iterator it1b = it1 + 1;
assert(!it1b->ambiguous());

if (it1b == it1e) {
out = copy(it2, it2e, out);
return true;
}

iterator it2a = it2e - 1;
if (it2e == it2 || !it2a->ambiguous()) {
return false;
}

unsigned ambiguous1 = it1->length();
unsigned ambiguous2 = it2a->length();
unsigned unambiguous1 = accumulate(it1b, it1e, 0, AddLength(lengths));
unsigned unambiguous2 = accumulate(it2, it2a, 0, AddLength(lengths));
if (ambiguous1 < unambiguous2 || ambiguous2 < unambiguous1) {
return false;
}

unsigned n = max(1U, max(ambiguous2 - unambiguous1, ambiguous1 - unambiguous2));
out = copy(it2, it2a, out);
*out++ = ContigNode(n, 'N');
out = copy(it1b, it1e, out);
return true;
}


template<class iterator, class oiterator>
static bool
alignAtSeed(
const Lengths& lengths,
iterator& it1,
iterator it1e,
iterator last1,
iterator& it2,
iterator last2,
oiterator& out)
{
assert(it1 != last1);
assert(it1->ambiguous());
assert(it1 + 1 != last1);
assert(!it1e->ambiguous());
assert(it2 != last2);

unsigned bestLen = UINT_MAX;
iterator bestIt2e;
for (iterator it2e = it2; (it2e = find(it2e, last2, *it1e)) != last2; ++it2e) {
oiterator myOut = out;
if (buildConsensus(lengths, it1, it1e, it2, it2e, myOut) &&
align(lengths, it1e, last1, it2e, last2, myOut)) {
unsigned len = myOut - out;
if (len <= bestLen) {
bestLen = len;
bestIt2e = it2e;
}
}
}
if (bestLen != UINT_MAX) {
bool good = buildConsensus(lengths, it1, it1e, it2, bestIt2e, out);
assert(good);
it1 = it1e;
it2 = bestIt2e;
return good;
} else
return false;
}


template<class iterator, class oiterator>
static bool
alignAmbiguous(
const Lengths& lengths,
iterator& it1,
iterator last1,
iterator& it2,
iterator last2,
oiterator& out)
{
assert(it1 != last1);
assert(it1->ambiguous());
assert(it1 + 1 != last1);
assert(it2 != last2);

for (iterator it1e = it1; it1e != last1; ++it1e) {
if (it1e->ambiguous())
continue;
if (alignAtSeed(lengths, it1, it1e, last1, it2, last2, out))
return true;
}

return alignCoordinates(lengths, it1, last1, it2, last2, out);
}


template<class iterator, class oiterator>
static bool
alignOne(
const Lengths& lengths,
iterator& it1,
iterator last1,
iterator& it2,
iterator last2,
oiterator& out)
{
unsigned n1 = last1 - it1, n2 = last2 - it2;
if (n1 <= n2 && equal(it1, last1, it2)) {
out = copy(it1, last1, out);
it1 += n1;
it2 += n1;
assert(it1 == last1);
return true;
} else if (n2 < n1 && equal(it2, last2, it1)) {
out = copy(it2, last2, out);
it1 += n2;
it2 += n2;
assert(it2 == last2);
return true;
}

return it1->ambiguous() && it2->ambiguous()
? (it1->length() > it2->length()
? alignAmbiguous(lengths, it1, last1, it2, last2, out)
: alignAmbiguous(lengths, it2, last2, it1, last1, out))
: it1->ambiguous()
? alignAmbiguous(lengths, it1, last1, it2, last2, out)
: it2->ambiguous() ? alignAmbiguous(lengths, it2, last2, it1, last1, out)
: (*out++ = *it1, *it1++ == *it2++);
}


template<class iterator, class oiterator>
static dir_type
align(
const Lengths& lengths,
iterator it1,
iterator last1,
iterator it2,
iterator last2,
oiterator& out)
{
assert(it1 != last1);
assert(it2 != last2);
while (it1 != last1 && it2 != last2)
if (!alignOne(lengths, it1, last1, it2, last2, out))
return DIR_X;
assert(it1 == last1 || it2 == last2);
out = copy(it1, last1, out);
out = copy(it2, last2, out);
return it1 == last1 && it2 == last2 ? DIR_B
: it1 == last1 ? DIR_F : it2 == last2 ? DIR_R : DIR_X;
}


static ContigPath
align(
const Lengths& lengths,
const ContigPath& p1,
const ContigPath& p2,
ContigPath::const_iterator pivot1,
ContigPath::const_iterator pivot2,
dir_type& orientation)
{
assert(*pivot1 == *pivot2);
ContigPath::const_reverse_iterator rit1 = ContigPath::const_reverse_iterator(pivot1 + 1),
rit2 = ContigPath::const_reverse_iterator(pivot2 + 1);
ContigPath alignmentr(p1.rend() - rit1 + p2.rend() - rit2);
ContigPath::iterator rout = alignmentr.begin();
dir_type alignedr = align(lengths, rit1, p1.rend(), rit2, p2.rend(), rout);
alignmentr.erase(rout, alignmentr.end());

ContigPath::const_iterator it1 = pivot1, it2 = pivot2;
ContigPath alignmentf(p1.end() - it1 + p2.end() - it2);
ContigPath::iterator fout = alignmentf.begin();
dir_type alignedf = align(lengths, it1, p1.end(), it2, p2.end(), fout);
alignmentf.erase(fout, alignmentf.end());

ContigPath consensus;
if (alignedr != DIR_X && alignedf != DIR_X) {
assert(!alignmentf.empty());
assert(!alignmentr.empty());
consensus.reserve(alignmentr.size() - 1 + alignmentf.size());
consensus.assign(alignmentr.rbegin(), alignmentr.rend() - 1);
consensus.insert(consensus.end(), alignmentf.begin(), alignmentf.end());

unsigned dirs = alignedr << 2 | alignedf;
static const dir_type DIRS[16] = {
DIR_X, 
DIR_X, 
DIR_X, 
DIR_X, 
DIR_X, 
DIR_B, 
DIR_R, 
DIR_R, 
DIR_X, 
DIR_F, 
DIR_B, 
DIR_F, 
DIR_X, 
DIR_F, 
DIR_R, 
DIR_B, 
};
assert(dirs < 16);
orientation = DIRS[dirs];
assert(orientation != DIR_X);
}
return consensus;
}


static pair<ContigNode, bool>
findPivot(const ContigPath& path1, const ContigPath& path2)
{
for (ContigPath::const_iterator it = path2.begin(); it != path2.end(); ++it) {
if (it->ambiguous())
continue;
if (count(path2.begin(), path2.end(), *it) == 1 &&
count(path1.begin(), path1.end(), *it) == 1)
return make_pair(*it, true);
}
return make_pair(ContigNode(0), false);
}


static ContigPath
align(
const Lengths& lengths,
const ContigPath& path1,
const ContigPath& path2,
ContigNode pivot,
dir_type& orientation)
{
if (&path1 == &path2) {
} else if (path1 == path2) {
orientation = DIR_B;
return path1;
} else {
ContigPath::const_iterator it =
search(path1.begin(), path1.end(), path2.begin(), path2.end());
if (it != path1.end()) {
orientation =
it == path1.begin() ? DIR_R : it + path2.size() == path1.end() ? DIR_F : DIR_B;
return path1;
}
}

if (find(path1.begin(), path1.end(), pivot) == path1.end() ||
find(path2.begin(), path2.end(), pivot) == path2.end()) {
bool good;
tie(pivot, good) = findPivot(path1, path2);
if (!good)
return ContigPath();
}
assert(find(path1.begin(), path1.end(), pivot) != path1.end());

ContigPath::const_iterator it2 = find(path2.begin(), path2.end(), pivot);
assert(it2 != path2.end());
if (&path1 != &path2) {
assert(count(it2 + 1, path2.end(), pivot) == 0);
}

ContigPath consensus;
for (ContigPath::const_iterator it1 = find_if(
path1.begin(), path1.end(), [&pivot](const ContigNode& c) { return c == pivot; });
it1 != path1.end();
it1 =
find_if(it1 + 1, path1.end(), [&pivot](const ContigNode& c) { return c == pivot; })) {
if (&*it1 == &*it2) {
continue;
}
consensus = align(lengths, path1, path2, it1, it2, orientation);
if (!consensus.empty())
return consensus;
}
return consensus;
}


static ContigPath
align(const Lengths& lengths, const ContigPath& path1, const ContigPath& path2, ContigNode pivot)
{
dir_type orientation;
return align(lengths, path1, path2, pivot, orientation);
}
