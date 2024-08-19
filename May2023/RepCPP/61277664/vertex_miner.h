#ifndef VERTEX_MINER_H
#define VERTEX_MINER_H
#include "miner.h"
typedef std::unordered_map<BaseEmbedding, Frequency> SimpleMap;
typedef QuickPattern<EdgeInducedEmbedding<StructuralElement>, StructuralElement> StrQPattern; 
typedef CanonicalGraph<EdgeInducedEmbedding<StructuralElement>, StructuralElement> StrCPattern; 
typedef std::unordered_map<StrQPattern, Frequency> StrQpMapFreq; 
typedef std::unordered_map<StrCPattern, Frequency> StrCgMapFreq; 
typedef PerThreadStorage<StrQpMapFreq> LocalStrQpMapFreq;
typedef PerThreadStorage<StrCgMapFreq> LocalStrCgMapFreq;

class VertexMiner : public Miner {
public:
VertexMiner(Graph *g, unsigned size = 3, int nthreads = 1) {
graph = g;
max_size = size;
degree_counting();
numThreads = nthreads;
}
virtual ~VertexMiner() {}
inline void extend_vertex(unsigned level, EmbeddingList& emb_list) {
UintList num_new_emb(emb_list.size());
#pragma omp parallel for
for (size_t pos = 0; pos < emb_list.size(); pos ++) {
VertexEmbedding emb(level+1);
get_embedding<VertexEmbedding>(level, pos, emb_list, emb);
num_new_emb[pos] = 0;
unsigned n = emb.size();
for (unsigned i = 0; i < n; ++i) {
VertexId src = emb.get_vertex(i);
IndexT row_begin = graph->edge_begin(src);
IndexT row_end = graph->edge_end(src);
for (IndexT e = row_begin; e < row_end; e++) {	
IndexT dst = graph->getEdgeDst(e);
if (!is_vertexInduced_automorphism(emb, i, src, dst)) {
num_new_emb[pos] ++;
}
}
}
}
UintList indices = parallel_prefix_sum<unsigned>(num_new_emb);
num_new_emb.clear();
auto new_size = indices.back();
assert(new_size < 4294967296); 
std::cout << "number of new embeddings: " << new_size << "\n";
emb_list.add_level(new_size);
#ifdef USE_WEDGE
if (level == 1 && max_size == 4) {
is_wedge.resize(emb_list.size());
std::fill(is_wedge.begin(), is_wedge.end(), 0);
}
#endif
for (size_t pos = 0; pos < emb_list.size(level); pos ++) {
VertexEmbedding emb(level+1);
get_embedding<VertexEmbedding>(level, pos, emb_list, emb);
auto start = indices[pos];
auto n = emb.size();
for (unsigned i = 0; i < n; ++i) {
VertexId src = emb.get_vertex(i);
IndexT row_begin = graph->edge_begin(src);
IndexT row_end = graph->edge_end(src);
for (IndexT e = row_begin; e < row_end; e++) {
IndexT dst = graph->getEdgeDst(e);
if (!is_vertexInduced_automorphism(emb, i, src, dst)) {
assert(start < indices.back());
if (n == 2 && max_size == 4)
emb_list.set_pid(start, find_motif_pattern_id(n, i, dst, emb, start));
emb_list.set_idx(level+1, start, pos);
emb_list.set_vid(level+1, start++, dst);
}
}
}
}
indices.clear();
}
inline void extend_vertex(unsigned level, EmbeddingList& emb_list, Accumulator<uint64_t> &num) {
UintList num_new_emb(emb_list.size());
#pragma omp parallel for
for (size_t pos = 0; pos < emb_list.size(); pos ++) {
BaseEmbedding emb(level+1);
get_embedding<BaseEmbedding>(level, pos, emb_list, emb);
VertexId vid = emb_list.get_vid(level, pos);
num_new_emb[pos] = 0;
IndexT row_begin = graph->edge_begin(vid);
IndexT row_end = graph->edge_end(vid);
for (IndexT e = row_begin; e < row_end; e++) {
IndexT dst = graph->getEdgeDst(e);
if (is_all_connected_dag(dst, emb, level)) {
if (level < max_size-2) num_new_emb[pos] ++;
else num += 1;
}
}
}
if (level == max_size-2) return;
UintList indices = parallel_prefix_sum<unsigned>(num_new_emb);
num_new_emb.clear();
auto new_size = indices.back();
assert(new_size < 4294967296); 
std::cout << "number of new embeddings: " << new_size << "\n";
emb_list.add_level(new_size);
#pragma omp parallel for
for (size_t pos = 0; pos < emb_list.size(level); pos ++) {
BaseEmbedding emb(level+1);
get_embedding<BaseEmbedding>(level, pos, emb_list, emb);
VertexId vid = emb_list.get_vid(level, pos);
unsigned start = indices[pos];
IndexT row_begin = graph->edge_begin(vid);
IndexT row_end = graph->edge_end(vid);
for (IndexT e = row_begin; e < row_end; e++) {
IndexT dst = graph->getEdgeDst(e);
if (is_all_connected_dag(dst, emb, level)) {
emb_list.set_idx(level+1, start, pos);
emb_list.set_vid(level+1, start++, dst);
}
}
}
indices.clear();
}
inline void aggregate(unsigned level, EmbeddingList& emb_list, std::vector<UlongAccu> &accumulators) {
#pragma omp parallel for
for (size_t pos = 0; pos < emb_list.size(); pos ++) {
VertexEmbedding emb(level+1);
get_embedding<VertexEmbedding>(level, pos, emb_list, emb);
unsigned n = emb.size();
if (n == 3) emb.set_pid(emb_list.get_pid(pos));
for (unsigned i = 0; i < n; ++i) {
VertexId src = emb.get_vertex(i);
IndexT row_begin = graph->edge_begin(src);
IndexT row_end = graph->edge_end(src);
for (IndexT e = row_begin; e < row_end; e++) {
IndexT dst = graph->getEdgeDst(e);
if (!is_vertexInduced_automorphism(emb, i, src, dst)) {
assert(n < 4);
unsigned pid = find_motif_pattern_id(n, i, dst, emb, pos);
assert(pid < accumulators.size());
accumulators[pid] += 1;
}
}
}
emb.clean();
}
}
inline void quick_aggregate(unsigned level, const EmbeddingList& emb_list) {
for (auto i = 0; i < numThreads; i++) qp_localmaps.getLocal(i)->clear();
#pragma omp parallel for
for (size_t pos = 0; pos < emb_list.size(level); pos ++) {
int tid = omp_get_thread_num();
StrQpMapFreq* qp_map = qp_localmaps.getLocal(tid);
VertexEmbedding emb(level+1);
get_embedding<VertexEmbedding>(level, pos, emb_list, emb);
unsigned n = emb.size();
for (unsigned i = 0; i < n; ++i) {
VertexId src = emb.get_vertex(i);
IndexT row_begin = graph->edge_begin(src);
IndexT row_end = graph->edge_end(src);
for (IndexT e = row_begin; e < row_end; e++) {
IndexT dst = graph->getEdgeDst(e);
if (!is_vertexInduced_automorphism(emb, i, src, dst)) {
std::vector<bool> connected;
get_connectivity(n, i, dst, emb, connected);
StrQPattern qp(n+1, connected);
if (qp_map->find(qp) != qp_map->end()) {
(*qp_map)[qp] += 1;
qp.clean();
} else (*qp_map)[qp] = 1;
}
}
}
emb.clean();
}
}
inline void canonical_aggregate() {
for (auto i = 0; i < numThreads; i++) cg_localmaps.getLocal(i)->clear();

}
inline void merge_qp_map() {
qp_map.clear();
for (unsigned i = 0; i < qp_localmaps.size(); i++) {
StrQpMapFreq qp_lmap = *qp_localmaps.getLocal(i);
for (auto element : qp_lmap) {
if (qp_map.find(element.first) != qp_map.end())
qp_map[element.first] += element.second;
else qp_map[element.first] = element.second;
}
}
}
inline void merge_cg_map() {
cg_map.clear();
for (unsigned i = 0; i < cg_localmaps.size(); i++) {
StrCgMapFreq cg_lmap = *cg_localmaps.getLocal(i);
for (auto element : cg_lmap) {
if (cg_map.find(element.first) != cg_map.end())
cg_map[element.first] += element.second;
else cg_map[element.first] = element.second;
}
}
}

void printout_motifs(std::vector<UlongAccu> &accumulators) {
std::cout << std::endl;
if (accumulators.size() == 2) {
std::cout << "\ttriangles\t" << accumulators[0].reduce() << std::endl;
std::cout << "\t3-chains\t" << accumulators[1].reduce() << std::endl;
} else if (accumulators.size() == 6) {
std::cout << "\t4-paths --> " << accumulators[0].reduce() << std::endl;
std::cout << "\t3-stars --> " << accumulators[1].reduce() << std::endl;
std::cout << "\t4-cycles --> " << accumulators[2].reduce() << std::endl;
std::cout << "\ttailed-triangles --> " << accumulators[3].reduce() << std::endl;
std::cout << "\tdiamonds --> " << accumulators[4].reduce() << std::endl;
std::cout << "\t4-cliques --> " << accumulators[5].reduce() << std::endl;
} else {
std::cout << "\ttoo many patterns to show\n";
}
std::cout << std::endl;
}
void printout_motifs(UintMap &p_map) {
assert(p_map.size() == 21);
std::cout << std::endl;
for (auto it = p_map.begin(); it != p_map.end(); ++it)
std::cout << "{" << it->first << "} --> " << it->second << std::endl;
std::cout << std::endl;
}
void printout_motifs() {
std::cout << std::endl;
for (auto it = cg_map.begin(); it != cg_map.end(); ++it)
std::cout << it->first << " --> " << it->second << std::endl;
std::cout << std::endl;
}

private:
unsigned max_size;
int numThreads;
std::vector<unsigned> is_wedge; 
StrQpMapFreq qp_map; 
StrCgMapFreq cg_map; 
LocalStrQpMapFreq qp_localmaps; 
LocalStrCgMapFreq cg_localmaps; 

template <typename EmbeddingTy>
inline void get_embedding(unsigned level, unsigned pos, const EmbeddingList& emb_list, EmbeddingTy &emb) {
VertexId vid = emb_list.get_vid(level, pos);
IndexTy idx = emb_list.get_idx(level, pos);
ElementType ele(vid);
emb.set_element(level, ele);
for (unsigned l = 1; l < level; l ++) {
VertexId u = emb_list.get_vid(level-l, idx);
ElementType ele(u);
emb.set_element(level-l, ele);
idx = emb_list.get_idx(level-l, idx);
}
ElementType ele0(idx);
emb.set_element(0, ele0);
}
inline bool is_vertexInduced_automorphism(const VertexEmbedding& emb, unsigned idx, VertexId src, VertexId dst) {
unsigned n = emb.size();
if (dst <= emb.get_vertex(0)) return true;
for (unsigned i = 1; i < n; ++i)
if (dst == emb.get_vertex(i)) return true;
for (unsigned i = 0; i < idx; ++i)
if (is_connected(emb.get_vertex(i), dst)) return true;
for (unsigned i = idx+1; i < n; ++i)
if (dst < emb.get_vertex(i)) return true;
return false;
}
inline unsigned find_motif_pattern_id(unsigned n, unsigned idx, VertexId dst, const VertexEmbedding& emb, unsigned pos = 0) {
unsigned pid = 0;
if (n == 2) { 
pid = 1; 
if (idx == 0) {
if (is_connected(emb.get_vertex(1), dst)) pid = 0; 
#ifdef USE_WEDGE
else if (max_size == 4) is_wedge[pos] = 1; 
#endif
}
} else if (n == 3) { 
unsigned num_edges = 1;
pid = emb.get_pid();
if (pid == 0) { 
for (unsigned j = idx+1; j < n; j ++)
if (is_connected(emb.get_vertex(j), dst)) num_edges ++;
pid = num_edges + 2; 
} else { 
assert(pid == 1);
std::vector<bool> connected(3, false);
connected[idx] = true;
for (unsigned j = idx+1; j < n; j ++) {
if (is_connected(emb.get_vertex(j), dst)) {
num_edges ++;
connected[j] = true;
}
}
if (num_edges == 1) {
pid = 0; 
unsigned center = 1;
#ifdef USE_WEDGE
if (is_wedge[pos]) center = 0;
#else
center = is_connected(emb.get_vertex(1), emb.get_vertex(2)) ? 1 : 0;
#endif
if (idx == center) pid = 1; 
} else if (num_edges == 2) {
pid = 2; 
unsigned center = 1;
#ifdef USE_WEDGE
if (is_wedge[pos]) center = 0;
#else
center = is_connected(emb.get_vertex(1), emb.get_vertex(2)) ? 1 : 0;
#endif
if (connected[center]) pid = 3; 
} else {
pid = 4; 
}
}
} else { 
std::vector<bool> connected;
get_connectivity(n, idx, dst, emb, connected);
Matrix A(n+1, std::vector<MatType>(n+1, 0));
gen_adj_matrix(n+1, connected, A);
std::vector<MatType> c(n+1, 0);
char_polynomial(n+1, A, c);
bliss::UintSeqHash h;
for (unsigned i = 0; i < n+1; ++i)
h.update((unsigned)c[i]);
pid = h.get_value();
}
return pid;
}
};

#endif 
