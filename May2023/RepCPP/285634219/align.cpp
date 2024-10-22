
#include <cstddef>
#include "../basic/value.h"
#include "align.h"
#include "../data/reference.h"
#include "../output/output_format.h"
#include "../util/queue.h"
#include "../output/output.h"
#include "query_mapper.h"
#include "../util/merge_sort.h"

using std::map;

struct Align_fetcher
{
static void init(size_t qbegin, size_t qend, vector<hit>::iterator begin, vector<hit>::iterator end)
{
it_ = begin;
end_ = end;
queue_ = auto_ptr<Queue>(new Queue(qbegin, qend));
}
void operator()(size_t query)
{
const unsigned q = (unsigned)query; 
const unsigned c = align_mode.query_contexts;
begin = it_;
while (it_ < end_ && it_->query_ / c == q)
++it_;
end = it_;
this->query = query;
}
bool get()
{
return (*queue_).get(*this) != Queue::end;
}
size_t query;
vector<hit>::iterator begin, end;
private:	
static vector<hit>::iterator it_, end_;
static auto_ptr<Queue> queue_;
};

auto_ptr<Queue> Align_fetcher::queue_;
vector<hit>::iterator Align_fetcher::it_;
vector<hit>::iterator Align_fetcher::end_;

void align_worker(size_t thread_id)
{
Align_fetcher hits; 
Statistics stat;
while (hits.get()) { 
if (hits.end == hits.begin) {
Text_buffer *buf = 0;
Output_sink::get().push(hits.query, buf);
continue;
}
Query_mapper mapper(hits.query, hits.begin, hits.end);
mapper.init();

stat.inc(Statistics::TARGET_HITS0, mapper.n_targets());
for (size_t i = 0; i < mapper.n_targets(); ++i)
mapper.ungapped_stage(i);

mapper.fill_source_ranges();
mapper.rank_targets(config.rank_ratio == -1 ? (mapper.query_seq(0).length() > 50 ? 0.6 : 0.9) : config.rank_ratio);
stat.inc(Statistics::TARGET_HITS1, mapper.n_targets());
const int cutoff = int(mapper.raw_score_cutoff() * config.score_ratio);
for (size_t i = 0; i < mapper.n_targets(); ++i)
mapper.greedy_stage(i, stat, cutoff);

mapper.fill_source_ranges();
mapper.rank_targets(config.rank_ratio2 == -1 ? (mapper.query_seq(0).length() > 50 ? 0.95 : 1.0) : config.rank_ratio2);
stat.inc(Statistics::TARGET_HITS2, mapper.n_targets());

for (size_t i = 0; i < mapper.n_targets(); ++i)
mapper.align_target(i, stat); 
Text_buffer *buf = 0;
buf = new Text_buffer;
const bool aligned = mapper.generate_output(*buf, stat);
if (aligned && !config.unaligned.empty())
query_aligned[hits.query] = true;
Output_sink::get().push(hits.query, buf);
}
statistics += stat;
}

void align_queries(Trace_pt_buffer &trace_pts, Output_stream* output_file)
{
const size_t max_size = (size_t)std::min(config.chunk_size*1e9 * 9 * 2 / config.lowmem, 2e9);
pair<size_t, size_t> query_range;
while (true) {
task_timer timer("Loading trace points", 3);
Trace_pt_list *v = new Trace_pt_list;
statistics.max(Statistics::TEMP_SPACE, trace_pts.load(*v, max_size, query_range));
if (query_range.second - query_range.first == 0) {
delete v;
break;
}
timer.go("Sorting trace points");
merge_sort(v->begin(), v->end(), config.threads_);
v->init();
timer.go("Computing alignments");
Align_fetcher::init(query_range.first, query_range.second, v->begin(), v->end());
Output_sink::instance = auto_ptr<Output_sink>(new Output_sink(query_range.first, output_file));
#if not defined(_OPENMP)
Thread_pool threads;
if (config.verbosity >= 3)
threads.push_back(launch_thread(heartbeat_worker, query_range.second));
for (size_t i = 0; i < (config.threads_align == 0 ? config.threads_ : config.threads_align); ++i)
threads.push_back(launch_thread(static_cast<void(*)(size_t)>(&align_worker), i));
threads.join_all();
#else
#pragma omp parallel
{
#pragma omp single
{
for (size_t i = 0; i < (config.threads_align == 0 ? config.threads_ : config.threads_align); ++i){
#pragma omp task 
align_worker(i);
}}}
#endif

timer.go("Deallocating buffers");
delete v;
}
}
