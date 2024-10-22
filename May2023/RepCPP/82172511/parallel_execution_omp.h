
#ifndef GRPPI_OMP_PARALLEL_EXECUTION_OMP_H
#define GRPPI_OMP_PARALLEL_EXECUTION_OMP_H

#ifdef GRPPI_OMP

#include "../common/mpmc_queue.h"
#include "../common/iterator.h"
#include "../common/execution_traits.h"
#include "../common/configuration.h"
#include "grppi/seq/sequential_execution.h"

#include <type_traits>
#include <tuple>

#include <omp.h>

namespace grppi {


class parallel_execution_omp {

public:

parallel_execution_omp() noexcept {};

parallel_execution_omp(int concurrency_degree) noexcept
:
concurrency_degree_{concurrency_degree}
{
omp_set_num_threads(concurrency_degree_);
}




parallel_execution_omp(int concurrency_degree, bool order) noexcept
:
concurrency_degree_{concurrency_degree},
ordering_{order}
{
omp_set_num_threads(concurrency_degree_);
}


void set_concurrency_degree(int degree) noexcept
{
concurrency_degree_ = degree;
omp_set_num_threads(concurrency_degree_);
}


int concurrency_degree() const noexcept
{
return concurrency_degree_;
}


void enable_ordering() noexcept { ordering_ = true; }


void disable_ordering() noexcept { ordering_ = false; }


bool is_ordered() const noexcept { return ordering_; }


void set_queue_attributes(int size, queue_mode mode) noexcept
{
queue_size_ = size;
queue_mode_ = mode;
}


template<typename T>
mpmc_queue <T> make_queue() const
{
return {queue_size_, queue_mode_};
}


template<typename T, typename ... Transformers>
mpmc_queue <T> &
get_output_queue(mpmc_queue <T> & queue, Transformers && ...) const
{
return queue;
}


template<typename T, typename ... Transformers>
mpmc_queue <T> get_output_queue(Transformers && ...) const
{
return std::move(make_queue<T>());
}


[[deprecated("Thread ids are deprecated.\n"
"If you have a specific use case file a bug")]]
int get_thread_id() const noexcept
{
int result;
#pragma omp parallel
{
result = omp_get_thread_num();
}
return result;
}


template<typename ... InputIterators, typename OutputIterator,
typename Transformer>
void map(std::tuple<InputIterators...> firsts,
OutputIterator first_out,
std::size_t sequence_size, Transformer transform_op) const;


template<typename InputIterator, typename Identity, typename Combiner>
auto reduce(InputIterator first, std::size_t sequence_size,
Identity && identity, Combiner && combine_op) const;


template<typename ... InputIterators, typename Identity,
typename Transformer, typename Combiner>
auto map_reduce(std::tuple<InputIterators...> firsts,
std::size_t sequence_size,
Identity && identity,
Transformer && transform_op, Combiner && combine_op) const;


template<typename ... InputIterators, typename OutputIterator,
typename StencilTransformer, typename Neighbourhood>
void stencil(std::tuple<InputIterators...> firsts, OutputIterator first_out,
std::size_t sequence_size,
StencilTransformer && transform_op,
Neighbourhood && neighbour_op) const;


template<typename Input, typename Divider, typename Solver, typename Combiner>
[[deprecated("Use new interface with predicate argument")]]
auto divide_conquer(Input && input,
Divider && divide_op,
Solver && solve_op,
Combiner && combine_op) const;


template<typename Input, typename Divider, typename Predicate, typename Solver, typename Combiner>
auto divide_conquer(Input && input,
Divider && divide_op,
Predicate && predicate_op,
Solver && solve_op,
Combiner && combine_op) const;



template<typename Generator, typename ... Transformers>
void pipeline(Generator && generate_op,
Transformers && ... transform_op) const;


template<typename InputType, typename Transformer, typename OutputType>
void
pipeline(mpmc_queue <InputType> & input_queue, Transformer && transform_op,
mpmc_queue <OutputType> & output_queue) const
{
do_pipeline(input_queue, std::forward<Transformer>(transform_op),
output_queue);
}


template<typename Population, typename Selection, typename Evolution,
typename Evaluation, typename Predicate>
void stream_pool(Population & population,
Selection && selection_op,
Evolution && evolve_op,
Evaluation && eval_op,
Predicate && termination_op) const;

private:

template<typename Input, typename Divider, typename Solver, typename Combiner>
auto divide_conquer(Input && input,
Divider && divide_op,
Solver && solve_op,
Combiner && combine_op,
std::atomic<int> & num_threads) const;

template<typename Input, typename Divider, typename Predicate, typename Solver, typename Combiner>
auto divide_conquer(Input && input,
Divider && divide_op,
Predicate && predicate_op,
Solver && solve_op,
Combiner && combine_op,
std::atomic<int> & num_threads) const;


template<typename Queue, typename Consumer,
requires_no_pattern <Consumer> = 0>
void do_pipeline(Queue & input_queue, Consumer && consume_op) const;

template<typename Inqueue, typename Transformer, typename output_type,
requires_no_pattern <Transformer> = 0>
void do_pipeline(Inqueue & input_queue, Transformer && transform_op,
mpmc_queue <output_type> & output_queue) const;

template<typename T, typename ... Others>
void do_pipeline(mpmc_queue <T> & in_q, mpmc_queue <T> & same_queue,
Others && ... ops) const;

template<typename T>
void do_pipeline(mpmc_queue <T> &) const {}

template<typename Queue, typename Transformer, typename ... OtherTransformers,
requires_no_pattern <Transformer> = 0>
void do_pipeline(Queue & input_queue, Transformer && transform_op,
OtherTransformers && ... other_ops) const;

template<typename Queue, typename Execution, typename Transformer,
template<typename, typename> class Context,
typename ... OtherTransformers,
requires_context <Context<Execution, Transformer>> = 0>
void do_pipeline(Queue & input_queue,
Context<Execution, Transformer> && context_op,
OtherTransformers && ... other_ops) const;

template<typename Queue, typename Execution, typename Transformer,
template<typename, typename> class Context,
typename ... OtherTransformers,
requires_context <Context<Execution, Transformer>> = 0>
void do_pipeline(Queue & input_queue,
Context<Execution, Transformer> & context_op,
OtherTransformers && ... other_ops) const
{
do_pipeline(input_queue, std::move(context_op),
std::forward<OtherTransformers>(other_ops)...);
}

template<typename Queue, typename FarmTransformer,
template<typename> class Farm,
requires_farm <Farm<FarmTransformer>> = 0>
void do_pipeline(Queue & input_queue,
Farm<FarmTransformer> & farm_obj) const
{
do_pipeline(input_queue, std::move(farm_obj));
}

template<typename Queue, typename FarmTransformer,
template<typename> class Farm,
requires_farm <Farm<FarmTransformer>> = 0>
void do_pipeline(Queue & input_queue,
Farm<FarmTransformer> && farm_obj) const;

template<typename Queue, typename FarmTransformer,
template<typename> class Farm,
typename ... OtherTransformers,
requires_farm <Farm<FarmTransformer>> = 0>
void do_pipeline(Queue & input_queue,
Farm<FarmTransformer> & farm_obj,
OtherTransformers && ... other_transform_ops) const
{
do_pipeline(input_queue, std::move(farm_obj),
std::forward<OtherTransformers>(other_transform_ops)...);
}

template<typename Queue, typename FarmTransformer,
template<typename> class Farm,
typename ... OtherTransformers,
requires_farm <Farm<FarmTransformer>> = 0>
void do_pipeline(Queue & input_queue,
Farm<FarmTransformer> && farm_obj,
OtherTransformers && ... other_transform_ops) const;

template<typename Queue, typename Predicate,
template<typename> class Filter,
requires_filter <Filter<Predicate>> = 0>
void do_pipeline(Queue & input_queue,
Filter<Predicate> & filter_obj) const
{
do_pipeline(input_queue, std::move(filter_obj));
}

template<typename Queue, typename Predicate,
template<typename> class Filter,
requires_filter <Filter<Predicate>> = 0>
void do_pipeline(Queue & input_queue,
Filter<Predicate> && filter_obj) const;

template<typename Queue, typename Predicate,
template<typename> class Filter,
typename ... OtherTransformers,
requires_filter <Filter<Predicate>> = 0>
void do_pipeline(Queue & input_queue,
Filter<Predicate> & filter_obj,
OtherTransformers && ... other_transform_ops) const
{
do_pipeline(input_queue, std::move(filter_obj),
std::forward<OtherTransformers>(other_transform_ops)...);
}

template<typename Queue, typename Predicate,
template<typename> class Filter,
typename ... OtherTransformers,
requires_filter <Filter<Predicate>> = 0>
void do_pipeline(Queue & input_queue,
Filter<Predicate> && filter_obj,
OtherTransformers && ... other_transform_ops) const;

template<typename Queue, typename Combiner, typename Identity,
template<typename C, typename I> class Reduce,
typename ... OtherTransformers,
requires_reduce <Reduce<Combiner, Identity>> = 0>
void
do_pipeline(Queue && input_queue, Reduce<Combiner, Identity> & reduce_obj,
OtherTransformers && ... other_transform_ops) const
{
do_pipeline(input_queue, std::move(reduce_obj),
std::forward<OtherTransformers>(other_transform_ops)...);
}

template<typename Queue, typename Combiner, typename Identity,
template<typename C, typename I> class Reduce,
typename ... OtherTransformers,
requires_reduce <Reduce<Combiner, Identity>> = 0>
void
do_pipeline(Queue && input_queue, Reduce<Combiner, Identity> && reduce_obj,
OtherTransformers && ... other_transform_ops) const;

template<typename Queue, typename Transformer, typename Predicate,
template<typename T, typename P> class Iteration,
typename ... OtherTransformers,
requires_iteration <Iteration<Transformer, Predicate>> = 0,
requires_no_pattern <Transformer> = 0>
void do_pipeline(Queue & input_queue,
Iteration<Transformer, Predicate> & iteration_obj,
OtherTransformers && ... other_transform_ops) const
{
do_pipeline(input_queue, std::move(iteration_obj),
std::forward<OtherTransformers>(other_transform_ops)...);
}

template<typename Queue, typename Transformer, typename Predicate,
template<typename T, typename P> class Iteration,
typename ... OtherTransformers,
requires_iteration <Iteration<Transformer, Predicate>> = 0,
requires_no_pattern <Transformer> = 0>
void do_pipeline(Queue & input_queue,
Iteration<Transformer, Predicate> && iteration_obj,
OtherTransformers && ... other_transform_ops) const;

template<typename Queue, typename Transformer, typename Predicate,
template<typename T, typename P> class Iteration,
typename ... OtherTransformers,
requires_iteration <Iteration<Transformer, Predicate>> = 0,
requires_pipeline <Transformer> = 0>
void do_pipeline(Queue & input_queue,
Iteration<Transformer, Predicate> && iteration_obj,
OtherTransformers && ... other_transform_ops) const;

template<typename Queue, typename ... Transformers,
template<typename...> class Pipeline,
typename ... OtherTransformers,
requires_pipeline <Pipeline<Transformers...>> = 0>
void do_pipeline(Queue & input_queue,
Pipeline<Transformers...> & pipeline_obj,
OtherTransformers && ... other_transform_ops) const
{
do_pipeline(input_queue, std::move(pipeline_obj),
std::forward<OtherTransformers>(other_transform_ops)...);
}

template<typename Queue, typename ... Transformers,
template<typename...> class Pipeline,
typename ... OtherTransformers,
requires_pipeline <Pipeline<Transformers...>> = 0>
void do_pipeline(Queue & input_queue,
Pipeline<Transformers...> && pipeline_obj,
OtherTransformers && ... other_transform_ops) const;

template<typename Queue, typename ... Transformers,
std::size_t ... I>
void do_pipeline_nested(
Queue & input_queue,
std::tuple<Transformers...> && transform_ops,
std::index_sequence<I...>) const;

private:


static int impl_concurrency_degree()
{
int result;
#pragma omp parallel
{
result = omp_get_num_threads();
}
return result;
}

private:

configuration<> config_{};

int concurrency_degree_ = config_.concurrency_degree();

bool ordering_ = config_.ordering();

int queue_size_ = config_.queue_size();

queue_mode queue_mode_ = config_.mode();
};


template<typename E>
constexpr bool is_parallel_execution_omp()
{
return std::is_same<E, parallel_execution_omp>::value;
}


template<>
constexpr bool is_supported<parallel_execution_omp>() { return true; }


template<>
constexpr bool supports_map<parallel_execution_omp>() { return true; }


template<>
constexpr bool supports_reduce<parallel_execution_omp>() { return true; }


template<>
constexpr bool supports_map_reduce<parallel_execution_omp>() { return true; }


template<>
constexpr bool supports_stencil<parallel_execution_omp>() { return true; }


template<>
constexpr bool
supports_divide_conquer<parallel_execution_omp>() { return true; }


template<>
constexpr bool supports_pipeline<parallel_execution_omp>() { return true; }


template<>
constexpr bool supports_stream_pool<parallel_execution_omp>() { return true; }

template<typename ... InputIterators, typename OutputIterator,
typename Transformer>
void parallel_execution_omp::map(
std::tuple<InputIterators...> firsts,
OutputIterator first_out,
std::size_t sequence_size, Transformer transform_op) const
{
#pragma omp parallel for
for (std::size_t i = 0; i < sequence_size; ++i) {
first_out[i] = apply_iterators_indexed(transform_op, firsts, i);
}
}

template<typename InputIterator, typename Identity, typename Combiner>
auto parallel_execution_omp::reduce(
InputIterator first, std::size_t sequence_size,
Identity && identity,
Combiner && combine_op) const
{
constexpr sequential_execution seq;

using result_type = std::decay_t<Identity>;
std::vector<result_type> partial_results(concurrency_degree_);
auto process_chunk = [&](InputIterator f, std::size_t sz, std::size_t id) {
partial_results[id] = seq.reduce(f, sz, std::forward<Identity>(identity),
std::forward<Combiner>(combine_op));
};

const auto chunk_size = sequence_size / concurrency_degree_;

#pragma omp parallel
{
#pragma omp single nowait
{
for (int i = 0; i < concurrency_degree_ - 1; ++i) {
const auto delta = chunk_size * i;
const auto chunk_first = std::next(first, delta);

#pragma omp task firstprivate (chunk_first, chunk_size, i)
{
process_chunk(chunk_first, chunk_size, i);
}
}

const auto delta = chunk_size * (concurrency_degree_ - 1);
const auto chunk_first = std::next(first, delta);
const auto chunk_sz = sequence_size - delta;
process_chunk(chunk_first, chunk_sz, concurrency_degree_ - 1);
#pragma omp taskwait
}
}

return seq.reduce(std::next(partial_results.begin()),
partial_results.size() - 1,
partial_results[0], std::forward<Combiner>(combine_op));
}

template<typename ... InputIterators, typename Identity,
typename Transformer, typename Combiner>
auto parallel_execution_omp::map_reduce(
std::tuple<InputIterators...> firsts,
std::size_t sequence_size,
Identity && identity,
Transformer && transform_op, Combiner && combine_op) const
{
constexpr sequential_execution seq;

using result_type = std::decay_t<Identity>;
std::vector<result_type> partial_results(concurrency_degree_);

auto process_chunk = [&](auto f, std::size_t sz, std::size_t i) {
partial_results[i] = seq.map_reduce(
f, sz,
std::forward<Identity>(identity),
std::forward<Transformer>(transform_op),
std::forward<Combiner>(combine_op));
};

const auto chunk_size = sequence_size / concurrency_degree_;

#pragma omp parallel
{
#pragma omp single nowait
{

for (int i = 0; i < concurrency_degree_ - 1; ++i) {
#pragma omp task firstprivate(i)
{
const auto delta = chunk_size * i;
const auto chunk_firsts = iterators_next(firsts, delta);
process_chunk(chunk_firsts, chunk_size, i);
}
}

const auto delta = chunk_size * (concurrency_degree_ - 1);
auto chunk_firsts = iterators_next(firsts, delta);
auto chunk_last = std::next(std::get<0>(firsts), sequence_size);
process_chunk(chunk_firsts,
std::distance(std::get<0>(chunk_firsts), chunk_last),
concurrency_degree_ - 1);
#pragma omp taskwait
}
}

return seq.reduce(partial_results.begin(),
partial_results.size(), std::forward<Identity>(identity),
std::forward<Combiner>(combine_op));
}

template<typename ... InputIterators, typename OutputIterator,
typename StencilTransformer, typename Neighbourhood>
void parallel_execution_omp::stencil(
std::tuple<InputIterators...> firsts, OutputIterator first_out,
std::size_t sequence_size,
StencilTransformer && transform_op,
Neighbourhood && neighbour_op) const
{
constexpr sequential_execution seq;
const auto chunk_size = sequence_size / concurrency_degree_;
auto process_chunk = [&](auto f, std::size_t sz, std::size_t delta) {
seq.stencil(f, std::next(first_out, delta), sz,
std::forward<StencilTransformer>(transform_op),
std::forward<Neighbourhood>(neighbour_op));
};

#pragma omp parallel
{
#pragma omp single nowait
{
for (int i = 0; i < concurrency_degree_ - 1; ++i) {
#pragma omp task firstprivate(i)
{
const auto delta = chunk_size * i;
const auto chunk_firsts = iterators_next(firsts, delta);
process_chunk(chunk_firsts, chunk_size, delta);
}
}

const auto delta = chunk_size * (concurrency_degree_ - 1);
const auto chunk_firsts = iterators_next(firsts, delta);
const auto chunk_last = std::next(std::get<0>(firsts), sequence_size);
process_chunk(chunk_firsts,
std::distance(std::get<0>(chunk_firsts), chunk_last), delta);

#pragma omp taskwait
}
}
}

template<typename Input, typename Divider, typename Predicate, typename Solver, typename Combiner>
auto parallel_execution_omp::divide_conquer(
Input && input,
Divider && divide_op,
Predicate && predicate_op,
Solver && solve_op,
Combiner && combine_op) const
{
std::atomic<int> num_threads{concurrency_degree_ - 1};

return divide_conquer(std::forward<Input>(input),
std::forward<Divider>(divide_op), std::forward<Predicate>(predicate_op),
std::forward<Solver>(solve_op),
std::forward<Combiner>(combine_op),
num_threads);
}


template<typename Input, typename Divider, typename Solver, typename Combiner>
auto parallel_execution_omp::divide_conquer(
Input && input,
Divider && divide_op,
Solver && solve_op,
Combiner && combine_op) const
{
std::atomic<int> num_threads{concurrency_degree_ - 1};

return divide_conquer(std::forward<Input>(input),
std::forward<Divider>(divide_op), std::forward<Solver>(solve_op),
std::forward<Combiner>(combine_op),
num_threads);
}

template<typename Generator, typename ... Transformers>
void parallel_execution_omp::pipeline(
Generator && generate_op,
Transformers && ... transform_ops) const
{
using namespace std;

using result_type = decay_t<typename result_of<Generator()>::type>;
auto output_queue = make_queue<pair<result_type, long>>();

#pragma omp parallel
{
#pragma omp single nowait
{
#pragma omp task shared(generate_op, output_queue)
{
long order = 0;
for (;;) {
auto item = generate_op();
output_queue.push(make_pair(item, order++));
if (!item) { break; }
}
}
do_pipeline(output_queue,
forward<Transformers>(transform_ops)...);
#pragma omp taskwait
}
}
}

template<typename Population, typename Selection, typename Evolution,
typename Evaluation, typename Predicate>
void parallel_execution_omp::stream_pool(
[[maybe_unused]] Population & population,
[[maybe_unused]] Selection && selection_op,
[[maybe_unused]] Evolution && evolve_op,
[[maybe_unused]] Evaluation && eval_op,
[[maybe_unused]] Predicate && termination_op) const
{
std::cerr << "stream_pool currently unimplemented on OpenMP\n";
std::abort();

}

template<typename Input, typename Divider, typename Predicate, typename Solver, typename Combiner>
auto parallel_execution_omp::divide_conquer(
Input && input,
Divider && divide_op,
Predicate && predicate_op,
Solver && solve_op,
Combiner && combine_op,
std::atomic<int> & num_threads) const
{
constexpr sequential_execution seq;
if (num_threads.load() <= 0) {
return seq.divide_conquer(std::forward<Input>(input),
std::forward<Divider>(divide_op),
std::forward<Predicate>(predicate_op),
std::forward<Solver>(solve_op),
std::forward<Combiner>(combine_op));
}

if (predicate_op(input)) { return solve_op(std::forward<Input>(input)); }
auto subproblems = divide_op(std::forward<Input>(input));

using subresult_type =
std::decay_t<typename std::result_of<Solver(Input)>::type>;
std::vector<subresult_type> partials(subproblems.size() - 1);

auto process_subproblems = [&, this](auto it, std::size_t div) {
partials[div] = this->divide_conquer(std::forward<Input>(*it),
std::forward<Divider>(divide_op),
std::forward<Predicate>(predicate_op),
std::forward<Solver>(solve_op),
std::forward<Combiner>(combine_op), num_threads);
};

int division = 0;
subresult_type subresult;

#pragma omp parallel
{
#pragma omp single nowait
{
auto i = subproblems.begin() + 1;
while (i != subproblems.end() && num_threads.load() > 0) {
#pragma omp task firstprivate(i, division) \
shared(partials, divide_op, solve_op, combine_op, num_threads)
{
process_subproblems(i, division);
}
num_threads--;
i++;
division++;
}

while (i != subproblems.end()) {
partials[division] = seq.divide_conquer(std::forward<Input>(*i++),
std::forward<Divider>(divide_op),
std::forward<Predicate>(predicate_op),
std::forward<Solver>(solve_op),
std::forward<Combiner>(combine_op));
}

if (num_threads.load() > 0) {
subresult = divide_conquer(std::forward<Input>(*subproblems.begin()),
std::forward<Divider>(divide_op),
std::forward<Predicate>(predicate_op),
std::forward<Solver>(solve_op),
std::forward<Combiner>(combine_op), num_threads);
}
else {
subresult = seq.divide_conquer(
std::forward<Input>(*subproblems.begin()),
std::forward<Divider>(divide_op),
std::forward<Predicate>(predicate_op),
std::forward<Solver>(solve_op),
std::forward<Combiner>(combine_op));
}
#pragma omp taskwait
}
}
return seq.reduce(partials.begin(), partials.size(),
std::forward<subresult_type>(subresult), combine_op);
}


template<typename Input, typename Divider, typename Solver, typename Combiner>
auto parallel_execution_omp::divide_conquer(
Input && input,
Divider && divide_op,
Solver && solve_op,
Combiner && combine_op,
std::atomic<int> & num_threads) const
{
constexpr sequential_execution seq;
if (num_threads.load() <= 0) {
return seq.divide_conquer(std::forward<Input>(input),
std::forward<Divider>(divide_op), std::forward<Solver>(solve_op),
std::forward<Combiner>(combine_op));
}

auto subproblems = divide_op(std::forward<Input>(input));
if (subproblems.size() <= 1) {
return solve_op(std::forward<Input>(input));
}

using subresult_type =
std::decay_t<typename std::result_of<Solver(Input)>::type>;
std::vector<subresult_type> partials(subproblems.size() - 1);

auto process_subproblems = [&, this](auto it, std::size_t div) {
partials[div] = this->divide_conquer(std::forward<Input>(*it),
std::forward<Divider>(divide_op), std::forward<Solver>(solve_op),
std::forward<Combiner>(combine_op), num_threads);
};

int division = 0;
subresult_type subresult;

#pragma omp parallel
{
#pragma omp single nowait
{
auto i = subproblems.begin() + 1;
while (i != subproblems.end() && num_threads.load() > 0) {
#pragma omp task firstprivate(i, division) \
shared(partials, divide_op, solve_op, combine_op, num_threads)
{
process_subproblems(i, division);
}
num_threads--;
i++;
division++;
}

while (i != subproblems.end()) {
partials[division] = seq.divide_conquer(std::forward<Input>(*i++),
std::forward<Divider>(divide_op), std::forward<Solver>(solve_op),
std::forward<Combiner>(combine_op));
}

if (num_threads.load() > 0) {
subresult = divide_conquer(std::forward<Input>(*subproblems.begin()),
std::forward<Divider>(divide_op), std::forward<Solver>(solve_op),
std::forward<Combiner>(combine_op), num_threads);
}
else {
subresult = seq.divide_conquer(
std::forward<Input>(*subproblems.begin()),
std::forward<Divider>(divide_op), std::forward<Solver>(solve_op),
std::forward<Combiner>(combine_op));
}
#pragma omp taskwait
}
}
return seq.reduce(partials.begin(), partials.size(),
std::forward<subresult_type>(subresult), combine_op);
}

template<typename Queue, typename Consumer,
requires_no_pattern <Consumer>>
void parallel_execution_omp::do_pipeline(Queue & input_queue,
Consumer && consume_op) const
{
using namespace std;
using input_type = typename Queue::value_type;

if (!is_ordered()) {
for (;;) {
auto item = input_queue.pop();
if (!item.first) { break; }
consume_op(*item.first);
}
return;
}

vector<input_type> elements;
long current = 0;
auto item = input_queue.pop();
while (item.first) {
if (current == item.second) {
consume_op(*item.first);
current++;
}
else {
elements.push_back(item);
}
auto it = find_if(elements.begin(), elements.end(),
[&](auto x) { return x.second == current; });
if (it != elements.end()) {
consume_op(*it->first);
elements.erase(it);
current++;
}
item = input_queue.pop();
}
while (elements.size() > 0) {
auto it = find_if(elements.begin(), elements.end(),
[&](auto x) { return x.second == current; });
if (it != elements.end()) {
consume_op(*it->first);
elements.erase(it);
current++;
}
}
}


template<typename Inqueue, typename Transformer, typename output_type,
requires_no_pattern <Transformer>>
void parallel_execution_omp::do_pipeline(Inqueue & input_queue,
Transformer && transform_op,
mpmc_queue <output_type> & output_queue) const
{
using namespace std;

using output_item_value_type = typename output_type::first_type::value_type;
for (;;) {
auto item{input_queue.pop()};
if (!item.first) { break; }
auto out = output_item_value_type{transform_op(*item.first)};
output_queue.push(make_pair(out, item.second));
}
}


template<typename Queue, typename Execution, typename Transformer,
template<typename, typename> class Context,
typename ... OtherTransformers,
requires_context <Context<Execution, Transformer>>>
void parallel_execution_omp::do_pipeline(Queue & input_queue,
Context<Execution, Transformer> && context_op,
OtherTransformers && ... other_ops) const
{
using namespace std;

using input_item_type = typename Queue::value_type;
using input_item_value_type = typename input_item_type::first_type::value_type;

using output_type = typename stage_return_type<input_item_value_type, Transformer>::type;
using output_optional_type = grppi::optional<output_type>;
using output_item_type = pair<output_optional_type, long>;

decltype(auto) output_queue =
get_output_queue<output_item_type>(other_ops...);

#pragma omp task shared(input_queue, context_op, output_queue)
{
context_op.execution_policy().pipeline(input_queue,
context_op.transformer(), output_queue);
output_queue.push(make_pair(output_optional_type{}, -1));
}

do_pipeline(output_queue,
forward<OtherTransformers>(other_ops)...);
#pragma omp taskwait
}

template<typename Queue, typename Transformer, typename ... OtherTransformers,
requires_no_pattern <Transformer>>
void parallel_execution_omp::do_pipeline(
Queue & input_queue,
Transformer && transform_op,
OtherTransformers && ... other_ops) const
{
using namespace std;
using input_type = typename Queue::value_type;
using input_value_type = typename input_type::first_type::value_type;
using result_type = typename result_of<Transformer(input_value_type)>::type;
using output_value_type = grppi::optional<result_type>;
using output_type = pair<output_value_type, long>;

decltype(auto) output_queue =
get_output_queue<output_type>(other_ops...);

#pragma omp task shared(transform_op, input_queue, output_queue)
{
for (;;) {
auto item = input_queue.pop();
if (!item.first) { break; }
auto out = output_value_type{transform_op(*item.first)};
output_queue.push(make_pair(out, item.second));
}
output_queue.push(make_pair(output_value_type{}, -1));
}

do_pipeline(output_queue,
forward<OtherTransformers>(other_ops)...);
}

template<typename Queue, typename FarmTransformer,
template<typename> class Farm,
requires_farm <Farm<FarmTransformer>>>
void parallel_execution_omp::do_pipeline(
Queue & input_queue,
Farm<FarmTransformer> && farm_obj) const
{
using namespace std;

for (int i = 0; i < farm_obj.cardinality(); ++i) {
#pragma omp task shared(farm_obj, input_queue)
{
auto item = input_queue.pop();
while (item.first) {
farm_obj(*item.first);
item = input_queue.pop();
}
input_queue.push(item);
}
}
#pragma omp taskwait
}

template<typename Queue, typename FarmTransformer,
template<typename> class Farm,
typename ... OtherTransformers,
requires_farm <Farm<FarmTransformer>>>
void parallel_execution_omp::do_pipeline(
Queue & input_queue,
Farm<FarmTransformer> && farm_obj,
OtherTransformers && ... other_transform_ops) const
{
using namespace std;

using input_type = typename Queue::value_type;
using input_value_type = typename input_type::first_type::value_type;

using result_type = typename stage_return_type<input_value_type, FarmTransformer>::type;
using output_optional_type = grppi::optional<result_type>;
using output_type = pair<output_optional_type, long>;

decltype(auto) output_queue =
get_output_queue<output_type>(other_transform_ops...);


atomic<int> done_threads{0};
int ntask = farm_obj.cardinality();
for (int i = 0; i < farm_obj.cardinality(); ++i) {
#pragma omp task shared(done_threads, output_queue, farm_obj, input_queue, ntask)
{
do_pipeline(input_queue, farm_obj.transformer(), output_queue);
done_threads++;
if (done_threads == ntask) {
output_queue.push(make_pair(output_optional_type{}, -1));
}
else {
input_queue.push(input_type{});
}
}
}
do_pipeline(output_queue,
forward<OtherTransformers>(other_transform_ops)...);
#pragma omp taskwait
}


template<typename Queue, typename Predicate,
template<typename> class Filter,
requires_filter <Filter<Predicate>>>
void parallel_execution_omp::do_pipeline(
Queue &,
Filter<Predicate> &&) const
{
}

template<typename Queue, typename Predicate,
template<typename> class Filter,
typename ... OtherTransformers,
requires_filter <Filter<Predicate>>>
void parallel_execution_omp::do_pipeline(
Queue & input_queue,
Filter<Predicate> && filter_obj,
OtherTransformers && ... other_transform_ops) const
{
using namespace std;
using input_type = typename Queue::value_type;
using input_value_type = typename input_type::first_type;
auto filter_queue = make_queue<input_type>();

if (is_ordered()) {
auto filter_task = [&]() {
{
auto item{input_queue.pop()};
while (item.first) {
if (filter_obj.keep()) {
if (filter_obj(*item.first)) {
filter_queue.push(item);
}
else {
filter_queue.push(make_pair(input_value_type{}, item.second));
}
}
else {
if (!filter_obj(*item.first)) {
filter_queue.push(item);
}
else {
filter_queue.push(make_pair(input_value_type{}, item.second));
}
}
item = input_queue.pop();
}
filter_queue.push(make_pair(input_value_type{}, -1));
}
};

decltype(auto) output_queue =
get_output_queue<input_type>(other_transform_ops...);


auto reorder_task = [&]() {
vector<input_type> elements;
int current = 0;
long order = 0;
auto item = filter_queue.pop();
for (;;) {
if (!item.first && item.second == -1) { break; }
if (item.second == current) {
if (item.first) {
output_queue.push(make_pair(item.first, order++));
}
current++;
}
else {
elements.push_back(item);
}
auto it = find_if(elements.begin(), elements.end(),
[&](auto x) { return x.second == current; });
if (it != elements.end()) {
if (it->first) {
output_queue.push(make_pair(it->first, order));
order++;
}
elements.erase(it);
current++;
}
item = filter_queue.pop();
}

while (elements.size() > 0) {
auto it = find_if(elements.begin(), elements.end(),
[&](auto x) { return x.second == current; });
if (it != elements.end()) {
if (it->first) {
output_queue.push(make_pair(it->first, order));
order++;
}
elements.erase(it);
current++;
}
item = filter_queue.pop();
}

output_queue.push(item);
};


#pragma omp task shared(filter_queue, filter_obj, input_queue)
{
filter_task();
}

#pragma omp task shared (output_queue, filter_queue)
{
reorder_task();
}

do_pipeline(output_queue,
forward<OtherTransformers>(other_transform_ops)...);

#pragma omp taskwait
}
else {
auto filter_task = [&]() {
auto item = input_queue.pop();
while (item.first) {
if (filter_obj(*item.first)) {
filter_queue.push(item);
}
item = input_queue.pop();
}
filter_queue.push(make_pair(input_value_type{}, -1));
};

#pragma omp task shared(filter_queue, filter_obj, input_queue)
{
filter_task();
}
do_pipeline(filter_queue,
std::forward<OtherTransformers>(other_transform_ops)...);
#pragma omp taskwait
}
}


template<typename Queue, typename Combiner, typename Identity,
template<typename C, typename I> class Reduce,
typename ... OtherTransformers,
requires_reduce <Reduce<Combiner, Identity>>>
void parallel_execution_omp::do_pipeline(
Queue && input_queue,
Reduce<Combiner, Identity> && reduce_obj,
OtherTransformers && ... other_transform_ops) const
{
using namespace std;

using output_item_value_type = grppi::optional<decay_t<Identity>>;
using output_item_type = pair<output_item_value_type, long>;

decltype(auto) output_queue =
get_output_queue<output_item_type>(other_transform_ops...);

auto reduce_task = [&]() {
auto item{input_queue.pop()};
int order = 0;
while (item.first) {
reduce_obj.add_item(std::forward<Identity>(*item.first));
item = input_queue.pop();
if (reduce_obj.reduction_needed()) {
constexpr sequential_execution seq;
auto red = reduce_obj.reduce_window(seq);
output_queue.push(make_pair(red, order++));
}
}
output_queue.push(make_pair(output_item_value_type{}, -1));
};

#pragma omp task shared(reduce_obj, input_queue, output_queue)
{
reduce_task();
}
do_pipeline(output_queue,
std::forward<OtherTransformers>(other_transform_ops)...);
#pragma omp taskwait
}

template<typename Queue, typename Transformer, typename Predicate,
template<typename T, typename P> class Iteration,
typename ... OtherTransformers,
requires_iteration <Iteration<Transformer, Predicate>>,
requires_no_pattern <Transformer>>
void parallel_execution_omp::do_pipeline(
Queue & input_queue,
Iteration<Transformer, Predicate> && iteration_obj,
OtherTransformers && ... other_transform_ops) const
{
using namespace std;

using input_item_type = typename decay_t<Queue>::value_type;
decltype(auto) output_queue =
get_output_queue<input_item_type>(other_transform_ops...);


auto iteration_task = [&]() {
for (;;) {
auto item = input_queue.pop();
if (!item.first) { break; }
auto value = iteration_obj.transform(*item.first);
auto new_item = input_item_type{value, item.second};
if (iteration_obj.predicate(value)) {
output_queue.push(new_item);
}
else {
input_queue.push(new_item);
}
}
while (!input_queue.empty()) {
auto item = input_queue.pop();
auto value = iteration_obj.transform(*item.first);
auto new_item = input_item_type{value, item.second};
if (iteration_obj.predicate(value)) {
output_queue.push(new_item);
}
else {
input_queue.push(new_item);
}
}
output_queue.push(input_item_type{{}, -1});
};

#pragma omp task shared(iteration_obj, input_queue, output_queue)
{
iteration_task();
}
do_pipeline(output_queue,
std::forward<OtherTransformers>(other_transform_ops)...);
#pragma omp taskwait

}

template<typename Queue, typename Transformer, typename Predicate,
template<typename T, typename P> class Iteration,
typename ... OtherTransformers,
requires_iteration <Iteration<Transformer, Predicate>>,
requires_pipeline <Transformer>>
void parallel_execution_omp::do_pipeline(
Queue &,
Iteration<Transformer, Predicate> &&,
OtherTransformers && ...) const
{
static_assert(!is_pipeline<Transformer>, "Not implemented");
}

template<typename Queue, typename ... Transformers,
template<typename...> class Pipeline,
typename ... OtherTransformers,
requires_pipeline <Pipeline<Transformers...>>>
void parallel_execution_omp::do_pipeline(
Queue & input_queue,
Pipeline<Transformers...> && pipeline_obj,
OtherTransformers && ... other_transform_ops) const
{
do_pipeline_nested(
input_queue,
std::tuple_cat(pipeline_obj.transformers(),
std::forward_as_tuple(other_transform_ops...)),
std::make_index_sequence<
sizeof...(Transformers) + sizeof...(OtherTransformers)>());
}

template<typename Queue, typename ... Transformers,
std::size_t ... I>
void parallel_execution_omp::do_pipeline_nested(
Queue & input_queue,
std::tuple<Transformers...> && transform_ops,
std::index_sequence<I...>) const
{
do_pipeline(input_queue,
std::forward<Transformers>(std::get<I>(transform_ops))...);
}

template<typename T, typename... Others>
void parallel_execution_omp::do_pipeline(mpmc_queue <T> &, mpmc_queue <T> &,
Others && ...) const {}


} 

#else 

namespace grppi {


struct parallel_execution_omp {};


template <typename E>
constexpr bool is_parallel_execution_omp() {
return false;
}

}

#endif 

#endif
