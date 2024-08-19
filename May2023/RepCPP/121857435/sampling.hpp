#pragma once

#include "defs.hpp"
#include "util/generator.hpp"
#include <algorithm>
#include <gsl/gsl_randist.h>
#include <tuple>

namespace bnmf_algs {
namespace util {

template <typename RandomIterator>
RandomIterator choice(RandomIterator cum_prob_begin,
RandomIterator cum_prob_end,
const bnmf_algs::util::gsl_rng_wrapper& gsl_rng) {
if (cum_prob_begin == cum_prob_end) {
return cum_prob_end;
}

auto max_val = *(std::prev(cum_prob_end));
double p = gsl_ran_flat(gsl_rng.get(), 0, max_val);

return std::upper_bound(cum_prob_begin, cum_prob_end, p);
}


template <typename Real, typename Integer>
void multinomial_mode(Integer num_trials, const vector_t<Real>& prob,
vector_t<Integer>& count, double eps = 1e-50) {
BNMF_ASSERT(prob.cols() == count.cols(),
"Number of event probabilities and counts differ in "
"util::multinomial_mode");
BNMF_ASSERT(
num_trials >= 0,
"Number of trials must be nonnegative in util::multinomial_mode");

if (prob.cols() == 0) {
return;
}

const auto num_events = prob.cols();
vector_t<Real> count_real;
vector_t<Real> diff;
{
vector_t<Real> normalized_probs = prob.array() + eps;
normalized_probs = normalized_probs.array() / normalized_probs.sum();

vector_t<Real> freq =
(num_trials + 0.5 * num_events) * normalized_probs;

count_real = freq.array().floor();
diff = freq - count_real;
count = count_real.template cast<Integer>();
}

const Integer total_count = count.sum();
if (total_count == num_trials) {
return;
}

std::vector<std::pair<int, Real>> fraction_heap;
{
vector_t<Real> fractions;
if (total_count < num_trials) {
fractions = (1 - diff.array()) / (count_real.array() + 1);
} else if (total_count > num_trials) {
fractions = diff.array() / (count_real.array() + eps);
}

for (int i = 0; i < num_events; ++i) {
fraction_heap.emplace_back(i, fractions(i));
}
}

auto ordering = [](const std::pair<int, Real>& elem_left,
const std::pair<int, Real>& elem_right) {
return elem_left.second > elem_right.second;
};
std::make_heap(fraction_heap.begin(), fraction_heap.end(), ordering);


for (Integer i = total_count; i < num_trials; ++i) {
std::pop_heap(fraction_heap.begin(), fraction_heap.end(), ordering);
int min_idx = fraction_heap.back().first;

++count(min_idx);
--diff(min_idx);
fraction_heap.back().second =
(1 - diff(min_idx)) / (count(min_idx) + 1);

std::push_heap(fraction_heap.begin(), fraction_heap.end(), ordering);
}

for (Integer i = total_count; i > num_trials; --i) {
std::pop_heap(fraction_heap.begin(), fraction_heap.end(), ordering);
int min_idx = fraction_heap.back().first;

--count(min_idx);
++diff(min_idx);
fraction_heap.back().second = diff(min_idx) / (count(min_idx) + eps);

std::push_heap(fraction_heap.begin(), fraction_heap.end(), ordering);
}
}


template <typename T>
matrix_t<T> dirichlet_mat(const matrix_t<T>& gamma_shape_mat,
size_t norm_axis) {
BNMF_ASSERT(norm_axis == 0 || norm_axis == 1,
"Axis must be 0 or 1 in util::dirichlet_mat");

matrix_t<T> result(gamma_shape_mat.rows(), gamma_shape_mat.cols());

util::gsl_rng_wrapper rnd_gen(gsl_rng_alloc(gsl_rng_taus), gsl_rng_free);
for (long i = 0; i < result.rows(); ++i) {
for (long j = 0; j < result.cols(); ++j) {
result(i, j) =
gsl_ran_gamma(rnd_gen.get(), gamma_shape_mat(i, j), 1);
}
}

if (norm_axis == 0) {
result = result.array().rowwise() / result.colwise().sum().array();
} else {
result = result.array().colwise() / result.rowwise().sum().array();
}

return result;
}
} 


namespace details {


template <typename Scalar> class SampleOnesNoReplaceComputer {
public:

explicit SampleOnesNoReplaceComputer(const matrix_t<Scalar>& X)
: m_heap(),
no_repl_comp([](const HeapElem& left, const HeapElem& right) {
return left.timestamp >= right.timestamp;
}),
rnd_gen(gsl_rng_alloc(gsl_rng_taus), gsl_rng_free) {

for (int i = 0; i < X.rows(); ++i) {
for (int j = 0; j < X.cols(); ++j) {
Scalar entry = X(i, j);
if (entry > 0) {
double timestamp =
gsl_ran_beta(rnd_gen.get(), entry + 1, 1);

m_heap.emplace_back(timestamp, entry, std::make_pair(i, j));
}
}
}
std::make_heap(m_heap.begin(), m_heap.end(), no_repl_comp);
}

void operator()(size_t curr_step, std::pair<int, int>& prev_val) {
if (m_heap.empty()) {
return;
}
std::pop_heap(m_heap.begin(), m_heap.end(), no_repl_comp);

auto& heap_entry = m_heap.back();
int ii, jj;
std::tie(ii, jj) = heap_entry.idx;

--heap_entry.matrix_entry;
if (heap_entry.matrix_entry <= 0) {
m_heap.pop_back();
} else {
heap_entry.timestamp +=
gsl_ran_beta(rnd_gen.get(), heap_entry.matrix_entry + 1, 1);
std::push_heap(m_heap.begin(), m_heap.end(), no_repl_comp);
}

prev_val.first = ii;
prev_val.second = jj;
}

private:

struct HeapElem {
HeapElem(double timestamp, Scalar matrix_entry, std::pair<int, int> idx)
: timestamp(timestamp), matrix_entry(matrix_entry),
idx(std::move(idx)) {}


double timestamp;


Scalar matrix_entry;


std::pair<int, int> idx;
};

private:

std::vector<HeapElem> m_heap;


std::function<bool(const HeapElem&, const HeapElem&)> no_repl_comp;


util::gsl_rng_wrapper rnd_gen;
};


template <typename Scalar> class SampleOnesReplaceComputer {
public:

explicit SampleOnesReplaceComputer(const matrix_t<Scalar>& X)
: cum_prob(), X_cols(X.cols()), X_sum(X.array().sum()),
rnd_gen(util::gsl_rng_wrapper(gsl_rng_alloc(gsl_rng_taus),
gsl_rng_free)) {

cum_prob = vector_t<Scalar>(X.rows() * X.cols());
auto X_arr = X.array();
cum_prob(0) = X_arr(0);
for (int i = 1; i < X_arr.size(); ++i) {
cum_prob(i) = cum_prob(i - 1) + X_arr(i);
}
}


void operator()(size_t curr_step, std::pair<int, int>& prev_val) {
Scalar* begin = cum_prob.data();
Scalar* end = cum_prob.data() + cum_prob.cols();
auto it = util::choice(begin, end, rnd_gen);
long m = it - begin;

prev_val.first = static_cast<int>(m / X_cols);
prev_val.second = static_cast<int>(m % X_cols);
}

private:
vector_t<Scalar> cum_prob;
long X_cols;
double X_sum;
util::gsl_rng_wrapper rnd_gen;
};

template <typename T> void check_sample_ones_params(const matrix_t<T>& X) {
BNMF_ASSERT(X.array().size() != 0,
"Matrix X must have at least one element");
BNMF_ASSERT((X.array() >= 0).all(), "Matrix X must be nonnegative");
BNMF_ASSERT((X.array() > std::numeric_limits<double>::epsilon()).any(),
"Matrix X must have at least one nonzero element");
}
} 

namespace util {

template <typename T>
util::Generator<std::pair<int, int>, details::SampleOnesNoReplaceComputer<T>>
sample_ones_noreplace(const matrix_t<T>& X) {
details::check_sample_ones_params(X);

auto num_samples = static_cast<size_t>(X.array().sum());
std::pair<int, int> init_val;
util::Generator<std::pair<int, int>,
details::SampleOnesNoReplaceComputer<T>>
gen(init_val, num_samples + 1,
details::SampleOnesNoReplaceComputer<T>(X));

++gen.begin();

return gen;
}


template <typename T>
util::Generator<std::pair<int, int>, details::SampleOnesReplaceComputer<T>>
sample_ones_replace(const matrix_t<T>& X, size_t num_samples) {
details::check_sample_ones_params(X);

std::pair<int, int> init_val;
util::Generator<std::pair<int, int>, details::SampleOnesReplaceComputer<T>>
gen(init_val, num_samples + 1,
details::SampleOnesReplaceComputer<T>(X));

++gen.begin();

return gen;
}
} 
} 
