
#include "sqs.hpp"
#if defined(_OPENMP)
#include <omp.h>
#endif
#if defined(USE_MPI)
#include <mpi.h>
#define TAG_BETTER_OBJECTIVE 1
#define TAG_COLLECT 2
#define HEAD_RANK 0
#endif
#include <atomic>
#include <chrono>
#include <signal.h>
#include <unordered_set>
#include <boost/circular_buffer.hpp>
#include <boost/log/trivial.hpp>

namespace sqsgenerator {

volatile sig_atomic_t do_shutdown = 0;
std::atomic<bool> shutdown_requested = false;
static_assert( std::atomic<bool>::is_always_lock_free );

std::string format_sqs_result(const sqsgenerator::SQSResult &result) {
std::stringstream message;
message << "{rank: " << result.rank() << ", objective: " << result.objective() << ", configuration: " << format_vector<sqsgenerator::species_t, int>(result.configuration()) << "}";
return message.str();
}

void log_settings(const std::string &function_name, const IterationSettings &settings) {
std::string mode = ((settings.mode() == random) ? "random" : "systematic");
BOOST_LOG_TRIVIAL(debug) << function_name << "::settings::mode = "  + mode;
BOOST_LOG_TRIVIAL(debug) << function_name << "::settings::num_atoms = " << settings.num_atoms();
BOOST_LOG_TRIVIAL(debug) << function_name << "::settings::num_species = " << settings.num_species();
BOOST_LOG_TRIVIAL(info) << function_name << "::settings::num_shells = " << settings.num_shells();
auto [shells, weights] = settings.shell_indices_and_weights();
BOOST_LOG_TRIVIAL(info) << function_name << "::settings::shell_weights = " + format_dict(shells, weights);
BOOST_LOG_TRIVIAL(info) << function_name << "::settings::num_iterations = " << settings.num_iterations();
BOOST_LOG_TRIVIAL(debug) << function_name << "::settings::num_output_configurations = " << settings.num_output_configurations();
BOOST_LOG_TRIVIAL(debug) << function_name << "::settings::num_pairs = " << settings.pair_list().size();
BOOST_LOG_TRIVIAL(trace) << function_name << "::settings::built_configuration = " << format_vector(settings.structure().rearranged(settings.arrange_backward()).configuration());
BOOST_LOG_TRIVIAL(trace) << function_name << "::settings::built_configuration_internal = " << format_vector(settings.structure().configuration());
BOOST_LOG_TRIVIAL(trace) << function_name << "::settings::packed_configuration = " << format_vector(settings.packed_configuraton());
formatter_t<std::tuple<size_t, size_t>> bounds_formatter = [](const std::tuple<size_t, size_t> &v) {
return "(" + std::to_string(static_cast<int>(std::get<0>(v)))
+ ", " + std::to_string(static_cast<int>(std::get<1>(v))) + ")";
};
BOOST_LOG_TRIVIAL(debug) << function_name << "::settings::shuffling_bounds = " << format_vector(settings.shuffling_bounds(), bounds_formatter);
}

inline
void count_pairs(const configuration_t &configuration, const std::vector<size_t> &pair_list,
std::vector<double> &bonds,
size_t nspecies,
bool clear) {
std::vector<size_t>::difference_type row_size{3};
size_t num_sro_params {nspecies * nspecies};
if (clear) std::fill(bonds.begin(), bonds.end(), 0.0);
for (auto it = pair_list.begin(); it != pair_list.end(); it += row_size) {
auto si {configuration[*it]};
auto sj {configuration[*(it + 1)]};
auto shell{*(it + 2)};
bonds[shell * num_sro_params + sj * nspecies + si]++;
if (si != sj) bonds[shell * num_sro_params + si * nspecies + sj]++;
}
}

inline
double calculate_pair_objective(parameter_storage_t &bonds, const parameter_storage_t &prefactors,
const parameter_storage_t &parameter_weights,
const parameter_storage_t &target_objectives) {
double total_objective{0.0};
size_t nparams{bonds.size()};
for (size_t i = 0; i < nparams; i++) {
bonds[i] = (1.0 - bonds[i] * prefactors[i]);
total_objective += parameter_weights[i] * std::abs(bonds[i] - target_objectives[i]);
}
return total_objective;
}

rank_iteration_map_t compute_ranks(const IterationSettings &settings, const std::vector<int> &threads_per_rank) {
auto thread_count {0};
rank_iteration_map_t rank_map;
int num_mpi_ranks {static_cast<int>(threads_per_rank.size())};
auto nthreads {std::accumulate(threads_per_rank.begin(), threads_per_rank.end(), 0)};

rank_t niterations(settings.num_iterations());
rank_t total = (settings.mode() == random) ? niterations :utils::total_permutations(settings.packed_configuraton());

for (int mpi_rank = 0; mpi_rank < num_mpi_ranks; mpi_rank++) {
auto threads_in_mpi_rank = threads_per_rank[mpi_rank];
std::map<int, std::tuple<rank_t, rank_t>> local_rank_map;
for (int local_thread_id = 0; local_thread_id < threads_in_mpi_rank; local_thread_id++) {
rank_t start_it = total / nthreads * thread_count;
rank_t end_it = start_it + total / nthreads;
if (settings.mode() == systematic) {
start_it++;
end_it++;
}
end_it = (thread_count == nthreads - 1) ? total : end_it;
local_rank_map.emplace(std::make_pair(local_thread_id, std::make_tuple(start_it, end_it)));
thread_count++;
}
rank_map.emplace(std::make_pair(mpi_rank, local_rank_map));
}
assert(thread_count == nthreads);
return rank_map;
}

std::vector<size_t> convert_pair_list(const std::vector<AtomPair> &pair_list) {
std::vector<size_t> result;
pair_shell_matrix_t::index i, j, _, shell_index;
for (const auto &pair : pair_list) {
std::tie(i, j, _, shell_index) = pair;
result.push_back(i);
result.push_back(j);
result.push_back(shell_index);
}
return result;
}

void handle_signal_sigint(int) {
do_shutdown = 1;
shutdown_requested = true;
}

#if defined(USE_MPI)

void handle_signal_sigterm(int) {
do_shutdown = 1;
shutdown_requested = true;
}
#endif

std::tuple<std::vector<SQSResult>, timing_map_t> do_pair_iterations(const IterationSettings &settings) {
typedef boost::circular_buffer<SQSResult> result_buffer_t; 
auto threads_per_rank {settings.threads_per_rank()};

#if defined(_WIN32) || defined(_WIN64)
typedef void (*signal_handler_ptr)(int);
signal_handler_ptr old_sigint_handler, old_sigabrt_handler;
old_sigint_handler = signal(SIGINT, handle_signal_sigint);
old_sigabrt_handler = signal(SIGABRT, handle_signal_sigint);
#else
struct sigaction old_sigint_handler;
{
struct sigaction new_sigint_handler;
new_sigint_handler.sa_handler = handle_signal_sigint;
sigemptyset(&new_sigint_handler.sa_mask);
new_sigint_handler.sa_flags = 0x0;
sigaction(SIGINT, &new_sigint_handler, &old_sigint_handler);
}
#endif

#if defined(USE_MPI)

#if defined(_WIN32) || defined(_WIN64)
signal_handler_ptr old_sigterm_handler;
old_sigterm_handler = signal(SIGTERM, handle_signal_sigterm);
#else
struct sigaction old_sigterm_handler;
{
struct sigaction new_sigterm_handler;
new_sigterm_handler.sa_handler = handle_signal_sigterm;
sigemptyset(&new_sigterm_handler.sa_mask);
new_sigterm_handler.sa_flags = 0x0;
sigaction(SIGTERM, &new_sigterm_handler, &old_sigterm_handler);
}
#endif
#endif

#if defined(USE_MPI)
bool mpi_initialized_in_current_context = false;
int mpi_all_gather_err, mpi_initialized, mpi_thread_level_support_provided, mpi_thread_level_support_required {MPI_THREAD_SERIALIZED};
MPI_Initialized(&mpi_initialized);
if (!mpi_initialized) { 
BOOST_LOG_TRIVIAL(info) << "do_pair_iterations:: MPI Runtime was not initialized. I'm going to do that!";
MPI_Init_thread(nullptr, nullptr, mpi_thread_level_support_required, &mpi_thread_level_support_provided);
mpi_initialized_in_current_context = true;
}
else MPI_Query_thread(&mpi_thread_level_support_provided); 
if (mpi_thread_level_support_provided < mpi_thread_level_support_required) throw std::runtime_error("MPI threading level support is not fulfilling the requirements. I need at least 'MPI_THREAD_SERIALIZED'");

int mpi_num_ranks, mpi_rank;
MPI_Comm_size(MPI_COMM_WORLD, &mpi_num_ranks);
MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

if (mpi_rank == HEAD_RANK) {
if (static_cast<int>(settings.threads_per_rank().size()) != mpi_num_ranks) throw std::runtime_error("Number if ranks does not match the number of entries in the thread map");
BOOST_LOG_TRIVIAL(info) << "do_pair_iterations::mpi::num_ranks = " << mpi_num_ranks;
log_settings("do_pair_iterations", settings);
}
#else
int mpi_rank {0};
int mpi_num_ranks {1};
log_settings("do_pair_iterations", settings);
#endif
if( threads_per_rank[mpi_rank] < 0) {
threads_per_rank[mpi_rank] = omp_get_max_threads();
BOOST_LOG_TRIVIAL(debug) << "do_pair_iterations::rank::" << mpi_rank << "::num_threads_requested::default = " << threads_per_rank[mpi_rank];

}
auto num_threads_per_rank = threads_per_rank[mpi_rank];
BOOST_LOG_TRIVIAL(debug) << "do_pair_iterations::rank::" << mpi_rank << "::num_threads_requested = " << num_threads_per_rank;

timing_map_t thread_timings;
rank_iteration_map_t iteration_ranks;
double best_objective{std::numeric_limits<double>::max()};
rank_t nperms = utils::total_permutations(settings.packed_configuraton());

result_buffer_t results(settings.num_output_configurations());

std::vector<size_t> pair_list(convert_pair_list(settings.pair_list()));
std::vector<size_t> hist(utils::configuration_histogram(settings.packed_configuraton()));

size_t nshells {settings.num_shells()};
size_t nspecies {settings.num_species()};
size_t nparams {nshells * nspecies * nspecies};

shuffling_bounds_t shuffling_bounds (settings.shuffling_bounds());

parameter_storage_t target_objectives (boost::to_flat_vector(settings.target_objective()));
parameter_storage_t parameter_weights (boost::to_flat_vector(settings.parameter_weights()));
parameter_storage_t parameter_prefactors (boost::to_flat_vector(settings.parameter_prefactors()));

omp_set_num_threads(num_threads_per_rank);
#pragma omp parallel default(shared) firstprivate(nspecies, nshells, nparams, mpi_rank, mpi_num_ranks)
{
uint64_t random_seed_local;
double best_objective_local {best_objective};
get_next_configuration_t get_next_configuration;
int thread_id {omp_get_thread_num()}, nthreads {omp_get_num_threads()};
configuration_t configuration_local(settings.packed_configuraton());
parameter_storage_t parameters_local(nparams);


#pragma omp single
{
thread_timings.emplace(mpi_rank,  std::vector<double>(nthreads));
bool need_redistribution_local {nthreads != threads_per_rank[mpi_rank]}, need_redistribution;
#if defined (USE_MPI)
bool need_redistribution_global;
bool need_redistribution_other[mpi_num_ranks]; 
MPI_Allgather(&need_redistribution_local, 1, MPI_CXX_BOOL, need_redistribution_other, 1, MPI_CXX_BOOL, MPI_COMM_WORLD);
for (int i = 0; i < mpi_num_ranks; i++) if (need_redistribution_other[i])  need_redistribution_global = need_redistribution_other[i];
BOOST_LOG_TRIVIAL(trace) << "do_pair_iterations::rank::" << mpi_rank << "::redistribution_requests = " +
format_vector(std::vector<bool>(need_redistribution_other, need_redistribution_other+mpi_num_ranks));

need_redistribution = need_redistribution_local or need_redistribution_global;
MPI_Barrier(MPI_COMM_WORLD);
#else
need_redistribution = need_redistribution_local;
#endif
if (need_redistribution) { 
threads_per_rank[mpi_rank] = nthreads;
#if defined(USE_MPI)
int buf_threads_per_rank[mpi_num_ranks]; 
mpi_all_gather_err = MPI_Allgather(&nthreads, 1, MPI_INT, buf_threads_per_rank, 1, MPI_INT, MPI_COMM_WORLD);
if (mpi_all_gather_err != MPI_SUCCESS) {
BOOST_LOG_TRIVIAL(error) << "do_pair_iterations::rank::" << mpi_rank << ": MPI_Alltoall failed";
throw std::runtime_error("MPI_Alltoall failed on rank #"+std::to_string(mpi_rank));
}
BOOST_LOG_TRIVIAL(trace) << "do_pair_iterations::rank::" << mpi_rank << "::actual_threads_per_rank = " +
format_vector(std::vector<int>(buf_threads_per_rank, buf_threads_per_rank+mpi_num_ranks));
for (int i = 0; i < mpi_num_ranks; i++) threads_per_rank[i] = buf_threads_per_rank[i];
#endif
}

iteration_ranks = compute_ranks(settings, threads_per_rank);
}
#pragma omp barrier
rank_t start_it, end_it;
std::tie(start_it, end_it) = iteration_ranks[mpi_rank][thread_id];
#pragma omp critical
{
BOOST_LOG_TRIVIAL(debug) << "do_pair_iterations::rank::" << mpi_rank << "::thread::" << thread_id << "::iteration_start = " << start_it;
BOOST_LOG_TRIVIAL(debug) << "do_pair_iterations::rank::" << mpi_rank << "::thread::" << thread_id << "::num_iterations = " << (end_it - start_it);
BOOST_LOG_TRIVIAL(debug) << "do_pair_iterations::rank::" << mpi_rank << "::thread::" << thread_id << "::iteration_end = " << end_it;
}

switch (settings.mode()) {
case iteration_mode::random: {
#pragma omp critical
{

auto current_time {std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now().time_since_epoch()).count()};
std::srand(current_time * (thread_id + 1));
random_seed_local = std::rand() * (thread_id + 1);
BOOST_LOG_TRIVIAL(trace) << "do_pair_iterations::rank::" << mpi_rank << "::thread::" << thread_id << "::random_seed = " << random_seed_local;
}
get_next_configuration = [&random_seed_local, &shuffling_bounds](configuration_t &c) {
shuffle_configuration(c, &random_seed_local, shuffling_bounds);
return true;
};
} break;
case iteration_mode::systematic: {
get_next_configuration = next_permutation;
unrank_permutation(configuration_local, hist, nperms, start_it);
} break;
}

time_point_t start_time = std::chrono::high_resolution_clock::now();
rank_t actual_iterations = 0;
for (rank_t i = start_it; i < end_it; i++) {
get_next_configuration(configuration_local);
count_pairs(configuration_local, pair_list, parameters_local, nspecies, true);
auto objective_local = calculate_pair_objective(parameters_local, parameter_prefactors, parameter_weights,
target_objectives);
if (objective_local <= best_objective_local) {

#if defined(USE_MPI)

#pragma omp critical
{
int have_message;
MPI_Request request;
MPI_Status request_status;
MPI_Iprobe(MPI_ANY_SOURCE, TAG_BETTER_OBJECTIVE, MPI_COMM_WORLD, &have_message, &request_status);
while (have_message) {
double global_best_objective;
int source_rank = request_status.MPI_SOURCE;
MPI_Irecv(&global_best_objective, 1, MPI_DOUBLE, source_rank, TAG_BETTER_OBJECTIVE, MPI_COMM_WORLD, &request);
MPI_Wait(&request, &request_status);
if (global_best_objective < best_objective) {
#pragma omp atomic write
best_objective = global_best_objective;
}
MPI_Iprobe(MPI_ANY_SOURCE, TAG_BETTER_OBJECTIVE, MPI_COMM_WORLD, &have_message, &request_status);
}
}
#endif
#if defined(_WIN32) || defined(_WIN64)
best_objective_local = best_objective;
#else
#pragma omp atomic read
best_objective_local = best_objective;
#endif

SQSResult result(objective_local, {-1}, configuration_local, parameters_local);
#pragma omp critical
results.push_back(result);
if (objective_local < best_objective_local){
#if defined(_WIN32) || defined(_WIN64)
#pragma omp critical
best_objective = objective_local;
#else
#pragma omp atomic write
best_objective = objective_local;
#endif

best_objective_local = objective_local;
#if defined(USE_MPI)

#pragma omp critical
{
auto handle_count{0};
MPI_Request req_notify[mpi_num_ranks - 1];
for (int rank = 0; rank < mpi_num_ranks; rank++) {
if (rank == mpi_rank) continue;
MPI_Isend(&objective_local, 1, MPI_DOUBLE, rank, TAG_BETTER_OBJECTIVE, MPI_COMM_WORLD,
&req_notify[handle_count++]);
}
}
#endif
}
}
if (do_shutdown && shutdown_requested.load() ) {
#pragma omp critical
{
BOOST_LOG_TRIVIAL(info) << "do_pair_iterations::rank::" << mpi_rank << "::thread::" << thread_id << "::sigint_received = true";
BOOST_LOG_TRIVIAL(info) << "do_pair_iterations::rank::" << mpi_rank << "::thread::" << thread_id << "::finished_iterations::" << (i - start_it) << "::out_of::" << (end_it - start_it);
}
actual_iterations = i;
break;
}

} 
rank_t actual_iterations_done = (do_shutdown && shutdown_requested.load()) ? actual_iterations - start_it : end_it - start_it;
time_point_t end_time = std::chrono::high_resolution_clock::now();
auto avg_loop_time = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count()) /(long)(actual_iterations_done);
#pragma omp critical
{
BOOST_LOG_TRIVIAL(debug) << "do_pair_iterations::rank::" << mpi_rank << "::thread::" << thread_id << "::avg_loop_time = " << avg_loop_time;
thread_timings[mpi_rank][thread_id] = avg_loop_time;
};
} 

#if defined(_WIN32) || defined(_WIN64)
signal(SIGINT, old_sigint_handler);
signal(SIGABRT, old_sigabrt_handler);
#else
sigaction(SIGINT, &old_sigint_handler, NULL);
#endif

#if defined(USE_MPI)
#if defined(_WIN32) || defined(_WIN64)
signal(SIGTERM, old_sigterm_handler);
#else
sigaction(SIGTERM, &old_sigterm_handler, NULL);
#endif
#endif

if ((do_shutdown && shutdown_requested.load())) {
BOOST_LOG_TRIVIAL(warning) << "do_pair_iterations::interrupt_message = \"Received SIGINT/SIGTERM results may be incomplete\"";


do_shutdown = 0;
shutdown_requested = false;
raise(SIGINT);
}
std::vector<SQSResult> tmp_results, final_results;
for (auto &r : results) tmp_results.push_back(std::move(r));

#if defined (USE_MPI)

int actual_threads_on_rank {static_cast<int>(thread_timings[mpi_rank].size())};
double buf_timings[actual_threads_on_rank];
species_t buf_conf[settings.num_atoms()];
std::vector<int> num_sqs_results, num_actual_threads_on_rank;
if (mpi_rank == HEAD_RANK)  {
num_sqs_results.resize(mpi_num_ranks);
num_actual_threads_on_rank.resize(mpi_num_ranks);
}

int num_results {static_cast<int>(results.size())};
MPI_Gather(&num_results, 1, MPI_INT, num_sqs_results.data(), 1, MPI_INT, HEAD_RANK, MPI_COMM_WORLD);
MPI_Gather(&actual_threads_on_rank, 1, MPI_INT, num_actual_threads_on_rank.data(), 1, MPI_INT, HEAD_RANK, MPI_COMM_WORLD);

double buf_par[nparams], buf_obj;
if (mpi_rank == HEAD_RANK) {
for (int other_rank = 0; other_rank < mpi_num_ranks; other_rank++) {
if (other_rank == HEAD_RANK) continue;
int num_sqs_results_of_rank = num_sqs_results[other_rank];

for (int j = 0; j < num_sqs_results_of_rank; j++) {
MPI_Status status;
MPI_Recv(&buf_par[0], static_cast<int>(nparams), MPI_DOUBLE, other_rank, TAG_COLLECT, MPI_COMM_WORLD, &status);
MPI_Recv(&buf_conf[0], static_cast<int>(settings.num_atoms()), MPI_INT, other_rank, TAG_COLLECT, MPI_COMM_WORLD, &status);
MPI_Recv(&buf_obj, 1, MPI_DOUBLE, other_rank, TAG_COLLECT, MPI_COMM_WORLD, &status);

SQSResult collected(buf_obj, {-1}, configuration_t(&buf_conf[0], &buf_conf[0]+settings.num_atoms()), parameter_storage_t(&buf_par[0], &buf_par[0] + nparams));
BOOST_LOG_TRIVIAL(trace) << "do_pair_iterations::rank::" << mpi_rank << "::got_result::" << j << "::from::" << other_rank << "::which_is = " << format_sqs_result(collected);
tmp_results.push_back(collected);
}
MPI_Status status_recv_timing;
int threads_on_other_rank = num_actual_threads_on_rank[other_rank];
MPI_Recv(&buf_timings[0], threads_on_other_rank, MPI_DOUBLE, other_rank, TAG_COLLECT, MPI_COMM_WORLD, &status_recv_timing);
thread_timings.emplace(other_rank, std::vector<double>(&buf_timings[0], &buf_timings[0]+threads_on_other_rank));
BOOST_LOG_TRIVIAL(trace) << "do_pair_iterations::rank::" << mpi_rank << "::received_thread_timings::from::" << other_rank;
}
}
else {
for (auto &r : tmp_results) {
std::copy(r.storage().begin(), r.storage().end(), buf_par);
std::copy(r.configuration().begin(), r.configuration().end(), buf_conf);
buf_obj = r.objective();
MPI_Send(&buf_par[0], static_cast<int>(nparams), MPI_DOUBLE, HEAD_RANK, TAG_COLLECT, MPI_COMM_WORLD);
MPI_Send(&buf_conf[0], static_cast<int>(settings.num_atoms()), MPI_INT, HEAD_RANK, TAG_COLLECT, MPI_COMM_WORLD);
MPI_Send(&buf_obj, 1, MPI_DOUBLE, HEAD_RANK, TAG_COLLECT, MPI_COMM_WORLD);
BOOST_LOG_TRIVIAL(trace) << "do_pair_iterations::rank::" << mpi_rank << "::send_result_to::" << HEAD_RANK << "::which_is = " << format_sqs_result(r);
}
std::copy(thread_timings[mpi_rank].begin(), thread_timings[mpi_rank].end(), buf_timings);
MPI_Send(&buf_timings[0], actual_threads_on_rank, MPI_DOUBLE, HEAD_RANK, TAG_COLLECT, MPI_COMM_WORLD);
BOOST_LOG_TRIVIAL(trace) << "do_pair_iterations::rank::" << mpi_rank << "::finished_sending_results";
}
#endif
std::unordered_set<rank_t> ranks;
int count = 0;
for (auto &r : tmp_results) {

count++;
configuration_t ordered_configuration(rearrange(r.configuration(), settings.arrange_backward()));
rank_t rank = rank_permutation(ordered_configuration, settings.num_species());
r.set_rank(rank);
r.set_configuration(settings.unpack_configuration(ordered_configuration));
BOOST_LOG_TRIVIAL(trace) << "do_pair_iterations::rank::" << mpi_rank << "::conf = " << format_sqs_result(r);
if (settings.mode() == random && !ranks.insert(rank).second) {
BOOST_LOG_TRIVIAL(trace) << "do_pair_iterations::rank::" << mpi_rank << "::duplicate_configuration = " << rank;
continue;
}
else final_results.push_back(std::move(r));
}
BOOST_LOG_TRIVIAL(info) << "do_pair_iterations::rank::" << mpi_rank << "::num_results = " << final_results.size();

#if defined(USE_MPI)
if (mpi_initialized_in_current_context) MPI_Finalize();
#endif
return std::make_tuple(final_results, thread_timings);
}

SQSResult do_pair_analysis(const IterationSettings &settings) {
log_settings("do_pair_analysis", settings);
size_t nshells {settings.num_shells()},
nspecies{settings.num_species()},
nparams {nshells * nspecies * nspecies};
configuration_t configuration(settings.packed_configuraton());

auto real_structure = settings.structure().rearranged(settings.arrange_backward());
auto real_shell_matrix = real_structure.shell_matrix(settings.shell_distances(), settings.atol(), settings.rtol());
auto real_pair_list = Structure::create_pair_list(real_shell_matrix, settings.shell_weights());
configuration = rearrange(configuration, settings.arrange_backward());
std::vector<size_t> pair_list(convert_pair_list(real_pair_list));
std::vector<size_t> hist(utils::configuration_histogram(settings.packed_configuraton()));

parameter_storage_t target_objectives (boost::to_flat_vector(settings.target_objective()));
parameter_storage_t parameter_weights (boost::to_flat_vector(settings.parameter_weights()));
parameter_storage_t parameter_prefactors (boost::to_flat_vector(settings.parameter_prefactors()));
parameter_storage_t sro_parameters(nparams);

count_pairs(configuration, pair_list, sro_parameters, nspecies, true);
double objective = calculate_pair_objective(sro_parameters, parameter_prefactors, parameter_weights, target_objectives);
rank_t configuration_rank = rank_permutation(configuration, nspecies);
SQSResult result(objective, configuration_rank, settings.unpack_configuration(configuration), sro_parameters);
return result;
}
}