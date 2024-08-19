



#ifndef BENCHMARK_BENCHMARK_H_
#define BENCHMARK_BENCHMARK_H_

#if __cplusplus >= 201103L || (defined(_MSVC_LANG) && _MSVC_LANG >= 201103L)
#define BENCHMARK_HAS_CXX11
#endif

#include <stdint.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iosfwd>
#include <map>
#include <set>
#include <string>
#include <vector>

#if defined(BENCHMARK_HAS_CXX11)
#include <initializer_list>
#include <type_traits>
#include <utility>
#endif

#if defined(_MSC_VER)
#include <intrin.h>  
#endif

#ifndef BENCHMARK_HAS_CXX11
#define BENCHMARK_DISALLOW_COPY_AND_ASSIGN(TypeName) \
TypeName(const TypeName&);                         \
TypeName& operator=(const TypeName&)
#else
#define BENCHMARK_DISALLOW_COPY_AND_ASSIGN(TypeName) \
TypeName(const TypeName&) = delete;                \
TypeName& operator=(const TypeName&) = delete
#endif

#if defined(__GNUC__)
#define BENCHMARK_UNUSED __attribute__((unused))
#define BENCHMARK_ALWAYS_INLINE __attribute__((always_inline))
#define BENCHMARK_NOEXCEPT noexcept
#define BENCHMARK_NOEXCEPT_OP(x) noexcept(x)
#elif defined(_MSC_VER) && !defined(__clang__)
#define BENCHMARK_UNUSED
#define BENCHMARK_ALWAYS_INLINE __forceinline
#if _MSC_VER >= 1900
#define BENCHMARK_NOEXCEPT noexcept
#define BENCHMARK_NOEXCEPT_OP(x) noexcept(x)
#else
#define BENCHMARK_NOEXCEPT
#define BENCHMARK_NOEXCEPT_OP(x)
#endif
#define __func__ __FUNCTION__
#else
#define BENCHMARK_UNUSED
#define BENCHMARK_ALWAYS_INLINE
#define BENCHMARK_NOEXCEPT
#define BENCHMARK_NOEXCEPT_OP(x)
#endif

#define BENCHMARK_INTERNAL_TOSTRING2(x) #x
#define BENCHMARK_INTERNAL_TOSTRING(x) BENCHMARK_INTERNAL_TOSTRING2(x)

#if defined(__GNUC__) || defined(__clang__)
#define BENCHMARK_BUILTIN_EXPECT(x, y) __builtin_expect(x, y)
#define BENCHMARK_DEPRECATED_MSG(msg) __attribute__((deprecated(msg)))
#else
#define BENCHMARK_BUILTIN_EXPECT(x, y) x
#define BENCHMARK_DEPRECATED_MSG(msg)
#define BENCHMARK_WARNING_MSG(msg)                           \
__pragma(message(__FILE__ "(" BENCHMARK_INTERNAL_TOSTRING( \
__LINE__) ") : warning note: " msg))
#endif

#if defined(__GNUC__) && !defined(__clang__)
#define BENCHMARK_GCC_VERSION (__GNUC__ * 100 + __GNUC_MINOR__)
#endif

#ifndef __has_builtin
#define __has_builtin(x) 0
#endif

#if defined(__GNUC__) || __has_builtin(__builtin_unreachable)
#define BENCHMARK_UNREACHABLE() __builtin_unreachable()
#elif defined(_MSC_VER)
#define BENCHMARK_UNREACHABLE() __assume(false)
#else
#define BENCHMARK_UNREACHABLE() ((void)0)
#endif

namespace benchmark {
class BenchmarkReporter;
class MemoryManager;

void Initialize(int* argc, char** argv);

bool ReportUnrecognizedArguments(int argc, char** argv);

size_t RunSpecifiedBenchmarks();
size_t RunSpecifiedBenchmarks(BenchmarkReporter* display_reporter);
size_t RunSpecifiedBenchmarks(BenchmarkReporter* display_reporter,
BenchmarkReporter* file_reporter);

void RegisterMemoryManager(MemoryManager* memory_manager);

namespace internal {
class Benchmark;
class BenchmarkImp;
class BenchmarkFamilies;

void UseCharPointer(char const volatile*);

Benchmark* RegisterBenchmarkInternal(Benchmark*);

int InitializeStreams();
BENCHMARK_UNUSED static int stream_init_anchor = InitializeStreams();

}  

#if (!defined(__GNUC__) && !defined(__clang__)) || defined(__pnacl__) || \
defined(__EMSCRIPTEN__)
#define BENCHMARK_HAS_NO_INLINE_ASSEMBLY
#endif

#ifndef BENCHMARK_HAS_NO_INLINE_ASSEMBLY
template <class Tp>
inline BENCHMARK_ALWAYS_INLINE void DoNotOptimize(Tp const& value) {
asm volatile("" : : "r,m"(value) : "memory");
}

template <class Tp>
inline BENCHMARK_ALWAYS_INLINE void DoNotOptimize(Tp& value) {
#if defined(__clang__)
asm volatile("" : "+r,m"(value) : : "memory");
#else
asm volatile("" : "+m,r"(value) : : "memory");
#endif
}

inline BENCHMARK_ALWAYS_INLINE void ClobberMemory() {
asm volatile("" : : : "memory");
}
#elif defined(_MSC_VER)
template <class Tp>
inline BENCHMARK_ALWAYS_INLINE void DoNotOptimize(Tp const& value) {
internal::UseCharPointer(&reinterpret_cast<char const volatile&>(value));
_ReadWriteBarrier();
}

inline BENCHMARK_ALWAYS_INLINE void ClobberMemory() { _ReadWriteBarrier(); }
#else
template <class Tp>
inline BENCHMARK_ALWAYS_INLINE void DoNotOptimize(Tp const& value) {
internal::UseCharPointer(&reinterpret_cast<char const volatile&>(value));
}
#endif

class Counter {
public:
enum Flags {
kDefaults = 0,
kIsRate = 1U << 0U,
kAvgThreads = 1U << 1U,
kAvgThreadsRate = kIsRate | kAvgThreads,
kIsIterationInvariant = 1U << 2U,
kIsIterationInvariantRate = kIsRate | kIsIterationInvariant,
kAvgIterations = 1U << 3U,
kAvgIterationsRate = kIsRate | kAvgIterations,

kInvert = 1U << 31U
};

enum OneK {
kIs1000 = 1000,
kIs1024 = 1024
};

double value;
Flags flags;
OneK oneK;

BENCHMARK_ALWAYS_INLINE
Counter(double v = 0., Flags f = kDefaults, OneK k = kIs1000)
: value(v), flags(f), oneK(k) {}

BENCHMARK_ALWAYS_INLINE operator double const&() const { return value; }
BENCHMARK_ALWAYS_INLINE operator double&() { return value; }
};

Counter::Flags inline operator|(const Counter::Flags& LHS,
const Counter::Flags& RHS) {
return static_cast<Counter::Flags>(static_cast<int>(LHS) |
static_cast<int>(RHS));
}

typedef std::map<std::string, Counter> UserCounters;

enum TimeUnit { kNanosecond, kMicrosecond, kMillisecond };

enum BigO { oNone, o1, oN, oNSquared, oNCubed, oLogN, oNLogN, oAuto, oLambda };

typedef uint64_t IterationCount;

typedef double(BigOFunc)(IterationCount);

typedef double(StatisticsFunc)(const std::vector<double>&);

namespace internal {
struct Statistics {
std::string name_;
StatisticsFunc* compute_;

Statistics(const std::string& name, StatisticsFunc* compute)
: name_(name), compute_(compute) {}
};

struct BenchmarkInstance;
class ThreadTimer;
class ThreadManager;

enum AggregationReportMode
#if defined(BENCHMARK_HAS_CXX11)
: unsigned
#else
#endif
{
ARM_Unspecified = 0,
ARM_Default = 1U << 0U,
ARM_FileReportAggregatesOnly = 1U << 1U,
ARM_DisplayReportAggregatesOnly = 1U << 2U,
ARM_ReportAggregatesOnly =
ARM_FileReportAggregatesOnly | ARM_DisplayReportAggregatesOnly
};

}  

class State {
public:
struct StateIterator;
friend struct StateIterator;

BENCHMARK_ALWAYS_INLINE StateIterator begin();
BENCHMARK_ALWAYS_INLINE StateIterator end();

bool KeepRunning();

bool KeepRunningBatch(IterationCount n);

void PauseTiming();

void ResumeTiming();

void SkipWithError(const char* msg);

bool error_occurred() const { return error_occurred_; }

void SetIterationTime(double seconds);

BENCHMARK_ALWAYS_INLINE
void SetBytesProcessed(int64_t bytes) {
counters["bytes_per_second"] =
Counter(static_cast<double>(bytes), Counter::kIsRate, Counter::kIs1024);
}

BENCHMARK_ALWAYS_INLINE
int64_t bytes_processed() const {
if (counters.find("bytes_per_second") != counters.end())
return static_cast<int64_t>(counters.at("bytes_per_second"));
return 0;
}

BENCHMARK_ALWAYS_INLINE
void SetComplexityN(int64_t complexity_n) { complexity_n_ = complexity_n; }

BENCHMARK_ALWAYS_INLINE
int64_t complexity_length_n() const { return complexity_n_; }

BENCHMARK_ALWAYS_INLINE
void SetItemsProcessed(int64_t items) {
counters["items_per_second"] =
Counter(static_cast<double>(items), benchmark::Counter::kIsRate);
}

BENCHMARK_ALWAYS_INLINE
int64_t items_processed() const {
if (counters.find("items_per_second") != counters.end())
return static_cast<int64_t>(counters.at("items_per_second"));
return 0;
}

void SetLabel(const char* label);

void BENCHMARK_ALWAYS_INLINE SetLabel(const std::string& str) {
this->SetLabel(str.c_str());
}

BENCHMARK_ALWAYS_INLINE
int64_t range(std::size_t pos = 0) const {
assert(range_.size() > pos);
return range_[pos];
}

BENCHMARK_DEPRECATED_MSG("use 'range(0)' instead")
int64_t range_x() const { return range(0); }

BENCHMARK_DEPRECATED_MSG("use 'range(1)' instead")
int64_t range_y() const { return range(1); }

BENCHMARK_ALWAYS_INLINE
IterationCount iterations() const {
if (BENCHMARK_BUILTIN_EXPECT(!started_, false)) {
return 0;
}
return max_iterations - total_iterations_ + batch_leftover_;
}

private
:  
IterationCount total_iterations_;

IterationCount batch_leftover_;

public:
const IterationCount max_iterations;

private:
bool started_;
bool finished_;
bool error_occurred_;

private:  
std::vector<int64_t> range_;

int64_t complexity_n_;

public:
UserCounters counters;
const int thread_index;
const int threads;

private:
State(IterationCount max_iters, const std::vector<int64_t>& ranges,
int thread_i, int n_threads, internal::ThreadTimer* timer,
internal::ThreadManager* manager);

void StartKeepRunning();
bool KeepRunningInternal(IterationCount n, bool is_batch);
void FinishKeepRunning();
internal::ThreadTimer* timer_;
internal::ThreadManager* manager_;

friend struct internal::BenchmarkInstance;
};

inline BENCHMARK_ALWAYS_INLINE bool State::KeepRunning() {
return KeepRunningInternal(1, false);
}

inline BENCHMARK_ALWAYS_INLINE bool State::KeepRunningBatch(IterationCount n) {
return KeepRunningInternal(n, true);
}

inline BENCHMARK_ALWAYS_INLINE bool State::KeepRunningInternal(IterationCount n,
bool is_batch) {
assert(n > 0);
assert(is_batch || n == 1);
if (BENCHMARK_BUILTIN_EXPECT(total_iterations_ >= n, true)) {
total_iterations_ -= n;
return true;
}
if (!started_) {
StartKeepRunning();
if (!error_occurred_ && total_iterations_ >= n) {
total_iterations_ -= n;
return true;
}
}
if (is_batch && total_iterations_ != 0) {
batch_leftover_ = n - total_iterations_;
total_iterations_ = 0;
return true;
}
FinishKeepRunning();
return false;
}

struct State::StateIterator {
struct BENCHMARK_UNUSED Value {};
typedef std::forward_iterator_tag iterator_category;
typedef Value value_type;
typedef Value reference;
typedef Value pointer;
typedef std::ptrdiff_t difference_type;

private:
friend class State;
BENCHMARK_ALWAYS_INLINE
StateIterator() : cached_(0), parent_() {}

BENCHMARK_ALWAYS_INLINE
explicit StateIterator(State* st)
: cached_(st->error_occurred_ ? 0 : st->max_iterations), parent_(st) {}

public:
BENCHMARK_ALWAYS_INLINE
Value operator*() const { return Value(); }

BENCHMARK_ALWAYS_INLINE
StateIterator& operator++() {
assert(cached_ > 0);
--cached_;
return *this;
}

BENCHMARK_ALWAYS_INLINE
bool operator!=(StateIterator const&) const {
if (BENCHMARK_BUILTIN_EXPECT(cached_ != 0, true)) return true;
parent_->FinishKeepRunning();
return false;
}

private:
IterationCount cached_;
State* const parent_;
};

inline BENCHMARK_ALWAYS_INLINE State::StateIterator State::begin() {
return StateIterator(this);
}
inline BENCHMARK_ALWAYS_INLINE State::StateIterator State::end() {
StartKeepRunning();
return StateIterator();
}

namespace internal {

typedef void(Function)(State&);

class Benchmark {
public:
virtual ~Benchmark();


Benchmark* Arg(int64_t x);

Benchmark* Unit(TimeUnit unit);

Benchmark* Range(int64_t start, int64_t limit);

Benchmark* DenseRange(int64_t start, int64_t limit, int step = 1);

Benchmark* Args(const std::vector<int64_t>& args);

Benchmark* ArgPair(int64_t x, int64_t y) {
std::vector<int64_t> args;
args.push_back(x);
args.push_back(y);
return Args(args);
}

Benchmark* Ranges(const std::vector<std::pair<int64_t, int64_t> >& ranges);

Benchmark* ArgName(const std::string& name);

Benchmark* ArgNames(const std::vector<std::string>& names);

Benchmark* RangePair(int64_t lo1, int64_t hi1, int64_t lo2, int64_t hi2) {
std::vector<std::pair<int64_t, int64_t> > ranges;
ranges.push_back(std::make_pair(lo1, hi1));
ranges.push_back(std::make_pair(lo2, hi2));
return Ranges(ranges);
}

Benchmark* Apply(void (*func)(Benchmark* benchmark));

Benchmark* RangeMultiplier(int multiplier);

Benchmark* MinTime(double t);

Benchmark* Iterations(IterationCount n);

Benchmark* Repetitions(int n);

Benchmark* ReportAggregatesOnly(bool value = true);

Benchmark* DisplayAggregatesOnly(bool value = true);

Benchmark* MeasureProcessCPUTime();

Benchmark* UseRealTime();

Benchmark* UseManualTime();

Benchmark* Complexity(BigO complexity = benchmark::oAuto);

Benchmark* Complexity(BigOFunc* complexity);

Benchmark* ComputeStatistics(std::string name, StatisticsFunc* statistics);


Benchmark* Threads(int t);

Benchmark* ThreadRange(int min_threads, int max_threads);

Benchmark* DenseThreadRange(int min_threads, int max_threads, int stride = 1);

Benchmark* ThreadPerCpu();

virtual void Run(State& state) = 0;

protected:
explicit Benchmark(const char* name);
Benchmark(Benchmark const&);
void SetName(const char* name);

int ArgsCnt() const;

private:
friend class BenchmarkFamilies;

std::string name_;
AggregationReportMode aggregation_report_mode_;
std::vector<std::string> arg_names_;       
std::vector<std::vector<int64_t> > args_;  
TimeUnit time_unit_;
int range_multiplier_;
double min_time_;
IterationCount iterations_;
int repetitions_;
bool measure_process_cpu_time_;
bool use_real_time_;
bool use_manual_time_;
BigO complexity_;
BigOFunc* complexity_lambda_;
std::vector<Statistics> statistics_;
std::vector<int> thread_counts_;

Benchmark& operator=(Benchmark const&);
};

}  

internal::Benchmark* RegisterBenchmark(const char* name,
internal::Function* fn);

#if defined(BENCHMARK_HAS_CXX11)
template <class Lambda>
internal::Benchmark* RegisterBenchmark(const char* name, Lambda&& fn);
#endif

void ClearRegisteredBenchmarks();

namespace internal {
class FunctionBenchmark : public Benchmark {
public:
FunctionBenchmark(const char* name, Function* func)
: Benchmark(name), func_(func) {}

virtual void Run(State& st);

private:
Function* func_;
};

#ifdef BENCHMARK_HAS_CXX11
template <class Lambda>
class LambdaBenchmark : public Benchmark {
public:
virtual void Run(State& st) { lambda_(st); }

private:
template <class OLambda>
LambdaBenchmark(const char* name, OLambda&& lam)
: Benchmark(name), lambda_(std::forward<OLambda>(lam)) {}

LambdaBenchmark(LambdaBenchmark const&) = delete;

private:
template <class Lam>
friend Benchmark* ::benchmark::RegisterBenchmark(const char*, Lam&&);

Lambda lambda_;
};
#endif

}  

inline internal::Benchmark* RegisterBenchmark(const char* name,
internal::Function* fn) {
return internal::RegisterBenchmarkInternal(
::new internal::FunctionBenchmark(name, fn));
}

#ifdef BENCHMARK_HAS_CXX11
template <class Lambda>
internal::Benchmark* RegisterBenchmark(const char* name, Lambda&& fn) {
using BenchType =
internal::LambdaBenchmark<typename std::decay<Lambda>::type>;
return internal::RegisterBenchmarkInternal(
::new BenchType(name, std::forward<Lambda>(fn)));
}
#endif

#if defined(BENCHMARK_HAS_CXX11) && \
(!defined(BENCHMARK_GCC_VERSION) || BENCHMARK_GCC_VERSION >= 409)
template <class Lambda, class... Args>
internal::Benchmark* RegisterBenchmark(const char* name, Lambda&& fn,
Args&&... args) {
return benchmark::RegisterBenchmark(
name, [=](benchmark::State& st) { fn(st, args...); });
}
#else
#define BENCHMARK_HAS_NO_VARIADIC_REGISTER_BENCHMARK
#endif

class Fixture : public internal::Benchmark {
public:
Fixture() : internal::Benchmark("") {}

virtual void Run(State& st) {
this->SetUp(st);
this->BenchmarkCase(st);
this->TearDown(st);
}

virtual void SetUp(const State&) {}
virtual void TearDown(const State&) {}
virtual void SetUp(State& st) { SetUp(const_cast<const State&>(st)); }
virtual void TearDown(State& st) { TearDown(const_cast<const State&>(st)); }

protected:
virtual void BenchmarkCase(State&) = 0;
};

}  


#if defined(__COUNTER__) && (__COUNTER__ + 1 == __COUNTER__ + 0)
#define BENCHMARK_PRIVATE_UNIQUE_ID __COUNTER__
#else
#define BENCHMARK_PRIVATE_UNIQUE_ID __LINE__
#endif

#define BENCHMARK_PRIVATE_NAME(n) \
BENCHMARK_PRIVATE_CONCAT(_benchmark_, BENCHMARK_PRIVATE_UNIQUE_ID, n)
#define BENCHMARK_PRIVATE_CONCAT(a, b, c) BENCHMARK_PRIVATE_CONCAT2(a, b, c)
#define BENCHMARK_PRIVATE_CONCAT2(a, b, c) a##b##c

#define BENCHMARK_PRIVATE_DECLARE(n)                                 \
static ::benchmark::internal::Benchmark* BENCHMARK_PRIVATE_NAME(n) \
BENCHMARK_UNUSED

#define BENCHMARK(n)                                     \
BENCHMARK_PRIVATE_DECLARE(n) =                         \
(::benchmark::internal::RegisterBenchmarkInternal( \
new ::benchmark::internal::FunctionBenchmark(#n, n)))

#define BENCHMARK_WITH_ARG(n, a) BENCHMARK(n)->Arg((a))
#define BENCHMARK_WITH_ARG2(n, a1, a2) BENCHMARK(n)->Args({(a1), (a2)})
#define BENCHMARK_WITH_UNIT(n, t) BENCHMARK(n)->Unit((t))
#define BENCHMARK_RANGE(n, lo, hi) BENCHMARK(n)->Range((lo), (hi))
#define BENCHMARK_RANGE2(n, l1, h1, l2, h2) \
BENCHMARK(n)->RangePair({{(l1), (h1)}, {(l2), (h2)}})

#ifdef BENCHMARK_HAS_CXX11

#define BENCHMARK_CAPTURE(func, test_case_name, ...)     \
BENCHMARK_PRIVATE_DECLARE(func) =                      \
(::benchmark::internal::RegisterBenchmarkInternal( \
new ::benchmark::internal::FunctionBenchmark(  \
#func "/" #test_case_name,                 \
[](::benchmark::State& st) { func(st, __VA_ARGS__); })))

#endif  

#define BENCHMARK_TEMPLATE1(n, a)                        \
BENCHMARK_PRIVATE_DECLARE(n) =                         \
(::benchmark::internal::RegisterBenchmarkInternal( \
new ::benchmark::internal::FunctionBenchmark(#n "<" #a ">", n<a>)))

#define BENCHMARK_TEMPLATE2(n, a, b)                                         \
BENCHMARK_PRIVATE_DECLARE(n) =                                             \
(::benchmark::internal::RegisterBenchmarkInternal(                     \
new ::benchmark::internal::FunctionBenchmark(#n "<" #a "," #b ">", \
n<a, b>)))

#ifdef BENCHMARK_HAS_CXX11
#define BENCHMARK_TEMPLATE(n, ...)                       \
BENCHMARK_PRIVATE_DECLARE(n) =                         \
(::benchmark::internal::RegisterBenchmarkInternal( \
new ::benchmark::internal::FunctionBenchmark(  \
#n "<" #__VA_ARGS__ ">", n<__VA_ARGS__>)))
#else
#define BENCHMARK_TEMPLATE(n, a) BENCHMARK_TEMPLATE1(n, a)
#endif

#define BENCHMARK_PRIVATE_DECLARE_F(BaseClass, Method)        \
class BaseClass##_##Method##_Benchmark : public BaseClass { \
public:                                                    \
BaseClass##_##Method##_Benchmark() : BaseClass() {        \
this->SetName(#BaseClass "/" #Method);                  \
}                                                         \
\
protected:                                                 \
virtual void BenchmarkCase(::benchmark::State&);          \
};

#define BENCHMARK_TEMPLATE1_PRIVATE_DECLARE_F(BaseClass, Method, a) \
class BaseClass##_##Method##_Benchmark : public BaseClass<a> {    \
public:                                                          \
BaseClass##_##Method##_Benchmark() : BaseClass<a>() {           \
this->SetName(#BaseClass "<" #a ">/" #Method);                \
}                                                               \
\
protected:                                                       \
virtual void BenchmarkCase(::benchmark::State&);                \
};

#define BENCHMARK_TEMPLATE2_PRIVATE_DECLARE_F(BaseClass, Method, a, b) \
class BaseClass##_##Method##_Benchmark : public BaseClass<a, b> {    \
public:                                                             \
BaseClass##_##Method##_Benchmark() : BaseClass<a, b>() {           \
this->SetName(#BaseClass "<" #a "," #b ">/" #Method);            \
}                                                                  \
\
protected:                                                          \
virtual void BenchmarkCase(::benchmark::State&);                   \
};

#ifdef BENCHMARK_HAS_CXX11
#define BENCHMARK_TEMPLATE_PRIVATE_DECLARE_F(BaseClass, Method, ...)       \
class BaseClass##_##Method##_Benchmark : public BaseClass<__VA_ARGS__> { \
public:                                                                 \
BaseClass##_##Method##_Benchmark() : BaseClass<__VA_ARGS__>() {        \
this->SetName(#BaseClass "<" #__VA_ARGS__ ">/" #Method);             \
}                                                                      \
\
protected:                                                              \
virtual void BenchmarkCase(::benchmark::State&);                       \
};
#else
#define BENCHMARK_TEMPLATE_PRIVATE_DECLARE_F(n, a) \
BENCHMARK_TEMPLATE1_PRIVATE_DECLARE_F(n, a)
#endif

#define BENCHMARK_DEFINE_F(BaseClass, Method)    \
BENCHMARK_PRIVATE_DECLARE_F(BaseClass, Method) \
void BaseClass##_##Method##_Benchmark::BenchmarkCase

#define BENCHMARK_TEMPLATE1_DEFINE_F(BaseClass, Method, a)    \
BENCHMARK_TEMPLATE1_PRIVATE_DECLARE_F(BaseClass, Method, a) \
void BaseClass##_##Method##_Benchmark::BenchmarkCase

#define BENCHMARK_TEMPLATE2_DEFINE_F(BaseClass, Method, a, b)    \
BENCHMARK_TEMPLATE2_PRIVATE_DECLARE_F(BaseClass, Method, a, b) \
void BaseClass##_##Method##_Benchmark::BenchmarkCase

#ifdef BENCHMARK_HAS_CXX11
#define BENCHMARK_TEMPLATE_DEFINE_F(BaseClass, Method, ...)            \
BENCHMARK_TEMPLATE_PRIVATE_DECLARE_F(BaseClass, Method, __VA_ARGS__) \
void BaseClass##_##Method##_Benchmark::BenchmarkCase
#else
#define BENCHMARK_TEMPLATE_DEFINE_F(BaseClass, Method, a) \
BENCHMARK_TEMPLATE1_DEFINE_F(BaseClass, Method, a)
#endif

#define BENCHMARK_REGISTER_F(BaseClass, Method) \
BENCHMARK_PRIVATE_REGISTER_F(BaseClass##_##Method##_Benchmark)

#define BENCHMARK_PRIVATE_REGISTER_F(TestName) \
BENCHMARK_PRIVATE_DECLARE(TestName) =        \
(::benchmark::internal::RegisterBenchmarkInternal(new TestName()))

#define BENCHMARK_F(BaseClass, Method)           \
BENCHMARK_PRIVATE_DECLARE_F(BaseClass, Method) \
BENCHMARK_REGISTER_F(BaseClass, Method);       \
void BaseClass##_##Method##_Benchmark::BenchmarkCase

#define BENCHMARK_TEMPLATE1_F(BaseClass, Method, a)           \
BENCHMARK_TEMPLATE1_PRIVATE_DECLARE_F(BaseClass, Method, a) \
BENCHMARK_REGISTER_F(BaseClass, Method);                    \
void BaseClass##_##Method##_Benchmark::BenchmarkCase

#define BENCHMARK_TEMPLATE2_F(BaseClass, Method, a, b)           \
BENCHMARK_TEMPLATE2_PRIVATE_DECLARE_F(BaseClass, Method, a, b) \
BENCHMARK_REGISTER_F(BaseClass, Method);                       \
void BaseClass##_##Method##_Benchmark::BenchmarkCase

#ifdef BENCHMARK_HAS_CXX11
#define BENCHMARK_TEMPLATE_F(BaseClass, Method, ...)                   \
BENCHMARK_TEMPLATE_PRIVATE_DECLARE_F(BaseClass, Method, __VA_ARGS__) \
BENCHMARK_REGISTER_F(BaseClass, Method);                             \
void BaseClass##_##Method##_Benchmark::BenchmarkCase
#else
#define BENCHMARK_TEMPLATE_F(BaseClass, Method, a) \
BENCHMARK_TEMPLATE1_F(BaseClass, Method, a)
#endif

#define BENCHMARK_MAIN()                                                \
int main(int argc, char** argv) {                                     \
::benchmark::Initialize(&argc, argv);                               \
if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1; \
::benchmark::RunSpecifiedBenchmarks();                              \
}                                                                     \
int main(int, char**)


namespace benchmark {

struct CPUInfo {
struct CacheInfo {
std::string type;
int level;
int size;
int num_sharing;
};

int num_cpus;
double cycles_per_second;
std::vector<CacheInfo> caches;
bool scaling_enabled;
std::vector<double> load_avg;

static const CPUInfo& Get();

private:
CPUInfo();
BENCHMARK_DISALLOW_COPY_AND_ASSIGN(CPUInfo);
};

struct SystemInfo {
std::string name;
static const SystemInfo& Get();

private:
SystemInfo();
BENCHMARK_DISALLOW_COPY_AND_ASSIGN(SystemInfo);
};

struct BenchmarkName {
std::string function_name;
std::string args;
std::string min_time;
std::string iterations;
std::string repetitions;
std::string time_type;
std::string threads;

std::string str() const;
};

class BenchmarkReporter {
public:
struct Context {
CPUInfo const& cpu_info;
SystemInfo const& sys_info;
size_t name_field_width;
static const char* executable_name;
Context();
};

struct Run {
static const int64_t no_repetition_index = -1;
enum RunType { RT_Iteration, RT_Aggregate };

Run()
: run_type(RT_Iteration),
error_occurred(false),
iterations(1),
threads(1),
time_unit(kNanosecond),
real_accumulated_time(0),
cpu_accumulated_time(0),
max_heapbytes_used(0),
complexity(oNone),
complexity_lambda(),
complexity_n(0),
report_big_o(false),
report_rms(false),
counters(),
has_memory_result(false),
allocs_per_iter(0.0),
max_bytes_used(0) {}

std::string benchmark_name() const;
BenchmarkName run_name;
RunType run_type;
std::string aggregate_name;
std::string report_label;  
bool error_occurred;
std::string error_message;

IterationCount iterations;
int64_t threads;
int64_t repetition_index;
int64_t repetitions;
TimeUnit time_unit;
double real_accumulated_time;
double cpu_accumulated_time;

double GetAdjustedRealTime() const;

double GetAdjustedCPUTime() const;

double max_heapbytes_used;

BigO complexity;
BigOFunc* complexity_lambda;
int64_t complexity_n;

const std::vector<internal::Statistics>* statistics;

bool report_big_o;
bool report_rms;

UserCounters counters;

bool has_memory_result;
double allocs_per_iter;
int64_t max_bytes_used;
};

BenchmarkReporter();

virtual bool ReportContext(const Context& context) = 0;

virtual void ReportRuns(const std::vector<Run>& report) = 0;

virtual void Finalize() {}

void SetOutputStream(std::ostream* out) {
assert(out);
output_stream_ = out;
}

void SetErrorStream(std::ostream* err) {
assert(err);
error_stream_ = err;
}

std::ostream& GetOutputStream() const { return *output_stream_; }

std::ostream& GetErrorStream() const { return *error_stream_; }

virtual ~BenchmarkReporter();

static void PrintBasicContext(std::ostream* out, Context const& context);

private:
std::ostream* output_stream_;
std::ostream* error_stream_;
};

class ConsoleReporter : public BenchmarkReporter {
public:
enum OutputOptions {
OO_None = 0,
OO_Color = 1,
OO_Tabular = 2,
OO_ColorTabular = OO_Color | OO_Tabular,
OO_Defaults = OO_ColorTabular
};
explicit ConsoleReporter(OutputOptions opts_ = OO_Defaults)
: output_options_(opts_),
name_field_width_(0),
prev_counters_(),
printed_header_(false) {}

virtual bool ReportContext(const Context& context);
virtual void ReportRuns(const std::vector<Run>& reports);

protected:
virtual void PrintRunData(const Run& report);
virtual void PrintHeader(const Run& report);

OutputOptions output_options_;
size_t name_field_width_;
UserCounters prev_counters_;
bool printed_header_;
};

class JSONReporter : public BenchmarkReporter {
public:
JSONReporter() : first_report_(true) {}
virtual bool ReportContext(const Context& context);
virtual void ReportRuns(const std::vector<Run>& reports);
virtual void Finalize();

private:
void PrintRunData(const Run& report);

bool first_report_;
};

class BENCHMARK_DEPRECATED_MSG(
"The CSV Reporter will be removed in a future release") CSVReporter
: public BenchmarkReporter {
public:
CSVReporter() : printed_header_(false) {}
virtual bool ReportContext(const Context& context);
virtual void ReportRuns(const std::vector<Run>& reports);

private:
void PrintRunData(const Run& report);

bool printed_header_;
std::set<std::string> user_counter_names_;
};

class MemoryManager {
public:
struct Result {
Result() : num_allocs(0), max_bytes_used(0) {}

int64_t num_allocs;

int64_t max_bytes_used;
};

virtual ~MemoryManager() {}

virtual void Start() = 0;

virtual void Stop(Result* result) = 0;
};

inline const char* GetTimeUnitString(TimeUnit unit) {
switch (unit) {
case kMillisecond:
return "ms";
case kMicrosecond:
return "us";
case kNanosecond:
return "ns";
}
BENCHMARK_UNREACHABLE();
}

inline double GetTimeUnitMultiplier(TimeUnit unit) {
switch (unit) {
case kMillisecond:
return 1e3;
case kMicrosecond:
return 1e6;
case kNanosecond:
return 1e9;
}
BENCHMARK_UNREACHABLE();
}

}  

#endif  
