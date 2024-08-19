
#pragma once

#include "includes/code_location.h"

#include <unordered_map>
#include <filesystem>
#include <chrono>
#include <thread>
#include <optional>
#include <list>
#include <mutex>


namespace Kratos::Internals {


template <class TTimeUnit>
class Profiler
{
private:
using TimeUnit = TTimeUnit;

using Duration = TimeUnit;

using Clock = std::chrono::high_resolution_clock;

class Item
{
public:
Item(CodeLocation&& rLocation);

private:
Item(std::size_t CallCount,
Duration CumulativeDuration,
Duration MinDuration,
Duration MaxDuration,
CodeLocation&& rLocation);

Item& operator+=(const Item& rOther);

private:
friend class Profiler;

unsigned mRecursionLevel;

std::size_t mCallCount;

Duration mCumulative;

Duration mMin;

Duration mMax;

CodeLocation mLocation;
}; 

struct SourceLocationHash
{
std::size_t operator()(const CodeLocation& r_argument) const
{
std::string string(r_argument.GetFileName());
string.append(std::to_string(r_argument.GetLineNumber()));
return std::hash<std::string>()(string);
}
};

struct SourceLocationEquality
{
bool operator()(const CodeLocation& r_lhs,
const CodeLocation& r_rhs) const
{
return (std::string(r_lhs.GetFileName()) == std::string(r_rhs.GetFileName())) && (r_lhs.GetLineNumber() == r_rhs.GetLineNumber());
}
};

public:
class Scope
{
public:
~Scope();

private:
Scope(Item& rItem);

Scope(Item& rItem, std::chrono::high_resolution_clock::time_point Begin);

Scope(Scope&&) = delete;

Scope(const Scope&) = delete;

Scope& operator=(Scope&&) = delete;

Scope& operator=(const Scope&) = delete;

private:
friend class Profiler;

Item& mrItem;

const std::chrono::high_resolution_clock::time_point mBegin;
}; 

using ItemMap = std::unordered_map<
CodeLocation,
Item,
SourceLocationHash,
SourceLocationEquality
>;

public:
Profiler();

Profiler(Profiler&& rOther) = default;

Profiler(std::filesystem::path&& r_outputPath);

~Profiler();

Profiler& operator=(Profiler&& rOther) = default;

[[nodiscard]] Item& Create(CodeLocation&& rItem);

[[nodiscard]] Scope Profile(Item& rItem);

ItemMap Aggregate() const;

void Write(std::ostream& rStream) const;

private:
Profiler(const Profiler&) = delete;

Profiler& operator=(const Profiler&) = delete;

private:

std::unordered_map<std::thread::id,std::list<Item>> mItemContainerMap;

Item mItem;

std::unique_ptr<Scope> mpScope;

std::filesystem::path mOutputPath;
}; 


template <class T>
std::ostream& operator<<(std::ostream& rStream, const Profiler<T>& rProfiler);


template <class TTimeUnit>
class ProfilerSingleton
{
public:
static Profiler<TTimeUnit>& Get() noexcept;

private:
static std::optional<Profiler<TTimeUnit>> mProfiler;

static std::mutex mMutex;
}; 


} 


#if defined(KRATOS_ENABLE_PROFILING)
#define KRATOS_DEFINE_SCOPE_PROFILER(KRATOS_TIME_UNIT, CODE_LOCATION)                                                     \
thread_local static auto& KRATOS_STATIC_PROFILER_REF = Kratos::Internals::ProfilerSingleton<KRATOS_TIME_UNIT>::Get(); \
thread_local static auto& KRATOS_SCOPE_PROFILED_ITEM = KRATOS_STATIC_PROFILER_REF.Create(CODE_LOCATION);              \
const auto KRATOS_SCOPE_PROFILER = KRATOS_STATIC_PROFILER_REF.Profile(KRATOS_SCOPE_PROFILED_ITEM)

#define KRATOS_PROFILE_SCOPE_MILLI(CODE_LOCATION) KRATOS_DEFINE_SCOPE_PROFILER(std::chrono::milliseconds, CODE_LOCATION)

#define KRATOS_PROFILE_SCOPE_MICRO(CODE_LOCATION) KRATOS_DEFINE_SCOPE_PROFILER(std::chrono::microseconds, CODE_LOCATION)

#define KRATOS_PROFILE_SCOPE_NANO(CODE_LOCATION) KRATOS_DEFINE_SCOPE_PROFILER(std::chrono::nanoseconds, CODE_LOCATION)

#define KRATOS_PROFILE_SCOPE(CODE_LOCATION) KRATOS_PROFILE_SCOPE_MICRO(CODE_LOCATION)

#else
#define KRATOS_PROFILE_SCOPE_MILLI(CODE_LOCATION)

#define KRATOS_PROFILE_SCOPE_MICRO(CODE_LOCATION)

#define KRATOS_PROFILE_SCOPE_NANO(CODE_LOCATION)

#define KRATOS_PROFILE_SCOPE(CODE_LOCATION)

#endif


#include "utilities/profiler_impl.h"
