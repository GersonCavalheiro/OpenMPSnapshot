
#pragma once

#include <tuple>
#include <limits>
#include <algorithm>
#include <mutex>


#include "includes/define.h"
#include "utilities/atomic_utilities.h"
#include "utilities/parallel_utilities.h"

namespace Kratos
{

namespace Internals
{

template <class TObjectType>
struct NullInitialized
{
static TObjectType Get()
{
return TObjectType();
}
};

template <class TValueType, std::size_t ArraySize>
struct NullInitialized<array_1d<TValueType,ArraySize>>
{
static array_1d<TValueType,ArraySize> Get()
{
array_1d<TValueType,ArraySize> array;
std::fill_n(array.begin(), ArraySize, NullInitialized<TValueType>::Get());
return array;
}
};
} 


/
template<class TDataType, class TReturnType = TDataType>
class SumReduction
{
public:
typedef TDataType   value_type;
typedef TReturnType return_type;

TReturnType mValue = Internals::NullInitialized<TReturnType>::Get(); 

TReturnType GetValue() const
{
return mValue;
}

void LocalReduce(const TDataType value){
mValue += value;
}

void ThreadSafeReduce(const SumReduction<TDataType, TReturnType>& rOther)
{
AtomicAdd(mValue, rOther.mValue);
}
};

template<class TDataType, class TReturnType = TDataType>
class SubReduction
{
public:
typedef TDataType   value_type;
typedef TReturnType return_type;

TReturnType mValue = Internals::NullInitialized<TReturnType>::Get(); 

TReturnType GetValue() const
{
return mValue;
}

void LocalReduce(const TDataType value){
mValue -= value;
}

void ThreadSafeReduce(const SubReduction<TDataType, TReturnType>& rOther)
{
AtomicAdd(mValue, rOther.mValue);
}
};

template<class TDataType, class TReturnType = TDataType>
class MaxReduction
{
public:
typedef TDataType   value_type;
typedef TReturnType return_type;

TReturnType mValue = std::numeric_limits<TReturnType>::lowest(); 

TReturnType GetValue() const
{
return mValue;
}

void LocalReduce(const TDataType value){
mValue = std::max(mValue,value);
}

void ThreadSafeReduce(const MaxReduction<TDataType, TReturnType>& rOther)
{
KRATOS_CRITICAL_SECTION
LocalReduce(rOther.mValue);
}
};

template<class TDataType, class TReturnType = TDataType>
class AbsMaxReduction
{
public:
typedef TDataType   value_type;
typedef TReturnType return_type;

TReturnType mValue = std::numeric_limits<TReturnType>::lowest(); 

TReturnType GetValue() const
{
return mValue;
}

void LocalReduce(const TDataType value){
mValue = (std::abs(mValue) < std::abs(value)) ? value : mValue;
}

void ThreadSafeReduce(const AbsMaxReduction<TDataType, TReturnType>& rOther)
{
KRATOS_CRITICAL_SECTION
LocalReduce(rOther.mValue);
}
};

template<class TDataType, class TReturnType = TDataType>
class MinReduction
{
public:
typedef TDataType   value_type;
typedef TReturnType return_type;

TReturnType mValue = std::numeric_limits<TReturnType>::max(); 

TReturnType GetValue() const
{
return mValue;
}

void LocalReduce(const TDataType value){
mValue = std::min(mValue,value);
}

void ThreadSafeReduce(const MinReduction<TDataType, TReturnType>& rOther)
{
KRATOS_CRITICAL_SECTION
LocalReduce(rOther.mValue);
}
};



template<class TDataType, class TReturnType = TDataType>
class AbsMinReduction
{
public:
typedef TDataType   value_type;
typedef TReturnType return_type;

TReturnType mValue = std::numeric_limits<TReturnType>::max(); 

TReturnType GetValue() const
{
return mValue;
}

void LocalReduce(const TDataType value){
mValue = (std::abs(mValue) < std::abs(value)) ? mValue : value;
}

void ThreadSafeReduce(const AbsMinReduction<TDataType, TReturnType>& rOther)
{
KRATOS_CRITICAL_SECTION
LocalReduce(rOther.mValue);
}
};


template<class TDataType, class TReturnType = std::vector<TDataType>>
class AccumReduction
{
public:
typedef TDataType   value_type;
typedef TReturnType return_type;

TReturnType mValue = TReturnType(); 

TReturnType GetValue() const
{
return mValue;
}

void LocalReduce(const TDataType value){
mValue.insert(mValue.end(), value);
}

void ThreadSafeReduce(const AccumReduction<TDataType, TReturnType>& rOther)
{
KRATOS_CRITICAL_SECTION
std::copy(rOther.mValue.begin(), rOther.mValue.end(), std::inserter(mValue, mValue.end()));
}
};

template<class MapType>
class MapReduction
{
public:
using value_type = typename MapType::value_type;
using return_type = MapType;

return_type mValue;

return_type GetValue() const
{
return mValue;
}

void LocalReduce(const value_type rValue){
mValue.emplace(rValue);
}

void ThreadSafeReduce(MapReduction<MapType>& rOther)
{
KRATOS_CRITICAL_SECTION
mValue.merge(rOther.mValue);
}
};

template <class... Reducer>
struct CombinedReduction {
typedef std::tuple<typename Reducer::value_type...> value_type;
typedef std::tuple<typename Reducer::return_type...> return_type;

std::tuple<Reducer...> mChild;

CombinedReduction() {}

return_type GetValue(){
return_type return_value;
fill_value<0>(return_value);
return return_value;
}

template <int I, class T>
typename std::enable_if<(I < sizeof...(Reducer)), void>::type
fill_value(T& v) {
std::get<I>(v) = std::get<I>(mChild).GetValue();
fill_value<I+1>(v);
};

template <int I, class T>
typename std::enable_if<(I == sizeof...(Reducer)), void>::type
fill_value(T& v) {}

template <class... T>
void LocalReduce(const std::tuple<T...> &&v) {
reduce_local<0>(v);
}

void ThreadSafeReduce(const CombinedReduction &other) {
reduce_global<0>(other);
}

private:

template <int I, class T>
typename std::enable_if<(I < sizeof...(Reducer)), void>::type
reduce_local(T &&v) {
std::get<I>(mChild).LocalReduce(std::get<I>(v));
reduce_local<I+1>(std::forward<T>(v));
};

template <int I, class T>
typename std::enable_if<(I == sizeof...(Reducer)), void>::type
reduce_local(T &&v) {
}

template <int I>
typename std::enable_if<(I < sizeof...(Reducer)), void>::type
reduce_global(const CombinedReduction &other) {
std::get<I>(mChild).ThreadSafeReduce(std::get<I>(other.mChild));
reduce_global<I+1>(other);
}

template <int I>
typename std::enable_if<(I == sizeof...(Reducer)), void>::type
reduce_global(const CombinedReduction &other) {
}
};

}  
