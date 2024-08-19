
#pragma once

#include <iostream>


#include "includes/define.h"
#include "includes/serializer.h"
#include "includes/key_hash.h"

namespace Kratos {

template<class TDataType>
class GlobalPointer {
private:

TDataType * mDataPointer;
#ifdef KRATOS_USING_MPI
int mRank;
#endif

public:

typedef TDataType element_type;


GlobalPointer() {
mDataPointer = nullptr;
#ifdef KRATOS_USING_MPI
this->mRank = 0;
#endif
};


GlobalPointer(TDataType Data) = delete;


GlobalPointer(TDataType * DataPointer, int Rank = 0)
: mDataPointer(DataPointer)
#ifdef KRATOS_USING_MPI
, mRank(Rank)
#endif
{
#ifndef KRATOS_USING_MPI
KRATOS_DEBUG_ERROR_IF(Rank != 0) << "trying to construct a global pointer with rank different from zero when kratos is not in MPI mode " << std::endl;
#endif
}


GlobalPointer(Kratos::shared_ptr<TDataType> DataPointer, int Rank = 0)
: mDataPointer(DataPointer.get())
#ifdef KRATOS_USING_MPI
, mRank(Rank) 
#endif
{
#ifndef KRATOS_USING_MPI
KRATOS_DEBUG_ERROR_IF(Rank != 0) << "trying to construct a global pointer with rank different from zero when kratos is not in MPI mode " << std::endl;
#endif
}


GlobalPointer(Kratos::intrusive_ptr<TDataType>& DataPointer, int Rank = 0)
: mDataPointer(DataPointer.get())
#ifdef KRATOS_USING_MPI
, mRank(Rank) 
#endif
{
#ifndef KRATOS_USING_MPI
KRATOS_DEBUG_ERROR_IF(Rank != 0) << "trying to construct a global pointer with rank different from zero when kratos is not in MPI mode " << std::endl;
#endif
}


GlobalPointer(Kratos::weak_ptr<TDataType> DataPointer, int Rank = 0)
: mDataPointer(DataPointer.lock().get())
#ifdef KRATOS_USING_MPI
, mRank(Rank) 
#endif
{
#ifndef KRATOS_USING_MPI
KRATOS_DEBUG_ERROR_IF(Rank != 0) << "trying to construct a global pointer with rank different from zero when kratos is not in MPI mode " << std::endl;
#endif
}


GlobalPointer(std::unique_ptr<TDataType> DataPointer, int Rank = 0) = delete;


GlobalPointer(const GlobalPointer & rOther)
: mDataPointer(rOther.mDataPointer)
#ifdef KRATOS_USING_MPI
, mRank(rOther.mRank)
#endif
{
}


GlobalPointer(const GlobalPointer && rOther)
: mDataPointer(std::move(rOther.mDataPointer))
#ifdef KRATOS_USING_MPI
, mRank(std::move(rOther.mRank))
#endif
{
}


GlobalPointer & operator=(const GlobalPointer & rOther) {
mDataPointer = rOther.mDataPointer;
#ifdef KRATOS_USING_MPI
mRank = rOther.mRank;
#endif

return *this;
}


~GlobalPointer() {
}

TDataType* get() {
return mDataPointer;
}

TDataType const* get() const {
return mDataPointer;
}


TDataType & operator*() {
return *mDataPointer;
}


TDataType const& operator*() const {
return *mDataPointer;
}


TDataType * operator->() {
return mDataPointer;
}

TDataType const* operator->() const {
return mDataPointer;
}


void save(char * buffer) const {
memcpy(buffer, this, sizeof(GlobalPointer));
}


void load(char * buffer) {
memcpy(this, buffer, sizeof(GlobalPointer));
}


int GetRank() const {
#ifdef KRATOS_USING_MPI
return this->mRank;
#else
return 0;
#endif
}

private:

friend class Serializer;



void save(Serializer& rSerializer) const
{
if(rSerializer.Is(Serializer::SHALLOW_GLOBAL_POINTERS_SERIALIZATION))
{
rSerializer.save("D", reinterpret_cast<std::size_t>(mDataPointer));
}
else
{
rSerializer.save("D", mDataPointer);
}
#ifdef KRATOS_USING_MPI
rSerializer.save("R", mRank);
#endif
}

void load(Serializer& rSerializer)
{
if(rSerializer.Is(Serializer::SHALLOW_GLOBAL_POINTERS_SERIALIZATION))
{
std::size_t tmp;
rSerializer.load("D", tmp);
mDataPointer = reinterpret_cast<TDataType*>(tmp);
}
else
{
rSerializer.load("D", mDataPointer);
}
#ifdef KRATOS_USING_MPI
rSerializer.load("R", mRank);
#endif
}

};

template< class TDataType >
struct GlobalPointerHasher
{

std::size_t operator()(const GlobalPointer<TDataType>& pGp) const
{
std::size_t seed = 0;
HashCombine(seed, &(*pGp) );
#ifdef KRATOS_USING_MPI
HashCombine(seed, pGp.GetRank());
#endif
return seed;
}
};


template< class TDataType >
struct GlobalPointerComparor
{

bool operator()(const GlobalPointer<TDataType>& pGp1, const GlobalPointer<TDataType>& pGp2) const
{
#ifdef KRATOS_USING_MPI
return ( &(*pGp1) == &(*pGp2)  &&  pGp1.GetRank() == pGp2.GetRank()  );
#else
return ( &(*pGp1) == &(*pGp2) );
#endif
}
};


template< class TDataType >
struct GlobalPointerCompare
{

bool operator()(const GlobalPointer<TDataType>& pGp1, const GlobalPointer<TDataType>& pGp2) const
{
#ifdef KRATOS_USING_MPI
return (pGp1.GetRank() == pGp2.GetRank()) ? (pGp1.get() < pGp2.get()) : (pGp1.GetRank() < pGp2.GetRank());
#else
return (pGp1.get() < pGp2.get());
#endif
}
};

template< class TDataType >
inline std::istream& operator >> (std::istream& rIStream,
GlobalPointer<TDataType>& rThis)
{return rIStream;};

template< class TDataType >
inline std::ostream& operator << (std::ostream& rOStream,
const GlobalPointer<TDataType>& rThis)
{

rOStream << reinterpret_cast<const std::size_t>(&*rThis) << " : " << rThis.GetRank();

return rOStream;
}

} 
