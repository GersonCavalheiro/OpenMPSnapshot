
#pragma once

#include <iostream>
#include <mutex>
#include "includes/ublas_interface.h"
#include "includes/serializer.h"
#include "includes/parallel_environment.h"
#include "containers/distributed_numbering.h"
#include "utilities/communication_coloring_utilities.h"
#include "containers/sparse_graph.h"
#include "containers/sparse_contiguous_row_graph.h"
#include "utilities/parallel_utilities.h"

#include <unordered_map>
#include <unordered_set>

#include "includes/define.h"


namespace Kratos
{








template< class TIndexType=std::size_t >
class DistributedSparseGraph final
{
public:
typedef TIndexType IndexType;
typedef int MpiIndexType;
typedef SparseContiguousRowGraph<IndexType> LocalGraphType; 
typedef SparseGraph<IndexType> NonLocalGraphType; 

KRATOS_CLASS_POINTER_DEFINITION(DistributedSparseGraph);


DistributedSparseGraph(const IndexType LocalSize,
DataCommunicator& rComm)
:
mpComm(&rComm),
mLocalGraph(LocalSize)
{
mNonLocalGraphs.resize(mpComm->Size(),false);
mNonLocalLocks = decltype(mNonLocalLocks)(mpComm->Size());

mpRowNumbering = Kratos::make_unique<DistributedNumbering<IndexType>>(*mpComm,LocalSize);
}


~DistributedSparseGraph() {}

inline const DataCommunicator& GetComm() const
{
return *mpComm;
}

inline const DataCommunicator* pGetComm() const
{
return mpComm;
}

inline const DistributedNumbering<IndexType>& GetRowNumbering() const
{
return *mpRowNumbering;
}

inline IndexType Size() const
{
return mpRowNumbering->Size();
}

inline IndexType LocalSize() const
{
return mpRowNumbering->LocalSize();
}

bool Has(const IndexType GlobalI, const IndexType GlobalJ) const
{
return mLocalGraph.Has(GetRowNumbering().LocalId(GlobalI),GlobalJ);
}

void ComputeLocalMinMaxColumnIndex(IndexType& rMinJ, IndexType& rMaxJ) const
{
rMaxJ = 0;
rMinJ = 0;
for(IndexType local_i = 0; local_i<mLocalGraph.Size(); ++local_i)
{
for(auto J : mLocalGraph[local_i] ) 
{
rMaxJ = std::max(rMaxJ, J);
rMinJ = std::min(rMinJ, J);
}
}
}

IndexType ComputeMaxGlobalColumnIndex() const
{
IndexType MinJ, MaxJ;
ComputeLocalMinMaxColumnIndex(MinJ,MaxJ);
return GetComm().MaxAll(MaxJ);
}


const typename LocalGraphType::GraphType::value_type& operator[](const IndexType& LocalPosition) const
{
return mLocalGraph[LocalPosition];
}

void Clear()
{
mLocalGraph.Clear();
mNonLocalGraphs.clear();
}

void AddEntry(const IndexType RowIndex, const IndexType ColIndex)
{
if(GetRowNumbering().IsLocal(RowIndex))
{
mLocalGraph.AddEntry(GetRowNumbering().LocalId(RowIndex), ColIndex);
}
else
{
IndexType owner = GetRowNumbering().OwnerRank(RowIndex);
const std::lock_guard<LockObject> scope_lock(mNonLocalLocks[owner]);
mNonLocalGraphs[owner].AddEntry(GetRowNumbering().RemoteLocalId(RowIndex,owner), ColIndex);
}
}

template<class TContainerType>
void AddEntries(const IndexType RowIndex, const TContainerType& rColIndices)
{
if(GetRowNumbering().IsLocal(RowIndex))
{
mLocalGraph.AddEntries(GetRowNumbering().LocalId(RowIndex), rColIndices);
}
else
{
IndexType owner = GetRowNumbering().OwnerRank(RowIndex);
mNonLocalLocks[owner].lock();
mNonLocalGraphs[owner].AddEntries(GetRowNumbering().RemoteLocalId(RowIndex,owner), rColIndices);
mNonLocalLocks[owner].unlock();
}
}

template<class TIteratorType>
void AddEntries(const IndexType RowIndex,
const TIteratorType& rColBegin,
const TIteratorType& rColEnd
)
{
if(GetRowNumbering().IsLocal(RowIndex))
{
mLocalGraph.AddEntries(GetRowNumbering().LocalId(RowIndex), rColBegin, rColEnd);
}
else
{
IndexType owner = GetRowNumbering().OwnerRank(RowIndex);
mNonLocalLocks[owner].lock();
mNonLocalGraphs[owner].AddEntries(GetRowNumbering().RemoteLocalId(RowIndex,owner), rColBegin, rColEnd);
mNonLocalLocks[owner].unlock();
}
}

template<class TContainerType>
void AddEntries(const TContainerType& rIndices)
{
for(auto I : rIndices)
{
if(GetRowNumbering().IsLocal(I))
{
mLocalGraph.AddEntries(GetRowNumbering().LocalId(I), rIndices);
}
else
{
IndexType owner = GetRowNumbering().OwnerRank(I);
mNonLocalLocks[owner].lock();
mNonLocalGraphs[owner].AddEntries(GetRowNumbering().RemoteLocalId(I,owner), rIndices);;
mNonLocalLocks[owner].unlock();
}
}
}

template<class TContainerType>
void AddEntries(const TContainerType& rRowIndices, const TContainerType& rColIndices)
{
for(auto I : rRowIndices)
{
AddEntries(I, rColIndices);
}
}


void Finalize()
{
std::vector<MpiIndexType> send_list;

for(unsigned int id = 0; id<mNonLocalGraphs.size(); ++id)
if( !mNonLocalGraphs[id].IsEmpty())
send_list.push_back(id);

auto colors = MPIColoringUtilities::ComputeCommunicationScheduling(send_list, *mpComm);

for(auto color : colors)
{
if(color >= 0) 
{


auto send_single_vector_repr = mNonLocalGraphs[color].ExportSingleVectorRepresentation();
const auto recv_single_vector_repr = mpComm->SendRecv(send_single_vector_repr, color, color);
mLocalGraph.AddFromSingleVectorRepresentation(recv_single_vector_repr);

}
}

}

const LocalGraphType& GetLocalGraph() const
{
return mLocalGraph;
}

const NonLocalGraphType& GetNonLocalGraph(IndexType Rank) const
{
return mNonLocalGraphs[Rank];
}

const DenseVector<NonLocalGraphType>& GetNonLocalGraphs() const
{
return mNonLocalGraphs;
}









std::string Info() const
{
std::stringstream buffer;
buffer << "DistributedSparseGraph" ;
return buffer.str();
}

void PrintInfo(std::ostream& rOStream) const
{
rOStream << "DistributedSparseGraph";
}

void PrintData(std::ostream& rOStream) const {}




protected:















private:


typename DistributedNumbering<IndexType>::UniquePointer mpRowNumbering = nullptr;
const DataCommunicator* mpComm;

LocalGraphType mLocalGraph;
DenseVector<NonLocalGraphType> mNonLocalGraphs;
std::vector<LockObject> mNonLocalLocks;

friend class Serializer;

void save(Serializer& rSerializer) const
{
}

void load(Serializer& rSerializer)
{
}









DistributedSparseGraph& operator=(DistributedSparseGraph const& rOther) = delete;

DistributedSparseGraph(DistributedSparseGraph const& rOther) = delete;


}; 






template<class TIndexType>
inline std::istream& operator >> (std::istream& rIStream,
DistributedSparseGraph<TIndexType>& rThis)
{
return rIStream;
}

template<class TIndexType>
inline std::ostream& operator << (std::ostream& rOStream,
const DistributedSparseGraph<TIndexType>& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}


}  
