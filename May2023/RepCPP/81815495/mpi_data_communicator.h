
#pragma once

#include <string>
#include <iostream>
#include <mpi.h>


#include "includes/define.h"
#include "includes/data_communicator.h"

#ifndef KRATOS_MPI_DATA_COMMUNICATOR_DECLARE_REDUCE_INTERFACE_FOR_TYPE
#define KRATOS_MPI_DATA_COMMUNICATOR_DECLARE_REDUCE_INTERFACE_FOR_TYPE(type)                                      \
type Sum(const type rLocalValue, const int Root) const override;                                                  \
std::vector<type> Sum(const std::vector<type>& rLocalValues, const int Root) const override;                      \
void Sum(const std::vector<type>& rLocalValues, std::vector<type>& rGlobalValues, const int Root) const override; \
type Min(const type rLocalValue, const int Root) const override;                                                  \
std::vector<type> Min(const std::vector<type>& rLocalValues, const int Root) const override;                      \
void Min(const std::vector<type>& rLocalValues, std::vector<type>& rGlobalValues, const int Root) const override; \
type Max(const type rLocalValue, const int Root) const override;                                                  \
std::vector<type> Max(const std::vector<type>& rLocalValues, const int Root) const override;                      \
void Max(const std::vector<type>& rLocalValues, std::vector<type>& rGlobalValues, const int Root) const override; \

#endif

#ifndef KRATOS_MPI_DATA_COMMUNICATOR_DECLARE_ALLREDUCE_INTERFACE_FOR_TYPE
#define KRATOS_MPI_DATA_COMMUNICATOR_DECLARE_ALLREDUCE_INTERFACE_FOR_TYPE(type)                      \
type SumAll(const type rLocalValue) const override;                                                  \
std::vector<type> SumAll(const std::vector<type>& rLocalValues) const override;                      \
void SumAll(const std::vector<type>& rLocalValues, std::vector<type>& rGlobalValues) const override; \
type MinAll(const type rLocalValue) const override;                                                  \
std::vector<type> MinAll(const std::vector<type>& rLocalValues) const override;                      \
void MinAll(const std::vector<type>& rLocalValues, std::vector<type>& rGlobalValues) const override; \
type MaxAll(const type rLocalValue) const override;                                                  \
std::vector<type> MaxAll(const std::vector<type>& rLocalValues) const override;                      \
void MaxAll(const std::vector<type>& rLocalValues, std::vector<type>& rGlobalValues) const override; \

#endif

#ifndef KRATOS_MPI_DATA_COMMUNICATOR_DECLARE_SCANSUM_INTERFACE_FOR_TYPE
#define KRATOS_MPI_DATA_COMMUNICATOR_DECLARE_SCANSUM_INTERFACE_FOR_TYPE(type)                         \
type ScanSum(const type rLocalValue) const override;                                                  \
std::vector<type> ScanSum(const std::vector<type>& rLocalValues) const override;                      \
void ScanSum(const std::vector<type>& rLocalValues, std::vector<type>& rGlobalValues) const override; \

#endif

#ifndef KRATOS_MPI_DATA_COMMUNICATOR_DECLARE_SENDRECV_INTERFACE_FOR_TYPE
#define KRATOS_MPI_DATA_COMMUNICATOR_DECLARE_SENDRECV_INTERFACE_FOR_TYPE(type)               \
type SendRecvImpl(                                                                           \
const type SendValue, const int SendDestination, const int SendTag,                      \
const int RecvSource, const int RecvTag) const override;                                 \
std::vector<type> SendRecvImpl(const std::vector<type>& rSendValues,                         \
const int SendDestination, const int SendTag,                                            \
const int RecvSource, const int RecvTag) const override;                                 \
void SendRecvImpl(                                                                           \
const type SendValue, const int SendDestination, const int SendTag,                      \
type& RecvValue, const int RecvSource, const int RecvTag) const override;                \
void SendRecvImpl(                                                                           \
const std::vector<type>& rSendValues, const int SendDestination, const int SendTag,      \
std::vector<type>& rRecvValues, const int RecvSource, const int RecvTag) const override; \
void SendImpl(const std::vector<type>& rSendValues,                                          \
const int SendDestination, const int SendTag = 0) const override;                        \
void RecvImpl(std::vector<type>& rRecvValues,                                                \
const int RecvSource, const int RecvTag = 0) const override;                             \

#endif

#ifndef KRATOS_MPI_DATA_COMMUNICATOR_DECLARE_BROADCAST_INTERFACE_FOR_TYPE
#define KRATOS_MPI_DATA_COMMUNICATOR_DECLARE_BROADCAST_INTERFACE_FOR_TYPE(type)         \
void BroadcastImpl(type& rBuffer, const int SourceRank) const override;                 \
void BroadcastImpl(std::vector<type>& rBuffer, const int SourceRank) const override;    \

#endif

#ifndef KRATOS_MPI_DATA_COMMUNICATOR_DECLARE_SCATTER_INTERFACE_FOR_TYPE
#define KRATOS_MPI_DATA_COMMUNICATOR_DECLARE_SCATTER_INTERFACE_FOR_TYPE(type)               \
std::vector<type> Scatter(                                                                  \
const std::vector<type>& rSendValues, const int SourceRank) const override;             \
void Scatter(                                                                               \
const std::vector<type>& rSendValues, std::vector<type>& rRecvValues,                   \
const int SourceRank) const override;                                                   \
std::vector<type> Scatterv(                                                                 \
const std::vector<std::vector<type>>& rSendValues, const int SourceRank) const override;\
void Scatterv(                                                                              \
const std::vector<type>& rSendValues,                                                   \
const std::vector<int>& rSendCounts, const std::vector<int>& rSendOffsets,              \
std::vector<type>& rRecvValues, const int SourceRank) const override;                   \

#endif

#ifndef KRATOS_MPI_DATA_COMMUNICATOR_DECLARE_GATHER_INTERFACE_FOR_TYPE
#define KRATOS_MPI_DATA_COMMUNICATOR_DECLARE_GATHER_INTERFACE_FOR_TYPE(type)                                \
std::vector<type> Gather(const std::vector<type>& rSendValues, const int DestinationRank) const override;   \
void Gather(                                                                                                \
const std::vector<type>& rSendValues, std::vector<type>& rRecvValues,                                   \
const int DestinationRank) const override;                                                              \
std::vector<std::vector<type>> Gatherv(                                                                     \
const std::vector<type>& rSendValues, const int DestinationRank) const override;                        \
void Gatherv(const std::vector<type>& rSendValues,                                                          \
std::vector<type>& rRecvValues,                                                                     \
const std::vector<int>& rRecvCounts,                                                                \
const std::vector<int>& rRecvOffsets,                                                               \
const int DestinationRank) const override;                                                          \
std::vector<type> AllGather(const std::vector<type>& rSendValues) const override;                           \
void AllGather(const std::vector<type>& rSendValues, std::vector<type>& rRecvValues) const override;        \
std::vector<std::vector<type>> AllGatherv(const std::vector<type>& rSendValues) const  override;            \
void AllGatherv(const std::vector<type>& rSendValues, std::vector<type>& rRecvValues,                       \
const std::vector<int>& rRecvCounts, const std::vector<int>& rRecvOffsets) const override;
#endif

#ifndef KRATOS_MPI_DATA_COMMUNICATOR_DECLARE_PUBLIC_INTERFACE_FOR_TYPE
#define KRATOS_MPI_DATA_COMMUNICATOR_DECLARE_PUBLIC_INTERFACE_FOR_TYPE(type)   \
KRATOS_MPI_DATA_COMMUNICATOR_DECLARE_REDUCE_INTERFACE_FOR_TYPE(type)    \
KRATOS_MPI_DATA_COMMUNICATOR_DECLARE_ALLREDUCE_INTERFACE_FOR_TYPE(type) \
KRATOS_MPI_DATA_COMMUNICATOR_DECLARE_SCANSUM_INTERFACE_FOR_TYPE(type)   \
KRATOS_MPI_DATA_COMMUNICATOR_DECLARE_SCATTER_INTERFACE_FOR_TYPE(type)   \
KRATOS_MPI_DATA_COMMUNICATOR_DECLARE_GATHER_INTERFACE_FOR_TYPE(type)    \

#endif

#ifndef KRATOS_MPI_DATA_COMMUNICATOR_DECLARE_IMPLEMENTATION_FOR_TYPE
#define KRATOS_MPI_DATA_COMMUNICATOR_DECLARE_IMPLEMENTATION_FOR_TYPE(type)   \
KRATOS_MPI_DATA_COMMUNICATOR_DECLARE_SENDRECV_INTERFACE_FOR_TYPE(type)  \
KRATOS_MPI_DATA_COMMUNICATOR_DECLARE_BROADCAST_INTERFACE_FOR_TYPE(type) \

#endif

namespace Kratos
{



class KRATOS_API(KRATOS_MPI_CORE) MPIDataCommunicator: public DataCommunicator
{
public:

KRATOS_CLASS_POINTER_DEFINITION(MPIDataCommunicator);


explicit MPIDataCommunicator(MPI_Comm MPIComm);

~MPIDataCommunicator() override;



static MPIDataCommunicator::UniquePointer Create(MPI_Comm MPIComm);

void Barrier() const override;

KRATOS_MPI_DATA_COMMUNICATOR_DECLARE_PUBLIC_INTERFACE_FOR_TYPE(int)
KRATOS_MPI_DATA_COMMUNICATOR_DECLARE_PUBLIC_INTERFACE_FOR_TYPE(unsigned int)
KRATOS_MPI_DATA_COMMUNICATOR_DECLARE_PUBLIC_INTERFACE_FOR_TYPE(long unsigned int)
KRATOS_MPI_DATA_COMMUNICATOR_DECLARE_PUBLIC_INTERFACE_FOR_TYPE(double)
KRATOS_MPI_DATA_COMMUNICATOR_DECLARE_SCATTER_INTERFACE_FOR_TYPE(char)


array_1d<double,3> Sum(const array_1d<double,3>& rLocalValue, const int Root) const override;

array_1d<double,3> Min(const array_1d<double,3>& rLocalValue, const int Root) const override;

array_1d<double,3> Max(const array_1d<double,3>& rLocalValue, const int Root) const override;

bool AndReduce(
const bool Value,
const int Root) const override;

Kratos::Flags AndReduce(
const Kratos::Flags Values,
const Kratos::Flags Mask,
const int Root) const override;

bool OrReduce(
const bool Value,
const int Root) const override;

Kratos::Flags OrReduce(
const Kratos::Flags Values,
const Kratos::Flags Mask,
const int Root) const override;


array_1d<double,3> SumAll(const array_1d<double,3>& rLocalValue) const override;

array_1d<double,3> MinAll(const array_1d<double,3>& rLocalValue) const override;

array_1d<double,3> MaxAll(const array_1d<double,3>& rLocalValue) const override;

bool AndReduceAll(const bool Value) const override;

Kratos::Flags AndReduceAll(const Kratos::Flags Values, const Kratos::Flags Mask) const override;

bool OrReduceAll(const bool Value) const override;

Kratos::Flags OrReduceAll(const Kratos::Flags Values, const Kratos::Flags Mask) const override;



static MPI_Comm GetMPICommunicator(const DataCommunicator& rDataCommunicator);


int Rank() const override;

int Size() const override;

bool IsDistributed() const override;

bool IsDefinedOnThisRank() const override;

bool IsNullOnThisRank() const override;


bool BroadcastErrorIfTrue(bool Condition, const int SourceRank) const override;

bool BroadcastErrorIfFalse(bool Condition, const int SourceRank) const override;

bool ErrorIfTrueOnAnyRank(bool Condition) const override;

bool ErrorIfFalseOnAnyRank(bool Condition) const override;


std::string Info() const override;

void PrintInfo(std::ostream &rOStream) const override;

void PrintData(std::ostream &rOStream) const override;


protected:

KRATOS_MPI_DATA_COMMUNICATOR_DECLARE_IMPLEMENTATION_FOR_TYPE(int)
KRATOS_MPI_DATA_COMMUNICATOR_DECLARE_IMPLEMENTATION_FOR_TYPE(unsigned int)
KRATOS_MPI_DATA_COMMUNICATOR_DECLARE_IMPLEMENTATION_FOR_TYPE(long unsigned int)
KRATOS_MPI_DATA_COMMUNICATOR_DECLARE_IMPLEMENTATION_FOR_TYPE(double)


void BroadcastImpl(std::string& rBroadcastValues, const int SourceRank) const override;


std::string SendRecvImpl(
const std::string& rSendValues, const int SendDestination, const int SendTag,
const int RecvSource, const int RecvTag) const override;

void SendRecvImpl(
const std::string& rSendValues, const int SendDestination, const int SendTag,
std::string& rRecvValues, const int RecvSource, const int RecvTag) const override;

void SendImpl(const std::string& rSendValues, const int SendDestination, const int SendTag = 0) const override;

void RecvImpl(std::string& rRecvValues, const int RecvSource, const int RecvTag = 0) const override;

private:

MPI_Comm mComm;


void CheckMPIErrorCode(const int ierr, const std::string& MPICallName) const;

template<class TDataType> void ReduceDetail(
const TDataType& rLocalValues,
TDataType& rReducedValues,
MPI_Op Operation,
const int Root) const;

template<class TDataType> TDataType ReduceDetail(
const TDataType& rLocalValues,
MPI_Op Operation,
const int Root) const;

template<class TDataType> std::vector<TDataType> ReduceDetailVector(
const std::vector<TDataType>& rLocalValues,
MPI_Op Operation,
const int Root) const;

template<class TDataType> void AllReduceDetail(
const TDataType& rLocalValues,
TDataType& rReducedValues,
MPI_Op Operation) const;

template<class TDataType> TDataType AllReduceDetail(
const TDataType& rLocalValues, MPI_Op Operation) const;

template<class TDataType> std::vector<TDataType> AllReduceDetailVector(
const std::vector<TDataType>& rLocalValues,
MPI_Op Operation) const;

template<class TDataType> void ScanDetail(
const TDataType& rLocalValues,
TDataType& rReducedValues,
MPI_Op Operation) const;

template<class TDataType> TDataType ScanDetail(
const TDataType rLocalValues,
MPI_Op Operation) const;

template<class TDataType> std::vector<TDataType> ScanDetail(
const std::vector<TDataType>& rLocalValues,
MPI_Op Operation) const;

template<class TDataType> void SendRecvDetail(
const TDataType& rSendMessage, const int SendDestination, const int SendTag,
TDataType& rRecvMessage, const int RecvSource, const int RecvTag) const;

template<class TDataType> TDataType SendRecvDetail(
const TDataType& rSendMessage,
const int SendDestination, const int SendTag,
const int RecvSource, const int RecvTag) const;

template<class TDataType> std::vector<TDataType> SendRecvDetail(
const std::vector<TDataType>& rSendMessage,
const int SendDestination, const int SendTag,
const int RecvSource, const int RecvTag) const;

template<class TDataType> void SendDetail(
const TDataType& rSendValues, const int SendDestination, const int SendTag) const;

template<class TDataType> void RecvDetail(
TDataType& rRecvValues, const int RecvSource, const int RecvTag) const;

template<class TDataType> void BroadcastDetail(
TDataType& rBuffer, const int SourceRank) const;

template<class TSendDataType, class TRecvDataType> void ScatterDetail(
const TSendDataType& rSendValues, TRecvDataType& rRecvValues, const int SourceRank) const;

template<class TDataType> std::vector<TDataType> ScatterDetail(
const std::vector<TDataType>& rSendValues, const int SourceRank) const;

template<class TDataType> void ScattervDetail(
const TDataType& rSendValues,
const std::vector<int>& rSendCounts, const std::vector<int>& rSendOffsets,
TDataType& rRecvValues, const int SourceRank) const;

template<class TDataType> std::vector<TDataType> ScattervDetail(
const std::vector<std::vector<TDataType>>& rSendValues,const int SourceRank) const;

template<class TSendDataType, class TRecvDataType> void GatherDetail(
const TSendDataType& rSendValues, TRecvDataType& rRecvValues, const int RecvRank) const;

template<class TDataType> std::vector<TDataType> GatherDetail(
const std::vector<TDataType>& rSendValues, const int DestinationRank) const;

template<class TDataType> void GathervDetail(
const TDataType& rSendValues, TDataType& rRecvValues,
const std::vector<int>& rRecvCounts, const std::vector<int>& rRecvOffsets,
const int RecvRank) const;

template<class TDataType> std::vector<std::vector<TDataType>>
GathervDetail(const std::vector<TDataType>& rSendValues, const int DestinationRank) const;

template<class TDataType> void AllGatherDetail(
const TDataType& rSendValues, TDataType& rRecvValues) const;

template<class TDataType> std::vector<TDataType> AllGatherDetail(
const std::vector<TDataType>& rSendValues) const;

template<class TDataType>
void AllGathervDetail(
const TDataType& rSendValues, TDataType& rRecvValues,
const std::vector<int>& rRecvCounts, const std::vector<int>& rRecvOffsets) const;

template<class TDataType>
std::vector<std::vector<TDataType>> AllGathervDetail(
const std::vector<TDataType>& rSendValues) const;

bool IsEqualOnAllRanks(const int LocalValue) const;

bool IsValidRank(const int Rank) const;

template<class TDataType> void ValidateScattervInput(
const TDataType& rSendValues,
const std::vector<int>& rSendCounts, const std::vector<int>& rSendOffsets,
TDataType& rRecvValues, const int SourceRank) const;

template<class TDataType> void ValidateGathervInput(
const TDataType& rSendValues, TDataType& rRecvValues,
const std::vector<int>& rRecvCounts, const std::vector<int>& rRecvOffsets,
const int RecvRank) const;

template<class TDataType>
void ValidateAllGathervInput(
const TDataType& rSendValues, TDataType& rRecvValues,
const std::vector<int>& rRecvCounts, const std::vector<int>& rRecvOffsets) const;

template<class TDataType> void PrepareScattervBuffers(
const std::vector<std::vector<TDataType>>& rInputMessage,
std::vector<TDataType>& rScattervMessage,
std::vector<int>& rMessageLengths,
std::vector<int>& rMessageDistances,
std::vector<TDataType>& rResult,
const int SourceRank) const;

template<class TDataType> void PrepareGathervBuffers(
const std::vector<TDataType>& rGathervInput,
std::vector<TDataType>& rGathervMessage,
std::vector<int>& rMessageLengths,
std::vector<int>& rMessageDistances,
const int DestinationRank) const;

template<class TDataType>
void PrepareAllGathervBuffers(
const std::vector<TDataType>& rGathervInput,
std::vector<TDataType>& rGathervMessage,
std::vector<int>& rMessageLengths,
std::vector<int>& rMessageDistances) const;

template<class TDataType> void PrepareGathervReturn(
const std::vector<TDataType>& rGathervMessage,
const std::vector<int>& rMessageLengths,
const std::vector<int>& rMessageDistances,
std::vector<std::vector<TDataType>>& rOutputMessage,
const int DestinationRank) const;

template<class TDataType>
void PrepareAllGathervReturn(
const std::vector<TDataType>& rGathervMessage,
const std::vector<int>& rMessageLengths,
const std::vector<int>& rMessageDistances,
std::vector<std::vector<TDataType>>& rOutputMessage) const;

template<class TValue> inline MPI_Datatype MPIDatatype(const TValue&) const;

template<class TContainer> inline void* MPIBuffer(TContainer& rValues) const;

template<class TContainer> inline const void* MPIBuffer(const TContainer& rValues) const;

template<class TContainer> inline int MPIMessageSize(const TContainer& rValues) const;


MPIDataCommunicator(MPIDataCommunicator const &rOther) = delete;

MPIDataCommunicator &operator=(MPIDataCommunicator const &rOther) = delete;


}; 



inline std::istream &operator>>(std::istream &rIStream,
MPIDataCommunicator &rThis)
{
return rIStream;
}

inline std::ostream &operator<<(std::ostream &rOStream,
const MPIDataCommunicator &rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}


} 

#undef KRATOS_MPI_DATA_COMMUNICATOR_DECLARE_REDUCE_INTERFACE_FOR_TYPE
#undef KRATOS_MPI_DATA_COMMUNICATOR_DECLARE_ALLREDUCE_INTERFACE_FOR_TYPE
#undef KRATOS_MPI_DATA_COMMUNICATOR_DECLARE_SCANSUM_INTERFACE_FOR_TYPE
#undef KRATOS_MPI_DATA_COMMUNICATOR_DECLARE_SENDRECV_INTERFACE_FOR_TYPE
#undef KRATOS_MPI_DATA_COMMUNICATOR_DECLARE_BROADCAST_INTERFACE_FOR_TYPE
#undef KRATOS_MPI_DATA_COMMUNICATOR_DECLARE_SCATTER_INTERFACE_FOR_TYPE
#undef KRATOS_MPI_DATA_COMMUNICATOR_DECLARE_GATHER_INTERFACE_FOR_TYPE
#undef KRATOS_MPI_DATA_COMMUNICATOR_DECLARE_PUBLIC_INTERFACE_FOR_TYPE
#undef KRATOS_MPI_DATA_COMMUNICATOR_DECLARE_IMPLEMENTATION_FOR_TYPE
