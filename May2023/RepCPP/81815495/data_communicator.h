
#pragma once

#include <string>
#include <iostream>
#include <type_traits>


#include "containers/array_1d.h"
#include "containers/flags.h"
#include "includes/define.h"
#include "includes/mpi_serializer.h"

#ifndef KRATOS_DATA_COMMUNICATOR_DEBUG_SIZE_CHECK
#define KRATOS_DATA_COMMUNICATOR_DEBUG_SIZE_CHECK(Size1, Size2, CheckedFunction) \
KRATOS_DEBUG_ERROR_IF(Size1 != Size2) \
<< "Input error in call to DataCommunicator::" << CheckedFunction \
<< ": The sizes of the local and distributed buffers do not match." << std::endl;
#endif


#ifndef KRATOS_BASE_DATA_COMMUNICATOR_DECLARE_REDUCE_INTERFACE_FOR_TYPE
#define KRATOS_BASE_DATA_COMMUNICATOR_DECLARE_REDUCE_INTERFACE_FOR_TYPE(type)                                       \
virtual type Sum(const type rLocalValue, const int Root) const { return rLocalValue; }                              \
virtual std::vector<type> Sum(const std::vector<type>& rLocalValues, const int Root) const {                        \
return rLocalValues;                                                                                            \
}                                                                                                                   \
virtual void Sum(const std::vector<type>& rLocalValues, std::vector<type>& rGlobalValues, const int Root) const {   \
KRATOS_DATA_COMMUNICATOR_DEBUG_SIZE_CHECK(rLocalValues.size(), rGlobalValues.size(), "Sum");                    \
rGlobalValues = Sum(rLocalValues, Root);                                                                        \
}                                                                                                                   \
virtual type Min(const type rLocalValue, const int Root) const { return rLocalValue; }                              \
virtual std::vector<type> Min(const std::vector<type>& rLocalValues, const int Root) const {                        \
return rLocalValues;                                                                                            \
}                                                                                                                   \
virtual void Min(const std::vector<type>& rLocalValues, std::vector<type>& rGlobalValues, const int Root) const {   \
KRATOS_DATA_COMMUNICATOR_DEBUG_SIZE_CHECK(rLocalValues.size(), rGlobalValues.size(), "Min");                    \
rGlobalValues = Min(rLocalValues, Root);                                                                        \
}                                                                                                                   \
virtual type Max(const type rLocalValue, const int Root) const { return rLocalValue; }                              \
virtual std::vector<type> Max(const std::vector<type>& rLocalValues, const int Root) const {                        \
return rLocalValues;                                                                                            \
}                                                                                                                   \
virtual void Max(const std::vector<type>& rLocalValues, std::vector<type>& rGlobalValues, const int Root) const {   \
KRATOS_DATA_COMMUNICATOR_DEBUG_SIZE_CHECK(rLocalValues.size(), rGlobalValues.size(), "Max");                    \
rGlobalValues = Max(rLocalValues, Root);                                                                        \
}                                                                                                                   \

#endif


#ifndef KRATOS_BASE_DATA_COMMUNICATOR_DECLARE_ALLREDUCE_INTERFACE_FOR_TYPE
#define KRATOS_BASE_DATA_COMMUNICATOR_DECLARE_ALLREDUCE_INTERFACE_FOR_TYPE(type)                        \
virtual type SumAll(const type rLocalValue) const { return rLocalValue; }                               \
virtual std::vector<type> SumAll(const std::vector<type>& rLocalValues) const {                         \
return rLocalValues;                                                                                \
}                                                                                                       \
virtual void SumAll(const std::vector<type>& rLocalValues, std::vector<type>& rGlobalValues) const {    \
KRATOS_DATA_COMMUNICATOR_DEBUG_SIZE_CHECK(rLocalValues.size(), rGlobalValues.size(), "SumAll");     \
rGlobalValues = SumAll(rLocalValues);                                                               \
}                                                                                                       \
virtual type MinAll(const type rLocalValue) const { return rLocalValue; }                               \
virtual std::vector<type> MinAll(const std::vector<type>& rLocalValues) const {                         \
return rLocalValues;                                                                                \
}                                                                                                       \
virtual void MinAll(const std::vector<type>& rLocalValues, std::vector<type>& rGlobalValues) const {    \
KRATOS_DATA_COMMUNICATOR_DEBUG_SIZE_CHECK(rLocalValues.size(), rGlobalValues.size(), "MinAll");     \
rGlobalValues = MinAll(rLocalValues);                                                               \
}                                                                                                       \
virtual type MaxAll(const type rLocalValue) const { return rLocalValue; }                               \
virtual std::vector<type> MaxAll(const std::vector<type>& rLocalValues) const {                         \
return rLocalValues;                                                                                \
}                                                                                                       \
virtual void MaxAll(const std::vector<type>& rLocalValues, std::vector<type>& rGlobalValues) const {    \
KRATOS_DATA_COMMUNICATOR_DEBUG_SIZE_CHECK(rLocalValues.size(), rGlobalValues.size(), "MaxAll");     \
rGlobalValues = MaxAll(rLocalValues);                                                               \
}                                                                                                       \

#endif


#ifndef KRATOS_BASE_DATA_COMMUNICATOR_DECLARE_SCANSUM_INTERFACE_FOR_TYPE
#define KRATOS_BASE_DATA_COMMUNICATOR_DECLARE_SCANSUM_INTERFACE_FOR_TYPE(type)                      \
virtual type ScanSum(const type rLocalValue) const { return rLocalValue; }                          \
virtual std::vector<type> ScanSum(const std::vector<type>& rLocalValues) const {                    \
return rLocalValues;                                                                            \
}                                                                                                   \
virtual void ScanSum(const std::vector<type>& rLocalValues, std::vector<type>& rPartialSums) const {\
KRATOS_DATA_COMMUNICATOR_DEBUG_SIZE_CHECK(rLocalValues.size(), rPartialSums.size(), "ScanSum"); \
rPartialSums = ScanSum(rLocalValues);                                                           \
}                                                                                                   \

#endif


#ifndef KRATOS_BASE_DATA_COMMUNICATOR_DECLARE_SENDRECV_INTERFACE_FOR_TYPE
#define KRATOS_BASE_DATA_COMMUNICATOR_DECLARE_SENDRECV_INTERFACE_FOR_TYPE(type)                                 \
virtual type SendRecvImpl(                                                                                      \
const type rSendValues, const int SendDestination, const int SendTag,                                       \
const int RecvSource, const int RecvTag) const {                                                            \
KRATOS_ERROR_IF( (Rank() != SendDestination) || (Rank() != RecvSource))                                     \
<< "Communication between different ranks is not possible with a serial DataCommunicator." << std::endl;    \
return rSendValues;                                                                                         \
}                                                                                                               \
virtual std::vector<type> SendRecvImpl(                                                                         \
const std::vector<type>& rSendValues, const int SendDestination, const int SendTag,                         \
const int RecvSource, const int RecvTag) const {                                                            \
KRATOS_ERROR_IF( (Rank() != SendDestination) || (Rank() != RecvSource))                                     \
<< "Communication between different ranks is not possible with a serial DataCommunicator." << std::endl;    \
return rSendValues;                                                                                         \
}                                                                                                               \
virtual void SendRecvImpl(                                                                                      \
const type rSendValues, const int SendDestination, const int SendTag,                                       \
type& rRecvValues, const int RecvSource, const int RecvTag) const {                                         \
rRecvValues = SendRecvImpl(rSendValues, SendDestination, SendTag, RecvSource, RecvTag);                     \
}                                                                                                               \
virtual void SendRecvImpl(                                                                                      \
const std::vector<type>& rSendValues, const int SendDestination, const int SendTag,                         \
std::vector<type>& rRecvValues, const int RecvSource, const int RecvTag) const {                            \
KRATOS_DATA_COMMUNICATOR_DEBUG_SIZE_CHECK(rSendValues.size(), rRecvValues.size(), "SendRecv");              \
rRecvValues = SendRecvImpl(rSendValues, SendDestination, SendTag, RecvSource, RecvTag);                     \
}                                                                                                               \
virtual void SendImpl(                                                                                          \
const std::vector<type>& rSendValues, const int SendDestination, const int SendTag = 0) const {             \
KRATOS_ERROR_IF(Rank() != SendDestination)                                                                  \
<< "Communication between different ranks is not possible with a serial DataCommunicator." << std::endl;    \
}                                                                                                               \
virtual void RecvImpl(std::vector<type>& rRecvValues, const int RecvSource, const int RecvTag = 0) const {      \
KRATOS_ERROR << "Calling serial DataCommunicator::Recv, which has no meaningful return." << std::endl;      \
}                                                                                                               \

#endif


#ifndef KRATOS_BASE_DATA_COMMUNICATOR_DECLARE_BROADCAST_INTERFACE_FOR_TYPE
#define KRATOS_BASE_DATA_COMMUNICATOR_DECLARE_BROADCAST_INTERFACE_FOR_TYPE(type)        \
virtual void BroadcastImpl(type& rBuffer, const int SourceRank) const {}                \
virtual void BroadcastImpl(std::vector<type>& rBuffer, const int SourceRank) const {}   \

#endif


#ifndef KRATOS_BASE_DATA_COMMUNICATOR_DECLARE_SCATTER_INTERFACE_FOR_TYPE
#define KRATOS_BASE_DATA_COMMUNICATOR_DECLARE_SCATTER_INTERFACE_FOR_TYPE(type)                                                              \
virtual std::vector<type> Scatter(const std::vector<type>& rSendValues, const int SourceRank) const {                                       \
KRATOS_ERROR_IF( Rank() != SourceRank )                                                                                                \
<< "Communication between different ranks is not possible with a serial DataCommunicator." << std::endl;                                \
return rSendValues;                                                                                                                     \
}                                                                                                                                           \
virtual void Scatter(                                                                                                                       \
const std::vector<type>& rSendValues, std::vector<type>& rRecvValues, const int SourceRank) const {                                     \
KRATOS_DATA_COMMUNICATOR_DEBUG_SIZE_CHECK(rSendValues.size(),rRecvValues.size(),"Scatter");                                             \
rRecvValues = Scatter(rSendValues, SourceRank);                                                                                         \
}                                                                                                                                           \
virtual std::vector<type> Scatterv(const std::vector<std::vector<type>>& rSendValues, const int SourceRank) const {                         \
KRATOS_ERROR_IF( Rank() != SourceRank )                                                                                                 \
<< "Communication between different ranks is not possible with a serial DataCommunicator." << std::endl;                                \
KRATOS_ERROR_IF( static_cast<unsigned int>(Size()) != rSendValues.size() )                                                              \
<< "Unexpected number of sends in DataCommuncatior::Scatterv (serial DataCommunicator always assumes a single process)." << std::endl;  \
return rSendValues[0];                                                                                                                  \
}                                                                                                                                           \
virtual void Scatterv(                                                                                                                      \
const std::vector<type>& rSendValues, const std::vector<int>& rSendCounts, const std::vector<int>& rSendOffsets,                        \
std::vector<type>& rRecvValues, const int SourceRank) const {                                                                           \
KRATOS_DATA_COMMUNICATOR_DEBUG_SIZE_CHECK(rRecvValues.size(), rSendValues.size(), "Scatterv (values check)");                           \
KRATOS_DATA_COMMUNICATOR_DEBUG_SIZE_CHECK(rSendCounts.size(), 1, "Scatterv (counts check)");                                            \
KRATOS_DATA_COMMUNICATOR_DEBUG_SIZE_CHECK(rSendOffsets.size(), 1, "Scatterv (offsets check)");                                          \
KRATOS_ERROR_IF( Rank() != SourceRank )                                                                                                 \
<< "Communication between different ranks is not possible with a serial DataCommunicator." << std::endl;                                \
rRecvValues = rSendValues;                                                                                                              \
}                                                                                                                                           \

#endif


#ifndef KRATOS_BASE_DATA_COMMUNICATOR_DECLARE_GATHER_INTERFACE_FOR_TYPE
#define KRATOS_BASE_DATA_COMMUNICATOR_DECLARE_GATHER_INTERFACE_FOR_TYPE(type)                                       \
virtual std::vector<type> Gather(const std::vector<type>& rSendValues, const int DestinationRank) const {           \
KRATOS_ERROR_IF( Rank() != DestinationRank )                                                                    \
<< "Communication between different ranks is not possible with a serial DataCommunicator." << std::endl;        \
return rSendValues;                                                                                             \
}                                                                                                                   \
virtual void Gather(                                                                                                \
const std::vector<type>& rSendValues, std::vector<type>& rRecvValues, const int DestinationRank) const {        \
KRATOS_DATA_COMMUNICATOR_DEBUG_SIZE_CHECK(rSendValues.size(),rRecvValues.size(),"Gather");                      \
rRecvValues = Gather(rSendValues, DestinationRank);                                                             \
}                                                                                                                   \
virtual std::vector<std::vector<type>> Gatherv(                                                                     \
const std::vector<type>& rSendValues, const int DestinationRank) const {                                        \
KRATOS_ERROR_IF( Rank() != DestinationRank )                                                                    \
<< "Communication between different ranks is not possible with a serial DataCommunicator." << std::endl;        \
return std::vector<std::vector<type>>{rSendValues};                                                             \
}                                                                                                                   \
virtual void Gatherv(                                                                                               \
const std::vector<type>& rSendValues, std::vector<type>& rRecvValues,                                           \
const std::vector<int>& rRecvCounts, const std::vector<int>& rRecvOffsets, const int DestinationRank) const {   \
KRATOS_DATA_COMMUNICATOR_DEBUG_SIZE_CHECK(rRecvValues.size(), rSendValues.size(), "Gatherv (values check)");    \
KRATOS_DATA_COMMUNICATOR_DEBUG_SIZE_CHECK(rRecvCounts.size(), 1, "Gatherv (counts check)");                     \
KRATOS_DATA_COMMUNICATOR_DEBUG_SIZE_CHECK(rRecvOffsets.size(), 1, "Gatherv (offset check)");                    \
KRATOS_ERROR_IF( Rank() != DestinationRank )                                                                    \
<< "Communication between different ranks is not possible with a serial DataCommunicator." << std::endl;        \
rRecvValues = rSendValues;                                                                                      \
}                                                                                                                   \
virtual std::vector<type> AllGather(const std::vector<type>& rSendValues) const { return rSendValues; }             \
virtual void AllGather(const std::vector<type>& rSendValues, std::vector<type>& rRecvValues) const {                \
KRATOS_DATA_COMMUNICATOR_DEBUG_SIZE_CHECK(rSendValues.size(),rRecvValues.size(),"AllGather");                   \
rRecvValues = AllGather(rSendValues);                                                                           \
}                                                                                                                   \
virtual std::vector<std::vector<type>> AllGatherv(const std::vector<type>& rSendValues) const {                     \
return std::vector<std::vector<type>>{rSendValues};                                                             \
}                                                                                                                   \
virtual void AllGatherv(const std::vector<type>& rSendValues, std::vector<type>& rRecvValues,                       \
const std::vector<int>& rRecvCounts, const std::vector<int>& rRecvOffsets) const {                              \
KRATOS_DATA_COMMUNICATOR_DEBUG_SIZE_CHECK(rRecvValues.size(), rSendValues.size(), "AllGatherv (values check)"); \
KRATOS_DATA_COMMUNICATOR_DEBUG_SIZE_CHECK(rRecvCounts.size(), 1, "AllGatherv (counts check)");                  \
KRATOS_DATA_COMMUNICATOR_DEBUG_SIZE_CHECK(rRecvOffsets.size(), 1, "AllGatherv (offset check)");                 \
rRecvValues = rSendValues;                                                                          \
}
#endif

#ifndef KRATOS_BASE_DATA_COMMUNICATOR_DECLARE_PUBLIC_INTERFACE_FOR_TYPE
#define KRATOS_BASE_DATA_COMMUNICATOR_DECLARE_PUBLIC_INTERFACE_FOR_TYPE(type)   \
KRATOS_BASE_DATA_COMMUNICATOR_DECLARE_REDUCE_INTERFACE_FOR_TYPE(type)    \
KRATOS_BASE_DATA_COMMUNICATOR_DECLARE_ALLREDUCE_INTERFACE_FOR_TYPE(type) \
KRATOS_BASE_DATA_COMMUNICATOR_DECLARE_SCANSUM_INTERFACE_FOR_TYPE(type)   \
KRATOS_BASE_DATA_COMMUNICATOR_DECLARE_SCATTER_INTERFACE_FOR_TYPE(type)   \
KRATOS_BASE_DATA_COMMUNICATOR_DECLARE_GATHER_INTERFACE_FOR_TYPE(type)    \

#endif

#ifndef KRATOS_BASE_DATA_COMMUNICATOR_DECLARE_IMPLEMENTATION_FOR_TYPE
#define KRATOS_BASE_DATA_COMMUNICATOR_DECLARE_IMPLEMENTATION_FOR_TYPE(type)   \
KRATOS_BASE_DATA_COMMUNICATOR_DECLARE_SENDRECV_INTERFACE_FOR_TYPE(type)  \
KRATOS_BASE_DATA_COMMUNICATOR_DECLARE_BROADCAST_INTERFACE_FOR_TYPE(type) \

#endif


namespace Kratos
{



class KRATOS_API(KRATOS_CORE) DataCommunicator
{
private:

template<typename T> class serialization_is_required {
private:

template<typename U> struct serialization_traits {
constexpr static bool is_std_vector = false;
constexpr static bool value_type_is_compound = false;
constexpr static bool value_type_is_bool = false;
};

template<typename U> struct serialization_traits<std::vector<U>> {
constexpr static bool is_std_vector = true;
constexpr static bool value_type_is_compound = std::is_compound<U>::value;
constexpr static bool value_type_is_bool = std::is_same<U, bool>::value;
};

constexpr static bool is_vector_of_simple_types = serialization_traits<T>::is_std_vector && !serialization_traits<T>::value_type_is_compound;
constexpr static bool is_vector_of_bools = serialization_traits<T>::is_std_vector && serialization_traits<T>::value_type_is_bool;

constexpr static bool is_vector_of_directly_communicable_type = is_vector_of_simple_types && !is_vector_of_bools;

public:
constexpr static bool value = std::is_compound<T>::value && !is_vector_of_directly_communicable_type;
};

template<bool value> struct TypeFromBool {};

template<typename T> void CheckSerializationForSimpleType(const T& rSerializedType, TypeFromBool<true>) const {}

template<typename T>
KRATOS_DEPRECATED_MESSAGE("Calling serialization-based communication for a simple type. Please implement direct communication support for this type.")
void CheckSerializationForSimpleType(const T& rSerializedType, TypeFromBool<false>) const {}

public:

KRATOS_CLASS_POINTER_DEFINITION(DataCommunicator);


DataCommunicator() {}

virtual ~DataCommunicator() {}



static DataCommunicator::UniquePointer Create()
{
return Kratos::make_unique<DataCommunicator>();
}


virtual void Barrier() const {}


KRATOS_BASE_DATA_COMMUNICATOR_DECLARE_PUBLIC_INTERFACE_FOR_TYPE(int)
KRATOS_BASE_DATA_COMMUNICATOR_DECLARE_PUBLIC_INTERFACE_FOR_TYPE(unsigned int)
KRATOS_BASE_DATA_COMMUNICATOR_DECLARE_PUBLIC_INTERFACE_FOR_TYPE(long unsigned int)
KRATOS_BASE_DATA_COMMUNICATOR_DECLARE_PUBLIC_INTERFACE_FOR_TYPE(double)
KRATOS_BASE_DATA_COMMUNICATOR_DECLARE_SCATTER_INTERFACE_FOR_TYPE(char)



virtual array_1d<double,3> Sum(const array_1d<double,3>& rLocalValue, const int Root) const
{
return rLocalValue;
}



virtual array_1d<double,3> Min(const array_1d<double,3>& rLocalValue, const int Root) const
{
return rLocalValue;
}


virtual array_1d<double,3> Max(const array_1d<double,3>& rLocalValue, const int Root) const
{
return rLocalValue;
}

virtual bool AndReduce(
const bool Value,
const int Root) const
{
return Value;
}

virtual Kratos::Flags AndReduce(
const Kratos::Flags Values,
const Kratos::Flags Mask,
const int Root) const
{
return Values;
}

virtual bool OrReduce(
const bool Value,
const int Root) const
{
return Value;
}

virtual Kratos::Flags OrReduce(
const Kratos::Flags Values,
const Kratos::Flags Mask,
const int Root) const
{
return Values;
}



virtual array_1d<double,3> SumAll(const array_1d<double,3>& rLocalValue) const
{
return rLocalValue;
}


virtual array_1d<double,3> MinAll(const array_1d<double,3>& rLocalValue) const
{
return rLocalValue;
}


virtual array_1d<double,3> MaxAll(const array_1d<double,3>& rLocalValue) const
{
return rLocalValue;
}

virtual bool AndReduceAll(const bool Value) const
{
return Value;
}

virtual Kratos::Flags AndReduceAll(const Kratos::Flags Values, const Kratos::Flags Mask) const
{
return Values;
}

virtual bool OrReduceAll(const bool Value) const
{
return Value;
}

virtual Kratos::Flags OrReduceAll(const Kratos::Flags Values, const Kratos::Flags Mask) const
{
return Values;
}



template<typename TObject>
void Broadcast(TObject& rBroadcastObject, const int SourceRank) const
{
this->BroadcastImpl(rBroadcastObject, SourceRank);
}



template<typename TObject>
TObject SendRecv(
const TObject& rSendObject, const int SendDestination, const int SendTag,
const int RecvSource, const int RecvTag) const
{
return this->SendRecvImpl(rSendObject, SendDestination, SendTag, RecvSource, RecvTag);
}


template<class TObject>
TObject SendRecv(
const TObject& rSendObject, const int SendDestination, const int RecvSource) const
{
return this->SendRecvImpl(rSendObject, SendDestination, 0, RecvSource, 0);
}


template<class TObject>
void SendRecv(
const TObject& rSendObject, const int SendDestination, const int SendTag,
TObject& rRecvObject, const int RecvSource, const int RecvTag) const
{
this->SendRecvImpl(rSendObject, SendDestination, SendTag, rRecvObject, RecvSource, RecvTag);
}


template<class TObject>
void SendRecv(
const TObject& rSendObject, const int SendDestination, TObject& rRecvObject, const int RecvSource) const
{
this->SendRecvImpl(rSendObject, SendDestination, 0, rRecvObject, RecvSource, 0);
}


template<typename TObject>
void Send(const TObject& rSendValues, const int SendDestination, const int SendTag = 0) const
{
this->SendImpl(rSendValues, SendDestination, SendTag);
}


template<typename TObject>
void Recv(TObject& rRecvObject, const int RecvSource, const int RecvTag = 0) const
{
this->RecvImpl(rRecvObject, RecvSource, RecvTag);
}



virtual int Rank() const
{
return 0;
}


virtual int Size() const
{
return 1;
}

virtual bool IsDistributed() const
{
return false;
}


virtual bool IsDefinedOnThisRank() const
{
return true;
}


virtual bool IsNullOnThisRank() const
{
return false;
}



KRATOS_DEPRECATED_MESSAGE("This function is deprecated, please retrieve the DataCommunicator through the ModelPart (or by name in special cases)")
static DataCommunicator& GetDefault();



virtual bool BroadcastErrorIfTrue(bool Condition, const int SourceRank) const
{
return Condition;
}


virtual bool BroadcastErrorIfFalse(bool Condition, const int SourceRank) const
{
return Condition;
}


virtual bool ErrorIfTrueOnAnyRank(bool Condition) const
{
return Condition;
}


virtual bool ErrorIfFalseOnAnyRank(bool Condition) const
{
return Condition;
}


virtual std::string Info() const
{
std::stringstream buffer;
PrintInfo(buffer);
return buffer.str();
}

virtual void PrintInfo(std::ostream &rOStream) const
{
rOStream << "DataCommunicator";
}

virtual void PrintData(std::ostream &rOStream) const
{
rOStream
<< "Serial do-nothing version of the Kratos wrapper for MPI communication.\n"
<< "Rank 0 of 1 assumed." << std::endl;
}


protected:


KRATOS_BASE_DATA_COMMUNICATOR_DECLARE_IMPLEMENTATION_FOR_TYPE(int)
KRATOS_BASE_DATA_COMMUNICATOR_DECLARE_IMPLEMENTATION_FOR_TYPE(unsigned int)
KRATOS_BASE_DATA_COMMUNICATOR_DECLARE_IMPLEMENTATION_FOR_TYPE(long unsigned int)
KRATOS_BASE_DATA_COMMUNICATOR_DECLARE_IMPLEMENTATION_FOR_TYPE(double)


virtual void BroadcastImpl(std::string& rBuffer, const int SourceRank) const {};


template<class TObject>
void BroadcastImpl(TObject& rBroadcastObject, const int SourceRank) const
{
CheckSerializationForSimpleType(rBroadcastObject, TypeFromBool<serialization_is_required<TObject>::value>());
if (this->IsDistributed())
{
unsigned int message_size;
std::string broadcast_message;
int rank = this->Rank();
if (rank == SourceRank)
{
MpiSerializer send_serializer;
send_serializer.save("data", rBroadcastObject);
broadcast_message = send_serializer.GetStringRepresentation();

message_size = broadcast_message.size();
}

this->Broadcast(message_size, SourceRank);

if (rank != SourceRank)
{
broadcast_message.resize(message_size);
}

this->Broadcast(broadcast_message, SourceRank);

if (rank != SourceRank)
{
MpiSerializer recv_serializer(broadcast_message);
recv_serializer.load("data", rBroadcastObject);
}
}
}


virtual void SendRecvImpl(
const std::string& rSendValues, const int SendDestination, const int SendTag,
std::string& rRecvValues, const int RecvSource, const int RecvTag) const
{
KRATOS_DATA_COMMUNICATOR_DEBUG_SIZE_CHECK(rSendValues.size(), rRecvValues.size(), "SendRecv");
rRecvValues = SendRecvImpl(rSendValues, SendDestination, SendTag, RecvSource, RecvTag);
}


virtual std::string SendRecvImpl(
const std::string& rSendValues, const int SendDestination, const int SendTag,
const int RecvSource, const int RecvTag) const
{
KRATOS_ERROR_IF( (Rank() != SendDestination) || (Rank() != RecvSource))
<< "Communication between different ranks is not possible with a serial DataCommunicator." << std::endl;
return rSendValues;
}


template<class TObject> TObject SendRecvImpl(
const TObject& rSendObject,
const int SendDestination, const int SendTag,
const int RecvSource, const int RecvTag) const
{
CheckSerializationForSimpleType(rSendObject, TypeFromBool<serialization_is_required<TObject>::value>());
if (this->IsDistributed())
{
MpiSerializer send_serializer;
send_serializer.save("data", rSendObject);
std::string send_message = send_serializer.GetStringRepresentation();

std::string recv_message = this->SendRecv(send_message, SendDestination, RecvSource);

MpiSerializer recv_serializer(recv_message);
TObject recv_object;
recv_serializer.load("data", recv_object);
return recv_object;
}
else
{
KRATOS_ERROR_IF( (Rank() != SendDestination) || (Rank() != RecvSource))
<< "Communication between different ranks is not possible with a serial DataCommunicator." << std::endl;

return rSendObject;
}
}


virtual void SendImpl(const std::string& rSendValues, const int SendDestination, const int SendTag) const
{
KRATOS_ERROR_IF(Rank() != SendDestination)
<< "Communication between different ranks is not possible with a serial DataCommunicator." << std::endl;
}


template<class TObject> void SendImpl(
const TObject& rSendObject, const int SendDestination, const int SendTag) const
{
CheckSerializationForSimpleType(rSendObject, TypeFromBool<serialization_is_required<TObject>::value>());
if (this->IsDistributed())
{
MpiSerializer send_serializer;
send_serializer.save("data", rSendObject);
std::string send_message = send_serializer.GetStringRepresentation();

this->SendImpl(send_message, SendDestination, SendTag);
}
else
{
KRATOS_ERROR_IF(Rank() != SendDestination)
<< "Communication between different ranks is not possible with a serial DataCommunicator." << std::endl;
}
}


virtual void RecvImpl(std::string& rRecvValues, const int RecvSource, const int RecvTag = 0) const
{
KRATOS_ERROR << "Calling serial DataCommunicator::Recv, which has no meaningful return." << std::endl;
}


template<class TObject> void RecvImpl(
TObject& rRecvObject, const int RecvSource, const int RecvTag = 0) const
{
CheckSerializationForSimpleType(rRecvObject, TypeFromBool<serialization_is_required<TObject>::value>());
if (this->IsDistributed())
{
std::string recv_message;

this->Recv(recv_message, RecvSource, RecvTag);

MpiSerializer recv_serializer(recv_message);
recv_serializer.load("data", rRecvObject);
}
else
{
KRATOS_ERROR_IF(Rank() != RecvSource)
<< "Communication between different ranks is not possible with a serial DataCommunicator." << std::endl;
}
}


private:


DataCommunicator(DataCommunicator const &rOther) = delete;

DataCommunicator &operator=(DataCommunicator const &rOther) = delete;


}; 




inline std::istream &operator>>(std::istream &rIStream,
DataCommunicator &rThis)
{
return rIStream;
}

inline std::ostream &operator<<(std::ostream &rOStream,
const DataCommunicator &rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}


} 

#undef KRATOS_DATA_COMMUNICATOR_DEBUG_SIZE_CHECK

#undef KRATOS_BASE_DATA_COMMUNICATOR_DECLARE_REDUCE_INTERFACE_FOR_TYPE
#undef KRATOS_BASE_DATA_COMMUNICATOR_DECLARE_ALLREDUCE_INTERFACE_FOR_TYPE
#undef KRATOS_BASE_DATA_COMMUNICATOR_DECLARE_SCANSUM_INTERFACE_FOR_TYPE
#undef KRATOS_BASE_DATA_COMMUNICATOR_DECLARE_SENDRECV_INTERFACE_FOR_TYPE
#undef KRATOS_BASE_DATA_COMMUNICATOR_DECLARE_BROADCAST_INTERFACE_FOR_TYPE
#undef KRATOS_BASE_DATA_COMMUNICATOR_DECLARE_SCATTER_INTERFACE_FOR_TYPE
#undef KRATOS_BASE_DATA_COMMUNICATOR_DECLARE_GATHER_INTERFACE_FOR_TYPE
#undef KRATOS_BASE_DATA_COMMUNICATOR_DECLARE_PUBLIC_INTERFACE_FOR_TYPE
#undef KRATOS_BASE_DATA_COMMUNICATOR_DECLARE_IMPLEMENTATION_FOR_TYPE
