
#pragma once

#include <string>
#include <iostream>
#include <type_traits>


#include "includes/define.h"
#include "includes/data_communicator.h"
#include "includes/global_pointer.h"
#include "includes/mpi_serializer.h"
#include "containers/global_pointers_vector.h"
#include "containers/global_pointers_unordered_map.h"
#include "includes/parallel_environment.h"
#include "utilities/communication_coloring_utilities.h"

namespace Kratos
{





template< class TPointerDataType >
class GlobalPointerCommunicator; 


template< class TPointerDataType, class TFunctorType >
class ResultsProxy
{
public:
using TSendType = typename std::result_of< TFunctorType(GlobalPointer<TPointerDataType>&)>::type;


ResultsProxy(
int current_rank,
GlobalPointersUnorderedMap< TPointerDataType, TSendType > NonLocalData,
TFunctorType UserFunctor,
GlobalPointerCommunicator<TPointerDataType>* pPointerComm
):
mCurrentRank(current_rank), mNonLocalData(NonLocalData), mUserFunctor(UserFunctor), mpPointerComm(pPointerComm)
{}

virtual ~ResultsProxy() {}


TSendType Get(GlobalPointer<TPointerDataType>& rGlobalPointer) const
{
if(rGlobalPointer.GetRank() == mCurrentRank)
return mUserFunctor(rGlobalPointer);
else {
auto non_local_gp = mNonLocalData.find(rGlobalPointer);
KRATOS_DEBUG_ERROR_IF(non_local_gp == mNonLocalData.end()) << "Missing entry in NonLocalData" << std::endl;
return non_local_gp->second;
}
}


TSendType Get(const GlobalPointer<TPointerDataType>& rGlobalPointer) const
{
if(rGlobalPointer.GetRank() == mCurrentRank)
return mUserFunctor(rGlobalPointer);
else {
auto non_local_gp = mNonLocalData.find(rGlobalPointer);
KRATOS_DEBUG_ERROR_IF(non_local_gp == mNonLocalData.end()) << "Missing entry in NonLocalData" << std::endl;
return non_local_gp->second;
}
}


bool Has(GlobalPointer<TPointerDataType>& rGlobalPointer) const 
{
if(rGlobalPointer.GetRank() == mCurrentRank)
return true;
else
return mNonLocalData.find(rGlobalPointer) != mNonLocalData.end();
}


bool Has(const GlobalPointer<TPointerDataType>& rGlobalPointer) const
{
if(rGlobalPointer.GetRank() == mCurrentRank)
return true;
else
return mNonLocalData.find(rGlobalPointer) != mNonLocalData.end();
}


void Update()
{
mpPointerComm->Update(mUserFunctor, mNonLocalData);
}

private:
const int mCurrentRank;  
GlobalPointersUnorderedMap< TPointerDataType, TSendType > mNonLocalData;  
TFunctorType mUserFunctor;  
GlobalPointerCommunicator<TPointerDataType>* mpPointerComm;  

};



template< class TPointerDataType >
class GlobalPointerCommunicator
{
public:

KRATOS_CLASS_POINTER_DEFINITION(GlobalPointerCommunicator);



GlobalPointerCommunicator(const DataCommunicator& rComm, GlobalPointersVector< TPointerDataType >& rGpList ):
mrDataCommunicator(rComm)
{
AddPointers(rGpList.ptr_begin(),rGpList.ptr_end());

if(mrDataCommunicator.IsDistributed())
{
ComputeCommunicationPlan();
}
}


template< class TIteratorType >
GlobalPointerCommunicator(const DataCommunicator& rComm, TIteratorType itBegin, TIteratorType itEnd):
mrDataCommunicator(rComm)
{
AddPointers(itBegin,itEnd);

if(mrDataCommunicator.IsDistributed())
{
ComputeCommunicationPlan();
}
}


template< class TFunctorType >
GlobalPointerCommunicator(const DataCommunicator& rComm, TFunctorType rFunctor):
mrDataCommunicator(rComm)
{
if(rComm.IsDistributed())
{
auto gps = rFunctor(rComm);
AddPointers(gps.ptr_begin(), gps.ptr_end());
ComputeCommunicationPlan();
}
}


template< class TFunctorType >
GlobalPointerCommunicator(TFunctorType rFunctor)
: GlobalPointerCommunicator(ParallelEnvironment::GetDefaultDataCommunicator(), rFunctor)
{}

virtual ~GlobalPointerCommunicator() {}


template< class TFunctorType >
ResultsProxy<
TPointerDataType,
TFunctorType 
> Apply(TFunctorType&& UserFunctor)
{
typedef typename ResultsProxy<TPointerDataType, TFunctorType >::TSendType SendType;

const int current_rank = mrDataCommunicator.Rank();

GlobalPointersUnorderedMap< TPointerDataType, SendType > non_local_data;

if(mrDataCommunicator.IsDistributed())
{
Update(UserFunctor, non_local_data);
}

return ResultsProxy<TPointerDataType, TFunctorType>(current_rank,non_local_data,UserFunctor, this );
}


template< class TFunctorType >
void Update(
TFunctorType& rUserFunctor,
GlobalPointersUnorderedMap< TPointerDataType, typename ResultsProxy<TPointerDataType, TFunctorType >::TSendType >& rNonLocalData)
{
for(auto color : mColors)
{
if(color >= 0) 
{
auto& gps_to_be_sent = mNonLocalPointers[color];

auto recv_global_pointers = mrDataCommunicator.SendRecv(gps_to_be_sent, color, color );

std::vector< typename ResultsProxy<TPointerDataType, TFunctorType >::TSendType > locally_gathered_data; 
for(auto& gp : recv_global_pointers.GetContainer())
locally_gathered_data.push_back( rUserFunctor(gp) );

auto remote_data = mrDataCommunicator.SendRecv(locally_gathered_data, color, color );

for(unsigned int i=0; i<remote_data.size(); ++i)
rNonLocalData[gps_to_be_sent(i)] = remote_data[i];
}
}
}






virtual std::string Info() const
{
std::stringstream buffer;
buffer << "GlobalPointerCommunicator" ;
return buffer.str();
}

virtual void PrintInfo(std::ostream& rOStream) const
{
rOStream << "GlobalPointerCommunicator";
}

virtual void PrintData(std::ostream& rOStream) const {}


protected:

std::unordered_map<int, GlobalPointersVector< TPointerDataType > > mNonLocalPointers; 

const DataCommunicator& mrDataCommunicator; 




template< class TIteratorType >
void AddPointers( TIteratorType begin, TIteratorType end)
{
if(mrDataCommunicator.IsDistributed())
{
const int current_rank = mrDataCommunicator.Rank();
for(auto it = begin; it != end; ++it)
{
auto& gp = *it;
if(gp.GetRank() != current_rank)
{
mNonLocalPointers[gp.GetRank()].push_back(gp);
}
}

for(auto& non_local : mNonLocalPointers)
non_local.second.Unique();
}
}


void ComputeCommunicationPlan()
{
std::vector<int> send_list;
send_list.reserve( mNonLocalPointers.size() );
for(auto& it : mNonLocalPointers)
send_list.push_back( it.first );

std::sort(send_list.begin(), send_list.end());
mColors = MPIColoringUtilities::ComputeCommunicationScheduling(send_list, mrDataCommunicator);
}





private:


std::vector<int> mColors; 






GlobalPointerCommunicator& operator=(GlobalPointerCommunicator const& rOther) {}

GlobalPointerCommunicator(GlobalPointerCommunicator const& rOther) {}


}; 




template< class TPointerDataType, class TSendType >
inline std::istream& operator >> (std::istream& rIStream,
GlobalPointerCommunicator<TPointerDataType>& rThis)
{
return rIStream;
}

template< class TPointerDataType, class TSendType >
inline std::ostream& operator << (std::ostream& rOStream,
const GlobalPointerCommunicator<TPointerDataType>& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}


}  


