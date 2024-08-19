
#pragma once

#include <string>
#include <iostream>


#include "includes/data_communicator.h"
#include "includes/global_pointer.h"
#include "containers/global_pointers_vector.h"
#include "containers/global_pointers_unordered_map.h"
#include "utilities/communication_coloring_utilities.h"

namespace Kratos
{







class GlobalPointerUtilities
{
public:

KRATOS_CLASS_POINTER_DEFINITION(GlobalPointerUtilities);


GlobalPointerUtilities()
{}

virtual ~GlobalPointerUtilities() {}


template< class TContainerType >
static std::unordered_map< int, GlobalPointer<typename TContainerType::value_type> > RetrieveGlobalIndexedPointersMap(
const TContainerType& rContainer,
const std::vector<int>& rIdList,
const DataCommunicator& rDataCommunicator
)
{
using GPType = GlobalPointer<typename TContainerType::value_type>;

std::unordered_map< int, GPType > global_pointers_list;
const int current_rank = rDataCommunicator.Rank();
const int world_size = rDataCommunicator.Size();

std::vector<int> remote_ids;

if(rDataCommunicator.IsDistributed()) {
for(const int id : rIdList ) {
const auto it = rContainer.find(id);

if( it != rContainer.end()) {
if(ObjectIsLocal(*it, current_rank)) {
global_pointers_list.emplace(id,GPType(&*it, current_rank));
} else {
remote_ids.push_back(id);
}
} else {
remote_ids.push_back(id);
}
}
} else {
for(const int id : rIdList ) {
const auto it = rContainer.find(id);
if( it != rContainer.end()) {
global_pointers_list.emplace(id,GPType(&*it, current_rank));
}
}
}

int master_rank = 0;

std::vector<int> all_remote_ids;
std::vector< std::vector<int> > collected_remote_ids(world_size);
std::unordered_map< int, GPType > all_non_local_gp_map;

for(int i=0; i<world_size; ++i) {
if(i != master_rank) {
if(current_rank == master_rank) { 
rDataCommunicator.Recv(collected_remote_ids[i],i);
} else if(current_rank == i) { 
rDataCommunicator.Send(remote_ids,master_rank);
}
} else { 
if(current_rank == master_rank) 
collected_remote_ids[i] = remote_ids;
}

if(current_rank == master_rank) {
for(const int id : collected_remote_ids[i])
all_remote_ids.push_back( id );
}
}



if(current_rank == master_rank) {
std::sort(all_remote_ids.begin(), all_remote_ids.end());
auto last = std::unique(all_remote_ids.begin(), all_remote_ids.end());
all_remote_ids.erase(last, all_remote_ids.end());
}

int number_of_all_remote_ids = all_remote_ids.size();
rDataCommunicator.Broadcast(number_of_all_remote_ids,master_rank);

if(current_rank != master_rank)
all_remote_ids.resize(number_of_all_remote_ids);

rDataCommunicator.Broadcast(all_remote_ids,master_rank);

for(int i=0; i<world_size; ++i) {
if(i != master_rank) {
if(current_rank == master_rank) {
std::unordered_map< int, GPType > recv_gps;
rDataCommunicator.Recv(recv_gps, i);

for(auto& it : recv_gps)
all_non_local_gp_map.emplace(it.first, it.second);
} else if(current_rank == i) {
auto non_local_gp_map = ComputeGpMap(rContainer, all_remote_ids, rDataCommunicator);
rDataCommunicator.Send(non_local_gp_map,master_rank);
}
} else {
auto recv_gps = ComputeGpMap(rContainer, all_remote_ids, rDataCommunicator);

for(auto& it : recv_gps)
all_non_local_gp_map.emplace(it.first, it.second);
}
}

for(int i=0; i<world_size; ++i) {
if(i != master_rank) {
if(current_rank == master_rank) { 
auto gp_list = ExtractById(all_non_local_gp_map,collected_remote_ids[i]);

rDataCommunicator.Send(gp_list,i);
} else if(current_rank == i) { 
std::unordered_map< int, GPType > gp_list;
rDataCommunicator.Recv(gp_list, master_rank);

for(auto& it : gp_list)
global_pointers_list.emplace(it.first, it.second);
}
} else {
auto gp_list = ExtractById(all_non_local_gp_map,collected_remote_ids[i]);

for(auto& it : gp_list)
global_pointers_list.emplace(it.first, it.second);
}
}

return global_pointers_list;
}


template< class TContainerType >
static GlobalPointersVector< typename TContainerType::value_type > LocalRetrieveGlobalPointers(
const TContainerType& rContainer,
const DataCommunicator& rDataCommunicator
)
{
std::vector<int> local_id_list;
local_id_list.reserve(rContainer.size());
for (const auto& r_entity : rContainer) {
local_id_list.push_back(r_entity.Id());
}

return RetrieveGlobalIndexedPointers(rContainer, local_id_list, rDataCommunicator);
}


template< class TContainerType >
static GlobalPointersVector< typename TContainerType::value_type > GlobalRetrieveGlobalPointers(
const TContainerType& rContainer,
const DataCommunicator& rDataCommunicator
)
{
const int world_size = rDataCommunicator.Size();

const int number_of_entities = rContainer.size();

std::vector<int> number_of_entities_per_partition(world_size);
std::vector<int> send_number_of_entities_per_partition(1, number_of_entities);
rDataCommunicator.AllGather(send_number_of_entities_per_partition, number_of_entities_per_partition);

std::vector<int> global_id_list, local_id_list;
local_id_list.reserve(rContainer.size());
for (const auto& r_entity : rContainer) {
local_id_list.push_back(r_entity.Id());
}

std::vector<int> recv_sizes(number_of_entities_per_partition);
int message_size = 0;
std::vector<int> recv_offsets(world_size, 0);
for (int i_rank = 0; i_rank < world_size; i_rank++) {
recv_offsets[i_rank] = message_size;
message_size += recv_sizes[i_rank];
}
global_id_list.resize(message_size);

rDataCommunicator.AllGatherv(local_id_list, global_id_list, recv_sizes, recv_offsets);

return RetrieveGlobalIndexedPointers(rContainer, global_id_list, rDataCommunicator);
}


template< class TContainerType >
static GlobalPointersVector< typename TContainerType::value_type > RetrieveGlobalIndexedPointers(
const TContainerType& rContainer,
const std::vector<int>& rIdList,
const DataCommunicator& rDataCommunicator
)
{
auto global_pointers_list = RetrieveGlobalIndexedPointersMap(rContainer, rIdList, rDataCommunicator);

const int current_rank = rDataCommunicator.Rank();

GlobalPointersVector< typename TContainerType::value_type > result;
result.reserve(rIdList.size());
for(unsigned int i=0; i<rIdList.size(); ++i) {
auto it = global_pointers_list.find(rIdList[i]);
if(it != global_pointers_list.end())
result.push_back( it->second );
else
KRATOS_ERROR << "The id " << rIdList[i] << " was not found for processor " << current_rank << std::endl;
}

return result;

}






virtual std::string Info() const
{
std::stringstream buffer;
buffer << "GlobalPointerUtilities" ;
return buffer.str();
}

virtual void PrintInfo(std::ostream& rOStream) const
{
rOStream << "GlobalPointerUtilities";
}

virtual void PrintData(std::ostream& rOStream) const {}


protected:







private:



static bool ObjectIsLocal(const Element& rElem, const int CurrentRank)
{
return true; 
}


static bool ObjectIsLocal(const Condition& rCond, const int CurrentRank)
{
return true; 
}


static bool ObjectIsLocal(const Node& rNode, const int CurrentRank)
{
return rNode.FastGetSolutionStepValue(PARTITION_INDEX) == CurrentRank;
}



template< class GPType >
static std::unordered_map< int, GPType > ExtractById(
std::unordered_map< int, GPType >& rGPList,
const std::vector<int>& rIds)
{
std::unordered_map< int, GPType > extracted_list;
for(auto id : rIds){
auto gp = rGPList[id];
extracted_list[id] = gp;
}
return extracted_list;
}


template< class TContainerType >
static std::unordered_map< int, GlobalPointer<typename TContainerType::value_type> > ComputeGpMap(
const TContainerType& rContainer,
const std::vector<int>& rIds,
const DataCommunicator& rDataCommunicator)
{
const int current_rank = rDataCommunicator.Rank();
std::unordered_map< int, GlobalPointer<typename TContainerType::value_type> > extracted_list;

if(rDataCommunicator.IsDistributed()) {
for(auto id : rIds) {
const auto it = rContainer.find(id);

if( it != rContainer.end()) {
if(ObjectIsLocal(*it, current_rank)){
extracted_list.emplace(id, GlobalPointer<typename TContainerType::value_type>(&*it, current_rank));
}
}
}
} else {
for(auto id : rIds) {
const auto it = rContainer.find(id);

if( it != rContainer.end()) {
extracted_list.emplace(id, GlobalPointer<typename TContainerType::value_type>(&*it, current_rank));
}
}
}
return extracted_list;
}





GlobalPointerUtilities& operator=(GlobalPointerUtilities const& rOther) = delete;

GlobalPointerUtilities(GlobalPointerUtilities const& rOther) = delete;


}; 




inline std::istream& operator >> (std::istream& rIStream,
GlobalPointerUtilities& rThis)
{
return rIStream;
}

inline std::ostream& operator << (std::ostream& rOStream,
const GlobalPointerUtilities& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}


}  