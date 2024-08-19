
#pragma once

#include <string>
#include <iostream>
#include <sstream>
#include <cstddef>
#include <atomic>


#include "includes/define.h"
#include "includes/lock_object.h"
#include "geometries/point.h"
#include "includes/dof.h"
#include "containers/pointer_vector_set.h"
#include "containers/variables_list_data_value_container.h"
#include "containers/flags.h"
#include "intrusive_ptr/intrusive_ptr.hpp"
#include "containers/global_pointers_vector.h"
#include "containers/data_value_container.h"
#include "containers/nodal_data.h"
#include "includes/kratos_flags.h"

namespace Kratos
{

class Element;







class Node : public Point, public Flags
{
public:

KRATOS_CLASS_INTRUSIVE_POINTER_DEFINITION(Node);

using NodeType = Node;

using BaseType = Point;

using PointType = Point;

using DofType = Dof<double>;

using IndexType = std::size_t;

using SizeType = std::size_t;

using DofsContainerType = std::vector<std::unique_ptr<Dof<double>>>;

using SolutionStepsNodalDataContainerType = VariablesListDataValueContainer;

using BlockType = VariablesListDataValueContainer::BlockType;


Node()
: BaseType()
, Flags()
, mNodalData(0)
, mDofs()
, mData()
, mInitialPosition()
, mNodeLock()
{
CreateSolutionStepData();
}

explicit Node(IndexType NewId )
: BaseType()
, Flags()
, mNodalData(NewId)
, mDofs()
, mData()
, mInitialPosition()
, mNodeLock()
{
KRATOS_ERROR <<  "Calling the default constructor for the node ... illegal operation!!" << std::endl;
CreateSolutionStepData();
}

Node(IndexType NewId, double const& NewX)
: BaseType(NewX)
, Flags()
, mNodalData(NewId)
, mDofs()
, mData()
, mInitialPosition(NewX)
, mNodeLock()
{
CreateSolutionStepData();
}

Node(IndexType NewId, double const& NewX, double const& NewY)
: BaseType(NewX, NewY)
, Flags()
, mNodalData(NewId)
, mDofs()
, mData()
, mInitialPosition(NewX, NewY)
, mNodeLock()
{
CreateSolutionStepData();
}

Node(IndexType NewId, double const& NewX, double const& NewY, double const& NewZ)
: BaseType(NewX, NewY, NewZ)
, Flags()
, mNodalData(NewId)
, mDofs()
, mData()
, mInitialPosition(NewX, NewY, NewZ)
, mNodeLock()
{
CreateSolutionStepData();
}

Node(IndexType NewId, PointType const& rThisPoint)
: BaseType(rThisPoint)
, Flags()
, mNodalData(NewId)
, mDofs()
, mData()
, mInitialPosition(rThisPoint)
, mNodeLock()
{
CreateSolutionStepData();
}


Node(Node const& rOtherNode) = delete;


template<class TVectorType>
Node(IndexType NewId, vector_expression<TVectorType> const&  rOtherCoordinates)
: BaseType(rOtherCoordinates)
, Flags()
, mNodalData(NewId)
, mDofs()
, mData()
, mInitialPosition(rOtherCoordinates)
, mNodeLock()
{
CreateSolutionStepData();
}




Node(IndexType NewId, std::vector<double> const&  rOtherCoordinates)
: BaseType(rOtherCoordinates)
, Flags()
, mNodalData(NewId)
, mDofs()
, mData()
, mInitialPosition()
, mNodeLock()
{
CreateSolutionStepData();
}

Node(IndexType NewId, double const& NewX, double const& NewY, double const& NewZ, VariablesList::Pointer  pVariablesList, BlockType const * ThisData, SizeType NewQueueSize = 1)
: BaseType(NewX, NewY, NewZ)
, Flags()
, mNodalData(NewId, pVariablesList,ThisData,NewQueueSize)
, mDofs()
, mData()
, mInitialPosition(NewX, NewY, NewZ)
, mNodeLock()
{
}

typename Node::Pointer Clone()
{
Node::Pointer p_new_node = Kratos::make_intrusive<Node >( this->Id(), (*this)[0], (*this)[1], (*this)[2]);
p_new_node->mNodalData = this->mNodalData;

Node::DofsContainerType& my_dofs = (this)->GetDofs();
for (typename DofsContainerType::const_iterator it_dof = my_dofs.begin(); it_dof != my_dofs.end(); it_dof++)
{
p_new_node->pAddDof(**it_dof);
}

p_new_node->mData = this->mData;
p_new_node->mInitialPosition = this->mInitialPosition;

p_new_node->Set(Flags(*this));

return p_new_node;
}

~Node() override
{
ClearSolutionStepsData();
}

/
template<class TVariableType>
inline unsigned int GetDofPosition(TVariableType const& rDofVariable) const
{
typename DofsContainerType::const_iterator it_dof = mDofs.end();
for(it_dof = mDofs.begin() ; it_dof != mDofs.end() ; it_dof++){
if((*it_dof)->GetVariable() == rDofVariable){
break;
}
}

return it_dof - mDofs.begin();
}


template<class TVariableType>
inline const DofType& GetDof(TVariableType const& rDofVariable, int pos) const
{
typename DofsContainerType::const_iterator it_begin = mDofs.begin();
typename DofsContainerType::const_iterator it_end = mDofs.end();
typename DofsContainerType::const_iterator it;
if(pos < it_end-it_begin)
{
it = it_begin + pos;
if( (*it)->GetVariable() == rDofVariable)
{
return **it;
}
}

for(auto it_dof = mDofs.begin() ; it_dof != mDofs.end() ; it_dof++){
if((*it_dof)->GetVariable() == rDofVariable){
return **it_dof;
}
}

KRATOS_ERROR <<  "Non-existent DOF in node #" << Id() << " for variable : " << rDofVariable.Name() << std::endl;
}


template<class TVariableType>
inline const DofType& GetDof(TVariableType const& rDofVariable) const
{
for(auto it_dof = mDofs.begin() ; it_dof != mDofs.end() ; it_dof++){
if((*it_dof)->GetVariable() == rDofVariable){
return **it_dof;
}
}

KRATOS_ERROR <<  "Non-existent DOF in node #" << Id() << " for variable : " << rDofVariable.Name() << std::endl;

}


DofsContainerType& GetDofs()
{
return mDofs;
}

const DofsContainerType& GetDofs() const
{
return mDofs;
}


template<class TVariableType>
inline typename DofType::Pointer pGetDof(TVariableType const& rDofVariable) const
{
for(auto it_dof = mDofs.begin() ; it_dof != mDofs.end() ; it_dof++){
if((*it_dof)->GetVariable() == rDofVariable){
return (*it_dof).get();
}
}

KRATOS_ERROR <<  "Non-existent DOF in node #" << Id() << " for variable : " << rDofVariable.Name() << std::endl;

}


template<class TVariableType>
inline typename DofType::Pointer pGetDof(
TVariableType const& rDofVariable,
int Position
) const
{
const auto it_begin = mDofs.begin();
const auto it_end = mDofs.end();
if(Position < it_end-it_begin) {
auto it_dof = it_begin + Position;
if( (*it_dof)->GetVariable() == rDofVariable) {
return (*it_dof).get();
}
}

for(auto it_dof = it_begin; it_dof != it_end; ++it_dof){
if((*it_dof)->GetVariable() == rDofVariable){
return (*it_dof).get();
}
}

KRATOS_ERROR <<  "Non-existent DOF in node #" << Id() << " for variable : " << rDofVariable.Name() << std::endl;
}



template<class TVariableType>
inline typename DofType::Pointer pAddDof(TVariableType const& rDofVariable)
{
KRATOS_TRY

for(auto it_dof = mDofs.begin() ; it_dof != mDofs.end() ; it_dof++){
if((*it_dof)->GetVariable() == rDofVariable){
return (*it_dof).get();
}
}

mDofs.push_back(Kratos::make_unique<DofType>(&mNodalData, rDofVariable));

DofType* p_new_dof = mDofs.back().get();

SortDofs();

return p_new_dof;

KRATOS_CATCH(*this);
}


inline typename DofType::Pointer pAddDof(DofType const& SourceDof)
{
KRATOS_TRY

for(auto it_dof = mDofs.begin() ; it_dof != mDofs.end() ; it_dof++){
if((*it_dof)->GetVariable() == SourceDof.GetVariable()){
if((*it_dof)->GetReaction() != SourceDof.GetReaction())
{
**it_dof = SourceDof;
(*it_dof)->SetNodalData(&mNodalData);
}
return (*it_dof).get();
}
}

mDofs.push_back(Kratos::make_unique<DofType>(SourceDof));
mDofs.back()->SetNodalData(&mNodalData);

DofType* p_new_dof = mDofs.back().get();

SortDofs();

return p_new_dof;

KRATOS_CATCH(*this);
}


template<class TVariableType, class TReactionType>
inline typename DofType::Pointer pAddDof(TVariableType const& rDofVariable, TReactionType const& rDofReaction)
{
KRATOS_TRY

for(auto it_dof = mDofs.begin() ; it_dof != mDofs.end() ; it_dof++){
if((*it_dof)->GetVariable() == rDofVariable){
(*it_dof)->SetReaction(rDofReaction);
return (*it_dof).get();
}
}

mDofs.push_back(Kratos::make_unique<DofType>(&mNodalData, rDofVariable, rDofReaction));

DofType* p_new_dof = mDofs.back().get();

SortDofs();

return p_new_dof;

KRATOS_CATCH(*this);

}


template<class TVariableType>
inline DofType& AddDof(TVariableType const& rDofVariable)
{
KRATOS_TRY

for(auto it_dof = mDofs.begin() ; it_dof != mDofs.end() ; it_dof++){
if((*it_dof)->GetVariable() == rDofVariable){
return **it_dof;
}
}

mDofs.push_back(Kratos::make_unique<DofType>(&mNodalData, rDofVariable));

DofType* p_new_dof = mDofs.back().get();

SortDofs();

return *p_new_dof;

KRATOS_CATCH(*this);

}


template<class TVariableType, class TReactionType>
inline DofType& AddDof(TVariableType const& rDofVariable, TReactionType const& rDofReaction)
{
KRATOS_TRY

for(auto it_dof = mDofs.begin() ; it_dof != mDofs.end() ; it_dof++){
if((*it_dof)->GetVariable() == rDofVariable){
(*it_dof)->SetReaction(rDofReaction);
return **it_dof;
}
}

mDofs.push_back(Kratos::make_unique<DofType>(&mNodalData, rDofVariable, rDofReaction));

DofType* p_new_dof = mDofs.back().get();

SortDofs();

return *p_new_dof;

KRATOS_CATCH(*this);

}



inline bool HasDofFor(const VariableData& rDofVariable) const
{
for(auto it_dof = mDofs.begin() ; it_dof != mDofs.end() ; it_dof++){
if((*it_dof)->GetVariable() == rDofVariable){
return true;
}
}
return false;
}

inline  bool IsFixed(const VariableData& rDofVariable) const
{
for(auto it_dof = mDofs.begin() ; it_dof != mDofs.end() ; it_dof++){
if((*it_dof)->GetVariable() == rDofVariable){
return (*it_dof)->IsFixed();
}
}
return false;
}


inline bool IsActive() const 
{
return IsDefined(ACTIVE) ? Is(ACTIVE) : true;
}


std::string Info() const override
{
std::stringstream buffer;
buffer << "Node #" << Id();
return buffer.str();
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << Info();
}

void PrintData(std::ostream& rOStream) const override
{
BaseType::PrintData(rOStream);
if(!mDofs.empty())
rOStream << std::endl << "    Dofs :" << std::endl;
for(typename DofsContainerType::const_iterator i = mDofs.begin() ; i != mDofs.end() ; i++)
rOStream << "        " << (*i)->Info() << std::endl;
}


protected:







private:


NodalData mNodalData;


DofsContainerType  mDofs;


DataValueContainer mData;


PointType mInitialPosition;

LockObject mNodeLock;

mutable std::atomic<int> mReferenceCounter{0};

friend void intrusive_ptr_add_ref(const NodeType* x)
{
x->mReferenceCounter.fetch_add(1, std::memory_order_relaxed);
}

friend void intrusive_ptr_release(const NodeType* x)
{
if (x->mReferenceCounter.fetch_sub(1, std::memory_order_release) == 1) {
std::atomic_thread_fence(std::memory_order_acquire);
delete x;
}
}


void SortDofs(){
std::sort(mDofs.begin(), mDofs.end(), [](Kratos::unique_ptr<DofType> const& First, Kratos::unique_ptr<DofType> const& Second)->bool{
return First->GetVariable().Key() < Second->GetVariable().Key();
});
}


friend class Serializer;

void save(Serializer& rSerializer) const override
{
KRATOS_SERIALIZE_SAVE_BASE_CLASS(rSerializer, Point );
KRATOS_SERIALIZE_SAVE_BASE_CLASS(rSerializer, Flags );
rSerializer.save("NodalData", &mNodalData); 
rSerializer.save("Data", mData);
rSerializer.save("Initial Position", mInitialPosition);
rSerializer.save("Data", mDofs);

}

void load(Serializer& rSerializer) override
{
KRATOS_SERIALIZE_LOAD_BASE_CLASS(rSerializer, Point );
KRATOS_SERIALIZE_LOAD_BASE_CLASS(rSerializer, Flags );
NodalData* p_nodal_data = &mNodalData;
rSerializer.load("NodalData", p_nodal_data);
rSerializer.load("Data", mData);
rSerializer.load("Initial Position", mInitialPosition);
rSerializer.load("Data", mDofs);
}





}; 







inline std::istream& operator >> (std::istream& rIStream,
Node& rThis);

inline std::ostream& operator << (std::ostream& rOStream,
const Node& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << " : ";
rThis.PrintData(rOStream);

return rOStream;
}


}  
