
#pragma once

#include <string>
#include <iostream>
#include <sstream>
#include <cstddef>


#include "includes/define.h"
#include "includes/serializer.h"
#include "includes/process_info.h"
#include "containers/data_value_container.h"
#include "includes/mesh.h"
#include "containers/geometry_container.h"
#include "includes/element.h"
#include "includes/condition.h"
#include "includes/communicator.h"
#include "includes/table.h"
#include "containers/pointer_vector_map.h"
#include "containers/pointer_hash_map_set.h"
#include "input_output/logger.h"
#include "includes/kratos_flags.h"
#include "includes/master_slave_constraint.h"
#include "containers/variable.h"
#include "containers/variable_data.h"

namespace Kratos
{






class Model;


class KRATOS_API(KRATOS_CORE) ModelPart final
: public DataValueContainer, public Flags
{
class GetModelPartName
{
public:
std::string const& operator()(const ModelPart& rModelPart) const
{
return rModelPart.Name();
}
};
public:

enum OwnershipType
{
Kratos_All,
Kratos_Local,
Kratos_Ghost,
Kratos_Ownership_Size
};



typedef std::size_t IndexType;

typedef std::size_t SizeType;

typedef Dof<double> DofType;
typedef std::vector< DofType::Pointer > DofsVectorType;
typedef Variable<double> DoubleVariableType;
typedef Matrix MatrixType;
typedef Vector VectorType;

typedef PointerVectorSet<DofType> DofsArrayType;

typedef Node NodeType;
typedef Geometry<NodeType> GeometryType;
typedef Properties PropertiesType;
typedef Element ElementType;
typedef Condition ConditionType;

typedef Mesh<NodeType, PropertiesType, ElementType, ConditionType> MeshType;

typedef PointerVector<MeshType> MeshesContainerType;

typedef MeshType::NodesContainerType NodesContainerType;


typedef MeshType::NodeIterator NodeIterator;


typedef MeshType::NodeConstantIterator NodeConstantIterator;



typedef MeshType::PropertiesContainerType PropertiesContainerType;


typedef MeshType::PropertiesIterator PropertiesIterator;


typedef MeshType::PropertiesConstantIterator PropertiesConstantIterator;



typedef MeshType::ElementsContainerType ElementsContainerType;


typedef MeshType::ElementIterator ElementIterator;


typedef MeshType::ElementConstantIterator ElementConstantIterator;

typedef MeshType::ConditionsContainerType ConditionsContainerType;


typedef MeshType::ConditionIterator ConditionIterator;


typedef MeshType::ConditionConstantIterator ConditionConstantIterator;

typedef Table<double,double> TableType;

typedef PointerVectorMap<SizeType, TableType> TablesContainerType;


typedef TablesContainerType::iterator TableIterator;


typedef TablesContainerType::const_iterator TableConstantIterator;

typedef MeshType::MasterSlaveConstraintType MasterSlaveConstraintType;
typedef MeshType::MasterSlaveConstraintContainerType MasterSlaveConstraintContainerType;


typedef MeshType::MasterSlaveConstraintIteratorType MasterSlaveConstraintIteratorType;


typedef MeshType::MasterSlaveConstraintConstantIteratorType MasterSlaveConstraintConstantIteratorType;


typedef GeometryContainer<GeometryType> GeometryContainerType;

typedef typename GeometryContainerType::GeometryIterator GeometryIterator;

typedef typename GeometryContainerType::GeometryConstantIterator GeometryConstantIterator;

typedef typename GeometryContainerType::GeometriesMapType GeometriesMapType;


typedef PointerHashMapSet<ModelPart, std::hash< std::string >, GetModelPartName, Kratos::shared_ptr<ModelPart> >  SubModelPartsContainerType;


typedef SubModelPartsContainerType::iterator SubModelPartIterator;


typedef SubModelPartsContainerType::const_iterator SubModelPartConstantIterator;


KRATOS_DEFINE_LOCAL_FLAG(ALL_ENTITIES);
KRATOS_DEFINE_LOCAL_FLAG(OVERWRITE_ENTITIES);



~ModelPart() override;


ModelPart & operator=(ModelPart const& rOther) = delete;

void Clear();

void Reset();


IndexType CreateSolutionStep();

IndexType CloneSolutionStep();



IndexType CloneTimeStep();

IndexType CreateTimeStep(double NewTime);

IndexType CloneTimeStep(double NewTime);

void OverwriteSolutionStepData(IndexType SourceSolutionStepIndex, IndexType DestinationSourceSolutionStepIndex);

Model& GetModel()
{
return mrModel;
}

const Model& GetModel() const
{
return mrModel;
}

void ReduceTimeStep(ModelPart& rModelPart, double NewTime);


SizeType NumberOfNodes(IndexType ThisIndex = 0) const
{
return GetMesh(ThisIndex).NumberOfNodes();
}


void AddNode(NodeType::Pointer pNewNode, IndexType ThisIndex = 0);


void AddNodes(std::vector<IndexType> const& NodeIds, IndexType ThisIndex = 0);


template<class TIteratorType >
void AddNodes(TIteratorType nodes_begin,  TIteratorType nodes_end, IndexType ThisIndex = 0)
{
KRATOS_TRY
ModelPart::NodesContainerType  aux;
ModelPart::NodesContainerType  aux_root; 
ModelPart* root_model_part = &this->GetRootModelPart();

for(TIteratorType it = nodes_begin; it!=nodes_end; it++)
{
auto it_found = root_model_part->Nodes().find(it->Id());
if(it_found == root_model_part->NodesEnd()) 
{
aux_root.push_back( *(it.base()) ); 
aux.push_back( *(it.base()) );
}
else 
{
if(&(*it_found) != &(*it))
KRATOS_ERROR << "attempting to add a new node with Id :" << it_found->Id() << ", unfortunately a (different) node with the same Id already exists" << std::endl;
else
aux.push_back( *(it.base()) );
}
}

for(auto it = aux_root.begin(); it!=aux_root.end(); it++)
root_model_part->Nodes().push_back( *(it.base()) );
root_model_part->Nodes().Unique();


ModelPart* current_part = this;
while(current_part->IsSubModelPart())
{
for(auto it = aux.begin(); it!=aux.end(); it++)
current_part->Nodes().push_back( *(it.base()) );

current_part->Nodes().Unique();

current_part = &(current_part->GetParentModelPart());
}

KRATOS_CATCH("")
}


NodeType::Pointer CreateNewNode(int Id, double x, double y, double z, VariablesList::Pointer pNewVariablesList, IndexType ThisIndex = 0);

NodeType::Pointer CreateNewNode(IndexType Id, double x, double y, double z, IndexType ThisIndex = 0);

NodeType::Pointer CreateNewNode(IndexType Id, double x, double y, double z, double* pThisData, IndexType ThisIndex = 0);

NodeType::Pointer CreateNewNode(IndexType NodeId, NodeType const& rSourceNode, IndexType ThisIndex = 0);

void AssignNode(NodeType::Pointer pThisNode, IndexType ThisIndex = 0);


bool HasNode(IndexType NodeId, IndexType ThisIndex = 0) const
{
return GetMesh(ThisIndex).HasNode(NodeId);
}


NodeType::Pointer pGetNode(IndexType NodeId, IndexType ThisIndex = 0)
{
return GetMesh(ThisIndex).pGetNode(NodeId);
}


const NodeType::Pointer pGetNode(const IndexType NodeId, const IndexType ThisIndex = 0) const
{
return GetMesh(ThisIndex).pGetNode(NodeId);
}


NodeType& GetNode(IndexType NodeId, IndexType ThisIndex = 0)
{
return GetMesh(ThisIndex).GetNode(NodeId);
}

const NodeType& GetNode(IndexType NodeId, IndexType ThisIndex = 0) const
{
return GetMesh(ThisIndex).GetNode(NodeId);
}


void RemoveNode(IndexType NodeId, IndexType ThisIndex = 0);


void RemoveNode(NodeType& ThisNode, IndexType ThisIndex = 0);


void RemoveNode(NodeType::Pointer pThisNode, IndexType ThisIndex = 0);


void RemoveNodeFromAllLevels(IndexType NodeId, IndexType ThisIndex = 0);


void RemoveNodeFromAllLevels(NodeType& ThisNode, IndexType ThisIndex = 0);


void RemoveNodeFromAllLevels(NodeType::Pointer pThisNode, IndexType ThisIndex = 0);


void RemoveNodes(Flags IdentifierFlag = TO_ERASE);


void RemoveNodesFromAllLevels(Flags IdentifierFlag = TO_ERASE);


ModelPart& GetRootModelPart();


const ModelPart& GetRootModelPart() const;

NodeIterator NodesBegin(IndexType ThisIndex = 0)
{
return GetMesh(ThisIndex).NodesBegin();
}

NodeConstantIterator NodesBegin(IndexType ThisIndex = 0) const
{
return GetMesh(ThisIndex).NodesBegin();
}

NodeIterator NodesEnd(IndexType ThisIndex = 0)
{
return GetMesh(ThisIndex).NodesEnd();
}

NodeConstantIterator NodesEnd(IndexType ThisIndex = 0) const
{
return GetMesh(ThisIndex).NodesEnd();
}

NodesContainerType& Nodes(IndexType ThisIndex = 0)
{
return GetMesh(ThisIndex).Nodes();
}

const NodesContainerType& Nodes(IndexType ThisIndex = 0) const
{
return GetMesh(ThisIndex).Nodes();
}

NodesContainerType::Pointer pNodes(IndexType ThisIndex = 0)
{
return GetMesh(ThisIndex).pNodes();
}

void SetNodes(NodesContainerType::Pointer pOtherNodes, IndexType ThisIndex = 0)
{
GetMesh(ThisIndex).SetNodes(pOtherNodes);
}

NodesContainerType::ContainerType& NodesArray(IndexType ThisIndex = 0)
{
return GetMesh(ThisIndex).NodesArray();
}

void AddNodalSolutionStepVariable(VariableData const& ThisVariable)
{
if (!HasNodalSolutionStepVariable(ThisVariable)) {
KRATOS_ERROR_IF((this->GetRootModelPart()).Nodes().size() != 0)
<< "Attempting to add the variable \"" << ThisVariable.Name()
<< "\" to the model part with name \"" << this->Name() << "\" which is not empty" << std::endl;

mpVariablesList->Add(ThisVariable);
}
}

bool HasNodalSolutionStepVariable(VariableData const& ThisVariable) const
{
return mpVariablesList->Has(ThisVariable);
}

VariablesList& GetNodalSolutionStepVariablesList()
{
return *mpVariablesList;
}

VariablesList const& GetNodalSolutionStepVariablesList() const
{
return *mpVariablesList;
}

VariablesList::Pointer pGetNodalSolutionStepVariablesList() const
{
return mpVariablesList;
}

void SetNodalSolutionStepVariablesList();

void SetNodalSolutionStepVariablesList(VariablesList::Pointer pNewVariablesList)
{
mpVariablesList = pNewVariablesList;
}

SizeType GetNodalSolutionStepDataSize()
{
return mpVariablesList->DataSize();
}

SizeType GetNodalSolutionStepTotalDataSize()
{
return mpVariablesList->DataSize() * mBufferSize;
}


SizeType NumberOfTables() const
{
return mTables.size();
}


void AddTable(IndexType TableId, TableType::Pointer pNewTable);


TableType::Pointer pGetTable(IndexType TableId)
{
return mTables(TableId);
}


TableType& GetTable(IndexType TableId)
{
return mTables[TableId];
}


void RemoveTable(IndexType TableId);


void RemoveTableFromAllLevels(IndexType TableId);


TableIterator TablesBegin()
{
return mTables.begin();
}

TableConstantIterator TablesBegin() const
{
return mTables.begin();
}

TableIterator TablesEnd()
{
return mTables.end();
}

TableConstantIterator TablesEnd() const
{
return mTables.end();
}

TablesContainerType& Tables()
{
return mTables;
}

TablesContainerType::ContainerType& TablesArray()
{
return mTables.GetContainer();
}


SizeType NumberOfMasterSlaveConstraints(IndexType ThisIndex = 0) const
{
return GetMesh(ThisIndex).NumberOfMasterSlaveConstraints();
}

MasterSlaveConstraintContainerType& MasterSlaveConstraints(IndexType ThisIndex = 0)
{
return GetMesh(ThisIndex).MasterSlaveConstraints();
}

const MasterSlaveConstraintContainerType& MasterSlaveConstraints(IndexType ThisIndex = 0) const
{
return GetMesh(ThisIndex).MasterSlaveConstraints();
}

MasterSlaveConstraintConstantIteratorType  MasterSlaveConstraintsBegin(IndexType ThisIndex = 0) const
{
return GetMesh(ThisIndex).MasterSlaveConstraintsBegin();
}

MasterSlaveConstraintConstantIteratorType  MasterSlaveConstraintsEnd(IndexType ThisIndex = 0) const
{
return GetMesh(ThisIndex).MasterSlaveConstraintsEnd();
}

MasterSlaveConstraintIteratorType  MasterSlaveConstraintsBegin(IndexType ThisIndex = 0)
{
return GetMesh(ThisIndex).MasterSlaveConstraintsBegin();
}

MasterSlaveConstraintIteratorType  MasterSlaveConstraintsEnd(IndexType ThisIndex = 0)
{
return GetMesh(ThisIndex).MasterSlaveConstraintsEnd();
}


void AddMasterSlaveConstraint(MasterSlaveConstraintType::Pointer pNewMasterSlaveConstraint, IndexType ThisIndex = 0);


void AddMasterSlaveConstraints(std::vector<IndexType> const& MasterSlaveConstraintIds, IndexType ThisIndex = 0);


template<class TIteratorType >
void AddMasterSlaveConstraints(TIteratorType constraints_begin,  TIteratorType constraints_end, IndexType ThisIndex = 0)
{
KRATOS_TRY
ModelPart::MasterSlaveConstraintContainerType  aux;
ModelPart::MasterSlaveConstraintContainerType  aux_root;
ModelPart* root_model_part = &this->GetRootModelPart();

for(TIteratorType it = constraints_begin; it!=constraints_end; it++)
{
auto it_found = root_model_part->MasterSlaveConstraints().find(it->Id());
if(it_found == root_model_part->MasterSlaveConstraintsEnd()) 
{
aux_root.push_back( *(it.base()) );
aux.push_back( *(it.base()) );
}
else 
{
if(&(*it_found) != &(*it))
KRATOS_ERROR << "attempting to add a new master-slave constraint with Id :" << it_found->Id() << ", unfortunately a (different) master-slave constraint with the same Id already exists" << std::endl;
else
aux.push_back( *(it.base()) );
}
}

for(auto it = aux_root.begin(); it!=aux_root.end(); it++)
root_model_part->MasterSlaveConstraints().push_back( *(it.base()) );
root_model_part->MasterSlaveConstraints().Unique();


ModelPart* current_part = this;
while(current_part->IsSubModelPart())
{
for(auto it = aux.begin(); it!=aux.end(); it++)
current_part->MasterSlaveConstraints().push_back( *(it.base()) );

current_part->MasterSlaveConstraints().Unique();

current_part = &(current_part->GetParentModelPart());
}

KRATOS_CATCH("")
}


MasterSlaveConstraint::Pointer CreateNewMasterSlaveConstraint(const std::string& ConstraintName,
IndexType Id,
DofsVectorType& rMasterDofsVector,
DofsVectorType& rSlaveDofsVector,
const MatrixType& RelationMatrix,
const VectorType& ConstantVector,
IndexType ThisIndex = 0);

MasterSlaveConstraint::Pointer CreateNewMasterSlaveConstraint(const std::string& ConstraintName,
IndexType Id,
NodeType& rMasterNode,
const DoubleVariableType& rMasterVariable,
NodeType& rSlaveNode,
const DoubleVariableType& rSlaveVariable,
const double Weight,
const double Constant,
IndexType ThisIndex = 0);


void RemoveMasterSlaveConstraint(IndexType MasterSlaveConstraintId, IndexType ThisIndex = 0);


void RemoveMasterSlaveConstraint(MasterSlaveConstraintType& ThisMasterSlaveConstraint, IndexType ThisIndex = 0);


void RemoveMasterSlaveConstraintFromAllLevels(IndexType MasterSlaveConstraintId, IndexType ThisIndex = 0);


void RemoveMasterSlaveConstraintFromAllLevels(MasterSlaveConstraintType& ThisMasterSlaveConstraint, IndexType ThisIndex = 0);


void RemoveMasterSlaveConstraints(Flags IdentifierFlag = TO_ERASE);


void RemoveMasterSlaveConstraintsFromAllLevels(Flags IdentifierFlag = TO_ERASE);


bool HasMasterSlaveConstraint(
const IndexType MasterSlaveConstraintId,
IndexType ThisIndex = 0
) const
{
return GetMesh(ThisIndex).HasMasterSlaveConstraint(MasterSlaveConstraintId);
}


MasterSlaveConstraintType::Pointer pGetMasterSlaveConstraint(IndexType ConstraintId, IndexType ThisIndex = 0);


MasterSlaveConstraintType& GetMasterSlaveConstraint(IndexType MasterSlaveConstraintId, IndexType ThisIndex = 0);

const MasterSlaveConstraintType& GetMasterSlaveConstraint(IndexType MasterSlaveConstraintId, IndexType ThisIndex = 0) const ;



SizeType NumberOfProperties(IndexType ThisIndex = 0) const;


void AddProperties(PropertiesType::Pointer pNewProperties, IndexType ThisIndex = 0);


bool HasProperties(IndexType PropertiesId, IndexType MeshIndex = 0) const;


bool RecursivelyHasProperties(IndexType PropertiesId, IndexType MeshIndex = 0) const;


PropertiesType::Pointer CreateNewProperties(IndexType PropertiesId, IndexType MeshIndex = 0);


PropertiesType::Pointer pGetProperties(IndexType PropertiesId, IndexType MeshIndex = 0);


const PropertiesType::Pointer pGetProperties(IndexType PropertiesId, IndexType MeshIndex = 0) const;


PropertiesType& GetProperties(IndexType PropertiesId, IndexType MeshIndex = 0);


const PropertiesType& GetProperties(IndexType PropertiesId, IndexType MeshIndex = 0) const;


bool HasProperties(
const std::string& rAddress,
IndexType MeshIndex = 0
) const;


PropertiesType::Pointer pGetProperties(
const std::string& rAddress,
IndexType MeshIndex = 0
);


const PropertiesType::Pointer pGetProperties(
const std::string& rAddress,
IndexType MeshIndex = 0
) const;


PropertiesType& GetProperties(
const std::string& rAddress,
IndexType MeshIndex = 0
);


const PropertiesType& GetProperties(
const std::string& rAddress,
IndexType MeshIndex = 0
) const;


void RemoveProperties(IndexType PropertiesId, IndexType ThisIndex = 0);


void RemoveProperties(PropertiesType& ThisProperties, IndexType ThisIndex = 0);


void RemoveProperties(PropertiesType::Pointer pThisProperties, IndexType ThisIndex = 0);


void RemovePropertiesFromAllLevels(IndexType PropertiesId, IndexType ThisIndex = 0);


void RemovePropertiesFromAllLevels(PropertiesType& ThisProperties, IndexType ThisIndex = 0);


void RemovePropertiesFromAllLevels(PropertiesType::Pointer pThisProperties, IndexType ThisIndex = 0);

PropertiesIterator PropertiesBegin(IndexType ThisIndex = 0)
{
return GetMesh(ThisIndex).PropertiesBegin();
}

PropertiesConstantIterator PropertiesBegin(IndexType ThisIndex = 0) const
{
return GetMesh(ThisIndex).PropertiesBegin();
}

PropertiesIterator PropertiesEnd(IndexType ThisIndex = 0)
{
return GetMesh(ThisIndex).PropertiesEnd();
}

PropertiesConstantIterator PropertiesEnd(IndexType ThisIndex = 0) const
{
return GetMesh(ThisIndex).PropertiesEnd();
}


PropertiesContainerType& rProperties(IndexType ThisIndex = 0)
{
return GetMesh(ThisIndex).Properties();
}

PropertiesContainerType::Pointer pProperties(IndexType ThisIndex = 0)
{
return GetMesh(ThisIndex).pProperties();
}

void SetProperties(PropertiesContainerType::Pointer pOtherProperties, IndexType ThisIndex = 0)
{
GetMesh(ThisIndex).SetProperties(pOtherProperties);
}

PropertiesContainerType::ContainerType& PropertiesArray(IndexType ThisIndex = 0)
{
return GetMesh(ThisIndex).PropertiesArray();
}


SizeType NumberOfElements(IndexType ThisIndex = 0) const
{
return GetMesh(ThisIndex).NumberOfElements();
}


void AddElement(ElementType::Pointer pNewElement, IndexType ThisIndex = 0);


void AddElements(std::vector<IndexType> const& ElementIds, IndexType ThisIndex = 0);


template<class TIteratorType >
void AddElements(TIteratorType elements_begin,  TIteratorType elements_end, IndexType ThisIndex = 0)
{
KRATOS_TRY
ModelPart::ElementsContainerType  aux;
ModelPart::ElementsContainerType  aux_root;
ModelPart* root_model_part = &this->GetRootModelPart();

for(TIteratorType it = elements_begin; it!=elements_end; it++)
{
auto it_found = root_model_part->Elements().find(it->Id());
if(it_found == root_model_part->ElementsEnd()) 
{
aux_root.push_back( *(it.base()) );
aux.push_back( *(it.base()) );
}
else 
{
if(&(*it_found) != &(*it))
KRATOS_ERROR << "attempting to add a new element with Id :" << it_found->Id() << ", unfortunately a (different) element with the same Id already exists" << std::endl;
else
aux.push_back( *(it.base()) );
}
}

for(auto it = aux_root.begin(); it!=aux_root.end(); it++)
root_model_part->Elements().push_back( *(it.base()) );
root_model_part->Elements().Unique();


ModelPart* current_part = this;
while(current_part->IsSubModelPart())
{
for(auto it = aux.begin(); it!=aux.end(); it++)
current_part->Elements().push_back( *(it.base()) );

current_part->Elements().Unique();

current_part = &(current_part->GetParentModelPart());
}

KRATOS_CATCH("")
}

ElementType::Pointer CreateNewElement(std::string ElementName,
IndexType Id, std::vector<IndexType> ElementNodeIds,
PropertiesType::Pointer pProperties, IndexType ThisIndex = 0);

ElementType::Pointer CreateNewElement(std::string ElementName,
IndexType Id, Geometry< Node >::PointsArrayType pElementNodes,
PropertiesType::Pointer pProperties, IndexType ThisIndex = 0);

ElementType::Pointer CreateNewElement(std::string ElementName,
IndexType Id, typename GeometryType::Pointer pGeometry,
PropertiesType::Pointer pProperties, IndexType ThisIndex = 0);


bool HasElement(IndexType ElementId, IndexType ThisIndex = 0) const
{
return GetMesh(ThisIndex).HasElement(ElementId);
}


ElementType::Pointer pGetElement(IndexType ElementId, IndexType ThisIndex = 0)
{
return GetMesh(ThisIndex).pGetElement(ElementId);
}


const ElementType::Pointer pGetElement(const IndexType ElementId, const IndexType ThisIndex = 0) const
{
return GetMesh(ThisIndex).pGetElement(ElementId);
}


ElementType& GetElement(IndexType ElementId, IndexType ThisIndex = 0)
{
return GetMesh(ThisIndex).GetElement(ElementId);
}

const ElementType& GetElement(IndexType ElementId, IndexType ThisIndex = 0) const
{
return GetMesh(ThisIndex).GetElement(ElementId);
}


void RemoveElement(IndexType ElementId, IndexType ThisIndex = 0);


void RemoveElement(ElementType& ThisElement, IndexType ThisIndex = 0);


void RemoveElement(ElementType::Pointer pThisElement, IndexType ThisIndex = 0);


void RemoveElementFromAllLevels(IndexType ElementId, IndexType ThisIndex = 0);


void RemoveElementFromAllLevels(ElementType& ThisElement, IndexType ThisIndex = 0);


void RemoveElementFromAllLevels(ElementType::Pointer pThisElement, IndexType ThisIndex = 0);


void RemoveElements(Flags IdentifierFlag = TO_ERASE);


void RemoveElementsFromAllLevels(Flags IdentifierFlag = TO_ERASE);

ElementIterator ElementsBegin(IndexType ThisIndex = 0)
{
return GetMesh(ThisIndex).ElementsBegin();
}

ElementConstantIterator ElementsBegin(IndexType ThisIndex = 0) const
{
return GetMesh(ThisIndex).ElementsBegin();
}

ElementIterator ElementsEnd(IndexType ThisIndex = 0)
{
return GetMesh(ThisIndex).ElementsEnd();
}

ElementConstantIterator ElementsEnd(IndexType ThisIndex = 0) const
{
return GetMesh(ThisIndex).ElementsEnd();
}

ElementsContainerType& Elements(IndexType ThisIndex = 0)
{
return GetMesh(ThisIndex).Elements();
}

const ElementsContainerType& Elements(IndexType ThisIndex = 0) const
{
return GetMesh(ThisIndex).Elements();
}

ElementsContainerType::Pointer pElements(IndexType ThisIndex = 0)
{
return GetMesh(ThisIndex).pElements();
}

void SetElements(ElementsContainerType::Pointer pOtherElements, IndexType ThisIndex = 0)
{
GetMesh(ThisIndex).SetElements(pOtherElements);
}

ElementsContainerType::ContainerType& ElementsArray(IndexType ThisIndex = 0)
{
return GetMesh(ThisIndex).ElementsArray();
}


SizeType NumberOfConditions(IndexType ThisIndex = 0) const
{
return GetMesh(ThisIndex).NumberOfConditions();
}


void AddCondition(ConditionType::Pointer pNewCondition, IndexType ThisIndex = 0);


void AddConditions(std::vector<IndexType> const& ConditionIds, IndexType ThisIndex = 0);


template<class TIteratorType >
void AddConditions(TIteratorType conditions_begin,  TIteratorType conditions_end, IndexType ThisIndex = 0)
{
KRATOS_TRY
ModelPart::ConditionsContainerType  aux;
ModelPart::ConditionsContainerType  aux_root;
ModelPart* root_model_part = &this->GetRootModelPart();

for(TIteratorType it = conditions_begin; it!=conditions_end; it++)
{
auto it_found = root_model_part->Conditions().find(it->Id());
if(it_found == root_model_part->ConditionsEnd()) 
{
aux.push_back( *(it.base()) );
aux_root.push_back( *(it.base()) );
}
else 
{
if(&(*it_found) != &(*it))
KRATOS_ERROR << "attempting to add a new Condition with Id :" << it_found->Id() << ", unfortunately a (different) Condition with the same Id already exists" << std::endl;
else
aux.push_back( *(it.base()) );
}
}

for(auto it = aux_root.begin(); it!=aux_root.end(); it++)
root_model_part->Conditions().push_back( *(it.base()) );
root_model_part->Conditions().Unique();


ModelPart* current_part = this;
while(current_part->IsSubModelPart())
{
for(auto it = aux.begin(); it!=aux.end(); it++)
current_part->Conditions().push_back( *(it.base()) );

current_part->Conditions().Unique();

current_part = &(current_part->GetParentModelPart());
}

KRATOS_CATCH("")
}

ConditionType::Pointer CreateNewCondition(std::string ConditionName,
IndexType Id, std::vector<IndexType> ConditionNodeIds,
PropertiesType::Pointer pProperties, IndexType ThisIndex = 0);

ConditionType::Pointer CreateNewCondition(std::string ConditionName,
IndexType Id, Geometry< Node >::PointsArrayType pConditionNodes,
PropertiesType::Pointer pProperties, IndexType ThisIndex = 0);

ConditionType::Pointer CreateNewCondition(std::string ConditionName,
IndexType Id, typename GeometryType::Pointer pGeometry,
PropertiesType::Pointer pProperties, IndexType ThisIndex = 0);


bool HasCondition(IndexType ConditionId, IndexType ThisIndex = 0) const
{
return GetMesh(ThisIndex).HasCondition(ConditionId);
}


ConditionType::Pointer pGetCondition(IndexType ConditionId, IndexType ThisIndex = 0)
{
return GetMesh(ThisIndex).pGetCondition(ConditionId);
}


const ConditionType::Pointer pGetCondition(const IndexType ConditionId, const IndexType ThisIndex = 0) const
{
return GetMesh(ThisIndex).pGetCondition(ConditionId);
}


ConditionType& GetCondition(IndexType ConditionId, IndexType ThisIndex = 0)
{
return GetMesh(ThisIndex).GetCondition(ConditionId);
}

const ConditionType& GetCondition(IndexType ConditionId, IndexType ThisIndex = 0) const
{
return GetMesh(ThisIndex).GetCondition(ConditionId);
}


void RemoveCondition(IndexType ConditionId, IndexType ThisIndex = 0);


void RemoveCondition(ConditionType& ThisCondition, IndexType ThisIndex = 0);


void RemoveCondition(ConditionType::Pointer pThisCondition, IndexType ThisIndex = 0);


void RemoveConditionFromAllLevels(IndexType ConditionId, IndexType ThisIndex = 0);


void RemoveConditionFromAllLevels(ConditionType& ThisCondition, IndexType ThisIndex = 0);


void RemoveConditionFromAllLevels(ConditionType::Pointer pThisCondition, IndexType ThisIndex = 0);


void RemoveConditions(Flags IdentifierFlag = TO_ERASE);


void RemoveConditionsFromAllLevels(Flags IdentifierFlag = TO_ERASE);

ConditionIterator ConditionsBegin(IndexType ThisIndex = 0)
{
return GetMesh(ThisIndex).ConditionsBegin();
}

ConditionConstantIterator ConditionsBegin(IndexType ThisIndex = 0) const
{
return GetMesh(ThisIndex).ConditionsBegin();
}

ConditionIterator ConditionsEnd(IndexType ThisIndex = 0)
{
return GetMesh(ThisIndex).ConditionsEnd();
}

ConditionConstantIterator ConditionsEnd(IndexType ThisIndex = 0) const
{
return GetMesh(ThisIndex).ConditionsEnd();
}

ConditionsContainerType& Conditions(IndexType ThisIndex = 0)
{
return GetMesh(ThisIndex).Conditions();
}

const ConditionsContainerType& Conditions(IndexType ThisIndex = 0) const
{
return GetMesh(ThisIndex).Conditions();
}

ConditionsContainerType::Pointer pConditions(IndexType ThisIndex = 0)
{
return GetMesh(ThisIndex).pConditions();
}

void SetConditions(ConditionsContainerType::Pointer pOtherConditions, IndexType ThisIndex = 0)
{
GetMesh(ThisIndex).SetConditions(pOtherConditions);
}

ConditionsContainerType::ContainerType& ConditionsArray(IndexType ThisIndex = 0)
{
return GetMesh(ThisIndex).ConditionsArray();
}


SizeType NumberOfGeometries() const
{
return mGeometries.NumberOfGeometries();
}


GeometryType::Pointer CreateNewGeometry(
const std::string& rGeometryTypeName,
const std::vector<IndexType>& rGeometryNodeIds
);


GeometryType::Pointer CreateNewGeometry(
const std::string& rGeometryTypeName,
GeometryType::PointsArrayType pGeometryNodes
);


GeometryType::Pointer CreateNewGeometry(
const std::string& rGeometryTypeName,
GeometryType::Pointer pGeometry
);


GeometryType::Pointer CreateNewGeometry(
const std::string& rGeometryTypeName,
const IndexType GeometryId,
const std::vector<IndexType>& rGeometryNodeIds
);


GeometryType::Pointer CreateNewGeometry(
const std::string& rGeometryTypeName,
const IndexType GeometryId,
GeometryType::PointsArrayType pGeometryNodes
);


GeometryType::Pointer CreateNewGeometry(
const std::string& rGeometryTypeName,
const IndexType GeometryId,
GeometryType::Pointer pGeometry
);


GeometryType::Pointer CreateNewGeometry(
const std::string& rGeometryTypeName,
const std::string& rGeometryIdentifierName,
const std::vector<IndexType>& rGeometryNodeIds
);


GeometryType::Pointer CreateNewGeometry(
const std::string& rGeometryTypeName,
const std::string& rGeometryIdentifierName,
GeometryType::PointsArrayType pGeometryNodes
);


GeometryType::Pointer CreateNewGeometry(
const std::string& rGeometryTypeName,
const std::string& rGeometryIdentifierName,
GeometryType::Pointer pGeometry
);

void AddGeometry(typename GeometryType::Pointer pNewGeometry);

void AddGeometries(std::vector<IndexType> const& GeometriesIds);

template<class TIteratorType >
void AddGeometries(TIteratorType GeometryBegin,  TIteratorType GeometriesEnd, IndexType ThisIndex = 0)
{
KRATOS_TRY
std::vector<GeometryType::Pointer> aux, aux_root;
ModelPart* p_root_model_part = &this->GetRootModelPart();

for(TIteratorType it = GeometryBegin; it!=GeometriesEnd; it++) {
auto it_found = p_root_model_part->Geometries().find(it->Id());
if(it_found == p_root_model_part->GeometriesEnd()) { 
aux_root.push_back( it.operator->() );
aux.push_back( it.operator->() );
} else { 
if(&(*it_found) != &(*it)) { 
KRATOS_ERROR << "Attempting to add a new geometry with Id :" << it_found->Id() << ", unfortunately a (different) element with the same Id already exists" << std::endl;
} else {
aux.push_back( it.operator->() );
}
}
}

for(auto& p_geom : aux_root) {
p_root_model_part->AddGeometry(p_geom);
}

ModelPart* p_current_part = this;
while(p_current_part->IsSubModelPart()) {
for(auto& p_geom : aux) {
p_current_part->AddGeometry(p_geom);
}

p_current_part = &(p_current_part->GetParentModelPart());
}

KRATOS_CATCH("")
}

typename GeometryType::Pointer pGetGeometry(IndexType GeometryId) {
return mGeometries.pGetGeometry(GeometryId);
}

const typename GeometryType::Pointer pGetGeometry(IndexType GeometryId) const {
return mGeometries.pGetGeometry(GeometryId);
}

typename GeometryType::Pointer pGetGeometry(std::string GeometryName) {
return mGeometries.pGetGeometry(GeometryName);
}

const typename GeometryType::Pointer pGetGeometry(std::string GeometryName) const {
return mGeometries.pGetGeometry(GeometryName);
}

GeometryType& GetGeometry(IndexType GeometryId) {
return mGeometries.GetGeometry(GeometryId);
}

const GeometryType& GetGeometry(IndexType GeometryId) const {
return mGeometries.GetGeometry(GeometryId);
}

GeometryType& GetGeometry(std::string GeometryName) {
return mGeometries.GetGeometry(GeometryName);
}

const GeometryType& GetGeometry(std::string GeometryName) const {
return mGeometries.GetGeometry(GeometryName);
}


bool HasGeometry(IndexType GeometryId) const {
return mGeometries.HasGeometry(GeometryId);
}

bool HasGeometry(std::string GeometryName) const {
return mGeometries.HasGeometry(GeometryName);
}


void RemoveGeometry(IndexType GeometryId);

void RemoveGeometry(std::string GeometryName);

void RemoveGeometryFromAllLevels(IndexType GeometryId);

void RemoveGeometryFromAllLevels(std::string GeometryName);


GeometryIterator GeometriesBegin() {
return mGeometries.GeometriesBegin();
}

GeometryConstantIterator GeometriesBegin() const {
return mGeometries.GeometriesBegin();
}

GeometryIterator GeometriesEnd() {
return mGeometries.GeometriesEnd();
}

GeometryConstantIterator GeometriesEnd() const {
return mGeometries.GeometriesEnd();
}


GeometriesMapType& Geometries()
{
return mGeometries.Geometries();
}

const GeometriesMapType& Geometries() const
{
return mGeometries.Geometries();
}


SizeType NumberOfSubModelParts() const
{
return mSubModelParts.size();
}


ModelPart& CreateSubModelPart(std::string const& NewSubModelPartName);


ModelPart& GetSubModelPart(std::string const& SubModelPartName);


const ModelPart& GetSubModelPart(std::string const& SubModelPartName) const;


ModelPart* pGetSubModelPart(std::string const& SubModelPartName);


void RemoveSubModelPart(std::string const& ThisSubModelPartName);


void RemoveSubModelPart(ModelPart& ThisSubModelPart);

SubModelPartIterator SubModelPartsBegin()
{
return mSubModelParts.begin();
}

SubModelPartConstantIterator SubModelPartsBegin() const
{
return mSubModelParts.begin();
}

SubModelPartIterator SubModelPartsEnd()
{
return mSubModelParts.end();
}

SubModelPartConstantIterator SubModelPartsEnd() const
{
return mSubModelParts.end();
}

SubModelPartsContainerType& SubModelParts()
{
return mSubModelParts;
}

const SubModelPartsContainerType& SubModelParts() const
{
return mSubModelParts;
}


ModelPart& GetParentModelPart();


const ModelPart& GetParentModelPart() const;


bool HasSubModelPart(std::string const& ThisSubModelPartName) const;


ProcessInfo& GetProcessInfo()
{
return *mpProcessInfo;
}

ProcessInfo const& GetProcessInfo() const
{
return *mpProcessInfo;
}

ProcessInfo::Pointer pGetProcessInfo()
{
return mpProcessInfo;
}

const ProcessInfo::Pointer pGetProcessInfo() const
{
return mpProcessInfo;
}

void SetProcessInfo(ProcessInfo::Pointer pNewProcessInfo)
{
mpProcessInfo = pNewProcessInfo;
}

void SetProcessInfo(ProcessInfo& NewProcessInfo)
{
*mpProcessInfo = NewProcessInfo;
}

SizeType NumberOfMeshes()
{
return mMeshes.size();
}

MeshType::Pointer pGetMesh(IndexType ThisIndex = 0)
{
return mMeshes(ThisIndex);
}

const MeshType::Pointer pGetMesh(IndexType ThisIndex = 0) const
{
return mMeshes(ThisIndex);
}

MeshType& GetMesh(IndexType ThisIndex = 0)
{
return mMeshes[ThisIndex];
}

MeshType const& GetMesh(IndexType ThisIndex = 0) const
{
return mMeshes[ThisIndex];
}

MeshesContainerType& GetMeshes()
{
return mMeshes;
}

MeshesContainerType const& GetMeshes() const
{
return mMeshes;
}

std::string& Name()
{
return mName;
}

std::string const& Name() const
{
return mName;
}

Communicator& GetCommunicator()
{
return *mpCommunicator;
}

Communicator const& GetCommunicator() const
{
return *mpCommunicator;
}

Communicator::Pointer pGetCommunicator()
{
return mpCommunicator;
}

void SetCommunicator(Communicator::Pointer pNewCommunicator)
{
mpCommunicator = pNewCommunicator;
}



std::string FullName() const
{
std::string full_name = this->Name();
if (this->IsSubModelPart()) {
full_name = this->GetParentModelPart().FullName() + "." + full_name;
}
return full_name;
}


std::vector<std::string> GetSubModelPartNames() const;


void SetBufferSize(IndexType NewBufferSize);


IndexType GetBufferSize() const
{
return mBufferSize;
}

virtual int Check() const;




bool IsSubModelPart() const
{
return (mpParentModelPart != NULL);
}

bool IsDistributed() const
{
return mpCommunicator->IsDistributed();
}


std::string Info() const override;

void PrintInfo(std::ostream& rOStream) const override;

void PrintData(std::ostream& rOStream) const override;

virtual void PrintInfo(std::ostream& rOStream, std::string const& PrefixString) const;

virtual void PrintData(std::ostream& rOStream, std::string const& PrefixString) const;





private:

friend class Model;

ModelPart(VariablesList::Pointer pVariableList, Model& rOwnerModel);

ModelPart(std::string const& NewName,VariablesList::Pointer pVariableList, Model& rOwnerModel);

ModelPart(std::string const& NewName, IndexType NewBufferSize,VariablesList::Pointer pVariableList, Model& rOwnerModel);

ModelPart(ModelPart const& rOther) = delete;





std::string mName; 

IndexType mBufferSize; 

ProcessInfo::Pointer mpProcessInfo; 

TablesContainerType mTables; 

MeshesContainerType mMeshes; 

GeometryContainerType mGeometries; 

VariablesList::Pointer mpVariablesList; 

Communicator::Pointer mpCommunicator; 

ModelPart* mpParentModelPart = NULL; 

SubModelPartsContainerType mSubModelParts; 

Model& mrModel; 





std::vector<IndexType> TrimComponentName(const std::string& rStringName) const
{
std::vector<IndexType> list_indexes;

std::stringstream ss(rStringName);
for (std::string index_string; std::getline(ss, index_string, '.'); ) {
list_indexes.push_back(std::stoi(index_string));
}

KRATOS_ERROR_IF(list_indexes.size() == 0) << "Properties:: Empty list of indexes when reading suproperties" << std::endl;

return list_indexes;
}


void SetBufferSizeSubModelParts(IndexType NewBufferSize);


void SetParentModelPart(ModelPart* pParentModelPart)
{
mpParentModelPart = pParentModelPart;
}

template <typename TEntitiesContainerType>
void AddEntities(TEntitiesContainerType const& Source, TEntitiesContainerType& rDestination, Flags Options)
{
}


[[ noreturn ]] void ErrorNonExistingSubModelPart(const std::string& rSubModelPartName) const;


friend class Serializer;

void save(Serializer& rSerializer) const override;

void load(Serializer& rSerializer) override;








}; 






KRATOS_API(KRATOS_CORE) inline std::istream & operator >>(std::istream& rIStream,
ModelPart& rThis)
{
return rIStream;
}
KRATOS_API(KRATOS_CORE) inline std::ostream & operator <<(std::ostream& rOStream,
const ModelPart& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}


} 
