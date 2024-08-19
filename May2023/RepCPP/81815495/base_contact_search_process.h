
#pragma once



#include "includes/model_part.h"
#include "includes/kratos_parameters.h"
#include "custom_processes/normal_gap_process.h"


#include "custom_includes/point_item.h"
#include "custom_conditions/paired_condition.h"


#include "spatial_containers/spatial_containers.h" 

namespace Kratos
{


typedef std::size_t SizeType;





template<SizeType TDim, SizeType TNumNodes, SizeType TNumNodesMaster = TNumNodes>
class KRATOS_API(CONTACT_STRUCTURAL_MECHANICS_APPLICATION) BaseContactSearchProcess
: public Process
{
public:

typedef ModelPart::NodesContainerType                    NodesArrayType;
typedef ModelPart::ConditionsContainerType          ConditionsArrayType;
typedef Node                                                NodeType;
typedef Geometry<NodeType>                                 GeometryType;

typedef std::size_t                                           IndexType;

typedef PointItem<Condition>                                  PointType;
typedef PointType::Pointer                             PointTypePointer;
typedef std::vector<PointTypePointer>                       PointVector;
typedef PointVector::iterator                             PointIterator;
typedef std::vector<double>                              DistanceVector;
typedef DistanceVector::iterator                       DistanceIterator;

typedef Bucket< 3ul, PointType, PointVector, PointTypePointer, PointIterator, DistanceIterator > BucketType;
typedef Tree< KDTreePartition<BucketType> > KDTree;

typedef NormalGapProcess<TDim, TNumNodes, TNumNodesMaster> NormalGapProcessType;

static constexpr double GapThreshold = 2.0e-3;

static constexpr double ZeroTolerance = std::numeric_limits<double>::epsilon();

KRATOS_CLASS_POINTER_DEFINITION( BaseContactSearchProcess );

KRATOS_DEFINE_LOCAL_FLAG( INVERTED_SEARCH );
KRATOS_DEFINE_LOCAL_FLAG( CREATE_AUXILIAR_CONDITIONS );
KRATOS_DEFINE_LOCAL_FLAG( MULTIPLE_SEARCHS );
KRATOS_DEFINE_LOCAL_FLAG( PREDEFINE_MASTER_SLAVE );
KRATOS_DEFINE_LOCAL_FLAG( PURE_SLIP );


enum class SearchTreeType {KdtreeInRadius = 0, KdtreeInBox = 1, KdtreeInRadiusWithOBB = 2, KdtreeInBoxWithOBB = 3, OctreeWithOBB = 4, Kdop = 5};

enum class CheckResult {Fail = 0, AlreadyInTheMap = 1, OK = 2};

enum class CheckGap {NoCheck = 0, DirectCheck = 1, MappingCheck = 2};

enum class TypeSolution {NormalContactStress = 0, ScalarLagrangeMultiplier = 1, VectorLagrangeMultiplier = 2, FrictionlessPenaltyMethod = 3, FrictionalPenaltyMethod = 4, OtherFrictionless = 5, OtherFrictional = 6};



BaseContactSearchProcess(
ModelPart& rMainModelPart,
Parameters ThisParameters =  Parameters(R"({})"),
Properties::Pointer pPairedProperties = nullptr
);

~BaseContactSearchProcess() override = default;


void operator()()
{
Execute();
}



void Execute() override;


void ExecuteInitialize() override;


void ExecuteInitializeSolutionStep() override;


void ExecuteFinalizeSolutionStep() override;


void InitializeMortarConditions();


virtual void ClearMortarConditions();


virtual void CheckContactModelParts();


void CreatePointListMortar();


void UpdatePointListMortar();


void UpdateMortarConditions();


void CheckMortarConditions();


void InvertSearch();


virtual void ResetContactOperators();


const Parameters GetDefaultParameters() const override;







std::string Info() const override
{
return "BaseContactSearchProcess";
}




void PrintInfo(std::ostream& rOStream) const override
{
rOStream << Info();
}



protected:



ModelPart& mrMainModelPart;                       
Parameters mThisParameters;                       
CheckGap mCheckGap;                               
TypeSolution mTypeSolution;                       
std::string mConditionName;                       
PointVector mPointListDestination;                

Properties::Pointer mpPairedProperties = nullptr; 




virtual void CleanModelPart(ModelPart& rModelPart);


virtual void CheckPairing(
ModelPart& rComputingModelPart,
IndexType& rConditionId
);


virtual void ComputeActiveInactiveNodes();


virtual void SetActiveNode(
NodeType& rNode,
const double CommonEpsilon,
const double ScaleFactor = 1.0
);


virtual void SetInactiveNode(NodeType& rNode);


virtual Condition::Pointer AddPairing(
ModelPart& rComputingModelPart,
IndexType& rConditionId,
GeometricalObject::Pointer pObjectSlave,
const array_1d<double, 3>& rSlaveNormal,
GeometricalObject::Pointer pObjectMaster,
const array_1d<double, 3>& rMasterNormal,
IndexMap::Pointer pIndexesPairs,
Properties::Pointer pProperties
);


CheckGap ConvertCheckGap(const std::string& str);



bool IsPureSlip();


bool IsNotPureSlip();


bool IsMultipleSearchs();


bool IsNotMultipleSearchs();


bool IsInvertedSearch();


bool IsNotInvertedSearch();




private:





void SearchUsingKDTree(
ModelPart& rSubContactModelPart,
ModelPart& rSubComputingContactModelPart
);


void SearchUsingOcTree(
ModelPart& rSubContactModelPart,
ModelPart& rSubComputingContactModelPart
);


void SetOriginDestinationModelParts(ModelPart& rModelPart);


void ClearScalarMortarConditions(NodesArrayType& rNodesArray);


void ClearComponentsMortarConditions(NodesArrayType& rNodesArray);


void ClearALMFrictionlessMortarConditions(NodesArrayType& rNodesArray);


inline CheckResult CheckGeometricalObject(
IndexMap::Pointer pIndexesPairs,
const GeometricalObject::Pointer pGeometricalObject1,
const GeometricalObject::Pointer pGeometricalObject2,
const bool InvertedSearch = false
);


inline CheckResult CheckCondition(
IndexMap::Pointer pIndexesPairs,
const Condition::Pointer pCond1,
const Condition::Pointer pCond2,
const bool InvertedSearch = false
);


void FillPointListDestination();


void ClearDestinationListAndAssignFlags(ModelPart& rSubContactModelPart);


inline IndexType PerformKDTreeSearch(
KDTree& rTreePoints,
PointVector& rPointsFound,
GeometryType& rGeometry,
const SearchTreeType TypeSearch = SearchTreeType::KdtreeInBox,
const double SearchFactor = 3.5,
const IndexType AllocationSize = 1000,
const bool Dynamic = false
);


inline IndexType GetMaximumConditionsIds();


void AddPotentialPairing(
ModelPart& rComputingModelPart,
IndexType& rConditionId,
GeometricalObject::Pointer pObjectSlave,
const array_1d<double, 3>& rSlaveNormal,
GeometricalObject::Pointer pObjectMaster,
const array_1d<double, 3>& rMasterNormal,
IndexMap::Pointer pIndexesPairs,
Properties::Pointer pProperties,
const double ActiveCheckFactor,
const bool FrictionalProblem
);


inline void ComputeMappedGap(const bool SearchOrientation);


inline void ComputeWeightedReaction();


inline void CreateAuxiliaryConditions(
ModelPart& rContactModelPart,
ModelPart& rComputingModelPart,
IndexType& rConditionId
);


void CreateDebugFile(
ModelPart& rModelPart,
const std::string& rName
);


static inline double Radius(GeometryType& ThisGeometry);


SearchTreeType ConvertSearchTree(const std::string& str);





}; 








template<SizeType TDim, SizeType TNumNodes, SizeType TNumNodesMaster>
inline std::istream& operator >> (std::istream& rIStream,
BaseContactSearchProcess<TDim, TNumNodes, TNumNodesMaster>& rThis);




template<SizeType TDim, SizeType TNumNodes, SizeType TNumNodesMaster>
inline std::ostream& operator << (std::ostream& rOStream,
const BaseContactSearchProcess<TDim, TNumNodes, TNumNodesMaster>& rThis)
{
return rOStream;
}


}  
