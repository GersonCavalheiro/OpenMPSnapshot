
#pragma once



#include "includes/define.h"
#include "processes/process.h"
#include "includes/model_part.h"
#include "includes/kratos_parameters.h"

#include "utilities/function_parser_utility.h"
#include "utilities/mortar_utilities.h"
namespace Kratos {





class KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) SetMovingLoadProcess : public Process
{
public:


KRATOS_CLASS_POINTER_DEFINITION(SetMovingLoadProcess);

typedef std::size_t SizeType;


SetMovingLoadProcess(ModelPart& rModelPart,
Parameters Parameters);



void ExecuteInitialize() override;


void ExecuteInitializeSolutionStep() override;


void ExecuteFinalizeSolutionStep() override;



virtual std::string Info() const override {
return "SetMovingLoadProcess";
}

void PrintInfo(std::ostream& rOStream) const override {
rOStream << "SetMovingLoadProcess";
}

void PrintData(std::ostream& rOStream) const override {
}


private:

ModelPart& mrModelPart;
Parameters mParameters;

std::vector<Condition> mSortedConditions;

std::vector<bool> mIsCondReversedVector;

array_1d<double, 3> mOriginPoint;

array_1d<int,3> mDirection;

double mCurrentDistance;

bool mUseLoadFunction;

bool mUseVelocityFunction;

std::vector<BasicGenericFunctionUtility> mLoadFunctions;



static std::vector<IndexType> FindNonRepeatingIndices(const std::vector<IndexType> IndicesVector);


std::vector<Condition> FindEndConditions();


std::vector<Condition> SortConditions(ModelPart::ConditionsContainerType& rUnsortedConditions, Condition& rFirstCondition);


static bool IsConditionReversed(const Condition& rCondition, const array_1d<int, 3> Direction);


static Condition& GetFirstCondition(const Point FirstPoint, const Point SecondPoint, const array_1d<int, 3> Direction, std::vector<Condition>& rEndConditions);


static Condition& GetFirstConditionFromCoord(const double FirstCoord, const double SecondCoord, const int Direction, std::vector<Condition>& EndConditions);


static bool IsSwapPoints(const double FirstCoord, const double SecondCoord, const int Direction);


void InitializeDistanceLoadInSortedVector();


friend class Serializer;

void save(Serializer& rSerializer) const override;
void load(Serializer& rSerializer) override;


}; 


}  
