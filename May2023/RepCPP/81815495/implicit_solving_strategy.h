
#pragma once






#include "includes/define.h"
#include "includes/model_part.h"
#include "solving_strategy.h"
#include "solving_strategies/schemes/scheme.h"
#include "solving_strategies/builder_and_solvers/builder_and_solver.h"
#include "includes/kratos_parameters.h"

namespace Kratos
{









template<class TSparseSpace, class TDenseSpace, class TLinearSolver>
class ImplicitSolvingStrategy : public SolvingStrategy<TSparseSpace, TDenseSpace>
{
public:

typedef SolvingStrategy<TSparseSpace, TDenseSpace>                              BaseType;

typedef typename BaseType::TDataType                                           TDataType;

typedef typename BaseType::TSystemMatrixType                           TSystemMatrixType;

typedef typename BaseType::TSystemVectorType                           TSystemVectorType;

typedef typename BaseType::TSystemMatrixPointerType             TSystemMatrixPointerType;

typedef typename BaseType::TSystemVectorPointerType             TSystemVectorPointerType;

typedef typename BaseType::LocalSystemMatrixType                   LocalSystemMatrixType;

typedef typename BaseType::LocalSystemVectorType                   LocalSystemVectorType;

typedef Scheme<TSparseSpace, TDenseSpace>                                    TSchemeType;

typedef BuilderAndSolver<TSparseSpace, TDenseSpace, TLinearSolver> TBuilderAndSolverType;

typedef ImplicitSolvingStrategy<TSparseSpace, TDenseSpace, TLinearSolver>      ClassType;

typedef typename BaseType::TDofType                                             TDofType;

typedef typename BaseType::DofsArrayType                                   DofsArrayType;

typedef typename BaseType::NodesArrayType                                 NodesArrayType;

typedef typename BaseType::ElementsArrayType                           ElementsArrayType;

typedef typename BaseType::ConditionsArrayType                       ConditionsArrayType;


KRATOS_CLASS_POINTER_DEFINITION(ImplicitSolvingStrategy);



explicit ImplicitSolvingStrategy() { }


explicit ImplicitSolvingStrategy(
ModelPart& rModelPart,
Parameters ThisParameters)
: BaseType(rModelPart, ThisParameters)
{
ThisParameters = this->ValidateAndAssignParameters(ThisParameters, this->GetDefaultParameters());
this->AssignSettings(ThisParameters);
}


explicit ImplicitSolvingStrategy(
ModelPart& rModelPart,
bool MoveMeshFlag = false)
: BaseType(rModelPart, MoveMeshFlag)
{
}


virtual ~ImplicitSolvingStrategy(){}




typename BaseType::Pointer Create(
ModelPart& rModelPart,
Parameters ThisParameters) const override
{
return Kratos::make_shared<ClassType>(rModelPart, ThisParameters);
}



Parameters GetDefaultParameters() const override
{
Parameters default_parameters = Parameters(R"(
{
"name"                         : "implicit_solving_strategy",
"build_level"                  : 2
})");

const Parameters base_default_parameters = BaseType::GetDefaultParameters();
default_parameters.RecursivelyAddMissingParameters(base_default_parameters);

return default_parameters;
}


static std::string Name()
{
return "implicit_solving_strategy";
}



void SetRebuildLevel(int Level) override
{
mRebuildLevel = Level;
mStiffnessMatrixIsBuilt = false;
}


int GetRebuildLevel() const override
{
return mRebuildLevel;
}


void SetStiffnessMatrixIsBuilt(const bool StiffnessMatrixIsBuilt)
{
mStiffnessMatrixIsBuilt = StiffnessMatrixIsBuilt;
}


bool GetStiffnessMatrixIsBuilt() const
{
return mStiffnessMatrixIsBuilt;
}


std::string Info() const override
{
return "ImplicitSolvingStrategy";
}


protected:

int mRebuildLevel;            
bool mStiffnessMatrixIsBuilt; 






void AssignSettings(const Parameters ThisParameters) override
{
BaseType::AssignSettings(ThisParameters);

mRebuildLevel = ThisParameters["build_level"].GetInt();
}






private:















ImplicitSolvingStrategy(const ImplicitSolvingStrategy& Other);


}; 





} 
