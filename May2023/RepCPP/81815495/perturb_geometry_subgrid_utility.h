
#pragma once



#include "custom_utilities/perturb_geometry_base_utility.h"
#include "linear_solvers/linear_solver.h"

namespace Kratos {



class KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) PerturbGeometrySubgridUtility
: public PerturbGeometryBaseUtility
{
public:


typedef LinearSolver<TDenseSpaceType, TDenseSpaceType>      LinearSolverType;

typedef LinearSolverType::Pointer                           LinearSolverPointerType;

typedef ModelPart::NodesContainerType::ContainerType        ResultNodesContainerType;

KRATOS_CLASS_POINTER_DEFINITION(PerturbGeometrySubgridUtility);


PerturbGeometrySubgridUtility( ModelPart& rInitialModelPart, LinearSolverPointerType pEigenSolver, Parameters Settings) :
PerturbGeometryBaseUtility(rInitialModelPart, Settings){
mpEigenSolver = pEigenSolver;
mMinDistanceSubgrid = Settings["min_distance_subgrid"].GetDouble();
}

~PerturbGeometrySubgridUtility() override
= default;



int CreateRandomFieldVectors() override;


std::string Info() const override
{
return "PerturbGeometrySubgridUtility";
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "PerturbGeometrySubgridUtility";
}

void PrintData(std::ostream& rOStream) const override
{
}


private:

LinearSolverPointerType mpEigenSolver;
double mMinDistanceSubgrid;

PerturbGeometrySubgridUtility& operator=(PerturbGeometrySubgridUtility const& rOther) = delete;

PerturbGeometrySubgridUtility(PerturbGeometrySubgridUtility const& rOther) = delete;


}; 


}