
#pragma once



#include "custom_utilities/perturb_geometry_base_utility.h"
#include "linear_solvers/linear_solver.h"

namespace Kratos {



class KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) PerturbGeometrySparseUtility
: public PerturbGeometryBaseUtility
{
public:


typedef LinearSolver<TSparseSpaceType, TDenseSpaceType> LinearSolverType;

typedef LinearSolverType::Pointer                       LinearSolverPointerType;

typedef ModelPart::NodesContainerType::ContainerType    ResultNodesContainerType;

typedef TSparseSpaceType::MatrixType                    SparseMatrixType;

KRATOS_CLASS_POINTER_DEFINITION(PerturbGeometrySparseUtility);


PerturbGeometrySparseUtility( ModelPart& rInitialModelPart, LinearSolverPointerType pEigenSolver, Parameters Settings) :
PerturbGeometryBaseUtility(rInitialModelPart, Settings){
mpEigenSolver = pEigenSolver;
}

~PerturbGeometrySparseUtility() override
= default;



int CreateRandomFieldVectors() override;

std::string Info() const override
{
return "PerturbGeometrySparseUtility";
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "PerturbGeometrySparseUtility";
}

void PrintData(std::ostream& rOStream) const override
{
}


private:

LinearSolverPointerType mpEigenSolver;

PerturbGeometrySparseUtility& operator=(PerturbGeometrySparseUtility const& rOther) = delete;

PerturbGeometrySparseUtility(PerturbGeometrySparseUtility const& rOther) = delete;


}; 


}