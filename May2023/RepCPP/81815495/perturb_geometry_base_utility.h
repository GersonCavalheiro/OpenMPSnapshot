
#pragma once



#include "includes/model_part.h"
#include "spaces/ublas_space.h"


namespace Kratos {



class KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) PerturbGeometryBaseUtility
{
public:


typedef TUblasSparseSpace<double> TSparseSpaceType;
typedef TUblasDenseSpace<double> TDenseSpaceType;

typedef TDenseSpaceType::MatrixPointerType DenseMatrixPointerType;

typedef TDenseSpaceType::VectorType DenseVectorType;

typedef TDenseSpaceType::MatrixType DenseMatrixType;

KRATOS_CLASS_POINTER_DEFINITION(PerturbGeometryBaseUtility);


PerturbGeometryBaseUtility( ModelPart& rInitialModelPart, Parameters Settings);

virtual ~PerturbGeometryBaseUtility() {}


virtual int CreateRandomFieldVectors() = 0;


void ApplyRandomFieldVectorsToGeometry(ModelPart& rThisModelPart, const std::vector<double>& variables );


virtual std::string Info() const
{
return "PerturbGeometryBaseUtility";
}

virtual void PrintInfo(std::ostream& rOStream) const
{
rOStream << "PerturbGeometryBaseUtility";
}

virtual void PrintData(std::ostream& rOStream) const
{
}


protected:


DenseMatrixPointerType mpPerturbationMatrix;

ModelPart& mrInitialModelPart;

double mCorrelationLength;

double mTruncationError;

int mEchoLevel;



double CorrelationFunction( ModelPart::NodeType& itNode1, ModelPart::NodeType& itNode2, double CorrelationLenth);


private:


double mMaximalDisplacement;

PerturbGeometryBaseUtility& operator=(PerturbGeometryBaseUtility const& rOther) = delete;

PerturbGeometryBaseUtility(PerturbGeometryBaseUtility const& rOther) = delete;


}; 


}