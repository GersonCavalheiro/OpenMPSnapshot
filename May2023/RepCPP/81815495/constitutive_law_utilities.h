
#pragma once




#include "includes/ublas_interface.h"
#include "includes/node.h"
#include "geometries/geometry.h"
#include "includes/constitutive_law.h"

namespace Kratos
{


typedef std::size_t SizeType;




template <SizeType TVoigtSize = 6>
class KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) ConstitutiveLawUtilities
{
public:

typedef std::size_t IndexType;

static constexpr SizeType Dimension = TVoigtSize == 6 ? 3 : 2;

static constexpr SizeType VoigtSize = TVoigtSize;

typedef Matrix MatrixType;

typedef Vector VectorType;

typedef array_1d<double, VoigtSize> BoundedVectorType;

typedef BoundedMatrix<double, Dimension, Dimension> BoundedMatrixType;

typedef BoundedMatrix<double, VoigtSize, VoigtSize> BoundedMatrixVoigtType;

typedef Node NodeType;

typedef Geometry<NodeType> GeometryType;

static constexpr double tolerance = std::numeric_limits<double>::epsilon();





static void CalculateGreenLagrangianStrain(
const MatrixType& rCauchyTensor,
VectorType& rStrainVector
);



static void PolarDecomposition(
const MatrixType& rFDeformationGradient,
MatrixType& rRMatrix,
MatrixType& rUMatrix
);


static void CalculateProjectionOperator(
const Vector& rStrainVector,
MatrixType& rProjectionOperator
);


template<class TVector>
static void CalculateI1Invariant(
const TVector& rStressVector,
double& rI1
)
{
rI1 = rStressVector[0];
for (IndexType i = 1; i < Dimension; ++i)
rI1 += rStressVector[i];
}


template<class TVector>
static void CalculateJ2Invariant(
const TVector& rStressVector,
const double I1,
BoundedVectorType& rDeviator,
double& rJ2
)
{
if constexpr (Dimension == 3) {
rDeviator = rStressVector;
const double p_mean = I1 / 3.0;
for (IndexType i = 0; i < Dimension; ++i)
rDeviator[i] -= p_mean;
rJ2 = 0.0;
for (IndexType i = 0; i < Dimension; ++i)
rJ2 += 0.5 * std::pow(rDeviator[i], 2);
for (IndexType i = Dimension; i < 6; ++i)
rJ2 += std::pow(rDeviator[i], 2);
} else {
rDeviator = rStressVector;
const double p_mean = I1 / 3.0;
for (IndexType i = 0; i < Dimension; ++i)
rDeviator[i] -= p_mean;
rJ2 = 0.5 * (std::pow(rDeviator[0], 2.0) + std::pow(rDeviator[1], 2.0) + std::pow(p_mean, 2.0)) + std::pow(rDeviator[2], 2.0);
}
}


template<class TVector>
static double CalculateVonMisesEquivalentStress(const TVector& rStressVector)
{
double I1, J2;
array_1d<double, VoigtSize> deviator = ZeroVector(VoigtSize);

ConstitutiveLawUtilities<VoigtSize>::CalculateI1Invariant(rStressVector, I1);
ConstitutiveLawUtilities<VoigtSize>::CalculateJ2Invariant(rStressVector, I1, deviator, J2);

return std::sqrt(3.0 * J2);
}


static void CalculateRotationOperatorVoigt(
const BoundedMatrixType &rOldOperator,
BoundedMatrixVoigtType &rNewOperator
);


template <class TVector>
static void CheckAndNormalizeVector(
TVector& rVector
)
{
const double norm = MathUtils<double>::Norm3(rVector);
if (norm > std::numeric_limits<double>::epsilon()) {
rVector /= norm;
} else {
KRATOS_ERROR << "The norm of one LOCAL_AXIS is null" << std::endl;
}
}


static void CalculateElasticMatrixPlaneStress(MatrixType &C, ConstitutiveLaw::Parameters &rValues);
static void CalculateElasticMatrixPlaneStrain(MatrixType &C, ConstitutiveLaw::Parameters &rValues);
static void CalculateElasticMatrix(MatrixType &C, ConstitutiveLaw::Parameters &rValues);

private:

}; 
} 
