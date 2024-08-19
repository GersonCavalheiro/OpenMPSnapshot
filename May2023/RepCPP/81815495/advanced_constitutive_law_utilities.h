
#pragma once



#include "includes/ublas_interface.h"
#include "includes/node.h"
#include "geometries/geometry.h"

namespace Kratos
{


typedef std::size_t SizeType;




template <SizeType TVoigtSize = 6>
class KRATOS_API(CONSTITUTIVE_LAWS_APPLICATION) AdvancedConstitutiveLawUtilities
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


static void CalculateI2Invariant(
const BoundedVectorType& rStressVector,
double& rI2
);


static void CalculateI3Invariant(
const BoundedVectorType& rStressVector,
double& rI3
);


template<class TVector>
static void CalculateJ2Invariant(
const TVector& rStressVector,
const double I1,
BoundedVectorType& rDeviator,
double& rJ2
)
{
noalias(rDeviator) = rStressVector;
const double p_mean = I1 / 3.0;
if constexpr (Dimension == 3) {
for (IndexType i = 0; i < Dimension; ++i)
rDeviator[i] -= p_mean;
rJ2 = 0.0;
for (IndexType i = 0; i < Dimension; ++i)
rJ2 += 0.5 * std::pow(rDeviator[i], 2);
for (IndexType i = Dimension; i < 6; ++i)
rJ2 += std::pow(rDeviator[i], 2);
} else {
for (IndexType i = 0; i < Dimension; ++i)
rDeviator[i] -= p_mean;
rJ2 = 0.5 * (std::pow(rDeviator[0], 2.0) + std::pow(rDeviator[1], 2.0) + std::pow(p_mean, 2.0)) + std::pow(rDeviator[2], 2.0);
}
}


static void CalculateJ3Invariant(
const BoundedVectorType& rDeviator,
double& rJ3
);


static void CalculateFirstVector(BoundedVectorType& rFirstVector);


static void CalculateSecondVector(
const BoundedVectorType& rDeviator,
const double J2,
BoundedVectorType& rSecondVector
);


static void CalculateThirdVector(
const BoundedVectorType& rDeviator,
const double J2,
BoundedVectorType& rThirdVector
);


static void CalculateLodeAngle(
const double J2,
const double J3,
double& rLodeAngle
);


static double CalculateCharacteristicLength(const GeometryType& rGeometry);


static double CalculateCharacteristicLengthOnReferenceConfiguration(const GeometryType& rGeometry);


static Matrix ComputeEquivalentSmallDeformationDeformationGradient(const Vector& rStrainVector);


static void CalculateAlmansiStrain(
const MatrixType& rLeftCauchyTensor,
VectorType& rStrainVector
);


static void CalculateHenckyStrain(
const MatrixType& rCauchyTensor,
VectorType& rStrainVector
);


static void CalculateBiotStrain(
const MatrixType& rCauchyTensor,
VectorType& rStrainVector
);


static void CalculatePrincipalStresses(
array_1d<double, Dimension>& rPrincipalStressVector,
const BoundedVectorType& rStressVector
);


static void CalculatePrincipalStressesWithCardano(
array_1d<double, Dimension>& rPrincipalStressVector,
const BoundedVectorType& rStressVector
);


static void SpectralDecomposition(
const BoundedVectorType& rStressVector,
BoundedVectorType& rStressVectorTension,
BoundedVectorType& rStressVectorCompression
);


static Matrix CalculateElasticDeformationGradient(
const MatrixType& rF,
const MatrixType& rFp
);


static MatrixType CalculateLinearPlasticDeformationGradientIncrement(
const BoundedVectorType& rPlasticPotentialDerivative,
const double PlasticConsistencyFactorIncrement
);


static Matrix CalculatePlasticDeformationGradientFromElastic(
const MatrixType& rF,
const MatrixType& rFp
);


static MatrixType CalculateExponentialElasticDeformationGradient(
const MatrixType& rTrialFe,
const BoundedVectorType& rPlasticPotentialDerivative,
const double PlasticConsistencyFactorIncrement,
const MatrixType& rRe
);


static void CalculatePlasticStrainFromFp(
const MatrixType& rFp,
Vector& rPlasticStrainVector
);


static MatrixType CalculateDirectElasticDeformationGradient(
const MatrixType& rElasticTrial,
const BoundedVectorType& rPlasticPotentialDerivative,
const double PlasticConsistencyFactorIncrement,
const MatrixType& rRe
);


static MatrixType CalculateExponentialPlasticDeformationGradientIncrement(
const BoundedVectorType& rPlasticPotentialDerivative,
const double PlasticConsistencyFactorIncrement,
const MatrixType& rRe
);


static MatrixType CalculateDirectPlasticDeformationGradientIncrement(
const BoundedVectorType& rPlasticPotentialDerivative,
const double PlasticConsistencyFactorIncrement,
const MatrixType& rRe
);


static void CalculateRotationOperatorEuler1(
const double EulerAngle1,
BoundedMatrix<double, 3, 3>& rRotationOperator
);


static void CalculateRotationOperatorEuler2(
const double EulerAngle2,
BoundedMatrix<double, 3, 3>& rRotationOperator
);


static void CalculateRotationOperatorEuler3(
const double EulerAngle3,
BoundedMatrix<double, 3, 3>& rRotationOperator
);


static void CalculateRotationOperator(
const double EulerAngle1, 
const double EulerAngle2, 
const double EulerAngle3, 
BoundedMatrix<double, 3, 3>& rRotationOperator
);


static double MacaullyBrackets(
const double Number);

}; 
} 