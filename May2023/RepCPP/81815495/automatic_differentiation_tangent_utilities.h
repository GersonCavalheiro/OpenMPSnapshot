
#pragma once



#include "custom_constitutive/auxiliary_files/yield_surfaces/generic_yield_surface.h"


namespace Kratos
{





template <class TYieldSurfaceType, SizeType TSofteningType>
class KRATOS_API(CONSTITUTIVE_LAWS_APPLICATION) AutomaticDifferentiationTangentUtilities
{
public:
typedef std::size_t SizeType;

typedef TYieldSurfaceType YieldSurfaceType;

static constexpr SizeType Dimension =  YieldSurfaceType::Dimension;

static constexpr SizeType VoigtSize = YieldSurfaceType::VoigtSize;

typedef Matrix MatrixType;

typedef Vector VectorType;

typedef array_1d<double, VoigtSize> BoundedVectorType;

typedef BoundedMatrix<double, Dimension, Dimension> BoundedMatrixType;

typedef BoundedMatrix<double, VoigtSize, VoigtSize> BoundedMatrixVoigtType;

typedef Node NodeType;

typedef Geometry<NodeType> GeometryType;






static void CalculateTangentTensorIsotropicDamage(ConstitutiveLaw::Parameters rValues);

};
}