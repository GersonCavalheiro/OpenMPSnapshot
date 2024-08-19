
#pragma once



#include "geometries/geometry.h"
#include "geometries/nurbs_curve_geometry.h"
#include "geometries/nurbs_shape_function_utilities/nurbs_interval.h"


namespace Kratos
{


template<class TContainerPointType, class TContainerPointEmbeddedType = TContainerPointType>
class BrepCurve
: public Geometry<typename TContainerPointType::value_type>
{
public:

KRATOS_CLASS_POINTER_DEFINITION( BrepCurve );

typedef typename TContainerPointType::value_type PointType;

typedef Geometry<typename TContainerPointType::value_type> BaseType;
typedef Geometry<typename TContainerPointType::value_type> GeometryType;
typedef typename GeometryType::Pointer GeometryPointer;

typedef GeometryData::IntegrationMethod IntegrationMethod;

typedef NurbsCurveGeometry<3, TContainerPointType> NurbsCurveType;

typedef typename BaseType::GeometriesArrayType GeometriesArrayType;

typedef typename BaseType::IndexType IndexType;
typedef typename BaseType::SizeType SizeType;

typedef typename BaseType::PointsArrayType PointsArrayType;
typedef typename BaseType::CoordinatesArrayType CoordinatesArrayType;
typedef typename BaseType::IntegrationPointsArrayType IntegrationPointsArrayType;


BrepCurve(
typename NurbsCurveType::Pointer pCurve)
: BaseType(PointsArrayType(), &msGeometryData)
, mpNurbsCurve(pCurve)
{
mIsTrimmed = false;
}

explicit BrepCurve(const PointsArrayType& ThisPoints)
: BaseType(ThisPoints, &msGeometryData)
{
}

BrepCurve(BrepCurve const& rOther)
: BaseType(rOther)
, mpNurbsCurve(rOther.mpNurbsCurve)
, mIsTrimmed(rOther.mIsTrimmed)
{
}

template<class TOtherContainerPointType, class TOtherContainerPointEmbeddedType>
explicit BrepCurve(
BrepCurve<TOtherContainerPointType, TOtherContainerPointEmbeddedType> const& rOther )
: BaseType(rOther)
, mpNurbsCurve(rOther.mpNurbsCurve)
, mIsTrimmed(rOther.mIsTrimmed)
{
}

~BrepCurve() override = default;


BrepCurve& operator=( const BrepCurve& rOther )
{
BaseType::operator=( rOther );
mpNurbsCurve = rOther.mpNurbsCurve;
mIsTrimmed = rOther.mIsTrimmed;
return *this;
}

template<class TOtherContainerPointType, class TOtherContainerPointEmbeddedType>
BrepCurve& operator=( BrepCurve<TOtherContainerPointType, TOtherContainerPointEmbeddedType> const & rOther )
{
BaseType::operator=( rOther );
mpNurbsCurve = rOther.mpNurbsCurve;
mIsTrimmed = rOther.mIsTrimmed;
return *this;
}


typename BaseType::Pointer Create(PointsArrayType const& ThisPoints) const override
{
return typename BaseType::Pointer(new BrepCurve(ThisPoints));
}



GeometryPointer pGetGeometryPart(const IndexType Index) override
{
const auto& const_this = *this;
return std::const_pointer_cast<GeometryType>(
const_this.pGetGeometryPart(Index));
}


const GeometryPointer pGetGeometryPart(const IndexType Index) const override
{
if (Index == GeometryType::BACKGROUND_GEOMETRY_INDEX)
return mpNurbsCurve;

KRATOS_ERROR << "Index " << Index << " not existing in BrepCurve: "
<< this->Id() << std::endl;
}


bool HasGeometryPart(const IndexType Index) const override
{
if (Index == GeometryType::BACKGROUND_GEOMETRY_INDEX)
return true;

return false;
}


void Calculate(
const Variable<array_1d<double, 3>>& rVariable,
array_1d<double, 3>& rOutput) const override
{
if (rVariable == CHARACTERISTIC_GEOMETRY_LENGTH)
{
mpNurbsCurve->Calculate(rVariable, rOutput);
}
}


SizeType PolynomialDegree(IndexType LocalDirectionIndex) const override
{
return mpNurbsCurve->PolynomialDegree(LocalDirectionIndex);
}



bool IsTrimmed() const
{
return mIsTrimmed;
}

SizeType PointsNumberInDirection(IndexType DirectionIndex) const override
{
return mpNurbsCurve->PointsNumberInDirection(DirectionIndex);
}


Point Center() const override
{
return mpNurbsCurve->Center();
}


int ProjectionPointGlobalToLocalSpace(
const CoordinatesArrayType& rPointGlobalCoordinates,
CoordinatesArrayType& rProjectedPointLocalCoordinates,
const double Tolerance = std::numeric_limits<double>::epsilon()
) const override
{
return mpNurbsCurve->ProjectionPointGlobalToLocalSpace(
rPointGlobalCoordinates, rProjectedPointLocalCoordinates, Tolerance);
}


CoordinatesArrayType& GlobalCoordinates(
CoordinatesArrayType& rResult,
const CoordinatesArrayType& rLocalCoordinates
) const override
{
mpNurbsCurve->GlobalCoordinates(rResult, rLocalCoordinates);

return rResult;
}


IntegrationInfo GetDefaultIntegrationInfo() const override
{
return mpNurbsCurve->GetDefaultIntegrationInfo();
}



void CreateIntegrationPoints(
IntegrationPointsArrayType& rIntegrationPoints,
IntegrationInfo& rIntegrationInfo) const override
{
std::vector<double> spans;
mpNurbsCurve->SpansLocalSpace(spans);

IntegrationPointUtilities::CreateIntegrationPoints1D(
rIntegrationPoints, spans, rIntegrationInfo);
}



void CreateQuadraturePointGeometries(
GeometriesArrayType& rResultGeometries,
IndexType NumberOfShapeFunctionDerivatives,
const IntegrationPointsArrayType& rIntegrationPoints,
IntegrationInfo& rIntegrationInfo) override
{
mpNurbsCurve->CreateQuadraturePointGeometries(
rResultGeometries, NumberOfShapeFunctionDerivatives, rIntegrationPoints, rIntegrationInfo);

for (IndexType i = 0; i < rResultGeometries.size(); ++i) {
rResultGeometries(i)->SetGeometryParent(this);
}
}


Vector& ShapeFunctionsValues(
Vector &rResult,
const CoordinatesArrayType& rCoordinates) const override
{
mpNurbsCurve->ShapeFunctionsValues(rResult, rCoordinates);

return rResult;
}

Matrix& ShapeFunctionsLocalGradients(
Matrix& rResult,
const CoordinatesArrayType& rCoordinates) const override
{
mpNurbsCurve->ShapeFunctionsLocalGradients(rResult, rCoordinates);

return rResult;
}


GeometryData::KratosGeometryFamily GetGeometryFamily() const override
{
return GeometryData::KratosGeometryFamily::Kratos_Brep;
}

GeometryData::KratosGeometryType GetGeometryType() const override
{
return GeometryData::KratosGeometryType::Kratos_Brep_Curve;
}


std::string Info() const override
{
return "BrepCurve";
}

void PrintInfo( std::ostream& rOStream ) const override
{
rOStream << "BrepCurve";
}

void PrintData( std::ostream& rOStream ) const override
{
BaseType::PrintData( rOStream );
std::cout << std::endl;
rOStream << "    BrepCurve " << std::endl;
}


private:

static const GeometryData msGeometryData;

static const GeometryDimension msGeometryDimension;


typename NurbsCurveType::Pointer mpNurbsCurve;


bool mIsTrimmed;


friend class Serializer;

void save( Serializer& rSerializer ) const override
{
KRATOS_SERIALIZE_SAVE_BASE_CLASS( rSerializer, BaseType );
rSerializer.save("NurbsCurve", mpNurbsCurve);
rSerializer.save("IsTrimmed", mIsTrimmed);
}

void load( Serializer& rSerializer ) override
{
KRATOS_SERIALIZE_LOAD_BASE_CLASS( rSerializer, BaseType );
rSerializer.load("NurbsCurve", mpNurbsCurve);
rSerializer.load("IsTrimmed", mIsTrimmed);
}

BrepCurve()
: BaseType( PointsArrayType(), &msGeometryData )
{}


}; 


template<class TContainerPointType, class TContainerPointEmbeddedType = TContainerPointType> inline std::istream& operator >> (
std::istream& rIStream,
BrepCurve<TContainerPointType, TContainerPointEmbeddedType>& rThis );

template<class TContainerPointType, class TContainerPointEmbeddedType = TContainerPointType> inline std::ostream& operator << (
std::ostream& rOStream,
const BrepCurve<TContainerPointType, TContainerPointEmbeddedType>& rThis )
{
rThis.PrintInfo( rOStream );
rOStream << std::endl;
rThis.PrintData( rOStream );
return rOStream;
}


template<class TContainerPointType, class TContainerPointEmbeddedType> const
GeometryData BrepCurve<TContainerPointType, TContainerPointEmbeddedType>::msGeometryData(
&msGeometryDimension,
GeometryData::IntegrationMethod::GI_GAUSS_1,
{}, {}, {});

template<class TContainerPointType, class TContainerPointEmbeddedType>
const GeometryDimension BrepCurve<TContainerPointType, TContainerPointEmbeddedType>::msGeometryDimension(3, 1);

}
