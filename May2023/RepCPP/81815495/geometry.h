
#pragma once

#include <typeinfo>


#include "geometries/geometry_data.h"
#include "geometries/point.h"
#include "containers/pointer_vector.h"
#include "containers/data_value_container.h"
#include "utilities/math_utils.h"
#include "input_output/logger.h"
#include "integration/integration_info.h"

namespace Kratos
{


template<class TPointType>
class Geometry
{
public:

typedef Geometry<TPointType> GeometryType;

KRATOS_CLASS_POINTER_DEFINITION( Geometry );


enum class QualityCriteria {
INRADIUS_TO_CIRCUMRADIUS,
AREA_TO_LENGTH,
SHORTEST_ALTITUDE_TO_LENGTH,
INRADIUS_TO_LONGEST_EDGE,
SHORTEST_TO_LONGEST_EDGE,
REGULARITY,
VOLUME_TO_SURFACE_AREA,
VOLUME_TO_EDGE_LENGTH,
VOLUME_TO_AVERAGE_EDGE_LENGTH,
VOLUME_TO_RMS_EDGE_LENGTH,
MIN_DIHEDRAL_ANGLE,
MAX_DIHEDRAL_ANGLE,
MIN_SOLID_ANGLE
};


enum class LumpingMethods {
ROW_SUM,
DIAGONAL_SCALING,
QUADRATURE_ON_NODES
};


typedef PointerVector<TPointType> PointsArrayType;


typedef GeometryData::IntegrationMethod IntegrationMethod;


typedef PointerVector<GeometryType> GeometriesArrayType;


typedef TPointType PointType;


typedef std::size_t IndexType;



typedef std::size_t SizeType;


typedef typename PointType::CoordinatesArrayType CoordinatesArrayType;



typedef IntegrationPoint<3> IntegrationPointType;


typedef std::vector<IntegrationPointType> IntegrationPointsArrayType;


typedef std::array<IntegrationPointsArrayType, static_cast<int>(GeometryData::IntegrationMethod::NumberOfIntegrationMethods)> IntegrationPointsContainerType;


typedef std::array<Matrix, static_cast<int>(GeometryData::IntegrationMethod::NumberOfIntegrationMethods)> ShapeFunctionsValuesContainerType;


typedef GeometryData::ShapeFunctionsLocalGradientsContainerType ShapeFunctionsLocalGradientsContainerType;


typedef DenseVector<Matrix > JacobiansType;


typedef GeometryData::ShapeFunctionsGradientsType ShapeFunctionsGradientsType;


typedef GeometryData::ShapeFunctionsSecondDerivativesType ShapeFunctionsSecondDerivativesType;


typedef GeometryData::ShapeFunctionsThirdDerivativesType ShapeFunctionsThirdDerivativesType;


typedef DenseVector<double> NormalType;

typedef typename PointType::Pointer PointPointerType;
typedef const PointPointerType ConstPointPointerType;
typedef TPointType& PointReferenceType;
typedef const TPointType& ConstPointReferenceType;
typedef std::vector<PointPointerType> PointPointerContainerType;

typedef typename PointsArrayType::iterator iterator;
typedef typename PointsArrayType::const_iterator const_iterator;

typedef typename PointsArrayType::ptr_iterator ptr_iterator;
typedef typename PointsArrayType::ptr_const_iterator ptr_const_iterator;
typedef typename PointsArrayType::difference_type difference_type;

static constexpr IndexType BACKGROUND_GEOMETRY_INDEX = std::numeric_limits<IndexType>::max();


Geometry()
: mId(GenerateSelfAssignedId())
, mpGeometryData(&GeometryDataInstance())
{
}

Geometry(IndexType GeomertyId)
: mpGeometryData(&GeometryDataInstance())
{
SetId(GeomertyId);
}

Geometry(const std::string& GeometryName)
: mId(GenerateId(GeometryName))
, mpGeometryData(&GeometryDataInstance())
{
}


Geometry(
const PointsArrayType &ThisPoints,
GeometryData const *pThisGeometryData = &GeometryDataInstance())
: mId(GenerateSelfAssignedId())
, mpGeometryData(pThisGeometryData)
, mPoints(ThisPoints)
{
}

Geometry(
IndexType GeometryId,
const PointsArrayType& ThisPoints,
GeometryData const* pThisGeometryData = &GeometryDataInstance())
: mpGeometryData(pThisGeometryData)
, mPoints(ThisPoints)
{
SetId(GeometryId);
}

Geometry(
const std::string& GeometryName,
const PointsArrayType& ThisPoints,
GeometryData const* pThisGeometryData = &GeometryDataInstance())
: mId(GenerateId(GeometryName))
, mpGeometryData(pThisGeometryData)
, mPoints(ThisPoints)
{
}


Geometry( const Geometry& rOther )
: mId(rOther.mId),
mpGeometryData(rOther.mpGeometryData),
mPoints(rOther.mPoints),
mData(rOther.mData)
{
}


template<class TOtherPointType>
Geometry( Geometry<TOtherPointType> const & rOther )
: mId(rOther.mId),
mpGeometryData(rOther.mpGeometryData),
mData(rOther.mData)
{
mPoints = new PointsArrayType(rOther.begin(), rOther.end());
}

virtual ~Geometry() {}

virtual GeometryData::KratosGeometryFamily GetGeometryFamily() const
{
return GeometryData::KratosGeometryFamily::Kratos_generic_family;
}

virtual GeometryData::KratosGeometryType GetGeometryType() const
{
return GeometryData::KratosGeometryType::Kratos_generic_type;
}



Geometry& operator=( const Geometry& rOther )
{
mpGeometryData = rOther.mpGeometryData;
mPoints = rOther.mPoints;
mData = rOther.mData;

return *this;
}


template<class TOtherPointType>
Geometry& operator=( Geometry<TOtherPointType> const & rOther )
{
this->clear();

for ( typename Geometry<TOtherPointType>::ptr_const_iterator i = rOther.ptr_begin() ; i != rOther.ptr_end() ; ++i )
push_back( typename PointType::Pointer( new PointType( **i ) ) );

mpGeometryData = rOther.mpGeometryData;

return *this;
}

operator PointsArrayType&()
{
return mPoints;
}


TPointType& operator[](const SizeType& i)
{
return mPoints[i];
}

TPointType const& operator[](const SizeType& i) const
{
return mPoints[i];
}

PointPointerType& operator()(const SizeType& i)
{
return mPoints(i);
}

ConstPointPointerType& operator()(const SizeType& i) const
{
return mPoints(i);
}


iterator                   begin()
{
return iterator(mPoints.begin());
}
const_iterator             begin() const
{
return const_iterator(mPoints.begin());
}
iterator                   end()
{
return iterator(mPoints.end());
}
const_iterator             end() const
{
return const_iterator(mPoints.end());
}
ptr_iterator               ptr_begin()
{
return mPoints.ptr_begin();
}
ptr_const_iterator         ptr_begin() const
{
return mPoints.ptr_begin();
}
ptr_iterator               ptr_end()
{
return mPoints.ptr_end();
}
ptr_const_iterator         ptr_end() const
{
return mPoints.ptr_end();
}
PointReferenceType        front()       
{
assert(!empty());
return mPoints.front();
}
ConstPointReferenceType  front() const 
{
assert(!empty());
return mPoints.front();
}
PointReferenceType        back()        
{
assert(!empty());
return mPoints.back();
}
ConstPointReferenceType  back() const  
{
assert(!empty());
return mPoints.back();
}

SizeType size() const
{
return mPoints.size();
}


SizeType PointsNumber() const {
return this->size();
}

virtual SizeType PointsNumberInDirection(IndexType LocalDirectionIndex) const
{
KRATOS_ERROR << "Trying to access PointsNumberInDirection from geometry base class." << std::endl;
}

SizeType max_size() const
{
return mPoints.max_size();
}

void swap(GeometryType& rOther)
{
mPoints.swap(rOther.mPoints);
}

void push_back(PointPointerType x)
{
mPoints.push_back(x);
}

void clear()
{
mPoints.clear();
}

void reserve(int dim)
{
mPoints.reserve(dim);
}

int capacity()
{
return mPoints.capacity();
}


PointPointerContainerType& GetContainer()
{
return mPoints.GetContainer();
}


const PointPointerContainerType& GetContainer() const
{
return mPoints.GetContainer();
}



DataValueContainer& GetData()
{
return mData;
}

DataValueContainer const& GetData() const
{
return mData;
}

void SetData(DataValueContainer const& rThisData)
{
mData = rThisData;
}


template<class TDataType> bool Has(const Variable<TDataType>& rThisVariable) const
{
return mData.Has(rThisVariable);
}


template<class TVariableType> void SetValue(
const TVariableType& rThisVariable,
typename TVariableType::Type const& rValue)
{
mData.SetValue(rThisVariable, rValue);
}


template<class TVariableType> typename TVariableType::Type& GetValue(
const TVariableType& rThisVariable)
{
return mData.GetValue(rThisVariable);
}

template<class TVariableType> typename TVariableType::Type const& GetValue(
const TVariableType& rThisVariable) const
{
return mData.GetValue(rThisVariable);
}




virtual void Assign(
const Variable<bool>& rVariable,
const bool Input) {}

virtual void Assign(
const Variable<int>& rVariable,
const int Input) {}

virtual void Assign(
const Variable<double>& rVariable,
const double Input) {}

virtual void Assign(
const Variable<array_1d<double, 2>>& rVariable,
const array_1d<double, 2>& rInput) {}

virtual void Assign(
const Variable<array_1d<double, 3>>& rVariable,
const array_1d<double, 3>& rInput) {}

virtual void Assign(
const Variable<array_1d<double, 6>>& rVariable,
const array_1d<double, 6>& rInput) {}

virtual void Assign(
const Variable<Vector>& rVariable,
const Vector& rInput) {}

virtual void Assign(
const Variable<Matrix>& rVariable,
const Matrix& rInput) {}



virtual void Calculate(
const Variable<bool>& rVariable,
bool& rOutput) const {}

virtual void Calculate(
const Variable<int>& rVariable,
int& rOutput) const {}

virtual void Calculate(
const Variable<double>& rVariable,
double& rOutput) const {}

virtual void Calculate(
const Variable<array_1d<double, 2>>& rVariable,
array_1d<double, 2>& rOutput) const {}

virtual void Calculate(
const Variable<array_1d<double, 3>>& rVariable,
array_1d<double, 3>& rOutput) const {}

virtual void Calculate(
const Variable<array_1d<double, 6>>& rVariable,
array_1d<double, 6>& rOutput) const {}

virtual void Calculate(
const Variable<Vector>& rVariable,
Vector& rOutput) const {}

virtual void Calculate(
const Variable<Matrix>& rVariable,
Matrix& rOutput) const {}



inline static bool HasSameType(
const GeometryType& rLHS,
const GeometryType& rRHS)
{
return (typeid(rLHS) == typeid(rRHS));
}


inline static bool HasSameType(
const GeometryType * rLHS,
const GeometryType* rRHS)
{
return GeometryType::HasSameType(*rLHS, *rRHS);
}


inline static bool HasSameGeometryType(const GeometryType& rLHS, const GeometryType& rRHS) {
return (rLHS.GetGeometryType() == rRHS.GetGeometryType());
}


inline static bool HasSameGeometryType(
const GeometryType* rLHS,
const GeometryType* rRHS)
{
return GeometryType::HasSameGeometryType(*rLHS, *rRHS);
}


inline static bool IsSame(
const GeometryType& rLHS,
const GeometryType& rRHS)
{
return GeometryType::HasSameType(rLHS, rRHS) && GeometryType::HasSameGeometryType(rLHS, rRHS);
}


inline static bool IsSame(
const GeometryType* rLHS,
const GeometryType* rRHS)
{
return GeometryType::HasSameType(*rLHS, *rRHS) && GeometryType::HasSameGeometryType(*rLHS, *rRHS);
}

bool empty() const
{
return mPoints.empty();
}



virtual Pointer Create(
PointsArrayType const& rThisPoints
) const
{
auto p_geom = this->Create(0, rThisPoints);

IndexType id = reinterpret_cast<IndexType>(p_geom.get());

p_geom->SetIdSelfAssigned(id);

p_geom->SetIdNotGeneratedFromString(id);

p_geom->SetIdWithoutCheck(id);

return p_geom;
}


virtual Pointer Create(
const IndexType NewGeometryId,
PointsArrayType const& rThisPoints
) const
{
return Pointer( new Geometry( NewGeometryId, rThisPoints, mpGeometryData));
}


Pointer Create(
const std::string& rNewGeometryName,
PointsArrayType const& rThisPoints
) const
{
auto p_geom = this->Create(0, rThisPoints);
p_geom->SetId(rNewGeometryName);
return p_geom;
}


virtual Pointer Create(
const GeometryType& rGeometry
) const
{
auto p_geom = this->Create(0, rGeometry);

IndexType id = reinterpret_cast<IndexType>(p_geom.get());

p_geom->SetIdSelfAssigned(id);

p_geom->SetIdNotGeneratedFromString(id);

p_geom->SetIdWithoutCheck(id);

return p_geom;
}


virtual Pointer Create(
const IndexType NewGeometryId,
const GeometryType& rGeometry
) const
{
auto p_geometry = Pointer( new Geometry( NewGeometryId, rGeometry.Points(), mpGeometryData));
p_geometry->SetData(rGeometry.GetData());
return p_geometry;
}


Pointer Create(
const std::string& rNewGeometryName,
const GeometryType& rGeometry
) const
{
auto p_geom = this->Create(0, rGeometry);
p_geom->SetId(rNewGeometryName);
return p_geom;
}


void ClonePoints()
{
for ( ptr_iterator i = this->ptr_begin() ; i != this->ptr_end() ; i++ )
*i = typename PointType::Pointer( new PointType( **i ) );
}



GeometryData const& GetGeometryData() const
{
return *mpGeometryData;
}


virtual void SetGeometryShapeFunctionContainer(
const GeometryShapeFunctionContainer<GeometryData::IntegrationMethod>&  rGeometryShapeFunctionContainer)
{
KRATOS_ERROR <<
"Calling SetGeometryShapeFunctionContainer from base geometry class."
<< std::endl;
}


IndexType const& Id() const
{
return mId;
}

bool IsIdGeneratedFromString()
{
return IsIdGeneratedFromString(mId);
}

bool IsIdSelfAssigned()
{
return IsIdSelfAssigned(mId);
}

void SetId(const IndexType Id)
{
KRATOS_ERROR_IF(IsIdGeneratedFromString(Id)
|| IsIdSelfAssigned(Id))
<< "Id: " << Id << " out of range. The Id must me lower than 2^62 = 4.61e+18. "
<< "Geometry being recognized as generated from string: " << IsIdGeneratedFromString(Id)
<< ", self assigned: " << IsIdSelfAssigned(Id) << "."
<< std::endl;

mId = Id;
}

void SetId(const std::string& rName)
{
mId = GenerateId(rName);
}

static inline IndexType GenerateId(const std::string& rName)
{
std::hash<std::string> string_hash_generator;
auto id = string_hash_generator(rName);

SetIdGeneratedFromString(id);

SetIdNotSelfAssigned(id);

return id;
}



virtual GeometryType& GetGeometryParent(IndexType Index) const
{
KRATOS_ERROR <<
"Calling GetGeometryParent from base geometry class."
<< std::endl;
}


virtual void SetGeometryParent(GeometryType* pGeometryParent)
{
KRATOS_ERROR <<
"Calling SetGeometryParent from base geometry class."
<< std::endl;
}



virtual GeometryType& GetGeometryPart(const IndexType Index)
{
return *pGetGeometryPart(Index);
}


virtual const GeometryType& GetGeometryPart(const IndexType Index) const
{
return *pGetGeometryPart(Index);
}


virtual typename GeometryType::Pointer pGetGeometryPart(const IndexType Index)
{
KRATOS_ERROR << "Calling base class 'pGetGeometryPart' method instead of derived function."
<< " Please check the definition in the derived class. " << *this << std::endl;
}


virtual const typename GeometryType::Pointer pGetGeometryPart(const IndexType Index) const
{
KRATOS_ERROR << "Calling base class 'pGetGeometryPart' method instead of derived function."
<< " Please check the definition in the derived class. " << *this << std::endl;
}


virtual void SetGeometryPart(
const IndexType Index,
GeometryType::Pointer pGeometry
)
{
KRATOS_ERROR << "Calling base class 'SetGeometryPart' method instead of derived function."
<< " Please check the definition in the derived class. " << *this << std::endl;
}


virtual IndexType AddGeometryPart(GeometryType::Pointer pGeometry)
{
KRATOS_ERROR << "Calling base class 'AddGeometryPart' method instead of derived function."
<< " Please check the definition in the derived class. " << *this << std::endl;
}


virtual void RemoveGeometryPart(GeometryType::Pointer pGeometry)
{
KRATOS_ERROR << "Calling base class 'RemoveGeometryPart' method instead of derived function."
<< " Please check the definition in the derived class. " << *this << std::endl;
}


virtual void RemoveGeometryPart(const IndexType Index)
{
KRATOS_ERROR << "Calling base class 'RemoveGeometryPart' method instead of derived function."
<< " Please check the definition in the derived class. " << *this << std::endl;
}


virtual bool HasGeometryPart(const IndexType Index) const
{
KRATOS_ERROR << "Calling base class 'HasGeometryPart' method instead of derived function."
<< " Please check the definition in the derived class. " << *this << std::endl;
}


virtual SizeType NumberOfGeometryParts() const
{
return 0;
}



virtual Vector& LumpingFactors(
Vector& rResult,
const LumpingMethods LumpingMethod = LumpingMethods::ROW_SUM
)  const
{
const SizeType number_of_nodes = this->size();
const SizeType local_space_dimension = this->LocalSpaceDimension();

if (rResult.size() != number_of_nodes)
rResult.resize(number_of_nodes, false);
noalias(rResult) = ZeroVector(number_of_nodes);

if (LumpingMethod == LumpingMethods::ROW_SUM) {
const IntegrationMethod integration_method = GetDefaultIntegrationMethod();
const GeometryType::IntegrationPointsArrayType& r_integrations_points = this->IntegrationPoints( integration_method );
const Matrix& r_Ncontainer = this->ShapeFunctionsValues(integration_method);

Vector detJ_vector(r_integrations_points.size());
DeterminantOfJacobian(detJ_vector, integration_method);

double domain_size = 0.0;
for ( IndexType point_number = 0; point_number < r_integrations_points.size(); ++point_number ) {
const double integration_weight = r_integrations_points[point_number].Weight() * detJ_vector[point_number];
const Vector& rN = row(r_Ncontainer,point_number);

domain_size += integration_weight;

for ( IndexType i = 0; i < number_of_nodes; ++i ) {
rResult[i] += rN[i] * integration_weight;
}
}

for ( IndexType i = 0; i < number_of_nodes; ++i ) {
rResult[i] /= domain_size;
}
} else if (LumpingMethod == LumpingMethods::DIAGONAL_SCALING) {
IntegrationMethod integration_method = GetDefaultIntegrationMethod();
int j = std::min(static_cast<int>(integration_method) + 1, 4);
integration_method = static_cast<IntegrationMethod>(j);
const GeometryType::IntegrationPointsArrayType& r_integrations_points = this->IntegrationPoints( integration_method );
const Matrix& r_Ncontainer = this->ShapeFunctionsValues(integration_method);

Vector detJ_vector(r_integrations_points.size());
DeterminantOfJacobian(detJ_vector, integration_method);

for ( IndexType point_number = 0; point_number < r_integrations_points.size(); ++point_number ) {
const double detJ = detJ_vector[point_number];
const double integration_weight = r_integrations_points[point_number].Weight() * detJ;
const Vector& rN = row(r_Ncontainer,point_number);

for ( IndexType i = 0; i < number_of_nodes; ++i ) {
rResult[i] += std::pow(rN[i], 2) * integration_weight;
}
}

double total_value = 0.0;
for ( IndexType i = 0; i < number_of_nodes; ++i ) {
total_value += rResult[i];
}
for ( IndexType i = 0; i < number_of_nodes; ++i ) {
rResult[i] /= total_value;
}
} else if (LumpingMethod == LumpingMethods::QUADRATURE_ON_NODES) {
const double domain_size = DomainSize();

Matrix local_coordinates(number_of_nodes, local_space_dimension);
PointsLocalCoordinates(local_coordinates);
Point local_point(ZeroVector(3));
array_1d<double, 3>& r_local_coordinates = local_point.Coordinates();

const GeometryType::IntegrationPointsArrayType& r_integrations_points = this->IntegrationPoints( GeometryData::IntegrationMethod::GI_GAUSS_1 ); 
const double weight = r_integrations_points[0].Weight()/static_cast<double>(number_of_nodes);
for ( IndexType point_number = 0; point_number < number_of_nodes; ++point_number ) {
for ( IndexType dim = 0; dim < local_space_dimension; ++dim ) {
r_local_coordinates[dim] = local_coordinates(point_number, dim);
}
const double detJ = DeterminantOfJacobian(local_point);
rResult[point_number] = weight * detJ/domain_size;
}
}

return rResult;
}



inline SizeType WorkingSpaceDimension() const
{
return mpGeometryData->WorkingSpaceDimension();
}


inline SizeType LocalSpaceDimension() const
{
return mpGeometryData->LocalSpaceDimension();
}


virtual SizeType PolynomialDegree(IndexType LocalDirectionIndex) const
{
KRATOS_ERROR << "Trying to access PolynomialDegree from geometry base class." << std::endl;
}



virtual double Length() const {
KRATOS_ERROR << "Calling base class 'Length' method instead of derived class one. Please check the definition of derived class. " << *this << std::endl;
return 0.0;
}


virtual double Area() const {
KRATOS_ERROR << "Calling base class 'Area' method instead of derived class one. Please check the definition of derived class. " << *this << std::endl;
return 0.0;
}


virtual double Volume() const {
KRATOS_ERROR << "Calling base class 'Volume' method instead of derived class one. Please check the definition of derived class. " << *this << std::endl;
return 0.0;
}


virtual double DomainSize() const {
const SizeType local_dimension = this->LocalSpaceDimension();
if (local_dimension == 1) { 
return this->Length();
} else if (local_dimension == 2) { 
return this->Area();
} else { 
return this->Volume();
}
return 0.0;
}


virtual double MinEdgeLength() const {
KRATOS_ERROR << "Calling base class 'MinEdgeLength' method instead of derived class one. Please check the definition of derived class. " << *this << std::endl;
return 0.0;
}


virtual double MaxEdgeLength() const {
KRATOS_ERROR << "Calling base class 'MaxEdgeLength' method instead of derived class one. Please check the definition of derived class. " << *this << std::endl;
return 0.0;
}


virtual double AverageEdgeLength() const {
KRATOS_ERROR << "Calling base class 'AverageEdgeLength' method instead of derived class one. Please check the definition of derived class. " << *this << std::endl;
return 0.0;
}


virtual double Circumradius() const {
KRATOS_ERROR << "Calling base class 'Circumradius' method instead of derived class one. Please check the definition of derived class. " << *this << std::endl;
return 0.0;
}


virtual double Inradius() const {
KRATOS_ERROR << "Calling base class 'Inradius' method instead of derived class one. Please check the definition of derived class. " << *this << std::endl;
return 0.0;
}


virtual bool HasIntersection(const GeometryType& ThisGeometry) const {
KRATOS_ERROR << "Calling base class 'HasIntersection' method instead of derived class one. Please check the definition of derived class. " << *this << std::endl;
return false;
}


virtual bool HasIntersection(const Point& rLowPoint, const Point& rHighPoint) const {
KRATOS_ERROR << "Calling base class 'HasIntersection' method instead of derived class one. Please check the definition of derived class. " << *this << std::endl;
return false;
}



virtual void BoundingBox(
TPointType& rLowPoint,
TPointType& rHighPoint
) const
{
rHighPoint = this->GetPoint( 0 );
rLowPoint  = this->GetPoint( 0 );
const SizeType dim = WorkingSpaceDimension();

for ( IndexType point = 1; point < PointsNumber(); ++point ) { 
const auto& r_point = this->GetPoint( point );
for ( IndexType i = 0; i < dim; ++i ) {
rHighPoint[i] = ( rHighPoint[i] < r_point[i] ) ? r_point[i] : rHighPoint[i];
rLowPoint[i]  = ( rLowPoint[i]  > r_point[i] ) ? r_point[i] : rLowPoint[i];
}
}
}


virtual Point Center() const
{
const SizeType points_number = this->size();

if ( points_number == 0 )
{
KRATOS_ERROR << "can not compute the ceneter of a geometry of zero points" << std::endl;
}

Point result = ( *this )[0];

for ( IndexType i = 1 ; i < points_number ; i++ )
{
result.Coordinates() += ( *this )[i];
}

const double temp = 1.0 / double( points_number );

result.Coordinates() *= temp;

return result;
}


virtual array_1d<double, 3> Normal(const CoordinatesArrayType& rPointLocalCoordinates) const
{
const SizeType local_space_dimension = this->LocalSpaceDimension();
const SizeType dimension = this->WorkingSpaceDimension();

KRATOS_ERROR_IF(dimension == local_space_dimension) << "Remember the normal can be computed just in geometries with a local dimension: "<< this->LocalSpaceDimension() << "smaller than the spatial dimension: " << this->WorkingSpaceDimension() << std::endl;

array_1d<double,3> tangent_xi = ZeroVector(3);
array_1d<double,3> tangent_eta = ZeroVector(3);

Matrix j_node = ZeroMatrix( dimension, local_space_dimension );
this->Jacobian( j_node, rPointLocalCoordinates);

if (dimension == 2) {
tangent_eta[2] = 1.0;
for (unsigned int i_dim = 0; i_dim < dimension; i_dim++) {
tangent_xi[i_dim]  = j_node(i_dim, 0);
}
} else {
for (unsigned int i_dim = 0; i_dim < dimension; i_dim++) {
tangent_xi[i_dim]  = j_node(i_dim, 0);
tangent_eta[i_dim] = j_node(i_dim, 1);
}
}

array_1d<double, 3> normal;
MathUtils<double>::CrossProduct(normal, tangent_xi, tangent_eta);
return normal;
}


virtual array_1d<double, 3> Normal(
IndexType IntegrationPointIndex) const
{
return Normal(IntegrationPointIndex, mpGeometryData->DefaultIntegrationMethod());
}


virtual array_1d<double, 3> Normal(
IndexType IntegrationPointIndex,
IntegrationMethod ThisMethod) const
{
const SizeType local_space_dimension = this->LocalSpaceDimension();
const SizeType dimension = this->WorkingSpaceDimension();

KRATOS_DEBUG_ERROR_IF(dimension == local_space_dimension)
<< "Remember the normal can be computed just in geometries with a local dimension: "
<< this->LocalSpaceDimension() << "smaller than the spatial dimension: "
<< this->WorkingSpaceDimension() << std::endl;

array_1d<double, 3> tangent_xi = ZeroVector(3);
array_1d<double, 3> tangent_eta = ZeroVector(3);

Matrix j_node = ZeroMatrix(dimension, local_space_dimension);
this->Jacobian(j_node, IntegrationPointIndex, ThisMethod);

if (dimension == 2) {
tangent_eta[2] = 1.0;
for (IndexType i_dim = 0; i_dim < dimension; i_dim++) {
tangent_xi[i_dim] = j_node(i_dim, 0);
}
}
else {
for (IndexType i_dim = 0; i_dim < dimension; i_dim++) {
tangent_xi[i_dim] = j_node(i_dim, 0);
tangent_eta[i_dim] = j_node(i_dim, 1);
}
}

array_1d<double, 3> normal;
MathUtils<double>::CrossProduct(normal, tangent_xi, tangent_eta);
return normal;
}


virtual array_1d<double, 3> UnitNormal(
const CoordinatesArrayType& rPointLocalCoordinates) const
{
array_1d<double, 3> normal = Normal(rPointLocalCoordinates);
const double norm_normal = norm_2(normal);
if (norm_normal > std::numeric_limits<double>::epsilon()) normal /= norm_normal;
else KRATOS_ERROR << "ERROR: The normal norm is zero or almost zero. Norm. normal: " << norm_normal << std::endl;
return normal;
}


virtual array_1d<double, 3> UnitNormal(
IndexType IntegrationPointIndex) const
{
return UnitNormal(IntegrationPointIndex, mpGeometryData->DefaultIntegrationMethod());
}


virtual array_1d<double, 3> UnitNormal(
IndexType IntegrationPointIndex,
IntegrationMethod ThisMethod) const
{
array_1d<double, 3> normal_vector = Normal(IntegrationPointIndex, ThisMethod);
const double norm_normal = norm_2(normal_vector);
if (norm_normal > std::numeric_limits<double>::epsilon())
normal_vector /= norm_normal;
else
KRATOS_ERROR
<< "ERROR: The normal norm is zero or almost zero: "
<< norm_normal << std::endl;
return normal_vector;
}



double Quality(const QualityCriteria qualityCriteria) const {
double quality = 0.0f;

if(qualityCriteria == QualityCriteria::INRADIUS_TO_CIRCUMRADIUS) {
quality = InradiusToCircumradiusQuality();
} else if(qualityCriteria == QualityCriteria::AREA_TO_LENGTH) {
quality = AreaToEdgeLengthRatio();
} else if(qualityCriteria == QualityCriteria::SHORTEST_ALTITUDE_TO_LENGTH) {
quality = ShortestAltitudeToEdgeLengthRatio();
} else if(qualityCriteria == QualityCriteria::INRADIUS_TO_LONGEST_EDGE) {
quality = InradiusToLongestEdgeQuality();
} else if(qualityCriteria == QualityCriteria::SHORTEST_TO_LONGEST_EDGE) {
quality = ShortestToLongestEdgeQuality();
} else if(qualityCriteria == QualityCriteria::REGULARITY) {
quality = RegularityQuality();
} else if(qualityCriteria == QualityCriteria::VOLUME_TO_SURFACE_AREA) {
quality = VolumeToSurfaceAreaQuality();
} else if(qualityCriteria == QualityCriteria::VOLUME_TO_EDGE_LENGTH) {
quality = VolumeToEdgeLengthQuality();
} else if(qualityCriteria == QualityCriteria::VOLUME_TO_AVERAGE_EDGE_LENGTH) {
quality = VolumeToAverageEdgeLength();
} else if(qualityCriteria == QualityCriteria::VOLUME_TO_RMS_EDGE_LENGTH) {
quality = VolumeToRMSEdgeLength();
} else if(qualityCriteria == QualityCriteria::MIN_DIHEDRAL_ANGLE) {
quality = MinDihedralAngle();
} else if (qualityCriteria == QualityCriteria::MAX_DIHEDRAL_ANGLE) {
quality = MaxDihedralAngle();
} else if(qualityCriteria == QualityCriteria::MIN_SOLID_ANGLE) {
quality = MinSolidAngle();
}

return quality;
}


virtual inline void ComputeDihedralAngles(Vector& rDihedralAngles )  const
{
KRATOS_ERROR << "Called the virtual function for ComputeDihedralAngles " << *this << std::endl;
}


virtual inline void ComputeSolidAngles(Vector& rSolidAngles )  const
{
KRATOS_ERROR << "Called the virtual function for ComputeDihedralAngles " << *this << std::endl;
}



const PointsArrayType& Points() const
{
return mPoints;
}


PointsArrayType& Points()
{
return mPoints;
}


const typename TPointType::Pointer pGetPoint( const int Index ) const
{
KRATOS_TRY
return mPoints( Index );
KRATOS_CATCH(mPoints)
}


typename TPointType::Pointer pGetPoint( const int Index )
{
KRATOS_TRY
return mPoints( Index );
KRATOS_CATCH(mPoints);
}


TPointType const& GetPoint( const int Index ) const
{
KRATOS_TRY
return mPoints[Index];
KRATOS_CATCH( mPoints);
}



TPointType& GetPoint( const int Index )
{
KRATOS_TRY
return mPoints[Index];
KRATOS_CATCH(mPoints);
}


virtual Matrix& PointsLocalCoordinates( Matrix& rResult ) const
{
KRATOS_ERROR << "Calling base class 'PointsLocalCoordinates' method instead of derived class one. Please check the definition of derived class. " << *this << std::endl;
return rResult;
}


virtual CoordinatesArrayType& PointLocalCoordinates(
CoordinatesArrayType& rResult,
const CoordinatesArrayType& rPoint
) const
{
KRATOS_ERROR_IF(WorkingSpaceDimension() != LocalSpaceDimension()) << "ERROR:: Attention, the Point Local Coordinates must be specialized for the current geometry" << std::endl;

Matrix J = ZeroMatrix( WorkingSpaceDimension(), LocalSpaceDimension() );

rResult.clear();

Vector DeltaXi = ZeroVector( LocalSpaceDimension() );

CoordinatesArrayType CurrentGlobalCoords( ZeroVector( 3 ) );

static constexpr double MaxNormPointLocalCoordinates = 30.0;
static constexpr std::size_t MaxIteratioNumberPointLocalCoordinates = 1000;
static constexpr double MaxTolerancePointLocalCoordinates = 1.0e-8;

for(std::size_t k = 0; k < MaxIteratioNumberPointLocalCoordinates; k++) {
CurrentGlobalCoords.clear();
DeltaXi.clear();

GlobalCoordinates( CurrentGlobalCoords, rResult );
noalias( CurrentGlobalCoords ) = rPoint - CurrentGlobalCoords;
InverseOfJacobian( J, rResult );
for(unsigned int i = 0; i < WorkingSpaceDimension(); i++) {
for(unsigned int j = 0; j < WorkingSpaceDimension(); j++) {
DeltaXi[i] += J(i,j)*CurrentGlobalCoords[j];
}
rResult[i] += DeltaXi[i];
}

const double norm2DXi = norm_2(DeltaXi);

if(norm2DXi > MaxNormPointLocalCoordinates) {
KRATOS_WARNING("Geometry") << "Computation of local coordinates failed at iteration " << k << std::endl;
break;
}

if(norm2DXi < MaxTolerancePointLocalCoordinates) {
break;
}
}

return rResult;
}



virtual bool IsInside(
const CoordinatesArrayType& rPointGlobalCoordinates,
CoordinatesArrayType& rResult,
const double Tolerance = std::numeric_limits<double>::epsilon()
) const
{
PointLocalCoordinates(
rResult,
rPointGlobalCoordinates);

if (IsInsideLocalSpace(rResult, Tolerance) == 0) {
return false;
}
return true;
}


virtual int IsInsideLocalSpace(
const CoordinatesArrayType& rPointLocalCoordinates,
const double Tolerance = std::numeric_limits<double>::epsilon()
) const
{
KRATOS_ERROR << "Calling IsInsideLocalSpace from base class."
<< " Please check the definition of derived class. "
<< *this << std::endl;
return 0;
}



virtual void SpansLocalSpace(
std::vector<double>& rSpans,
IndexType LocalDirectionIndex = 0) const
{
KRATOS_ERROR <<
"Calling SpansLocalSpace of geometry base class. Please check derived definitions. "
<< *this << std::endl;
}



bool HasIntegrationMethod( IntegrationMethod ThisMethod ) const
{
return ( mpGeometryData->HasIntegrationMethod( ThisMethod ) );
}



IntegrationMethod GetDefaultIntegrationMethod() const
{
return mpGeometryData->DefaultIntegrationMethod();
}

virtual IntegrationInfo GetDefaultIntegrationInfo() const
{
return IntegrationInfo(LocalSpaceDimension(), GetDefaultIntegrationMethod());
}


virtual bool IsSymmetric() const
{
return false;
}



virtual GeometriesArrayType GenerateBoundariesEntities() const
{
const SizeType dimension = this->LocalSpaceDimension();
if (dimension == 3) {
return this->GenerateFaces();
} else if (dimension == 2) {
return this->GenerateEdges();
} else { 
return this->GeneratePoints();
}
}



virtual GeometriesArrayType GeneratePoints() const
{
GeometriesArrayType points;

const auto& p_points = this->Points();
for (IndexType i_point = 0; i_point < p_points.size(); ++i_point) {
PointsArrayType point_array;
point_array.push_back(p_points(i_point));
auto p_point_geometry = Kratos::make_shared<Geometry<TPointType>>(point_array);
points.push_back(p_point_geometry);
}

return points;
}



virtual SizeType EdgesNumber() const
{
KRATOS_ERROR << "Calling base class EdgesNumber method instead of derived class one. Please check the definition of derived class. " << *this << std::endl;
}


KRATOS_DEPRECATED_MESSAGE("This is legacy version (use GenerateEdgesInstead)") virtual GeometriesArrayType Edges( void )
{
return this->GenerateEdges();
}


virtual GeometriesArrayType GenerateEdges() const
{
KRATOS_ERROR << "Calling base class Edges method instead of derived class one. Please check the definition of derived class. " << *this << std::endl;
}





virtual SizeType FacesNumber() const
{
KRATOS_ERROR << "Calling base class FacesNumber method instead of derived class one. Please check the definition of derived class. " << *this << std::endl;
}


KRATOS_DEPRECATED_MESSAGE("This is legacy version (use GenerateEdgesInstead)") virtual GeometriesArrayType Faces( void )
{
const SizeType dimension = this->LocalSpaceDimension();
if (dimension == 3) {
return this->GenerateFaces();
} else {
return this->GenerateEdges();
}
}


virtual GeometriesArrayType GenerateFaces() const
{
KRATOS_ERROR << "Calling base class GenerateFaces method instead of derived class one. Please check the definition of derived class. " << *this << std::endl;
}

virtual void NumberNodesInFaces (DenseVector<unsigned int>& rNumberNodesInFaces) const
{
KRATOS_ERROR << "Calling base class NumberNodesInFaces method instead of derived class one. Please check the definition of derived class. " << *this << std::endl;
}

virtual void NodesInFaces (DenseMatrix<unsigned int>& rNodesInFaces) const
{
KRATOS_ERROR << "Calling base class NodesInFaces method instead of derived class one. Please check the definition of derived class. " << *this << std::endl;
}













SizeType IntegrationPointsNumber() const
{
return mpGeometryData->IntegrationPoints().size();
}


SizeType IntegrationPointsNumber( IntegrationMethod ThisMethod ) const
{
return mpGeometryData->IntegrationPointsNumber( ThisMethod );
}



const IntegrationPointsArrayType& IntegrationPoints() const
{
return mpGeometryData->IntegrationPoints();
}


const IntegrationPointsArrayType& IntegrationPoints( IntegrationMethod ThisMethod ) const
{
return mpGeometryData->IntegrationPoints( ThisMethod );
}


virtual void CreateIntegrationPoints(
IntegrationPointsArrayType& rIntegrationPoints,
IntegrationInfo& rIntegrationInfo) const
{
IntegrationMethod integration_method = rIntegrationInfo.GetIntegrationMethod(0);
for (IndexType i = 1; i < LocalSpaceDimension(); ++i) {
KRATOS_ERROR_IF(integration_method != rIntegrationInfo.GetIntegrationMethod(i))
<< "Default creation of integration points only valid if integration method is not varying per direction." << std::endl;
}
rIntegrationPoints = IntegrationPoints(integration_method);
}



virtual void CreateQuadraturePointGeometries(
GeometriesArrayType& rResultGeometries,
IndexType NumberOfShapeFunctionDerivatives,
const IntegrationPointsArrayType& rIntegrationPoints,
IntegrationInfo& rIntegrationInfo)
{
KRATOS_ERROR << "Calling CreateQuadraturePointGeometries from geometry base class."
<< " Please check the definition of derived class. "
<< *this << std::endl;
}


virtual void CreateQuadraturePointGeometries(
GeometriesArrayType& rResultGeometries,
IndexType NumberOfShapeFunctionDerivatives,
IntegrationInfo& rIntegrationInfo)
{
IntegrationPointsArrayType IntegrationPoints;
CreateIntegrationPoints(IntegrationPoints, rIntegrationInfo);

this->CreateQuadraturePointGeometries(
rResultGeometries,
NumberOfShapeFunctionDerivatives,
IntegrationPoints,
rIntegrationInfo);
}



virtual CoordinatesArrayType& GlobalCoordinates(
CoordinatesArrayType& rResult,
CoordinatesArrayType const& LocalCoordinates
) const
{
noalias( rResult ) = ZeroVector( 3 );

Vector N( this->size() );
ShapeFunctionsValues( N, LocalCoordinates );

for ( IndexType i = 0 ; i < this->size() ; i++ )
noalias( rResult ) += N[i] * (*this)[i];

return rResult;
}


void GlobalCoordinates(
CoordinatesArrayType& rResult,
IndexType IntegrationPointIndex
) const
{
this->GlobalCoordinates(rResult, IntegrationPointIndex, GetDefaultIntegrationMethod());
}


void GlobalCoordinates(
CoordinatesArrayType& rResult,
IndexType IntegrationPointIndex,
const IntegrationMethod ThisMethod
) const
{
noalias(rResult) = ZeroVector(3);

const Matrix& N = this->ShapeFunctionsValues(ThisMethod);

for (IndexType i = 0; i < this->size(); i++)
noalias(rResult) += N(IntegrationPointIndex, i) * (*this)[i];
}


virtual CoordinatesArrayType& GlobalCoordinates(
CoordinatesArrayType& rResult,
CoordinatesArrayType const& LocalCoordinates,
Matrix& DeltaPosition
) const
{
constexpr std::size_t dimension = 3;
noalias( rResult ) = ZeroVector( 3 );
if (DeltaPosition.size2() != 3)
DeltaPosition.resize(DeltaPosition.size1(), dimension,false);

Vector N( this->size() );
ShapeFunctionsValues( N, LocalCoordinates );

for ( IndexType i = 0 ; i < this->size() ; i++ )
noalias( rResult ) += N[i] * ((*this)[i] + row(DeltaPosition, i));

return rResult;
}


virtual void GlobalSpaceDerivatives(
std::vector<CoordinatesArrayType>& rGlobalSpaceDerivatives,
const CoordinatesArrayType& rLocalCoordinates,
const SizeType DerivativeOrder) const
{
if (DerivativeOrder == 0)
{
if (rGlobalSpaceDerivatives.size() != 1)
rGlobalSpaceDerivatives.resize(1);

this->GlobalCoordinates(
rGlobalSpaceDerivatives[0],
rLocalCoordinates);
}
else if (DerivativeOrder == 1)
{
const double local_space_dimension = LocalSpaceDimension();
const SizeType points_number = this->size();

if (rGlobalSpaceDerivatives.size() != 1 + local_space_dimension)
rGlobalSpaceDerivatives.resize(1 + local_space_dimension);

this->GlobalCoordinates(
rGlobalSpaceDerivatives[0],
rLocalCoordinates);

Matrix shape_functions_gradients(points_number, local_space_dimension);
this->ShapeFunctionsLocalGradients(shape_functions_gradients, rLocalCoordinates);

for (IndexType i = 0; i < points_number; ++i) {
const array_1d<double, 3>& r_coordinates = (*this)[i].Coordinates();
for (IndexType k = 0; k < WorkingSpaceDimension(); ++k) {
const double value = r_coordinates[k];
for (IndexType m = 0; m < local_space_dimension; ++m) {
rGlobalSpaceDerivatives[m + 1][k] += value * shape_functions_gradients(i, m);
}
}
}

return;
}
else
{
KRATOS_ERROR << "Calling GlobalDerivatives within geometry.h."
<< " Please check the definition within derived class. "
<< *this << std::endl;
}
}


virtual void GlobalSpaceDerivatives(
std::vector<CoordinatesArrayType>& rGlobalSpaceDerivatives,
IndexType IntegrationPointIndex,
const SizeType DerivativeOrder) const
{
if (DerivativeOrder == 0)
{
if (rGlobalSpaceDerivatives.size() != 1)
rGlobalSpaceDerivatives.resize(1);

GlobalCoordinates(
rGlobalSpaceDerivatives[0],
IntegrationPointIndex);
}
else if (DerivativeOrder == 1)
{
const double local_space_dimension = LocalSpaceDimension();
const SizeType points_number = this->size();

if (rGlobalSpaceDerivatives.size() != 1 + local_space_dimension)
rGlobalSpaceDerivatives.resize(1 + local_space_dimension);

this->GlobalCoordinates(
rGlobalSpaceDerivatives[0],
IntegrationPointIndex);

for (IndexType k = 0; k < local_space_dimension; ++k)
{
rGlobalSpaceDerivatives[1 + k] = ZeroVector(3);
}

const Matrix& r_shape_functions_gradient_in_integration_point = this->ShapeFunctionLocalGradient(IntegrationPointIndex);

for (IndexType i = 0; i < points_number; ++i) {
const array_1d<double, 3>& r_coordinates = (*this)[i].Coordinates();
for (IndexType k = 0; k < WorkingSpaceDimension(); ++k) {
const double value = r_coordinates[k];
for (IndexType m = 0; m < local_space_dimension; ++m) {
rGlobalSpaceDerivatives[m + 1][k] += value * r_shape_functions_gradient_in_integration_point(i, m);
}
}
}
}
else
{
KRATOS_ERROR << "Calling GlobalDerivatives within geometry.h."
<< " Please check the definition within derived class. "
<< *this << std::endl;
}
}



KRATOS_DEPRECATED_MESSAGE("This method is deprecated. Use either \'ProjectionPointLocalToLocalSpace\' or \'ProjectionPointGlobalToLocalSpace\' instead.")
virtual int ProjectionPoint(
const CoordinatesArrayType& rPointGlobalCoordinates,
CoordinatesArrayType& rProjectedPointGlobalCoordinates,
CoordinatesArrayType& rProjectedPointLocalCoordinates,
const double Tolerance = std::numeric_limits<double>::epsilon()
) const
{
KRATOS_ERROR << "Calling ProjectionPoint within geometry base class."
<< " Please check the definition within derived class. "
<< *this << std::endl;
}


virtual int ProjectionPointLocalToLocalSpace(
const CoordinatesArrayType& rPointLocalCoordinates,
CoordinatesArrayType& rProjectionPointLocalCoordinates,
const double Tolerance = std::numeric_limits<double>::epsilon()
) const
{
KRATOS_ERROR << "Calling ProjectionPointLocalToLocalSpace within geometry base class."
<< " Please check the definition within derived class. "
<< *this << std::endl;
}


virtual int ProjectionPointGlobalToLocalSpace(
const CoordinatesArrayType& rPointGlobalCoordinates,
CoordinatesArrayType& rProjectionPointLocalCoordinates,
const double Tolerance = std::numeric_limits<double>::epsilon()
) const
{
KRATOS_ERROR << "Calling ProjectionPoinGlobalToLocalSpace within geometry base class."
<< " Please check the definition within derived class. "
<< *this << std::endl;
}


KRATOS_DEPRECATED_MESSAGE("This method is deprecated. Use either \'ClosestPointLocalToLocalSpace\' or \'ClosestPointGlobalToLocalSpace\' instead. Please note that \'rClosestPointGlobalCoordinates\' returns unmodified original value.")
virtual int ClosestPoint(
const CoordinatesArrayType& rPointGlobalCoordinates,
CoordinatesArrayType& rClosestPointGlobalCoordinates,
CoordinatesArrayType& rClosestPointLocalCoordinates,
const double Tolerance = std::numeric_limits<double>::epsilon()
) const
{
return ClosestPointGlobalToLocalSpace(rPointGlobalCoordinates, rClosestPointLocalCoordinates, Tolerance);
}


KRATOS_DEPRECATED_MESSAGE("This method is deprecated. Use either \'ClosestPointLocalToLocalSpace\' or \'ClosestPointGlobalToLocalSpace\' instead.")
virtual int ClosestPoint(
const CoordinatesArrayType& rPointGlobalCoordinates,
CoordinatesArrayType& rClosestPointGlobalCoordinates,
const double Tolerance = std::numeric_limits<double>::epsilon()
) const
{
CoordinatesArrayType local_coordinates(ZeroVector(3));
const int result = ClosestPointGlobalToLocalSpace(rPointGlobalCoordinates, local_coordinates, Tolerance);

if (result == 1) {
this->GlobalCoordinates(rClosestPointGlobalCoordinates, local_coordinates);
}

return result;
}


KRATOS_DEPRECATED_MESSAGE("This method is deprecated. Use either \'ClosestPointLocalToLocalSpace\' or \'ClosestPointGlobalToLocalSpace\' instead.")
virtual int ClosestPointLocalCoordinates(
const CoordinatesArrayType& rPointGlobalCoordinates,
CoordinatesArrayType& rClosestPointLocalCoordinates,
const double Tolerance = std::numeric_limits<double>::epsilon()
) const
{
return ClosestPointGlobalToLocalSpace(rPointGlobalCoordinates, rClosestPointLocalCoordinates, Tolerance);
}


virtual int ClosestPointLocalToLocalSpace(
const CoordinatesArrayType& rPointLocalCoordinates,
CoordinatesArrayType& rClosestPointLocalCoordinates,
const double Tolerance = std::numeric_limits<double>::epsilon()
) const
{
const int projection_result = ProjectionPointLocalToLocalSpace(
rPointLocalCoordinates,
rClosestPointLocalCoordinates,
Tolerance);

if (projection_result == 1) {
return IsInsideLocalSpace(
rClosestPointLocalCoordinates,
Tolerance);
} else {
return -1;
}
}


virtual int ClosestPointGlobalToLocalSpace(
const CoordinatesArrayType& rPointGlobalCoordinates,
CoordinatesArrayType& rClosestPointLocalCoordinates,
const double Tolerance = std::numeric_limits<double>::epsilon()
) const
{
const int projection_result = ProjectionPointGlobalToLocalSpace(
rPointGlobalCoordinates,
rClosestPointLocalCoordinates,
Tolerance);

if (projection_result == 1) {
return IsInsideLocalSpace(
rClosestPointLocalCoordinates,
Tolerance);
} else {
return -1;
}
}


virtual double CalculateDistance(
const CoordinatesArrayType& rPointGlobalCoordinates,
const double Tolerance = std::numeric_limits<double>::epsilon()
) const
{
CoordinatesArrayType local_coordinates(ZeroVector(3));
if (ClosestPointGlobalToLocalSpace(rPointGlobalCoordinates, local_coordinates, Tolerance) < 1) {
return std::numeric_limits<double>::max();
}

CoordinatesArrayType global_coordinates(ZeroVector(3));
this->GlobalCoordinates(global_coordinates, local_coordinates);

return norm_2(rPointGlobalCoordinates - global_coordinates);
}



JacobiansType& Jacobian( JacobiansType& rResult ) const
{
Jacobian( rResult, mpGeometryData->DefaultIntegrationMethod() );
return rResult;
}


virtual JacobiansType& Jacobian( JacobiansType& rResult,
IntegrationMethod ThisMethod ) const
{
if( rResult.size() != this->IntegrationPointsNumber( ThisMethod ) )
rResult.resize( this->IntegrationPointsNumber( ThisMethod ), false );

for ( unsigned int pnt = 0; pnt < this->IntegrationPointsNumber( ThisMethod ); pnt++ ) {
this->Jacobian( rResult[pnt], pnt, ThisMethod);
}

return rResult;
}


virtual JacobiansType& Jacobian( JacobiansType& rResult, IntegrationMethod ThisMethod, Matrix & DeltaPosition ) const
{
if( rResult.size() != this->IntegrationPointsNumber( ThisMethod ) )
rResult.resize( this->IntegrationPointsNumber( ThisMethod ), false );

for ( unsigned int pnt = 0; pnt < this->IntegrationPointsNumber( ThisMethod ); pnt++ ) {
this->Jacobian( rResult[pnt], pnt, ThisMethod, DeltaPosition);
}
return rResult;
}


Matrix& Jacobian( Matrix& rResult, IndexType IntegrationPointIndex ) const
{
Jacobian( rResult, IntegrationPointIndex, mpGeometryData->DefaultIntegrationMethod() );
return rResult;
}


virtual Matrix& Jacobian( Matrix& rResult, IndexType IntegrationPointIndex, IntegrationMethod ThisMethod ) const
{
const SizeType working_space_dimension = this->WorkingSpaceDimension();
const SizeType local_space_dimension = this->LocalSpaceDimension();
if(rResult.size1() != working_space_dimension || rResult.size2() != local_space_dimension)
rResult.resize( working_space_dimension, local_space_dimension, false );

const Matrix& r_shape_functions_gradient_in_integration_point = ShapeFunctionsLocalGradients( ThisMethod )[ IntegrationPointIndex ];

rResult.clear();
const SizeType points_number = this->PointsNumber();
for (IndexType i = 0; i < points_number; ++i ) {
const array_1d<double, 3>& r_coordinates = (*this)[i].Coordinates();
for(IndexType k = 0; k< working_space_dimension; ++k) {
const double value = r_coordinates[k];
for(IndexType m = 0; m < local_space_dimension; ++m) {
rResult(k,m) += value * r_shape_functions_gradient_in_integration_point(i,m);
}
}
}

return rResult;
}


virtual Matrix& Jacobian( Matrix& rResult, IndexType IntegrationPointIndex, IntegrationMethod ThisMethod, const Matrix& rDeltaPosition ) const
{
const SizeType working_space_dimension = this->WorkingSpaceDimension();
const SizeType local_space_dimension = this->LocalSpaceDimension();
if(rResult.size1() != working_space_dimension || rResult.size2() != local_space_dimension)
rResult.resize( working_space_dimension, local_space_dimension, false );

const Matrix& r_shape_functions_gradient_in_integration_point = ShapeFunctionsLocalGradients( ThisMethod )[ IntegrationPointIndex ];

rResult.clear();
const SizeType points_number = this->PointsNumber();
for (IndexType i = 0; i < points_number; ++i ) {
const array_1d<double, 3>& r_coordinates = (*this)[i].Coordinates();
for(IndexType k = 0; k< working_space_dimension; ++k) {
const double value = r_coordinates[k] - rDeltaPosition(i,k);
for(IndexType m = 0; m < local_space_dimension; ++m) {
rResult(k,m) += value * r_shape_functions_gradient_in_integration_point(i,m);
}
}
}

return rResult;
}


virtual Matrix& Jacobian( Matrix& rResult, const CoordinatesArrayType& rCoordinates ) const
{
const SizeType working_space_dimension = this->WorkingSpaceDimension();
const SizeType local_space_dimension = this->LocalSpaceDimension();
const SizeType points_number = this->PointsNumber();
if(rResult.size1() != working_space_dimension || rResult.size2() != local_space_dimension)
rResult.resize( working_space_dimension, local_space_dimension, false );

Matrix shape_functions_gradients(points_number, local_space_dimension);
ShapeFunctionsLocalGradients( shape_functions_gradients, rCoordinates );

rResult.clear();
for (IndexType i = 0; i < points_number; ++i ) {
const array_1d<double, 3>& r_coordinates = (*this)[i].Coordinates();
for(IndexType k = 0; k< working_space_dimension; ++k) {
const double value = r_coordinates[k];
for(IndexType m = 0; m < local_space_dimension; ++m) {
rResult(k,m) += value * shape_functions_gradients(i,m);
}
}
}

return rResult;
}



virtual Matrix& Jacobian( Matrix& rResult, const CoordinatesArrayType& rCoordinates, Matrix& rDeltaPosition ) const
{
const SizeType working_space_dimension = this->WorkingSpaceDimension();
const SizeType local_space_dimension = this->LocalSpaceDimension();
const SizeType points_number = this->PointsNumber();
if(rResult.size1() != working_space_dimension || rResult.size2() != local_space_dimension)
rResult.resize( working_space_dimension, local_space_dimension, false );

Matrix shape_functions_gradients(points_number, local_space_dimension);
ShapeFunctionsLocalGradients( shape_functions_gradients, rCoordinates );

rResult.clear();
for (IndexType i = 0; i < points_number; ++i ) {
const array_1d<double, 3>& r_coordinates = (*this)[i].Coordinates();
for(IndexType k = 0; k< working_space_dimension; ++k) {
const double value = r_coordinates[k] - rDeltaPosition(i,k);
for(IndexType m = 0; m < local_space_dimension; ++m) {
rResult(k,m) += value * shape_functions_gradients(i,m);
}
}
}

return rResult;
}


Vector& DeterminantOfJacobian( Vector& rResult ) const
{
DeterminantOfJacobian( rResult, mpGeometryData->DefaultIntegrationMethod() );
return rResult;
}


virtual Vector& DeterminantOfJacobian( Vector& rResult, IntegrationMethod ThisMethod ) const
{
if( rResult.size() != this->IntegrationPointsNumber( ThisMethod ) )
rResult.resize( this->IntegrationPointsNumber( ThisMethod ), false );

Matrix J( this->WorkingSpaceDimension(), this->LocalSpaceDimension());
for ( unsigned int pnt = 0; pnt < this->IntegrationPointsNumber( ThisMethod ); pnt++ ) {
this->Jacobian( J, pnt, ThisMethod);
rResult[pnt] = MathUtils<double>::GeneralizedDet(J);
}
return rResult;
}


double DeterminantOfJacobian( IndexType IntegrationPointIndex ) const
{
return DeterminantOfJacobian( IntegrationPointIndex, mpGeometryData->DefaultIntegrationMethod() );
}


virtual double DeterminantOfJacobian( IndexType IntegrationPointIndex, IntegrationMethod ThisMethod ) const
{
Matrix J( this->WorkingSpaceDimension(), this->LocalSpaceDimension());
this->Jacobian( J, IntegrationPointIndex, ThisMethod);
return MathUtils<double>::GeneralizedDet(J);
}



virtual double DeterminantOfJacobian( const CoordinatesArrayType& rPoint ) const
{
Matrix J( this->WorkingSpaceDimension(), this->LocalSpaceDimension());
this->Jacobian( J, rPoint);
return MathUtils<double>::GeneralizedDet(J);
}



JacobiansType& InverseOfJacobian( JacobiansType& rResult ) const
{
InverseOfJacobian( rResult, mpGeometryData->DefaultIntegrationMethod() );
return rResult;
}


virtual JacobiansType& InverseOfJacobian( JacobiansType& rResult, IntegrationMethod ThisMethod ) const
{
Jacobian(rResult, ThisMethod); 

double detJ;
Matrix Jinv(this->LocalSpaceDimension(), this->WorkingSpaceDimension());
for ( unsigned int pnt = 0; pnt < this->IntegrationPointsNumber( ThisMethod ); pnt++ ) {
MathUtils<double>::GeneralizedInvertMatrix(rResult[pnt], Jinv, detJ);
noalias(rResult[pnt]) = Jinv;
}
return rResult;
}


Matrix& InverseOfJacobian( Matrix& rResult, IndexType IntegrationPointIndex ) const
{
InverseOfJacobian( rResult, IntegrationPointIndex, mpGeometryData->DefaultIntegrationMethod() );
return rResult;
}


virtual Matrix& InverseOfJacobian( Matrix& rResult, IndexType IntegrationPointIndex, IntegrationMethod ThisMethod ) const
{
Jacobian(rResult,IntegrationPointIndex, ThisMethod); 

double detJ;
Matrix Jinv(this->WorkingSpaceDimension(), this->WorkingSpaceDimension());

MathUtils<double>::GeneralizedInvertMatrix(rResult, Jinv, detJ);
noalias(rResult) = Jinv;

return rResult;
}


virtual Matrix& InverseOfJacobian( Matrix& rResult, const CoordinatesArrayType& rCoordinates ) const
{
Jacobian(rResult,rCoordinates); 

double detJ;
Matrix Jinv(this->WorkingSpaceDimension(), this->WorkingSpaceDimension());

MathUtils<double>::GeneralizedInvertMatrix(rResult, Jinv, detJ);
noalias(rResult) = Jinv;

return rResult;
}





const Matrix& ShapeFunctionsValues() const
{
return mpGeometryData->ShapeFunctionsValues();
}



virtual Vector& ShapeFunctionsValues (Vector &rResult, const CoordinatesArrayType& rCoordinates) const
{
KRATOS_ERROR << "Calling base class ShapeFunctionsValues method instead of derived class one. Please check the definition of derived class. " << *this << std::endl;
return rResult;
}


const Matrix& ShapeFunctionsValues( IntegrationMethod ThisMethod )  const
{
return mpGeometryData->ShapeFunctionsValues( ThisMethod );
}


double ShapeFunctionValue( IndexType IntegrationPointIndex, IndexType ShapeFunctionIndex ) const
{
return mpGeometryData->ShapeFunctionValue( IntegrationPointIndex, ShapeFunctionIndex );
}


double ShapeFunctionValue( IndexType IntegrationPointIndex, IndexType ShapeFunctionIndex, IntegrationMethod ThisMethod ) const
{
return mpGeometryData->ShapeFunctionValue( IntegrationPointIndex, ShapeFunctionIndex, ThisMethod );
}


virtual double ShapeFunctionValue( IndexType ShapeFunctionIndex, const CoordinatesArrayType& rCoordinates ) const
{
KRATOS_ERROR << "Calling base class ShapeFunctionValue method instead of derived class one. Please check the definition of derived class. " << *this << std::endl;

return 0;
}



const ShapeFunctionsGradientsType& ShapeFunctionsLocalGradients() const
{
return mpGeometryData->ShapeFunctionsLocalGradients();
}


const ShapeFunctionsGradientsType& ShapeFunctionsLocalGradients( IntegrationMethod ThisMethod ) const
{
return mpGeometryData->ShapeFunctionsLocalGradients( ThisMethod );
}


const Matrix& ShapeFunctionLocalGradient( IndexType IntegrationPointIndex )  const
{
return mpGeometryData->ShapeFunctionLocalGradient( IntegrationPointIndex );
}


const Matrix& ShapeFunctionLocalGradient(IndexType IntegrationPointIndex , IntegrationMethod ThisMethod)  const
{
return mpGeometryData->ShapeFunctionLocalGradient(IntegrationPointIndex, ThisMethod);
}

const Matrix& ShapeFunctionLocalGradient(IndexType IntegrationPointIndex, IndexType ShapeFunctionIndex, IntegrationMethod ThisMethod)  const
{
return mpGeometryData->ShapeFunctionLocalGradient(IntegrationPointIndex, ShapeFunctionIndex, ThisMethod);
}



virtual Matrix& ShapeFunctionsLocalGradients( Matrix& rResult, const CoordinatesArrayType& rPoint ) const
{
KRATOS_ERROR << "Calling base class ShapeFunctionsLocalGradients method instead of derived class one. Please check the definition of derived class. " << *this << std::endl;
return rResult;
}


const Matrix& ShapeFunctionDerivatives(
IndexType DerivativeOrderIndex,
IndexType IntegrationPointIndex,
IntegrationMethod ThisMethod) const
{
return mpGeometryData->ShapeFunctionDerivatives(
DerivativeOrderIndex, IntegrationPointIndex, ThisMethod);
}


const Matrix& ShapeFunctionDerivatives(
IndexType DerivativeOrderIndex,
IndexType IntegrationPointIndex) const
{
return mpGeometryData->ShapeFunctionDerivatives(
DerivativeOrderIndex, IntegrationPointIndex, GetDefaultIntegrationMethod());
}


virtual ShapeFunctionsSecondDerivativesType& ShapeFunctionsSecondDerivatives( ShapeFunctionsSecondDerivativesType& rResult, const CoordinatesArrayType& rPoint ) const
{
KRATOS_ERROR << "Calling base class ShapeFunctionsSecondDerivatives method instead of derived class one. Please check the definition of derived class. " << *this << std::endl;
return rResult;
}


virtual ShapeFunctionsThirdDerivativesType& ShapeFunctionsThirdDerivatives( ShapeFunctionsThirdDerivativesType& rResult, const CoordinatesArrayType& rPoint ) const
{
KRATOS_ERROR << "Calling base class ShapeFunctionsThirdDerivatives method instead of derived class one. Please check the definition of derived class. " << *this << std::endl;
return rResult;
}


void ShapeFunctionsIntegrationPointsGradients( ShapeFunctionsGradientsType& rResult ) const
{
ShapeFunctionsIntegrationPointsGradients( rResult, mpGeometryData->DefaultIntegrationMethod() );
}

virtual void ShapeFunctionsIntegrationPointsGradients(
ShapeFunctionsGradientsType& rResult,
IntegrationMethod ThisMethod ) const
{
KRATOS_ERROR_IF_NOT(this->WorkingSpaceDimension() == this->LocalSpaceDimension())
<< "\'ShapeFunctionsIntegrationPointsGradients\' is not defined for current geometry type as gradients are only defined in the local space." << std::endl;

const unsigned int integration_points_number = this->IntegrationPointsNumber( ThisMethod );

if ( integration_points_number == 0 )
KRATOS_ERROR << "This integration method is not supported" << *this << std::endl;

if ( rResult.size() != integration_points_number )
rResult.resize(  this->IntegrationPointsNumber( ThisMethod ), false  );

const ShapeFunctionsGradientsType& DN_De = ShapeFunctionsLocalGradients( ThisMethod );

Matrix Jinv(this->LocalSpaceDimension(), this->WorkingSpaceDimension());
for ( unsigned int pnt = 0; pnt < integration_points_number; pnt++ ) {
if (rResult[pnt].size1() != (*this).size() || rResult[pnt].size2() != this->LocalSpaceDimension())
rResult[pnt].resize( (*this).size(), this->LocalSpaceDimension(), false );
this->InverseOfJacobian(Jinv,pnt, ThisMethod);
noalias(rResult[pnt]) =  prod( DN_De[pnt], Jinv );
}
}

virtual void ShapeFunctionsIntegrationPointsGradients(
ShapeFunctionsGradientsType& rResult,
Vector& rDeterminantsOfJacobian,
IntegrationMethod ThisMethod ) const
{
KRATOS_ERROR_IF_NOT(this->WorkingSpaceDimension() == this->LocalSpaceDimension())
<< "\'ShapeFunctionsIntegrationPointsGradients\' is not defined for current geometry type as gradients are only defined in the local space." << std::endl;

const unsigned int integration_points_number = this->IntegrationPointsNumber( ThisMethod );

if ( integration_points_number == 0 )
KRATOS_ERROR << "This integration method is not supported " << *this << std::endl;

if ( rResult.size() != integration_points_number )
rResult.resize(  this->IntegrationPointsNumber( ThisMethod ), false  );
if (rDeterminantsOfJacobian.size() != integration_points_number)
rDeterminantsOfJacobian.resize(this->IntegrationPointsNumber(ThisMethod), false);

const ShapeFunctionsGradientsType& DN_De = ShapeFunctionsLocalGradients( ThisMethod );

Matrix J(this->WorkingSpaceDimension(), this->LocalSpaceDimension());
Matrix Jinv(this->LocalSpaceDimension(), this->WorkingSpaceDimension());
double DetJ;
for ( unsigned int pnt = 0; pnt < integration_points_number; pnt++ ) {
if (rResult[pnt].size1() != (*this).size() || rResult[pnt].size2() != this->LocalSpaceDimension())
rResult[pnt].resize( (*this).size(), this->LocalSpaceDimension(), false );
this->Jacobian(J,pnt, ThisMethod);
MathUtils<double>::GeneralizedInvertMatrix(J, Jinv, DetJ);
noalias(rResult[pnt]) =  prod( DN_De[pnt], Jinv );
rDeterminantsOfJacobian[pnt] = DetJ;
}
}

KRATOS_DEPRECATED_MESSAGE("This is signature of \'ShapeFunctionsIntegrationPointsGradients\' is legacy (use any of the alternatives without shape functions calculation).")
virtual void ShapeFunctionsIntegrationPointsGradients(
ShapeFunctionsGradientsType& rResult,
Vector& rDeterminantsOfJacobian,
IntegrationMethod ThisMethod,
Matrix& ShapeFunctionsIntegrationPointsValues) const
{

ShapeFunctionsIntegrationPointsGradients(rResult, rDeterminantsOfJacobian, ThisMethod);
ShapeFunctionsIntegrationPointsValues = ShapeFunctionsValues(ThisMethod);
}

virtual int Check() const
{
KRATOS_TRY

return 0;

KRATOS_CATCH("")
}


virtual std::string Info() const
{
std::stringstream buffer;
buffer << "Geometry # "
<< std::to_string(mId) << ": "
<< LocalSpaceDimension() << "-dimensional geometry in "
<< WorkingSpaceDimension() << "D space";

return buffer.str();
}

virtual std::string Name() const {
std::string geometryName = "BaseGeometry";
KRATOS_ERROR << "Base geometry does not have a name." << std::endl;
return geometryName;
}

virtual void PrintInfo(std::ostream& rOStream) const
{
rOStream << Info();
}

virtual void PrintName(std::ostream& rOstream) const {
rOstream << Name() << std::endl;
}

virtual void PrintData(std::ostream& rOStream) const
{
if (mpGeometryData) {
mpGeometryData->PrintData(rOStream);
}

rOStream << std::endl;
rOStream << std::endl;

for (unsigned int i = 0; i < this->size(); ++i) {
rOStream << "\tPoint " << i + 1 << "\t : ";
mPoints[i].PrintData(rOStream);
rOStream << std::endl;
}

rOStream << "\tCenter\t : ";

Center().PrintData(rOStream);

rOStream << std::endl;
rOStream << std::endl;




}




protected:


void SetGeometryData(GeometryData const* pGeometryData)
{
mpGeometryData = pGeometryData;
}




virtual double InradiusToCircumradiusQuality() const {
KRATOS_ERROR << "Calling base class 'InradiusToCircumradiusQuality' method instead of derived class one. Please check the definition of derived class. " << *this << std::endl;
return 0.0;
}


virtual double AreaToEdgeLengthRatio() const {
KRATOS_ERROR << "Calling base class 'AreaToEdgeLengthRatio' method instead of derived class one. Please check the definition of derived class. " << *this << std::endl;
return 0.0;
}


virtual double ShortestAltitudeToEdgeLengthRatio() const {
KRATOS_ERROR << "Calling base class 'ShortestAltitudeToEdgeLengthRatio' method instead of derived class one. Please check the definition of derived class. " << *this << std::endl;
return 0.0;
}


virtual double InradiusToLongestEdgeQuality() const {
KRATOS_ERROR << "Calling base class 'InradiusToLongestEdgeQuality' method instead of derived class one. Please check the definition of derived class. " << *this << std::endl;
return 0.0;
}


virtual double ShortestToLongestEdgeQuality() const {
KRATOS_ERROR << "Calling base class 'ShortestToLongestEdgeQuality' method instead of derived class one. Please check the definition of derived class. " << *this << std::endl;
return 0.0;
}


virtual double RegularityQuality() const {
KRATOS_ERROR << "Calling base class 'RegularityQuality' method instead of derived class one. Please check the definition of derived class. " << *this << std::endl;
return 0.0;
}


virtual double VolumeToSurfaceAreaQuality() const {
KRATOS_ERROR << "Calling base class 'VolumeToSurfaceAreaQuality' method instead of derived class one. Please check the definition of derived class. " << *this << std::endl;
return 0.0;
}


virtual double VolumeToEdgeLengthQuality() const {
KRATOS_ERROR << "Calling base class 'VolumeToEdgeLengthQuality' method instead of derived class one. Please check the definition of derived class. " << *this << std::endl;
return 0.0;
}


virtual double VolumeToAverageEdgeLength() const {
KRATOS_ERROR << "Calling base class 'VolumeToAverageEdgeLength' method instead of derived class one. Please check the definition of derived class. " << *this << std::endl;
return 0.0;
}


virtual double VolumeToRMSEdgeLength() const {
KRATOS_ERROR << "Calling base class 'VolumeToRMSEdgeLength' method instead of derived class one. Please check the definition of derived class. " << *this << std::endl;
return 0.0;
}


virtual double MinDihedralAngle() const {
KRATOS_ERROR << "Calling base class 'MinDihedralAngle' method instead of derived class one. Please check the definition of derived class. " << *this << std::endl;
return 0.0;
}


virtual double MaxDihedralAngle() const {
KRATOS_ERROR << "Calling base class 'MaxDihedralAngle' method instead of derived class one. Please check the definition of derived class. " << *this << std::endl;
return 0.0;
}


virtual double MinSolidAngle() const {
KRATOS_ERROR << "Calling base class 'MinSolidAngle' method instead of derived class one. Please check the definition of derived class. " << *this << std::endl;
return 0.0;
}






private:

IndexType mId;

GeometryData const* mpGeometryData;

static const GeometryDimension msGeometryDimension;

PointsArrayType mPoints;

DataValueContainer mData;



IndexType GenerateSelfAssignedId() const
{
IndexType id = reinterpret_cast<IndexType>(this);

SetIdSelfAssigned(id);

SetIdNotGeneratedFromString(id);

return id;
}

static inline bool IsIdGeneratedFromString(IndexType Id)
{
return Id & (IndexType(1) << (sizeof(IndexType) * 8 - 1));
}

static inline void SetIdGeneratedFromString(IndexType& Id)
{
Id |= (IndexType(1) << (sizeof(IndexType) * 8 - 1));
}

static inline void SetIdNotGeneratedFromString(IndexType& Id)
{
Id &= ~(IndexType(1) << (sizeof(IndexType) * 8 - 1));
}

static inline bool IsIdSelfAssigned(IndexType Id)
{
return Id & (IndexType(1) << (sizeof(IndexType) * 8 - 2));
}

static inline void SetIdSelfAssigned(IndexType& Id)
{
Id |= (IndexType(1) << (sizeof(IndexType) * 8 - 2));
}

static inline void SetIdNotSelfAssigned(IndexType& Id)
{
Id &= ~(IndexType(1) << (sizeof(IndexType) * 8 - 2));
}


friend class Serializer;

virtual void save( Serializer& rSerializer ) const
{
rSerializer.save("Id", mId);
rSerializer.save( "Points", mPoints);
rSerializer.save("Data", mData);
}

virtual void load( Serializer& rSerializer )
{
rSerializer.load("Id", mId);
rSerializer.load( "Points", mPoints );
rSerializer.load("Data", mData);
}


void SetIdWithoutCheck(const IndexType Id)
{
mId = Id;
}

static const GeometryData& GeometryDataInstance()
{
IntegrationPointsContainerType integration_points = {};
ShapeFunctionsValuesContainerType shape_functions_values = {};
ShapeFunctionsLocalGradientsContainerType shape_functions_local_gradients = {};
static GeometryData s_geometry_data(
&msGeometryDimension,
GeometryData::IntegrationMethod::GI_GAUSS_1,
integration_points,
shape_functions_values,
shape_functions_local_gradients);

return s_geometry_data;
}







template<class TOtherPointType> friend class Geometry;




}; 






template<class TPointType>
inline std::istream& operator >> ( std::istream& rIStream,
Geometry<TPointType>& rThis );

template<class TPointType>
inline std::ostream& operator << ( std::ostream& rOStream,
const Geometry<TPointType>& rThis )
{
rThis.PrintInfo( rOStream );
rOStream << std::endl;
rThis.PrintData( rOStream );

return rOStream;
}


template<class TPointType>
const GeometryDimension Geometry<TPointType>::msGeometryDimension(3, 3);

}  
