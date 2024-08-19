#pragma once

#include "includes/serializer.h"

#include "custom_conditions/U_Pw_condition.hpp"
#include "custom_conditions/U_Pw_face_load_condition.hpp"
#include "custom_utilities/element_utilities.hpp"
#include "custom_utilities/condition_utilities.hpp"
#include "geo_mechanics_application_variables.h"

namespace Kratos
{

template< unsigned int TDim, unsigned int TNumNodes >
class KRATOS_API(GEO_MECHANICS_APPLICATION) UPwLysmerAbsorbingCondition : public UPwFaceLoadCondition<TDim,TNumNodes>
{

public:

KRATOS_CLASS_INTRUSIVE_POINTER_DEFINITION(UPwLysmerAbsorbingCondition);

using IndexType = std::size_t;
using PropertiesType = Properties;
using NodeType = Node;
using GeometryType = Geometry<NodeType>;
using NodesArrayType = GeometryType::PointsArrayType;
using VectorType = Vector;
using MatrixType = Matrix;


UPwLysmerAbsorbingCondition() : UPwFaceLoadCondition<TDim,TNumNodes>() {}

UPwLysmerAbsorbingCondition( IndexType NewId, GeometryType::Pointer pGeometry ) : UPwFaceLoadCondition<TDim,TNumNodes>(NewId, pGeometry) {}

UPwLysmerAbsorbingCondition( IndexType NewId, GeometryType::Pointer pGeometry, PropertiesType::Pointer pProperties ) : UPwFaceLoadCondition<TDim,TNumNodes>(NewId, pGeometry, pProperties) {}

~UPwLysmerAbsorbingCondition() override {}


Condition::Pointer Create(IndexType NewId,NodesArrayType const& ThisNodes,PropertiesType::Pointer pProperties ) const override;


void GetValuesVector(Vector& rValues, int Step) const override;


void GetFirstDerivativesVector(Vector& rValues, int Step) const override;


void CalculateRightHandSide(VectorType& rRightHandSideVector,
const ProcessInfo& rCurrentProcessInfo) override;


void CalculateDampingMatrix(MatrixType& rDampingMatrix, const ProcessInfo& rCurrentProcessInfo) override;


void CalculateLocalSystem(MatrixType& rLhsMatrix, VectorType& rRightHandSideVector, const ProcessInfo& rCurrentProcessInfo) override;


protected:

static constexpr SizeType N_DOF = TNumNodes * TDim;
static constexpr SizeType CONDITION_SIZE = TNumNodes * TDim + TNumNodes;

using ElementMatrixType = BoundedMatrix<double, N_DOF, N_DOF>;
using DimensionMatrixType = BoundedMatrix<double, TDim, TDim>;

struct NormalLysmerAbsorbingVariables
{
double rho; 
double Ec; 
double G; 
double n; 
double vp; 
double vs; 
double p_factor; 
double s_factor; 
double virtual_thickness;
Vector EcNodes;
Vector GNodes;
Vector SaturationNodes;
Vector rhoNodes;

DimensionMatrixType CAbsMatrix; 
DimensionMatrixType KAbsMatrix; 
};





void AddLHS(MatrixType& rLeftHandSideMatrix, const ElementMatrixType& rUMatrix);


void CalculateAndAddRHS(VectorType& rRightHandSideVector, const MatrixType& rStiffnessMatrix);


void CalculateRotationMatrix(DimensionMatrixType& rRotationMatrix, const Element::GeometryType& rGeom);


void GetNeighbourElementVariables(NormalLysmerAbsorbingVariables& rVariables, const ProcessInfo& rCurrentProcessInfo);


void GetVariables(NormalLysmerAbsorbingVariables& rVariables, const ProcessInfo& rCurrentProcessInfo);


void CalculateNodalDampingMatrix(NormalLysmerAbsorbingVariables& rVariables, const Element::GeometryType& rGeom);


void CalculateNodalStiffnessMatrix(NormalLysmerAbsorbingVariables& rVariables, const Element::GeometryType& rGeom);


Matrix CalculateExtrapolationMatrixNeighbour(const Element& rNeighbourElement);

private:
using hashmap = std::unordered_multimap<DenseVector<int>, std::vector<Condition::Pointer>, KeyHasherRange<DenseVector<int>>, KeyComparorRange<DenseVector<int>>>;


void CalculateRotationMatrix2DLine(DimensionMatrixType& rRotationMatrix, const Element::GeometryType& rGeom);


void CalculateConditionStiffnessMatrix(ElementMatrixType& rStiffnessMatrix, const ProcessInfo& rCurrentProcessInfo);

friend class Serializer;

void save(Serializer& rSerializer) const override
{
KRATOS_SERIALIZE_SAVE_BASE_CLASS( rSerializer, Condition )
}

void load(Serializer& rSerializer) override
{
KRATOS_SERIALIZE_LOAD_BASE_CLASS( rSerializer, Condition )
}

}; 

} 
