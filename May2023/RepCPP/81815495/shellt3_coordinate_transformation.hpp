
#pragma once



#include "shellt3_local_coordinate_system.hpp"

namespace Kratos
{

class ShellT3_CoordinateTransformation
{

public:

KRATOS_CLASS_POINTER_DEFINITION(ShellT3_CoordinateTransformation);

typedef Element::GeometryType GeometryType;

typedef Vector VectorType;

typedef Matrix MatrixType;

public:

ShellT3_CoordinateTransformation(const GeometryType::Pointer& pGeometry)
: mpGeometry(pGeometry)
{
}

virtual ~ShellT3_CoordinateTransformation()
{
}

private:

ShellT3_CoordinateTransformation(const ShellT3_CoordinateTransformation& other);

ShellT3_CoordinateTransformation& operator = (const ShellT3_CoordinateTransformation& other);

public:

virtual ShellT3_CoordinateTransformation::Pointer Create(GeometryType::Pointer pGeometry)const
{
return ShellT3_CoordinateTransformation::Pointer(new ShellT3_CoordinateTransformation(pGeometry));
}

virtual void Initialize()
{
}

virtual void InitializeSolutionStep()
{
}

virtual void FinalizeSolutionStep()
{
}

virtual void InitializeNonLinearIteration()
{
}

virtual void FinalizeNonLinearIteration()
{
}

virtual ShellT3_LocalCoordinateSystem CreateReferenceCoordinateSystem()const
{
const GeometryType& geom = GetGeometry();
return ShellT3_LocalCoordinateSystem(geom[0].GetInitialPosition(),
geom[1].GetInitialPosition(),
geom[2].GetInitialPosition());
}

virtual ShellT3_LocalCoordinateSystem CreateLocalCoordinateSystem()const
{
return CreateReferenceCoordinateSystem();
}

virtual Vector CalculateLocalDisplacements(const ShellT3_LocalCoordinateSystem& LCS,
const VectorType& globalDisplacements)
{
MatrixType R(18, 18);
LCS.ComputeTotalRotationMatrix(R);
return prod(R, globalDisplacements);
}

virtual void FinalizeCalculations(const ShellT3_LocalCoordinateSystem& LCS,
const VectorType& globalDisplacements,
const VectorType& localDisplacements,
MatrixType& rLeftHandSideMatrix,
VectorType& rRightHandSideVector,
const bool RHSrequired,
const bool LHSrequired)
{
MatrixType R(18, 18);
LCS.ComputeTotalRotationMatrix(R);

if (LHSrequired) {
MatrixType temp(18, 18);
noalias(temp) = prod(trans(R), rLeftHandSideMatrix);
noalias(rLeftHandSideMatrix) = prod(temp, R);
}

if (RHSrequired) {
rRightHandSideVector = prod(trans(R), rRightHandSideVector);
}
}

virtual MatrixType GetNodalDeformationalRotationTensor(const ShellT3_LocalCoordinateSystem& LCS,
const Vector& globalDisplacements,
size_t nodeid)
{
return IdentityMatrix(3);
}

virtual MatrixType GetNodalDeformationalRotationTensor(const ShellT3_LocalCoordinateSystem& LCS,
const Vector& globalDisplacements,
const Vector& N)
{
return IdentityMatrix(3);
}

public:

inline const GeometryType& GetGeometry()const
{
return *mpGeometry;
}

protected:

ShellT3_CoordinateTransformation()
: mpGeometry(GeometryType::Pointer())
{
}

private:

GeometryType::Pointer mpGeometry;

private:

friend class Serializer;

virtual void save(Serializer& rSerializer) const
{
rSerializer.save("pGeom", mpGeometry);
}

virtual void load(Serializer& rSerializer)
{
rSerializer.load("pGeom", mpGeometry);
}

};

}
