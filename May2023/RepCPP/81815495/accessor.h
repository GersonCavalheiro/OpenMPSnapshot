
# pragma once



#include "geometries/geometry.h"
#include "includes/process_info.h"
#include "includes/node.h"

namespace Kratos
{


class Properties;


class KRATOS_API(KRATOS_CORE) Accessor
{
public:

using GeometryType = Geometry<Node>;

KRATOS_CLASS_POINTER_DEFINITION(Accessor);


Accessor() = default;

virtual ~Accessor() = default;

Accessor(const Accessor& rOther) {}



virtual double GetValue(
const Variable<double>& rVariable,
const Properties& rProperties,
const GeometryType& rGeometry,
const Vector& rShapeFunctionVector,
const ProcessInfo& rProcessInfo
) const;


virtual Vector GetValue(
const Variable<Vector>& rVariable,
const Properties& rProperties,
const GeometryType& rGeometry,
const Vector& rShapeFunctionVector,
const ProcessInfo& rProcessInfo
) const;


virtual bool GetValue(
const Variable<bool>& rVariable,
const Properties& rProperties,
const GeometryType& rGeometry,
const Vector& rShapeFunctionVector,
const ProcessInfo& rProcessInfo
) const;


virtual int GetValue(
const Variable<int>& rVariable,
const Properties& rProperties,
const GeometryType& rGeometry,
const Vector& rShapeFunctionVector,
const ProcessInfo& rProcessInfo
) const;


virtual Matrix GetValue(
const Variable<Matrix>& rVariable,
const Properties& rProperties,
const GeometryType& rGeometry,
const Vector& rShapeFunctionVector,
const ProcessInfo& rProcessInfo
) const;


virtual array_1d<double, 3> GetValue(
const Variable<array_1d<double, 3>>& rVariable,
const Properties& rProperties,
const GeometryType& rGeometry,
const Vector& rShapeFunctionVector,
const ProcessInfo& rProcessInfo
) const;


virtual array_1d<double, 6> GetValue(
const Variable<array_1d<double, 6>>& rVariable,
const Properties& rProperties,
const GeometryType& rGeometry,
const Vector& rShapeFunctionVector,
const ProcessInfo& rProcessInfo
) const;


virtual array_1d<double, 4> GetValue(
const Variable<array_1d<double, 4>>& rVariable,
const Properties& rProperties,
const GeometryType& rGeometry,
const Vector& rShapeFunctionVector,
const ProcessInfo& rProcessInfo
) const;


virtual array_1d<double, 9> GetValue(
const Variable<array_1d<double, 9>>& rVariable,
const Properties& rProperties,
const GeometryType& rGeometry,
const Vector& rShapeFunctionVector,
const ProcessInfo& rProcessInfo
) const;


virtual std::string GetValue(
const Variable<std::string>& rVariable,
const Properties& rProperties,
const GeometryType& rGeometry,
const Vector& rShapeFunctionVector,
const ProcessInfo& rProcessInfo
) const;

virtual Accessor::UniquePointer Clone() const;


virtual std::string Info() const
{
std::stringstream buffer;
buffer << "Accessor" ;

return buffer.str();
}

virtual void PrintInfo(std::ostream& rOStream) const  {rOStream << "Accessor";}

virtual void PrintData(std::ostream& rOStream) const {rOStream << "virtual method of the base Accessor class";}


private:


friend class Serializer;

void save(Serializer& rSerializer) const
{}

void load(Serializer& rSerializer)
{}


}; 


} 