
#pragma once



#include "geometries/geometry.h"
namespace Kratos
{


class IntegrationUtilities
{
public:

template<class TPointType>
static GeometryData::IntegrationMethod GetIntegrationMethodForExactMassMatrixEvaluation(const Geometry<TPointType>& rGeometry)
{
GeometryData::IntegrationMethod integration_method = rGeometry.GetDefaultIntegrationMethod();
if (integration_method == GeometryData::IntegrationMethod::GI_GAUSS_1)
integration_method = GeometryData::IntegrationMethod::GI_GAUSS_2;
else if(integration_method == GeometryData::IntegrationMethod::GI_GAUSS_2)
integration_method = GeometryData::IntegrationMethod::GI_GAUSS_3;
else if(integration_method == GeometryData::IntegrationMethod::GI_GAUSS_3)
integration_method = GeometryData::IntegrationMethod::GI_GAUSS_4;
else if(integration_method == GeometryData::IntegrationMethod::GI_GAUSS_4)
integration_method = GeometryData::IntegrationMethod::GI_GAUSS_5;
return integration_method;
}


template<class TGeometryType>
static inline double ComputeDomainSize(const TGeometryType& rGeometry)
{
const auto& r_integration_points = rGeometry.IntegrationPoints();
const auto number_gp = r_integration_points.size();
Vector temp(number_gp);
temp = rGeometry.DeterminantOfJacobian(temp);
double domain_size = 0.0;
for (unsigned int i = 0; i < number_gp; ++i) {
domain_size += temp[i] * r_integration_points[i].Weight();
}
return domain_size;
}


template<class TPointType>
static inline double ComputeDomainSize(
const Geometry<TPointType>& rGeometry,
const typename Geometry<TPointType>::IntegrationMethod IntegrationMethod
)
{
const auto& r_integration_points = rGeometry.IntegrationPoints( IntegrationMethod );
const auto number_gp = r_integration_points.size();
Vector temp(number_gp);
temp = rGeometry.DeterminantOfJacobian(temp, IntegrationMethod);
double domain_size = 0.0;
for (unsigned int i = 0; i < number_gp; ++i) {
domain_size += temp[i] * r_integration_points[i].Weight();
}
return domain_size;
}


template<class TPointType>
static inline double ComputeArea2DGeometry(const Geometry<TPointType>& rGeometry)
{
const auto integration_method = rGeometry.GetDefaultIntegrationMethod();
const auto& r_integration_points = rGeometry.IntegrationPoints( integration_method );
double volume = 0.0;
Matrix J(2, 2);
for ( unsigned int i = 0; i < r_integration_points.size(); i++ ) {
rGeometry.Jacobian( J, i, integration_method);
volume += MathUtils<double>::Det2(J) * r_integration_points[i].Weight();
}

return volume;
}


template<class TPointType>
static inline double ComputeVolume3DGeometry(const Geometry<TPointType>& rGeometry)
{
const auto integration_method = rGeometry.GetDefaultIntegrationMethod();
const auto& r_integration_points = rGeometry.IntegrationPoints( integration_method );
double volume = 0.0;
Matrix J(3, 3);
for ( unsigned int i = 0; i < r_integration_points.size(); i++ ) {
rGeometry.Jacobian( J, i, integration_method);
volume += MathUtils<double>::Det3(J) * r_integration_points[i].Weight();
}

return volume;
}

};

}  
