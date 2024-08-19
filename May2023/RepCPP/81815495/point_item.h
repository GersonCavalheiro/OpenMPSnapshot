
#pragma once



#include "geometries/point.h"

namespace Kratos
{






template<class TEntity>
class PointItem
: public Point
{
public:


typedef Point BaseType;

KRATOS_CLASS_POINTER_DEFINITION( PointItem );


PointItem():
BaseType(),
mpOriginEntity(nullptr)
{}

PointItem(const array_1d<double, 3>& rCoordinates)
:BaseType(rCoordinates),
mpOriginEntity(nullptr)
{}

PointItem(typename TEntity::Pointer pEntity):
mpOriginEntity(pEntity)
{
UpdatePoint();
}

PointItem(
const array_1d<double, 3>& rCoordinates,
typename TEntity::Pointer pEntity
):
BaseType(rCoordinates),
mpOriginEntity(pEntity)
{}

PointItem(const PointItem& rRHS):
BaseType(rRHS),
mpOriginEntity(rRHS.mpOriginEntity)
{
}

~PointItem() override= default;




BaseType GetPoint()
{
BaseType Point(this->Coordinates());

return Point;
}


void SetPoint(const BaseType& rPoint)
{
this->Coordinates() = rPoint.Coordinates();
}



void SetEntity(typename TEntity::Pointer pEntity)
{
mpOriginEntity = pEntity;
}



typename TEntity::Pointer GetEntity()
{
#ifdef KRATOS_DEBUG
KRATOS_ERROR_IF(mpOriginEntity.get() == nullptr) << "TEntity no initialized in the PointItem class" << std::endl;
#endif
return mpOriginEntity;
}



void Check()
{
KRATOS_TRY;

auto aux_coord = std::make_shared<array_1d<double, 3>>(this->Coordinates());
KRATOS_ERROR_IF(!aux_coord) << "Coordinates no initialized in the PointItem class" << std::endl;
KRATOS_ERROR_IF(mpOriginEntity.get() == nullptr) << "TEntity no initialized in the PointItem class" << std::endl;

KRATOS_CATCH("Error checking the PointItem");
}


void UpdatePoint()
{
noalias(this->Coordinates()) = mpOriginEntity->GetGeometry().Center().Coordinates();
}

private:

typename TEntity::Pointer mpOriginEntity; 

}; 






}  
