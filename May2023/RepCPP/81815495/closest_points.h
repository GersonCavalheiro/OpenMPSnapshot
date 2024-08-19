

#pragma once

#include <set>
#include <limits>


#include "includes/define.h"
#include "includes/indexed_object.h"
#include "includes/serializer.h"
#include "geometries/point.h"

namespace Kratos
{

class KRATOS_API(MAPPING_APPLICATION) PointWithId : public IndexedObject, public Point
{
public:
using IndexedObject::IndexType;

PointWithId(const IndexType NewId, const CoordinatesArrayType& rCoords, const double Distance);

PointWithId(const PointWithId& rOther);

PointWithId& operator=(const PointWithId& rOther) = delete;

bool operator<(const PointWithId& rOther) const;

double GetDistance() const { return mDistance; }

private:

double mDistance;



PointWithId() = default;

friend class Serializer;

void save(Serializer &rSerializer) const override;

void load(Serializer &rSerializer) override;

};


class KRATOS_API(MAPPING_APPLICATION) ClosestPointsContainer
{
public:
using ContainerType = std::set<PointWithId>;

explicit ClosestPointsContainer(const std::size_t MaxSize);
ClosestPointsContainer(const std::size_t MaxSize, const double MaxDistance);

explicit ClosestPointsContainer(const ClosestPointsContainer& rOther);

ClosestPointsContainer& operator=(const ClosestPointsContainer& rOther) = delete;

bool operator==(const ClosestPointsContainer& rOther) const;

void Add(const PointWithId& rPoint);

void Merge(const ClosestPointsContainer& rOther);

ContainerType& GetPoints() { return mClosestPoints; }

const ContainerType& GetPoints() const { return mClosestPoints; }

private:

ContainerType mClosestPoints;
std::size_t mMaxSize;
double mMaxDistance = std::numeric_limits<double>::max();


void LimitToMaxSize();


ClosestPointsContainer() = default;

friend class Serializer;

void save(Serializer &rSerializer) const;

void load(Serializer &rSerializer);

};


}  