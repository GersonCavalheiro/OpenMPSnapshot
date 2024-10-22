
#pragma once

#include <string>
#include <iostream>
#include <cstddef>
#include <vector>


#include "tree.h"

namespace Kratos
{







template< class TLeafType >
class KDTreePartitionBase : public TreeNode< TLeafType::Dimension,
typename TLeafType::PointType,
typename TLeafType::PointerType,
typename TLeafType::IteratorType,
typename TLeafType::DistanceIteratorType >
{
public:

KRATOS_CLASS_POINTER_DEFINITION(KDTreePartitionBase);

typedef TLeafType LeafType;

typedef typename LeafType::PointType PointType;

typedef typename LeafType::ContainerType ContainerType;

typedef typename LeafType::IteratorType IteratorType;

typedef typename LeafType::DistanceIteratorType DistanceIteratorType;

typedef typename LeafType::PointerType PointerType;

typedef typename LeafType::DistanceFunction DistanceFunction;

enum { Dimension = LeafType::Dimension };

typedef TreeNode<Dimension,PointType, PointerType, IteratorType, DistanceIteratorType> TreeNodeType;

typedef typename TreeNodeType::CoordinateType CoordinateType;

typedef typename TreeNodeType::SizeType SizeType;

typedef typename TreeNodeType::IndexType IndexType;

typedef typename LeafType::SearchStructureType SearchStructureType;



KDTreePartitionBase(IndexType CutingDimension, CoordinateType Position,
CoordinateType LeftEnd, CoordinateType RightEnd,
TreeNodeType* pLeftChild = NULL, TreeNodeType* pRightChild = NULL)
: mCutingDimension(CutingDimension), mPosition(Position), mLeftEnd(LeftEnd), mRightEnd(RightEnd)
{
mpChilds[0] = pLeftChild;
mpChilds[1] = pRightChild;
}

void PrintData(std::ostream& rOStream, std::string const& Perfix = std::string()) const override
{
rOStream << Perfix << "Partition at ";
switch(mCutingDimension)
{
case 0:
rOStream << "X =";
break;
case 1:
rOStream << "Y =";
break;
case 2:
rOStream << "Z =";
break;
default:
rOStream << mCutingDimension << " in";
break;
}
rOStream << mPosition << " from " << mLeftEnd << " to " << mRightEnd << std::endl;

mpChilds[0]->PrintData(rOStream, Perfix + "  ");
mpChilds[1]->PrintData(rOStream, Perfix + "  ");

}

virtual ~KDTreePartitionBase()
{
delete mpChilds[0];
delete mpChilds[1];
}



void SearchNearestPoint(
PointType const& rThisPoint,
PointerType& rResult,
CoordinateType& rResultDistance
) override
{
SearchStructureType Auxiliar;
for(SizeType i = 0 ; i < Dimension; i++)
Auxiliar.residual_distance[i] = 0.00;
SearchNearestPoint(rThisPoint, rResult, rResultDistance, Auxiliar );
}

void SearchNearestPoint(
PointType const& rThisPoint,
PointerType& rResult,
CoordinateType& rResultDistance,
SearchStructureType& Auxiliar
) override
{
CoordinateType temp = Auxiliar.residual_distance[mCutingDimension];
CoordinateType distance_to_partition = rThisPoint[mCutingDimension] - mPosition;

if( distance_to_partition < 0.0 ) 
{
mpChilds[0]->SearchNearestPoint(rThisPoint, rResult, rResultDistance, Auxiliar );

Auxiliar.residual_distance[mCutingDimension] = distance_to_partition * distance_to_partition;
Auxiliar.distance_to_partition2 = Auxiliar.residual_distance[0];
for(SizeType i = 1; i < Dimension; i++)
Auxiliar.distance_to_partition2 += Auxiliar.residual_distance[i];
if( rResultDistance > Auxiliar.distance_to_partition2 )
mpChilds[1]->SearchNearestPoint(rThisPoint, rResult, rResultDistance, Auxiliar );

}
else  
{
mpChilds[1]->SearchNearestPoint(rThisPoint, rResult, rResultDistance, Auxiliar );

Auxiliar.residual_distance[mCutingDimension] = distance_to_partition * distance_to_partition;
Auxiliar.distance_to_partition2 = Auxiliar.residual_distance[0];
for(SizeType i = 1; i < Dimension; i++)
Auxiliar.distance_to_partition2 += Auxiliar.residual_distance[i];
if( rResultDistance > Auxiliar.distance_to_partition2 )
mpChilds[0]->SearchNearestPoint( rThisPoint, rResult, rResultDistance, Auxiliar );
}
Auxiliar.residual_distance[mCutingDimension] = temp;

}

void SearchInRadius(
PointType const& ThisPoint,
CoordinateType const& Radius,
CoordinateType const& Radius2,
IteratorType& Results,
DistanceIteratorType& ResultsDistances,
SizeType& NumberOfResults,
SizeType const& MaxNumberOfResults
) override
{
SearchStructureType Auxiliar;
for(SizeType i = 0 ; i < Dimension; i++)
Auxiliar.residual_distance[i] = 0.00;
SearchInRadius(ThisPoint, Radius, Radius2, Results, ResultsDistances, NumberOfResults, MaxNumberOfResults, Auxiliar );
}

void SearchInRadius(
PointType const& ThisPoint,
CoordinateType const& Radius,
CoordinateType const& Radius2,
IteratorType& Results,
DistanceIteratorType& ResultsDistances,
SizeType& NumberOfResults,
SizeType const& MaxNumberOfResults,
SearchStructureType& Auxiliar
) override
{
const CoordinateType temp = Auxiliar.residual_distance[mCutingDimension];
const CoordinateType distance_to_partition = ThisPoint[mCutingDimension] - mPosition;

if(distance_to_partition < 0) 
{
mpChilds[0]->SearchInRadius(ThisPoint, Radius, Radius2, Results, ResultsDistances, NumberOfResults, MaxNumberOfResults, Auxiliar );

Auxiliar.residual_distance[mCutingDimension] = distance_to_partition * distance_to_partition;
Auxiliar.distance_to_partition2 = Auxiliar.residual_distance[0];
for(SizeType i = 1; i < Dimension; i++)
Auxiliar.distance_to_partition2 += Auxiliar.residual_distance[i];
if( Radius2 >= Auxiliar.distance_to_partition2 )
mpChilds[1]->SearchInRadius(ThisPoint, Radius, Radius2, Results, ResultsDistances, NumberOfResults, MaxNumberOfResults, Auxiliar );
}
else 
{
mpChilds[1]->SearchInRadius(ThisPoint, Radius, Radius2, Results, ResultsDistances, NumberOfResults, MaxNumberOfResults, Auxiliar );

Auxiliar.residual_distance[mCutingDimension] = distance_to_partition * distance_to_partition;
Auxiliar.distance_to_partition2 = Auxiliar.residual_distance[0];
for(SizeType i = 1; i < Dimension; i++)
Auxiliar.distance_to_partition2 += Auxiliar.residual_distance[i];
if( Radius2 >= Auxiliar.distance_to_partition2 )
mpChilds[0]->SearchInRadius(ThisPoint, Radius, Radius2, Results, ResultsDistances, NumberOfResults, MaxNumberOfResults, Auxiliar );
}
Auxiliar.residual_distance[mCutingDimension] = temp;

}

void SearchInRadius(
PointType const& ThisPoint,
CoordinateType const& Radius,
CoordinateType const& Radius2,
IteratorType& Results,
SizeType& NumberOfResults,
SizeType const& MaxNumberOfResults
) override
{
SearchStructureType Auxiliar;
for(SizeType i = 0 ; i < Dimension; i++)
Auxiliar.residual_distance[i] = 0.00;
SearchInRadius(ThisPoint, Radius, Radius2, Results, NumberOfResults, MaxNumberOfResults, Auxiliar );
}

void SearchInRadius(
PointType const& ThisPoint,
CoordinateType const& Radius,
CoordinateType const& Radius2,
IteratorType& Results,
SizeType& NumberOfResults,
SizeType const& MaxNumberOfResults,
SearchStructureType& Auxiliar
) override
{
CoordinateType temp = Auxiliar.residual_distance[mCutingDimension];
CoordinateType distance_to_partition = ThisPoint[mCutingDimension] - mPosition;

if(distance_to_partition < 0) 
{
mpChilds[0]->SearchInRadius(ThisPoint, Radius, Radius2, Results, NumberOfResults, MaxNumberOfResults, Auxiliar );

Auxiliar.residual_distance[mCutingDimension] = distance_to_partition * distance_to_partition;
Auxiliar.distance_to_partition2 = Auxiliar.residual_distance[0];
for(SizeType i = 1; i < Dimension; i++)
Auxiliar.distance_to_partition2 += Auxiliar.residual_distance[i];
if( Radius2 >= Auxiliar.distance_to_partition2 )
mpChilds[1]->SearchInRadius(ThisPoint, Radius, Radius2, Results, NumberOfResults, MaxNumberOfResults, Auxiliar );
Auxiliar.residual_distance[mCutingDimension] = temp;
}
else 
{
mpChilds[1]->SearchInRadius(ThisPoint, Radius, Radius2, Results, NumberOfResults, MaxNumberOfResults, Auxiliar );

Auxiliar.residual_distance[mCutingDimension] = distance_to_partition * distance_to_partition;
Auxiliar.distance_to_partition2 = Auxiliar.residual_distance[0];
for(SizeType i = 1; i < Dimension; i++)
Auxiliar.distance_to_partition2 += Auxiliar.residual_distance[i];
if( Radius2 >= Auxiliar.distance_to_partition2 )
mpChilds[0]->SearchInRadius(ThisPoint, Radius, Radius2, Results, NumberOfResults, MaxNumberOfResults, Auxiliar );
Auxiliar.residual_distance[mCutingDimension] = temp;
}

}

void SearchInBox(
PointType const& SearchMinPoint,
PointType const& SearchMaxPoint,
IteratorType& Results,
SizeType& NumberOfResults,
SizeType const& MaxNumberOfResults
) override
{
if( SearchMinPoint[mCutingDimension] <= mPosition )
mpChilds[0]->SearchInBox(SearchMinPoint,SearchMaxPoint,Results,NumberOfResults,MaxNumberOfResults);
if( SearchMaxPoint[mCutingDimension] >= mPosition )
mpChilds[1]->SearchInBox(SearchMinPoint,SearchMaxPoint,Results,NumberOfResults,MaxNumberOfResults);
}





public:



private:

IndexType mCutingDimension;
CoordinateType mPosition;   
CoordinateType mLeftEnd;    
CoordinateType mRightEnd;   
TreeNodeType* mpChilds[2];  

};



template< class TLeafType >
class KDTreePartition : public KDTreePartitionBase<TLeafType>
{
public:

KRATOS_CLASS_POINTER_DEFINITION(KDTreePartition);

typedef KDTreePartitionBase<TLeafType> BaseType;

typedef TLeafType LeafType;

typedef typename LeafType::PointType PointType;

typedef typename LeafType::ContainerType ContainerType;

typedef typename LeafType::IteratorType IteratorType;

typedef typename LeafType::DistanceIteratorType DistanceIteratorType;

typedef typename LeafType::PointerType PointerType;

typedef typename LeafType::DistanceFunction DistanceFunction;

enum { Dimension = LeafType::Dimension };

typedef TreeNode<Dimension,PointType, PointerType, IteratorType, DistanceIteratorType> TreeNodeType;

typedef typename TreeNodeType::CoordinateType CoordinateType;

typedef typename TreeNodeType::SizeType SizeType;

typedef typename TreeNodeType::IndexType IndexType;

typedef typename LeafType::SearchStructureType SearchStructureType;



KDTreePartition( IndexType CutingDimension, CoordinateType Position, CoordinateType LeftEnd, CoordinateType RightEnd,
TreeNodeType* pLeftChild = NULL, TreeNodeType* pRightChild = NULL )
: BaseType(CutingDimension,Position,LeftEnd,RightEnd,pLeftChild,pRightChild) {}

~KDTreePartition() {}




static IteratorType Partition(
IteratorType PointsBegin,
IteratorType PointsEnd,
IndexType& rCuttingDimension,
CoordinateType& rCuttingValue
)
{
const SizeType n = SearchUtils::PointerDistance(PointsBegin, PointsEnd);
rCuttingDimension = MaxSpread(PointsBegin, PointsEnd);
IteratorType partition = PointsBegin + n / 2;

MedianSplit(PointsBegin, partition, PointsEnd, rCuttingDimension, rCuttingValue);

return partition;
}

static SizeType MaxSpread(
IteratorType PointsBegin,
IteratorType PointsEnd
)
{
SizeType max_dimension = 0;					
CoordinateType max_spread = 0;				




CoordinateType min[Dimension];
CoordinateType max[Dimension];
for (SizeType d = 0; d < Dimension; d++)
{
min[d] = (**PointsBegin)[d];
max[d] = (**PointsBegin)[d];
}
for (IteratorType i_point = PointsBegin; i_point != PointsEnd; i_point++)
{
for (SizeType d = 0; d < Dimension; d++)
{
CoordinateType c = (**i_point)[d];
if (c < min[d])
min[d] = c;
else if (c > max[d])
max[d] = c;
}
}
max_dimension = 0;
max_spread = max[0] - min[0];
for (SizeType d = 1; d < Dimension; d++)
{
CoordinateType spread = max[d] - min[d];
if (spread > max_spread)
{
max_spread = spread;
max_dimension = d;
}
}

return max_dimension;
}

static void MedianSplit(
IteratorType PointsBegin,
IteratorType PartitionPosition,
IteratorType PointsEnd,
IndexType CuttingDimension,
CoordinateType& rCuttingValue
)
{
IteratorType left  = PointsBegin;
IteratorType right = PointsEnd - 1;
while (left < right)   
{

IteratorType i = left; 
if ((**i)[CuttingDimension] > (**right)[CuttingDimension])
std::swap(*i,*right);

CoordinateType value = (**i)[CuttingDimension];
IteratorType j = right;
for(;;)
{
while ((**(++i))[CuttingDimension] < value)
;

while ((**(--j))[CuttingDimension] > value)
;
if (i < j) std::swap(*i,*j);
else break;
}
std::swap(*left,*j);

if (j > PartitionPosition)
right = j - 1;
else if (j < PartitionPosition)
left = j + 1;
else break;
}

CoordinateType max = (**PointsBegin)[CuttingDimension];
IteratorType k = PointsBegin;
IteratorType last = PartitionPosition;

for(IteratorType i = PointsBegin ; i != PartitionPosition ; ++i)
if((**i)[CuttingDimension] > max)
{
max = (**i)[CuttingDimension];
k = i;
}

if(k != PointsBegin)
std::swap(--last, k);

rCuttingValue = ((**last)[CuttingDimension] + (**PartitionPosition)[CuttingDimension])/2.0;
}

public:

static TreeNodeType* Construct(
IteratorType PointsBegin,
IteratorType PointsEnd,
const PointType& HighPoint,
const PointType& LowPoint,
SizeType BucketSize
)
{
SizeType number_of_points = SearchUtils::PointerDistance(PointsBegin,PointsEnd);
if (number_of_points == 0)
return NULL;
else if (number_of_points <= BucketSize)
{
return new LeafType(PointsBegin, PointsEnd);
}
else
{
IndexType cutting_dimension;
CoordinateType cutting_value;

IteratorType partition = Partition(PointsBegin, PointsEnd, cutting_dimension, cutting_value);

PointType partition_high_point;
partition_high_point.Coordinates() =  HighPoint.Coordinates();
PointType partition_low_point;
partition_low_point.Coordinates() =  LowPoint.Coordinates();

partition_high_point[cutting_dimension] = cutting_value;
partition_low_point[cutting_dimension] = cutting_value;

return new KDTreePartition( cutting_dimension, cutting_value,
HighPoint[cutting_dimension], LowPoint[cutting_dimension],
Construct(PointsBegin, partition, partition_high_point, LowPoint, BucketSize),
Construct(partition, PointsEnd, HighPoint, partition_low_point, BucketSize) );

}
}

};



template< class TLeafType >
class KDTreePartitionAverageSplit : public KDTreePartitionBase<TLeafType>
{
public:

KRATOS_CLASS_POINTER_DEFINITION(KDTreePartitionAverageSplit);

typedef KDTreePartitionBase<TLeafType> BaseType;

typedef TLeafType LeafType;

typedef typename LeafType::PointType PointType;

typedef typename LeafType::ContainerType ContainerType;

typedef typename LeafType::IteratorType IteratorType;

typedef typename LeafType::DistanceIteratorType DistanceIteratorType;

typedef typename LeafType::PointerType PointerType;

typedef typename LeafType::DistanceFunction DistanceFunction;

enum { Dimension = LeafType::Dimension };

typedef TreeNode<Dimension,PointType, PointerType, IteratorType, DistanceIteratorType> TreeNodeType;

typedef typename TreeNodeType::CoordinateType CoordinateType;

typedef typename TreeNodeType::SizeType SizeType;

typedef typename TreeNodeType::IndexType IndexType;

typedef typename LeafType::SearchStructureType SearchStructureType;



KDTreePartitionAverageSplit(
IndexType CutingDimension,
CoordinateType Position,
CoordinateType LeftEnd,
CoordinateType RightEnd,
TreeNodeType* pLeftChild = NULL,
TreeNodeType* pRightChild = NULL
)
: BaseType(CutingDimension,Position,LeftEnd,RightEnd,pLeftChild,pRightChild) 
{
}

virtual ~KDTreePartitionAverageSplit() {}




static IteratorType Partition(
IteratorType PointsBegin,
IteratorType PointsEnd,
IndexType& rCuttingDimension,
CoordinateType& rCuttingValue
)
{

rCuttingDimension = MaxSpread( PointsBegin, PointsEnd, rCuttingValue );

IteratorType partition;
AverageSplit(PointsBegin, partition, PointsEnd, rCuttingDimension, rCuttingValue);

return partition;

}

static SizeType MaxSpread(
IteratorType PointsBegin,
IteratorType PointsEnd,
CoordinateType& AverageValue
)
{
SizeType max_dimension = 0;					
CoordinateType max_spread = 0;				
AverageValue = 0.0;

CoordinateType size = static_cast<CoordinateType>(SearchUtils::PointerDistance(PointsBegin,PointsEnd));

CoordinateType min[Dimension];
CoordinateType max[Dimension];
CoordinateType Average[Dimension];
for (SizeType d = 0; d < Dimension; d++)
{
Average[d] = 0.0;
min[d] = (**PointsBegin)[d];
max[d] = (**PointsBegin)[d];
}
for (IteratorType i_point = PointsBegin; i_point != PointsEnd; i_point++)
{
for (SizeType d = 0; d < Dimension; d++)
{
CoordinateType c = (**i_point)[d];
Average[d] += c;
if (c < min[d])
min[d] = c;
else if (c > max[d])
max[d] = c;
}
}
max_dimension = 0;
max_spread = max[0] - min[0];
AverageValue = Average[0] / size;
for (SizeType d = 1; d < Dimension; d++)
{
CoordinateType spread = max[d] - min[d];
if (spread > max_spread)
{
max_spread = spread;
max_dimension = d;
AverageValue = Average[d] / size;
}
}

return max_dimension;
}


static void AverageSplit(
IteratorType PointsBegin,
IteratorType& PartitionPosition,
IteratorType PointsEnd,
IndexType& CuttingDimension,
CoordinateType& rCuttingValue
)
{
IteratorType left  = PointsBegin;
IteratorType right = PointsEnd - 1;
for(;;)
{
while( left  < PointsEnd   && (**left)[CuttingDimension]  <  rCuttingValue ) left++;
while( right > PointsBegin && (**right)[CuttingDimension] >= rCuttingValue ) right--;
if (left <= right) std::swap(*left,*right);
else break;
}

PartitionPosition = left;
}

public:
static TreeNodeType* Construct(
IteratorType PointsBegin,
IteratorType PointsEnd,
PointType HighPoint,
PointType LowPoint,
SizeType BucketSize
)
{
SizeType number_of_points = SearchUtils::PointerDistance(PointsBegin,PointsEnd);
if (number_of_points == 0)
return NULL;
else if (number_of_points <= BucketSize)
{
return new LeafType(PointsBegin, PointsEnd);
}
else
{
IndexType cutting_dimension;
CoordinateType cutting_value;

IteratorType partition = Partition(PointsBegin, PointsEnd, cutting_dimension, cutting_value);

PointType partition_high_point = HighPoint;
PointType partition_low_point = LowPoint;

partition_high_point[cutting_dimension] = cutting_value;
partition_low_point[cutting_dimension] = cutting_value;

return new KDTreePartitionAverageSplit( cutting_dimension, cutting_value,
HighPoint[cutting_dimension], LowPoint[cutting_dimension],
Construct(PointsBegin, partition, partition_high_point, LowPoint, BucketSize),
Construct(partition, PointsEnd, HighPoint, partition_low_point, BucketSize) );

}
}

};


template< class TLeafType >
class KDTreePartitionMidPointSplit : public KDTreePartitionBase<TLeafType>
{
public:

KRATOS_CLASS_POINTER_DEFINITION(KDTreePartitionMidPointSplit);

typedef KDTreePartitionBase<TLeafType> BaseType;

typedef TLeafType LeafType;

typedef typename LeafType::PointType PointType;

typedef typename LeafType::ContainerType ContainerType;

typedef typename LeafType::IteratorType IteratorType;

typedef typename LeafType::DistanceIteratorType DistanceIteratorType;

typedef typename LeafType::PointerType PointerType;

typedef typename LeafType::DistanceFunction DistanceFunction;

enum { Dimension = LeafType::Dimension };

typedef TreeNode<Dimension,PointType, PointerType, IteratorType, DistanceIteratorType> TreeNodeType;

typedef typename TreeNodeType::CoordinateType CoordinateType;

typedef typename TreeNodeType::SizeType SizeType;

typedef typename TreeNodeType::IndexType IndexType;

typedef typename LeafType::SearchStructureType SearchStructureType;


KDTreePartitionMidPointSplit( IndexType CutingDimension, CoordinateType Position, CoordinateType LeftEnd, CoordinateType RightEnd,
TreeNodeType* pLeftChild = NULL, TreeNodeType* pRightChild = NULL)
: BaseType(CutingDimension,Position,LeftEnd,RightEnd,pLeftChild,pRightChild) {}

virtual ~KDTreePartitionMidPointSplit() {}



static IteratorType Partition(
IteratorType PointsBegin,
IteratorType PointsEnd,
PointType const& HighPoint,
PointType const& LowPoint,
IndexType& rCuttingDimension,
CoordinateType& rCuttingValue
)
{
rCuttingDimension = MaxSpread( PointsBegin, PointsEnd, HighPoint, LowPoint, rCuttingValue );

return Split(PointsBegin, PointsEnd, rCuttingDimension, rCuttingValue);
}

static SizeType MaxSpread(
IteratorType PointsBegin,
IteratorType PointsEnd,
PointType const& HighPoint,
PointType const& LowPoint,
CoordinateType& CuttingValue
)
{


CoordinateType min[Dimension];
CoordinateType max[Dimension];
for (SizeType d = 0; d < Dimension; d++)
{
min[d] = (**PointsBegin)[d];
max[d] = (**PointsBegin)[d];
}
for (IteratorType i_point = PointsBegin; i_point != PointsEnd; i_point++)
for (SizeType d = 0; d < Dimension; d++)
{
CoordinateType c = (**i_point)[d];
if (c < min[d])
min[d] = c;
else if (c > max[d])
max[d] = c;
}
SizeType max_dimension = 0;
CoordinateType max_spread = max[0] - min[0];
for (SizeType d = 1; d < Dimension; d++)
{
CoordinateType spread = max[d] - min[d];
if (spread > max_spread)
{
max_spread = spread;
max_dimension = d;
}
}
CuttingValue = (max[max_dimension]+min[max_dimension]) / 2.00;

return max_dimension;
}

static IteratorType Split(
IteratorType PointsBegin,
IteratorType PointsEnd,
IndexType& CuttingDimension,
CoordinateType& rCuttingValue
)
{
IteratorType left  = PointsBegin;
IteratorType right = PointsEnd - 1;
for(;;)
{
while( (**left)[CuttingDimension] < rCuttingValue ) left++;
while( (**right)[CuttingDimension] >= rCuttingValue ) right--;
if (left < right) std::swap(*left,*right);
else break;
}

return left;
}

public:
static TreeNodeType* Construct(
IteratorType PointsBegin,
IteratorType PointsEnd,
PointType HighPoint,
PointType LowPoint,
SizeType BucketSize
)
{
SizeType number_of_points = SearchUtils::PointerDistance(PointsBegin,PointsEnd);
if (number_of_points == 0)
return NULL;
else if (number_of_points <= BucketSize)
{
return new LeafType(PointsBegin, PointsEnd);
}
else
{
IndexType cutting_dimension;
CoordinateType cutting_value;

IteratorType partition = Partition(PointsBegin, PointsEnd, HighPoint, LowPoint, cutting_dimension, cutting_value);

PointType partition_high_point = HighPoint;
PointType partition_low_point = LowPoint;

partition_high_point[cutting_dimension] = cutting_value;
partition_low_point[cutting_dimension] = cutting_value;

return new KDTreePartitionMidPointSplit( cutting_dimension, cutting_value,
HighPoint[cutting_dimension], LowPoint[cutting_dimension],
Construct(PointsBegin, partition, partition_high_point, LowPoint, BucketSize),
Construct(partition, PointsEnd, HighPoint, partition_low_point, BucketSize) );

}
}

};

}  