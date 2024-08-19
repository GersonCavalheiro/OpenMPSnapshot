
#pragma once

#include <string>
#include <iostream>
#include <cmath>


#include "search_structure.h"
#include "utilities/parallel_utilities.h"

namespace Kratos
{






template<
std::size_t TDimension,
class TPointType,
class TPointerType,
class TIteratorType,
class TDistanceIteratorType,
class TIteratorIteratorType = typename std::vector<TIteratorType>::iterator
>
class TreeNode
{
public:


KRATOS_CLASS_POINTER_DEFINITION(TreeNode);

typedef std::size_t SizeType;
typedef std::size_t IndexType;
typedef double CoordinateType;

typedef TPointType PointType;
typedef TPointerType PointerType;
typedef TIteratorType IteratorType;
typedef TDistanceIteratorType DistanceIteratorType;

typedef TreeNode<TDimension,TPointType,TPointerType,TIteratorType,TDistanceIteratorType> TreeNodeType;

typedef typename std::vector<IteratorType>::iterator IteratorIteratorType;

typedef SearchStructure<IndexType,SizeType,CoordinateType,TIteratorType,IteratorIteratorType,TDimension> SearchStructureType;

virtual void PrintData(std::ostream& rOStream, std::string const& Perfix = std::string()) const {}

TreeNode() {}

virtual ~TreeNode() {}

virtual void SearchNearestPoint(PointType const& ThisPoint, PointerType& rResult, CoordinateType& rResultDistance) {}

virtual void SearchNearestPoint(PointType const& ThisPoint, PointerType& rResult, CoordinateType& rResultDistance,
SearchStructureType& Auxiliar) {}

virtual void SearchInRadius(PointType const& ThisPoint, CoordinateType const& Radius, CoordinateType const& Radius2, IteratorType& Results,
DistanceIteratorType& ResultsDistances, SizeType& NumberOfResults, SizeType const& MaxNumberOfResults)
{
return;
}

virtual void SearchInRadius(PointType const& ThisPoint, CoordinateType const& Radius, CoordinateType const& Radius2, IteratorType& Results,
DistanceIteratorType& ResultsDistances, SizeType& NumberOfResults, SizeType const& MaxNumberOfResults, SearchStructureType& Auxiliar)
{
return;
}

virtual void SearchInRadius(PointType const& ThisPoint, CoordinateType const& Radius, CoordinateType const& Radius2, IteratorType& Results,
SizeType& NumberOfResults, SizeType const& MaxNumberOfResults)
{
return;
}

virtual void SearchInRadius(PointType const& ThisPoint, CoordinateType const& Radius, CoordinateType const& Radius2, IteratorType& Results,
SizeType& NumberOfResults, SizeType const& MaxNumberOfResults, SearchStructureType& Auxiliar)
{
return;
}

virtual void SearchInBox(PointType const& SearchMinPoint, PointType const& SearchMaxPoint, IteratorType& Results, SizeType& NumberOfResults,
SizeType const& MaxNumberOfResults ) 
{
return;
}


static IteratorType& NullIterator()
{
return msNull;
}

static PointerType& NullPointer()
{
return msNullPointer;
}

static TreeNode& NullLeaf()
{
return msNullLeaf;
}

private:
static IteratorType msNull;
static PointerType msNullPointer;
static TreeNode msNullLeaf;
};

template<std::size_t TDimension, class TPointType, class TPointerType, class TIteratorType, class TDistanceIteratorType, class TIteratorIteratorType>
typename TreeNode<TDimension, TPointType, TPointerType, TIteratorType, TDistanceIteratorType, TIteratorIteratorType>::IteratorType
TreeNode<TDimension, TPointType, TPointerType, TIteratorType, TDistanceIteratorType, TIteratorIteratorType>::msNull;

template<std::size_t TDimension, class TPointType, class TPointerType, class TIteratorType, class TDistanceIteratorType, class TIteratorIteratorType>
typename TreeNode<TDimension, TPointType, TPointerType, TIteratorType, TDistanceIteratorType, TIteratorIteratorType>::PointerType
TreeNode<TDimension, TPointType, TPointerType, TIteratorType, TDistanceIteratorType, TIteratorIteratorType>::msNullPointer;

template<std::size_t TDimension, class TPointType, class TPointerType, class TIteratorType, class TDistanceIteratorType, class TIteratorIteratorType>
TreeNode<TDimension, TPointType, TPointerType, TIteratorType, TDistanceIteratorType, TIteratorIteratorType>
TreeNode<TDimension, TPointType, TPointerType, TIteratorType, TDistanceIteratorType, TIteratorIteratorType>::msNullLeaf;


template< class TPartitionType >
class Tree
{
public:

class Partitions
{
public:
explicit Partitions( const std::size_t NumPartitions ) : mNumPartitions(NumPartitions) {}
~Partitions() {};
std::size_t mNumPartitions;
};

KRATOS_CLASS_POINTER_DEFINITION(Tree);
using PartitionType = TPartitionType;

using LeafType = typename PartitionType::LeafType;

using PointType = typename PartitionType::PointType;

using IteratorType = typename PartitionType::IteratorType;

using DistanceIteratorType = typename PartitionType::DistanceIteratorType;

using PointerType = typename PartitionType::PointerType;

using DistanceFunction = typename PartitionType::DistanceFunction;

static constexpr std::size_t Dimension = PartitionType::Dimension;

using NodeType = TreeNode<Dimension,PointType,PointerType,IteratorType,DistanceIteratorType> ;

using CoordinateType = typename NodeType::CoordinateType;

using SizeType = typename NodeType::SizeType;

using IndexType = typename NodeType::IndexType;

using SearchStructureType = typename PartitionType::SearchStructureType;



Tree(
IteratorType PointsBegin,
IteratorType PointsEnd,
SizeType BucketSize = 1
) : mBucketSize(BucketSize),
mPointsBegin(PointsBegin),
mPointsEnd(PointsEnd)
{
if(mPointsBegin == mPointsEnd)
return;

for(SizeType i = 0 ; i < Dimension ; i++) {
mBoundingBoxHighPoint[i] = (**mPointsBegin)[i];
mBoundingBoxLowPoint[i] = (**mPointsBegin)[i];
}

for(IteratorType point_iterator = mPointsBegin ; point_iterator != mPointsEnd ; point_iterator++) {
for(SizeType i = 0 ; i < Dimension ; i++) {
if((**point_iterator)[i] > mBoundingBoxHighPoint[i]) {
mBoundingBoxHighPoint[i] = (**point_iterator)[i];
} else if((**point_iterator)[i] < mBoundingBoxLowPoint[i]) {
mBoundingBoxLowPoint[i]  = (**point_iterator)[i];
}
}
}

mRoot = TPartitionType::Construct(mPointsBegin, mPointsEnd, mBoundingBoxHighPoint, mBoundingBoxLowPoint, mBucketSize);
}


Tree(
IteratorType PointsBegin,
IteratorType PointsEnd,
Partitions Parts
) : mPointsBegin(PointsBegin), 
mPointsEnd(PointsEnd)
{
if(mPointsBegin == mPointsEnd)
return;

SizeType NumPoints = SearchUtils::PointerDistance(mPointsBegin,mPointsEnd);
mBucketSize = static_cast<std::size_t>( (double) NumPoints / (double) Parts.mNumPartitions ) + 1;

mBoundingBoxHighPoint = **mPointsBegin;
mBoundingBoxLowPoint = **mPointsBegin;
for(IteratorType point_iterator = mPointsBegin ; point_iterator != mPointsEnd ; point_iterator++) {
for(SizeType i = 0 ; i < Dimension ; i++) {
if((**point_iterator)[i] > mBoundingBoxHighPoint[i]) {
mBoundingBoxHighPoint[i] = (**point_iterator)[i];
} else if((**point_iterator)[i] < mBoundingBoxLowPoint[i]) {
mBoundingBoxLowPoint[i] = (**point_iterator)[i];
}
}
}

mRoot = TPartitionType::Construct(mPointsBegin, mPointsEnd, mBoundingBoxHighPoint, mBoundingBoxLowPoint, mBucketSize);
}

virtual ~Tree()
{
delete mRoot;
}



PointerType ExistPoint(
PointerType const& ThisPoint,
CoordinateType const Tolerance = static_cast<CoordinateType>(10.0*DBL_EPSILON) 
)
{
PointerType Result = *mPointsBegin;
CoordinateType ResultDistance = static_cast<CoordinateType>(DBL_MAX);
mRoot->SearchNearestPoint(*ThisPoint, Result, ResultDistance);
if (ResultDistance<Tolerance*Tolerance)
return Result;
return NodeType::NullPointer();
}

PointerType SearchNearestPoint(
PointType const& ThisPoint,
CoordinateType& rResultDistance
)
{
PointerType Result = *mPointsBegin;
rResultDistance = static_cast<CoordinateType>(DBL_MAX); 

mRoot->SearchNearestPoint(ThisPoint, Result, rResultDistance);

return Result;
}

PointerType SearchNearestPoint(PointType const& ThisPoint)
{
PointerType Result = *mPointsBegin; 
CoordinateType rResultDistance = static_cast<CoordinateType>(DBL_MAX); 

mRoot->SearchNearestPoint(ThisPoint, Result, rResultDistance);

return Result;
}

SizeType SearchInRadius(
PointType const& ThisPoint,
CoordinateType Radius,
IteratorType Results,
DistanceIteratorType ResultsDistances,
SizeType MaxNumberOfResults
)
{
const CoordinateType Radius2 = Radius * Radius;

SizeType NumberOfResults = 0;
mRoot->SearchInRadius(ThisPoint, Radius, Radius2, Results, ResultsDistances, NumberOfResults, MaxNumberOfResults);

return NumberOfResults;
}

SizeType SearchInRadius(
PointType const& ThisPoint,
CoordinateType Radius,
IteratorType Results,
SizeType MaxNumberOfResults
)
{
const CoordinateType Radius2 = Radius * Radius;

SizeType NumberOfResults = 0;
mRoot->SearchInRadius(ThisPoint, Radius, Radius2, Results, NumberOfResults, MaxNumberOfResults);
return NumberOfResults;
}

SizeType SearchInBox(
PointType const& MinPointBox,
PointType const& MaxPointBox,
IteratorType Results,
SizeType MaxNumberOfResults
)
{
SizeType NumberOfResults = 0;
mRoot->SearchInBox(MinPointBox,MaxPointBox,Results,NumberOfResults,MaxNumberOfResults);
return NumberOfResults;
}


PointType& BoundingBoxLowPoint()
{
return mBoundingBoxLowPoint;
}

PointType& BoundingBoxHighPoint()
{
return mBoundingBoxHighPoint;
}



virtual std::string Info() const
{
return "Tree";
}

virtual void PrintInfo(std::ostream& rOStream) const
{
rOStream << "Tree";
}

virtual void PrintData(std::ostream& rOStream, std::string const& Perfix = std::string()) const
{
mRoot->PrintData(rOStream, "  ");
}


protected:







private:

static LeafType msEmptyLeaf;


SizeType mBucketSize;

PointType mBoundingBoxLowPoint;
PointType mBoundingBoxHighPoint;

IteratorType mPointsBegin;
IteratorType mPointsEnd;

NodeType* mRoot;






Tree& operator=(Tree const& rOther);

Tree(Tree const& rOther);

}; 

template< class TPartitionType >
typename Tree<TPartitionType>::LeafType Tree<TPartitionType>::msEmptyLeaf;



template<class TPartitionType>
inline std::istream& operator >> (std::istream& rIStream, Tree<TPartitionType>& rThis);

template<class TPartitionType>
inline std::ostream& operator << (std::ostream& rOStream, const Tree<TPartitionType>& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}

}  