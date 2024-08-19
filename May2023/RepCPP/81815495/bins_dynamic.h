
#pragma once

#include <array>
#include <cmath>
#include <algorithm>


#include "tree.h"
#include "utilities/parallel_utilities.h"

namespace Kratos
{

template<
std::size_t TDimension,
class TPointType,
class TContainerType,
class TPointerType = typename TContainerType::value_type,
class TIteratorType = typename TContainerType::iterator,
class TDistanceIteratorType = typename std::vector<double>::iterator,
class TDistanceFunction = Kratos::SearchUtils::SquaredDistanceFunction<TDimension,TPointType>
>
class BinsDynamic : public TreeNode<TDimension,TPointType, TPointerType, TIteratorType, TDistanceIteratorType, typename std::vector<TPointerType>::iterator >
{

public:

KRATOS_CLASS_POINTER_DEFINITION(BinsDynamic);

enum { Dimension = TDimension };

typedef TPointType                                  PointType;
typedef TContainerType                              ContainerType;
typedef TIteratorType                               IteratorType;
typedef TDistanceIteratorType                       DistanceIteratorType;
typedef TPointerType                                PointerType;
typedef TDistanceFunction                           DistanceFunction;

typedef TreeNode<Dimension,TPointType,TPointerType,TIteratorType,TDistanceIteratorType> TreeNodeType;

typedef typename TreeNodeType::CoordinateType       CoordinateType;  
typedef typename TreeNodeType::SizeType             SizeType;        
typedef typename TreeNodeType::IndexType            IndexType;       

typedef Tvector<CoordinateType,Dimension>           CoordinateArray;
typedef Tvector<SizeType,Dimension>                 SizeArray;
typedef Tvector<IndexType,Dimension>                IndexArray;

typedef typename TreeNodeType::IteratorIteratorType IteratorIteratorType;
typedef typename TreeNodeType::SearchStructureType  SearchStructureType;

typedef std::vector<PointerType>                    LocalContainerType;
typedef typename LocalContainerType::iterator       LocalIterator;

typedef Tvector<IndexType,Dimension>               CellType;
typedef std::vector<LocalContainerType>             CellContainerType;

typedef Kratos::SearchUtils::SearchNearestInRange<PointType,PointerType,LocalIterator,DistanceFunction,CoordinateType> SearchNearestInRange;
typedef Kratos::SearchUtils::SearchRadiusInRange<PointType,LocalIterator,DistanceIteratorType,DistanceFunction,SizeType,CoordinateType,IteratorType> SearchRadiusInRange;
typedef Kratos::SearchUtils::SearchBoxInRange<PointType,LocalIterator,SizeType,Dimension,IteratorType> SearchBoxInRange;

typedef std::vector<CoordinateType>                 CoordinateVectorType;
typedef std::vector<IteratorType>                   IteratorVectorType;
typedef std::vector<DistanceIteratorType>           DistanceIteratorVectorType;

typedef LocalContainerType                          PointVector;
typedef LocalIterator                               PointIterator;
typedef TreeNodeType                                LeafType;

public:

/
CellContainerType& GetCellContainer() {
return mCells;
}


SizeArray& GetDivisions() {
return mN;
}


CoordinateArray& GetCellSize() {
return mCellSize;
}


PointType& GetMinPoint() {
return mMinPoint;
}


PointType& GetMaxPoint() {
return mMaxPoint;
}

/
void CalculateCellSize(std::size_t ApproximatedSize)
{
std::size_t average_number_of_cells = static_cast<std::size_t>(std::pow(static_cast<double>(ApproximatedSize), 1.00 / Dimension));

std::array<double, 3> lengths;
double average_length = 0.00;

for (int i = 0; i < Dimension; i++) {
lengths[i] = mMaxPoint[i] - mMinPoint[i];
average_length += lengths[i];
}
average_length *= 1.00 / 3.00;

if (average_length < std::numeric_limits<double>::epsilon()) {
for(int i = 0; i < Dimension; i++) {
mN[i] = 1;
}
return;
}

for (int i = 0; i < Dimension; i++) {
mN[i] = static_cast<std::size_t>(lengths[i] / average_length * (double)average_number_of_cells) + 1;

if (mN[i] > 1) {
mCellSize[i] = lengths[i] / mN[i];
} else {
mCellSize[i] = average_length;
}

mInvCellSize[i] = 1.00 / mCellSize[i];
}
}


void AssignCellSize( CoordinateType BoxSize )
{
for(SizeType i = 0 ; i < Dimension ; i++)
{
mCellSize[i] = BoxSize;
mInvCellSize[i] = 1.00 / mCellSize[i];
mN[i] = static_cast<SizeType>( (mMaxPoint[i]-mMinPoint[i]) / mCellSize[i]) + 1;
}
}


void AllocateCellsContainer()
{
SizeType Size = 1;
for(SizeType i = 0 ; i < Dimension ; i++)
Size *= mN[i];
mCells.resize(Size);
}


void GenerateBins()
{

for(IteratorType i_point = mPointBegin ; i_point != mPointEnd ; i_point++)
mCells[CalculateIndex(**i_point)].push_back(*i_point);

}


IndexType CalculatePosition( CoordinateType const& ThisCoord, SizeType ThisDimension )
{
CoordinateType d_index = (ThisCoord - mMinPoint[ThisDimension]) * mInvCellSize[ThisDimension];
IndexType index = static_cast<IndexType>( (d_index < 0.00) ? 0.00 : d_index );
return  (index > mN[ThisDimension]-1) ? mN[ThisDimension]-1 : index;
}


IndexType CalculateIndex( PointType const& ThisPoint )
{
IndexType Index = 0;
for(SizeType iDim = Dimension-1 ; iDim > 0 ; iDim--)
{
Index += CalculatePosition(ThisPoint[iDim],iDim);
Index *= mN[iDim-1];
}
Index += CalculatePosition(ThisPoint[0],0);
return Index;
}


IndexType CalculateIndex( CellType const& ThisIndex )
{
IndexType Index = 0;
for(SizeType iDim = Dimension-1 ; iDim > 0 ; iDim--)
{
Index += ThisIndex[iDim];
Index *= mN[iDim-1];
}
Index += ThisIndex[0];
return Index;
}


CellType CalculateCell( PointType const& ThisPoint )
{
CellType Cell;
for(SizeType i = 0 ; i < Dimension ; i++)
Cell[i] = CalculatePosition(ThisPoint[i],i);
return Cell;
}

CellType CalculateCell( PointType const& ThisPoint, CoordinateType Radius )
{
CellType Cell;
for(SizeType i = 0 ; i < Dimension ; i++)
Cell[i] = CalculatePosition(ThisPoint[i]+Radius,i);
return Cell;
}


void AddPoint( PointerType const& ThisPoint )
{
mCells[CalculateIndex(*ThisPoint)].push_back(ThisPoint);
mNumCells++;
}


PointerType ExistPoint( PointerType const& ThisPoint, CoordinateType const Tolerance = static_cast<CoordinateType>(10.0*DBL_EPSILON) )
{
PointerType Nearest;
CoordinateType Distance = static_cast<CoordinateType>(DBL_MAX);
bool Found;
SearchStructureType Box( CalculateCell(*ThisPoint,-Tolerance), CalculateCell(*ThisPoint,Tolerance), mN );
SearchNearestInBox( *ThisPoint, Nearest, Distance, Box, Found );
if(Found)
return Nearest;
return this->NullPointer();
}


PointerType SearchNearestPoint( PointType const& ThisPoint )
{
if( mPointBegin == mPointEnd )
return this->NullPointer();

PointerType Result            = *mPointBegin;
CoordinateType ResultDistance = static_cast<CoordinateType>(DBL_MAX);
SearchStructureType Box( CalculateCell(ThisPoint), mN );
SearchNearestPointLocal( ThisPoint, Result, ResultDistance, Box );
return Result;
}


PointerType SearchNearestPoint( PointType const& ThisPoint, CoordinateType& ResultDistance )
{
if( mPointBegin == mPointEnd )
return this->NullPointer();

PointerType Result = *mPointBegin;
ResultDistance     = static_cast<CoordinateType>(DBL_MAX);
SearchStructureType Box( CalculateCell(ThisPoint), mN );
SearchNearestPointLocal( ThisPoint, Result, ResultDistance, Box);
return Result;
}


PointerType SearchNearestPoint( PointType const& ThisPoint, CoordinateType& rResultDistance, SearchStructureType& Box )
{
PointerType Result = *mPointBegin; 
rResultDistance    = static_cast<CoordinateType>(DBL_MAX);
Box.Set( CalculateCell(ThisPoint), mN );
SearchNearestPointLocal( ThisPoint, Result, rResultDistance, Box);
return Result;
}


void SearchNearestPoint( PointType const& ThisPoint, PointerType& rResult, CoordinateType& rResultDistance ) override
{
SearchStructureType Box;
Box.Set( CalculateCell(ThisPoint), mN );
SearchNearestPointLocal(ThisPoint,rResult,rResultDistance,Box);
}


void SearchNearestPoint( PointType const& ThisPoint, PointerType& rResult, CoordinateType& rResultDistance, SearchStructureType& Box ) override
{
Box.Set( CalculateCell(ThisPoint), mN );
SearchNearestPointLocal( ThisPoint, rResult, rResultDistance, Box );
}


void SearchNearestPoint( PointType* const& ThisPoints, SizeType const& NumberOfPoints, IteratorType &Results, std::vector<CoordinateType> ResultsDistances)
{
IndexPartition<SizeType>(NumberOfPoints).for_each(
[&](SizeType iPoint)
{ Results[iPoint] = SearchNearestPoint(ThisPoints[iPoint],ResultsDistances[iPoint]); }
);
}


void SearchNearestPointLocal( PointType const& ThisPoint, PointerType& rResult, CoordinateType& rResultDistance, SearchStructureType& Box )
{
if( mPointBegin == mPointEnd )
return;

bool Found = false;

Box.Set( CalculateCell(ThisPoint), mN );

++Box;
SearchNearestInBox( ThisPoint, rResult, rResultDistance, Box, Found );
while(!Found)
{
++Box;
SearchNearestInBox( ThisPoint, rResult, rResultDistance, Box, Found );
}

}


SizeType SearchInRadius( PointType const& ThisPoint, CoordinateType const& Radius, IteratorType Results,
DistanceIteratorType ResultsDistances, SizeType const& MaxNumberOfResults )
{
CoordinateType Radius2 = Radius * Radius;
SizeType NumberOfResults = 0;
SearchStructureType Box( CalculateCell(ThisPoint,-Radius), CalculateCell(ThisPoint,Radius), mN );
SearchInRadiusLocal( ThisPoint, Radius, Radius2, Results, ResultsDistances, NumberOfResults, MaxNumberOfResults, Box );
return NumberOfResults;
}


SizeType SearchInRadius( PointType const& ThisPoint, CoordinateType const& Radius, IteratorType Results,
DistanceIteratorType ResultsDistances, SizeType const& MaxNumberOfResults, SearchStructureType& Box )
{
CoordinateType Radius2 = Radius * Radius;
SizeType NumberOfResults = 0;
Box.Set( CalculateCell(ThisPoint,-Radius), CalculateCell(ThisPoint,Radius), mN );
SearchInRadiusLocal( ThisPoint, Radius, Radius2, Results, ResultsDistances, NumberOfResults, MaxNumberOfResults, Box );
return NumberOfResults;
}


void SearchInRadius( PointType const& ThisPoint, CoordinateType const& Radius, CoordinateType const& Radius2, IteratorType& Results,
DistanceIteratorType& ResultsDistances, SizeType& NumberOfResults, SizeType const& MaxNumberOfResults ) override
{
SearchStructureType Box( CalculateCell(ThisPoint,-Radius), CalculateCell(ThisPoint,Radius), mN );
SearchInRadiusLocal( ThisPoint, Radius, Radius2, Results, ResultsDistances, NumberOfResults, MaxNumberOfResults, Box);
}


void SearchInRadius( PointType const& ThisPoint, CoordinateType const& Radius, CoordinateType const& Radius2, IteratorType& Results,
DistanceIteratorType& ResultsDistances, SizeType& NumberOfResults, SizeType const& MaxNumberOfResults, SearchStructureType& Box ) override
{
Box.Set( CalculateCell(ThisPoint,-Radius), CalculateCell(ThisPoint,Radius), mN );
SearchInRadiusLocal( ThisPoint, Radius, Radius2, Results, ResultsDistances, NumberOfResults, MaxNumberOfResults, Box);
}


void SearchInRadius( PointerType const& ThisPoints, SizeType const& NumberOfPoints, CoordinateVectorType const& Radius, IteratorVectorType Results,
DistanceIteratorVectorType ResultsDistances, std::vector<SizeType>& NumberOfResults, SizeType const& MaxNumberOfResults )
{
IndexPartition<SizeType>(NumberOfPoints).for_each(
[&](SizeType iPoint)
{ NumberOfResults[iPoint] = SearchInRadius(ThisPoints[iPoint],Radius[iPoint],Results[iPoint],ResultsDistances[iPoint],MaxNumberOfResults); }
);
}


void SearchInRadiusLocal( PointType const& ThisPoint, CoordinateType const& Radius, CoordinateType const& Radius2, IteratorType& Results,
DistanceIteratorType& ResultsDistances, SizeType& NumberOfResults, SizeType const& MaxNumberOfResults,
SearchStructure<IndexType,SizeType,CoordinateType,IteratorType,IteratorIteratorType,1>& Box )
{
for(IndexType I = Box.Axis[0].Begin() ; I <= Box.Axis[0].End() ; I += Box.Axis[0].Block )
SearchRadiusInRange()(mCells[I].begin(),mCells[I].end(),ThisPoint,Radius2,Results,ResultsDistances,NumberOfResults,MaxNumberOfResults);
}

void SearchInRadiusLocal( PointType const& ThisPoint, CoordinateType const& Radius, CoordinateType const& Radius2, IteratorType& Results,
DistanceIteratorType& ResultsDistances, SizeType& NumberOfResults, SizeType const& MaxNumberOfResults,
SearchStructure<IndexType,SizeType,CoordinateType,IteratorType,IteratorIteratorType,2>& Box )
{
for(IndexType II = Box.Axis[1].Begin() ; II <= Box.Axis[1].End() ; II += Box.Axis[1].Block )
for(IndexType I = II + Box.Axis[0].Begin() ; I <= II + Box.Axis[0].End() ; I += Box.Axis[0].Block )
SearchRadiusInRange()(mCells[I].begin(),mCells[I].end(),ThisPoint,Radius2,Results,ResultsDistances,NumberOfResults,MaxNumberOfResults);
}

void SearchInRadiusLocal( PointType const& ThisPoint, CoordinateType const& Radius, CoordinateType const& Radius2, IteratorType& Results,
DistanceIteratorType& ResultsDistances, SizeType& NumberOfResults, SizeType const& MaxNumberOfResults,
SearchStructure<IndexType,SizeType,CoordinateType,IteratorType,IteratorIteratorType,3>& Box )
{
for(IndexType III = Box.Axis[2].Begin() ; III <= Box.Axis[2].End() ; III += Box.Axis[2].Block )
for(IndexType II = III + Box.Axis[1].Begin() ; II <= III + Box.Axis[1].End() ; II += Box.Axis[1].Block )
for(IndexType I = II + Box.Axis[0].Begin() ; I <= II + Box.Axis[0].End() ; I += Box.Axis[0].Block )
SearchRadiusInRange()(mCells[I].begin(),mCells[I].end(),ThisPoint,Radius2,Results,ResultsDistances,NumberOfResults,MaxNumberOfResults);
}


SizeType SearchInRadius( PointType const& ThisPoint, CoordinateType Radius, IteratorType Results, SizeType MaxNumberOfResults )
{
CoordinateType Radius2 = Radius * Radius;
SizeType NumberOfResults = 0;
SearchStructureType Box( CalculateCell(ThisPoint,-Radius), CalculateCell(ThisPoint,Radius), mN );
SearchInRadiusLocal( ThisPoint, Radius, Radius2, Results, NumberOfResults, MaxNumberOfResults, Box );
return NumberOfResults;
}


SizeType SearchInRadius( PointType const& ThisPoint, CoordinateType Radius, IteratorType Results,
SizeType MaxNumberOfResults, SearchStructureType& Box )
{
CoordinateType Radius2 = Radius * Radius;
SizeType NumberOfResults = 0;
Box.Set( CalculateCell(ThisPoint,-Radius), CalculateCell(ThisPoint,Radius), mN );
SearchInRadiusLocal( ThisPoint, Radius, Radius2, Results, NumberOfResults, MaxNumberOfResults, Box );
return NumberOfResults;
}


void SearchInRadius( PointType const& ThisPoint, CoordinateType const& Radius, CoordinateType const& Radius2, IteratorType& Results,
SizeType& NumberOfResults, SizeType const& MaxNumberOfResults ) override
{
SearchStructureType Box( CalculateCell(ThisPoint,-Radius), CalculateCell(ThisPoint,Radius), mN );
SearchInRadiusLocal( ThisPoint, Radius, Radius2, Results, NumberOfResults, MaxNumberOfResults, Box );
}


void SearchInRadius( PointType const& ThisPoint, CoordinateType const& Radius, CoordinateType const& Radius2, IteratorType& Results,
SizeType& NumberOfResults, SizeType const& MaxNumberOfResults, SearchStructureType& Box ) override
{
Box.Set( CalculateCell(ThisPoint,-Radius), CalculateCell(ThisPoint,Radius), mN );
SearchInRadiusLocal( ThisPoint, Radius, Radius2, Results, NumberOfResults, MaxNumberOfResults, Box );
}



void SearchInRadiusLocal( PointType const& ThisPoint, CoordinateType const& Radius, CoordinateType const& Radius2, IteratorType& Results,
SizeType& NumberOfResults, SizeType const& MaxNumberOfResults,
SearchStructure<IndexType,SizeType,CoordinateType,IteratorType,IteratorIteratorType,1>& Box )
{
for(IndexType I = Box.Axis[0].Begin() ; I <= Box.Axis[0].End() ; I++ )
SearchRadiusInRange()(mCells[I].begin(),mCells[I].end(),ThisPoint,Radius2,Results,NumberOfResults,MaxNumberOfResults);
}

void SearchInRadiusLocal( PointType const& ThisPoint, CoordinateType const& Radius, CoordinateType const& Radius2, IteratorType& Results,
SizeType& NumberOfResults, SizeType const& MaxNumberOfResults,
SearchStructure<IndexType,SizeType,CoordinateType,IteratorType,IteratorIteratorType,2>& Box )
{
for(IndexType II = Box.Axis[1].Begin() ; II <= Box.Axis[1].End() ; II += Box.Axis[1].Block )
for(IndexType I = II + Box.Axis[0].Begin() ; I <= II + Box.Axis[0].End() ; I++ )
SearchRadiusInRange()(mCells[I].begin(),mCells[I].end(),ThisPoint,Radius2,Results,NumberOfResults,MaxNumberOfResults);
}

void SearchInRadiusLocal( PointType const& ThisPoint, CoordinateType const& Radius, CoordinateType const& Radius2, IteratorType& Results,
SizeType& NumberOfResults, SizeType const& MaxNumberOfResults,
SearchStructure<IndexType,SizeType,CoordinateType,IteratorType,IteratorIteratorType,3>& Box )
{
for(IndexType III = Box.Axis[2].Begin() ; III <= Box.Axis[2].End() ; III += Box.Axis[2].Block )
for(IndexType II = III + Box.Axis[1].Begin() ; II <= III + Box.Axis[1].End() ; II += Box.Axis[1].Block )
for(IndexType I = II + Box.Axis[0].Begin() ; I <= II + Box.Axis[0].End() ; I++ )
SearchRadiusInRange()(mCells[I].begin(),mCells[I].end(),ThisPoint,Radius2,Results,NumberOfResults,MaxNumberOfResults);
}


void SearchNearestInBox( PointType const& ThisPoint, PointerType& ResultPoint, CoordinateType& ResultDistance,
SearchStructure<IndexType,SizeType,CoordinateType,IteratorType,IteratorIteratorType,1>& Box, bool& Found )
{
Found = false;
for(IndexType I = Box.Axis[0].Begin() ; I <= Box.Axis[0].End() ; I += Box.Axis[0].Block )
SearchNearestInRange()( mCells[I].begin(), mCells[I].end(), ThisPoint, ResultPoint, ResultDistance, Found );
}

void SearchNearestInBox( PointType const& ThisPoint, PointerType& ResultPoint, CoordinateType& ResultDistance,
SearchStructure<IndexType,SizeType,CoordinateType,IteratorType,IteratorIteratorType,2>& Box, bool& Found )
{
Found = false;
for(IndexType II = Box.Axis[1].Begin() ; II <= Box.Axis[1].End() ; II += Box.Axis[1].Block )
for(IndexType I = II + Box.Axis[0].Begin() ; I <= II + Box.Axis[0].End() ; I += Box.Axis[0].Block )
SearchNearestInRange()( mCells[I].begin(), mCells[I].end(), ThisPoint, ResultPoint, ResultDistance, Found );
}

void SearchNearestInBox( PointType const& ThisPoint, PointerType& ResultPoint, CoordinateType& ResultDistance,
SearchStructure<IndexType,SizeType,CoordinateType,IteratorType,IteratorIteratorType,3>& Box, bool& Found )
{
Found = false;
for(IndexType III = Box.Axis[2].Begin() ; III <= Box.Axis[2].End() ; III += Box.Axis[2].Block )
for(IndexType II = III + Box.Axis[1].Begin() ; II <= III + Box.Axis[1].End() ; II += Box.Axis[1].Block )
for(IndexType I = II + Box.Axis[0].Begin() ; I <= II + Box.Axis[0].End() ; I += Box.Axis[0].Block )
SearchNearestInRange()( mCells[I].begin(), mCells[I].end(), ThisPoint, ResultPoint, ResultDistance, Found );
}


SizeType SearchInBox( PointType const& SearchMinPoint, PointType const& SearchMaxPoint, IteratorType Results,
SizeType MaxNumberOfResults )
{
SizeType NumberOfResults = 0;
SearchStructureType Box( CalculateCell(SearchMinPoint), CalculateCell(SearchMaxPoint), mN );
SearchInBoxLocal( SearchMinPoint, SearchMaxPoint, Results, NumberOfResults, MaxNumberOfResults, Box );
return NumberOfResults;
}


void SearchInBox(PointType const& SearchMinPoint, PointType const& SearchMaxPoint, IteratorType& Results, SizeType& NumberOfResults,
SizeType const& MaxNumberOfResults ) override
{
NumberOfResults = 0;
SearchStructureType Box( CalculateCell(SearchMinPoint), CalculateCell(SearchMaxPoint), mN );
SearchInBoxLocal( SearchMinPoint, SearchMaxPoint, Results, NumberOfResults, MaxNumberOfResults, Box );
}


void SearchInBoxLocal( PointType const& SearchMinPoint, PointType const& SearchMaxPoint, IteratorType& ResultsPoint,
SizeType& NumberOfResults, SizeType const& MaxNumberOfResults,
SearchStructure<IndexType,SizeType,CoordinateType,IteratorType,IteratorIteratorType,1>& Box )
{
for(IndexType I = Box.Axis[0].Begin() ; I <= Box.Axis[0].End() ; I += Box.Axis[0].Block )
SearchBoxInRange()(SearchMinPoint,SearchMaxPoint,mCells[I].begin(),mCells[I].end(),ResultsPoint,NumberOfResults,MaxNumberOfResults);
}

void SearchInBoxLocal( PointType const& SearchMinPoint, PointType const& SearchMaxPoint, IteratorType& ResultsPoint,
SizeType& NumberOfResults, SizeType const& MaxNumberOfResults,
SearchStructure<IndexType,SizeType,CoordinateType,IteratorType,IteratorIteratorType,2>& Box )
{
for(IndexType II = Box.Axis[1].Begin() ; II <= Box.Axis[1].End() ; II += Box.Axis[1].Block )
for(IndexType I = II + Box.Axis[0].Begin() ; I <= II + Box.Axis[0].End() ; I += Box.Axis[0].Block )
SearchBoxInRange()(SearchMinPoint,SearchMaxPoint,mCells[I].begin(),mCells[I].end(),ResultsPoint,NumberOfResults,MaxNumberOfResults);
}

void SearchInBoxLocal( PointType const& SearchMinPoint, PointType const& SearchMaxPoint, IteratorType& ResultsPoint,
SizeType& NumberOfResults, SizeType const& MaxNumberOfResults,
SearchStructure<IndexType,SizeType,CoordinateType,IteratorType,IteratorIteratorType,3>& Box )
{
for(IndexType III = Box.Axis[2].Begin() ; III <= Box.Axis[2].End() ; III += Box.Axis[2].Block )
for(IndexType II = III + Box.Axis[1].Begin() ; II <= III + Box.Axis[1].End() ; II += Box.Axis[1].Block )
for(IndexType I = II + Box.Axis[0].Begin() ; I <= II + Box.Axis[0].End() ; I += Box.Axis[0].Block )
SearchBoxInRange()(SearchMinPoint,SearchMaxPoint,mCells[I].begin(),mCells[I].end(),ResultsPoint,NumberOfResults,MaxNumberOfResults);
}


virtual std::string Info() const
{
return "BinsDynamic";
}

virtual void PrintInfo(std::ostream& rOStream) const
{
rOStream << "BinsDynamic";
}

void PrintData(std::ostream& rOStream, std::string const& Perfix = std::string()) const override
{
rOStream << Perfix << "Bin[" << SearchUtils::PointerDistance(mPointBegin, mPointEnd) << "] : " << std::endl;
for(typename CellContainerType::const_iterator i_cell = mCells.begin() ; i_cell != mCells.end() ; i_cell++)
{
rOStream << Perfix << "[ " ;
for(typename LocalContainerType::const_iterator i_point = i_cell->begin() ; i_point != i_cell->end() ; i_point++)
rOStream << **i_point << "    ";
rOStream << " ]" << std::endl;
}
rOStream << std::endl;
}

void PrintSize( std::ostream& rout )
{
rout << " BinsSize: ";
for(SizeType i = 0 ; i < Dimension ; i++)
rout << "[" << mN[i] << "]";
rout << std::endl;
}

void PrintBox( std::ostream& rout )
{
rout << " BinsBox: Min [";
mMinPoint.Print(rout);
rout <<       "];  Max [";
mMaxPoint.Print(rout);
rout <<       "];  Size [";
mCellSize.Print(rout);
rout << "]" << std::endl;
}

BinsDynamic& operator=(BinsDynamic const& rOther);

BinsDynamic(BinsDynamic const& rOther);

private:

IteratorType     mPointBegin;
IteratorType     mPointEnd;

PointType        mMinPoint;
PointType        mMaxPoint;
CoordinateArray  mCellSize;
CoordinateArray  mInvCellSize;
SizeArray        mN;
SizeType         mNumCells;

CellContainerType mCells;


public:
static TreeNodeType* Construct(IteratorType PointsBegin, IteratorType PointsEnd, PointType MaxPoint, PointType MinPoint, SizeType BucketSize)
{

SizeType number_of_points = SearchUtils::PointerDistance(PointsBegin,PointsEnd);
if (number_of_points == 0)
return NULL;
else
{
return new BinsDynamic( PointsBegin, PointsEnd, MinPoint, MaxPoint, BucketSize );
}

}

};

template<
std::size_t TDimension,
class TPointType,
class TContainerType,
class TPointerType,
class TIteratorType,
class TDistanceIteratorType,
class TDistanceFunction >
std::ostream & operator<<( std::ostream& rOStream,
BinsDynamic<TDimension,TPointType,TContainerType,TPointerType,TIteratorType,TDistanceIteratorType,TDistanceFunction>& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintSize(rOStream);
rThis.PrintData(rOStream);
return rOStream;
}



}
