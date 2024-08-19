
#pragma once



#include "tree.h"
#include "utilities/parallel_utilities.h"


namespace Kratos
{

template<  std::size_t TDimension,
class TPointType,
class TContainerType,
class TPointerType = typename TContainerType::value_type,
class TIteratorType = typename TContainerType::iterator,
class TDistanceIteratorType = typename std::vector<double>::iterator,
class TDistanceFunction = Kratos::SearchUtils::SquaredDistanceFunction<TDimension,TPointType> >
class Bins : public TreeNode<TDimension,TPointType, TPointerType, TIteratorType, TDistanceIteratorType>
{


public:

enum { Dimension = TDimension };

typedef TPointType                                  PointType;
typedef TContainerType                              ContainerType;
typedef TIteratorType                               IteratorType;
typedef TDistanceIteratorType                       DistanceIteratorType;
typedef TPointerType                                PointerType;
typedef TDistanceFunction                           DistanceFunction;

typedef TreeNode<Dimension,PointType,PointerType,IteratorType,DistanceIteratorType> TreeNodeType;

typedef typename TreeNodeType::SizeType             SizeType;
typedef typename TreeNodeType::IndexType            IndexType;
typedef typename TreeNodeType::CoordinateType       CoordinateType;

typedef Tvector<CoordinateType,Dimension>           CoordinateArray;
typedef Tvector<SizeType,Dimension>                 SizeArray;
typedef Tvector<IndexType,Dimension>                IndexArray;

typedef typename TreeNodeType::IteratorIteratorType IteratorIteratorType;
typedef typename TreeNodeType::SearchStructureType  SearchStructureType;

typedef std::vector<PointerType>                    LocalContainerType;
typedef typename LocalContainerType::iterator       LocalIterator;

typedef Tvector<IndexType,TDimension>               CellType;

typedef std::vector<IteratorType>                   IteratorVector;
typedef typename IteratorVector::iterator           IteratorIterator;
typedef typename IteratorVector::const_iterator     IteratorConstIterator;

typedef Kratos::SearchUtils::SearchNearestInRange<PointType,PointerType,IteratorType,DistanceFunction,CoordinateType> SearchNearestInRange;
typedef Kratos::SearchUtils::SearchRadiusInRange<PointType,IteratorType,DistanceIteratorType,DistanceFunction,SizeType,CoordinateType> SearchRadiusInRange;
typedef Kratos::SearchUtils::SearchBoxInRange<PointType,IteratorType,SizeType,TDimension> SearchBoxInRange;

typedef LocalContainerType                          PointVector;
typedef LocalIterator                               PointIterator;
typedef TreeNodeType                                LeafType;

KRATOS_CLASS_POINTER_DEFINITION(Bins);


public:


Bins() : mPointBegin(this->NullIterator()), mPointEnd(this->NullIterator()) {};



Bins( IteratorType const& PointBegin, IteratorType const& PointEnd, SizeType BucketSize = 1 )
: mPointBegin(PointBegin), mPointEnd(PointEnd)
{
auto NumPoints = std::distance(mPointBegin, mPointEnd);

if(mPointBegin==mPointEnd)
return;

CalculateBoundingBox();
CalculateCellSize(NumPoints);
AllocateCellsContainer();
GenerateBins();
}


Bins( IteratorType const& PointBegin, IteratorType const& PointEnd, PointType const& MinPoint, PointType const& MaxPoint, SizeType BucketSize = 1 )
: mPointBegin(PointBegin), mPointEnd(PointEnd)
{
auto NumPoints = std::distance(mPointBegin, mPointEnd);

if(mPointBegin==mPointEnd)
return;

for(SizeType i = 0 ; i < TDimension ; i++)
{
mMinPoint[i] = MinPoint[i];
mMaxPoint[i] = MaxPoint[i];
}

CalculateCellSize(NumPoints);
AllocateCellsContainer();
GenerateBins();
}


Bins( IteratorType const& PointBegin, IteratorType const& PointEnd, CoordinateType cellsize, SizeType BucketSize = 1 )
: mPointBegin(PointBegin), mPointEnd(PointEnd)
{
if(mPointBegin==mPointEnd)
return;

CalculateBoundingBox();
AssignCellSize(cellsize);
AllocateCellsContainer();
GenerateBins();
}

~Bins() override { }

/
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

/
PointerType ExistPoint( PointerType const& ThisPoint, CoordinateType const Tolerance = static_cast<CoordinateType>(10.0*DBL_EPSILON) )
{
PointerType Nearest;
CoordinateType Distance = static_cast<CoordinateType>(DBL_MAX);
bool Found;
SearchStructureType Box( CalculateCell(*ThisPoint), mN, mIndexCellBegin );
SearchNearestInBox( *ThisPoint, Nearest, Distance, Box, Found );
if(Found)
return Nearest;
return this->NullPointer();
}


PointerType SearchNearestPointInner( PointerType& ThisPoint )
{
PointerType Result            = *mPointBegin;                           
CoordinateType ResultDistance = static_cast<CoordinateType>(DBL_MAX);
SearchStructureType Box( CalculateCell(*ThisPoint), mN, mIndexCellBegin );
SearchNearestPointLocalInner( ThisPoint, Result, ResultDistance, Box );
return Result;
}


PointerType SearchNearestPoint( PointType const& ThisPoint )
{
PointerType Result            = *mPointBegin;                           
CoordinateType ResultDistance = static_cast<CoordinateType>(DBL_MAX);
SearchStructureType Box( CalculateCell(ThisPoint), mN, mIndexCellBegin );
SearchNearestPointLocal( ThisPoint, Result, ResultDistance, Box );
return Result;
}


PointerType SearchNearestPoint( PointType const& ThisPoint, CoordinateType& rResultDistance )
{
PointerType Result = *mPointBegin;                           
rResultDistance    = static_cast<CoordinateType>(DBL_MAX);
SearchStructureType Box( CalculateCell(ThisPoint), mN, mIndexCellBegin );
SearchNearestPointLocal( ThisPoint, Result, rResultDistance, Box);
return Result;
}


PointerType SearchNearestPoint( PointType const& ThisPoint, CoordinateType& rResultDistance, SearchStructureType& Box )
{
PointerType Result            = *mPointBegin;                           
rResultDistance = static_cast<CoordinateType>(DBL_MAX);
Box.Set( CalculateCell(ThisPoint), mN, mIndexCellBegin );
SearchNearestPointLocal( ThisPoint, Result, rResultDistance, Box);
return Result;
}


void SearchNearestPoint( PointType const& ThisPoint, PointerType& rResult, CoordinateType& rResultDistance ) override
{
SearchStructureType Box( CalculateCell(ThisPoint), mN, mIndexCellBegin );
SearchNearestPointLocal(ThisPoint,rResult,rResultDistance,Box);
}


void SearchNearestPoint( PointType const& ThisPoint, PointerType& rResult, CoordinateType& rResultDistance, SearchStructureType& Box ) override
{
Box.Set( CalculateCell(ThisPoint), mN, mIndexCellBegin );
SearchNearestPointLocal( ThisPoint, rResult, rResultDistance, Box );
}


void SearchNearestPoint( PointerType const& ThisPoints, SizeType const& NumberOfPoints, IteratorType &Results, std::vector<CoordinateType> ResultsDistances)
{
IndexPartition<SizeType>(NumberOfPoints).for_each(
[&](SizeType iPoint)
{ Results[iPoint] = SearchNearestPoint((&(*ThisPoints))[iPoint],ResultsDistances[iPoint]); }
);
}


void SearchNearestPointLocal( PointType const& ThisPoint, PointerType& rResult, CoordinateType& rResultDistance, SearchStructureType& Box )
{
if( mPointBegin == mPointEnd )
return;

bool Found;

++Box;
SearchNearestInBox( ThisPoint, rResult, rResultDistance, Box, Found );

while(!Found)
{
++Box;
SearchNearestInBox( ThisPoint, rResult, rResultDistance, Box, Found );
}

}


void SearchNearestPointLocalInner( PointerType& ThisPoint, PointerType& rResult, CoordinateType& rResultDistance, SearchStructureType& Box )
{
if( mPointBegin == mPointEnd )
return;

bool Found;

++Box;
SearchNearestInBoxInner( ThisPoint, rResult, rResultDistance, Box, Found );
while(!Found)
{
++Box;
SearchNearestInBoxInner( ThisPoint, rResult, rResultDistance, Box, Found );
}

}


SizeType SearchInRadius( PointType const& ThisPoint, CoordinateType const& Radius, IteratorType Results,
DistanceIteratorType ResultsDistances, SizeType const& MaxNumberOfResults )
{
CoordinateType Radius2 = Radius * Radius;
SizeType NumberOfResults = 0;
SearchStructureType Box( CalculateCell(ThisPoint,-Radius), CalculateCell(ThisPoint,Radius), mN, mIndexCellBegin );
SearchInRadiusLocal( ThisPoint, Radius, Radius2, Results, ResultsDistances, NumberOfResults, MaxNumberOfResults, Box );
return NumberOfResults;
}


SizeType SearchInRadius( PointType const& ThisPoint, CoordinateType const& Radius, IteratorType Results,
DistanceIteratorType ResultsDistances, SizeType const& MaxNumberOfResults, SearchStructureType& Box )
{
CoordinateType Radius2 = Radius * Radius;
SizeType NumberOfResults = 0;
Box.Set( CalculateCell(ThisPoint,-Radius), CalculateCell(ThisPoint,Radius), mN, mIndexCellBegin );
SearchInRadiusLocal( ThisPoint, Radius, Radius2, Results, ResultsDistances, NumberOfResults, MaxNumberOfResults, Box );
return NumberOfResults;
}


void SearchInRadius( PointType const& ThisPoint, CoordinateType const& Radius, CoordinateType const& Radius2, IteratorType& Results,
DistanceIteratorType& ResultsDistances, SizeType& NumberOfResults, SizeType const& MaxNumberOfResults ) override
{
SearchStructureType Box( CalculateCell(ThisPoint,-Radius), CalculateCell(ThisPoint,Radius), mN, mIndexCellBegin );
SearchInRadiusLocal( ThisPoint, Radius, Radius2, Results, ResultsDistances, NumberOfResults, MaxNumberOfResults, Box);
}


void SearchInRadius( PointType const& ThisPoint, CoordinateType const& Radius, CoordinateType const& Radius2, IteratorType& Results,
DistanceIteratorType& ResultsDistances, SizeType& NumberOfResults, SizeType const& MaxNumberOfResults, SearchStructureType& Box ) override
{
Box.Set( CalculateCell(ThisPoint,-Radius), CalculateCell(ThisPoint,Radius), mN, mIndexCellBegin );
SearchInRadiusLocal( ThisPoint, Radius, Radius2, Results, ResultsDistances, NumberOfResults, MaxNumberOfResults, Box);
}


void SearchInRadius( PointerType const& ThisPoints, SizeType const& NumberOfPoints, std::vector<CoordinateType> const& Radius, std::vector<IteratorType> Results,
std::vector<DistanceIteratorType> ResultsDistances, std::vector<SizeType>& NumberOfResults, SizeType const& MaxNumberOfResults )
{
IndexPartition<SizeType>(NumberOfPoints).for_each(
[&](SizeType iPoint)
{ NumberOfResults[iPoint] = SearchInRadius((&(*ThisPoints))[iPoint],Radius[iPoint],Results[iPoint],ResultsDistances[iPoint],MaxNumberOfResults); }
);
}



void SearchInRadiusLocal( PointType const& ThisPoint, CoordinateType const& Radius, CoordinateType const& Radius2, IteratorType& Results,
DistanceIteratorType& ResultsDistances, SizeType& NumberOfResults, SizeType const& MaxNumberOfResults,
SearchStructure<IndexType,SizeType,CoordinateType,IteratorType,IteratorIteratorType,1>& Box )
{
SearchRadiusInRange()(*(Box.RowBegin),*(Box.RowEnd),ThisPoint,Radius2,Results,ResultsDistances,NumberOfResults,MaxNumberOfResults);
}

void SearchInRadiusLocal( PointType const& ThisPoint, CoordinateType const& Radius, CoordinateType const& Radius2, IteratorType& Results,
DistanceIteratorType& ResultsDistances, SizeType& NumberOfResults, SizeType const& MaxNumberOfResults,
SearchStructure<IndexType,SizeType,CoordinateType,IteratorType,IteratorIteratorType,2>& Box )
{
for(IndexType I = Box.Axis[1].Begin() ; I <= Box.Axis[1].End() ; I += Box.Axis[1].Block )
SearchRadiusInRange()(Box.RowBegin[I],Box.RowEnd[I],ThisPoint,Radius2,Results,ResultsDistances,NumberOfResults,MaxNumberOfResults);
}

void SearchInRadiusLocal( PointType const& ThisPoint, CoordinateType const& Radius, CoordinateType const& Radius2, IteratorType& Results,
DistanceIteratorType& ResultsDistances, SizeType& NumberOfResults, SizeType const& MaxNumberOfResults,
SearchStructure<IndexType,SizeType,CoordinateType,IteratorType,IteratorIteratorType,3>& Box )
{
for(IndexType II = Box.Axis[2].Begin() ; II <= Box.Axis[2].End() ; II += Box.Axis[2].Block )
for(IndexType I = II + Box.Axis[1].Begin() ; I <= II + Box.Axis[1].End() ; I += Box.Axis[1].Block )
SearchRadiusInRange()(Box.RowBegin[I],Box.RowEnd[I],ThisPoint,Radius2,Results,ResultsDistances,NumberOfResults,MaxNumberOfResults);
}


SizeType SearchInRadius( PointType const& ThisPoint, CoordinateType Radius, IteratorType Results, SizeType MaxNumberOfResults )
{
CoordinateType Radius2 = Radius * Radius;
SizeType NumberOfResults = 0;
SearchStructureType Box( CalculateCell(ThisPoint,-Radius), CalculateCell(ThisPoint,Radius), mN, mIndexCellBegin );
SearchInRadiusLocal( ThisPoint, Radius, Radius2, Results, NumberOfResults, MaxNumberOfResults, Box );
return NumberOfResults;
}


SizeType SearchInRadius( PointType const& ThisPoint, CoordinateType Radius, IteratorType Results,
SizeType MaxNumberOfResults, SearchStructureType& Box )
{
CoordinateType Radius2 = Radius * Radius;
SizeType NumberOfResults = 0;
Box.Set( CalculateCell(ThisPoint,-Radius), CalculateCell(ThisPoint,Radius), mN, mIndexCellBegin );
SearchInRadiusLocal( ThisPoint, Radius, Radius2, Results, NumberOfResults, MaxNumberOfResults, Box );
return NumberOfResults;
}


void SearchInRadius( PointType const& ThisPoint, CoordinateType const& Radius, CoordinateType const& Radius2, IteratorType& Results,
SizeType& NumberOfResults, SizeType const& MaxNumberOfResults ) override
{
SearchStructureType Box( CalculateCell(ThisPoint,-Radius), CalculateCell(ThisPoint,Radius), mN, mIndexCellBegin );
SearchInRadiusLocal( ThisPoint, Radius, Radius2, Results, NumberOfResults, MaxNumberOfResults, Box );
}


void SearchInRadius( PointType const& ThisPoint, CoordinateType const& Radius, CoordinateType const& Radius2, IteratorType& Results,
SizeType& NumberOfResults, SizeType const& MaxNumberOfResults, SearchStructureType& Box ) override
{
Box.Set( CalculateCell(ThisPoint,-Radius), CalculateCell(ThisPoint,Radius), mN, mIndexCellBegin );
SearchInRadiusLocal( ThisPoint, Radius, Radius2, Results, NumberOfResults, MaxNumberOfResults, Box );
}


void SearchInRadiusLocal( PointType const& ThisPoint, CoordinateType const& Radius, CoordinateType const& Radius2, IteratorType& Results,
SizeType& NumberOfResults, SizeType const& MaxNumberOfResults,
SearchStructure<IndexType,SizeType,CoordinateType,IteratorType,IteratorIteratorType,1>& Box )
{
SearchRadiusInRange()(*(Box.RowBegin),*(Box.RowEnd),ThisPoint,Radius2,Results,NumberOfResults,MaxNumberOfResults);
}

void SearchInRadiusLocal( PointType const& ThisPoint, CoordinateType const& Radius, CoordinateType const& Radius2, IteratorType& Results,
SizeType& NumberOfResults, SizeType const& MaxNumberOfResults,
SearchStructure<IndexType,SizeType,CoordinateType,IteratorType,IteratorIteratorType,2>& Box )
{
for(IndexType I = Box.Axis[1].Begin() ; I <= Box.Axis[1].End() ; I += Box.Axis[1].Block )
SearchRadiusInRange()(Box.RowBegin[I],Box.RowEnd[I],ThisPoint,Radius2,Results,NumberOfResults,MaxNumberOfResults);
}

void SearchInRadiusLocal( PointType const& ThisPoint, CoordinateType const& Radius, CoordinateType const& Radius2, IteratorType& Results,
SizeType& NumberOfResults, SizeType const& MaxNumberOfResults,
SearchStructure<IndexType,SizeType,CoordinateType,IteratorType,IteratorIteratorType,3>& Box )
{
for(IndexType II = Box.Axis[2].Begin() ; II <= Box.Axis[2].End() ; II += Box.Axis[2].Block )
for(IndexType I = II + Box.Axis[1].Begin() ; I <= II + Box.Axis[1].End() ; I += Box.Axis[1].Block )
SearchRadiusInRange()(Box.RowBegin[I],Box.RowEnd[I],ThisPoint,Radius2,Results,NumberOfResults,MaxNumberOfResults);
}


void SearchNearestInBox( PointType const& ThisPoint, PointerType& ResultPoint, CoordinateType& ResultDistance,
SearchStructure<IndexType,SizeType,CoordinateType,IteratorType,IteratorIteratorType,1>& Box, bool& Found )
{
Found = false;
SearchNearestInRange()( *(Box.RowBegin), *(Box.RowEnd), ThisPoint, ResultPoint, ResultDistance, Found );
}

void SearchNearestInBox( PointType const& ThisPoint, PointerType& ResultPoint, CoordinateType& ResultDistance,
SearchStructure<IndexType,SizeType,CoordinateType,IteratorType,IteratorIteratorType,2>& Box, bool& Found )
{
Found = false;
for(IndexType I = Box.Axis[1].Begin() ; I <= Box.Axis[1].End() ; I += Box.Axis[1].Block )
SearchNearestInRange()( Box.RowBegin[I], Box.RowEnd[I], ThisPoint, ResultPoint, ResultDistance, Found );
}

void SearchNearestInBox( PointType const& ThisPoint, PointerType& ResultPoint, CoordinateType& ResultDistance,
SearchStructure<IndexType,SizeType,CoordinateType,IteratorType,IteratorIteratorType,3>& Box, bool& Found )
{
Found = false;
for(IndexType II = Box.Axis[2].Begin() ; II <= Box.Axis[2].End() ; II += Box.Axis[2].Block )
for(IndexType I = II + Box.Axis[1].Begin() ; I <= II + Box.Axis[1].End() ; I += Box.Axis[1].Block )
SearchNearestInRange()( Box.RowBegin[I], Box.RowEnd[I], ThisPoint, ResultPoint, ResultDistance, Found );
}


void SearchNearestInBoxInner( PointerType& ThisPoint, PointerType& ResultPoint, CoordinateType& ResultDistance,
SearchStructure<IndexType,SizeType,CoordinateType,IteratorType,IteratorIteratorType,1>& Box, bool& Found )
{
Found = false;
SearchNearestInnerInRange( *(Box.RowBegin), *(Box.RowEnd), ThisPoint, ResultPoint, ResultDistance, Found );
}

void SearchNearestInBoxInner( PointerType& ThisPoint, PointerType& ResultPoint, CoordinateType& ResultDistance,
SearchStructure<IndexType,SizeType,CoordinateType,IteratorType,IteratorIteratorType,2>& Box, bool& Found )
{
Found = false;
for(IndexType I = Box.Axis[1].Begin() ; I <= Box.Axis[1].End() ; I += Box.Axis[1].Block )
SearchNearestInnerInRange( Box.RowBegin[I], Box.RowEnd[I], ThisPoint, ResultPoint, ResultDistance, Found );
}

void SearchNearestInBoxInner( PointerType& ThisPoint, PointerType& ResultPoint, CoordinateType& ResultDistance,
SearchStructure<IndexType,SizeType,CoordinateType,IteratorType,IteratorIteratorType,3>& Box, bool& Found )
{
Found = false;
for(IndexType II = Box.Axis[2].Begin() ; II <= Box.Axis[2].End() ; II += Box.Axis[2].Block )
for(IndexType I = II + Box.Axis[1].Begin() ; I <= II + Box.Axis[1].End() ; I += Box.Axis[1].Block )
SearchNearestInnerInRange( Box.RowBegin[I], Box.RowEnd[I], ThisPoint, ResultPoint, ResultDistance, Found );
}


void SearchNearestInnerInRange( const IteratorType& RangeBegin, const IteratorType& RangeEnd, PointerType& ThisPoint,
PointerType& Result, CoordinateType& Distance, bool& Found )
{
CoordinateType NewDistance;
for(IteratorType Point = RangeBegin ; Point != RangeEnd ; Point++)
{
NewDistance = TDistanceFunction()(**Point,*ThisPoint);
if( NewDistance < Distance && *Point != ThisPoint)
{
Result = *Point;
Distance = NewDistance;
Found = true;
}
}
}


SizeType SearchInBox( PointType const& SearchMinPoint, PointType const& SearchMaxPoint, IteratorType Results,
SizeType MaxNumberOfResults )
{
SizeType NumberOfResults = 0;
SearchStructureType Box( CalculateCell(SearchMinPoint), CalculateCell(SearchMaxPoint), mN, mIndexCellBegin );
SearchInBoxLocal( SearchMinPoint, SearchMaxPoint, Results, NumberOfResults, MaxNumberOfResults, Box );
return NumberOfResults;
}


void SearchInBox(PointType const& SearchMinPoint, PointType const& SearchMaxPoint, IteratorType& Results, SizeType& NumberOfResults,
SizeType const& MaxNumberOfResults ) override
{
NumberOfResults = 0;
SearchStructureType Box( CalculateCell(SearchMinPoint), CalculateCell(SearchMaxPoint), mN, mIndexCellBegin );
SearchInBoxLocal( SearchMinPoint, SearchMaxPoint, Results, NumberOfResults, MaxNumberOfResults, Box );
}


void SearchInBoxLocal( PointType const& SearchMinPoint, PointType const& SearchMaxPoint, IteratorType& ResultsPoint,
SizeType& NumberOfResults, SizeType const& MaxNumberOfResults,
SearchStructure<IndexType,SizeType,CoordinateType,IteratorType,IteratorIteratorType,1>& Box )
{
SearchBoxInRange()(SearchMinPoint,SearchMaxPoint,*(Box.RowBegin),*(Box.RowEnd),ResultsPoint,NumberOfResults,MaxNumberOfResults);
}

void SearchInBoxLocal( PointType const& SearchMinPoint, PointType const& SearchMaxPoint, IteratorType& ResultsPoint,
SizeType& NumberOfResults, SizeType const& MaxNumberOfResults,
SearchStructure<IndexType,SizeType,CoordinateType,IteratorType,IteratorIteratorType,2>& Box )
{
for(IndexType I = Box.Axis[1].Begin() ; I <= Box.Axis[1].End() ; I += Box.Axis[1].Block )
SearchBoxInRange()(SearchMinPoint,SearchMaxPoint,Box.RowBegin[I],Box.RowEnd[I],ResultsPoint,NumberOfResults,MaxNumberOfResults);
}

void SearchInBoxLocal( PointType const& SearchMinPoint, PointType const& SearchMaxPoint, IteratorType& ResultsPoint,
SizeType& NumberOfResults, SizeType const& MaxNumberOfResults,
SearchStructure<IndexType,SizeType,CoordinateType,IteratorType,IteratorIteratorType,3>& Box )
{
for(IndexType II = Box.Axis[2].Begin() ; II <= Box.Axis[2].End() ; II += Box.Axis[2].Block )
for(IndexType I = II + Box.Axis[1].Begin() ; I <= II + Box.Axis[1].End() ; I += Box.Axis[1].Block )
SearchBoxInRange()(SearchMinPoint,SearchMaxPoint,Box.RowBegin[I],Box.RowEnd[I],ResultsPoint,NumberOfResults,MaxNumberOfResults);
}


virtual std::string Info() const
{
return "BinsContainer";
}

virtual void PrintInfo(std::ostream& rOStream) const
{
rOStream << "BinsContainer";
}

void PrintData(std::ostream& rOStream, std::string const& Perfix = std::string()) const override
{
rOStream << Perfix << "Bin[" << SearchUtils::PointerDistance(mPointBegin, mPointEnd) << "] : " << std::endl;
for(IteratorConstIterator i_cell = mIndexCell.begin() ; i_cell != mIndexCell.end()-1 ; i_cell++)
{
rOStream << Perfix << "[ " ;
for(IteratorType i_point = *i_cell ; i_point != *(i_cell+1) ; i_point++)
rOStream << **i_point << " ";
rOStream << "]" << std::endl;
}
rOStream << std::endl;
}

void PrintSize( std::ostream& rout )
{
rout << " BinsSize: ";
for(SizeType i = 0 ; i < TDimension ; i++)
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

Bins& operator=(Bins const& rOther);

Bins(Bins const& rOther);

private:

IteratorType    mPointBegin;
IteratorType    mPointEnd;

PointType       mMinPoint;
PointType       mMaxPoint;
CoordinateArray mCellSize;
CoordinateArray mInvCellSize;
SizeArray       mN;

IteratorVector           mIndexCell;
IteratorIterator         mIndexCellBegin;
IteratorIterator         mIndexCellEnd;


public:

static TreeNodeType* Construct(IteratorType PointsBegin, IteratorType PointsEnd, const PointType& MaxPoint, const PointType& MinPoint, SizeType BucketSize)
{
SizeType number_of_points = SearchUtils::PointerDistance(PointsBegin,PointsEnd);
if (number_of_points == 0)
return NULL;
else
{
return new Bins( PointsBegin, PointsEnd, MinPoint, MaxPoint, BucketSize );
}
}


};

template< std::size_t TDimension, class TPointType, class TContainerType, class TPointerType,
class TIteratorType, class TDistanceIteratorType, class TDistanceFunction >
std::ostream & operator<<( std::ostream& rOStream, Bins<TDimension,TPointType,TContainerType,TPointerType,TIteratorType,TDistanceIteratorType,TDistanceFunction>& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintSize(rOStream);
rThis.PrintData(rOStream);
return rOStream;
}

}
