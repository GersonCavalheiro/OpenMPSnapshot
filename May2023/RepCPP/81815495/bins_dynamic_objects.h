
#pragma once

#include <string>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <array>

#include "tree.h"
#include "cell.h"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace Kratos
{








template<class TConfigure>
class BinsObjectDynamic {
public:

enum { Dimension = TConfigure::Dimension };

typedef TConfigure                                    Configure;
typedef typename TConfigure::PointType                PointType;
typedef typename TConfigure::PointerType              PointerType;
typedef typename TConfigure::ContainerType            ContainerType;
typedef typename TConfigure::IteratorType             IteratorType;
typedef typename TConfigure::ResultContainerType      ResultContainerType;
typedef typename TConfigure::ResultIteratorType       ResultIteratorType;
typedef typename TConfigure::DistanceIteratorType     DistanceIteratorType;

typedef TreeNode<Dimension, PointType, PointerType, IteratorType,  typename TConfigure::DistanceIteratorType> TreeNodeType;

typedef typename TreeNodeType::CoordinateType         CoordinateType;  
typedef typename TreeNodeType::SizeType               SizeType;        
typedef typename TreeNodeType::IndexType              IndexType;       

typedef Tvector<CoordinateType,Dimension>             CoordinateArray;
typedef Tvector<SizeType,Dimension>                   SizeArray;
typedef Tvector<IndexType,Dimension>                  IndexArray;

typedef typename TreeNodeType::IteratorIteratorType   IteratorIteratorType;
typedef typename TreeNodeType::SearchStructureType    SearchStructureType;

typedef Cell<Configure>                               CellType;
typedef std::vector<CellType>                         CellContainerType;
typedef typename CellContainerType::iterator          CellContainerIterator;

KRATOS_CLASS_POINTER_DEFINITION(BinsObjectDynamic);


BinsObjectDynamic() {}


BinsObjectDynamic (IteratorType const& ObjectsBegin, IteratorType const& ObjectsEnd)
: mObjectsBegin(ObjectsBegin), mObjectsEnd(ObjectsEnd) {

mObjectsSize = SearchUtils::PointerDistance(mObjectsBegin,mObjectsEnd);

CalculateBoundingBox();                                         
CalculateCellSize(mObjectsSize);                                
AllocateContainer();                                            
GenerateBins();                                                 
}


BinsObjectDynamic (IteratorType const& ObjectsBegin, IteratorType const& ObjectsEnd, CoordinateType CellSize)
: mObjectsBegin(ObjectsBegin), mObjectsEnd(ObjectsEnd) {

mObjectsSize = SearchUtils::PointerDistance(mObjectsBegin,mObjectsEnd);

CalculateBoundingBox();                                         
AssignCellSize(CellSize);                                       
AllocateContainer();                                            
GenerateBins();                                                 
}


BinsObjectDynamic (const PointType& MinPoint, const PointType& MaxPoint, CoordinateType CellSize)
: mObjectsSize(0), mObjectsBegin(0), mObjectsEnd(0) {

for(SizeType i = 0; i < Dimension; i++) {
mMinPoint[i] = MinPoint[i];
mMaxPoint[i] = MaxPoint[i];
}

AssignCellSize(CellSize);                                       
AllocateContainer();                                            
}


BinsObjectDynamic (const PointType& MinPoint, const PointType& MaxPoint, SizeType NumPoints)
: mObjectsSize(0), mObjectsBegin(0), mObjectsEnd(0) {

for(SizeType i = 0; i < Dimension; i++) {
mMinPoint[i] = MinPoint[i];
mMaxPoint[i] = MaxPoint[i];
}

CalculateCellSize(NumPoints);                                 
AllocateContainer();                                         
}

virtual ~BinsObjectDynamic() {}



SizeType SearchObjects(PointerType& ThisObject, ResultContainerType& Result) {
PointType Low, High;
SearchStructureType Box;

TConfigure::CalculateBoundingBox(ThisObject, Low, High);
Box.Set( CalculateCell(Low), CalculateCell(High), mN );
SearchInBoxLocal(ThisObject, Result, Box );

return Result.size();
}


SizeType SearchObjects(PointerType& ThisObject, ResultIteratorType& Result, const SizeType& MaxNumberOfResults) {
PointType Low, High;
SearchStructureType Box;
SizeType NumberOfResults = 0;

TConfigure::CalculateBoundingBox(ThisObject, Low, High);
Box.Set( CalculateCell(Low), CalculateCell(High), mN );
SearchInBoxLocal(ThisObject, Result, NumberOfResults, MaxNumberOfResults, Box );

return NumberOfResults;
}


SizeType SearchObjectsInCell(const PointType& ThisPoint, ResultIteratorType Result) {
KRATOS_ERROR << "Missing implementation of SearchObjectsInCell(PointerType, ResultIteratorType)" << std::endl;
}


SizeType SearchObjectsInCell(const PointType& ThisPoint, ResultIteratorType Result, const SizeType& MaxNumberOfResults) {
IndexType icell = CalculateIndex(ThisPoint);

if(mCells[icell].Size() < MaxNumberOfResults) {
for(IteratorType i_object = mCells[icell].Begin() ; i_object != mCells[icell].End(); i_object++, Result++) {
*Result = *i_object;
}
return mCells[icell].Size();
} else {
return std::numeric_limits<SizeType>::max();
}
}


SizeType SearchObjectsExclusive(PointerType& ThisObject, ResultIteratorType& Result) {
PointType Low, High;
SearchStructureType Box;

TConfigure::CalculateBoundingBox(ThisObject, Low, High);
Box.Set( CalculateCell(Low), CalculateCell(High), mN );
SearchObjectLocalExclusive(ThisObject, Result, Box );

return Result.size();
}


SizeType SearchObjectsExclusive(PointerType& ThisObject, ResultIteratorType& Result, const SizeType& MaxNumberOfResults) {
PointType Low, High;
SearchStructureType Box;
SizeType NumberOfResults = 0;

TConfigure::CalculateBoundingBox(ThisObject, Low, High);
Box.Set( CalculateCell(Low), CalculateCell(High), mN );
SearchObjectLocalExclusive(ThisObject, Result, NumberOfResults, MaxNumberOfResults, Box );

return NumberOfResults;
}


SizeType SearchObjectsInRadius(PointerType& ThisObject, const double& Radius, ResultIteratorType& Results) {
KRATOS_ERROR << "Missing implementation of SearchObjectsInRadius(PointerType, const double, ResultIteratorType)" << std::endl;
}


SizeType SearchObjectsInRadius(PointerType& ThisObject, const double& Radius, ResultIteratorType& Results, const SizeType& MaxNumberOfResults) {
PointType Low, High;
SearchStructureType Box;
SizeType NumberOfResults = 0;

TConfigure::CalculateBoundingBox(ThisObject, Low, High, Radius);
Box.Set( CalculateCell(Low), CalculateCell(High), mN );
SearchInRadius(ThisObject, Radius, Results, NumberOfResults, MaxNumberOfResults, Box );

return NumberOfResults;
}


SizeType SearchObjectsInRadius(PointerType& ThisObject, const double& Radius, ResultIteratorType& Results, DistanceIteratorType ResultDistances) {
KRATOS_ERROR << "Missing implementation of SearchObjectsInRadius(PointerType, const double, ResultIteratorType, DistanceIteratorType)" << std::endl;
}


SizeType SearchObjectsInRadius(PointerType& ThisObject, const double& Radius, ResultIteratorType& Results, DistanceIteratorType ResultDistances, const SizeType& MaxNumberOfResults) {
PointType Low, High;
SearchStructureType Box;
SizeType NumberOfResults = 0;

TConfigure::CalculateBoundingBox(ThisObject, Low, High, Radius);
Box.Set( CalculateCell(Low), CalculateCell(High), mN );
SearchInRadius(ThisObject, Radius, Results, ResultDistances, NumberOfResults, MaxNumberOfResults, Box );

return NumberOfResults;
}


virtual SizeType SearchObjectsInRadiusExclusive(PointerType& ThisObject, const double& Radius, ResultIteratorType& Results) {
KRATOS_ERROR << "Missing implementation of SearchObjectsInRadiusExclusive(PointerType, const double, ResultIteratorType)" << std::endl;
}


virtual SizeType SearchObjectsInRadiusExclusive(PointerType& ThisObject, const double& Radius, ResultIteratorType& Results, const SizeType& MaxNumberOfResults) {
PointType Low, High;
SearchStructureType Box;
SizeType NumberOfResults = 0;

TConfigure::CalculateBoundingBox(ThisObject, Low, High, Radius);
Box.Set( CalculateCell(Low), CalculateCell(High), mN );
SearchInRadiusExclusive(ThisObject, Radius, Results, NumberOfResults, MaxNumberOfResults, Box );

return NumberOfResults;
}


virtual SizeType SearchObjectsInRadiusExclusive(PointerType& ThisObject, const double& Radius, ResultIteratorType& Results, DistanceIteratorType ResultDistances) {
KRATOS_ERROR << "Missing implementation of SearchObjectsInRadiusExclusive(PointerType, const double, ResultIteratorType, DistanceIteratorType)" << std::endl;
}


virtual SizeType SearchObjectsInRadiusExclusive(PointerType& ThisObject, const double& Radius, ResultIteratorType& Results, DistanceIteratorType ResultDistances, const SizeType& MaxNumberOfResults) {
PointType Low, High;
SearchStructureType Box;
SizeType NumberOfResults = 0;

TConfigure::CalculateBoundingBox(ThisObject, Low, High, Radius);
Box.Set( CalculateCell(Low), CalculateCell(High), mN );
SearchInRadiusExclusive(ThisObject, Radius, Results, ResultDistances, NumberOfResults, MaxNumberOfResults, Box );

return NumberOfResults;
}



void SearchObjectsInRadius(IteratorType const& ThisObjects, SizeType const& NumberOfObjects, const std::vector<double>& Radius, std::vector<std::vector<PointerType> >& Results, std::vector<SizeType>& NumberOfResults, SizeType const& MaxNumberOfResults) {

struct tls_type
{
PointType Low;
PointType High;
SearchStructureType Box;
};

IndexPartition<std::size_t>(NumberOfObjects).for_each(tls_type(), [&](std::size_t i, tls_type& rTLS){
ResultIteratorType ResultsPointer = Results[i].begin();
NumberOfResults[i] = 0;

TConfigure::CalculateBoundingBox(ThisObjects[i], rTLS.Low, rTLS.High, Radius[i]);
rTLS.Box.Set( CalculateCell(rTLS.Low), CalculateCell(rTLS.High), mN );

SearchInRadius(ThisObjects[i], Radius[i], ResultsPointer, NumberOfResults[i], MaxNumberOfResults, rTLS.Box );
});
}


void SearchObjectsInRadius(IteratorType const& ThisObjects, SizeType const& NumberOfObjects, const std::vector<double>& Radius, std::vector<std::vector<PointerType> >& Results, std::vector<std::vector<double> >& ResultsDistances, std::vector<SizeType>& NumberOfResults, SizeType const& MaxNumberOfResults) {
PointType Low, High;
SearchStructureType Box;

#pragma omp parallel for private(Low,High,Box)
for(int i = 0; i < static_cast<int>(NumberOfObjects); i++) {
ResultIteratorType ResultsPointer            = Results[i].begin();
DistanceIteratorType ResultsDistancesPointer = ResultsDistances[i].begin();

NumberOfResults[i] = 0;

TConfigure::CalculateBoundingBox(ThisObjects[i], Low, High, Radius[i]);
Box.Set( CalculateCell(Low), CalculateCell(High), mN );

SearchInRadius(ThisObjects[i], Radius[i], ResultsPointer, ResultsDistancesPointer, NumberOfResults[i], MaxNumberOfResults, Box );
}
}


virtual void SearchObjectsInRadiusExclusive(IteratorType const& ThisObjects, SizeType const& NumberOfObjects, const std::vector<double>& Radius, std::vector<std::vector<PointerType> >& Results, std::vector<SizeType>& NumberOfResults, SizeType const& MaxNumberOfResults) {
PointType Low, High;
SearchStructureType Box;

#pragma omp parallel for private(Low,High,Box)
for(int i = 0; i < static_cast<int>(NumberOfObjects); i++) {
ResultIteratorType ResultsPointer            = Results[i].begin();

NumberOfResults[i] = 0;

TConfigure::CalculateBoundingBox(ThisObjects[i], Low, High, Radius[i]);
Box.Set( CalculateCell(Low), CalculateCell(High), mN );

SearchInRadiusExclusive(ThisObjects[i], Radius[i], ResultsPointer, NumberOfResults[i], MaxNumberOfResults, Box );
}
}


virtual void SearchObjectsInRadiusExclusive(IteratorType const& ThisObjects, SizeType const& NumberOfObjects, const std::vector<double>& Radius, std::vector<std::vector<PointerType> >& Results, std::vector<std::vector<double> >& ResultsDistances, std::vector<SizeType>& NumberOfResults, SizeType const& MaxNumberOfResults) {
PointType Low, High;
SearchStructureType Box;

#pragma omp parallel for private(Low,High,Box)
for(int i = 0; i < static_cast<int>(NumberOfObjects); i++) {
ResultIteratorType ResultsPointer            = Results[i].begin();
DistanceIteratorType ResultsDistancesPointer = ResultsDistances[i].begin();

NumberOfResults[i] = 0;

TConfigure::CalculateBoundingBox(ThisObjects[i], Low, High, Radius[i]);
Box.Set( CalculateCell(Low), CalculateCell(High), mN );

SearchInRadiusExclusive(ThisObjects[i], Radius[i], ResultsPointer, ResultsDistancesPointer, NumberOfResults[i], MaxNumberOfResults, Box );
}
}



virtual void AddObject(const PointerType& ThisObject) {
PointType Low, High;
SearchStructureType Box;

TConfigure::CalculateBoundingBox(ThisObject, Low, High);
Box.Set( CalculateCell(Low), CalculateCell(High), mN );
FillObject(Box,ThisObject);

mObjectsSize++;
}


void RemoveObject(const PointerType& ThisObject) {
PointType Low, High;
SearchStructureType Box;

TConfigure::CalculateBoundingBox(ThisObject, Low, High);
Box.Set( CalculateCell(Low), CalculateCell(High), mN );
RemoveObjectLocal(Box,ThisObject);

mObjectsSize--;
}





template<class GenericCoordType>
IndexArray CalculateCell(const GenericCoordType& ThisObject) {
IndexArray IndexCell;

for(SizeType i = 0 ; i < Dimension ; i++) {
IndexCell[i] = CalculatePosition(ThisObject[i],i);
}

return IndexCell;
}


template<class GenericCoordType>
IndexType CalculateIndex(const GenericCoordType& ThisObject) {
IndexType Index = 0;

for(SizeType iDim = Dimension-1 ; iDim > 0 ; iDim--) {
Index += CalculatePosition(ThisObject[iDim],iDim);
Index *= mN[iDim-1];
}

Index += CalculatePosition(ThisObject[0],0);

return Index;
}


virtual IndexType CalculatePosition(CoordinateType const& ThisCoord, const SizeType& ThisDimension) {
CoordinateType d_index = (ThisCoord - mMinPoint[ThisDimension]) * mInvCellSize[ThisDimension];
IndexType index = static_cast<IndexType>( (d_index < 0.00) ? 0.00 : d_index );

return  (index > mN[ThisDimension]-1) ? mN[ThisDimension]-1 : index;
}



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




virtual std::string Info() const {
return "BinsObjectDynamic" ;
}


virtual void PrintInfo(std::ostream& rOStream) const {
rOStream << Info();
}


virtual void PrintData(std::ostream& rOStream, std::string const& Perfix = std::string()) const {
rOStream << " BinsSize: ";
for(SizeType i = 0 ; i < Dimension ; i++) {
rOStream << "[" << mN[i] << "]";
}
rOStream << std::endl;
rOStream << "  CellSize: ";
for(SizeType i = 0 ; i < Dimension ; i++) {
rOStream << "[" << mCellSize[i] << "]";
}
rOStream << std::endl;
SizeType nn = 0;
for(SizeType i = 0 ; i < mCells.size(); i++) {
nn += mCells[i].Size();
}
rOStream << "NumPointers: " << nn << std::endl;
}


void PrintSize(std::ostream& rout) {
rout << " BinsSize: ";
for(SizeType i = 0 ; i < Dimension ; i++) {
rout << "[" << mN[i] << "]";
}
rout << std::endl;
}


void PrintBox(std::ostream& rout) {
rout << " BinsBox: Min [";
mMinPoint.Print(rout);
rout <<       "];  Max [";
mMaxPoint.Print(rout);
rout <<       "];  Size [";
mCellSize.Print(rout);
rout << "]" << std::endl;
}

protected:









virtual void CalculateBoundingBox()
{
PointType Low, High;
TConfigure::CalculateBoundingBox(*mObjectsBegin,mMinPoint,mMaxPoint);

#ifdef _OPENMP
SizeType number_of_threads = omp_get_max_threads();
#else
SizeType number_of_threads = 1;
#endif

std::vector<SizeType> node_partition;
CreatePartition(number_of_threads, mObjectsSize, node_partition);

std::vector<PointType> Max(number_of_threads);
std::vector<PointType> Min(number_of_threads);

for(SizeType k=0; k<number_of_threads; k++ )
{
Max[k] = mMaxPoint;
Min[k] = mMinPoint;
}

IteratorType i_begin = mObjectsBegin;
IteratorType i_end   = mObjectsEnd;

for (IteratorType i_object = i_begin ; i_object != i_end ; i_object++ )
{
TConfigure::CalculateBoundingBox(*i_object, Low, High);
for(SizeType i = 0 ; i < Dimension ; i++)
{
mMaxPoint[i] = (mMaxPoint[i] < High[i]) ? High[i] : mMaxPoint[i];
mMinPoint[i] = (mMinPoint[i] > Low[i])  ? Low[i]  : mMinPoint[i];
}
}

auto Epsilon = PointType{mMaxPoint - mMinPoint};

for(SizeType i = 0 ; i < Dimension ; i++)
{
mMaxPoint[i] += Epsilon[i] * 0.01;
mMinPoint[i] -= Epsilon[i] * 0.01;
}
}


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


void AssignCellSize(CoordinateType CellSize)
{
for(SizeType i = 0 ; i < Dimension ; i++)
{
mCellSize[i] = CellSize;
mInvCellSize[i] = 1.00 / mCellSize[i];
mN[i] = static_cast<SizeType>( (mMaxPoint[i]-mMinPoint[i]) / mCellSize[i]) + 1;
}
}

virtual void GenerateBins()
{
PointType Low, High;
SearchStructureType Box;
for(IteratorType i_object = mObjectsBegin ; i_object != mObjectsEnd ; i_object++)
{
TConfigure::CalculateBoundingBox(*i_object, Low, High);
Box.Set( CalculateCell(Low), CalculateCell(High), mN );
FillObject(Box, *i_object);
}
}

void SearchInBoxLocal(PointerType& ThisObject, ResultIteratorType& Result, SizeType& NumberOfResults, const SizeType& MaxNumberOfResults,
SearchStructure<IndexType,SizeType,CoordinateType,IteratorType,IteratorIteratorType,1>& Box )
{
PointType  MinCell, MaxCell;
PointType  MinBox, MaxBox;
MinCell[0] = static_cast<CoordinateType>(Box.Axis[0].Min) * mCellSize[0] + mMinPoint[0];  
MaxCell[0] = MinCell[0] + mCellSize[0];
for(IndexType I = Box.Axis[0].Begin() ; I <= Box.Axis[0].End() ; I += Box.Axis[0].Block, MinCell[0] += mCellSize[0], MaxCell[0] += mCellSize[0])
if(TConfigure::IntersectionBox(ThisObject, MinCell, MaxCell))
mCells[I].SearchObjects(ThisObject, Result, NumberOfResults, MaxNumberOfResults);
}

void SearchInBoxLocal(PointerType& ThisObject, ResultIteratorType& Result, SizeType& NumberOfResults, const SizeType& MaxNumberOfResults,
SearchStructure<IndexType,SizeType,CoordinateType,IteratorType,IteratorIteratorType,2>& Box )
{
PointType  MinCell, MaxCell;
PointType  MinBox, MaxBox;

for(SizeType i = 0; i < 2; i++)
{
MinBox[i] = static_cast<CoordinateType>(Box.Axis[i].Min) * mCellSize[i] + mMinPoint[i];
MaxBox[i] = MinBox[i] + mCellSize[i];
}

MinCell[1] = MinBox[1];
MaxCell[1] = MaxBox[1];
for(IndexType II = Box.Axis[1].Begin() ; II <= Box.Axis[1].End() ; II += Box.Axis[1].Block, MinCell[1] += mCellSize[1], MaxCell[1] += mCellSize[1] )
{
MinCell[0] = MinBox[0];
MaxCell[0] = MaxBox[0];
for(IndexType I = II + Box.Axis[0].Begin() ; I <= II + Box.Axis[0].End() ; I += Box.Axis[0].Block, MinCell[0] += mCellSize[0], MaxCell[0] += mCellSize[0] )
{
if(TConfigure::IntersectionBox(ThisObject, MinCell, MaxCell))
mCells[I].SearchObjects(ThisObject, Result, NumberOfResults, MaxNumberOfResults);
}
}
}

void SearchInBoxLocal(PointerType& ThisObject, ResultIteratorType& Result,
SizeType& NumberOfResults, const SizeType& MaxNumberOfResults,
SearchStructure<IndexType,SizeType,CoordinateType,IteratorType,IteratorIteratorType,3>& Box )
{

PointType  MinCell, MaxCell;
PointType  MinBox, MaxBox;

for(SizeType i = 0; i < 3; i++)
{
MinBox[i] = static_cast<CoordinateType>(Box.Axis[i].Min) * mCellSize[i] + mMinPoint[i];  
MaxBox[i] = MinBox[i] + mCellSize[i];
}

MinCell[2] = MinBox[2];
MaxCell[2] = MaxBox[2];
for(IndexType III = Box.Axis[2].Begin() ; III <= Box.Axis[2].End() ; III += Box.Axis[2].Block, MinCell[2] += mCellSize[2], MaxCell[2] += mCellSize[2] )
{
MinCell[1] = MinBox[1];
MaxCell[1] = MaxBox[1];
for(IndexType II = III + Box.Axis[1].Begin() ; II <= III + Box.Axis[1].End() ; II += Box.Axis[1].Block, MinCell[1] += mCellSize[1], MaxCell[1] += mCellSize[1] )
{
MinCell[0] = MinBox[0];
MaxCell[0] = MaxBox[0];
for(IndexType I = II + Box.Axis[0].Begin() ; I <= II + Box.Axis[0].End() ; I += Box.Axis[0].Block, MinCell[0] += mCellSize[0], MaxCell[0] += mCellSize[0] )
{
if(TConfigure::IntersectionBox(ThisObject, MinCell, MaxCell))
{
mCells[I].SearchObjects(ThisObject, Result, NumberOfResults, MaxNumberOfResults);
}
}
}
}
}


void SearchInBoxLocal(PointerType& ThisObject, ResultContainerType& Result,
SearchStructure<IndexType,SizeType,CoordinateType,IteratorType,IteratorIteratorType,1>& Box )
{

PointType  MinCell, MaxCell;
PointType  MinBox, MaxBox;
MinCell[0] = static_cast<CoordinateType>(Box.Axis[0].Min) * mCellSize[0] + mMinPoint[0];  
MaxCell[0] = MinCell[0] + mCellSize[0];
for(IndexType I = Box.Axis[0].Begin() ; I <= Box.Axis[0].End() ; I += Box.Axis[0].Block )
{
if(TConfigure::IntersectionBox(ThisObject, MinCell, MaxCell))
mCells[I].SearchObjects(ThisObject, Result);
}
}

void SearchInBoxLocal(PointerType& ThisObject, ResultContainerType& Result,
SearchStructure<IndexType,SizeType,CoordinateType,IteratorType,IteratorIteratorType,2>& Box )
{
PointType  MinCell, MaxCell;
PointType  MinBox, MaxBox;

for(SizeType i = 0; i < 2; i++)
{
MinBox[i] = static_cast<CoordinateType>(Box.Axis[i].Min) * mCellSize[i] + mMinPoint[i];
MaxBox[i] = MinBox[i] + mCellSize[i];
}

MinCell[1] = MinBox[1];
MaxCell[1] = MaxBox[1];

for(IndexType II = Box.Axis[1].Begin() ; II <= Box.Axis[1].End() ; II += Box.Axis[1].Block )
{
MinCell[0] = MinBox[0];
MaxCell[0] = MaxBox[0];
for(IndexType I = II + Box.Axis[0].Begin() ; I <= II + Box.Axis[0].End() ; I += Box.Axis[0].Block )
{
if(TConfigure::IntersectionBox(ThisObject, MinCell, MaxCell))
mCells[I].SearchObjects(ThisObject, Result);
}
}
}

void SearchInBoxLocal(PointerType& ThisObject, ResultContainerType& Result,
SearchStructure<IndexType,SizeType,CoordinateType,IteratorType,IteratorIteratorType,3>& Box )
{
PointType  MinCell, MaxCell;
PointType  MinBox, MaxBox;

for(SizeType i = 0; i < 3; i++)
{
MinBox[i] = static_cast<CoordinateType>(Box.Axis[i].Min) * mCellSize[i] + mMinPoint[i];  
MaxBox[i] = MinBox[i] + mCellSize[i];
}

MinCell[2] = MinBox[2];
MaxCell[2] = MaxBox[2];

for(IndexType III = Box.Axis[2].Begin() ; III <= Box.Axis[2].End() ; III += Box.Axis[2].Block )
{
MinCell[1] = MinBox[1];
MaxCell[1] = MaxBox[1];
for(IndexType II = III + Box.Axis[1].Begin() ; II <= III + Box.Axis[1].End() ; II += Box.Axis[1].Block )
{
MinCell[0] = MinBox[0];
MaxCell[0] = MaxBox[0];
for(IndexType I = II + Box.Axis[0].Begin() ; I <= II + Box.Axis[0].End() ; I += Box.Axis[0].Block )
{
if(TConfigure::IntersectionBox(ThisObject, MinCell, MaxCell))
{
mCells[I].SearchObjects(ThisObject, Result);
}
}
}
}
}


void SearchObjectLocalExclusive(PointerType& ThisObject, ResultIteratorType& Result, SizeType& NumberOfResults, const SizeType& MaxNumberOfResults,
SearchStructure<IndexType,SizeType,CoordinateType,IteratorType,IteratorIteratorType,1>& Box )
{
PointType  MinCell, MaxCell;
PointType  MinBox, MaxBox;
MinCell[0] = static_cast<CoordinateType>(Box.Axis[0].Min) * mCellSize[0] + mMinPoint[0];  
MaxCell[0] = MinCell[0] + mCellSize[0];
for(IndexType I = Box.Axis[0].Begin() ; I <= Box.Axis[0].End() ; I += Box.Axis[0].Block, MinCell[0] += mCellSize[0], MaxCell[0] += mCellSize[0])
if(TConfigure::IntersectionBox(ThisObject, MinCell, MaxCell))
mCells[I].SearchObjectsExclusive(ThisObject, Result, NumberOfResults, MaxNumberOfResults);
}

void SearchObjectLocalExclusive(PointerType& ThisObject, ResultIteratorType& Result, SizeType& NumberOfResults, const SizeType& MaxNumberOfResults,
SearchStructure<IndexType,SizeType,CoordinateType,IteratorType,IteratorIteratorType,2>& Box )
{
PointType  MinCell, MaxCell;
PointType  MinBox, MaxBox;

for(SizeType i = 0; i < 2; i++)
{
MinBox[i] = static_cast<CoordinateType>(Box.Axis[i].Min) * mCellSize[i] + mMinPoint[i];
MaxBox[i] = MinBox[i] + mCellSize[i];
}

MinCell[1] = MinBox[1];
MaxCell[1] = MaxBox[1];
for(IndexType II = Box.Axis[1].Begin() ; II <= Box.Axis[1].End() ; II += Box.Axis[1].Block, MinCell[1] += mCellSize[1], MaxCell[1] += mCellSize[1] )
{
MinCell[0] = MinBox[0];
MaxCell[0] = MaxBox[0];
for(IndexType I = II + Box.Axis[0].Begin() ; I <= II + Box.Axis[0].End() ; I += Box.Axis[0].Block, MinCell[0] += mCellSize[0], MaxCell[0] += mCellSize[0] )
{
if(TConfigure::IntersectionBox(ThisObject, MinCell, MaxCell))
mCells[I].SearchObjectsExclusive(ThisObject, Result, NumberOfResults, MaxNumberOfResults);
}
}
}

void SearchObjectLocalExclusive(PointerType& ThisObject, ResultIteratorType& Result,
SizeType& NumberOfResults, const SizeType& MaxNumberOfResults,
SearchStructure<IndexType,SizeType,CoordinateType,IteratorType,IteratorIteratorType,3>& Box )
{

PointType  MinCell, MaxCell;
PointType  MinBox, MaxBox;

for(SizeType i = 0; i < 3; i++)
{
MinBox[i] = static_cast<CoordinateType>(Box.Axis[i].Min) * mCellSize[i] + mMinPoint[i];  
MaxBox[i] = MinBox[i] + mCellSize[i];
}

MinCell[2] = MinBox[2];
MaxCell[2] = MaxBox[2];
for(IndexType III = Box.Axis[2].Begin() ; III <= Box.Axis[2].End() ; III += Box.Axis[2].Block, MinCell[2] += mCellSize[2], MaxCell[2] += mCellSize[2] )
{
MinCell[1] = MinBox[1];
MaxCell[1] = MaxBox[1];
for(IndexType II = III + Box.Axis[1].Begin() ; II <= III + Box.Axis[1].End() ; II += Box.Axis[1].Block, MinCell[1] += mCellSize[1], MaxCell[1] += mCellSize[1] )
{
MinCell[0] = MinBox[0];
MaxCell[0] = MaxBox[0];
for(IndexType I = II + Box.Axis[0].Begin() ; I <= II + Box.Axis[0].End() ; I += Box.Axis[0].Block, MinCell[0] += mCellSize[0], MaxCell[0] += mCellSize[0] )
{
if(TConfigure::IntersectionBox(ThisObject, MinCell, MaxCell))
{
mCells[I].SearchObjectsExclusive(ThisObject, Result, NumberOfResults, MaxNumberOfResults);
}
}
}
}
}


void SearchObjectLocalExclusive(PointerType& ThisObject, ResultContainerType& Result,
SearchStructure<IndexType,SizeType,CoordinateType,IteratorType,IteratorIteratorType,1>& Box )
{
PointType  MinCell, MaxCell;
PointType  MinBox, MaxBox;
MinCell[0] = static_cast<CoordinateType>(Box.Axis[0].Min) * mCellSize[0] + mMinPoint[0];  
MaxCell[0] = MinCell[0] + mCellSize[0];
for(IndexType I = Box.Axis[0].Begin() ; I <= Box.Axis[0].End() ; I += Box.Axis[0].Block )
if(TConfigure::IntersectionBox(ThisObject, MinCell, MaxCell))
mCells[I].SearchObjectsExclusive(ThisObject, Result);

}

void SearchObjectLocalExclusive(PointerType& ThisObject, ResultContainerType& Result,
SearchStructure<IndexType,SizeType,CoordinateType,IteratorType,IteratorIteratorType,2>& Box )
{
PointType  MinCell, MaxCell;
PointType  MinBox, MaxBox;

for(SizeType i = 0; i < 2; i++)
{
MinBox[i] = static_cast<CoordinateType>(Box.Axis[i].Min) * mCellSize[i] + mMinPoint[i];
MaxBox[i] = MinBox[i] + mCellSize[i];
}

MinCell[1] = MinBox[1];
MaxCell[1] = MaxBox[1];

for(IndexType II = Box.Axis[1].Begin() ; II <= Box.Axis[1].End() ; II += Box.Axis[1].Block )
{
MinCell[0] = MinBox[0];
MaxCell[0] = MaxBox[0];
for(IndexType I = II + Box.Axis[0].Begin() ; I <= II + Box.Axis[0].End() ; I += Box.Axis[0].Block )
{
if(TConfigure::IntersectionBox(ThisObject, MinCell, MaxCell))
mCells[I].SearchObjectsExclusive(ThisObject, Result);
}
}
}

void SearchObjectLocalExclusive(PointerType& ThisObject, ResultContainerType& Result,
SearchStructure<IndexType,SizeType,CoordinateType,IteratorType,IteratorIteratorType,3>& Box )
{
PointType  MinCell, MaxCell;
PointType  MinBox, MaxBox;

for(SizeType i = 0; i < 3; i++)
{
MinBox[i] = static_cast<CoordinateType>(Box.Axis[i].Min) * mCellSize[i] + mMinPoint[i];  
MaxBox[i] = MinBox[i] + mCellSize[i];
}

MinCell[2] = MinBox[2];
MaxCell[2] = MaxBox[2];

for(IndexType III = Box.Axis[2].Begin() ; III <= Box.Axis[2].End() ; III += Box.Axis[2].Block )
{
MinCell[2] = MinBox[2];
MaxCell[2] = MaxBox[2];
for(IndexType II = III + Box.Axis[1].Begin() ; II <= III + Box.Axis[1].End() ; II += Box.Axis[1].Block )
{
MinCell[0] = MinBox[0];
MaxCell[0] = MaxBox[0];
for(IndexType I = II + Box.Axis[0].Begin() ; I <= II + Box.Axis[0].End() ; I += Box.Axis[0].Block )
{
if(TConfigure::IntersectionBox(ThisObject, MinCell, MaxCell))
{
mCells[I].SearchObjectsExclusive(ThisObject, Result);
}
}
}
}
}


void SearchInRadius(PointerType& ThisObject, CoordinateType const& Radius, ResultIteratorType& Result, SizeType& NumberOfResults, const SizeType& MaxNumberOfResults,
SearchStructure<IndexType,SizeType,CoordinateType,IteratorType,IteratorIteratorType,1>& Box )
{
PointType  MinCell, MaxCell;
PointType  MinBox, MaxBox;

MinCell[0] = static_cast<CoordinateType>(Box.Axis[0].Min) * mCellSize[0] + mMinPoint[0];  
MaxCell[0] = MinCell[0] + mCellSize[0];

for(IndexType I = Box.Axis[0].Begin() ; I <= Box.Axis[0].End() ; I += Box.Axis[0].Block, MinCell[0] += mCellSize[0], MaxCell[0] += mCellSize[0])
if(TConfigure::IntersectionBox(ThisObject, MinCell, MaxCell, Radius))
mCells[I].SearchObjectsInRaius(ThisObject, Radius, Result, NumberOfResults, MaxNumberOfResults);
}

void SearchInRadius(PointerType& ThisObject, CoordinateType const& Radius, ResultIteratorType& Result, SizeType& NumberOfResults, const SizeType& MaxNumberOfResults,
SearchStructure<IndexType,SizeType,CoordinateType,IteratorType,IteratorIteratorType,2>& Box )
{
PointType  MinCell, MaxCell;
PointType  MinBox, MaxBox;

for(SizeType i = 0; i < 2; i++)
{
MinBox[i] = static_cast<CoordinateType>(Box.Axis[i].Min) * mCellSize[i] + mMinPoint[i];
MaxBox[i] = MinBox[i] + mCellSize[i];
}

MinCell[1] = MinBox[1];
MaxCell[1] = MaxBox[1];
for(IndexType II = Box.Axis[1].Begin() ; II <= Box.Axis[1].End() ; II += Box.Axis[1].Block, MinCell[1] += mCellSize[1], MaxCell[1] += mCellSize[1] )
{
MinCell[0] = MinBox[0];
MaxCell[0] = MaxBox[0];
for(IndexType I = II + Box.Axis[0].Begin() ; I <= II + Box.Axis[0].End() ; I += Box.Axis[0].Block, MinCell[0] += mCellSize[0], MaxCell[0] += mCellSize[0] )
{
if(TConfigure::IntersectionBox(ThisObject, MinCell, MaxCell, Radius))
mCells[I].SearchObjectsInRaius(ThisObject, Radius, Result, NumberOfResults, MaxNumberOfResults);
}
}
}

void SearchInRadius(PointerType& ThisObject, CoordinateType const& Radius, ResultIteratorType& Result, SizeType& NumberOfResults, const SizeType& MaxNumberOfResults,
SearchStructure<IndexType,SizeType,CoordinateType,IteratorType,IteratorIteratorType,3>& Box )
{

PointType  MinCell, MaxCell;
PointType  MinBox, MaxBox;

for(SizeType i = 0; i < 3; i++)
{
MinBox[i] = static_cast<CoordinateType>(Box.Axis[i].Min) * mCellSize[i] + mMinPoint[i];  
MaxBox[i] = MinBox[i] + mCellSize[i];
}

MinCell[2] = MinBox[2];
MaxCell[2] = MaxBox[2];
for(IndexType III = Box.Axis[2].Begin() ; III <= Box.Axis[2].End() ; III += Box.Axis[2].Block, MinCell[2] += mCellSize[2], MaxCell[2] += mCellSize[2] )
{
MinCell[1] = MinBox[1];
MaxCell[1] = MaxBox[1];
for(IndexType II = III + Box.Axis[1].Begin() ; II <= III + Box.Axis[1].End() ; II += Box.Axis[1].Block, MinCell[1] += mCellSize[1], MaxCell[1] += mCellSize[1] )
{
MinCell[0] = MinBox[0];
MaxCell[0] = MaxBox[0];
for(IndexType I = II + Box.Axis[0].Begin() ; I <= II + Box.Axis[0].End() ; I += Box.Axis[0].Block, MinCell[0] += mCellSize[0], MaxCell[0] += mCellSize[0] )
{
if(TConfigure::IntersectionBox(ThisObject, MinCell, MaxCell, Radius))
{
mCells[I].SearchObjectsInRadius(ThisObject, Radius, Result, NumberOfResults, MaxNumberOfResults);
}
}
}
}
}


void SearchInRadius(PointerType& ThisObject, CoordinateType const& Radius, ResultIteratorType& Result, DistanceIteratorType ResultDistances, SizeType& NumberOfResults, const SizeType& MaxNumberOfResults,
SearchStructure<IndexType,SizeType,CoordinateType,IteratorType,IteratorIteratorType,1>& Box )
{
PointType  MinCell, MaxCell;
PointType  MinBox, MaxBox;

MinCell[0] = static_cast<CoordinateType>(Box.Axis[0].Min) * mCellSize[0] + mMinPoint[0];  
MaxCell[0] = MinCell[0] + mCellSize[0];

for(IndexType I = Box.Axis[0].Begin() ; I <= Box.Axis[0].End() ; I += Box.Axis[0].Block, MinCell[0] += mCellSize[0], MaxCell[0] += mCellSize[0])
if(TConfigure::IntersectionBox(ThisObject, MinCell, MaxCell, Radius))
mCells[I].SearchObjectsInRaius(ThisObject, Radius, Result, ResultDistances, NumberOfResults, MaxNumberOfResults);
}

void SearchInRadius(PointerType& ThisObject, CoordinateType const& Radius, ResultIteratorType& Result, DistanceIteratorType ResultDistances, SizeType& NumberOfResults, const SizeType& MaxNumberOfResults,
SearchStructure<IndexType,SizeType,CoordinateType,IteratorType,IteratorIteratorType,2>& Box )
{
PointType  MinCell, MaxCell;
PointType  MinBox, MaxBox;

for(SizeType i = 0; i < 2; i++)
{
MinBox[i] = static_cast<CoordinateType>(Box.Axis[i].Min) * mCellSize[i] + mMinPoint[i];
MaxBox[i] = MinBox[i] + mCellSize[i];
}

MinCell[1] = MinBox[1];
MaxCell[1] = MaxBox[1];
for(IndexType II = Box.Axis[1].Begin() ; II <= Box.Axis[1].End() ; II += Box.Axis[1].Block, MinCell[1] += mCellSize[1], MaxCell[1] += mCellSize[1] )
{
MinCell[0] = MinBox[0];
MaxCell[0] = MaxBox[0];
for(IndexType I = II + Box.Axis[0].Begin() ; I <= II + Box.Axis[0].End() ; I += Box.Axis[0].Block, MinCell[0] += mCellSize[0], MaxCell[0] += mCellSize[0] )
{
if(TConfigure::IntersectionBox(ThisObject, MinCell, MaxCell, Radius))
mCells[I].SearchObjectsInRaius(ThisObject, Radius, Result, ResultDistances, NumberOfResults, MaxNumberOfResults);
}
}
}

void SearchInRadius(PointerType& ThisObject, CoordinateType const& Radius, ResultIteratorType& Result, DistanceIteratorType ResultDistances, SizeType& NumberOfResults, const SizeType& MaxNumberOfResults,
SearchStructure<IndexType,SizeType,CoordinateType,IteratorType,IteratorIteratorType,3>& Box )
{

PointType  MinCell, MaxCell;
PointType  MinBox, MaxBox;

for(SizeType i = 0; i < 3; i++)
{
MinBox[i] = static_cast<CoordinateType>(Box.Axis[i].Min) * mCellSize[i] + mMinPoint[i];  
MaxBox[i] = MinBox[i] + mCellSize[i];
}

MinCell[2] = MinBox[2];
MaxCell[2] = MaxBox[2];
for(IndexType III = Box.Axis[2].Begin() ; III <= Box.Axis[2].End() ; III += Box.Axis[2].Block, MinCell[2] += mCellSize[2], MaxCell[2] += mCellSize[2] )
{
MinCell[1] = MinBox[1];
MaxCell[1] = MaxBox[1];
for(IndexType II = III + Box.Axis[1].Begin() ; II <= III + Box.Axis[1].End() ; II += Box.Axis[1].Block, MinCell[1] += mCellSize[1], MaxCell[1] += mCellSize[1] )
{
MinCell[0] = MinBox[0];
MaxCell[0] = MaxBox[0];
for(IndexType I = II + Box.Axis[0].Begin() ; I <= II + Box.Axis[0].End() ; I += Box.Axis[0].Block, MinCell[0] += mCellSize[0], MaxCell[0] += mCellSize[0] )
{
if(TConfigure::IntersectionBox(ThisObject, MinCell, MaxCell, Radius))
{
mCells[I].SearchObjectsInRadius(ThisObject, Radius, Result, ResultDistances, NumberOfResults, MaxNumberOfResults);
}
}
}
}
}


virtual void SearchInRadiusExclusive(PointerType& ThisObject, CoordinateType const& Radius, ResultIteratorType& Result, SizeType& NumberOfResults, const SizeType& MaxNumberOfResults,
SearchStructure<IndexType,SizeType,CoordinateType,IteratorType,IteratorIteratorType,1>& Box )
{
PointType  MinCell, MaxCell;
PointType  MinBox, MaxBox;

MinCell[0] = static_cast<CoordinateType>(Box.Axis[0].Min) * mCellSize[0] + mMinPoint[0];  
MaxCell[0] = MinCell[0] + mCellSize[0];

for(IndexType I = Box.Axis[0].Begin() ; I <= Box.Axis[0].End() ; I += Box.Axis[0].Block, MinCell[0] += mCellSize[0], MaxCell[0] += mCellSize[0])
if(TConfigure::IntersectionBox(ThisObject, MinCell, MaxCell, Radius))
mCells[I].SearchObjectsInRadiusExclusive(ThisObject, Radius, Result, NumberOfResults, MaxNumberOfResults);
}

virtual void SearchInRadiusExclusive(PointerType& ThisObject, CoordinateType const& Radius, ResultIteratorType& Result, SizeType& NumberOfResults, const SizeType& MaxNumberOfResults,
SearchStructure<IndexType,SizeType,CoordinateType,IteratorType,IteratorIteratorType,2>& Box )
{
PointType  MinCell, MaxCell;
PointType  MinBox, MaxBox;

for(SizeType i = 0; i < 2; i++)
{
MinBox[i] = static_cast<CoordinateType>(Box.Axis[i].Min) * mCellSize[i] + mMinPoint[i];
MaxBox[i] = MinBox[i] + mCellSize[i];
}

MinCell[1] = MinBox[1];
MaxCell[1] = MaxBox[1];
for(IndexType II = Box.Axis[1].Begin() ; II <= Box.Axis[1].End() ; II += Box.Axis[1].Block, MinCell[1] += mCellSize[1], MaxCell[1] += mCellSize[1] )
{
MinCell[0] = MinBox[0];
MaxCell[0] = MaxBox[0];
for(IndexType I = II + Box.Axis[0].Begin() ; I <= II + Box.Axis[0].End() ; I += Box.Axis[0].Block, MinCell[0] += mCellSize[0], MaxCell[0] += mCellSize[0] )
{
if(TConfigure::IntersectionBox(ThisObject, MinCell, MaxCell, Radius))
mCells[I].SearchObjectsInRadiusExclusive(ThisObject, Radius, Result, NumberOfResults, MaxNumberOfResults);
}
}
}

virtual void SearchInRadiusExclusive(PointerType& ThisObject, CoordinateType const& Radius, ResultIteratorType& Result, SizeType& NumberOfResults, const SizeType& MaxNumberOfResults,
SearchStructure<IndexType,SizeType,CoordinateType,IteratorType,IteratorIteratorType,3>& Box )
{

PointType  MinCell, MaxCell;
PointType  MinBox, MaxBox;

for(SizeType i = 0; i < 3; i++)
{
MinBox[i] = static_cast<CoordinateType>(Box.Axis[i].Min) * mCellSize[i] + mMinPoint[i];  
MaxBox[i] = MinBox[i] + mCellSize[i];
}

MinCell[2] = MinBox[2];
MaxCell[2] = MaxBox[2];
for(IndexType III = Box.Axis[2].Begin() ; III <= Box.Axis[2].End() ; III += Box.Axis[2].Block, MinCell[2] += mCellSize[2], MaxCell[2] += mCellSize[2] )
{
MinCell[1] = MinBox[1];
MaxCell[1] = MaxBox[1];
for(IndexType II = III + Box.Axis[1].Begin() ; II <= III + Box.Axis[1].End() ; II += Box.Axis[1].Block, MinCell[1] += mCellSize[1], MaxCell[1] += mCellSize[1] )
{
MinCell[0] = MinBox[0];
MaxCell[0] = MaxBox[0];
for(IndexType I = II + Box.Axis[0].Begin() ; I <= II + Box.Axis[0].End() ; I += Box.Axis[0].Block, MinCell[0] += mCellSize[0], MaxCell[0] += mCellSize[0] )
{
if(TConfigure::IntersectionBox(ThisObject, MinCell, MaxCell, Radius))
{
mCells[I].SearchObjectsInRadiusExclusive(ThisObject, Radius, Result, NumberOfResults, MaxNumberOfResults);
}
}
}
}
}


virtual void SearchInRadiusExclusive(PointerType& ThisObject, CoordinateType const& Radius, ResultIteratorType& Result, DistanceIteratorType ResultDistances, SizeType& NumberOfResults, const SizeType& MaxNumberOfResults,
SearchStructure<IndexType,SizeType,CoordinateType,IteratorType,IteratorIteratorType,1>& Box )
{
PointType  MinCell, MaxCell;
PointType  MinBox, MaxBox;

MinCell[0] = static_cast<CoordinateType>(Box.Axis[0].Min) * mCellSize[0] + mMinPoint[0];  
MaxCell[0] = MinCell[0] + mCellSize[0];

for(IndexType I = Box.Axis[0].Begin() ; I <= Box.Axis[0].End() ; I += Box.Axis[0].Block, MinCell[0] += mCellSize[0], MaxCell[0] += mCellSize[0])
if(TConfigure::IntersectionBox(ThisObject, MinCell, MaxCell, Radius))
mCells[I].SearchObjectsInRadiusExclusive(ThisObject, Radius, Result, ResultDistances, NumberOfResults, MaxNumberOfResults);
}

virtual void SearchInRadiusExclusive(PointerType& ThisObject, CoordinateType const& Radius, ResultIteratorType& Result, DistanceIteratorType ResultDistances, SizeType& NumberOfResults, const SizeType& MaxNumberOfResults,
SearchStructure<IndexType,SizeType,CoordinateType,IteratorType,IteratorIteratorType,2>& Box )
{
PointType  MinCell, MaxCell;
PointType  MinBox, MaxBox;

for(SizeType i = 0; i < 2; i++)
{
MinBox[i] = static_cast<CoordinateType>(Box.Axis[i].Min) * mCellSize[i] + mMinPoint[i];
MaxBox[i] = MinBox[i] + mCellSize[i];
}

MinCell[1] = MinBox[1];
MaxCell[1] = MaxBox[1];
for(IndexType II = Box.Axis[1].Begin() ; II <= Box.Axis[1].End() ; II += Box.Axis[1].Block, MinCell[1] += mCellSize[1], MaxCell[1] += mCellSize[1] )
{
MinCell[0] = MinBox[0];
MaxCell[0] = MaxBox[0];
for(IndexType I = II + Box.Axis[0].Begin() ; I <= II + Box.Axis[0].End() ; I += Box.Axis[0].Block, MinCell[0] += mCellSize[0], MaxCell[0] += mCellSize[0] )
{
if(TConfigure::IntersectionBox(ThisObject, MinCell, MaxCell, Radius))
mCells[I].SearchObjectsInRadiusExclusive(ThisObject, Radius, Result, ResultDistances, NumberOfResults, MaxNumberOfResults);
}
}
}

virtual void SearchInRadiusExclusive(PointerType& ThisObject, CoordinateType const& Radius, ResultIteratorType& Result, DistanceIteratorType ResultDistances, SizeType& NumberOfResults, const SizeType& MaxNumberOfResults,
SearchStructure<IndexType,SizeType,CoordinateType,IteratorType,IteratorIteratorType,3>& Box )
{

PointType  MinCell, MaxCell;
PointType  MinBox, MaxBox;

for(SizeType i = 0; i < 3; i++)
{
MinBox[i] = static_cast<CoordinateType>(Box.Axis[i].Min) * mCellSize[i] + mMinPoint[i];  
MaxBox[i] = MinBox[i] + mCellSize[i];
}

MinCell[2] = MinBox[2];
MaxCell[2] = MaxBox[2];
for(IndexType III = Box.Axis[2].Begin() ; III <= Box.Axis[2].End() ; III += Box.Axis[2].Block, MinCell[2] += mCellSize[2], MaxCell[2] += mCellSize[2] )
{
MinCell[1] = MinBox[1];
MaxCell[1] = MaxBox[1];
for(IndexType II = III + Box.Axis[1].Begin() ; II <= III + Box.Axis[1].End() ; II += Box.Axis[1].Block, MinCell[1] += mCellSize[1], MaxCell[1] += mCellSize[1] )
{
MinCell[0] = MinBox[0];
MaxCell[0] = MaxBox[0];
for(IndexType I = II + Box.Axis[0].Begin() ; I <= II + Box.Axis[0].End() ; I += Box.Axis[0].Block, MinCell[0] += mCellSize[0], MaxCell[0] += mCellSize[0] )
{
if(TConfigure::IntersectionBox(ThisObject, MinCell, MaxCell, Radius))
{
mCells[I].SearchObjectsInRadiusExclusive(ThisObject, Radius, Result, ResultDistances, NumberOfResults, MaxNumberOfResults);
}
}
}
}
}

void FillObject( SearchStructure<IndexType,SizeType,CoordinateType,IteratorType,IteratorIteratorType,1>& Box, const PointerType& i_object)
{
PointType  MinCell, MaxCell;

MinCell[0] = static_cast<CoordinateType>(Box.Axis[0].Min) * mCellSize[0] + mMinPoint[0];  
MaxCell[0] = MinCell[0] + mCellSize[0];
for(IndexType I = Box.Axis[0].Begin() ; I <= Box.Axis[0].End() ; I += Box.Axis[0].Block, MinCell[0]+=mCellSize[0], MaxCell[0]+=mCellSize[0] )
{
if(TConfigure::IntersectionBox(i_object, MinCell, MaxCell))
mCells[I].Add(i_object);
}
}

void FillObject( SearchStructure<IndexType,SizeType,CoordinateType,IteratorType,IteratorIteratorType,2>& Box, const PointerType& i_object)
{
PointType  MinCell, MaxCell;
PointType  MinBox, MaxBox;

for(SizeType i = 0; i < 2; i++)
{
MinBox[i] = static_cast<CoordinateType>(Box.Axis[i].Min) * mCellSize[i] + mMinPoint[i];
MaxBox[i] = MinBox[i] + mCellSize[i];
}

MinCell[1] = MinBox[1];
MaxCell[1] = MaxBox[1];
for(IndexType II = Box.Axis[1].Begin() ; II <= Box.Axis[1].End() ; II += Box.Axis[1].Block, MinCell[1]+=mCellSize[1], MaxCell[1]+=mCellSize[1] )
{
MinCell[0] = MinBox[0];
MaxCell[0] = MaxBox[0];
for(IndexType I = II + Box.Axis[0].Begin() ; I <= II + Box.Axis[0].End() ; I += Box.Axis[0].Block, MinCell[0]+=mCellSize[0], MaxCell[0]+=mCellSize[0] )
{
if(TConfigure::IntersectionBox(i_object,MinCell,MaxCell))
mCells[I].Add(i_object);
}
}
}

virtual void FillObject( SearchStructure<IndexType,SizeType,CoordinateType,IteratorType,IteratorIteratorType,3>& Box, const PointerType& i_object)
{
PointType  MinCell, MaxCell;
PointType  MinBox, MaxBox;

for(SizeType i = 0; i < 3; i++)
{
MinBox[i] = static_cast<CoordinateType>(Box.Axis[i].Min) * mCellSize[i] + mMinPoint[i];  
MaxBox[i] = MinBox[i] + mCellSize[i];
}

MinCell[2] = MinBox[2];
MaxCell[2] = MaxBox[2];
for(IndexType III = Box.Axis[2].Begin() ; III <= Box.Axis[2].End() ; III += Box.Axis[2].Block, MinCell[2]+=mCellSize[2], MaxCell[2]+=mCellSize[2] )
{
MinCell[1] = MinBox[1];
MaxCell[1] = MaxBox[1];
for(IndexType II = III + Box.Axis[1].Begin() ; II <= III + Box.Axis[1].End() ; II += Box.Axis[1].Block, MinCell[1]+=mCellSize[1], MaxCell[1]+=mCellSize[1] )
{
MinCell[0] = MinBox[0];
MaxCell[0] = MaxBox[0];
for(IndexType I = II + Box.Axis[0].Begin() ; I <= II + Box.Axis[0].End() ; I += Box.Axis[0].Block, MinCell[0]+=mCellSize[0], MaxCell[0]+=mCellSize[0] )
{
if(TConfigure::IntersectionBox(i_object,MinCell,MaxCell))
mCells[I].Add(i_object);
}
}
}
}

void RemoveObjectLocal( SearchStructure<IndexType,SizeType,CoordinateType,IteratorType,IteratorIteratorType,1>& Box, const PointerType& i_object)
{
PointType  MinCell, MaxCell;

MinCell[0] = static_cast<CoordinateType>(Box.Axis[0].Min) * mCellSize[0] + mMinPoint[0];  
MaxCell[0] = MinCell[0] + mCellSize[0];
for(IndexType I = Box.Axis[0].Begin() ; I <= Box.Axis[0].End() ; I += Box.Axis[0].Block, MinCell[0]+=mCellSize[0], MaxCell[0]+=mCellSize[0] )
{
if(TConfigure::IntersectionBox(i_object, MinCell, MaxCell))
mCells[I].Remove(i_object);
}
}

void RemoveObjectLocal( SearchStructure<IndexType,SizeType,CoordinateType,IteratorType,IteratorIteratorType,2>& Box, const PointerType& i_object)
{
PointType  MinCell, MaxCell;
PointType  MinBox, MaxBox;

for(SizeType i = 0; i < 2; i++)
{
MinBox[i] = static_cast<CoordinateType>(Box.Axis[i].Min) * mCellSize[i] + mMinPoint[i];
MaxBox[i] = MinBox[i] + mCellSize[i];
}

MinCell[1] = MinBox[1];
MaxCell[1] = MaxBox[1];
for(IndexType II = Box.Axis[1].Begin() ; II <= Box.Axis[1].End() ; II += Box.Axis[1].Block, MinCell[1]+=mCellSize[1], MaxCell[1]+=mCellSize[1] )
{
MinCell[0] = MinBox[0];
MaxCell[0] = MaxBox[0];
for(IndexType I = II + Box.Axis[0].Begin() ; I <= II + Box.Axis[0].End() ; I += Box.Axis[0].Block, MinCell[0]+=mCellSize[0], MaxCell[0]+=mCellSize[0] )
{
if(TConfigure::IntersectionBox(i_object,MinCell,MaxCell))
mCells[I].Remove(i_object);
}
}
}

void RemoveObjectLocal( SearchStructure<IndexType,SizeType,CoordinateType,IteratorType,IteratorIteratorType,3>& Box, const PointerType& i_object)
{
PointType  MinCell, MaxCell;
PointType  MinBox, MaxBox;

for(SizeType i = 0; i < 3; i++)
{
MinBox[i] = static_cast<CoordinateType>(Box.Axis[i].Min) * mCellSize[i] + mMinPoint[i];  
MaxBox[i] = MinBox[i] + mCellSize[i];
}

MinCell[2] = MinBox[2];
MaxCell[2] = MaxBox[2];
for(IndexType III = Box.Axis[2].Begin() ; III <= Box.Axis[2].End() ; III += Box.Axis[2].Block, MinCell[2]+=mCellSize[2], MaxCell[2]+=mCellSize[2] )
{
MinCell[1] = MinBox[1];
MaxCell[1] = MaxBox[1];
for(IndexType II = III + Box.Axis[1].Begin() ; II <= III + Box.Axis[1].End() ; II += Box.Axis[1].Block, MinCell[1]+=mCellSize[1], MaxCell[1]+=mCellSize[1] )
{
MinCell[0] = MinBox[0];
MaxCell[0] = MaxBox[0];
for(IndexType I = II + Box.Axis[0].Begin() ; I <= II + Box.Axis[0].End() ; I += Box.Axis[0].Block, MinCell[0]+=mCellSize[0], MaxCell[0]+=mCellSize[0] )
{
if(TConfigure::IntersectionBox(i_object,MinCell,MaxCell))
mCells[I].Remove(i_object);
}
}
}
}

void AllocateContainer()
{
SizeType Size = mN[0];
for(SizeType i = 1 ; i < Dimension ; i++)
Size *= mN[i];
mCells.resize(Size);
}

inline void CreatePartition(SizeType number_of_threads, const SizeType number_of_rows, std::vector<SizeType>& partitions)
{
partitions.resize(number_of_threads+1);
SizeType partition_size = number_of_rows / number_of_threads;
partitions[0] = 0;
partitions[number_of_threads] = number_of_rows;
for(SizeType i = 1; i<number_of_threads; i++)
partitions[i] = partitions[i-1] + partition_size ;
}













PointType    mMinPoint;
PointType    mMaxPoint;

SizeType     mObjectsSize;
IteratorType mObjectsBegin;
IteratorType mObjectsEnd;

CoordinateArray  mCellSize;
CoordinateArray  mInvCellSize;
SizeArray        mN;

CellContainerType mCells;  

private:












public:
BinsObjectDynamic<TConfigure> & operator=(const BinsObjectDynamic<TConfigure> & rOther)
{
mMinPoint            = rOther.mMinPoint;
mMaxPoint            = rOther.mMaxPoint;
mObjectsBegin        = rOther.mObjectsBegin;
mObjectsEnd          = rOther.mObjectsEnd;
mObjectsSize         = rOther.mObjectsSize;
mCellSize            = rOther.mCellSize;
mInvCellSize         = rOther.mInvCellSize;
mN                   = rOther.mN;
mCells               = rOther.mCells;
return *this;
}

BinsObjectDynamic(const BinsObjectDynamic& rOther)
{
*this =  rOther;
}

template<class T>
BinsObjectDynamic(const BinsObjectDynamic<T>& rOther)
{
*this =  rOther;
}

}; 






template<class TConfigure>
inline std::istream& operator >> (std::istream& rIStream,
BinsObjectDynamic<TConfigure>& rThis)
{
return rIStream;
}


template<class TConfigure>
inline std::ostream& operator << (std::ostream& rOStream,
const BinsObjectDynamic<TConfigure> & rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}


}  
