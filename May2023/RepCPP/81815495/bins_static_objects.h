
#pragma once

#include <string>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <array>

#include "tree.h"

#ifdef _OPENMP
#include <omp.h>
#endif


namespace Kratos
{








template<class TConfigure>
class BinsObjectStatic
{
public:

enum { Dimension = TConfigure::Dimension };

typedef TConfigure                                  Configure;
typedef typename TConfigure::PointType              PointType;
typedef typename TConfigure::PointerType            PointerType;
typedef typename TConfigure::ContainerType          ContainerType;
typedef typename TConfigure::IteratorType           IteratorType;
typedef typename TConfigure::ResultContainerType    ResultContainerType;
typedef typename TConfigure::ResultIteratorType     ResultIteratorType;

typedef TreeNode<Dimension, PointType, PointerType, IteratorType,  typename TConfigure::DistanceIteratorType> TreeNodeType;

typedef typename TreeNodeType::CoordinateType       CoordinateType;  
typedef typename TreeNodeType::SizeType             SizeType;        
typedef typename TreeNodeType::IndexType            IndexType;       

typedef Tvector<CoordinateType,Dimension>           CoordinateArray;
typedef Tvector<SizeType,Dimension>                 SizeArray;
typedef Tvector<IndexType,Dimension>                IndexArray;

typedef typename TreeNodeType::IteratorIteratorType IteratorIteratorType;
typedef typename TreeNodeType::SearchStructureType  SearchStructureType;

typedef std::vector<PointerType>                    LocalContainerType;
typedef typename LocalContainerType::iterator       LocalIteratorType;

typedef typename TConfigure::ContainerContactType   ContainerContactType;
typedef typename TConfigure::IteratorContactType    IteratorContactType;

typedef std::vector<IndexType>                      IndexContainer;
typedef typename IndexContainer::iterator           IndexIterator;

typedef std::vector<IteratorType>                   IteratorVector;
typedef typename IteratorVector::iterator           IteratorIterator;
typedef typename IteratorVector::const_iterator     IteratorConstIterator;

KRATOS_CLASS_POINTER_DEFINITION(BinsObjectStatic);


BinsObjectStatic() {}

BinsObjectStatic (IteratorType const& ObjectsBegin, IteratorType const& ObjectsEnd)
: mObjectsBegin(ObjectsBegin), mObjectsEnd(ObjectsEnd)
{
auto mNumPoints = std::distance(mObjectsBegin, mObjectsEnd);

CalculateBoundingBox();
CalculateCellSize(mNumPoints);
GenerateBins();
}

BinsObjectStatic (IteratorType const& ObjectsBegin, IteratorType const& ObjectsEnd, const SizeType Nx, const SizeType Ny, const SizeType Nz )
: mObjectsBegin(ObjectsBegin), mObjectsEnd(ObjectsEnd)
{
CalculateBoundingBox();

mN[0] = Nx;
mN[1] = Ny;
mN[2] = Nz;

double delta[Dimension];
SizeType index = 0;
for(SizeType i = 0 ; i < Dimension ; i++)
{
delta[i] = mMaxPoint[i] - mMinPoint[i];
if ( delta[i] > delta[index] )
index = i;
delta[i] = (delta[i] == 0.00) ? 1.00 : delta[i];
}

for(SizeType i = 0 ; i < Dimension ; i++)
{
mCellSize[i] = delta[i] / mN[i];
mInvCellSize[i] = 1.00 / mCellSize[i];
}

GenerateBins();
}


virtual ~BinsObjectStatic() {}











virtual std::string Info() const
{
return "BinsObjectStatic : ";
}

virtual void PrintInfo(std::ostream& rOStream) const
{
rOStream << Info();
}

virtual void PrintData(std::ostream& rOStream) const
{
rOStream << "   Container Size: ";
for(SizeType i = 0 ; i < Dimension ; i++)
rOStream << "[" << mN[i] << "]";
rOStream << std::endl;
rOStream << "   Cell Size: ";
for(SizeType i = 0 ; i < Dimension ; i++)
rOStream << "[" << mCellSize[i] << "]";
rOStream << std::endl;
rOStream << "   Contained Objects: " << SearchUtils::PointerDistance(mObjectsBegin,mObjectsEnd) << std::endl;
rOStream << "   Total Object Storaged: " << mObjectList.size() << std::endl;
}

void PrintSize( std::ostream& rout )
{
rout << "  Container Size: ";
for(SizeType i = 0 ; i < Dimension ; i++)
rout << "[" << this->mN[i] << "]";
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
/

void SearchObjectRow(PointerType& ThisObject, LocalIteratorType RowBegin, LocalIteratorType RowEnd, ResultIteratorType& Result, SizeType& NumberOfResults, const SizeType& MaxNumberOfResults)
{
for(LocalIteratorType iter = RowBegin ; iter != RowEnd && NumberOfResults < MaxNumberOfResults ; iter++)
{
if(TConfigure::Intersection(ThisObject,*iter))
{
if( std::find(Result-NumberOfResults, Result, *iter) == Result )
{
*Result = *iter;
Result++;
NumberOfResults++;
}
}
}
}


void SearchInBoxLocal(PointerType& ThisObject, ResultIteratorType& Result, SizeType& NumberOfResults, const SizeType& MaxNumberOfResults,
SearchStructure<IndexType,SizeType,CoordinateType,IteratorType,IteratorIteratorType,1>& Box )
{
SearchObjectRow(ThisObject,mObjectList.begin()+mObjectsAccess[Box.Axis[0].Begin()],mObjectList.begin()+mObjectsAccess[Box.Axis[0].End()+1],Result,NumberOfResults,MaxNumberOfResults);
}

void SearchInBoxLocal(PointerType& ThisObject, ResultIteratorType& Result, SizeType& NumberOfResults, const SizeType& MaxNumberOfResults,
SearchStructure<IndexType,SizeType,CoordinateType,IteratorType,IteratorIteratorType,2>& Box )
{
for(IndexType I = Box.Axis[1].Begin() ; I <= Box.Axis[1].End() ; I += Box.Axis[1].Block )
SearchObjectRow(ThisObject,mObjectList.begin()+mObjectsAccess[I+Box.Axis[0].Begin()],mObjectList.begin()+mObjectsAccess[I+Box.Axis[0].End()+1],Result,NumberOfResults,MaxNumberOfResults);
}

void SearchInBoxLocal(PointerType& ThisObject, ResultIteratorType& Result, SizeType& NumberOfResults, const SizeType& MaxNumberOfResults,
SearchStructure<IndexType,SizeType,CoordinateType,IteratorType,IteratorIteratorType,3>& Box )
{
for(IndexType II = Box.Axis[2].Begin() ; II <= Box.Axis[2].End() ; II += Box.Axis[2].Block )
for(IndexType I = II + Box.Axis[1].Begin() ; I <= II + Box.Axis[1].End() ; I += Box.Axis[1].Block )
SearchObjectRow(ThisObject,mObjectList.begin()+mObjectsAccess[I+Box.Axis[0].Begin()],mObjectList.begin()+mObjectsAccess[I+Box.Axis[0].End()+1],Result,NumberOfResults,MaxNumberOfResults);
}

void SearchInBoxLocal_(PointerType& ThisObject, ResultIteratorType& Result, SizeType& NumberOfResults, const SizeType& MaxNumberOfResults,
SearchStructure<IndexType,SizeType,CoordinateType,IteratorType,IteratorIteratorType,3>& Box )
{
PointType  MinCell, MaxCell;
PointType  MinBox, MaxBox;
IndexType  objects_begin, objects_end;

for(SizeType i = 0; i < 3; i++)
{
MinBox[i] = static_cast<CoordinateType>(Box.Axis[i].Min) * mCellSize[i] + mMinPoint[i];  
MaxBox[i] = MinBox[i] + mCellSize[i];
}
CoordinateType MaxBox_ = static_cast<CoordinateType>(Box.Axis[0].Min+1) * mCellSize[0] + mMinPoint[0];  
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
objects_begin = mObjectsAccess[II + Box.Axis[0].Begin()];
for(IndexType I = II + Box.Axis[0].Begin() ; I <= II + Box.Axis[0].End() ; I += Box.Axis[0].Block, MinCell[0]+=mCellSize[0], MaxCell[0]+=mCellSize[0] )
{
if(TConfigure::IntersectionBox(ThisObject,MinCell,MaxCell))
{
objects_begin = mObjectsAccess[I];
break;
}
}
MinCell[0] = MaxBox_-mCellSize[0];
MaxCell[0] = MaxBox_;
objects_end = mObjectsAccess[II+Box.Axis[0].End()+1];
for(IndexType I = II + Box.Axis[0].End() ; I >= II + Box.Axis[0].Begin() ; I -= Box.Axis[0].Block, MinCell[0]-=mCellSize[0], MaxCell[0]-=mCellSize[0] )
{
if(TConfigure::IntersectionBox(ThisObject,MinCell,MaxCell))
{
objects_end = mObjectsAccess[I+1];
break;
}
}
SearchObjectRow(ThisObject,mObjectList.begin()+objects_begin,mObjectList.begin()+objects_end,Result,NumberOfResults,MaxNumberOfResults);
}
}
}



void SearchObjectRow(PointerType& ThisObject, LocalIteratorType RowBegin, LocalIteratorType RowEnd, ResultContainerType& Results )
{
for(LocalIteratorType iter = RowBegin ; iter != RowEnd ; iter++)
{
if(TConfigure::Intersection(ThisObject,*iter))
if( std::find(Results.begin(), Results.end(), *iter) == Results.end() )
Results.push_back(*iter);
}
}


void SearchInBoxLocal(PointerType& ThisObject, ResultContainerType& Results,
SearchStructure<IndexType,SizeType,CoordinateType,IteratorType,IteratorIteratorType,1>& Box )
{
SearchObjectRow(ThisObject,mObjectList.begin()+mObjectsAccess[Box.Axis[0].Begin()],mObjectList.begin()+mObjectsAccess[Box.Axis[0].End()+1],Results);
}

void SearchInBoxLocal(PointerType& ThisObject, ResultContainerType& Results,
SearchStructure<IndexType,SizeType,CoordinateType,IteratorType,IteratorIteratorType,2>& Box )
{
for(IndexType I = Box.Axis[1].Begin() ; I <= Box.Axis[1].End() ; I += Box.Axis[1].Block )
SearchObjectRow(ThisObject,mObjectList.begin()+mObjectsAccess[I+Box.Axis[0].Begin()],mObjectList.begin()+mObjectsAccess[I+Box.Axis[0].End()+1],Results);
}

void SearchInBoxLocal(PointerType& ThisObject, ResultContainerType& Results,
SearchStructure<IndexType,SizeType,CoordinateType,IteratorType,IteratorIteratorType,3>& Box )
{
for(IndexType II = Box.Axis[2].Begin() ; II <= Box.Axis[2].End() ; II += Box.Axis[2].Block )
for(IndexType I = II + Box.Axis[1].Begin() ; I <= II + Box.Axis[1].End() ; I += Box.Axis[1].Block )
SearchObjectRow(ThisObject,mObjectList.begin()+mObjectsAccess[I+Box.Axis[0].Begin()],mObjectList.begin()+mObjectsAccess[I+Box.Axis[0].End()+1],Results);
}




Tvector<IndexType,Dimension>  CalculateCell( const PointType& ThisPoint )
{
Tvector<IndexType,Dimension>  Cell;
for(SizeType i = 0 ; i < Dimension ; i++)
Cell[i] = CalculatePosition(ThisPoint[i],i);
return Cell;
}



void FillObject( SearchStructure<IndexType,SizeType,CoordinateType,IteratorType,IteratorIteratorType,1>& Box, const PointerType& object)
{
PointType  MinCell, MaxCell;

MinCell[0] = static_cast<CoordinateType>(Box.Axis[0].Min) * mCellSize[0] + mMinPoint[0];  
MaxCell[0] = MinCell[0] + mCellSize[0];
for(IndexType I = Box.Axis[0].Begin() ; I <= Box.Axis[0].End() ; I += Box.Axis[0].Block, MinCell[0]+=mCellSize[0], MaxCell[0]+=mCellSize[0] )
if(TConfigure::IntersectionBox(object,MinCell,MaxCell))
mObjectList[mObjectsAccess[I]++] = object;
}


void FillObject( SearchStructure<IndexType,SizeType,CoordinateType,IteratorType,IteratorIteratorType,2>& Box, const PointerType& object)
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
if(TConfigure::IntersectionBox(object,MinCell,MaxCell))
mObjectList[mObjectsAccess[I]++] = object;
}
}


void FillObject( SearchStructure<IndexType,SizeType,CoordinateType,IteratorType,IteratorIteratorType,3>& Box, const PointerType&  object)
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
if(TConfigure::IntersectionBox(object,MinCell,MaxCell))
mObjectList[mObjectsAccess[I]++] = object;
}
}
}



IndexType CalculatePosition( CoordinateType const& ThisCoord, SizeType& ThisDimension )
{
CoordinateType d_index = (ThisCoord - mMinPoint[ThisDimension]) * mInvCellSize[ThisDimension];
IndexType index = static_cast<IndexType>( (d_index < 0.00) ? 0.00 : d_index );
return  (index > mN[ThisDimension]-1) ? mN[ThisDimension]-1 : index;

}













private:



PointType    mMinPoint;
PointType    mMaxPoint;

IteratorType mObjectsBegin;
IteratorType mObjectsEnd;

CoordinateArray mCellSize;
CoordinateArray mInvCellSize;
SizeArray       mN;

LocalContainerType  mObjectList;
IndexContainer      mObjectsAccess;







inline void CreatePartition(unsigned int number_of_threads, const int number_of_rows, std::vector<unsigned int>& partitions)
{
partitions.resize(number_of_threads+1);
int partition_size = number_of_rows / number_of_threads;
partitions[0] = 0;
partitions[number_of_threads] = number_of_rows;
for(unsigned int i = 1; i<number_of_threads; i++)
partitions[i] = partitions[i-1] + partition_size ;
}








BinsObjectStatic& operator=(BinsObjectStatic const& rOther) {}

BinsObjectStatic(BinsObjectStatic const& rOther) {}




}; 






template<class TConfigure>
inline std::istream& operator >> (std::istream& rIStream,BinsObjectStatic<TConfigure>& rThis)
{
return rIStream;
}


template<class TConfigure>
inline std::ostream& operator << (std::ostream& rOStream,
const BinsObjectStatic<TConfigure> & rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}


}  


