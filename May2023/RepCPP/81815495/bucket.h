
#pragma once



#include "tree.h"

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
class Bucket  : public TreeNode<TDimension,TPointType,TPointerType,TIteratorType,TDistanceIteratorType>
{
public:

KRATOS_CLASS_POINTER_DEFINITION(Bucket);

typedef TreeNode<TDimension, TPointType, TPointerType, TIteratorType, TDistanceIteratorType> BaseType;

typedef TPointType PointType;

typedef TContainerType ContainerType;

typedef TIteratorType IteratorType;

typedef TDistanceIteratorType DistanceIteratorType;

typedef TPointerType PointerType;

typedef TDistanceFunction  DistanceFunction;

typedef typename BaseType::SizeType       SizeType;

typedef typename BaseType::IndexType      IndexType;

typedef typename BaseType::CoordinateType CoordinateType;

enum { Dimension = TDimension };

typedef typename BaseType::SearchStructureType SearchStructureType;

typedef Kratos::SearchUtils::SearchNearestInRange<PointType,PointerType,IteratorType,DistanceFunction,CoordinateType> SearchNearestInRange;
typedef Kratos::SearchUtils::SearchRadiusInRange<PointType,IteratorType,DistanceIteratorType,DistanceFunction,SizeType,CoordinateType> SearchRadiusInRange;
typedef Kratos::SearchUtils::SearchBoxInRange<PointType,IteratorType,SizeType,TDimension> SearchBoxInRange;



Bucket()
: mPointsBegin(this->NullIterator()), mPointsEnd(this->NullIterator())
{}

Bucket(IteratorType PointsBegin,IteratorType PointsEnd)
: mPointsBegin(PointsBegin), mPointsEnd(PointsEnd)
{}

virtual ~Bucket() {}







IteratorType Begin()
{
return mPointsBegin;
}

IteratorType End()
{
return mPointsEnd;
}

void SearchNearestPoint(PointType const& ThisPoint, PointerType& rResult, CoordinateType& rResultDistance ) override
{
if(mPointsBegin == mPointsEnd)
return;
SearchNearestInRange()( mPointsBegin, mPointsEnd, ThisPoint, rResult, rResultDistance );
}

void SearchNearestPoint(PointType const& ThisPoint, PointerType& Result, CoordinateType& ResultDistance, SearchStructureType& Auxiliar ) override
{
SearchNearestPoint(ThisPoint,Result,ResultDistance);
}

void SearchInRadius(PointType const& ThisPoint, CoordinateType const& Radius, CoordinateType const& Radius2, IteratorType& Results,
DistanceIteratorType& ResultsDistances, SizeType& NumberOfResults, SizeType const& MaxNumberOfResults) override
{
if(mPointsBegin == mPointsEnd)
return;
SearchRadiusInRange()(mPointsBegin,mPointsEnd,ThisPoint,Radius2,Results,ResultsDistances,NumberOfResults,MaxNumberOfResults);
}

void SearchInRadius(PointType const& ThisPoint, CoordinateType const& Radius, CoordinateType const& Radius2, IteratorType& Results,
DistanceIteratorType& ResultsDistances, SizeType& NumberOfResults, SizeType const& MaxNumberOfResults, SearchStructureType& Auxiliar) override
{
SearchInRadius(ThisPoint,Radius,Radius2,Results,ResultsDistances,NumberOfResults,MaxNumberOfResults);
}


void SearchInRadius(PointType const& ThisPoint, CoordinateType const& Radius, CoordinateType const& Radius2, IteratorType& Results,
SizeType& NumberOfResults, SizeType const& MaxNumberOfResults) override
{
if(mPointsBegin == mPointsEnd)
return;
SearchRadiusInRange()(mPointsBegin,mPointsEnd,ThisPoint,Radius2,Results,NumberOfResults,MaxNumberOfResults);
}

void SearchInRadius(PointType const& ThisPoint, CoordinateType const& Radius, CoordinateType const& Radius2, IteratorType& Results,
SizeType& NumberOfResults, SizeType const& MaxNumberOfResults, SearchStructureType& Auxiliar) override
{
SearchInRadius(ThisPoint,Radius,Radius2,Results,NumberOfResults,MaxNumberOfResults);
}

void SearchInBox(PointType const& SearchMinPoint, PointType const& SearchMaxPoint, IteratorType& Results, SizeType& NumberOfResults,
SizeType const& MaxNumberOfResults ) override
{
SearchBoxInRange()(SearchMinPoint,SearchMaxPoint,mPointsBegin,mPointsEnd,Results,NumberOfResults,MaxNumberOfResults);
}







virtual std::string Info() const
{
return "Bucket";
}

virtual void PrintInfo(std::ostream& rOStream) const
{
rOStream << "Bucket";
}

void PrintData(std::ostream& rOStream, std::string const& Perfix = std::string()) const override
{
rOStream << Perfix << "Leaf[" << SearchUtils::PointerDistance(mPointsBegin, mPointsEnd) << "] : ";
for(IteratorType i = mPointsBegin ; i != mPointsEnd ; i++)
rOStream << **i << "    ";
rOStream << std::endl;
}





protected:






















private:


IteratorType mPointsBegin;
IteratorType mPointsEnd;











Bucket& operator=(Bucket const& rOther);

Bucket(Bucket const& rOther);



}; 






template<
std::size_t TDimension,
class TPointType,
class TContainerType,
class TPointerType,
class TIteratorType,
class TDistanceIteratorType,
class TDistanceFunction >
inline std::istream& operator >> (std::istream& rIStream,
Bucket<TDimension, TPointType, TContainerType, TPointerType, TIteratorType, TDistanceIteratorType, TDistanceFunction>& rThis);

template<
std::size_t TDimension,
class TPointType,
class TContainerType,
class TPointerType,
class TIteratorType,
class TDistanceIteratorType,
class TDistanceFunction >
inline std::ostream& operator << (std::ostream& rOStream,
const Bucket<TDimension, TPointType, TContainerType, TPointerType, TIteratorType, TDistanceIteratorType, TDistanceFunction>& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}


}  


