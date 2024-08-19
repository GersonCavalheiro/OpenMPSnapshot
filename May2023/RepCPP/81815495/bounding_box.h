
#pragma once



#include "includes/define.h"

namespace Kratos
{



template <typename TPointType>
class BoundingBox 
{
public:

KRATOS_CLASS_POINTER_DEFINITION(BoundingBox);


BoundingBox()
{
std::fill(GetMinPoint().begin(), GetMinPoint().end(), 0.0);
std::fill(GetMaxPoint().begin(), GetMaxPoint().end(), 0.0);
};

BoundingBox(TPointType const& MinPoint, TPointType const& MaxPoint) :
mMinMaxPoints{MinPoint,MaxPoint} {}

BoundingBox( const BoundingBox &Other) :
mMinMaxPoints(Other.mMinMaxPoints) {}


template<typename TIteratorType>
BoundingBox(TIteratorType const& PointsBegin, TIteratorType const& PointsEnd) 
{
Set(PointsBegin, PointsEnd);
}

virtual ~BoundingBox(){}


BoundingBox& operator=(BoundingBox const& rOther)
{
GetMinPoint() = rOther.GetMinPoint();
GetMaxPoint() = rOther.GetMaxPoint();

return *this;
}


template<typename TIteratorType>
void Set(TIteratorType const& PointsBegin, TIteratorType const& PointsEnd)
{
if (PointsBegin == PointsEnd) {
std::fill(GetMinPoint().begin(), GetMinPoint().end(), 0.0);
std::fill(GetMaxPoint().begin(), GetMaxPoint().end(), 0.0);
return;
}

for (unsigned int i = 0; i < Dimension; i++) {
GetMinPoint()[i] = (*PointsBegin)[i];
GetMaxPoint()[i] = (*PointsBegin)[i];
}

Extend(PointsBegin, PointsEnd);
}

template<typename TIteratorType>
void Extend(TIteratorType const& PointsBegin, TIteratorType const& PointsEnd)
{
for (TIteratorType i_point = PointsBegin; i_point != PointsEnd; i_point++){
for (unsigned int i = 0; i < Dimension; i++) {
if ((*i_point)[i] < GetMinPoint()[i]) GetMinPoint()[i] = (*i_point)[i];
if ((*i_point)[i] > GetMaxPoint()[i]) GetMaxPoint()[i] = (*i_point)[i];
}
}
}

void Extend(const double Margin)
{
for (unsigned int i = 0; i < Dimension; i++){
GetMinPoint()[i] -= Margin;
GetMaxPoint()[i] += Margin;
}

}


TPointType& GetMinPoint() { return mMinMaxPoints[0]; }

TPointType const& GetMinPoint() const { return mMinMaxPoints[0]; }

TPointType& GetMaxPoint() { return mMinMaxPoints[1]; }

TPointType const& GetMaxPoint() const { return mMinMaxPoints[1]; }



virtual std::string Info() const
{
std::stringstream buffer;
buffer << "BoundingBox" ;
return buffer.str();
}

virtual void PrintInfo(std::ostream& rOStream) const {rOStream << "BoundingBox";}

virtual void PrintData(std::ostream& rOStream) const {
rOStream << "   MinPoint : [" << GetMinPoint()[0] << ","  << GetMinPoint()[1] << ","  << GetMinPoint()[2] << "]" << std::endl;
rOStream << "   MaxPoint : [" << GetMaxPoint()[0] << ","  << GetMaxPoint()[1] << ","  << GetMaxPoint()[2] << "]" << std::endl;
}


private:

static constexpr unsigned int Dimension = 3;


std::array<TPointType, 2> mMinMaxPoints;  


}; 



template <typename TPointType>
inline std::istream& operator >> (std::istream& rIStream,
BoundingBox<TPointType>& rThis){
return rIStream;
}

template <typename TPointType>
inline std::ostream& operator << (std::ostream& rOStream,
const BoundingBox<TPointType>& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}


}  
