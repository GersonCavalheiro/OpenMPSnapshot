
#pragma once

#include <string>
#include <iostream>
#include <cmath>
#include <algorithm>



namespace Kratos
{






template< class  TConfigure>
class Cell
{
public:

typedef std::size_t  SizeType;
typedef typename TConfigure::PointType               PointType;
typedef typename TConfigure::PointerType             PointerType;
typedef typename TConfigure::ContainerType           ContainerType;
typedef typename TConfigure::IteratorType            IteratorType;
typedef typename TConfigure::ResultContainerType     ResultContainerType;
typedef typename TConfigure::ResultIteratorType      ResultIteratorType;
typedef typename TConfigure::DistanceIteratorType	 DistanceIteratorType;

typedef std::vector<PointerType>     LocalContainerType;
typedef typename LocalContainerType::iterator LocalIteratorType;

KRATOS_CLASS_POINTER_DEFINITION(Cell);


Cell()
{
}

virtual ~Cell() {}


void Add(const PointerType& ThisObject)
{
mObjects.push_back(ThisObject);
}

void Remove(const PointerType& ThisObject)
{
mObjects.erase(std::remove(mObjects.begin(),mObjects.end(),ThisObject),mObjects.end());
}

void Remove(const std::size_t Index)
{
std::swap(mObjects[Index], mObjects.back());
mObjects.pop_back();
}

void Clear()
{
mObjects.clear();
}


void AllocateCell(const std::size_t size)
{
mObjects.reserve(size);
}


Cell& operator=(Cell const& rOther)
{
mObjects   = rOther.mObjects;
return *this;
}


Cell(Cell const& rOther) :
mObjects(rOther.mObjects)
{
}

void SearchObjects(PointerType& rThisObject, ResultIteratorType& Result, SizeType& NumberOfResults, const SizeType& MaxNumberOfResults)
{
for(LocalIteratorType i_object = Begin() ; i_object != End()  && NumberOfResults < MaxNumberOfResults ; i_object++)
{
if(TConfigure::Intersection(rThisObject, *i_object))
{
ResultIteratorType repeated_object = std::find(Result-NumberOfResults, Result, *i_object);
if(repeated_object==Result)
{
*Result   = *i_object;
Result++;
NumberOfResults++;
}
}
}
}

void SearchObjects(PointerType& rThisObject, ResultContainerType& Result)
{
for(LocalIteratorType i_object = Begin() ; i_object != End(); i_object++)
{
if(TConfigure::Intersection(rThisObject, *i_object))
{
ResultIteratorType repeated_object = std::find(Result.begin(), Result.end(), *i_object);
if(repeated_object==Result.end())
{
Result.push_back(*i_object);
}
}
}
}

void SearchObjectsExclusive(PointerType& rThisObject, ResultIteratorType& Result, SizeType& NumberOfResults, const SizeType& MaxNumberOfResults)
{
for(LocalIteratorType i_object = Begin() ; i_object != End()  && NumberOfResults < MaxNumberOfResults ; i_object++)
{
if( rThisObject != *i_object )
{
if(TConfigure::Intersection(rThisObject, *i_object))
{
ResultIteratorType repeated_object = std::find(Result-NumberOfResults, Result, *i_object);
if(repeated_object==Result)
{
*Result   = *i_object;
Result++;
NumberOfResults++;
}
}
}
}
}

void SearchObjectsExclusive(PointerType& rThisObject, ResultContainerType& Result)
{
for(LocalIteratorType i_object = Begin() ; i_object != End(); i_object++)
{
if( rThisObject != *i_object )
{
if(TConfigure::Intersection(rThisObject, *i_object))
{
ResultIteratorType repeated_object = std::find(Result.begin(), Result.end(), *i_object);
if(repeated_object==Result.end())
{
Result.push_back(*i_object);
}
}
}
}
}

void SearchObjectsInRadius(PointerType& rThisObject, double const& Radius, ResultIteratorType& Result, SizeType& NumberOfResults, const SizeType& MaxNumberOfResults)
{
for(LocalIteratorType i_object = Begin() ; i_object != End()  && NumberOfResults < MaxNumberOfResults ; i_object++)
{
if(TConfigure::Intersection(rThisObject, *i_object, Radius))
{
ResultIteratorType repeated_object = std::find(Result-NumberOfResults, Result, *i_object);
if(repeated_object==Result)
{
*Result   = *i_object;
Result++;
NumberOfResults++;
}
}
}
}

void SearchObjectsInRadiusExclusive(PointerType& rThisObject, double const& Radius, ResultIteratorType& Result, SizeType& NumberOfResults, const SizeType& MaxNumberOfResults)
{
for(LocalIteratorType i_object = Begin() ; i_object != End()  && NumberOfResults < MaxNumberOfResults ; i_object++)
{
if( rThisObject != *i_object )
{
if(TConfigure::Intersection(rThisObject, *i_object, Radius))
{
ResultIteratorType repeated_object = std::find(Result-NumberOfResults, Result, *i_object);
if(repeated_object==Result)
{
*Result   = *i_object;
Result++;
NumberOfResults++;
}
}
}
}
}

void SearchObjectsInRadius(PointerType& rThisObject, double const& Radius, ResultIteratorType& Result, DistanceIteratorType& Distances, SizeType& NumberOfResults, const SizeType& MaxNumberOfResults)
{
for(LocalIteratorType i_object = Begin() ; i_object != End()  && NumberOfResults < MaxNumberOfResults ; i_object++)
{
if(TConfigure::Intersection(rThisObject, *i_object, Radius))
{
ResultIteratorType repeated_object = std::find(Result-NumberOfResults, Result, *i_object);
if(repeated_object==Result)
{
double distance = 0;
TConfigure::Distance(rThisObject,*i_object,distance); 
*Result   = *i_object;
Result++;
*Distances = distance;
Distances++;
NumberOfResults++;
}
}
}
}

void SearchObjectsInRadiusExclusive(PointerType& rThisObject, double const& Radius, ResultIteratorType& Result, DistanceIteratorType& Distances, SizeType& NumberOfResults, const SizeType& MaxNumberOfResults)
{
for(LocalIteratorType i_object = Begin() ; i_object != End()  && NumberOfResults < MaxNumberOfResults ; i_object++)
{
if( rThisObject != *i_object )
{
if(TConfigure::Intersection(rThisObject, *i_object, Radius))
{
ResultIteratorType repeated_object = std::find(Result-NumberOfResults, Result, *i_object);
if(repeated_object==Result)
{
double distance = 0;
TConfigure::Distance(rThisObject,*i_object,distance); 
*Result   = *i_object;
Result++;
*Distances = distance;
Distances++;
NumberOfResults++;
}
}
}
}
}

LocalIteratorType Begin()
{
return mObjects.begin();
}

LocalIteratorType End()
{
return mObjects.end();
}

SizeType Size()
{
return mObjects.size();
}

LocalIteratorType Begin() const
{
return mObjects.begin();
}

LocalIteratorType End() const
{
return mObjects.end();
}

SizeType Size() const
{
return mObjects.size();
}

PointerType GetObject(std::size_t Index)
{
return mObjects[Index];
}






virtual std::string Info() const
{
return "Cell Class ";
}

virtual void PrintInfo(std::ostream& rOStream) const
{
return;
}

virtual void PrintData(std::ostream& rOStream) const
{
return;
}


protected:







private:


std::vector<PointerType> mObjects;







}; 






template< class  TConfigure>
inline std::istream& operator >> (std::istream& rIStream,
Cell<TConfigure>& rThis)
{
return rIStream;
}

template< class  TConfigure>
inline std::ostream& operator << (std::ostream& rOStream,
const Cell<TConfigure>& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}


}  


