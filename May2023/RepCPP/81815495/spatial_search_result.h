
#pragma once

#include <string>
#include <iostream>


#include "includes/define.h"


namespace Kratos
{



template <typename TObjectType>
class SpatialSearchResult
{
public:

KRATOS_CLASS_POINTER_DEFINITION(SpatialSearchResult);

using TPointerType = GlobalPointer<TObjectType>;


SpatialSearchResult()
: mpObject(nullptr),
mDistance(0.0),
mIsObjectFound(false),
mIsDistanceCalculated(false)
{
}

SpatialSearchResult(
TObjectType* pObject,
const int Rank = 0
) : mpObject(pObject, Rank),
mDistance(0.0),
mIsObjectFound(false),
mIsDistanceCalculated(false)
{
if (mpObject.get() != nullptr)
mIsObjectFound = true;
}

SpatialSearchResult(SpatialSearchResult const& ) = default;

SpatialSearchResult(SpatialSearchResult&& ) = default;

virtual ~SpatialSearchResult(){}


SpatialSearchResult& operator=(SpatialSearchResult const& ) = default;



void Reset()
{
mpObject = nullptr;
mDistance = 0.0;
mIsObjectFound = false;
mIsDistanceCalculated = false;
}


TPointerType Get() {
return mpObject;
}

TPointerType const Get() const {
return mpObject;
}

void Set(TObjectType* pObject) {
mpObject = pObject;
mIsObjectFound = true;
}

double GetDistance() const {
return mDistance;
}

void SetDistance(const double TheDistance) {
mDistance = TheDistance;
mIsDistanceCalculated = true;
}


bool IsObjectFound() const { 
return mIsObjectFound; 
}

bool IsDistanceCalculated() const { 
return mIsDistanceCalculated; 
}


virtual std::string Info() const
{
std::stringstream buffer;
buffer << "SpatialSearchResult" ;
return buffer.str();
}

virtual void PrintInfo(std::ostream& rOStream) const {rOStream << "SpatialSearchResult";}

virtual void PrintData(std::ostream& rOStream) const {}


private:

TPointerType mpObject;      
double mDistance;           
bool mIsObjectFound;        
bool mIsDistanceCalculated; 



friend class Serializer;

void save(Serializer& rSerializer) const
{
rSerializer.save("Object", mpObject);
rSerializer.save("Distance", mDistance);
rSerializer.save("Is Object Found", mIsObjectFound);
rSerializer.save("Is Distance Calculated", mIsDistanceCalculated);
}

void load(Serializer& rSerializer)
{
rSerializer.load("Object", mpObject);
rSerializer.load("Distance", mDistance);
rSerializer.load("Is Object Found", mIsObjectFound);
rSerializer.load("Is Distance Calculated", mIsDistanceCalculated);
}


}; 



template <typename TObjectType>
inline std::istream& operator >> (std::istream& rIStream,
SpatialSearchResult<TObjectType>& rThis){
return rIStream;
}

template <typename TObjectType>
inline std::ostream& operator << (std::ostream& rOStream,
const SpatialSearchResult<TObjectType>& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}


}  


