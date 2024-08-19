
#pragma once



#include "includes/define.h"
#include "custom_searching/interface_object.h"

namespace Kratos
{



class MapperInterfaceInfo
{
public:

KRATOS_CLASS_POINTER_DEFINITION(MapperInterfaceInfo);

typedef std::size_t IndexType;

typedef typename InterfaceObject::CoordinatesArrayType CoordinatesArrayType;

typedef InterfaceObject::NodeType NodeType;
typedef InterfaceObject::GeometryType GeometryType;


enum class InfoType
{
Dummy
};


MapperInterfaceInfo() = default;

explicit MapperInterfaceInfo(const CoordinatesArrayType& rCoordinates,
const IndexType SourceLocalSystemIndex,
const IndexType SourceRank)
: mSourceLocalSystemIndex(SourceLocalSystemIndex),
mCoordinates(rCoordinates),
mSourceRank(SourceRank)
{}

virtual ~MapperInterfaceInfo() = default;



virtual void ProcessSearchResult(const InterfaceObject& rInterfaceObject) = 0;


virtual void ProcessSearchResultForApproximation(const InterfaceObject& rInterfaceObject) {}

virtual MapperInterfaceInfo::Pointer Create(const CoordinatesArrayType& rCoordinates,
const IndexType SourceLocalSystemIndex,
const IndexType SourceRank) const = 0;

virtual MapperInterfaceInfo::Pointer Create() const = 0;


virtual InterfaceObject::ConstructionType GetInterfaceObjectType() const = 0;

IndexType GetLocalSystemIndex() const { return mSourceLocalSystemIndex; }

IndexType GetSourceRank() const { return mSourceRank; }

bool GetLocalSearchWasSuccessful() const { return mLocalSearchWasSuccessful; }

bool GetIsApproximation() const { return mIsApproximation; }

CoordinatesArrayType& Coordinates()
{
return mCoordinates;
}


virtual void GetValue(int& rValue, const InfoType ValueType) const { KRATOS_ERROR << "Base class function called!" << std::endl; }
virtual void GetValue(std::size_t& rValue, const InfoType ValueType) const { KRATOS_ERROR << "Base class function called!" << std::endl; }
virtual void GetValue(double& rValue, const InfoType ValueType) const { KRATOS_ERROR << "Base class function called!" << std::endl; }
virtual void GetValue(bool& rValue, const InfoType ValueType) const { KRATOS_ERROR << "Base class function called!" << std::endl; }
virtual void GetValue(GeometryType& rValue, const InfoType ValueType) const { KRATOS_ERROR << "Base class function called!" << std::endl; }

virtual void GetValue(std::vector<int>& rValue, const InfoType ValueType) const { KRATOS_ERROR << "Base class function called!" << std::endl; }
virtual void GetValue(std::vector<std::size_t>& rValue, const InfoType ValueType) const { KRATOS_ERROR << "Base class function called!" << std::endl; }
virtual void GetValue(std::vector<double>& rValue, const InfoType ValueType) const { KRATOS_ERROR << "Base class function called!" << std::endl; }
virtual void GetValue(std::vector<bool>& rValue, const InfoType ValueType) const { KRATOS_ERROR << "Base class function called!" << std::endl; }
virtual void GetValue(std::vector<GeometryType>& rValue, const InfoType ValueType) const { KRATOS_ERROR << "Base class function called!" << std::endl; }


virtual std::string Info() const
{
return "MapperInterfaceInfo";
}

virtual void PrintInfo(std::ostream& rOStream) const {}

virtual void PrintData(std::ostream& rOStream) const {}


protected:

IndexType mSourceLocalSystemIndex;

CoordinatesArrayType mCoordinates;
IndexType mSourceRank = 0;


void SetLocalSearchWasSuccessful()
{
mLocalSearchWasSuccessful = true;
mIsApproximation = false;
}

void SetIsApproximation()
{
mLocalSearchWasSuccessful = true;

mIsApproximation = true;
}


private:

bool mIsApproximation = false;

bool mLocalSearchWasSuccessful = false; 


friend class Serializer;

virtual void save(Serializer& rSerializer) const
{
rSerializer.save("LocalSysIdx", mSourceLocalSystemIndex);
rSerializer.save("IsApproximation", mIsApproximation);
}

virtual void load(Serializer& rSerializer)
{
rSerializer.load("LocalSysIdx", mSourceLocalSystemIndex);
rSerializer.load("IsApproximation", mIsApproximation);
}


}; 





}  
