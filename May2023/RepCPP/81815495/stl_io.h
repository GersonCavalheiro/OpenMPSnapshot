
#pragma once

#include <string>
#include <iostream>
#include <filesystem>


#include "includes/define.h"
#include "includes/io.h"

namespace Kratos
{







class KRATOS_API(KRATOS_CORE) StlIO 
: public IO
{
public:

using GeometriesMapType = ModelPart::GeometriesMapType;
using NodesArrayType = Element::NodesArrayType;

KRATOS_CLASS_POINTER_DEFINITION(StlIO);


StlIO(
std::filesystem::path const& Filename,
Parameters ThisParameters = Parameters());

StlIO(
Kratos::shared_ptr<std::iostream> pInputStream,
Parameters ThisParameters = Parameters());

virtual ~StlIO(){}



static Parameters GetDefaultParameters();

void ReadModelPart(ModelPart & rThisModelPart) override;

void WriteModelPart(const ModelPart & rThisModelPart) override;




std::string Info() const override;

void PrintInfo(std::ostream& rOStream) const override;

void PrintData(std::ostream& rOStream) const override;


protected:


Parameters mParameters;

std::size_t mNextNodeId = 0;
std::size_t mNextElementId = 0;
std::size_t mNextConditionId = 0;






private:


Kratos::shared_ptr<std::iostream> mpInputStream;
Flags mOptions;



void ReadSolid(
ModelPart & rThisModelPart,
const std::function<void(ModelPart&, NodesArrayType&)>& rCreateEntityFunctor );

void ReadFacet(
ModelPart & rThisModelPart,
const std::function<void(ModelPart&, NodesArrayType&)>& rCreateEntityFunctor);

void ReadLoop(
ModelPart & rThisModelPart,
const std::function<void(ModelPart&, NodesArrayType&)>& rCreateEntityFunctor);

Point ReadPoint();

void ReadKeyword(std::string const& Keyword);

template<class TContainerType>
void WriteEntityBlock(const TContainerType& rThisEntities);

void WriteGeometryBlock(const GeometriesMapType& rThisGeometries);

void WriteFacet(const GeometryType & rGeom);

bool IsValidGeometry(
const Geometry<Node>& rGeometry,
std::size_t& rNumDegenerateGeos) const;




StlIO& operator=(StlIO const& rOther);

StlIO(StlIO const& rOther);


}; 





inline std::istream& operator >> (std::istream& rIStream,
StlIO& rThis);

inline std::ostream& operator << (std::ostream& rOStream,
const StlIO& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}


}  
