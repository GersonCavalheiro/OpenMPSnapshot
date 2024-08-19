
#pragma once

#include <string>
#include <iostream>
#include <unordered_map>


#include "includes/define.h"
#include "containers/model.h"
#include "processes/process.h"
#include "includes/kratos_parameters.h"
#include "includes/model_part.h"
#include "includes/global_pointer_variables.h"

namespace Kratos
{





template<class TContainerType>
class KRATOS_API(KRATOS_CORE) FindGlobalNodalEntityNeighboursProcess : public Process
{
public:

using IndexType = std::size_t;

using NodeType = ModelPart::NodeType;

using EntityType = typename TContainerType::value_type;

using GlobalEntityPointersVectorType = GlobalPointersVector<EntityType>;

using NeighbourMapType = std::unordered_map<int, GlobalPointersVector<EntityType>>;

using NonLocalMapType =  std::unordered_map<int, NeighbourMapType>;

using IdMapType = std::unordered_map<int, std::vector<int>>;

KRATOS_CLASS_POINTER_DEFINITION(FindGlobalNodalEntityNeighboursProcess);


FindGlobalNodalEntityNeighboursProcess(
Model& rModel,
Parameters Params);

FindGlobalNodalEntityNeighboursProcess(
ModelPart& rModelPart);

FindGlobalNodalEntityNeighboursProcess(
ModelPart& rModelPart,
const Variable<GlobalEntityPointersVectorType>& rOutputVariable);

~FindGlobalNodalEntityNeighboursProcess() override = default;


FindGlobalNodalEntityNeighboursProcess<EntityType>& operator=(FindGlobalNodalEntityNeighboursProcess<EntityType> const& rOther) = delete;


void Execute() override;

void Clear() override;

KRATOS_DEPRECATED_MESSAGE("This is legacy version (use Clear)") void ClearNeighbours() { Clear(); }

IdMapType GetNeighbourIds(const ModelPart::NodesContainerType& rNodes);

const Parameters GetDefaultParameters() const override;


std::string Info() const override
{
return "FindGlobalNodalEntityNeighboursProcess";
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "FindGlobalNodalEntityNeighboursProcess";
}

void PrintData(std::ostream& rOStream) const override
{
}


private:

Model& mrModel;

std::string mModelPartName;

const Variable<GlobalEntityPointersVectorType>& mrOutputVariable;


static TContainerType& GetContainer(ModelPart& rModelPart);

static const Variable<GlobalEntityPointersVectorType>& GetDefaultOutputVariable();


}; 



template<class TContainerType>
inline std::istream& operator >> (std::istream& rIStream,
FindGlobalNodalEntityNeighboursProcess<TContainerType>& rThis);

template<class TContainerType>
inline std::ostream& operator << (std::ostream& rOStream,
const FindGlobalNodalEntityNeighboursProcess<TContainerType>& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}

}  
