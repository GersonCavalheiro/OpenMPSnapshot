
#pragma once

#include <unordered_map>


#include "includes/model_part.h"
#include "includes/kratos_parameters.h"
#include "includes/constitutive_law.h"
#include "geometries/geometry_data.h"
#include "utilities/geometry_utilities.h"

namespace Kratos
{





namespace
{
template< class TContainerType>
std::vector<std::string> GetDofsListFromGenericEntitiesSpecifications(const TContainerType& rContainer);

static std::unordered_map<std::string, GeometryData::KratosGeometryType> GenerateStringGeometryMap()
{
std::unordered_map<std::string, GeometryData::KratosGeometryType> my_map;
for (unsigned int i = 0; i < static_cast<unsigned int>(GeometryData::KratosGeometryType::NumberOfGeometryTypes); ++i) {
const auto type = static_cast<GeometryData::KratosGeometryType>(i);
my_map.insert({GeometryUtils::GetGeometryName(type), type});
}
return my_map;
}

static std::unordered_map<std::string, GeometryData::KratosGeometryType> string_geometry_map = GenerateStringGeometryMap();

static std::unordered_map<std::string, std::size_t> string_dimension_map = {
{"2D",2},
{"3D",3}
};
}


class KRATOS_API(KRATOS_CORE) SpecificationsUtilities
{
public:

KRATOS_CLASS_POINTER_DEFINITION( SpecificationsUtilities );


enum class TimeIntegration
{
Static   = 0,
Implicit = 1,
Explicit = 2
};


enum class Framework
{
Lagrangian = 0,
Eulerian   = 1,
ALE        = 2
};



SpecificationsUtilities() = delete;




static void AddMissingVariables(ModelPart& rModelPart);


static void AddMissingVariablesFromEntitiesList(
ModelPart& rModelPart,
const Parameters EntitiesList
);


static void AddMissingVariablesFromSpecifications(
ModelPart& rModelPart,
const Parameters SpecificationsParameters,
const std::string EntityName = "NOT_DEFINED"
);


static void AddMissingDofs(ModelPart& rModelPart);


static void AddMissingDofsFromEntitiesList(
ModelPart& rModelPart,
const Parameters EntitiesList
);


static void AddMissingDofsFromSpecifications(
ModelPart& rModelPart,
const Parameters SpecificationsParameters,
const std::string EntityName = "NOT_DEFINED"
);


static std::vector<std::string> GetDofsListFromSpecifications(const ModelPart& rModelPart);


static std::vector<std::string> GetDofsListFromElementsSpecifications(const ModelPart& rModelPart);


static std::vector<std::string> GetDofsListFromConditionsSpecifications(const ModelPart& rModelPart);


static void DetermineFlagsUsed(const ModelPart& rModelPart);


static std::vector<std::string> DetermineTimeIntegration(const ModelPart& rModelPart);


static std::string DetermineFramework(const ModelPart& rModelPart);


static bool DetermineSymmetricLHS(const ModelPart& rModelPart);


static bool DeterminePositiveDefiniteLHS(const ModelPart& rModelPart);


static bool DetermineIfCompatibleGeometries(const ModelPart& rModelPart);


static bool DetermineIfRequiresTimeIntegration(const ModelPart& rModelPart);


static bool CheckCompatibleConstitutiveLaws(const ModelPart& rModelPart);


static int CheckGeometricalPolynomialDegree(const ModelPart& rModelPart);


static Parameters GetDocumention(const ModelPart& rModelPart);

}; 


}  
