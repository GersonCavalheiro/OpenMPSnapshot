
#pragma once




namespace Kratos
{




class ModelPart;


namespace CppTestsUtilities
{

void KRATOS_API(KRATOS_CORE) Create2DGeometry(
ModelPart& rModelPart,
const std::string& rEntityName = "Element2D3N",
const bool Initialize = true,
const bool Elements = true
);


void KRATOS_API(KRATOS_CORE) CreateTestModelPartTriangle2D3N(ModelPart& rModelPart);


void KRATOS_API(KRATOS_CORE) Create2DQuadrilateralsGeometry(
ModelPart& rModelPart, 
const std::string& rEntityName = "Element2D4N",
const bool Initialize = true,
const bool Elements = true
);


void KRATOS_API(KRATOS_CORE) Create3DGeometry(
ModelPart& rModelPart,
const std::string& rElementName = "Element3D4N",
const bool Initialize = true
);


void KRATOS_API(KRATOS_CORE) CreateTestModelPartTetrahedra3D4N(ModelPart& rModelPart);


void KRATOS_API(KRATOS_CORE) Create3DHexahedraGeometry(
ModelPart& rModelPart,
const std::string& rElementName = "Element3D8N",
const bool Initialize = true
);


void KRATOS_API(KRATOS_CORE) Create3DQuadraticGeometry(
ModelPart& rModelPart, 
const std::string& rElementName = "Element3D10N",
const bool Initialize = true
);

}; 
}  
