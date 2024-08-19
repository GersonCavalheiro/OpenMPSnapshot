
#pragma once



#include "includes/define.h"
#include "includes/kratos_application.h"
#include "custom_searching/interface_object.h"
#include "custom_modelers/mapping_geometries_modeler.h"

namespace Kratos
{







class KRATOS_API(MAPPING_APPLICATION) KratosMappingApplication : public KratosApplication
{
public:


KRATOS_CLASS_POINTER_DEFINITION(KratosMappingApplication);


KratosMappingApplication();

~KratosMappingApplication() override {}





void Register() override;








std::string Info() const override
{
return "KratosMappingApplication";
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << Info();
PrintData(rOStream);
}

void PrintData(std::ostream& rOStream) const override
{
KRATOS_WATCH("in my application");
KRATOS_WATCH(KratosComponents<VariableData>::GetComponents().size() );

rOStream << "Variables:" << std::endl;
KratosComponents<VariableData>().PrintData(rOStream);
rOStream << std::endl;
rOStream << "Elements:" << std::endl;
KratosComponents<Element>().PrintData(rOStream);
rOStream << std::endl;
rOStream << "Conditions:" << std::endl;
KratosComponents<Condition>().PrintData(rOStream);
}




protected:















private:



const InterfaceObject           mInterfaceObject;
const InterfaceNode             mInterfaceNode;
const InterfaceGeometryObject   mInterfaceGeometryObject;

const MappingGeometriesModeler  mMappingGeometriesModeler;










KratosMappingApplication& operator=(KratosMappingApplication const& rOther);

KratosMappingApplication(KratosMappingApplication const& rOther);



}; 




}  