
#include "includes/model_part.h"

#if !defined(KRATOS_ACTIVATION_UTILITIES )
#define  KRATOS_ACTIVATION_UTILITIES









#include "utilities/math_utils.h"
#include "includes/kratos_flags.h"

namespace Kratos
{



























class ActivationUtilities
{
public:


typedef ModelPart::NodesContainerType NodesArrayType;
typedef ModelPart::ConditionsContainerType ConditionsArrayType;


















void ActivateElementsAndConditions( ModelPart& rmodel_part,
const Variable< double >& rVariable,
const double reference_value,
bool active_if_lower_than_reference)
{
KRATOS_TRY

ModelPart::ElementsContainerType::iterator el_begin = rmodel_part.ElementsBegin();
ModelPart::ConditionsContainerType::iterator cond_begin = rmodel_part.ConditionsBegin();

#pragma omp parallel for
for(int i=0; i<static_cast<int>(rmodel_part.Elements().size()); i++)
{
ModelPart::ElementsContainerType::iterator it = el_begin + i;

const Geometry< Node >& geom = it->GetGeometry();
it->Set(ACTIVE,false);

for(unsigned int k=0; k<geom.size(); k++)
{
if( geom[k].FastGetSolutionStepValue(rVariable) < reference_value) 
{
it->Set(ACTIVE,true);
break;
}
}
}

#pragma omp parallel for
for(int i=0; i<static_cast<int>(rmodel_part.Conditions().size()); i++)
{
ModelPart::ConditionsContainerType::iterator it = cond_begin + i;

const Geometry< Node >& geom = it->GetGeometry();
it->Set(ACTIVE,false);
for(unsigned int k=0; k<geom.size(); k++)
{
if( geom[k].FastGetSolutionStepValue(rVariable) < reference_value) 
{
it->Set(ACTIVE,true);
break;
}

}
}

if( active_if_lower_than_reference == false) 
{
#pragma omp parallel for
for(int i=0; i<static_cast<int>(rmodel_part.Elements().size()); i++)
{
ModelPart::ElementsContainerType::iterator it = el_begin + i;
it->Flip(ACTIVE);
}

#pragma omp parallel for
for(int i=0; i<static_cast<int>(rmodel_part.Conditions().size()); i++)
{
ModelPart::ConditionsContainerType::iterator it = cond_begin + i;
it->Flip(ACTIVE);
}
}

KRATOS_CATCH("")

}




















private:





































}; 









}  

#endif 

