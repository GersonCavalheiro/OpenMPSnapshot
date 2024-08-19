

#if !defined(KRATOS_ENERGY_UTILITIES_H_INCLUDED)
#define  KRATOS_ENERGY_UTILITIES_H_INCLUDED




#include "utilities/timer.h"

#include "includes/variables.h"
#include "includes/model_part.h"
#include "utilities/openmp_utils.h"
#include "utilities/math_utils.h"
#include "solid_mechanics_application_variables.h"

namespace Kratos
{




class EnergyUtilities
{
public:


typedef ModelPart::ElementsContainerType                  ElementsContainerType;
typedef ModelPart::MeshType::GeometryType                          GeometryType;


EnergyUtilities(){ mEchoLevel = 0;  mParallel = true; };

EnergyUtilities(bool Parallel){ mEchoLevel = 0;  mParallel = Parallel; };


virtual ~EnergyUtilities(){};






/
virtual void SetEchoLevel(int Level)
{
mEchoLevel = Level;
}

int GetEchoLevel()
{
return mEchoLevel;
}










protected:














private:


int mEchoLevel;

bool mParallel;












}; 





} 

#endif 


