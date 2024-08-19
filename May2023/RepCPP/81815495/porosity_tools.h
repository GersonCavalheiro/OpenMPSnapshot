#ifndef KRATOS_POROSITY_TOOLS_H
#define KRATOS_POROSITY_TOOLS_H


#include "includes/variables.h"


#include <limits>
#include <iostream>
#include <iomanip>


#ifdef _OPENMP
#include <omp.h>
#endif


#include "includes/define.h"
#include "includes/model_part.h"
#include "utilities/openmp_utils.h"
#include "real_field.h"

namespace Kratos
{
class PorosityUtils
{
public:
typedef ModelPart::NodesContainerType::ContainerType::iterator NodesIteratorType;

KRATOS_CLASS_POINTER_DEFINITION(PorosityUtils);


PorosityUtils(RealField& porosity_field):mPorosityField(porosity_field){}


virtual ~PorosityUtils(){}


void CalculatePorosity(ModelPart& r_model_part, const ProcessInfo& r_current_process_info)
{
double time = r_current_process_info[TIME];
const int nnodes = r_model_part.Nodes().size();

#pragma omp parallel for
for (int i = 0; i < nnodes; ++i){
ModelPart::NodeIterator node_it = r_model_part.NodesBegin() + i;
array_1d<double, 3> coor;
coor[0] = node_it->X();
coor[1] = node_it->Y();
coor[2] = node_it->Z();
node_it->FastGetSolutionStepValue(FLUID_FRACTION) = mPorosityField.Evaluate(time, coor);
}

}






virtual std::string Info() const
{
return "";
}


virtual void PrintInfo(std::ostream& rOStream) const
{
}


virtual void PrintData(std::ostream& rOStream) const
{
}




protected:














private:



RealField mPorosityField;








PorosityUtils & operator=(PorosityUtils const& rOther);



}; 

}  

#endif 
