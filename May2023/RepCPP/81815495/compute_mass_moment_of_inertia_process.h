
#pragma once



#include "processes/process.h"
#include "includes/model_part.h"

namespace Kratos
{





class KRATOS_API(STRUCTURAL_MECHANICS_APPLICATION) ComputeMassMomentOfInertiaProcess
: public Process
{
public:

KRATOS_CLASS_POINTER_DEFINITION(ComputeMassMomentOfInertiaProcess);


ComputeMassMomentOfInertiaProcess(
ModelPart& rThisModelPart,
const Point& rPoint1,
const Point& rPoint2
):mrThisModelPart(rThisModelPart) , mrPoint1(rPoint1), mrPoint2(rPoint2)
{ }

~ComputeMassMomentOfInertiaProcess() override = default;







void Execute() override;






std::string Info() const override
{
return "ComputeMassMomentOfInertiaProcess";
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "ComputeMassMomentOfInertiaProcess";
}

void PrintData(std::ostream& rOStream) const override
{
}




protected:














private:



ModelPart& mrThisModelPart; 
const Point& mrPoint1;      
const Point& mrPoint2;      









ComputeMassMomentOfInertiaProcess& operator=(ComputeMassMomentOfInertiaProcess const& rOther) = delete;

ComputeMassMomentOfInertiaProcess(ComputeMassMomentOfInertiaProcess const& rOther) = delete;



}; 






}
