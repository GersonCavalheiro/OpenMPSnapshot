
#pragma once

#include <string>
#include <iostream>


#include "processes/apply_ray_casting_process.h"

namespace Kratos
{



template<std::size_t TDim>
class KRATOS_API(KRATOS_CORE) ApplyRayCastingInterfaceRecognitionProcess : public ApplyRayCastingProcess<TDim>
{

public:

KRATOS_CLASS_POINTER_DEFINITION(ApplyRayCastingInterfaceRecognitionProcess);

KRATOS_REGISTRY_ADD_PROTOTYPE("Processes.KratosMultiphysics", ApplyRayCastingInterfaceRecognitionProcess<TDim>)
KRATOS_REGISTRY_ADD_PROTOTYPE("Processes.All", ApplyRayCastingInterfaceRecognitionProcess<TDim>)


using BaseType = ApplyRayCastingProcess<TDim>;



ApplyRayCastingInterfaceRecognitionProcess() = default;


ApplyRayCastingInterfaceRecognitionProcess(
Model& rModel,
Parameters ThisParameters);


ApplyRayCastingInterfaceRecognitionProcess(
FindIntersectedGeometricalObjectsProcess& TheFindIntersectedObjectsProcess,
Parameters ThisParameters = Parameters());


~ApplyRayCastingInterfaceRecognitionProcess() override 
{}


ApplyRayCastingInterfaceRecognitionProcess(ApplyRayCastingInterfaceRecognitionProcess const& rOther) = delete;

ApplyRayCastingInterfaceRecognitionProcess(ApplyRayCastingInterfaceRecognitionProcess&& rOther) = delete;

ApplyRayCastingInterfaceRecognitionProcess& operator=(ApplyRayCastingInterfaceRecognitionProcess const& rOther) = delete;

ApplyRayCastingInterfaceRecognitionProcess& operator=(ApplyRayCastingInterfaceRecognitionProcess&& rOther) = delete;


const Parameters GetDefaultParameters() const override;

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "ApplyRayCastingInterfaceRecognitionProcess";
}

void PrintData(std::ostream& rOStream) const override
{
}

protected:









std::function<void(Node&, const double)> CreateApplyNodalFunction() const override;








}; 




template<std::size_t TDim>
inline std::istream& operator >> (
std::istream& rIStream,
ApplyRayCastingInterfaceRecognitionProcess<TDim>& rThis);

template<std::size_t TDim>
inline std::ostream& operator << (
std::ostream& rOStream,
const ApplyRayCastingInterfaceRecognitionProcess<TDim>& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}

} 
