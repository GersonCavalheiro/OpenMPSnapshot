
#pragma once




#include "includes/define.h"
#include "includes/element.h"

namespace Kratos
{

enum class TracedStressType
{
FX,
FY,
FZ,
MX,
MY,
MZ,
FXX,
FXY,
FXZ,
FYX,
FYY,
FYZ,
FZX,
FZY,
FZZ,
MXX,
MXY,
MXZ,
MYX,
MYY,
MYZ,
MZX,
MZY,
MZZ,
PK2,
VON_MISES_STRESS
};

enum class StressTreatment
{
Mean,
Node,
GaussPoint
};

namespace StressResponseDefinitions
{

TracedStressType ConvertStringToTracedStressType(const std::string& Str);

StressTreatment ConvertStringToStressTreatment(const std::string& Str);

} 



class StressCalculation
{
public:

typedef std::size_t IndexType;
typedef std::size_t SizeType;

static void CalculateStressOnNode(Element& rElement,
const TracedStressType rTracedStressType,
Vector& rOutput,
const ProcessInfo& rCurrentProcessInfo);

static void CalculateStressOnGP(Element& rElement,
const TracedStressType rTracedStressType,
Vector& rOutput,
const ProcessInfo& rCurrentProcessInfo);

private:

static void CalculateStressOnGPLinearTruss(Element& rElement,
const TracedStressType rTracedStressType,
Vector& rOutput,
const ProcessInfo& rCurrentProcessInfo);

static void CalculateStressOnGPTruss(Element& rElement,
const TracedStressType rTracedStressType,
Vector& rOutput,
const ProcessInfo& rCurrentProcessInfo);

static void CalculateStressOnGPShell(Element& rElement,
const TracedStressType rTracedStressType,
Vector& rOutput,
const ProcessInfo& rCurrentProcessInfo);

static void CalculateStressBeam(Element& rElement,
const TracedStressType rTracedStressType,
std::vector< array_1d<double, 3 > >& rStressVector,
const ProcessInfo& rCurrentProcessInfo,
int& rDirection);

static void CalculateStressOnGPBeam(Element& rElement,
const TracedStressType rTracedStressType,
Vector& rOutput,
const ProcessInfo& rCurrentProcessInfo);

static void CalculateStressOnNodeBeam(Element& rElement,
const TracedStressType rTracedStressType,
Vector& rOutput,
const ProcessInfo& rCurrentProcessInfo);

static void CalculateStressOnGPSmallDisplacement(Element& rElement,
const TracedStressType rTracedStressType,
Vector& rOutput,
const ProcessInfo& rCurrentProcessInfo);

};  

}  


