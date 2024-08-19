
#pragma once






#include "includes/define.h"
#include "includes/serializer.h"
#include "includes/properties.h"
#include "geometries/geometry.h"
#include "includes/process_info.h"

namespace Kratos
{


class KRATOS_API(GEO_MECHANICS_APPLICATION) RetentionLaw
{
public:

using ProcessInfoType = ProcessInfo;
using SizeType = std::size_t;
using GeometryType = Geometry<Node>;


KRATOS_CLASS_POINTER_DEFINITION(RetentionLaw);



class Parameters
{
KRATOS_CLASS_POINTER_DEFINITION(Parameters);



private:

double mFluidPressure = 0.0;
double mMeanStress = 0.0;
double mTemperature = 0.0;
double mVolumetricStrain = 0.0;

const ProcessInfo &mrCurrentProcessInfo;
const Properties &mrMaterialProperties;
const GeometryType &mrElementGeometry;

public:
Parameters(const GeometryType &rElementGeometry,
const Properties &rMaterialProperties,
const ProcessInfo &rCurrentProcessInfo) 
: mrCurrentProcessInfo(rCurrentProcessInfo)
,mrMaterialProperties(rMaterialProperties)
,mrElementGeometry(rElementGeometry)
{};

~Parameters() = default;

void SetVolumetricStrain(const double rVolumetricStrain) { mVolumetricStrain = rVolumetricStrain; };
void SetMeanStress      (const double rMeanStress)       { mMeanStress = rMeanStress; };
void SetFluidPressure   (const double rFluidPressure)    { mFluidPressure = rFluidPressure; };
void SetTemperature     (const double rTemperature)      { mTemperature = rTemperature; };

double GetVolumetricStrain() const { return mVolumetricStrain; }
double GetMeanStress()       const { return mMeanStress;       }
double GetFluidPressure()    const { return mFluidPressure;    }
double GetTemperature()      const { return mTemperature;      }

const ProcessInfo &GetProcessInfo() const
{
return mrCurrentProcessInfo;
}
const Properties &GetMaterialProperties() const
{
return mrMaterialProperties;
}
const GeometryType &GetElementGeometry() const
{
return mrElementGeometry;
}

}; 

RetentionLaw() = default;

virtual ~RetentionLaw() = default;


virtual RetentionLaw::Pointer Clone() const = 0;


virtual double &CalculateValue(Parameters &rParameters,
const Variable<double> &rThisVariable,
double &rValue) = 0;

virtual double CalculateSaturation(Parameters &rParameters) = 0;

virtual double CalculateEffectiveSaturation(Parameters &rParameters) = 0;

virtual double CalculateDerivativeOfSaturation(Parameters &rParameters) = 0;

virtual double CalculateRelativePermeability(Parameters &rParameters) = 0;

virtual double CalculateBishopCoefficient(Parameters &rParameters) = 0;


virtual void InitializeMaterial(const Properties& rMaterialProperties,
const GeometryType& rElementGeometry,
const Vector& rShapeFunctionsValues);

virtual void Initialize(Parameters &rParameters);


virtual void InitializeSolutionStep(Parameters &rParameters);


virtual void FinalizeSolutionStep(Parameters &rParameters);


virtual void Finalize(Parameters &rParameters);


virtual void ResetMaterial(const Properties &rMaterialProperties,
const GeometryType &rElementGeometry,
const Vector &rShapeFunctionsValues);


virtual int Check(const Properties &rMaterialProperties,
const ProcessInfo &rCurrentProcessInfo) = 0;


inline static bool HasSameType(const RetentionLaw &rLHS, const RetentionLaw &rRHS)
{
return (typeid(rLHS) == typeid(rRHS));
}


inline static bool HasSameType(const RetentionLaw *rLHS, const RetentionLaw *rRHS)
{
return RetentionLaw::HasSameType(*rLHS, *rRHS);
}

virtual std::string Info() const
{
std::stringstream buffer;
buffer << "RetentionLaw";
return buffer.str();
}

virtual void PrintInfo(std::ostream &rOStream) const
{
rOStream << "RetentionLaw";
}

virtual void PrintData(std::ostream &rOStream) const
{
rOStream << "RetentionLaw has no data";
}

private:
friend class Serializer;

virtual void save(Serializer &rSerializer) const;

virtual void load(Serializer &rSerializer);

}; 

inline std::istream &operator>>(std::istream &rIStream,
RetentionLaw &rThis);

inline std::ostream &operator<<(std::ostream &rOStream,
const RetentionLaw &rThis)
{
rThis.PrintInfo(rOStream);
rOStream << " : " << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}

} 
