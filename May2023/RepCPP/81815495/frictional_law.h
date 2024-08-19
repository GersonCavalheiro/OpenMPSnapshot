
#pragma once



#include "includes/node.h"
#include "custom_conditions/paired_condition.h"
#include "includes/process_info.h"
#include "includes/mortar_classes.h"

namespace Kratos
{






class KRATOS_API(CONTACT_STRUCTURAL_MECHANICS_APPLICATION) FrictionalLaw
{
public:


typedef Node NodeType;

typedef std::size_t IndexType;

typedef std::size_t SizeType;

static constexpr double ZeroTolerance = std::numeric_limits<double>::epsilon();

KRATOS_CLASS_POINTER_DEFINITION( FrictionalLaw );



FrictionalLaw()
{
}

FrictionalLaw(const FrictionalLaw& rhs)
{
}

virtual ~FrictionalLaw()
{
}




virtual double GetFrictionCoefficient(
const NodeType& rNode,
const PairedCondition& rCondition,
const ProcessInfo& rCurrentProcessInfo
);


virtual double GetThresholdValue(
const NodeType& rNode,
const PairedCondition& rCondition,
const ProcessInfo& rCurrentProcessInfo
);




virtual std::string Info() const
{
return "FrictionalLaw";
}

virtual void PrintInfo(std::ostream& rOStream) const
{
rOStream << Info() << std::endl;
}

virtual void PrintData(std::ostream& rOStream) const
{
rOStream << Info() << std::endl;
}


protected:








private:





friend class Serializer;

virtual void save(Serializer& rSerializer) const
{
}

virtual void load(Serializer& rSerializer)
{
}


}; 






}  