
#pragma once



#include "processes/process.h"
#include "containers/model.h"

namespace Kratos
{






class PointGeometry
: public Point
{
public:


typedef Point BaseType;

typedef Node NodeType;

KRATOS_CLASS_POINTER_DEFINITION( PointGeometry );


PointGeometry():
BaseType()
{}

PointGeometry(const double X, const double Y, const double Z)
: BaseType(X, Y, Z)
{}

PointGeometry(Geometry<NodeType>::Pointer pGeometry):
mpGeometry(pGeometry)
{
UpdatePoint();
}

PointGeometry(const PointGeometry& rRHS):
BaseType(rRHS),
mpGeometry(rRHS.mpGeometry)
{
}

~PointGeometry() override= default;




Geometry<NodeType>::Pointer pGetGeometry()
{
return mpGeometry;
}


void UpdatePoint()
{
noalias(this->Coordinates()) = mpGeometry->Center().Coordinates();
}

private:

Geometry<NodeType>::Pointer mpGeometry = nullptr; 


}; 


struct CalculateDistanceToPathSettings
{
constexpr static bool SaveAsHistoricalVariable = true;
constexpr static bool SaveAsNonHistoricalVariable = false;
};


template<bool THistorical = true>
class KRATOS_API(KRATOS_CORE) CalculateDistanceToPathProcess
: public Process
{
public:

typedef std::size_t IndexType;

typedef std::size_t SizeType;

typedef Node NodeType;

KRATOS_CLASS_POINTER_DEFINITION(CalculateDistanceToPathProcess);



CalculateDistanceToPathProcess(
Model& rModel,
Parameters ThisParameters
);

~CalculateDistanceToPathProcess() override
{
}




Process::Pointer Create(
Model& rModel,
Parameters ThisParameters
) override
{
return Kratos::make_shared<CalculateDistanceToPathProcess>(rModel, ThisParameters);
}


void Execute() override;


const Parameters GetDefaultParameters() const override;




std::string Info() const override
{
return "CalculateDistanceToPathProcess";
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "CalculateDistanceToPathProcess";
}

void PrintData(std::ostream& rOStream) const override
{
}


protected:







private:


Model& mrModel;                             
Parameters mThisParameters;                 
const Variable<double>* mpDistanceVariable; 




void CalculateDistance(
ModelPart& rModelPart,
std::vector<Geometry<NodeType>::Pointer>& rVectorSegments
);


void CalculateDistanceByBruteForce(
ModelPart& rModelPart,
std::vector<Geometry<NodeType>::Pointer>& rVectorSegments
);




CalculateDistanceToPathProcess& operator=(CalculateDistanceToPathProcess const& rOther);



}; 



template<bool THistorical>
inline std::istream& operator >> (std::istream& rIStream,
CalculateDistanceToPathProcess<THistorical>& rThis);

template<bool THistorical>
inline std::ostream& operator << (std::ostream& rOStream,
const CalculateDistanceToPathProcess<THistorical>& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}

} 