
#pragma once




#include "metis_divide_heterogeneous_input_process.h"


namespace Kratos {






class KRATOS_API(METIS_APPLICATION) MetisDivideSubModelPartsHeterogeneousInputProcess : public MetisDivideHeterogeneousInputProcess {
public:


KRATOS_CLASS_POINTER_DEFINITION(MetisDivideSubModelPartsHeterogeneousInputProcess);

using BaseType = MetisDivideHeterogeneousInputProcess;


MetisDivideSubModelPartsHeterogeneousInputProcess(IO& rIO, Parameters Settings, SizeType NumberOfPartitions, int Dimension = 3, int Verbosity = 0, bool SynchronizeConditions = false)
: MetisDivideHeterogeneousInputProcess(rIO, NumberOfPartitions, Dimension, Verbosity, SynchronizeConditions),
mSettings(Settings) {}

~MetisDivideSubModelPartsHeterogeneousInputProcess() override {
}






std::string Info() const override {
std::stringstream buffer;
buffer << "MetisDivideSubModelPartsHeterogeneousInputProcess" ;
return buffer.str();

}

void PrintInfo(std::ostream& rOStream) const override {
rOStream << "MetisDivideSubModelPartsHeterogeneousInputProcess";
}

void PrintData(std::ostream& rOStream) const override {}



protected:

Parameters mSettings;




void GetNodesPartitions(std::vector<idxtype> &rNodePartition, SizeType &rNumNodes) override;





private:








MetisDivideSubModelPartsHeterogeneousInputProcess& operator=(MetisDivideSubModelPartsHeterogeneousInputProcess const& rOther);

}; 






}  
