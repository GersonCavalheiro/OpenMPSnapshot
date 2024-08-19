
#pragma once

#include "includes/model_part_io.h"
#include "custom_processes/metis_divide_heterogeneous_input_process.h"


namespace Kratos
{






class KRATOS_API(METIS_APPLICATION) MetisDivideHeterogeneousInputInMemoryProcess : public MetisDivideHeterogeneousInputProcess
{
public:

KRATOS_CLASS_POINTER_DEFINITION(MetisDivideHeterogeneousInputInMemoryProcess);

typedef MetisDivideHeterogeneousInputProcess BaseType;

using BaseType::SizeType;
using BaseType::GraphType;
using BaseType::idxtype;


MetisDivideHeterogeneousInputInMemoryProcess(IO& rIO, ModelPartIO& rSerialIO, const DataCommunicator& rDataComm, int Dimension = 3, int Verbosity = 0, bool SynchronizeConditions = false):
BaseType(rIO,rDataComm.Size(),Dimension,Verbosity,SynchronizeConditions), mrSerialIO(rSerialIO), mrDataComm(rDataComm)
{
KRATOS_ERROR_IF_NOT(mrDataComm.IsDistributed()) << "DataCommunicator must be distributed!" << std::endl;
}

virtual ~MetisDivideHeterogeneousInputInMemoryProcess()
{
}



void operator()()
{
this->Execute();
}




void Execute() override;






std::string Info() const override
{
return "MetisDivideHeterogeneousInputInMemoryProcess";
}

void PrintInfo(std::ostream& rOStream) const override
{
rOStream << "MetisDivideHeterogeneousInputInMemoryProcess";
}

void PrintData(std::ostream& rOStream) const override
{
}





protected:















private:



ModelPartIO& mrSerialIO;

const DataCommunicator& mrDataComm;













}; 







}
