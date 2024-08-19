
#pragma once

#include <iostream>
#include <vector>
#include <pybind11/pybind11.h>


#include "includes/serializer.h"

namespace Kratos
{






class ProcessFactoryUtility
{
public:


KRATOS_CLASS_POINTER_DEFINITION( ProcessFactoryUtility );

typedef pybind11::object ObjectType;

typedef pybind11::list     ListType;


ProcessFactoryUtility()= default;


ProcessFactoryUtility(ListType& ProcessesList);


ProcessFactoryUtility(ObjectType& rProcess);

virtual ~ProcessFactoryUtility()= default;



ProcessFactoryUtility& operator=(ProcessFactoryUtility const& rOther)
= default;




void AddProcess(ObjectType& rProcess);



void AddProcesses(ListType& ProcessesList);



void ExecuteMethod(const std::string& rNameMethod);



void ExecuteInitialize();



void ExecuteBeforeSolutionLoop();



void ExecuteInitializeSolutionStep();



void ExecuteFinalizeSolutionStep();



void ExecuteBeforeOutputStep();



void ExecuteAfterOutputStep();



void ExecuteFinalize();



void IsOutputStep();



void PrintOutput();



void Clear();






virtual std::string Info() const
{
std::stringstream buffer;
buffer << "ProcessFactoryUtility. Number of processes:" << mProcesses.size();
return buffer.str();
}


virtual void PrintInfo(std::ostream& rOStream) const
{
rOStream << "ProcessFactoryUtility. Number of processes:" << mProcesses.size();
}


virtual void PrintData(std::ostream& rOStream) const
{
rOStream << "ProcessFactoryUtility. Number of processes:" << mProcesses.size();
}

private:

std::vector<ObjectType> mProcesses; 





friend class Serializer;

virtual void save(Serializer& rSerializer) const
{
}

virtual void load(Serializer& rSerializer)
{
}


}; 





inline std::istream & operator >>(std::istream& rIStream,
ProcessFactoryUtility& rThis);


inline std::ostream & operator <<(std::ostream& rOStream,
const ProcessFactoryUtility& rThis)
{
rThis.PrintInfo(rOStream);
rOStream << " : " << std::endl;
rThis.PrintData(rOStream);

return rOStream;
}


}  
