
#pragma once

#include <pybind11/pybind11.h>


#include "includes/define_python.h"

namespace Kratos::Python
{

namespace py = pybind11;






template<class TContainerType, class TVariableType>
class VariableIndexingPython 
{
public:

KRATOS_CLASS_POINTER_DEFINITION(VariableIndexingPython);


VariableIndexingPython() {}

VariableIndexingPython(const VariableIndexingPython& rOther);

virtual ~VariableIndexingPython() {}



template <class TClassType>
void visit(TClassType& ThisClass) const
{
ThisClass
.def("__contains__", &VariableIndexingPython<TContainerType, TVariableType>::DataValueContainerHas)
.def("__setitem__", &VariableIndexingPython<TContainerType, TVariableType>::DataValueContainerSetValue)
.def("__getitem__", &VariableIndexingPython<TContainerType, TVariableType>::DataValueContainerGetValue)
.def("Has", &VariableIndexingPython<TContainerType, TVariableType>::DataValueContainerHas)
.def("SetValue", &VariableIndexingPython<TContainerType, TVariableType>::DataValueContainerSetValue)
.def("GetValue", &VariableIndexingPython<TContainerType, TVariableType>::DataValueContainerGetValue)
;
}





private:




static void DataValueContainerSetValue(TContainerType&  rData, TVariableType const& rV, typename TVariableType::Type const& rValue)
{
rData.SetValue(rV, rValue);
}

static typename TVariableType::Type DataValueContainerGetValue(TContainerType const& rData, TVariableType const& rV)
{
return rData.GetValue(rV);
}

inline
static typename TVariableType::Type const& DataValueContainerGetReference(TContainerType const& rData, TVariableType const& rV)
{
return rData.GetValue(rV);
}


static bool DataValueContainerHas(TContainerType const& rData, TVariableType const& rV)
{
return rData.Has(rV);
}






VariableIndexingPython& operator=(const VariableIndexingPython& rOther);


}; 





}  


