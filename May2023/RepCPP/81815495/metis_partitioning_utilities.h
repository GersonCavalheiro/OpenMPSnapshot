
#pragma once

#include <string>
#include <iostream>




#include "includes/define.h"


namespace Kratos {




class KRATOS_API(METIS_APPLICATION) MetisPartitioningUtilities
{
public:

KRATOS_CLASS_POINTER_DEFINITION(MetisPartitioningUtilities);


MetisPartitioningUtilities() = delete;

MetisPartitioningUtilities(MetisPartitioningUtilities const& rOther) = delete;


MetisPartitioningUtilities& operator=(MetisPartitioningUtilities const& rOther) = delete;




}; 



} 
