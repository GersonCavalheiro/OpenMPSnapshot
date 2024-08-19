


#pragma once


#include <fstream>
#include <iostream>
#include <boost/serialization/string.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/archive/xml_oarchive.hpp>
#include "utils/traceparser/tracetypes.h"

struct rgb
{
template< class Archive >
void serialize( Archive & ar, const unsigned int version )
{
ar & boost::serialization::make_nvp( "red", red );
ar & boost::serialization::make_nvp( "green", green );
ar & boost::serialization::make_nvp( "blue", blue );
}


ParaverColor red;
ParaverColor green;
ParaverColor blue;

bool operator==( const rgb& b ) const
{
return red == b.red && green == b.green && blue == b.blue;
}

bool operator!=( const rgb& b ) const
{
return !( red == b.red && green == b.green && blue == b.blue );
}

bool operator<( const rgb& b ) const
{
return (red < b.red) ||
(red == b.red && blue < b.blue ) ||
(red == b.red && blue == b.blue && green < b.green);
}
};
