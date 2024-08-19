

#pragma once

#include <string>
#include <fstream>

template< class StreamT >
void prvGetLine( StreamT& s, std::string& line )
{
s.getline( line );
}


inline void prvGetLine( std::fstream& s, std::string& line )
{
std::getline( s, line );
}
