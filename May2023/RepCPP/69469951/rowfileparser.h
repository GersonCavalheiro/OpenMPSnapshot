


#pragma once

#include <cstddef>
#include <string>
#include <vector>
#include <map>
#include "tracetypes.h"

template< typename dummy = std::nullptr_t >
class RowFileParser
{
public:
static bool openRowFileParser( const std::string& filename, RowFileParser<dummy>& outRowFile );

RowFileParser() = default;
RowFileParser( const std::string& filename );
~RowFileParser() = default;

void dumpToFile( const std::string& filename ) const;

std::string getRowLabel( TTraceLevel whichLevel, TObjectOrder whichRow ) const;
void pushBack( TTraceLevel whichLevel, const std::string& rowLabel );

size_t getMaxLength( TTraceLevel whichLevel = TTraceLevel::NONE ) const;

protected:

private:
std::map<TTraceLevel, std::tuple< std::string, size_t, std::vector<std::string> > > levelLabels;

size_t globalMaxLength;

void dumpLevel( const std::tuple< std::string, size_t, std::vector<std::string> >& whichLevel, std::ofstream& whichFile ) const;
};

#include "rowfileparser.cpp"
