


#pragma once

#include <map>
#include "paraverkerneltypes.h"
#include "utils/traceparser/pcffileparser.h"

class StateLabels
{
public:
static const std::string unknownLabel;

StateLabels();
StateLabels( const PCFFileParser<>& pcfParser );
~StateLabels();

void getStates( std::vector<TState>& onVector ) const;
bool getStateLabel( TState state, std::string& onStr ) const;

protected:

private:
std::map<TState, std::string> stateLabel;
};



