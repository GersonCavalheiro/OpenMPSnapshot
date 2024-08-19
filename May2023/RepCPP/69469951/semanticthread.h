


#pragma once


#include "semanticfunction.h"

class SemanticThread : public SemanticFunction
{
public:
SemanticThread()
{}
~SemanticThread()
{}

virtual bool validRecord( MemoryTrace::iterator *record )
{
TRecordType type = record->getRecordType();
TRecordType mask = getValidateMask();

if ( type == EMPTYREC )
return true;

if( mask == STATE + EVENT && ( type & STATE || type & EVENT ) )
return true;

if ( mask & RSEND )
{
if ( type & RSEND )
return true;
else
mask -= RSEND;
}
else if ( mask & RRECV )
{
if ( type & RRECV )
return true;
else
mask -= RRECV;
}
return ( ( mask & type ) == mask );
}

protected:
virtual const TRecordType getValidateMask() = 0;

private:
};



