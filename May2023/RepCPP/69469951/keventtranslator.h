


#pragma once



#include <string>
#include <vector>

#include "eventtranslator.h"
#include "ktraceeditsequence.h"

class ProgressController;
class KernelConnection;

class KEventTranslator : public EventTranslator
{
public:
KEventTranslator( const KernelConnection *myKernel,
std::string traceIn,
std::string traceOut,
std::string traceReference,
ProgressController *progress = nullptr );
~KEventTranslator();

virtual void execute( std::string traceIn,
std::string traceOut,
ProgressController *progress = nullptr ) override;

virtual bool translationEmpty() override;

private:
TraceEditSequence *mySequence;
bool tranlationEmpty;

std::vector<std::string> traces;

};


