


#pragma once


#include <string>
#include "paraverkerneltypes.h"

class KernelConnection;
class ProgressController;

class EventTranslator
{
public:
static EventTranslator *create( const KernelConnection *whichKernel,
std::string traceIn,
std::string traceOut,
std::string traceReference,
ProgressController *progress );

static std::string getID();
static std::string getName();
static std::string getExtension();

virtual ~EventTranslator()
{}

virtual void execute( std::string traceIn,
std::string traceOut,
ProgressController *progress = nullptr ) = 0;

virtual bool translationEmpty() = 0;

private:
static std::string traceToolID;
static std::string traceToolName;
static std::string traceToolExtension;
};


class EventTranslatorProxy : public EventTranslator
{
public:
virtual ~EventTranslatorProxy();

virtual void execute( std::string traceIn,
std::string traceOut,
ProgressController *progress = nullptr ) override;

virtual bool translationEmpty() override;

private:
EventTranslator *myEventTranslator;
const KernelConnection *myKernel;

EventTranslatorProxy( const KernelConnection *whichKernel,
std::string traceIn,
std::string traceOut,
std::string traceReference,
ProgressController *progress );

friend EventTranslator *EventTranslator::create( const KernelConnection *whichKernel,
std::string traceIn,
std::string traceOut,
std::string traceReference,
ProgressController *progress );
};



