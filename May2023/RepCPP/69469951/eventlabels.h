


#pragma once

#include <functional>
#include <set>
#include <map>
#include "paraverkerneltypes.h"
#include "utils/traceparser/pcffileparser.h"

class EventLabels
{
public:
static const std::string unknownLabel;

EventLabels();
EventLabels( const PCFFileParser<>& pcfParser );
~EventLabels();

void getTypes( std::vector<TEventType>& onVector ) const;
void getTypes( std::function<void( TEventType, const std::string& )> insert ) const;
bool getEventTypeLabel( TEventType type, std::string& onStr ) const;
bool getEventValueLabel( TEventType type, TEventValue value, std::string& onStr ) const;
bool getEventValueLabel( TEventValue value, std::string& onStr ) const;
bool getValues( TEventType type, std::vector<std::string> &values ) const;
bool getValues( TEventType type, std::map<TEventValue, std::string> &values ) const;
void getValues( TEventType type, std::function<void( TEventValue, const std::string& )> insert ) const;

bool getEventType( const std::string& whichTypeLabel, std::vector<TEventType>& onVector ) const;
bool getEventValue( const std::string& whichValueLabel, std::multimap< TEventType, TEventValue >& onMap ) const;

protected:

private:
std::map<TEventType, std::string> eventType2Label;
std::map<TEventType, std::map<TEventValue, std::string> > eventValue2Label;

std::map<std::string, TEventType> label2eventType;
std::map<std::string, std::multimap< TEventType, TEventValue > > label2eventValue;
};



