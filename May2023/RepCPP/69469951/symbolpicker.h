


#pragma once


#include <vector>
#include <set>
#include "paraverkerneltypes.h"

class EventLabels;


class EventTypeSymbolPicker
{
public:
EventTypeSymbolPicker();
~EventTypeSymbolPicker();

void clear();

void insert( TEventType whichType );
void insert( std::string whichLabel );

bool pick( const EventLabels& eventLabels, std::vector<TEventType>& onVector ) const;

private:
std::vector<TEventType> eventTypes;
std::vector<std::string> eventTypeLabels;

bool makepick( const EventLabels& eventLabels, TEventType eventType, const std::string& eventLabel, std::vector<TEventType>& onVector ) const;
};



class EventValueSymbolPicker
{
public:
EventValueSymbolPicker();
~EventValueSymbolPicker();

void clear();

void insert( TSemanticValue whichValue );
void insert( std::string whichLabel );

bool pick( const EventLabels& eventLabels, std::vector<TSemanticValue>& onVector ) const;

bool getMultipleValuesFound() const;

private:
std::vector<TSemanticValue> eventValues;
std::vector<std::string> eventValueLabels;
bool multipleValuesFound;

bool makepick( const EventLabels& eventLabels, TSemanticValue eventValue,
const std::string& eventLabel,
std::set<TSemanticValue>& onValues ) const;
};


