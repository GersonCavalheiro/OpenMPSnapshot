


#pragma once


#include <string>
#include <vector>
#include <map>
#include "ptools_prv_types.h"

using std::string;
using std::vector;
using std::map;

class EventDescription;

class EventList
{
public:
static EventList *getInstance();

~EventList();

void init();

bool insert( string          stringID,
bool            usedInExtrae,
prvEventType_t  mpitID,
prvEventType_t  whichType,
prvEventValue_t whichValue,
string          whichStrType,
string          whichStrValue,
bool            isChangingState,
prvState_t      whichStateTransition );

bool insert( string            stringID,
prvEventType_t    mpitID,
EventDescription *whichEvent );

EventDescription *getByStringID( string whichStringID ) const;
EventDescription *getByMpitID( prvEventType_t whichMpitID ) const;
EventDescription *getByTypeValue( prvEventType_t  whichType,
prvEventValue_t whichValue ) const;

void getUsed( vector<EventDescription *>& onVector ) const;

protected:

private:
EventList();

static EventList *instance;

vector<EventDescription *> events;
map<string, EventDescription *> stringMap;
map<prvEventType_t, EventDescription *> mpitMap;
map<prvEventType_t, map<prvEventValue_t, EventDescription *> > typeValueMap;
};



