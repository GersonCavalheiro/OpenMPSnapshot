


#pragma once


#include <string>
#include "ptools_prv_types.h"

using std::string;

class EventDescription
{
public:
EventDescription();

EventDescription( bool            usedInExtrae,
prvEventType_t  whichType,
prvEventValue_t whichValue,
string          whichStrType,
string          whichStrValue,
bool            isChangingState,
prvState_t      whichStateTransition
);

~EventDescription();

bool            getUsedInExtrae() const;
prvEventType_t  getType() const;
prvEventValue_t getValue() const;
string          getStrType() const;
string          getStrValue() const;
bool            getChangeState() const;
prvState_t      getStateTransition() const;
bool            getUsed() const;
void            setUsed( bool newValue );

protected:
bool            inExtrae;
prvEventType_t  type;
prvEventValue_t value;
string          strType;
string          strValue;
bool            changeState;
prvState_t      stateTransition;
bool            used;

private:

};



