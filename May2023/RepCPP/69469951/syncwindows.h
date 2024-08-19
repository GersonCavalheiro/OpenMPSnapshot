


#pragma once


#include <map>
#include "paraverkerneltypes.h"

typedef unsigned int TGroupId;

class Timeline;
class Histogram;

class SyncWindows
{
public:
static SyncWindows *getInstance();
~SyncWindows();

bool addWindow( Timeline *whichWindow, TGroupId whichGroup = 0 );
bool addWindow( Histogram *whichWindow, TGroupId whichGroup = 0 );
void removeWindow( Timeline *whichWindow, TGroupId whichGroup = 0 );
void removeWindow( Histogram *whichWindow, TGroupId whichGroup = 0 );
void removeAllWindows( TGroupId whichGroup = 0 );
int getNumWindows( TGroupId whichGroup ) const;

TGroupId newGroup();
TGroupId getNumGroups() const;
void getGroups( std::vector< TGroupId >& groups ) const;
void removeAllGroups();

void broadcastTime( TGroupId whichGroup, Timeline *sendWindow, TTime beginTime, TTime endTime );
void broadcastTime( TGroupId whichGroup, Histogram *sendWindow, TTime beginTime, TTime endTime );
void getGroupTimes( TGroupId whichGroup, TTime& beginTime, TTime& endTime ) const;

private:
SyncWindows();

static SyncWindows *instance;
std::map<TGroupId, std::vector<Timeline *> > syncGroupsTimeline;
std::map<TGroupId, std::vector<Histogram *> > syncGroupsHistogram;
TGroupId lastNewGroup;
bool removingAll;

void broadcastTimeTimelines( TGroupId whichGroup, Timeline *sendWindow, TTime beginTime, TTime endTime );
void broadcastTimeHistograms( TGroupId whichGroup, Histogram *sendWindow, TTime beginTime, TTime endTime );
};


