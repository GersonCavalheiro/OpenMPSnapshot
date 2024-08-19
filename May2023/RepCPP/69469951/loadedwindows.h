


#pragma once


#include <map>
#include <vector>
#include "paraverkerneltypes.h"

class Timeline;
class Histogram;
class Trace;

typedef PRV_UINT32 TWindowID;

class LoadedWindows
{
public:
~LoadedWindows();

static LoadedWindows *getInstance();
static bool validDataWindow( Timeline *dataWindow, Timeline *controlWindow );
static bool validLevelDataWindow( Timeline *dataWindow, Timeline *controlWindow );
static bool notInParents( Timeline *whichWindow, Timeline *inParents );

TWindowID add( Timeline *whichWindow );
TWindowID add( Histogram *whichHisto );
void eraseWindow( TWindowID id );
void eraseWindow( Timeline *whichWindow );
void eraseHisto( TWindowID id );
void eraseHisto( Histogram *whichHisto );
Timeline *getWindow( TWindowID id ) const;
Histogram *getHisto( TWindowID id ) const;
bool emptyWindows() const;
bool emptyHistograms() const;
void getAll( std::vector<Timeline *>& onVector ) const;
void getAll( std::vector<TWindowID>& onVector ) const;

void getAll( std::vector<Histogram *>& onVector ) const;
void getAll( Trace *whichTrace, std::vector< Timeline *>& onVector ) const;
void getDerivedCompatible( Trace *whichTrace, std::vector< Timeline *>& onVector ) const;
void getDerivedCompatible( Trace *whichTrace, std::vector<TWindowID>& onVector ) const;
void getAll( Trace *whichTrace, std::vector< Histogram *>& onVector ) const;

void getValidControlWindow( Timeline *dataWindow, Timeline *controlWindow, std::vector<TWindowID>& onVector ) const;
void getValidDataWindow( Timeline *controlWindow, Timeline *extraWindow,
std::vector<TWindowID>& onVector ) const;
protected:

private:
LoadedWindows();

static LoadedWindows *instance;

std::map<TWindowID, Timeline *> windows;
std::map<TWindowID, Histogram *> histograms;
TWindowID currentID;
TWindowID currentHistoID;

};



