


#pragma once


#include <map>

#include "window.h"
#include "histogram.h"
#include "gtimeline.h"
#include "ghistogram.h"

constexpr char STR_COPY[] =                   "Copy\tCTRL+C";
constexpr char STR_PASTE[] =                  "Paste";
constexpr char STR_PASTE_DEFAULT_SPECIAL[] =  "Default\tCTRL+V";
constexpr char STR_PASTE_SPECIAL[] =          "Paste Special...";
constexpr char STR_TIME[] =                   "Time";
constexpr char STR_SIZE[] =                   "Size";
constexpr char STR_OBJECTS[] =                "Objects";
constexpr char STR_DURATION[] =               "Duration";
constexpr char STR_PASTE_SEMANTIC_SORT[] =    "Semantic Sort";
constexpr char STR_SEMANTIC_SCALE[] =         "Semantic Scale";
constexpr char STR_FILTER[] =                 "Filter";
constexpr char STR_FILTER_COMMS[] =           "Communications";
constexpr char STR_FILTER_COMMS_XT[] =        "Filter:Communications";
constexpr char STR_FILTER_EVENTS[] =          "Events";
constexpr char STR_FILTER_EVENTS_XT[] =       "Filter:Events";
constexpr char STR_FILTER_ALL[] =             "All";
constexpr char STR_CLONE[] =                  "Clone";
constexpr char STR_RENAME[] =                 "Rename\tF2";
constexpr char STR_FIT_TIME[] =               "Fit Time Scale";
constexpr char STR_FIT_SEMANTIC[] =           "Fit Semantic Scale";
constexpr char STR_FIT_OBJECTS[] =            "Fit Objects";
constexpr char STR_CONTROL_SCALE[] =          "Control Scale";
constexpr char STR_CONTROL_DIMENSIONS[] =     "Control Dimensions";
constexpr char STR_3D_SCALE[] =               "3D scale";
constexpr char STR_SYNCHRONIZE[] =            "Synchronize";
constexpr char STR_SYNC_REMOVE_GROUP[] =      "Remove group";
constexpr char STR_SYNC_REMOVE_ALL_GROUPS[] = "Remove all groups";
constexpr char STR_SYNC_NEW_GROUP[] =         "New group";
constexpr char STR_SAVE_IMAGE[] =             "Save Image...";
constexpr char STR_SAVE_IMAGE_LEGEND[] =      "Image Legend...";
constexpr char STR_UNDO_ZOOM[] =              "Undo Zoom\tCTRL+U";
constexpr char STR_REDO_ZOOM[] =              "Redo Zoom\tCTRL+R";
constexpr char STR_AUTOFIT_CONTROL_ZERO[] =   "Auto Fit Control Scale Zeros";
constexpr char STR_CUSTOM_PALETTE[] =         "Custom Palette";

class gTimeline;
class gHistogram;

class gPasteWindowProperties
{
public:
static gPasteWindowProperties *getInstance();

~gPasteWindowProperties();

void copy( gTimeline *whichTimeline );
void copy( gHistogram* whichHistogram );

void paste( gTimeline* destinyTimeline, const std::string property );
void paste( gHistogram* destinyHistogram, const std::string property );

bool isAllowed( gTimeline* destinyTimeline, const std::string tag );
bool isAllowed( gHistogram* destinyHistogram, const std::string tag );

TRecordTime getBeginTime();
TRecordTime getEndTime();

void verifyRemove( gTimeline *destinyTimeline );
void verifyRemove( gHistogram *destinyHistogram );

private:
static gPasteWindowProperties *pasteWindowProperties;

gTimeline  *sourceTimeline;
gHistogram *sourceHistogram;
std::map < const std::string, std::vector< std::vector < std::vector< bool > > > > allowed;

gPasteWindowProperties();

void commonMenuSettings( );
void commonTimeSettings( TRecordTime destinyTraceEndTime );
void commonFilterSettings( gTimeline *destinyTimeline );
bool seekAllowed( const std::string property, int destiny, gTimeline* destinyTimeline );
bool seekAllowed( const std::string property, int destiny, gHistogram* destinyHistogram );
};


