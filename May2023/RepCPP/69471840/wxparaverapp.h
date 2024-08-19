

#pragma once





#include "wx/image.h"
#include "paravermain.h"
#include <wx/snglinst.h>
#include "wx/dirctrl.h"
#include "wx/filefn.h"
#include <time.h>
#ifdef TRACING_ENABLED
#include "extrae_user_events.h"
#endif

#ifdef _WIN32
const char PATH_SEP('\\');
#else
const char PATH_SEP('/');
#endif



class stServer;




bool launchBrowser( const wxString& htmlFile );



class wxparaverApp: public wxApp
{    
DECLARE_CLASS( wxparaverApp )
DECLARE_EVENT_TABLE()

public:
wxparaverApp();

void Init();

virtual bool OnInit();

virtual int OnRun();

virtual int OnExit();

int FilterEvent(wxEvent& event);

#if !defined _MSC_VER && !defined __MINGW32__
static void handler( int signum );
void presetUserSignals();
#endif

void ActivateGlobalTiming( wxDialog* whichDialog );
void DeactivateGlobalTiming();

void ParseCommandLine( wxCmdLineParser& paraverCommandLineParser );




TEventType GetEventTypeForCode() const { return eventTypeForCode ; }
void SetEventTypeForCode(TEventType value) { eventTypeForCode = value ; }

bool GetGlobalTiming() const { return globalTiming ; }
void SetGlobalTiming(bool value) { globalTiming = value ; }

TTime GetGlobalTimingBegin() const { return globalTimingBegin ; }
void SetGlobalTimingBegin(TTime value) { globalTimingBegin = value ; }

bool GetGlobalTimingBeginIsSet() const { return globalTimingBeginIsSet ; }
void SetGlobalTimingBeginIsSet(bool value) { globalTimingBeginIsSet = value ; }

wxDialog* GetGlobalTimingCallDialog() const { return globalTimingCallDialog ; }
void SetGlobalTimingCallDialog(wxDialog* value) { globalTimingCallDialog = value ; }

TTime GetGlobalTimingEnd() const { return globalTimingEnd ; }
void SetGlobalTimingEnd(TTime value) { globalTimingEnd = value ; }


static paraverMain* mainWindow;
static wxCmdLineEntryDesc argumentsParseSyntax[];
void ValidateSession( bool setValidate );

private:
TEventType eventTypeForCode;
bool globalTiming;
TTime globalTimingBegin;
bool globalTimingBeginIsSet;
wxDialog* globalTimingCallDialog;
TTime globalTimingEnd;
bool invalidateNoConnect;

wxLocale m_locale;

wxSingleInstanceChecker *m_checker;

stServer *m_server;

void PrintVersion();
};



DECLARE_APP(wxparaverApp)
