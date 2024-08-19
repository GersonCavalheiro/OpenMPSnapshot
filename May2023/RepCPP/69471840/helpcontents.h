

#pragma once




#include "wx/html/htmlwin.h"
#include "wx/statline.h"

#include "wx/help.h"    
#include "wx/fs_zip.h"  



class wxHtmlWindow;
class wxStaticLine;



#define ID_HELPCONTENTS 10192
#define ID_HTMLWINDOW 10193
#define ID_BUTTON_INDEX 10194
#define ID_BITMAPBUTTON_BACK 10217
#define ID_BITMAPBUTTON_FORWARD 10218
#define ID_BITMAPBUTTON_DOWNLOAD 10000
#define ID_BUTTON_CLOSE 10195
#define SYMBOL_HELPCONTENTS_STYLE wxCAPTION|wxRESIZE_BORDER|wxSYSTEM_MENU|wxCLOSE_BOX|wxTAB_TRAVERSAL
#define SYMBOL_HELPCONTENTS_TITLE _("Help Contents")
#define SYMBOL_HELPCONTENTS_IDNAME ID_HELPCONTENTS
#define SYMBOL_HELPCONTENTS_SIZE wxSize(600, 600)
#define SYMBOL_HELPCONTENTS_POSITION wxDefaultPosition

#define SYMBOL_TUTORIALSBROWSER_TITLE _("Tutorials")




enum class TContents 
{
HELP,
TUTORIAL
};

class HelpContents: public wxDialog
{
DECLARE_DYNAMIC_CLASS( HelpContents )
DECLARE_EVENT_TABLE()

public:

HelpContents();
HelpContents( wxWindow* parent,
const wxString& whichHelpContentsRoot,
const bool whichLookForContents = true,
wxWindowID id = SYMBOL_HELPCONTENTS_IDNAME,
const wxString& caption = SYMBOL_HELPCONTENTS_TITLE,
const wxPoint& pos = SYMBOL_HELPCONTENTS_POSITION,
const wxSize& size = SYMBOL_HELPCONTENTS_SIZE,
long style = SYMBOL_HELPCONTENTS_STYLE );
bool Create( wxWindow* parent,
wxWindowID id = SYMBOL_HELPCONTENTS_IDNAME,
const wxString& caption = SYMBOL_HELPCONTENTS_TITLE,
const wxPoint& pos = SYMBOL_HELPCONTENTS_POSITION,
const wxSize& size = SYMBOL_HELPCONTENTS_SIZE,
long style = SYMBOL_HELPCONTENTS_STYLE );

static HelpContents* createObject( TContents whichObject,
wxWindow* parent,
const wxString& whichHelpContentsRoot,
const bool whichLookForContents = true,
wxWindowID id = SYMBOL_HELPCONTENTS_IDNAME,
const wxString& caption = SYMBOL_HELPCONTENTS_TITLE,
const wxPoint& pos = SYMBOL_HELPCONTENTS_POSITION,
const wxSize& size = SYMBOL_HELPCONTENTS_SIZE,
long style = SYMBOL_HELPCONTENTS_STYLE);

~HelpContents();

void Init();

void CreateControls();


void OnHtmlwindowLinkClicked( wxHtmlLinkEvent& event );

void OnButtonIndexClick( wxCommandEvent& event );

void OnBitmapbuttonBackClick( wxCommandEvent& event );

void OnBitmapbuttonBackUpdate( wxUpdateUIEvent& event );

void OnBitmapbuttonForwardClick( wxCommandEvent& event );

void OnBitmapbuttonForwardUpdate( wxUpdateUIEvent& event );

void OnButtonCloseClick( wxCommandEvent& event );



wxBitmap GetBitmapResource( const wxString& name );

wxIcon GetIconResource( const wxString& name );

static bool ShowToolTips();

wxHtmlWindow* htmlWindow;
wxBitmapButton* buttonIndex;
wxBitmapButton* buttonHistoryBack;
wxBitmapButton* buttonHistoryForward;
wxStaticLine* staticLineDownloadSeparator;
wxBitmapButton* buttonDownloadTutorial;

bool SetHelpContentsRoot( const std::string& whichRoot );
bool SetHelpContentsRoot( const wxString& whichRoot );

const std::string GetHelpContentsRootStr();
const wxString GetHelpContentsRoot();

void LoadHtml( const wxString& htmlFile );
bool SetHelpContents( const wxString& whichHelpContents );

static bool isHtmlDoc( const wxString& whichPath );
static bool isHtmlReferenceInDoc( const wxString& whichPath );

protected:
wxString helpContentsRoot;
bool lookForContents;
wxString currentHelpContentsDir;
wxString indexFileName;
wxString subindexLink;
wxString dialogCaption;

std::string getCurrentHelpContentsFullPath();
std::string getHrefFullPath( wxHtmlLinkEvent &event );
bool matchHrefExtension( wxHtmlLinkEvent &event, const wxString extension );

const wxString appendIndexHtmlToURL( const wxString& path );
void appendHelpContents( const wxString& title, const wxString& path, wxString& htmlDoc );
bool helpContentsFound( wxArrayString & tutorials );
bool DetectHelpContentsIndexInPath( const wxString& whichTutorial );

virtual const wxString getTitle( int numTutorial, const wxString& path );
virtual void buildIndexTemplate( wxString title, wxString filePrefix );
virtual void buildIndex();
virtual void linkToWebPage( wxString& htmlDoc );
virtual void helpMessage( wxString& htmlDoc );
};


class TutorialsBrowser: public HelpContents
{
DECLARE_DYNAMIC_CLASS( TutorialsBrowser )
DECLARE_EVENT_TABLE()

public:
TutorialsBrowser()
{}
TutorialsBrowser( wxWindow* parent,
const wxString& whichHelpContentsRoot,
wxWindowID id = wxID_ANY,
const wxString& caption = SYMBOL_TUTORIALSBROWSER_TITLE,
const wxPoint& pos = SYMBOL_HELPCONTENTS_POSITION,
const wxSize& size = SYMBOL_HELPCONTENTS_SIZE,
long style = SYMBOL_HELPCONTENTS_STYLE );
~TutorialsBrowser();

void OnHtmlwindowLinkClicked( wxHtmlLinkEvent& event );

void OnButtonDownloadClick(  wxCommandEvent& event );

protected:
const wxString getTitle( int numTutorial, const wxString& path );
void linkToWebPage( wxString& htmlDoc );
void buildIndex();
void helpMessage( wxString& htmlDoc );
};


