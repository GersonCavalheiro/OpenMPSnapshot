


#pragma once



#include <wx/panel.h>
#include <wx/propdlg.h>
#include <wx/button.h>
#include <wx/sizer.h>
#include <wx/checklst.h>
#include <wx/textctrl.h>

#include <wx/regex.h>
#include <wx/checkbox.h>
#include <wx/valtext.h> 
#include <wx/stattext.h>

#include <map>

#include "paraverkerneltypes.h"
#include "selectionmanagement.h"
#include "histogram.h"


class Timeline;



#define ID_ROWSSELECTIONDIALOG 10078

#define SYMBOL_ROWSSELECTIONDIALOG_STYLE wxCAPTION|wxRESIZE_BORDER|wxSYSTEM_MENU|wxCLOSE_BOX
#define SYMBOL_ROWSSELECTIONDIALOG_TITLE _("Objects Selection")
#define SYMBOL_ROWSSELECTIONDIALOG_IDNAME ID_ROWSSELECTIONDIALOG
#define SYMBOL_ROWSSELECTIONDIALOG_SIZE wxSize(400, 300)
#define SYMBOL_ROWSSELECTIONDIALOG_POSITION wxDefaultPosition


class RowsSelectionDialog: public wxPropertySheetDialog
{    
DECLARE_DYNAMIC_CLASS( RowsSelectionDialog )
DECLARE_EVENT_TABLE()

public:

RowsSelectionDialog();
RowsSelectionDialog( wxWindow* parent,
Timeline *whichWindow,
SelectionManagement< TObjectOrder, TTraceLevel > *whichSelectedRows,
wxWindowID id = SYMBOL_ROWSSELECTIONDIALOG_IDNAME,
const wxString& caption = SYMBOL_ROWSSELECTIONDIALOG_TITLE,
bool parentIsGtimeline = false,
const wxPoint& pos = SYMBOL_ROWSSELECTIONDIALOG_POSITION,
const wxSize& size = SYMBOL_ROWSSELECTIONDIALOG_SIZE,
long style = SYMBOL_ROWSSELECTIONDIALOG_STYLE );

RowsSelectionDialog( wxWindow* parent,
Histogram* histogram,
SelectionManagement< TObjectOrder, TTraceLevel > *whichSelectedRows,
wxWindowID id = SYMBOL_ROWSSELECTIONDIALOG_IDNAME,
const wxString& caption = SYMBOL_ROWSSELECTIONDIALOG_TITLE,
bool parentIsGtimeline = false,
const wxPoint& pos = SYMBOL_ROWSSELECTIONDIALOG_POSITION,
const wxSize& size = SYMBOL_ROWSSELECTIONDIALOG_SIZE,
long style = SYMBOL_ROWSSELECTIONDIALOG_STYLE );


bool Create( wxWindow* parent,
wxWindowID id = SYMBOL_ROWSSELECTIONDIALOG_IDNAME,
const wxString& caption = SYMBOL_ROWSSELECTIONDIALOG_TITLE,
const wxPoint& pos = SYMBOL_ROWSSELECTIONDIALOG_POSITION,
const wxSize& size = SYMBOL_ROWSSELECTIONDIALOG_SIZE,
long style = SYMBOL_ROWSSELECTIONDIALOG_STYLE );

~RowsSelectionDialog();

void Init();

void CreateControls();

wxBitmap GetBitmapResource( const wxString& name );

wxIcon GetIconResource( const wxString& name );

static bool ShowToolTips();

int GetSelections( TTraceLevel whichLevel, wxArrayInt &selections );

virtual bool TransferDataFromWindow();

bool ShouldChangeTimelineZoom() const { return shouldChangeTimelineZoom; }
TObjectOrder GetNewBeginZoom() const { return beginZoom; }
TObjectOrder GetNewEndZoom() const { return endZoom; }

private:
Timeline *myTimeline;
Histogram *myHistogram;

SelectionManagement< TObjectOrder, TTraceLevel > *mySelectedRows;

TTraceLevel minLevel; 
TTraceLevel myLevel;
std::vector< wxButton * > selectionButtons;
std::vector< wxCheckListBox* > levelCheckList;

bool parentIsGtimeline;   

bool shouldChangeTimelineZoom;
TObjectOrder beginZoom;
TObjectOrder endZoom;

Trace *myTrace;

std::map< TTraceLevel , std::vector< TObjectOrder > >selectedIndex;

bool lockedByUpdate;
std::vector< wxStaticText *> messageMatchesFound;
std::vector< wxCheckBox *> checkBoxPosixBasicRegExp;
std::vector< wxTextCtrl *> textCtrlRegularExpr;
std::vector< wxButton * > applyButtons;
std::vector< wxButton * > helpRE;
std::vector< wxRegEx * > validRE;

wxString getMyToolTip( const bool posixBasicRegExpTip );
void OnCheckBoxMatchPosixRegExpClicked( wxCommandEvent& event );
wxTextValidator *getValidator( bool basicPosixRegExprMode ); 
wxString buildRegularExpressionString( const wxString& enteredRE );
int countMatches( int iTab, wxRegEx *&levelRE );
void checkMatches( const int &iTab, wxRegEx *&levelRE );
void OnRegularExpressionApply( wxCommandEvent& event );
void OnRegularExpressionHelp( wxCommandEvent& event );
void OnCheckListBoxSelected( wxCommandEvent& event );

void OnSelectAllButtonClicked( wxCommandEvent& event );
void OnUnselectAllButtonClicked( wxCommandEvent& event );
void OnInvertButtonClicked( wxCommandEvent& event );
void buildPanel( const wxString& title, TTraceLevel level );

void ZoomAwareTransferData( const wxArrayInt &dialogSelections,
const std::vector< TObjectOrder > &timelineZoomRange );
void OnOkClick( wxCommandEvent& event );
};


