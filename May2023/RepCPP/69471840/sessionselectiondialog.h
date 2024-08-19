

#pragma once





#include <wx/filename.h>
#include <wx/textfile.h>
#include <wx/dir.h>
#include <vector>
#include <map>
#include <algorithm>
#include "boost/date_time/posix_time/posix_time.hpp"






#define ID_SESSIONSELECTIONDIALOG 10000
#define ID_SESSIONBOX 10001
#define SYMBOL_SESSIONSELECTIONDIALOG_STYLE wxCAPTION|wxRESIZE_BORDER|wxSYSTEM_MENU|wxCLOSE_BOX|wxTAB_TRAVERSAL
#define SYMBOL_SESSIONSELECTIONDIALOG_TITLE _("Session Selection Dialog")
#define SYMBOL_SESSIONSELECTIONDIALOG_IDNAME ID_SESSIONSELECTIONDIALOG
#define SYMBOL_SESSIONSELECTIONDIALOG_SIZE wxSize(400, 300)
#define SYMBOL_SESSIONSELECTIONDIALOG_POSITION wxDefaultPosition




class SessionSelectionDialog: public wxDialog
{    
DECLARE_DYNAMIC_CLASS( SessionSelectionDialog )
DECLARE_EVENT_TABLE()

public:
SessionSelectionDialog();
SessionSelectionDialog( wxString folderPath, bool isInitialized = false );
SessionSelectionDialog( wxWindow* parent, wxString folderPath, bool isInitialized = false, wxWindowID id = SYMBOL_SESSIONSELECTIONDIALOG_IDNAME, const wxString& caption = SYMBOL_SESSIONSELECTIONDIALOG_TITLE, const wxPoint& pos = SYMBOL_SESSIONSELECTIONDIALOG_POSITION, const wxSize& size = SYMBOL_SESSIONSELECTIONDIALOG_SIZE, long style = SYMBOL_SESSIONSELECTIONDIALOG_STYLE );

bool Create( wxWindow* parent, wxString folderPath, bool isInitialized, wxWindowID id = SYMBOL_SESSIONSELECTIONDIALOG_IDNAME, const wxString& caption = SYMBOL_SESSIONSELECTIONDIALOG_TITLE, const wxPoint& pos = SYMBOL_SESSIONSELECTIONDIALOG_POSITION, const wxSize& size = SYMBOL_SESSIONSELECTIONDIALOG_SIZE, long style = SYMBOL_SESSIONSELECTIONDIALOG_STYLE );

~SessionSelectionDialog();

void Init();

void CreateControls();


void OnSessionboxSelected( wxCommandEvent& event );

void OnSessionboxDoubleClicked( wxCommandEvent& event );

void OnCancelClick( wxCommandEvent& event );

void OnOkClick( wxCommandEvent& event );

void OnOkUpdate( wxUpdateUIEvent& event );

bool OnCreate();
bool OnCreateNoDialog();


bool GetIsInitialized() const { return isInitialized ; }
void SetIsInitialized(bool value) { isInitialized = value ; }

std::map< wxString, wxString > GetLinksPerFileName() const { return linksPerFileName ; }
void SetLinksPerFileName(std::map< wxString, wxString > value) { linksPerFileName = value ; }

wxBitmap GetBitmapResource( const wxString& name );

wxIcon GetIconResource( const wxString& name );

static bool ShowToolTips();

wxString GetSessionPath();
wxArrayString GetSessionPaths();

wxStaticText* textDialogDescription;
wxListBox* listSessions;
wxButton* buttonCancel;
wxButton* buttonLoad;
private:
bool isInitialized;
std::map< wxString, wxString > linksPerFileName;

wxString myPath;
wxString folderPath;
wxArrayString allFilesInDir;
wxString FormatFileName( wxString fileName );
};
