

#pragma once




#include <wx/filectrl.h>
#include <wx/filename.h>
#include <wx/textfile.h>
#include <wx/dir.h>
#include <map>
#include "cfg.h"




class wxFileCtrl;



#define ID_LOADCFGDIALOG 10000
#define ID_SEARCHCTRL 10004
#define ID_FILE_NAVIGATOR 10001
#define ID_TEXTLOADDESCRIPTION 10003
#define SYMBOL_LOADCFGDIALOG_STYLE wxCAPTION|wxRESIZE_BORDER|wxSYSTEM_MENU|wxCLOSE_BOX|wxTAB_TRAVERSAL
#define SYMBOL_LOADCFGDIALOG_TITLE _("Load CFG Dialog")
#define SYMBOL_LOADCFGDIALOG_IDNAME ID_LOADCFGDIALOG
#define SYMBOL_LOADCFGDIALOG_SIZE wxSize(800, 600)
#define SYMBOL_LOADCFGDIALOG_POSITION wxDefaultPosition




class LoadCFGDialog: public wxDialog
{    
DECLARE_DYNAMIC_CLASS( LoadCFGDialog )
DECLARE_EVENT_TABLE()

public:
LoadCFGDialog();
LoadCFGDialog( wxWindow* parent, wxString directoryPath = _( "" ), wxWindowID id = SYMBOL_LOADCFGDIALOG_IDNAME, const wxString& caption = SYMBOL_LOADCFGDIALOG_TITLE, const wxPoint& pos = SYMBOL_LOADCFGDIALOG_POSITION, const wxSize& size = SYMBOL_LOADCFGDIALOG_SIZE, long style = SYMBOL_LOADCFGDIALOG_STYLE );

bool Create( wxWindow* parent, wxWindowID id = SYMBOL_LOADCFGDIALOG_IDNAME, const wxString& caption = SYMBOL_LOADCFGDIALOG_TITLE, const wxPoint& pos = SYMBOL_LOADCFGDIALOG_POSITION, const wxSize& size = SYMBOL_LOADCFGDIALOG_SIZE, long style = SYMBOL_LOADCFGDIALOG_STYLE );


~LoadCFGDialog();

void Init();

void CreateControls();


void OnSearchctrlEnter( wxCommandEvent& event );

void OnTextloaddescriptionUpdate( wxUpdateUIEvent& event );

void OnCancelClick( wxCommandEvent& event );

void OnOkClick( wxCommandEvent& event );

void OnOkUpdate( wxUpdateUIEvent& event );


void OnFileNavigatorDoubleClick( wxFileCtrlEvent& event );

wxString GetFilePath();



wxBitmap GetBitmapResource( const wxString& name );

wxIcon GetIconResource( const wxString& name );

static bool ShowToolTips();

wxTextCtrl* searchBar;
wxFileCtrl* fileNavigator;
wxTextCtrl* textDescription;
wxButton* buttonCancel;
wxButton* buttonLoad;
wxString directoryStartingPath;
wxString selectedCfgFilePath;
std::map< wxString, wxString > linksPerFileName;

};
