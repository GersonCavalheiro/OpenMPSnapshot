

#include "wx/wxprec.h"

#ifdef __BORLANDC__
#pragma hdrstop
#endif

#ifndef WX_PRECOMP
#include "wx/wx.h"
#endif


#include "loadcfgdialog.h"

#include "paravermain.h"




IMPLEMENT_DYNAMIC_CLASS( LoadCFGDialog, wxDialog )




BEGIN_EVENT_TABLE( LoadCFGDialog, wxDialog )

EVT_TEXT_ENTER( ID_SEARCHCTRL, LoadCFGDialog::OnSearchctrlEnter )
EVT_UPDATE_UI( ID_TEXTLOADDESCRIPTION, LoadCFGDialog::OnTextloaddescriptionUpdate )
EVT_BUTTON( wxID_CANCEL, LoadCFGDialog::OnCancelClick )
EVT_BUTTON( wxID_OK, LoadCFGDialog::OnOkClick )
EVT_UPDATE_UI( wxID_OK, LoadCFGDialog::OnOkUpdate )

EVT_FILECTRL_FILEACTIVATED( ID_FILE_NAVIGATOR, LoadCFGDialog::OnFileNavigatorDoubleClick )

END_EVENT_TABLE()




LoadCFGDialog::LoadCFGDialog()
{
Init();
}


LoadCFGDialog::LoadCFGDialog( wxWindow* parent, wxString directoryStartingPath, wxWindowID id, const wxString& caption, const wxPoint& pos, const wxSize& size, long style ):
directoryStartingPath( directoryStartingPath )
{
Init();
Create(parent, id, caption, pos, size, style);
}




bool LoadCFGDialog::Create( wxWindow* parent, wxWindowID id, const wxString& caption, const wxPoint& pos, const wxSize& size, long style )
{
SetExtraStyle(wxWS_EX_VALIDATE_RECURSIVELY|wxWS_EX_BLOCK_EVENTS);
wxDialog::Create( parent, id, caption, pos, size, style );

CreateControls();
Centre();

searchBar->ChangeValue( directoryStartingPath );
fileNavigator->SetDirectory( directoryStartingPath );

return true;
}




LoadCFGDialog::~LoadCFGDialog()
{
}




void LoadCFGDialog::Init()
{
searchBar = NULL;
fileNavigator = NULL;
textDescription = NULL;
buttonCancel = NULL;
buttonLoad = NULL;
}




void LoadCFGDialog::CreateControls()
{    
LoadCFGDialog* itemDialog1 = this;

wxBoxSizer* itemBoxSizer2 = new wxBoxSizer(wxVERTICAL);
itemDialog1->SetSizer(itemBoxSizer2);

searchBar = new wxTextCtrl( itemDialog1, ID_SEARCHCTRL, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
itemBoxSizer2->Add(searchBar, 0, wxGROW|wxALL, 5);

fileNavigator = new wxFileCtrl( itemDialog1,ID_FILE_NAVIGATOR,wxEmptyString,wxEmptyString,"CFG (*.cfg)|*.cfg",wxSIMPLE_BORDER|wxFC_OPEN,wxDefaultPosition,wxDefaultSize );
itemBoxSizer2->Add(fileNavigator, 3, wxGROW|wxALL, 5);

wxStaticBox* itemStaticBoxSizer6Static = new wxStaticBox(itemDialog1, wxID_STATIC, _("Description"));
wxStaticBoxSizer* itemStaticBoxSizer6 = new wxStaticBoxSizer(itemStaticBoxSizer6Static, wxHORIZONTAL);
itemBoxSizer2->Add(itemStaticBoxSizer6, 1, wxGROW|wxALL, 5);

textDescription = new wxTextCtrl( itemDialog1, ID_TEXTLOADDESCRIPTION, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_MULTILINE|wxTE_READONLY|wxNO_BORDER );
if (LoadCFGDialog::ShowToolTips())
textDescription->SetToolTip(_("Shows the description of a configuration (.cfg) file."));
itemStaticBoxSizer6->Add(textDescription, 1, wxGROW|wxALL, 5);

wxStdDialogButtonSizer* itemStdDialogButtonSizer8 = new wxStdDialogButtonSizer;

itemBoxSizer2->Add(itemStdDialogButtonSizer8, 0, wxGROW|wxALL, 5);
buttonCancel = new wxButton( itemDialog1, wxID_CANCEL, _("Cancel"), wxDefaultPosition, wxDefaultSize, 0 );
itemStdDialogButtonSizer8->AddButton(buttonCancel);

buttonLoad = new wxButton( itemDialog1, wxID_OK, _("&Load"), wxDefaultPosition, wxDefaultSize, 0 );
itemStdDialogButtonSizer8->AddButton(buttonLoad);

itemStdDialogButtonSizer8->Realize();


#if !defined( __WXGTK3__ ) && !defined( __WXOSX__ ) && !defined( __WXMSW__ )
searchBar->Hide();
#endif

searchBar->ChangeValue( directoryStartingPath );
fileNavigator->SetDirectory( directoryStartingPath );
}

bool LoadCFGDialog::ShowToolTips()
{
return true;
}



wxBitmap LoadCFGDialog::GetBitmapResource( const wxString& name )
{
wxUnusedVar(name);
return wxNullBitmap;
}



wxIcon LoadCFGDialog::GetIconResource( const wxString& name )
{
wxUnusedVar(name);
return wxNullIcon;
}




void LoadCFGDialog::OnCancelClick( wxCommandEvent& event )
{
event.Skip();
}




void LoadCFGDialog::OnOkClick( wxCommandEvent& event )
{
EndModal( wxID_OK );
}




void LoadCFGDialog::OnOkUpdate( wxUpdateUIEvent& event )
{
buttonLoad->Enable( !fileNavigator->GetFilename().IsEmpty() &&
CFGLoader::isCFGFile( std::string( fileNavigator->GetPath().mb_str() ) ) );
}


wxString LoadCFGDialog::GetFilePath()
{
return selectedCfgFilePath;
}




void LoadCFGDialog::OnSearchctrlEnter( wxCommandEvent& event )
{
wxString myPath = searchBar->GetValue();
if ( wxDirExists( myPath ) ) 
{
fileNavigator->SetDirectory( myPath );
textDescription->Clear();
}
}




void LoadCFGDialog::OnTextloaddescriptionUpdate( wxUpdateUIEvent& event )
{
wxString myPath = fileNavigator->GetPath();
selectedCfgFilePath = myPath;

std::string description = "";
if ( !fileNavigator->GetFilename().IsEmpty() )
{
if ( !CFGLoader::isCFGFile( std::string( myPath.mb_str() ) ) )
description = "*Not a Paraver CFG file!*";
else if( !CFGLoader::loadDescription( std::string( myPath.mb_str() ), description ) )
description = "*No description available*";
}

textDescription->ChangeValue( description );
}


void LoadCFGDialog::OnFileNavigatorDoubleClick( wxFileCtrlEvent& event )
{
if( !CFGLoader::isCFGFile( std::string( fileNavigator->GetPath().mb_str() ) ) )
return;

EndModal( wxID_OK );
}
