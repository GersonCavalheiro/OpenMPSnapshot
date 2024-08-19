

#include "wx/wxprec.h"

#ifdef __BORLANDC__
#pragma hdrstop
#endif

#ifndef WX_PRECOMP
#include "wx/wx.h"
#endif


#include "exitdialog.h"

#include "../icons/logout.xpm"




IMPLEMENT_DYNAMIC_CLASS( ExitDialog, wxDialog )




BEGIN_EVENT_TABLE( ExitDialog, wxDialog )

EVT_BUTTON( ID_BUTTON_SAVE_EXIT, ExitDialog::OnButtonSaveExitClick )
EVT_BUTTON( ID_BUTTON_CANCEL, ExitDialog::OnButtonCancelClick )
EVT_BUTTON( ID_BUTTON_CLOSE_NO_SAVE, ExitDialog::OnButtonCloseNoSaveClick )

END_EVENT_TABLE()




ExitDialog::ExitDialog()
{
Init();
}

ExitDialog::ExitDialog( wxWindow* parent, wxWindowID id, const wxString& caption, const wxPoint& pos, const wxSize& size, long style )
{
Init();
Create(parent, id, caption, pos, size, style);
}




bool ExitDialog::Create( wxWindow* parent, wxWindowID id, const wxString& caption, const wxPoint& pos, const wxSize& size, long style )
{
SetExtraStyle(wxWS_EX_VALIDATE_RECURSIVELY|wxWS_EX_BLOCK_EVENTS);
wxDialog::Create( parent, id, caption, pos, size, style );

CreateControls();
if (GetSizer())
{
GetSizer()->SetSizeHints(this);
}
Centre();
return true;
}




ExitDialog::~ExitDialog()
{
}




void ExitDialog::Init()
{
saveExitButton = nullptr;
cancelButton = nullptr;
noSaveExitButton = nullptr;
}




void ExitDialog::CreateControls()
{    
ExitDialog* itemDialog1 = this;

wxBoxSizer* itemBoxSizer2 = new wxBoxSizer(wxVERTICAL);
itemDialog1->SetSizer(itemBoxSizer2);

wxBoxSizer* itemBoxSizer1 = new wxBoxSizer(wxHORIZONTAL);
itemBoxSizer2->Add(itemBoxSizer1, 0, wxGROW|wxALL, 5);

wxStaticBitmap* itemStaticBitmap1 = new wxStaticBitmap( itemDialog1, wxID_STATIC, itemDialog1->GetBitmapResource(wxT("icons/logout.xpm")), wxDefaultPosition, wxSize(48, 48), 0 );
itemBoxSizer1->Add(itemStaticBitmap1, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5);

wxStaticText* itemStaticText2 = new wxStaticText( itemDialog1, wxID_STATIC, _("Some windows are already open... Do you want to save this session before closing?"), wxDefaultPosition, wxSize(270, -1), 0 );
itemBoxSizer1->Add(itemStaticText2, 1, wxGROW|wxALL, 5);

itemBoxSizer2->Add(5, 0, 1, wxGROW|wxALL, 5);

wxBoxSizer* itemBoxSizer3 = new wxBoxSizer(wxHORIZONTAL);
itemBoxSizer2->Add(itemBoxSizer3, 0, wxGROW|wxALL, 5);

saveExitButton = new wxButton( itemDialog1, ID_BUTTON_SAVE_EXIT, _("Save and exit"), wxDefaultPosition, wxDefaultSize, 0 );
itemBoxSizer3->Add(saveExitButton, 1, wxGROW|wxALL, 5);

cancelButton = new wxButton( itemDialog1, ID_BUTTON_CANCEL, _("Cancel"), wxDefaultPosition, wxDefaultSize, 0 );
itemBoxSizer3->Add(cancelButton, 1, wxGROW|wxALL, 5);

noSaveExitButton = new wxButton( itemDialog1, ID_BUTTON_CLOSE_NO_SAVE, _("Close without saving"), wxDefaultPosition, wxDefaultSize, 0 );
itemBoxSizer3->Add(noSaveExitButton, 1, wxGROW|wxALL, 5);

}





bool ExitDialog::ShowToolTips()
{
return true;
}



wxBitmap ExitDialog::GetBitmapResource( const wxString& name )
{
wxUnusedVar(name);
if (name == wxT("icons/logout.xpm"))
{
wxBitmap bitmap(logout_xpm);
return bitmap;
}
return wxNullBitmap;
}



wxIcon ExitDialog::GetIconResource( const wxString& name )
{
wxUnusedVar(name);
return wxNullIcon;
}



void ExitDialog::OnButtonSaveExitClick( wxCommandEvent& event )
{
EndModal( wxID_NO );
}



void ExitDialog::OnButtonCloseNoSaveClick( wxCommandEvent& event )
{
EndModal( wxID_YES );
}



void ExitDialog::OnButtonCancelClick( wxCommandEvent& event )
{
EndModal( wxID_CANCEL );
}
