

#include "wx/wxprec.h"

#ifdef __BORLANDC__
#pragma hdrstop
#endif

#ifndef WX_PRECOMP
#include "wx/wx.h"
#endif


#include "wxparaverapp.h"
#include "saveimagedialog.h"
#include "labelconstructor.h"





IMPLEMENT_DYNAMIC_CLASS( SaveImageDialog, wxDialog )




BEGIN_EVENT_TABLE( SaveImageDialog, wxDialog )

EVT_TEXT_ENTER( ID_TEXTPATH, SaveImageDialog::OnTextpathEnter )
EVT_UPDATE_UI( ID_FILENAVIGATOR, SaveImageDialog::OnFilenavigatorUpdate )
EVT_CHECKBOX( ID_SAVEIMAGECHECKBOX, SaveImageDialog::OnSaveimagecheckboxClick )
EVT_CHECKBOX( ID_SAVELEGENDCHECKBOX, SaveImageDialog::OnSavelegendcheckboxClick )
EVT_BUTTON( wxID_OK, SaveImageDialog::OnOkClick )
EVT_UPDATE_UI( wxID_OK, SaveImageDialog::OnOkUpdate )
EVT_BUTTON( wxID_CANCEL, SaveImageDialog::OnCancelClick )

EVT_FILECTRL_SELECTIONCHANGED( ID_FILENAVIGATOR, SaveImageDialog::OnFileNavigatorChanged )
EVT_FILECTRL_FOLDERCHANGED( ID_FILENAVIGATOR, SaveImageDialog::OnFileNavigatorChanged )
EVT_FILECTRL_FILTERCHANGED( ID_FILENAVIGATOR, SaveImageDialog::OnFileNavigatorChanged )

END_EVENT_TABLE()

wxString SaveImageDialog::directoryStartingPath = _( "" );



SaveImageDialog::SaveImageDialog()
{
Init();
}

SaveImageDialog::SaveImageDialog( wxWindow* parent, wxString& whichStartingPath, wxString whichFileName, bool isItHistogram, wxString whichLegendSuffix, wxWindowID id, const wxString& caption, const wxPoint& pos, const wxSize& size, long style )
: defaultFileName( whichFileName ), isHistogram( isItHistogram ), legendSuffix( whichLegendSuffix )
{
if ( directoryStartingPath.IsEmpty() )
{
directoryStartingPath = whichStartingPath;
}
Init();
Create(parent, id, caption, pos, size, style);
}




bool SaveImageDialog::Create( wxWindow* parent, wxWindowID id, const wxString& caption, const wxPoint& pos, const wxSize& size, long style )
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




SaveImageDialog::~SaveImageDialog()
{
}




void SaveImageDialog::Init()
{
sizerMain = NULL;
sizerPath = NULL;
textPath = NULL;
fileNavigator = NULL;
imageToSaveSizer = NULL;
imageSizer = NULL;
imageCheckbox = NULL;
imageFileName = NULL;
legendSizer = NULL;
legendCheckbox = NULL;
legendFileName = NULL;
buttonSave = NULL;
buttonCancel = NULL;
}




void SaveImageDialog::CreateControls()
{    
SaveImageDialog* itemDialog1 = this;

sizerMain = new wxBoxSizer(wxVERTICAL);
itemDialog1->SetSizer(sizerMain);

sizerPath = new wxBoxSizer(wxHORIZONTAL);
sizerMain->Add(sizerPath, 0, wxGROW|wxALL, 5);

wxStaticText* itemStaticText2 = new wxStaticText( itemDialog1, wxID_STATIC, _("Path:"), wxDefaultPosition, wxDefaultSize, 0 );
sizerPath->Add(itemStaticText2, 1, wxALIGN_CENTER_VERTICAL|wxLEFT|wxTOP, 5);

textPath = new wxTextCtrl( itemDialog1, ID_TEXTPATH, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_PROCESS_ENTER );
sizerPath->Add(textPath, 5, wxALIGN_CENTER_VERTICAL|wxRIGHT, 5);

fileNavigator = new wxFileCtrl( itemDialog1,ID_FILENAVIGATOR,wxEmptyString,wxEmptyString,"BMP (*.bmp)|*.bmp|JPG (*.jpg;*.jpeg)|*.jpg;*.jpeg|PNG (*.png)|*.png|XPM (*.xpm)|*.xpm",wxSIMPLE_BORDER|wxFC_SAVE,wxDefaultPosition,wxSize(600, 400) );
sizerMain->Add(fileNavigator, 1, wxGROW|wxLEFT|wxRIGHT|wxTOP, 5);

wxStaticBox* itemStaticBoxSizer4Static = new wxStaticBox(itemDialog1, wxID_ANY, _(" Image to save "));
imageToSaveSizer = new wxStaticBoxSizer(itemStaticBoxSizer4Static, wxVERTICAL);
sizerMain->Add(imageToSaveSizer, 0, wxGROW|wxLEFT|wxRIGHT, 5);

imageSizer = new wxBoxSizer(wxHORIZONTAL);
imageToSaveSizer->Add(imageSizer, 0, wxGROW, 5);

imageCheckbox = new wxCheckBox( itemDialog1, ID_SAVEIMAGECHECKBOX, _("Timeline"), wxDefaultPosition, wxDefaultSize, 0 );
imageCheckbox->SetValue(true);
imageSizer->Add(imageCheckbox, 1, wxALIGN_CENTER_VERTICAL|wxLEFT, 5);

imageFileName = new wxTextCtrl( itemDialog1, ID_SAVEIMAGETEXTCTRL, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_READONLY );
imageSizer->Add(imageFileName, 4, wxALIGN_CENTER_VERTICAL|wxRIGHT|wxTOP, 5);

legendSizer = new wxBoxSizer(wxHORIZONTAL);
imageToSaveSizer->Add(legendSizer, 0, wxGROW, 5);

legendCheckbox = new wxCheckBox( itemDialog1, ID_SAVELEGENDCHECKBOX, _("Legend"), wxDefaultPosition, wxDefaultSize, 0 );
legendCheckbox->SetValue(true);
legendSizer->Add(legendCheckbox, 1, wxALIGN_CENTER_VERTICAL|wxLEFT, 5);

legendFileName = new wxTextCtrl( itemDialog1, ID_SAVELEGENDTEXTCTRL, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_READONLY );
legendSizer->Add(legendFileName, 4, wxALIGN_CENTER_VERTICAL|wxRIGHT|wxTOP|wxBOTTOM, 5);

wxStdDialogButtonSizer* itemStdDialogButtonSizer1 = new wxStdDialogButtonSizer;

sizerMain->Add(itemStdDialogButtonSizer1, 0, wxGROW|wxALL, 5);
buttonSave = new wxButton( itemDialog1, wxID_OK, _("&Save"), wxDefaultPosition, wxDefaultSize, 0 );
buttonSave->Enable(false);
itemStdDialogButtonSizer1->AddButton(buttonSave);

buttonCancel = new wxButton( itemDialog1, wxID_CANCEL, _("&Cancel"), wxDefaultPosition, wxDefaultSize, 0 );
itemStdDialogButtonSizer1->AddButton(buttonCancel);

itemStdDialogButtonSizer1->Realize();

fileTypeText = _( ".png" );

#if !defined( __WXGTK3__ ) && !defined( __WXOSX__ ) && !defined( __WXMSW__ )
sizerMain->Hide( sizerPath, true );
#endif

if ( isHistogram )
{
imageCheckbox->Hide(); 
imageToSaveSizer->Hide( (wxSizer*) legendSizer, true );
}

Layout();

textPath->ChangeValue( directoryStartingPath );
fileNavigator->SetDirectory( directoryStartingPath );
fileNavigator->SetFilterIndex( 2 );
fileNavigator->SetFilename( defaultFileName );

updateFileNamesAndPaths();
}


void SaveImageDialog::updateFileNamesAndPaths() 
{
wxFileName myPath = fileNavigator->GetPath();

if( wxDir::Exists( fileNavigator->GetDirectory() ) )
{
textPath->ChangeValue( fileNavigator->GetDirectory() );
directoryStartingPath = fileNavigator->GetDirectory();
}

fileTypeText = "." + LabelConstructor::getImageFileSuffix( static_cast< TImageFormat >( fileNavigator->GetFilterIndex() ) );

setImageFileName();
}




void SaveImageDialog::OnTextpathEnter( wxCommandEvent& event )
{
wxString myPath = textPath->GetValue(); 
if ( wxDirExists( myPath ) ) 
{
fileNavigator->SetDirectory( myPath );
directoryStartingPath = myPath;
}
}


void SaveImageDialog::setImageFileName()
{
wxString fileName = fileNavigator->GetFilename();
if ( fileName.Find( "." ) != wxNOT_FOUND )
{
wxString selectedFileSuffix = wxString( LabelConstructor::getImageFileSuffix( static_cast< TImageFormat >( fileNavigator->GetFilterIndex() ) ) );
if ( fileNavigator->GetFilename().AfterLast( '.' ) == selectedFileSuffix )
fileName = fileNavigator->GetFilename().BeforeLast( '.' );
else
{
for( int tif = (int)TImageFormat::BMP; tif <= (int)TImageFormat::XPM ; ++tif )
{
wxString ext = wxString( LabelConstructor::getImageFileSuffix( static_cast< TImageFormat >( tif ) ) );
if ( fileNavigator->GetFilename().AfterLast( '.' ) == ext )
{
fileName = fileNavigator->GetFilename().BeforeLast( '.' );
fileNavigator->SetFilterIndex( tif );
break;
}
}
}
}

selectedImageFilePath = fileName + fileTypeText;
if ( imageCheckbox->IsChecked() && imageFileName->GetValue() != selectedImageFilePath )
imageFileName->ChangeValue( selectedImageFilePath );
else if( !imageCheckbox->IsChecked() && !imageFileName->IsEmpty() )
imageFileName->Clear();

selectedLegendFilePath = fileName + legendSuffix + fileTypeText;
if ( legendCheckbox->IsChecked() && legendFileName->GetValue() != selectedLegendFilePath ) 
legendFileName->ChangeValue( selectedLegendFilePath );
else if( !legendCheckbox->IsChecked() && !legendFileName->IsEmpty() )
legendFileName->Clear();
}



void SaveImageDialog::OnSaveimagecheckboxClick( wxCommandEvent& event )
{
setImageFileName();
}




void SaveImageDialog::OnSavelegendcheckboxClick( wxCommandEvent& event )
{
setImageFileName();
}




void SaveImageDialog::OnOkClick( wxCommandEvent& event )
{
if( imageCheckbox->IsChecked() )
{
wxString tmpImageFile = GetImageFilePath();
if( wxFileExists( tmpImageFile ) )
{
wxMessageDialog tmpDialog( this, tmpImageFile + " exists.\n\nDo yo want to overwrite it?", "File exists", wxYES_NO | wxICON_QUESTION );
if( tmpDialog.ShowModal() == wxID_NO )
return;
}
}

if( legendCheckbox->IsChecked() )
{
wxString tmpLegendFile = GetLegendFilePath();
if( wxFileExists( tmpLegendFile ) )
{
wxMessageDialog tmpDialog( this, tmpLegendFile + " exists.\n\nDo yo want to overwrite it?", "File exists", wxYES_NO | wxICON_QUESTION );
if( tmpDialog.ShowModal() == wxID_NO )
return;
}
}

if ( ( imageCheckbox->IsChecked() || legendCheckbox->IsChecked() ) && !textPath->IsEmpty() )
{
directoryStartingPath = textPath->GetValue();
EndModal( wxID_OK );
}
}




void SaveImageDialog::OnCancelClick( wxCommandEvent& event )
{
event.Skip();
}




bool SaveImageDialog::ShowToolTips()
{
return true;
}



wxBitmap SaveImageDialog::GetBitmapResource( const wxString& name )
{
wxUnusedVar(name);
return wxNullBitmap;
}



wxIcon SaveImageDialog::GetIconResource( const wxString& name )
{
wxUnusedVar(name);
return wxNullIcon;
}

wxString SaveImageDialog::GetImageFilePath()
{
return directoryStartingPath + PATH_SEP + selectedImageFilePath;
}

wxString SaveImageDialog::GetLegendFilePath()
{
return directoryStartingPath + PATH_SEP + selectedLegendFilePath;
}


bool SaveImageDialog::DialogSavesImage()
{
return imageCheckbox->IsChecked();
}

bool SaveImageDialog::DialogSavesLegend()
{
return !isHistogram && legendCheckbox->IsChecked();
}


int SaveImageDialog::GetFilterIndex()
{
return fileNavigator->GetFilterIndex();
}




void SaveImageDialog::OnOkUpdate( wxUpdateUIEvent& event )
{
event.Enable( ( imageCheckbox->IsChecked() || legendCheckbox->IsChecked() ) &&
wxDir::Exists( textPath->GetValue() ) &&
!fileNavigator->GetFilename().IsEmpty() );
}


void SaveImageDialog::OnFileNavigatorChanged( wxFileCtrlEvent& event )
{
if( fileNavigator == nullptr )
return;

updateFileNamesAndPaths();
}




void SaveImageDialog::OnFilenavigatorUpdate( wxUpdateUIEvent& event )
{
setImageFileName();
}

