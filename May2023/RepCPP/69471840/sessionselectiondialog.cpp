

#include "wx/wxprec.h"

#ifdef __BORLANDC__
#pragma hdrstop
#endif

#ifndef WX_PRECOMP
#include "wx/wx.h"
#endif


#include "sessionselectiondialog.h"

using namespace std;





IMPLEMENT_DYNAMIC_CLASS( SessionSelectionDialog, wxDialog )




BEGIN_EVENT_TABLE( SessionSelectionDialog, wxDialog )

EVT_LISTBOX( ID_SESSIONBOX, SessionSelectionDialog::OnSessionboxSelected )
EVT_LISTBOX_DCLICK( ID_SESSIONBOX, SessionSelectionDialog::OnSessionboxDoubleClicked )
EVT_BUTTON( wxID_CANCEL, SessionSelectionDialog::OnCancelClick )
EVT_BUTTON( wxID_OK, SessionSelectionDialog::OnOkClick )
EVT_UPDATE_UI( wxID_OK, SessionSelectionDialog::OnOkUpdate )

END_EVENT_TABLE()




SessionSelectionDialog::SessionSelectionDialog()
{
Init();
OnCreate();
}


SessionSelectionDialog::SessionSelectionDialog( wxString folderPath, bool isInitialized )
{
Init();
this->folderPath = folderPath;
this->isInitialized = isInitialized;
OnCreateNoDialog();
}

SessionSelectionDialog::SessionSelectionDialog( wxWindow* parent, wxString folderPath, bool isInitialized, wxWindowID id, const wxString& caption, const wxPoint& pos, const wxSize& size, long style )
{
Init();
Create(parent, folderPath, isInitialized, id, caption, pos, size, style);
OnCreate();
}




bool SessionSelectionDialog::Create( wxWindow* parent, wxString folderPath, bool isInitialized, wxWindowID id, const wxString& caption, const wxPoint& pos, const wxSize& size, long style )
{
SetExtraStyle(wxWS_EX_VALIDATE_RECURSIVELY|wxWS_EX_BLOCK_EVENTS);
wxDialog::Create( parent, id, caption, pos, size, style );

CreateControls();
if (GetSizer())
{
GetSizer()->SetSizeHints(this);
}
Centre();
this->folderPath = folderPath;
this->isInitialized = isInitialized;
return true;
}




SessionSelectionDialog::~SessionSelectionDialog()
{
}




void SessionSelectionDialog::Init()
{
textDialogDescription = nullptr;
listSessions = nullptr;
buttonCancel = nullptr;
buttonLoad = nullptr;
}




void SessionSelectionDialog::CreateControls()
{    
SessionSelectionDialog* itemDialog1 = this;

wxBoxSizer* itemBoxSizer2 = new wxBoxSizer(wxVERTICAL);
itemDialog1->SetSizer(itemBoxSizer2);

textDialogDescription = new wxStaticText( itemDialog1, wxID_STATIC, _("Paraver closed unexpectedly. Do you want to load any of your last crashed auto-saved Paraver sessions?"), wxDefaultPosition, wxDefaultSize, 0 );
itemBoxSizer2->Add(textDialogDescription, 0, wxALIGN_CENTER_HORIZONTAL|wxALL, 5);

wxStaticBox* itemStaticBoxSizer1Static = new wxStaticBox(itemDialog1, wxID_ANY, _("List of crashed sessions"));
wxStaticBoxSizer* itemStaticBoxSizer1 = new wxStaticBoxSizer(itemStaticBoxSizer1Static, wxHORIZONTAL);
itemBoxSizer2->Add(itemStaticBoxSizer1, 3, wxGROW|wxALL, 5);

wxArrayString listSessionsStrings;
listSessions = new wxListBox( itemDialog1, ID_SESSIONBOX, wxDefaultPosition, wxSize(500, 270), listSessionsStrings, wxLB_SINGLE );
itemStaticBoxSizer1->Add(listSessions, 1, wxGROW|wxALL, 5);

wxStdDialogButtonSizer* itemStdDialogButtonSizer2 = new wxStdDialogButtonSizer;

itemBoxSizer2->Add(itemStdDialogButtonSizer2, 0, wxGROW|wxALL, 5);
buttonCancel = new wxButton( itemDialog1, wxID_CANCEL, _("&Cancel"), wxDefaultPosition, wxDefaultSize, 0 );
itemStdDialogButtonSizer2->AddButton(buttonCancel);

buttonLoad = new wxButton( itemDialog1, wxID_OK, _("&Load Session"), wxDefaultPosition, wxDefaultSize, 0 );
itemStdDialogButtonSizer2->AddButton(buttonLoad);

itemStdDialogButtonSizer2->Realize();



textDialogDescription->Show( !isInitialized );
textDialogDescription->GetFont().SetWeight( wxFONTWEIGHT_BOLD );
if ( isInitialized )
{
textDialogDescription->SetLabel( _("Select one of your last auto-saved Paraver sessions") );
itemStaticBoxSizer1Static->SetLabel( _("List of auto-saved sessions") );
}
}




void SessionSelectionDialog::OnSessionboxSelected( wxCommandEvent& event )
{
if ( listSessions->GetSelection() != wxNOT_FOUND )
myPath = linksPerFileName[ listSessions->GetString( listSessions->GetSelection() ) ];
}




void SessionSelectionDialog::OnSessionboxDoubleClicked( wxCommandEvent& event )
{
EndModal( wxID_OK );
}




void SessionSelectionDialog::OnOkUpdate( wxUpdateUIEvent& event )
{
buttonLoad->Enable( listSessions->GetSelection() != wxNOT_FOUND );
}




bool SessionSelectionDialog::ShowToolTips()
{
return true;
}



wxBitmap SessionSelectionDialog::GetBitmapResource( const wxString& name )
{
wxUnusedVar(name);
return wxNullBitmap;
}



wxIcon SessionSelectionDialog::GetIconResource( const wxString& name )
{
wxUnusedVar(name);
return wxNullIcon;
}


bool SessionSelectionDialog::OnCreate()
{
if ( wxDirExists( folderPath ) ) 
{
wxArrayString filesInDir;
if ( isInitialized )
wxDir::GetAllFiles( folderPath, &filesInDir, wxT( "*.session" ), wxDIR_FILES );
else
wxDir::GetAllFiles( folderPath, &filesInDir, wxT( "*0.session" ), wxDIR_FILES );

if ( filesInDir.size() == 0 )
return false;

listSessions->Clear();
linksPerFileName.clear();

map< boost::posix_time::ptime, wxString, std::greater< boost::posix_time::ptime > > dtToFile;
for ( size_t i = 0 ; i < filesInDir.size() ; ++i )
{
#ifdef _WIN32
wxString datetime = filesInDir[ i ].AfterLast( '\\' ).AfterFirst( '_' ).Left( 15 );
#else
wxString datetime = filesInDir[ i ].AfterLast( '/' ).AfterFirst( '_' ).Left( 15 );
#endif
datetime[ 8 ] = 'T';

boost::posix_time::ptime dt;
dt = boost::posix_time::from_iso_string( std::string( datetime.mb_str() ) );

dtToFile.insert( std::pair< boost::posix_time::ptime, wxString >( dt , filesInDir[ i ] ) );
}

map< boost::posix_time::ptime, wxString, std::greater< boost::posix_time::ptime > >::iterator it;
for ( it = dtToFile.begin(); it != dtToFile.end(); ++it )
{
wxString fileName = FormatFileName( (* it ).second.AfterLast( '/' ) );
listSessions->Append( fileName );
linksPerFileName[ fileName ] = (* it ).second;
}
}
return true;
}


bool SessionSelectionDialog::OnCreateNoDialog()
{
if ( wxDirExists( folderPath ) ) 
{
wxArrayString filesInDir;
if ( isInitialized )
wxDir::GetAllFiles( folderPath, &filesInDir, wxT( "*.session" ), wxDIR_FILES );
else
wxDir::GetAllFiles( folderPath, &filesInDir, wxT( "*0.session" ), wxDIR_FILES );

if ( filesInDir.size() == 0 )
return false;

linksPerFileName.clear();

map< boost::posix_time::ptime, wxString, std::greater< boost::posix_time::ptime > > dtToFile;
for ( size_t i = 0 ; i < filesInDir.size() ; ++i )
{
#ifdef _WIN32
wxString datetime = filesInDir[ i ].AfterLast( '\\' ).AfterFirst( '_' ).Left( 15 );
#else
wxString datetime = filesInDir[ i ].AfterLast( '/' ).AfterFirst( '_' ).Left( 15 );
#endif
datetime[ 8 ] = 'T';

boost::posix_time::ptime dt;
dt = boost::posix_time::from_iso_string( std::string( datetime.mb_str() ) );

dtToFile.insert( std::pair< boost::posix_time::ptime, wxString >( dt , filesInDir[ i ] ) );
}

map< boost::posix_time::ptime, wxString, std::greater< boost::posix_time::ptime > >::iterator it;
for ( it = dtToFile.begin(); it != dtToFile.end(); ++it )
{
allFilesInDir.push_back( (* it ).second );
}
}
return true;
}



wxString SessionSelectionDialog::FormatFileName( wxString fileName )
{
std::string fileStringStd = std::string( fileName.mb_str() ) ;
wxArrayString parts;  

std::size_t end, begin = 0;
char delim = '_';
end = fileStringStd.find( delim );


wxString subPart;
while ( end != std::string::npos ) 
{
subPart = wxString( fileStringStd.substr( begin, end - begin ).c_str(), wxConvUTF8 );
parts.push_back( subPart );
begin = end + 1;
end = fileStringStd.find( delim, begin );
}
subPart = wxString( fileStringStd.substr( begin, end - begin ).c_str(), wxConvUTF8 );
parts.push_back( subPart );

wxString dmy = parts[ 1 ];
wxString hms = parts[ 2 ]; 

dmy = dmy.Mid( 6, 2 ) +  
wxT( "/" ) +
dmy.Mid( 4, 2 ) +
wxT( "/" ) +
dmy.Mid( 0, 4 );


hms = hms.Mid( 0, 2 ) +
wxT( ":" ) +
hms.Mid( 2, 2 ) +
wxT( ":" ) +
hms.Mid( 4, 2 );

wxString crash = ( parts[3] == wxT( "0.session" ) ? wxT( " [Crashed]" ) : _( "" ) );

return dmy +
wxT( " " ) +
hms + 
crash;
}



void SessionSelectionDialog::OnOkClick( wxCommandEvent& event )
{
event.Skip();
}




void SessionSelectionDialog::OnCancelClick( wxCommandEvent& event )
{
event.Skip();
}



wxString SessionSelectionDialog::GetSessionPath()
{
return myPath;
}



wxArrayString SessionSelectionDialog::GetSessionPaths()
{
return allFilesInDir;
}

