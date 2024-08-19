


#include "wx/wxprec.h"

#ifdef __BORLANDC__
#pragma hdrstop
#endif

#ifndef WX_PRECOMP
#include "wx/wx.h"
#endif


#include "saveconfigurationdialog.h"
#include "advancedsaveconfiguration.h"


using namespace std;



IMPLEMENT_DYNAMIC_CLASS( SaveConfigurationDialog, wxDialog )




BEGIN_EVENT_TABLE( SaveConfigurationDialog, wxDialog )

EVT_CHOICE( ID_CHOICE_TRACE_SELECTOR, SaveConfigurationDialog::OnChoiceTraceSelectorSelected )
EVT_BUTTON( ID_BUTTON_SET_ALL_TIMELINES, SaveConfigurationDialog::OnButtonSetAllTimelinesClick )
EVT_BUTTON( ID_BUTTON_UNSET_ALL_TIMELINES, SaveConfigurationDialog::OnButtonUnsetAllTimelinesClick )
EVT_BUTTON( ID_BUTTON_SET_ALL_HISTOGRAMS, SaveConfigurationDialog::OnButtonSetAllHistogramsClick )
EVT_BUTTON( ID_BUTTON_UNSET_ALL_HISTOGRAMS, SaveConfigurationDialog::OnButtonUnsetAllHistogramsClick )
EVT_BUTTON( wxID_SAVE, SaveConfigurationDialog::OnSaveClick )

END_EVENT_TABLE()




SaveConfigurationDialog::SaveConfigurationDialog()
{
Init();
}

SaveConfigurationDialog::SaveConfigurationDialog( wxWindow* parent, wxWindowID id, const wxString& caption, const wxPoint& pos, const wxSize& size, long style )
{
Init();
Create(parent, id, caption, pos, size, style);
}




bool SaveConfigurationDialog::Create( wxWindow* parent, wxWindowID id, const wxString& caption, const wxPoint& pos, const wxSize& size, long style )
{
SetExtraStyle(wxWS_EX_BLOCK_EVENTS);
wxDialog::Create( parent, id, caption, pos, size, style );

CreateControls();
if (GetSizer())
{
GetSizer()->SetSizeHints(this);
}
Centre();
return true;
}




SaveConfigurationDialog::~SaveConfigurationDialog()
{
}




void SaveConfigurationDialog::Init()
{
initialTrace = nullptr;
choiceTraceSelector = NULL;
listTimelines = NULL;
buttonSetAllTimelines = NULL;
buttonUnsetAllTimelines = NULL;
listHistograms = NULL;
buttonSetAllHistograms = NULL;
buttonUnsetAllHistograms = NULL;
optRelativeBegin = NULL;
optRelativeEnd = NULL;
optComputeSemantic = NULL;
radioAllTrace = NULL;
radioAllWindow = NULL;
optComputeGradient = NULL;
textDescription = NULL;
checkboxSaveCFGBasicMode = NULL;
}




void SaveConfigurationDialog::CreateControls()
{    
SaveConfigurationDialog* itemDialog1 = this;

wxBoxSizer* itemBoxSizer2 = new wxBoxSizer(wxVERTICAL);
itemDialog1->SetSizer(itemBoxSizer2);

wxBoxSizer* itemBoxSizer3 = new wxBoxSizer(wxHORIZONTAL);
itemBoxSizer2->Add(itemBoxSizer3, 2, wxGROW|wxALL, 0);

wxBoxSizer* itemBoxSizer4 = new wxBoxSizer(wxVERTICAL);
itemBoxSizer3->Add(itemBoxSizer4, 2, wxGROW|wxALL, 5);

wxArrayString choiceTraceSelectorStrings;
choiceTraceSelector = new wxChoice( itemDialog1, ID_CHOICE_TRACE_SELECTOR, wxDefaultPosition, wxDefaultSize, choiceTraceSelectorStrings, 0 );
itemBoxSizer4->Add(choiceTraceSelector, 0, wxGROW|wxLEFT|wxRIGHT|wxTOP, 5);

wxBoxSizer* itemBoxSizer6 = new wxBoxSizer(wxHORIZONTAL);
itemBoxSizer4->Add(itemBoxSizer6, 1, wxGROW|wxLEFT|wxRIGHT|wxTOP, 5);

wxBoxSizer* itemBoxSizer7 = new wxBoxSizer(wxVERTICAL);
itemBoxSizer6->Add(itemBoxSizer7, 1, wxGROW|wxRIGHT|wxTOP, 5);

wxStaticText* itemStaticText8 = new wxStaticText( itemDialog1, wxID_STATIC, _("Timelines"), wxDefaultPosition, wxDefaultSize, 0 );
itemBoxSizer7->Add(itemStaticText8, 0, wxALIGN_CENTER_HORIZONTAL|wxLEFT|wxRIGHT, 5);

wxArrayString listTimelinesStrings;
listTimelines = new wxCheckListBox( itemDialog1, ID_LISTTIMELINES, wxDefaultPosition, wxSize(200, -1), listTimelinesStrings, wxLB_SINGLE );
itemBoxSizer7->Add(listTimelines, 1, wxGROW|wxRIGHT|wxTOP|wxBOTTOM, 5);

wxBoxSizer* itemBoxSizer10 = new wxBoxSizer(wxHORIZONTAL);
itemBoxSizer7->Add(itemBoxSizer10, 0, wxALIGN_CENTER_HORIZONTAL, 5);

buttonSetAllTimelines = new wxButton( itemDialog1, ID_BUTTON_SET_ALL_TIMELINES, _("Set all"), wxDefaultPosition, wxDefaultSize, 0 );
itemBoxSizer10->Add(buttonSetAllTimelines, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5);

buttonUnsetAllTimelines = new wxButton( itemDialog1, ID_BUTTON_UNSET_ALL_TIMELINES, _("Unset all"), wxDefaultPosition, wxDefaultSize, 0 );
itemBoxSizer10->Add(buttonUnsetAllTimelines, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5);

wxBoxSizer* itemBoxSizer13 = new wxBoxSizer(wxVERTICAL);
itemBoxSizer6->Add(itemBoxSizer13, 1, wxGROW|wxLEFT|wxTOP, 5);

wxStaticText* itemStaticText14 = new wxStaticText( itemDialog1, wxID_STATIC, _("Histograms"), wxDefaultPosition, wxDefaultSize, 0 );
itemBoxSizer13->Add(itemStaticText14, 0, wxALIGN_CENTER_HORIZONTAL|wxLEFT|wxRIGHT, 5);

wxArrayString listHistogramsStrings;
listHistograms = new wxCheckListBox( itemDialog1, ID_LISTHISTOGRAMS, wxDefaultPosition, wxSize(200, -1), listHistogramsStrings, wxLB_SINGLE );
itemBoxSizer13->Add(listHistograms, 1, wxGROW|wxTOP|wxBOTTOM, 5);

wxBoxSizer* itemBoxSizer16 = new wxBoxSizer(wxHORIZONTAL);
itemBoxSizer13->Add(itemBoxSizer16, 0, wxALIGN_CENTER_HORIZONTAL, 5);

buttonSetAllHistograms = new wxButton( itemDialog1, ID_BUTTON_SET_ALL_HISTOGRAMS, _("Set all"), wxDefaultPosition, wxDefaultSize, 0 );
itemBoxSizer16->Add(buttonSetAllHistograms, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5);

buttonUnsetAllHistograms = new wxButton( itemDialog1, ID_BUTTON_UNSET_ALL_HISTOGRAMS, _("Unset all"), wxDefaultPosition, wxDefaultSize, 0 );
itemBoxSizer16->Add(buttonUnsetAllHistograms, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5);

wxBoxSizer* itemBoxSizer19 = new wxBoxSizer(wxVERTICAL);
itemBoxSizer3->Add(itemBoxSizer19, 1, wxGROW|wxALL, 5);

wxStaticBox* itemStaticBoxSizer20Static = new wxStaticBox(itemDialog1, wxID_ANY, _(" Timeline options "));
wxStaticBoxSizer* itemStaticBoxSizer20 = new wxStaticBoxSizer(itemStaticBoxSizer20Static, wxVERTICAL);
itemBoxSizer19->Add(itemStaticBoxSizer20, 1, wxGROW|wxALL, 5);

optRelativeBegin = new wxCheckBox( itemDialog1, ID_CHECKBEGIN, _("Relative begin time"), wxDefaultPosition, wxDefaultSize, 0 );
optRelativeBegin->SetValue(false);
itemStaticBoxSizer20->Add(optRelativeBegin, 1, wxALIGN_LEFT|wxALL, 5);

optRelativeEnd = new wxCheckBox( itemDialog1, ID_CHECKEND, _("Relative end time"), wxDefaultPosition, wxDefaultSize, 0 );
optRelativeEnd->SetValue(false);
itemStaticBoxSizer20->Add(optRelativeEnd, 1, wxALIGN_LEFT|wxALL, 5);

optComputeSemantic = new wxCheckBox( itemDialog1, ID_CHECKSEMANTIC, _("Compute semantic scale"), wxDefaultPosition, wxDefaultSize, 0 );
optComputeSemantic->SetValue(false);
itemStaticBoxSizer20->Add(optComputeSemantic, 1, wxALIGN_LEFT|wxALL, 5);

wxStaticBox* itemStaticBoxSizer24Static = new wxStaticBox(itemDialog1, wxID_ANY, _(" Histogram options "));
wxStaticBoxSizer* itemStaticBoxSizer24 = new wxStaticBoxSizer(itemStaticBoxSizer24Static, wxVERTICAL);
itemBoxSizer19->Add(itemStaticBoxSizer24, 1, wxGROW|wxLEFT|wxRIGHT|wxTOP, 5);

radioAllTrace = new wxRadioButton( itemDialog1, ID_RADIOALLTRACE, _("All trace"), wxDefaultPosition, wxDefaultSize, wxRB_GROUP );
radioAllTrace->SetValue(false);
itemStaticBoxSizer24->Add(radioAllTrace, 1, wxALIGN_LEFT|wxALL, 5);

radioAllWindow = new wxRadioButton( itemDialog1, ID_RADIOALLWINDOW, _("All window"), wxDefaultPosition, wxDefaultSize, 0 );
radioAllWindow->SetValue(false);
itemStaticBoxSizer24->Add(radioAllWindow, 1, wxGROW|wxALL, 5);

wxStaticLine* itemStaticLine27 = new wxStaticLine( itemDialog1, wxID_STATIC, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
itemStaticBoxSizer24->Add(itemStaticLine27, 1, wxGROW|wxLEFT|wxRIGHT, 5);

optComputeGradient = new wxCheckBox( itemDialog1, ID_CHECKGRADIENT, _("Compute gradient limits"), wxDefaultPosition, wxDefaultSize, 0 );
optComputeGradient->SetValue(false);
itemStaticBoxSizer24->Add(optComputeGradient, 1, wxALIGN_LEFT|wxALL, 5);

wxStaticText* itemStaticText29 = new wxStaticText( itemDialog1, wxID_STATIC, _("Description"), wxDefaultPosition, wxDefaultSize, 0 );
itemBoxSizer2->Add(itemStaticText29, 0, wxALIGN_LEFT|wxALL, 5);

textDescription = new wxTextCtrl( itemDialog1, ID_TEXTDESCRIPTION, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_MULTILINE );
itemBoxSizer2->Add(textDescription, 1, wxGROW|wxLEFT|wxRIGHT|wxBOTTOM, 10);

wxBoxSizer* itemBoxSizer31 = new wxBoxSizer(wxHORIZONTAL);
itemBoxSizer2->Add(itemBoxSizer31, 0, wxGROW|wxALL, 5);

itemBoxSizer31->Add(5, 5, 1, wxALIGN_CENTER_VERTICAL|wxALL, 5);

checkboxSaveCFGBasicMode = new wxCheckBox( itemDialog1, ID_CHECKBOX_SAVE_BASIC_MODE, _("Save whole CFG in basic mode"), wxDefaultPosition, wxDefaultSize, 0 );
checkboxSaveCFGBasicMode->SetValue(false);
if (SaveConfigurationDialog::ShowToolTips())
checkboxSaveCFGBasicMode->SetToolTip(_("In basic mode, for every timeline or histogram selected in this dialog you can declare which properties can be renamed or hidden. Differences will be displayed in main window every time CFG's been loaded.\nIn normal mode all properties are showed in main window."));
itemBoxSizer31->Add(checkboxSaveCFGBasicMode, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5);

itemBoxSizer31->Add(5, 5, 1, wxALIGN_CENTER_VERTICAL|wxALL, 5);

wxStdDialogButtonSizer* itemStdDialogButtonSizer35 = new wxStdDialogButtonSizer;

itemBoxSizer31->Add(itemStdDialogButtonSizer35, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5);
wxButton* itemButton36 = new wxButton( itemDialog1, wxID_CANCEL, _("&Cancel"), wxDefaultPosition, wxDefaultSize, 0 );
itemStdDialogButtonSizer35->AddButton(itemButton36);

wxButton* itemButton37 = new wxButton( itemDialog1, wxID_SAVE, _("&Save"), wxDefaultPosition, wxDefaultSize, 0 );
itemStdDialogButtonSizer35->AddButton(itemButton37);

itemStdDialogButtonSizer35->Realize();

}




bool SaveConfigurationDialog::ShowToolTips()
{
return true;
}



wxBitmap SaveConfigurationDialog::GetBitmapResource( const wxString& name )
{
wxUnusedVar(name);
return wxNullBitmap;
}



wxIcon SaveConfigurationDialog::GetIconResource( const wxString& name )
{
wxUnusedVar(name);
return wxNullIcon;
}


const CFGS4DLinkedPropertiesManager& SaveConfigurationDialog::getLinkedPropertiesManager() const
{
return linkedProperties;
}


bool SaveConfigurationDialog::TransferDataToWindow()
{
set< string > auxTraces;

for( vector<Timeline *>::iterator it = timelines.begin(); it != timelines.end(); ++it )
{
auxTraces.insert( (*it)->getTrace()->getTraceNameNumbered() );
}

for( vector<Histogram *>::iterator it = histograms.begin(); it != histograms.end(); ++it )
{
auxTraces.insert( (*it)->getTrace()->getTraceNameNumbered() );
}

traces.push_back( "All traces" );
traces.insert( ++traces.begin(), auxTraces.begin(), auxTraces.end() );

wxString aux;
int firstSelection = 0;
int pos = 0;
for( vector< string >::iterator it = traces.begin(); it != traces.end(); ++it )
{
aux << wxString::FromUTF8( (*it).c_str() );
choiceTraceSelector->Append( aux );
aux.clear();

if ( initialTrace != nullptr )
{
if( initialTrace->getTraceNameNumbered() != (*it) )
{
pos++;
}
else
{
firstSelection = pos;
}
}
}

choiceTraceSelector->SetSelection( firstSelection );

wxArrayString items;
displayedTimelines.clear();
for( vector<Timeline *>::iterator it = timelines.begin(); it != timelines.end(); ++it )
{
if ( firstSelection == 0 ||
initialTrace->getTraceNameNumbered() == (*it)->getTrace()->getTraceNameNumbered()  )
{
items.Add( wxString::FromUTF8(
(*it)->getName().c_str() ) +
_( " @ " ) +
wxString::FromUTF8( (*it)->getTrace()->getTraceNameNumbered().c_str() ) );

displayedTimelines.push_back( *it ); 
}
}
if( !items.empty() )
listTimelines->InsertItems( items, 0 );

items.Clear();
displayedHistograms.clear();
for( vector<Histogram *>::iterator it = histograms.begin(); it != histograms.end(); ++it )
{
if ( firstSelection == 0 ||
initialTrace->getTraceNameNumbered() == (*it)->getTrace()->getTraceNameNumbered()  )
{
items.Add( wxString::FromUTF8(
(*it)->getName().c_str() ) +
_( " @ " ) +
wxString::FromUTF8( (*it)->getTrace()->getTraceNameNumbered().c_str() ) );

displayedHistograms.push_back( *it ); 
}
}
if( !items.empty() )
listHistograms->InsertItems( items, 0 );

optRelativeBegin->SetValue( options.windowBeginTimeRelative );
optRelativeEnd->SetValue( options.windowScaleRelative );
optComputeSemantic->SetValue( options.windowComputeYMaxOnLoad );
radioAllTrace->SetValue( options.histoAllTrace );
optComputeGradient->SetValue( options.histoComputeGradient );

checkboxSaveCFGBasicMode->SetValue( options.enabledCFG4DMode );

return true;
}


bool SaveConfigurationDialog::TransferDataFromWindow()
{
vector<Timeline *> tmpTimelines;
for( size_t i = 0; i < listTimelines->GetCount(); ++i )
{
if( listTimelines->IsChecked( i ) )
tmpTimelines.push_back( displayedTimelines[ i ] );
}
selectedTimelines.clear();
selectedTimelines = tmpTimelines;

vector<Histogram *> tmpHistograms;
for( size_t i = 0; i < listHistograms->GetCount(); ++i )
{
if( listHistograms->IsChecked( i ) )
tmpHistograms.push_back( displayedHistograms[ i ] );
}
selectedHistograms.clear();
selectedHistograms = tmpHistograms;

options.windowBeginTimeRelative = optRelativeBegin->GetValue();
options.windowScaleRelative = optRelativeEnd->GetValue();
options.windowComputeYMaxOnLoad = optComputeSemantic->GetValue();
options.histoAllTrace = radioAllTrace->GetValue();
options.histoComputeGradient = optComputeGradient->GetValue();
options.description = std::string( textDescription->GetValue().mb_str() );

options.enabledCFG4DMode = checkboxSaveCFGBasicMode->GetValue();

return true;
}




void SaveConfigurationDialog::OnSaveClick( wxCommandEvent& event )
{
TransferDataFromWindow();
if( selectedTimelines.begin() == selectedTimelines.end() && 
selectedHistograms.begin() == selectedHistograms.end() )
{
wxMessageDialog message( this, _( "No timeline or histogram selected." ), _( "Warning" ), wxOK );
message.ShowModal();
}
else
{
if ( options.enabledCFG4DMode )
{
AdvancedSaveConfiguration tagEditorDialog( (wxWindow *)this, selectedTimelines, selectedHistograms );
if ( tagEditorDialog.ShowModal() == wxID_OK )
{
linkedProperties = tagEditorDialog.getLinkedPropertiesManager();
EndModal( wxID_OK );
}
}
else
{
EndModal( wxID_OK );
}
}
}




void SaveConfigurationDialog::OnButtonSetAllTimelinesClick( wxCommandEvent& event )
{
for( unsigned int i = 0; i <= listTimelines->GetCount(); ++i )
{
listTimelines->Check( i, true );
}
}




void SaveConfigurationDialog::OnButtonUnsetAllTimelinesClick( wxCommandEvent& event )
{
for( unsigned int i = 0; i <= listTimelines->GetCount(); ++i )
{
listTimelines->Check( i, false );
}
}




void SaveConfigurationDialog::OnButtonSetAllHistogramsClick( wxCommandEvent& event )
{
for( unsigned  int i = 0; i <= listHistograms->GetCount(); ++i )
{
listHistograms->Check( i, true );
}
}




void SaveConfigurationDialog::OnButtonUnsetAllHistogramsClick( wxCommandEvent& event )
{
for( unsigned int i = 0; i <= listHistograms->GetCount(); ++i )
{
listHistograms->Check( i, false );
}
}




void SaveConfigurationDialog::OnChoiceTraceSelectorSelected( wxCommandEvent& event )
{
int index = choiceTraceSelector->GetSelection();
string selectedTrace;
if ( index > 0 )
selectedTrace = traces[ index ];

wxArrayString items;

listTimelines->Clear();
displayedTimelines.clear();
for( vector< Timeline * >::iterator it = timelines.begin(); it != timelines.end(); ++it )
{
string currentTrace = (*it)->getTrace()->getTraceNameNumbered();
if ( index == 0 || selectedTrace == currentTrace )
{
items.Add( wxString::FromUTF8( (*it)->getName().c_str() ) +
_( " @ " ) +
wxString::FromUTF8( currentTrace.c_str() ) );
displayedTimelines.push_back( *it );
}
}

if( !items.IsEmpty() )
listTimelines->InsertItems( items, 0 );

listHistograms->Clear();
items.Clear();
displayedHistograms.clear();
for( vector<Histogram *>::iterator it = histograms.begin(); it != histograms.end(); ++it )
{
string currentTrace = (*it)->getTrace()->getTraceNameNumbered();
if ( index == 0 || selectedTrace == currentTrace )
{
items.Add( wxString::FromUTF8( (*it)->getName().c_str() ) +
_( " @ " ) +
wxString::FromUTF8( currentTrace.c_str() ) );
displayedHistograms.push_back( *it );
}
}

if( !items.IsEmpty() )
listHistograms->InsertItems( items, 0 );
}

