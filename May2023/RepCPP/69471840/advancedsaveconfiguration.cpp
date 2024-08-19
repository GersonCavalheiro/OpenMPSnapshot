



#ifdef __BORLANDC__
#pragma hdrstop
#endif

#ifndef WX_PRECOMP
#include "wx/wx.h"
#endif

#include "advancedsaveconfiguration.h"
#include "labelconstructor.h"
#include <wx/statline.h>


using namespace std;

class OriginalNameData : public wxObject
{
public:
std::string myOriginalName;
};

class CheckboxLinkData : public wxObject
{
public:
CheckboxLinkData()
{}

~CheckboxLinkData()
{}

void setPropertyName( const string& whichName )
{
propertyName = whichName;
}

void setData( Timeline *whichWindow )
{
myWindow = whichWindow;
myHistogram = nullptr;
}

void setData( Histogram *whichHistogram )
{
myWindow = nullptr;
myHistogram = whichHistogram;
}

string& getPropertyName()
{
return propertyName;
}

void getData( Timeline *&onWindow )
{
onWindow = myWindow;
}

void getData( Histogram *&onHistogram )
{
onHistogram = myHistogram;
}

private:
string propertyName;
Timeline *myWindow;
Histogram *myHistogram;
};



IMPLEMENT_DYNAMIC_CLASS( AdvancedSaveConfiguration, wxDialog )




BEGIN_EVENT_TABLE( AdvancedSaveConfiguration, wxDialog )

EVT_CHOICE( ID_CHOICE_WINDOW, AdvancedSaveConfiguration::OnChoiceWindowSelected )
EVT_TOGGLEBUTTON( ID_TOGGLEBUTTON_LIST_SELECTED, AdvancedSaveConfiguration::OnToggleOnlySelectedClick )
EVT_BUTTON( wxID_CANCEL, AdvancedSaveConfiguration::OnCancelClick )
EVT_BUTTON( wxID_SAVE, AdvancedSaveConfiguration::OnSaveClick )

END_EVENT_TABLE()

const wxString AdvancedSaveConfiguration::KParamSeparator = _( PARAM_SEPARATOR );
const wxString AdvancedSaveConfiguration::KSuffixSeparator = _( "_" );
const wxString AdvancedSaveConfiguration::KTextCtrlSuffix = AdvancedSaveConfiguration::KSuffixSeparator +
_( "TXTCTRL" );
const wxString AdvancedSaveConfiguration::KCheckBoxSuffix = AdvancedSaveConfiguration::KSuffixSeparator +
_( "CHECKBOX" );
const wxString AdvancedSaveConfiguration::KButtonSuffix   = AdvancedSaveConfiguration::KSuffixSeparator +
_( "BUTTON" );



AdvancedSaveConfiguration::AdvancedSaveConfiguration()
{
Init();
}

AdvancedSaveConfiguration::AdvancedSaveConfiguration( wxWindow* parent,
wxWindowID id,
const wxString& caption,
const wxPoint& pos,
const wxSize& size,
long style )
{
Init();
Create(parent, id, caption, pos, size, style);
}

AdvancedSaveConfiguration::AdvancedSaveConfiguration( wxWindow* parent,
const vector< Timeline * > &whichTimelines,
const vector< Histogram * > &whichHistograms,
TEditorMode whichMode,
wxWindowID id,
const wxString& caption,
const wxPoint& pos,
const wxSize& size,
long style )
{
Init();

timelines  = whichTimelines;
histograms = whichHistograms;
editionMode = whichMode;

switch ( editionMode )
{
case TEditorMode::HISTOGRAM_STATISTIC_TAGS:
for( vector< Histogram * >::iterator it = histograms.begin(); it != histograms.end(); ++it )
{
backupHistogramsCFG4DStatisticsAliasList[ *it ] = (*it)->getCFG4DStatisticsAliasList();
}
break;

case TEditorMode::PROPERTIES_TAGS:
for( vector< Timeline * >::iterator it = timelines.begin(); it != timelines.end(); ++it )
{
backupTimelinesCFG4DEnabled[ *it ] = (*it)->getCFG4DEnabled();
backupTimelinesCFG4DMode[ *it ] = (*it)->getCFG4DMode();
backupTimelinesCFG4DAliasList[ *it ] = (*it)->getCFG4DAliasList();
backupTimelinesCFG4DParamAlias[ *it ] = (*it)->getCFG4DParamAliasList();
}

for( vector< Histogram * >::iterator it = histograms.begin(); it != histograms.end(); ++it )
{
backupHistogramsCFG4DEnabled[ *it ] = (*it)->getCFG4DEnabled();
backupHistogramsCFG4DMode[ *it ] = (*it)->getCFG4DMode();
backupHistogramsCFG4DAliasList[ *it ] = (*it)->getCFG4DAliasList();
backupHistogramsCFG4DStatisticsAliasList[ *it ] = (*it)->getCFG4DStatisticsAliasList();
}
break;

default:
break;
}



Create( parent, id, caption, pos, size, style );
}




bool AdvancedSaveConfiguration::Create( wxWindow* parent, wxWindowID id, const wxString& caption, const wxPoint& pos, const wxSize& size, long style )
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




AdvancedSaveConfiguration::~AdvancedSaveConfiguration()
{
DisconnectWidgetsTagsPanel( !toggleOnlySelected->GetValue() );
}




void AdvancedSaveConfiguration::Init()
{
choiceWindow = nullptr;
scrolledWindow = nullptr;
scrolledLinkProperties = nullptr;
toggleOnlySelected = nullptr;
buttonSave = nullptr;
isTimeline = true;
currentItem = 0;
editionMode = TEditorMode::PROPERTIES_TAGS;
}




void AdvancedSaveConfiguration::CreateControls()
{    

AdvancedSaveConfiguration* itemDialog1 = this;

wxBoxSizer* itemBoxSizer2 = new wxBoxSizer(wxVERTICAL);
itemDialog1->SetSizer(itemBoxSizer2);

wxBoxSizer* itemBoxSizer3 = new wxBoxSizer(wxHORIZONTAL);
itemBoxSizer2->Add(itemBoxSizer3, 1, wxGROW|wxALL, 5);

wxBoxSizer* itemBoxSizer4 = new wxBoxSizer(wxVERTICAL);
itemBoxSizer3->Add(itemBoxSizer4, 1, wxGROW|wxRIGHT|wxFIXED_MINSIZE, 5);

wxBoxSizer* itemBoxSizer5 = new wxBoxSizer(wxHORIZONTAL);
itemBoxSizer4->Add(itemBoxSizer5, 0, wxGROW|wxALL|wxFIXED_MINSIZE, 5);

wxArrayString choiceWindowStrings;
choiceWindow = new wxChoice( itemDialog1, ID_CHOICE_WINDOW, wxDefaultPosition, wxDefaultSize, choiceWindowStrings, 0 );
itemBoxSizer5->Add(choiceWindow, 1, wxALIGN_CENTER_VERTICAL|wxALL, 5);

scrolledWindow = new wxScrolledWindow( itemDialog1, ID_SCROLLEDWINDOW1, wxDefaultPosition, wxSize(450, 400), wxSUNKEN_BORDER|wxVSCROLL|wxTAB_TRAVERSAL );
itemBoxSizer4->Add(scrolledWindow, 1, wxGROW|wxALL|wxFIXED_MINSIZE, 5);
scrolledWindow->SetScrollbars(15, 15, 0, 0);

wxBoxSizer* itemBoxSizer8 = new wxBoxSizer(wxVERTICAL);
itemBoxSizer3->Add(itemBoxSizer8, 1, wxGROW|wxLEFT, 5);

wxStaticBox* itemStaticBoxSizer1Static = new wxStaticBox(itemDialog1, wxID_ANY, _("Link editor"));
wxStaticBoxSizer* itemStaticBoxSizer1 = new wxStaticBoxSizer(itemStaticBoxSizer1Static, wxHORIZONTAL);
itemBoxSizer8->Add(itemStaticBoxSizer1, 1, wxGROW|wxALL, 1);

scrolledLinkProperties = new wxScrolledWindow( itemDialog1, ID_SCROLLED_LINK_PROPERTIES, wxDefaultPosition, wxDefaultSize, wxHSCROLL|wxVSCROLL );
itemStaticBoxSizer1->Add(scrolledLinkProperties, 1, wxGROW|wxALL, 0);
scrolledLinkProperties->SetScrollbars(1, 1, 0, 0);

wxBoxSizer* itemBoxSizer10 = new wxBoxSizer(wxHORIZONTAL);
itemBoxSizer2->Add(itemBoxSizer10, 0, wxGROW|wxALL, 5);

toggleOnlySelected = new wxToggleButton( itemDialog1, ID_TOGGLEBUTTON_LIST_SELECTED, _("View selected"), wxDefaultPosition, wxDefaultSize, 0 );
toggleOnlySelected->SetValue(false);
toggleOnlySelected->SetName(wxT("List Selected"));
itemBoxSizer10->Add(toggleOnlySelected, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5);

itemBoxSizer10->Add(5, 5, 1, wxALIGN_CENTER_VERTICAL|wxALL, 5);

wxStdDialogButtonSizer* itemStdDialogButtonSizer13 = new wxStdDialogButtonSizer;

itemBoxSizer10->Add(itemStdDialogButtonSizer13, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5);
wxButton* itemButton14 = new wxButton( itemDialog1, wxID_CANCEL, _("&Cancel"), wxDefaultPosition, wxDefaultSize, 0 );
itemStdDialogButtonSizer13->AddButton(itemButton14);

buttonSave = new wxButton( itemDialog1, wxID_SAVE, _("&Save"), wxDefaultPosition, wxDefaultSize, 0 );
itemStdDialogButtonSizer13->AddButton(buttonSave);

itemStdDialogButtonSizer13->Realize();



for ( vector< Timeline * >::iterator it = timelines.begin(); it != timelines.end(); ++it )
{
choiceWindow->Append( BuildName( *it ) );
initLinks( *it );
}

for ( vector< Histogram * >::iterator it = histograms.begin(); it != histograms.end(); ++it )
{
choiceWindow->Append( BuildName( *it ) );
initLinks( *it );
}

updateLinkPropertiesWidgets();

currentItem = 0;
bool showFullList = !toggleOnlySelected->GetValue();
if ( timelines.size() > 0 )
{
isTimeline = true;
BuildTagsPanel( timelines[ currentItem ], showFullList );
}
else
{
isTimeline = false;
BuildTagsPanel( histograms[ currentItem ], showFullList );
}

choiceWindow->SetSelection( currentItem );

if ( editionMode == TEditorMode::HISTOGRAM_STATISTIC_TAGS )
{
buttonSave->SetLabel( _("Ok") );
choiceWindow->Enable( false );
scrolledLinkProperties->Hide();
}
}


void AdvancedSaveConfiguration::initLinks( Timeline *whichTimeline )
{
auto aliasList = whichTimeline->getCFG4DAliasList();
for( auto elem : aliasList )
insertLinkInUnlinkedManager( elem.first, elem.second, whichTimeline );

TWindowLevel semLevel;
string semFunction;
TParamIndex numParameter;
auto aliasParamList = whichTimeline->getCFG4DParamAliasList();
for( auto elem : aliasParamList )
{
whichTimeline->splitCFG4DParamAliasKey( elem.first, semLevel, semFunction, numParameter );

insertLinkInUnlinkedManager( whichTimeline->getCFG4DParameterOriginalName( semLevel, numParameter ), elem.second, whichTimeline );
}
}


void AdvancedSaveConfiguration::initLinks( Histogram *whichHistogram )
{
auto aliasList = whichHistogram->getCFG4DAliasList();
for( auto elem : aliasList )
insertLinkInUnlinkedManager( elem.first, elem.second, whichHistogram );
}


wxString AdvancedSaveConfiguration::BuildName( Timeline *current )
{
return  ( wxString::FromUTF8( current->getName().c_str() ) + _( " @ " ) +
wxString::FromUTF8( current->getTrace()->getTraceNameNumbered().c_str() ) );
}


wxString AdvancedSaveConfiguration::BuildName( Histogram *current )
{
return  ( wxString::FromUTF8( current->getName().c_str() ) + _( " @ " ) +
wxString::FromUTF8( current->getTrace()->getTraceNameNumbered().c_str() ) );
}


void AdvancedSaveConfiguration::CleanTagsPanel( bool showFullList )
{
DisconnectWidgetsTagsPanel( showFullList );
scrolledWindow->DestroyChildren();
}


void AdvancedSaveConfiguration::DisconnectWidgetsTagsPanel( bool showFullList )
{
for( map< string, string >::iterator it = renamedTag.begin(); it != renamedTag.end(); ++it )
{
if ( allowedLevel( it->first ) && ( showFullList || enabledTag[ it->first ] ))
{
wxString currentCheckBoxName = wxString::FromUTF8( it->first.c_str() );
GetCheckBoxByName( currentCheckBoxName )->Disconnect(
wxEVT_COMMAND_CHECKBOX_CLICKED,
wxCommandEventHandler( AdvancedSaveConfiguration::OnCheckBoxPropertyClicked ),
nullptr,
this );
}
}
}


void AdvancedSaveConfiguration::BuildTagMaps( const map< string, string > &renamedTagMap,
const bool showFullList )
{
map< string, bool > auxEnabledFullTagList;
map< string, string > auxRenamedFullTagsList;

for( vector< string >::const_iterator it = fullTagList.begin(); it != fullTagList.end(); ++it )
{
bool aliasExists = renamedTagMap.find( *it ) != renamedTagMap.end();

auxEnabledFullTagList[ *it ] = aliasExists;

if ( aliasExists )
{
auxRenamedFullTagsList[ *it ] =  renamedTagMap.find( *it )->second;
}
else
{
if ( showFullList )
{
auxRenamedFullTagsList[ *it ] = *it;
}
else
{
auxRenamedFullTagsList[ *it ] = string( "" );
}
}
}

enabledTag = auxEnabledFullTagList;
renamedTag = auxRenamedFullTagsList;
}


void AdvancedSaveConfiguration::parseSemanticParameterTag( const wxString& whichTag,
string& onSemanticLevel,
string& onFunction,
TParamIndex& onNumParameter )
{
onSemanticLevel = whichTag.BeforeFirst( KParamSeparator[0] ).mb_str();
onFunction = whichTag.AfterLast( KParamSeparator[0] ).BeforeFirst( wxChar('.') ).mb_str();
istringstream tmpValue(
string( whichTag.BeforeLast( KParamSeparator[0] ).AfterFirst( KParamSeparator[0] ).mb_str() ) );
tmpValue >> onNumParameter;
}


void AdvancedSaveConfiguration::InsertParametersToTagMaps( const vector< Timeline::TParamAliasKey > &fullParamList, 
const Timeline::TParamAlias &renamedParamAlias,
const bool showFullList )
{
vector< string > auxFullTagList;
map< string, bool > auxEnabledFullTagsList;
map< string, string > auxRenamedFullTagsList;

TWindowLevel semanticLevel;
string function, paramAlias;
string innerKey;
TParamIndex numParameter;
bool enabled;
Timeline *currentWindow = timelines[ currentItem ]; 
vector< Timeline::TParamAliasKey > semanticLevelParamKeys;

for( vector< string >::const_iterator it = fullTagList.begin(); it != fullTagList.end(); ++it )
{
if ( allowedLevel( *it ) )
{
auxFullTagList.push_back( *it );
auxRenamedFullTagsList[ *it ] = renamedTag[ *it ];
auxEnabledFullTagsList[ *it ] = enabledTag[ *it ];

semanticLevelParamKeys = currentWindow->getCFG4DParamKeysBySemanticLevel( *it, fullParamList );

TParamIndex currentParam = 0;
for( vector< Timeline::TParamAliasKey >::const_iterator it2 = semanticLevelParamKeys.begin();
it2 != semanticLevelParamKeys.end(); ++it2 )
{
currentWindow->splitCFG4DParamAliasKey( *it2, semanticLevel, function, numParameter );

innerKey = currentWindow->getCFG4DParameterOriginalName( semanticLevel, numParameter );

if ( renamedParamAlias.find( *it2 ) != renamedParamAlias.end() )
{
enabled = true;
paramAlias = currentWindow->getCFG4DParamAlias( *it2 );
}
else
{
enabled = false;
paramAlias = currentWindow->getFunctionParamName( semanticLevel, currentParam );
}

auxFullTagList.push_back( innerKey );
auxEnabledFullTagsList[ innerKey ] = enabled;
auxRenamedFullTagsList[ innerKey ] = paramAlias;

currentParam++;
}
}
}

fullTagList = auxFullTagList;
enabledTag  = auxEnabledFullTagsList;
renamedTag  = auxRenamedFullTagsList;
}


bool AdvancedSaveConfiguration::allowedLevel( const string &tag )
{
bool allowed = false;


if ( isTimeline )
{
Timeline *currentWindow = timelines[ currentItem ];
if ( tag == SingleTimelinePropertyLabels[ SINGLE_COMPOSEWORKLOAD ] ||
tag == SingleTimelinePropertyLabels[ SINGLE_WORKLOAD ] )
{
if ( currentWindow->getLevel() == TTraceLevel::WORKLOAD )
allowed = true;
}
else if ( tag == SingleTimelinePropertyLabels[ SINGLE_COMPOSEAPPL ] ||
tag == SingleTimelinePropertyLabels[ SINGLE_APPLICATION ] )
{
if ( currentWindow->getLevel() >= TTraceLevel::WORKLOAD && currentWindow->getLevel() <= TTraceLevel::APPLICATION )
allowed = true;
}
else if ( tag == SingleTimelinePropertyLabels[ SINGLE_COMPOSETASK ] ||
tag == SingleTimelinePropertyLabels[ SINGLE_TASK ] )
{
if ( currentWindow->getLevel() >= TTraceLevel::WORKLOAD && currentWindow->getLevel() <= TTraceLevel::TASK )
allowed = true;
}
else if ( tag == SingleTimelinePropertyLabels[ SINGLE_COMPOSETHREAD ] ||
tag == SingleTimelinePropertyLabels[ SINGLE_THREAD ] )
{
allowed = true;
}
else if ( tag == SingleTimelinePropertyLabels[ SINGLE_COMPOSESYSTEM ] ||
tag == SingleTimelinePropertyLabels[ SINGLE_SYSTEM ] )
{
if ( currentWindow->getLevel() == TTraceLevel::SYSTEM )
allowed = true;
}
else if ( tag == SingleTimelinePropertyLabels[ SINGLE_COMPOSENODE ] ||
tag == SingleTimelinePropertyLabels[ SINGLE_NODE ] )
{
if ( currentWindow->getLevel() >= TTraceLevel::SYSTEM && currentWindow->getLevel() <= TTraceLevel::NODE )
allowed = true;
}
else if ( tag == SingleTimelinePropertyLabels[ SINGLE_COMPOSECPU ] ||
tag == SingleTimelinePropertyLabels[ SINGLE_CPU ] )
{
if ( currentWindow->getLevel() >= TTraceLevel::SYSTEM && currentWindow->getLevel() <= TTraceLevel::CPU )
allowed = true;
}
else
allowed = true;
}
else
allowed = true;

return allowed;
}


wxBoxSizer *AdvancedSaveConfiguration::BuildTagRowWidgets( map< string, string >::iterator it,
bool showFullList )
{
wxBoxSizer *auxBoxSizer = nullptr;
wxBoxSizer *auxBoxSizerLeft = nullptr;
wxCheckBox *auxCheckBox;
wxTextCtrl *auxTextCtrl;
wxButton   *auxButton;

wxString rowLabel;
wxString rowBaseName;

if ( showFullList || enabledTag[ it->first ] )
{
auxBoxSizer = new wxBoxSizer( wxHORIZONTAL );
auxBoxSizerLeft = new wxBoxSizer( wxHORIZONTAL );

rowLabel = wxString::FromUTF8( it->first.c_str() );
rowBaseName = rowLabel;
if ( rowLabel.AfterLast( KParamSeparator[0] ) != rowLabel )
{
rowLabel = rowLabel.AfterLast( KParamSeparator[0] );
auxBoxSizerLeft->Add( 0, 0, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5 );
}

auxCheckBox = new wxCheckBox( scrolledWindow,
wxID_ANY,
rowLabel,
wxDefaultPosition,
wxDefaultSize,
0,
wxDefaultValidator,
rowBaseName + KCheckBoxSuffix );
auxCheckBox->SetValue( enabledTag[ it->first ] );

auxBoxSizerLeft->Add( auxCheckBox, 2, wxALIGN_LEFT | wxGROW | wxALL, 2 );
auxBoxSizer->Add( auxBoxSizerLeft, 2, wxALIGN_LEFT | wxALL, 2 );

wxArrayString forbiddenChars;
forbiddenChars.Add( wxT("|") );
wxTextValidator excludeVerticalBar( wxFILTER_EXCLUDE_CHAR_LIST );
excludeVerticalBar.SetExcludes( forbiddenChars );

auxTextCtrl = new wxTextCtrl( scrolledWindow,
wxID_ANY,
wxString::FromUTF8( it->second.c_str() ),
wxDefaultPosition,
wxDefaultSize,
0,
excludeVerticalBar,
rowBaseName + KTextCtrlSuffix ); 
auxTextCtrl->Enable( enabledTag[ it->first ] );
auxTextCtrl->SetValidator( excludeVerticalBar );
OriginalNameData *tmpDataName = new OriginalNameData();
tmpDataName->myOriginalName = it->first;
auxTextCtrl->Connect( wxEVT_COMMAND_TEXT_UPDATED,
wxCommandEventHandler( AdvancedSaveConfiguration::OnTextCtrlPropertyChanged ),
tmpDataName,
this ); 

auxBoxSizer->Add( auxTextCtrl, 2, wxEXPAND | wxGROW | wxALL, 2 );

if ( editionMode == TEditorMode::PROPERTIES_TAGS )
{
if( wxString::FromUTF8( it->first.c_str() ) == _( "Statistic" ) )
{
auxButton = new wxButton( scrolledWindow,
wxID_ANY, _("..."),
wxDefaultPosition,
wxDefaultSize,
wxBU_EXACTFIT,
wxDefaultValidator,
wxString::FromUTF8( it->first.c_str() ) + KButtonSuffix );
auxButton->Enable( enabledTag[ it->first ] );
auxBoxSizer->Add( auxButton, 1, wxALIGN_CENTER_VERTICAL | wxALL, 2 );

auxButton->Connect( wxEVT_COMMAND_BUTTON_CLICKED,
wxCommandEventHandler( AdvancedSaveConfiguration::OnStatisticsButtonClick ),
nullptr,
this ); 

}
else
{
auxBoxSizer->Add(2, 2, 1, wxALIGN_CENTER_VERTICAL|wxALL, 1 );
}
}

auxCheckBox->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED,
wxCommandEventHandler( AdvancedSaveConfiguration::OnCheckBoxPropertyClicked ),
nullptr,
this ); 
}

return auxBoxSizer;
}


void AdvancedSaveConfiguration::OnTextCtrlPropertyChanged( wxCommandEvent &event )
{
string tmpOriginalName  = ( ( OriginalNameData *)event.m_callbackUserData )->myOriginalName;
string tmpCustomName = string( event.GetString().mb_str() );

if ( isTimeline && linkedManager.existsWindow( tmpOriginalName, timelines[ currentItem ] ) )
{
linkedManager.setCustomName( tmpOriginalName, tmpCustomName );
updateAliasForLinkedWindows( tmpOriginalName, tmpCustomName );
}
else if ( !isTimeline && linkedManager.existsWindow( tmpOriginalName, histograms[ currentItem ] ) )
{
linkedManager.setCustomName( tmpOriginalName, tmpCustomName );
updateAliasForLinkedWindows( tmpOriginalName, tmpCustomName );
}

updateLinkPropertiesWidgets();
}


void AdvancedSaveConfiguration::BuildTagWidgets( const bool showFullList )
{
wxBoxSizer *auxBoxSizer;
wxBoxSizer *boxSizerCurrentItem = new wxBoxSizer( wxVERTICAL );

map< string, string >::iterator it;
for( vector< string >::const_iterator itOrd = fullTagList.begin(); itOrd != fullTagList.end(); ++itOrd )
{
if ( allowedLevel( *itOrd ) )
{
it = renamedTag.find( *itOrd );

auxBoxSizer = BuildTagRowWidgets( it, showFullList );
if ( auxBoxSizer != nullptr )
{
boxSizerCurrentItem->Add( auxBoxSizer, 0, wxGROW|wxALL, 2 );
}
}
}

scrolledWindow->SetSizer( boxSizerCurrentItem );
scrolledWindow->FitInside();
}


void AdvancedSaveConfiguration::BuildTagsPanel( Timeline *currentWindow, const bool showFullList )
{
fullTagList = currentWindow->getCFG4DFullTagList();
BuildTagMaps( currentWindow->getCFG4DAliasList(), showFullList );
if ( editionMode == TEditorMode::PROPERTIES_TAGS )
{
InsertParametersToTagMaps( currentWindow->getCFG4DCurrentSelectedFullParamList(),
currentWindow->getCFG4DParamAliasList(),
showFullList );
}

BuildTagWidgets( showFullList );
}


void AdvancedSaveConfiguration::BuildTagsPanel( Histogram *currentHistogram, const bool showFullList )
{
int selected;

switch ( editionMode )
{
case TEditorMode::HISTOGRAM_STATISTIC_TAGS:
selected = ( currentHistogram->isCommunicationStat( currentHistogram->getCurrentStat() ) )? 0 : 1;
currentHistogram->getStatisticsLabels( fullTagList, selected );
BuildTagMaps( currentHistogram->getCFG4DStatisticsAliasList(), showFullList );
break;

case TEditorMode::PROPERTIES_TAGS:
fullTagList = currentHistogram->getCFG4DFullTagList();
BuildTagMaps( currentHistogram->getCFG4DAliasList(), showFullList );
break;

default:
break;
}

BuildTagWidgets( showFullList );
}



bool AdvancedSaveConfiguration::ShowToolTips()
{
return true;
}



wxBitmap AdvancedSaveConfiguration::GetBitmapResource( const wxString& name )
{
wxUnusedVar(name);
return wxNullBitmap;
}



wxIcon AdvancedSaveConfiguration::GetIconResource( const wxString& name )
{
wxUnusedVar(name);
return wxNullIcon;
}

const CFGS4DLinkedPropertiesManager& AdvancedSaveConfiguration::getLinkedPropertiesManager() const
{
return linkedManager;
}

wxCheckBox *AdvancedSaveConfiguration::GetCheckBoxByName( const wxString& widgetName ) const
{
wxString currentCheckBoxName = widgetName + KCheckBoxSuffix;
wxWindow *relatedwxWidget = scrolledWindow->FindWindowByName( currentCheckBoxName );
return static_cast<wxCheckBox *>( relatedwxWidget );
}


wxTextCtrl *AdvancedSaveConfiguration::GetTextCtrlByName( const wxString& widgetName ) const
{
wxString currentTextCtrlName = widgetName + KTextCtrlSuffix;
wxWindow *relatedwxWidget = scrolledWindow->FindWindowByName( currentTextCtrlName );
return static_cast<wxTextCtrl *>( relatedwxWidget );
}


wxButton *AdvancedSaveConfiguration::GetButtonByName( const wxString& widgetName ) const
{
wxString currentButtonName = widgetName + KButtonSuffix;
wxWindow *relatedwxWidget = scrolledWindow->FindWindowByName( currentButtonName );
return static_cast<wxButton *>( relatedwxWidget );
}


template< class T >
void AdvancedSaveConfiguration::insertLinkInUnlinkedManager( const std::string& originalName, const std::string& newCustomName, T *whichWindow )
{
bool existsCustomName = ( unlinkedManager.getLinksSize( originalName ) + linkedManager.getLinksSize( originalName ) ) > 0;

unlinkedManager.insertLink( originalName, whichWindow );

if( !existsCustomName )
unlinkedManager.setCustomName( originalName, newCustomName );
}


void AdvancedSaveConfiguration::OnCheckBoxPropertyClicked( wxCommandEvent& event )
{
wxCheckBox *currentCheckBox = static_cast<wxCheckBox *>( event.GetEventObject() );

wxString currentTextCtrlName = currentCheckBox->GetName().BeforeLast( KSuffixSeparator[0] );
GetTextCtrlByName( currentTextCtrlName )->Enable( currentCheckBox->GetValue() );

wxButton *relatedButton = GetButtonByName( currentTextCtrlName );
if ( relatedButton != nullptr )
relatedButton->Enable( currentCheckBox->GetValue() );

if( editionMode == TEditorMode::PROPERTIES_TAGS )
{
string tmpOriginalName = std::string( currentTextCtrlName.mb_str() );
if( currentCheckBox->GetValue() )
{
if( isTimeline )
insertLinkInUnlinkedManager( tmpOriginalName, std::string( GetTextCtrlByName( currentTextCtrlName )->GetValue().mb_str() ), timelines[ currentItem ] );
else
insertLinkInUnlinkedManager( tmpOriginalName, std::string( GetTextCtrlByName( currentTextCtrlName )->GetValue().mb_str() ), histograms[ currentItem ] );
}
else
{
if( isTimeline )
{
unlinkedManager.removeLink( tmpOriginalName, timelines[ currentItem ] );
linkedManager.removeLink( tmpOriginalName, timelines[ currentItem ] );
}
else
{
unlinkedManager.removeLink( tmpOriginalName, histograms[ currentItem ] );
linkedManager.removeLink( tmpOriginalName, histograms[ currentItem ] );
}
}

updateLinkPropertiesWidgets();
}
}


void AdvancedSaveConfiguration::PreparePanel( bool showFullList )
{
wxTextCtrl *currentTextCtrl;
map< string, string > auxMap;

if ( isTimeline )
{
auxMap = timelines[ currentItem ]->getCFG4DAliasList();
}
else
{
switch ( editionMode )
{
case TEditorMode::HISTOGRAM_STATISTIC_TAGS:
auxMap = histograms[ currentItem ]->getCFG4DStatisticsAliasList();
break;
case TEditorMode::PROPERTIES_TAGS:
auxMap = histograms[ currentItem ]->getCFG4DAliasList();
break;
default:
break;
}
}

for( map< string, string >::iterator it = renamedTag.begin(); it != renamedTag.end(); ++it )
{
if ( !allowedLevel( it->first ) || ( !showFullList && !enabledTag[ it->first ] ))
continue;

wxString currentTagName = wxString::FromUTF8( it->first.c_str() );

currentTextCtrl = GetTextCtrlByName( currentTagName );
if ( GetCheckBoxByName( currentTagName )->GetValue() && currentTextCtrl->GetValue().IsEmpty() )
{
if ( auxMap.find( it->first ) != auxMap.end() )
{
currentTextCtrl->SetValue( wxString::FromUTF8( auxMap.find( it->first )->second.c_str() ) );
}
else
{
currentTextCtrl->SetValue( wxString::FromUTF8( it->first.c_str() ) );
}
}
}
}

void AdvancedSaveConfiguration::TransferDataFromPanel( bool showFullList )
{
map< string, string > auxActivePropertyTags;
Timeline::TParamAliasKey auxParamKey;
Timeline::TParamAlias auxActiveParametersTags;
string semanticLevel;
string function;
TParamIndex numParameter;
string newAlias;

for( map< string, string >::iterator it = renamedTag.begin(); it != renamedTag.end(); ++it )
{
if ( !allowedLevel( it->first ) || ( !showFullList && !enabledTag[ it->first ] ))
continue;

wxString currentTagName = wxString::FromUTF8( it->first.c_str() );
enabledTag[ it->first ] = GetCheckBoxByName( currentTagName )->GetValue();

if ( enabledTag[ it->first ] )
{
if ( currentTagName.AfterLast( KParamSeparator[0] ) != currentTagName )
{
if ( isTimeline )  
{
parseSemanticParameterTag( currentTagName, semanticLevel, function, numParameter );

auxParamKey = timelines[ currentItem ]->buildCFG4DParamAliasKey( semanticLevel, function, numParameter );
newAlias = GetTextCtrlByName( currentTagName )->GetValue().mb_str();

auxActiveParametersTags[ auxParamKey ] = newAlias;
}
}
else
{
auxActivePropertyTags[ it->first ] = GetTextCtrlByName( currentTagName )->GetValue().mb_str();
}
}
}

renamedTag = auxActivePropertyTags;

if ( isTimeline )
{
timelines[ currentItem ]->setCFG4DEnabled( true );
timelines[ currentItem ]->setCFG4DMode( true );
timelines[ currentItem ]->setCFG4DAliasList( renamedTag );
timelines[ currentItem ]->setCFG4DParamAlias( auxActiveParametersTags );
}
else
{
histograms[ currentItem ]->setCFG4DEnabled( true );
histograms[ currentItem ]->setCFG4DMode( true );

switch ( editionMode )
{
case TEditorMode::HISTOGRAM_STATISTIC_TAGS:
histograms[ currentItem ]->setCFG4DStatisticsAliasList( renamedTag );
break;
case TEditorMode::PROPERTIES_TAGS:
histograms[ currentItem ]->setCFG4DAliasList( renamedTag );
default:
break;
}
}
}



void AdvancedSaveConfiguration::OnSaveClick( wxCommandEvent& event )
{
bool showFullList = !toggleOnlySelected->GetValue();
PreparePanel( showFullList );
TransferDataFromPanel( showFullList );
EndModal( wxID_OK );
}

void AdvancedSaveConfiguration::RefreshList( bool showFullList )
{
PreparePanel( showFullList );
TransferDataFromPanel( showFullList );
CleanTagsPanel( showFullList );

currentItem = GetSelectionIndexCorrected( choiceWindow->GetSelection(), isTimeline );
if ( isTimeline )
{
BuildTagsPanel( timelines[ currentItem ], showFullList );
}
else
{
BuildTagsPanel( histograms[ currentItem ], showFullList );
}
}





void AdvancedSaveConfiguration::OnChoiceWindowSelected( wxCommandEvent& event )
{
RefreshList( !toggleOnlySelected->GetValue() );
}


int AdvancedSaveConfiguration::GetSelectionIndexCorrected( int index, bool &isTimeline )
{
isTimeline = ( index <= (int)timelines.size() - 1 );
if ( !isTimeline )
{
index = index - timelines.size();
}

return index;
}


void AdvancedSaveConfiguration::OnStatisticsButtonClick( wxCommandEvent& event )
{
vector< Timeline * > dummy;
vector< Histogram * > onlyCurrentHistogram;

onlyCurrentHistogram.push_back( histograms[ currentItem ] );

AdvancedSaveConfiguration statisticsEditorDialog(
(wxWindow *)this,
dummy,
onlyCurrentHistogram,
TEditorMode::HISTOGRAM_STATISTIC_TAGS,
wxID_ANY,
_("Save Basic CFG - Statistics Editor"),
wxPoint( GetPosition().x + 20 , GetPosition().y + 20 ) ); 

if ( statisticsEditorDialog.ShowModal() == wxID_OK )
{
}
}




void AdvancedSaveConfiguration::OnToggleOnlySelectedClick( wxCommandEvent& event )
{
bool previousState = toggleOnlySelected->GetValue();

PreparePanel( previousState );
TransferDataFromPanel( previousState );
CleanTagsPanel( previousState );

bool currentState = !toggleOnlySelected->GetValue();

currentItem = GetSelectionIndexCorrected( choiceWindow->GetSelection(), isTimeline );
if ( isTimeline )
{
BuildTagsPanel( timelines[ currentItem ], currentState );
}
else
{
BuildTagsPanel( histograms[ currentItem ], currentState );
}
}





void AdvancedSaveConfiguration::OnCancelClick( wxCommandEvent& event )
{
switch ( editionMode )
{
case TEditorMode::HISTOGRAM_STATISTIC_TAGS:
for( vector< Histogram * >::iterator it = histograms.begin(); it != histograms.end(); ++it )
{
(*it)->setCFG4DStatisticsAliasList( backupHistogramsCFG4DStatisticsAliasList[ *it ] );
}
break;

case TEditorMode::PROPERTIES_TAGS:
for( vector< Timeline * >::iterator it = timelines.begin(); it != timelines.end(); ++it )
{
(*it)->setCFG4DEnabled( backupTimelinesCFG4DEnabled[ *it ] );
(*it)->setCFG4DMode( backupTimelinesCFG4DMode[ *it ] );
(*it)->setCFG4DAliasList( backupTimelinesCFG4DAliasList[ *it ] );
(*it)->setCFG4DParamAlias( backupTimelinesCFG4DParamAlias[ *it ] );
}

for( vector< Histogram * >::iterator it = histograms.begin(); it != histograms.end(); ++it )
{
(*it)->setCFG4DEnabled( backupHistogramsCFG4DEnabled[ *it ] );
(*it)->setCFG4DMode( backupHistogramsCFG4DMode[ *it ] );
(*it)->setCFG4DAliasList( backupHistogramsCFG4DAliasList[ *it ] );
(*it)->setCFG4DStatisticsAliasList( backupHistogramsCFG4DStatisticsAliasList[ *it ] );
}
break;

default:
break;
}

EndModal( wxID_CANCEL );
}

void AdvancedSaveConfiguration::OnCheckBoxLinkWindowClicked( wxCommandEvent& event )
{
Timeline *tmpWin;
Histogram *tmpHisto;
CheckboxLinkData *tmpData = ( CheckboxLinkData *)event.m_callbackUserData;

if( event.IsChecked() )
{
bool existsCustomName = linkedManager.getLinksSize( tmpData->getPropertyName() ) > 0;
string tmpCustomName;
if( !existsCustomName )
tmpCustomName = unlinkedManager.getCustomName( tmpData->getPropertyName() );
else
tmpCustomName = linkedManager.getCustomName( tmpData->getPropertyName() );

tmpData->getData( tmpWin );
if( tmpWin != nullptr )
{
unlinkedManager.removeLink( tmpData->getPropertyName(), tmpWin );
linkedManager.insertLink( tmpData->getPropertyName(), tmpWin );
if ( tmpWin == timelines[ currentItem ] )
GetTextCtrlByName( wxString::FromUTF8( tmpData->getPropertyName().c_str() ) )->ChangeValue( wxString::FromUTF8( tmpCustomName.c_str() ) );
else
{
setTimelineCFG4DAlias( tmpWin, tmpData->getPropertyName(), tmpCustomName );
}
}
else
{
tmpData->getData( tmpHisto );
if( tmpHisto != nullptr )
{
unlinkedManager.removeLink( tmpData->getPropertyName(), tmpHisto );
linkedManager.insertLink( tmpData->getPropertyName(), tmpHisto );
if ( tmpHisto == histograms[ currentItem ] )
GetTextCtrlByName( wxString::FromUTF8( tmpData->getPropertyName().c_str() ) )->ChangeValue( wxString::FromUTF8( tmpCustomName.c_str() ) );
else
tmpHisto->setCFG4DAlias( tmpData->getPropertyName(), tmpCustomName );
}
}

if( !existsCustomName )
linkedManager.setCustomName( tmpData->getPropertyName(), tmpCustomName );
}
else
{
bool existsCustomName = unlinkedManager.getLinksSize( tmpData->getPropertyName() ) > 0;
string tmpCustomName = linkedManager.getCustomName( tmpData->getPropertyName() );

tmpData->getData( tmpWin );
if( tmpWin != nullptr )
{
linkedManager.removeLink( tmpData->getPropertyName(), tmpWin );
unlinkedManager.insertLink( tmpData->getPropertyName(), tmpWin );
}
else
{
tmpData->getData( tmpHisto );
if( tmpHisto != nullptr )
{
linkedManager.removeLink( tmpData->getPropertyName(), tmpHisto );
unlinkedManager.insertLink( tmpData->getPropertyName(), tmpHisto );
}
}

if( !existsCustomName )
unlinkedManager.setCustomName( tmpData->getPropertyName(), tmpCustomName );
}

updateLinkPropertiesWidgets();
}

template <typename WindowType>
void AdvancedSaveConfiguration::buildLinkWindowWidget( wxBoxSizer *boxSizerLinks,
const string& propertyName,
WindowType *whichWindow,
bool checked )
{
wxBoxSizer *boxSizerLinkWindow;
boxSizerLinkWindow = new wxBoxSizer( wxHORIZONTAL );
boxSizerLinkWindow->Add( 0, 0, 0, wxALIGN_CENTER_VERTICAL|wxLEFT, 20 );

wxCheckBox *auxCheckBox = new wxCheckBox( scrolledLinkProperties,
wxID_ANY,
BuildName( whichWindow ),
wxDefaultPosition,
wxDefaultSize,
0,
wxDefaultValidator );
auxCheckBox->SetToolTip( wxT( "Link/Unlink window to property" ) );
auxCheckBox->SetValue( checked );
CheckboxLinkData *tmpData = new CheckboxLinkData;
tmpData->setPropertyName( propertyName );
tmpData->setData( whichWindow );
auxCheckBox->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED,
wxCommandEventHandler( AdvancedSaveConfiguration::OnCheckBoxLinkWindowClicked ),
tmpData,
this ); 

boxSizerLinkWindow->Add( auxCheckBox, 1, wxALIGN_CENTER_VERTICAL | wxALL, 2 );  
boxSizerLinks->Add( boxSizerLinkWindow );
}

void AdvancedSaveConfiguration::buildWindowsSetWidgets( const string& propertyName, wxBoxSizer *boxSizerLinks, bool checked )
{
TWindowsSet tmpWinSet;
THistogramsSet tmpHistoSet;

if( checked )
{
linkedManager.getLinks( propertyName, tmpWinSet );
linkedManager.getLinks( propertyName, tmpHistoSet );
}
else
{
unlinkedManager.getLinks( propertyName, tmpWinSet );
unlinkedManager.getLinks( propertyName, tmpHistoSet );
}

for( TWindowsSet::iterator itWin = tmpWinSet.begin(); itWin != tmpWinSet.end(); ++itWin )
{
buildLinkWindowWidget( boxSizerLinks, propertyName, *itWin, checked );
}

for( THistogramsSet::iterator itHisto = tmpHistoSet.begin(); itHisto != tmpHistoSet.end(); ++itHisto )
{
buildLinkWindowWidget( boxSizerLinks, propertyName, *itHisto, checked );
}
}

void AdvancedSaveConfiguration::OnCheckBoxLinkPropertyClicked( wxCommandEvent& event )
{
string tmpOriginalName = ( ( OriginalNameData *)event.m_callbackUserData )->myOriginalName;

if( event.IsChecked() )
{
bool existsCustomName = linkedManager.getLinksSize( tmpOriginalName ) > 0;
string tmpCustomName;
if( !existsCustomName )
tmpCustomName = unlinkedManager.getCustomName( tmpOriginalName );
else
tmpCustomName = linkedManager.getCustomName( tmpOriginalName );

TWindowsSet tmpWinSet;
unlinkedManager.getLinks( tmpOriginalName, tmpWinSet );
for( TWindowsSet::iterator it = tmpWinSet.begin(); it != tmpWinSet.end(); ++it )
{
unlinkedManager.removeLink( tmpOriginalName, *it );
linkedManager.insertLink( tmpOriginalName, *it );
if ( (*it) == timelines[ currentItem ] )
GetTextCtrlByName( wxString::FromUTF8( tmpOriginalName.c_str() ) )->ChangeValue( wxString::FromUTF8( tmpCustomName.c_str() ) );
else
{
setTimelineCFG4DAlias( *it, tmpOriginalName, tmpCustomName );
}
}

THistogramsSet tmpHistoSet;
unlinkedManager.getLinks( tmpOriginalName, tmpHistoSet );
for( THistogramsSet::iterator it = tmpHistoSet.begin(); it != tmpHistoSet.end(); ++it )
{
unlinkedManager.removeLink( tmpOriginalName, *it );
linkedManager.insertLink( tmpOriginalName, *it ); 
if ( (*it) == histograms[ currentItem ] )
GetTextCtrlByName( wxString::FromUTF8( tmpOriginalName.c_str() ) )->ChangeValue( wxString::FromUTF8( tmpCustomName.c_str() ) );
else
(*it)->setCFG4DAlias( tmpOriginalName, tmpCustomName );
}

if( !existsCustomName )
linkedManager.setCustomName( tmpOriginalName, tmpCustomName );
}
else
{
string tmpCustomName = linkedManager.getCustomName( tmpOriginalName );

TWindowsSet tmpWinSet;
linkedManager.getLinks( tmpOriginalName, tmpWinSet );
for( TWindowsSet::iterator it = tmpWinSet.begin(); it != tmpWinSet.end(); ++it )
{
linkedManager.removeLink( tmpOriginalName, *it );
unlinkedManager.insertLink( tmpOriginalName, *it );
}

THistogramsSet tmpHistoSet;
linkedManager.getLinks( tmpOriginalName, tmpHistoSet );
for( THistogramsSet::iterator it = tmpHistoSet.begin(); it != tmpHistoSet.end(); ++it )
{
linkedManager.removeLink( tmpOriginalName, *it ); 
unlinkedManager.insertLink( tmpOriginalName, *it );
}

unlinkedManager.setCustomName( tmpOriginalName, tmpCustomName );
}

updateLinkPropertiesWidgets();
}


void AdvancedSaveConfiguration::updateAliasForLinkedWindows( std::string whichOriginalName, 
std::string whichCustomName )
{
TWindowsSet tmpWin;
linkedManager.getLinks( whichOriginalName, tmpWin );
for( TWindowsSet::iterator it = tmpWin.begin(); it != tmpWin.end(); ++it )
{
setTimelineCFG4DAlias( *it, whichOriginalName, whichCustomName );
}

THistogramsSet tmpHisto;
linkedManager.getLinks( whichOriginalName, tmpHisto );
for( THistogramsSet::iterator it = tmpHisto.begin(); it != tmpHisto.end(); ++it )
{
(*it)->setCFG4DAlias( whichOriginalName, whichCustomName );
}
}


void AdvancedSaveConfiguration::OnLinkedPropertiesNameChanged( wxCommandEvent &event )
{
string tmpOriginalName = ( ( OriginalNameData *)event.m_callbackUserData )->myOriginalName;
string tmpCustomName = string( event.GetString().mb_str() );

unlinkedManager.setCustomName( tmpOriginalName, tmpCustomName );
linkedManager.setCustomName( tmpOriginalName, tmpCustomName );

updateAliasForLinkedWindows( tmpOriginalName, tmpCustomName );

if( isTimeline )
{
TWindowsSet tmpWin;
linkedManager.getLinks( tmpOriginalName, tmpWin );
if ( tmpWin.find( timelines[ currentItem ] ) != tmpWin.end() )
GetTextCtrlByName( wxString::FromUTF8( tmpOriginalName.c_str() ) )->ChangeValue( wxString::FromUTF8( tmpCustomName.c_str() ) );
}
else
{
THistogramsSet tmpHisto;
linkedManager.getLinks( tmpOriginalName, tmpHisto );
if ( tmpHisto.find( histograms[ currentItem ] ) != tmpHisto.end() )
GetTextCtrlByName( wxString::FromUTF8( tmpOriginalName.c_str() ) )->ChangeValue( wxString::FromUTF8( tmpCustomName.c_str() ) );
}
}


void AdvancedSaveConfiguration::updateLinkPropertiesWidgets()
{
set<string> links;
unlinkedManager.getLinksName( links );
linkedManager.getLinksName( links );

scrolledLinkProperties->DestroyChildren();

wxBoxSizer *boxSizerLinks = new wxBoxSizer( wxVERTICAL );

for( set<string>::iterator it = links.begin(); it != links.end(); ++it )
{
wxBoxSizer *boxSizerOriginalName = new wxBoxSizer( wxHORIZONTAL );

wxString originalNameLabel = wxString::FromUTF8( (*it).c_str() );
OriginalNameData *tmpDataCheck = new OriginalNameData();
tmpDataCheck->myOriginalName = *it;
wxString fullOriginalNameLabel = originalNameLabel;
if ( originalNameLabel.AfterLast( KParamSeparator[0] ) != originalNameLabel )
originalNameLabel = originalNameLabel.AfterLast( KParamSeparator[0] );

wxCheckBox *auxCheckBox = new wxCheckBox( scrolledLinkProperties,
wxID_ANY,
originalNameLabel,
wxDefaultPosition,
wxDefaultSize,
0,
wxDefaultValidator,
fullOriginalNameLabel + KCheckBoxSuffix );
auxCheckBox->SetValue( unlinkedManager.getLinksSize( *it ) == 0 );
auxCheckBox->SetToolTip( wxT( "Link/Unlink all windows" ) );
auxCheckBox->Connect( wxEVT_COMMAND_CHECKBOX_CLICKED,
wxCommandEventHandler( AdvancedSaveConfiguration::OnCheckBoxLinkPropertyClicked ),
tmpDataCheck,
this ); 

boxSizerOriginalName->Add( auxCheckBox, 1, wxEXPAND | wxALL, 2 );

wxString tmpCustomName;
if( linkedManager.getLinksSize( *it ) > 0 )
tmpCustomName = wxString::FromUTF8( linkedManager.getCustomName( *it ).c_str() );
else
tmpCustomName = wxString::FromUTF8( unlinkedManager.getCustomName( *it ).c_str() );

wxArrayString forbiddenChars;
forbiddenChars.Add( wxT("|") );
wxTextValidator excludeVerticalBar( wxFILTER_EXCLUDE_CHAR_LIST );
excludeVerticalBar.SetExcludes( forbiddenChars );

wxTextCtrl *customNameText = new wxTextCtrl( scrolledLinkProperties,
wxID_ANY,
tmpCustomName,
wxDefaultPosition,
wxDefaultSize,
0,
excludeVerticalBar );
customNameText->SetToolTip( wxT( "Custom name for linked property" ) );
OriginalNameData *tmpDataText = new OriginalNameData();
tmpDataText->myOriginalName = *it;
customNameText->Connect( wxEVT_COMMAND_TEXT_UPDATED,
wxCommandEventHandler( AdvancedSaveConfiguration::OnLinkedPropertiesNameChanged ),
tmpDataText,
this ); 

boxSizerOriginalName->Add( customNameText, 2, wxEXPAND | wxALL, 2 );

boxSizerLinks->Add( boxSizerOriginalName, 0, wxEXPAND );

buildWindowsSetWidgets( *it, boxSizerLinks, false );
buildWindowsSetWidgets( *it, boxSizerLinks, true );

boxSizerLinks->Add( new wxStaticLine( scrolledLinkProperties ), 0, wxEXPAND|wxALL, 3 );
}

scrolledLinkProperties->SetSizer( boxSizerLinks );
scrolledLinkProperties->FitInside();
}


void AdvancedSaveConfiguration::setTimelineCFG4DAlias( Timeline *whichWindow,
const string& whichOriginalName,
const string& whichCustomName )
{
if ( whichOriginalName.find( PARAM_SEPARATOR ) != string::npos )
{
string semanticLevel;
string function;
TParamIndex numParameter;

parseSemanticParameterTag( whichOriginalName, semanticLevel, function, numParameter );
whichWindow->setCFG4DParamAlias( semanticLevel, function, numParameter, whichCustomName );
}
else
whichWindow->setCFG4DAlias( whichOriginalName, whichCustomName );
}
