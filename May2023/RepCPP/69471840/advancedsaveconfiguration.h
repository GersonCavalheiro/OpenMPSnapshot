


#pragma once





#include "wx/tglbtn.h"

#include <vector>
#include <map>
#include <set>

#include "paraverkerneltypes.h"
#include "window.h"
#include "histogram.h"
#include "cfgs4d.h"

using std::multimap;



class wxToggleButton;



#define ID_ADVANCEDSAVECONFIGURATION 10186
#define ID_CHOICE_WINDOW 10185
#define ID_SCROLLEDWINDOW1 10187
#define ID_SCROLLED_LINK_PROPERTIES 10000
#define ID_TOGGLEBUTTON_LIST_SELECTED 10197
#define SYMBOL_ADVANCEDSAVECONFIGURATION_STYLE wxCAPTION|wxRESIZE_BORDER|wxSYSTEM_MENU|wxCLOSE_BOX|wxTAB_TRAVERSAL
#define SYMBOL_ADVANCEDSAVECONFIGURATION_TITLE _("Save Basic CFG - Properties Editor")
#define SYMBOL_ADVANCEDSAVECONFIGURATION_IDNAME ID_ADVANCEDSAVECONFIGURATION
#define SYMBOL_ADVANCEDSAVECONFIGURATION_SIZE wxDefaultSize
#define SYMBOL_ADVANCEDSAVECONFIGURATION_POSITION wxDefaultPosition




enum class TEditorMode
{
PROPERTIES_TAGS,
HISTOGRAM_STATISTIC_TAGS
};

class AdvancedSaveConfiguration: public wxDialog
{    
DECLARE_DYNAMIC_CLASS( AdvancedSaveConfiguration )
DECLARE_EVENT_TABLE()

public:

AdvancedSaveConfiguration();
AdvancedSaveConfiguration( wxWindow* parent,
wxWindowID id = SYMBOL_ADVANCEDSAVECONFIGURATION_IDNAME,
const wxString& caption = SYMBOL_ADVANCEDSAVECONFIGURATION_TITLE,
const wxPoint& pos = SYMBOL_ADVANCEDSAVECONFIGURATION_POSITION,
const wxSize& size = SYMBOL_ADVANCEDSAVECONFIGURATION_SIZE,
long style = SYMBOL_ADVANCEDSAVECONFIGURATION_STYLE );
AdvancedSaveConfiguration( wxWindow* parent,
const std::vector< Timeline * > &whichTimelines,
const std::vector< Histogram * > &whichHistograms,
TEditorMode mode = TEditorMode::PROPERTIES_TAGS,
wxWindowID id = SYMBOL_ADVANCEDSAVECONFIGURATION_IDNAME,
const wxString& caption = SYMBOL_ADVANCEDSAVECONFIGURATION_TITLE,
const wxPoint& pos = SYMBOL_ADVANCEDSAVECONFIGURATION_POSITION,
const wxSize& size = SYMBOL_ADVANCEDSAVECONFIGURATION_SIZE,
long style = SYMBOL_ADVANCEDSAVECONFIGURATION_STYLE );

bool Create( wxWindow* parent, wxWindowID id = SYMBOL_ADVANCEDSAVECONFIGURATION_IDNAME, const wxString& caption = SYMBOL_ADVANCEDSAVECONFIGURATION_TITLE, const wxPoint& pos = SYMBOL_ADVANCEDSAVECONFIGURATION_POSITION, const wxSize& size = SYMBOL_ADVANCEDSAVECONFIGURATION_SIZE, long style = SYMBOL_ADVANCEDSAVECONFIGURATION_STYLE );

~AdvancedSaveConfiguration();

void Init();

void CreateControls();


void OnChoiceWindowSelected( wxCommandEvent& event );

void OnToggleOnlySelectedClick( wxCommandEvent& event );

void OnCancelClick( wxCommandEvent& event );

void OnSaveClick( wxCommandEvent& event );



wxBitmap GetBitmapResource( const wxString& name );

wxIcon GetIconResource( const wxString& name );

const CFGS4DLinkedPropertiesManager& getLinkedPropertiesManager() const;

static bool ShowToolTips();

wxChoice* choiceWindow;
wxScrolledWindow* scrolledWindow;
wxScrolledWindow* scrolledLinkProperties;
wxToggleButton* toggleOnlySelected;
wxButton* buttonSave;


protected:
const static wxString KParamSeparator;
const static wxString KPreffixSeparator;
const static wxString KSuffixSeparator;
const static wxString KCheckBoxSuffix;
const static wxString KTextCtrlSuffix;
const static wxString KButtonSuffix;

private:
bool isTimeline;
int currentItem;                   
std::vector< Timeline * > timelines;
std::vector< Histogram * > histograms;

std::map< Timeline *, bool > backupTimelinesCFG4DEnabled;
std::map< Timeline *, bool > backupTimelinesCFG4DMode;
std::map< Timeline *, std::map< std::string, std::string > > backupTimelinesCFG4DAliasList;
std::map< Timeline *, Timeline::TParamAlias > backupTimelinesCFG4DParamAlias;
std::map< Histogram *, bool > backupHistogramsCFG4DEnabled;
std::map< Histogram *, bool > backupHistogramsCFG4DMode;
std::map< Histogram *, std::map< std::string, std::string > > backupHistogramsCFG4DAliasList;
std::map< Histogram *, std::map< std::string, std::string > > backupHistogramsCFG4DStatisticsAliasList;

std::vector< std::string > fullTagList;
std::map< std::string, std::string > renamedTag;
std::map< std::string, bool > enabledTag;
bool enabledCFG4DMode;

TEditorMode editionMode;

CFGS4DLinkedPropertiesManager unlinkedManager;
CFGS4DLinkedPropertiesManager linkedManager;

int GetSelectionIndexCorrected( int index, bool &isTimeline );

void initLinks( Timeline *whichTimeline );
void initLinks( Histogram *whichHistogram );

wxString BuildName( Timeline *current );
wxString BuildName( Histogram *current );

bool allowedLevel( const std::string &tag );

void BuildTagMaps( const std::map< std::string, std::string > &renamedTagMap,
const bool showFullList );
void parseSemanticParameterTag( const wxString& whichTag,
std::string& onSemanticLevel,
std::string& onFunction,
TParamIndex& onNumParameter );
void InsertParametersToTagMaps( const std::vector< Timeline::TParamAliasKey > &fullParamList,
const Timeline::TParamAlias &renamedParamAlias,
const bool showFullList );
wxBoxSizer *BuildTagRowWidgets( std::map< std::string, std::string >::iterator it,
bool showFullList );
void BuildTagWidgets( const bool showFullList );
void BuildTagsPanel( Timeline *currentWindow, const bool showFullList );
void BuildTagsPanel( Histogram *currentHistogram, const bool showFullList );

void PreparePanel( bool showFullList );
void TransferDataFromPanel( bool showFullList );

void DisconnectWidgetsTagsPanel( bool showFullList );
void CleanTagsPanel( bool showFullList );

wxCheckBox *GetCheckBoxByName( const wxString& widgetName ) const;
wxTextCtrl *GetTextCtrlByName( const wxString& widgetName ) const;
wxButton *GetButtonByName( const wxString& widgetName ) const;

template< class T >
void insertLinkInUnlinkedManager( const std::string& originalName, const std::string& newCustomName, T *whichWindow );

void OnCheckBoxPropertyClicked( wxCommandEvent& event );
void OnCheckBoxLinkWindowClicked( wxCommandEvent& event );
void OnStatisticsButtonClick( wxCommandEvent& event );
void OnTextCtrlPropertyChanged( wxCommandEvent &event );

void RefreshList( bool showFullList );

void buildWindowsSetWidgets( const std::string& propertyName, wxBoxSizer *boxSizerLinks, bool checked );
template <typename WindowType>
void buildLinkWindowWidget( wxBoxSizer *boxSizerLinks, 
const std::string& propertyName,
WindowType *whichWindow,
bool checked );
void updateLinkPropertiesWidgets();
void updateAliasForLinkedWindows( std::string whichOriginalName, std::string whichCustomName );

void OnCheckBoxLinkPropertyClicked( wxCommandEvent& event );
void OnLinkedPropertiesNameChanged( wxCommandEvent& event );

void setTimelineCFG4DAlias( Timeline *whichWindow,
const std::string& whichOriginalName,
const std::string& whichCustomName );

};
