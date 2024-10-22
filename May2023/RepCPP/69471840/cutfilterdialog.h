

#pragma once





#include "include/filebrowserbutton.h"
#include "wx/notebook.h"
#include "wx/statline.h"
#include "wx/spinctrl.h"

#include <string>
#include <vector>



class FileBrowserButton;
class wxBoxSizer;
class wxNotebook;
class wxSpinCtrl;

#include "traceoptions.h"

#include "tracecutter.h"
#include "tracefilter.h"
#include "tracesoftwarecounters.h"
#include "wxparaverapp.h"



#define ID_CUTFILTERDIALOG 10103
#define ID_TEXTCTRL_CUT_FILTER_INPUT_TRACE 10247
#define ID_BUTTON_FILE_BROWSER_INPUT_TRACE 10246
#define ID_TEXTCTRL_CUT_FILTER_OUTPUT_TRACE 10248
#define ID_BUTTON_FILE_BROWSER_OUTPUT_TRACE 10249
#define ID_CHECKBOX_LOAD_RESULTING_TRACE 10152
#define ID_CHECKBOX_RUN_APP_WITH_RESULTING_TRACE 10002
#define ID_TEXTCTRL_CUT_FILTER_XML 10250
#define ID_FILEBROWSERBUTTON_CUT_FILTER_XML 10251
#define ID_CHECKLISTBOX_EXECUTION_CHAIN 10107
#define ID_BITMAPBUTTON_PUSH_UP_FILTER 10109
#define ID_BITMAPBUTTON_PUSH_DOWN_FILTER 10001
#define ID_BUTTON_SAVE_XML 10154
#define ID_NOTEBOOK_CUT_FILTER_OPTIONS 10108
#define ID_PANEL_CUTTER 10111
#define ID_RADIOBUTTON_CUTTER_CUT_BY_TIME 10116
#define ID_RADIOBUTTON_CUTTER_CUT_BY_PERCENT 10117
#define ID_TEXTCTRL_CUTTER_BEGIN_CUT 10118
#define ID_TEXTCTRL_CUTTER_END_CUT 10119
#define ID_TEXTCTRL_CUTTER_TASKS 10157
#define ID_BUTTON_CUTTER_SELECT_REGION 10114
#define ID_BUTTON_CUTTER_ALL_WINDOW 10198
#define ID_BUTTON_CUTTER_ALL_TRACE 10115
#define ID_CHECKBOX_CHECK_CUTTER_ORIGINAL_TIME 10120
#define ID_CHECKBOX_CUTTER_REMOVE_FIRST_STATE 10121
#define ID_CHECKBOX_CUTTER_BREAK_STATES 10122
#define ID_CHECKBOX_CUTTER_REMOVE_LAST_STATE 10123
#define ID_CHECKBOX_CUTTER_KEEP_EVENTS 10216
#define ID_CHECKBOX_CUTTER_KEEP_EVENTS_WITHOUT_STATES 10000
#define ID_SPINCTRL_CUTTER_MAXIMUM_SIZE 10147
#define ID_PANEL_FILTER 10112
#define ID_CHECKBOX_FILTER_DISCARD_STATE 10124
#define ID_CHECKBOX_FILTER_DISCARD_EVENT 10125
#define ID_CHECKBOX_FILTER_DISCARD_COMMUNICATION 10126
#define ID_CHECKLISTBOX_FILTER_STATES 10128
#define ID_BUTTON_FILTER_SELECT_ALL 10129
#define ID_BUTTON_FILTER_UNSELECT_ALL 10130
#define ID_TEXTCTRL_FILTER_MIN_BURST_TIME 10151
#define ID_LISTBOX_FILTER_EVENTS 10141
#define ID_BUTTON_FILTER_ADD 10142
#define ID_BUTTON_FILTER_DELETE 10143
#define ID_CHECKBOX_FILTER_DISCARD_LISTED_EVENTS 10155
#define ID_SPINCTRL_FILTER_SIZE 10127
#define ID_PANEL_SOFTWARE_COUNTERS 10113
#define ID_RADIOBUTTON_SC_ON_INTERVALS 10131
#define ID_RADIOBUTTON_SC_ON_STATES 10132
#define ID_TEXTCTRL_SC_SAMPLING_INTERVAL 10133
#define ID_TEXTCTRL_SC_MINIMUM_BURST_TIME 10134
#define ID_CHECKLISTBOX_SC_SELECTED_EVENTS 10148
#define ID_BUTTON_SC_SELECTED_EVENTS_ADD 10149
#define ID_BUTTON_SC_SELECTED_EVENTS_DELETE 10150
#define ID_RADIOBUTTON_SC_COUNT_EVENTS 10135
#define ID_RADIOBUTTON8 10136
#define ID_CHECKBOX_SC_REMOVE_STATES 10137
#define ID_CHECKBOX_SC_SUMMARIZE_USEFUL 10138
#define ID_CHECKBOX_SC_GLOBAL_COUNTERS 10139
#define ID_CHECKBOX_SC_ONLY_IN_BURSTS_COUNTING 10140
#define ID_LISTBOX_SC_KEEP_EVENTS 10144
#define ID_BUTTON_SC_KEEP_EVENTS_ADD 10145
#define ID_BUTTON_SC_KEEP_EVENTS_DELETE 10146
#define SYMBOL_CUTFILTERDIALOG_STYLE wxCAPTION|wxRESIZE_BORDER|wxSYSTEM_MENU|wxCLOSE_BOX|wxTAB_TRAVERSAL
#define SYMBOL_CUTFILTERDIALOG_TITLE _("Cut & Filter")
#define SYMBOL_CUTFILTERDIALOG_IDNAME ID_CUTFILTERDIALOG
#define SYMBOL_CUTFILTERDIALOG_SIZE wxSize(400, 300)
#define SYMBOL_CUTFILTERDIALOG_POSITION wxDefaultPosition




class CutFilterDialog: public wxDialog
{    
DECLARE_DYNAMIC_CLASS( CutFilterDialog )
DECLARE_EVENT_TABLE()

public:
CutFilterDialog();
CutFilterDialog(
wxWindow* parent,
const wxString& whichXMLConfigurationFile = wxT( "" ),
wxWindowID id = SYMBOL_CUTFILTERDIALOG_IDNAME,
const wxString& caption = SYMBOL_CUTFILTERDIALOG_TITLE,
const wxPoint& pos = SYMBOL_CUTFILTERDIALOG_POSITION,
const wxSize& size = SYMBOL_CUTFILTERDIALOG_SIZE,
long style = SYMBOL_CUTFILTERDIALOG_STYLE );

bool Create( wxWindow* parent, wxWindowID id = SYMBOL_CUTFILTERDIALOG_IDNAME, const wxString& caption = SYMBOL_CUTFILTERDIALOG_TITLE, const wxPoint& pos = SYMBOL_CUTFILTERDIALOG_POSITION, const wxSize& size = SYMBOL_CUTFILTERDIALOG_SIZE, long style = SYMBOL_CUTFILTERDIALOG_STYLE );

~CutFilterDialog();

void Init();

void CreateControls();


void OnInitDialog( wxInitDialogEvent& event );

void OnIdle( wxIdleEvent& event );

void OnKeyDown( wxKeyEvent& event );

void OnTextctrlCutFilterInputTraceTextUpdated( wxCommandEvent& event );

void OnTextctrlCutFilterXmlTextUpdated( wxCommandEvent& event );

void OnChecklistboxExecutionChainDoubleClicked( wxCommandEvent& event );

void OnCheckListExecutionChainSelected( wxCommandEvent& event );

void OnChecklistboxExecutionChainToggled( wxCommandEvent& event );

void OnBitmapbuttonPushUpFilterClick( wxCommandEvent& event );

void OnBitmapbuttonPushDownFilterClick( wxCommandEvent& event );

void OnButtonSaveXmlClick( wxCommandEvent& event );

void OnButtonSaveXmlUpdate( wxUpdateUIEvent& event );

void OnNotebookCutFilterOptionsPageChanged( wxNotebookEvent& event );

void OnRadiobuttonCutterCutByTimeSelected( wxCommandEvent& event );

void OnRadiobuttonCutterCutByPercentSelected( wxCommandEvent& event );

void OnButtonCutterSelectRegionClick( wxCommandEvent& event );

void OnButtonCutterSelectRegionUpdate( wxUpdateUIEvent& event );

void OnButtonCutterAllWindowClick( wxCommandEvent& event );

void OnButtonCutterAllWindowUpdate( wxUpdateUIEvent& event );

void OnButtonCutterAllTraceClick( wxCommandEvent& event );

void OnCheckboxCheckCutterOriginalTimeUpdate( wxUpdateUIEvent& event );

void OnCheckboxCutterKeepEventsUpdate( wxUpdateUIEvent& event );

void OnCheckboxFilterDiscardStateUpdate( wxUpdateUIEvent& event );

void OnCheckboxFilterDiscardEventUpdate( wxUpdateUIEvent& event );

void OnCheckboxFilterDiscardCommunicationUpdate( wxUpdateUIEvent& event );

void OnButtonFilterSelectAllClick( wxCommandEvent& event );

void OnButtonFilterUnselectAllClick( wxCommandEvent& event );

void OnButtonFilterAddClick( wxCommandEvent& event );

void OnButtonFilterDeleteClick( wxCommandEvent& event );

void OnPanelSoftwareCountersUpdate( wxUpdateUIEvent& event );

void OnButtonScSelectedEventsAddClick( wxCommandEvent& event );

void OnButtonScSelectedEventsDeleteClick( wxCommandEvent& event );

void OnButtonScKeepEventsAddClick( wxCommandEvent& event );

void OnButtonScKeepEventsDeleteClick( wxCommandEvent& event );

void OnApplyClick( wxCommandEvent& event );

void OnApplyUpdate( wxUpdateUIEvent& event );



bool GetChangedXMLParameters() const { return changedXMLParameters ; }
void SetChangedXMLParameters(bool value) { changedXMLParameters = value ; }

std::vector< std::string > GetFilterToolOrder() const { return filterToolOrder ; }
void SetFilterToolOrder(std::vector< std::string > value) { filterToolOrder = value ; }

std::string GetGlobalXMLsPath() const { return globalXMLsPath ; }
void SetGlobalXMLsPath(std::string value) { globalXMLsPath = value ; }

bool GetLoadResultingTrace() const { return loadResultingTrace ; }
void SetLoadResultingTrace(bool value) { loadResultingTrace = value ; }

KernelConnection * GetLocalKernel() const { return localKernel ; }
void SetLocalKernel(KernelConnection * value) { localKernel = value ; }

std::string GetNameDestinyTrace() const { return nameDestinyTrace ; }
void SetNameDestinyTrace(std::string value) { nameDestinyTrace = value ; }

std::string GetNameSourceTrace() const { return nameSourceTrace ; }
void SetNameSourceTrace(std::string value) { nameSourceTrace = value ; }

bool GetNewXMLsPath() const { return newXMLsPath ; }
void SetNewXMLsPath(bool value) { newXMLsPath = value ; }

bool GetRunAppWithResultingTrace() const { return runAppWithResultingTrace ; }
void SetRunAppWithResultingTrace(bool value) { runAppWithResultingTrace = value ; }

TraceOptions * GetTraceOptions() const { return traceOptions ; }
void SetTraceOptions(TraceOptions * value) { traceOptions = value ; }

bool GetWaitingGlobalTiming() const { return waitingGlobalTiming ; }
void SetWaitingGlobalTiming(bool value) { waitingGlobalTiming = value ; }

wxBitmap GetBitmapResource( const wxString& name );

wxIcon GetIconResource( const wxString& name );

static bool ShowToolTips();

void UpdateExecutionChain();

TTime formatTime( const wxString whichTime );
wxString formatTime( double value );
double formatPercent(const wxString whichPercent );
wxString formatPercent( double value );

std::string GetTraceFileName();
std::vector< int > GetToolsOrder();
bool LoadResultingTrace();

void CheckCommonOptions( bool &previousWarning, bool showWarning = false );
void CheckCutterOptions( bool &previousWarning );
void CheckFilterOptions( bool &previousWarning );
void CheckSoftwareCountersOptions( bool &previousWarning );

bool CheckStringTasks( wxString taskStr );
bool CheckEventsLine( wxString eventsStr );
void GetEventsFromLine( wxString eventsStr,
TraceOptions::TFilterTypes &eventTypes,
int &lastType );
void SetEventLine( TraceOptions::TFilterTypes eventTypes, int current );
void SetEventsList( TraceOptions::TFilterTypes types, int lastType ); 
void GetEventsList( TraceOptions::TFilterTypes &types, int &lastType );
void CheckStatesList( size_t begin, bool value );
void CheckStatesList( TraceOptions::TStateNames statesList );


bool SetSoftwareCountersEventsListToString( std::string listEvents, wxListBox *selectedEvents );
char *GetSoftwareCountersEventsListToString( wxListBox *selectedEvents );

void TransferWindowToCommonData( bool previousWarning );
void TransferWindowToCutterData( bool previousWarning );
void TransferWindowToFilterData( bool previousWarning );
void TransferWindowToSoftwareCountersData( bool previousWarning );

void TransferCommonDataToWindow( std::vector< std::string > order );
void readTimes( bool byTime, TTime &whichBeginTime, TTime &whichEndTime );
void TransferCutterDataToWindow( TraceOptions *traceOptions );
void TransferFilterDataToWindow( TraceOptions *traceOptions );
void TransferSoftwareCountersDataToWindow( TraceOptions *traceOptions );
void TransferDataToWindow( std::vector< std::string > order, TraceOptions* traceOptions );

void TransferXMLDataToWindow( TraceOptions *traceOptions,

std::vector< std::string > &toolIDsOrder );

bool GetLoadedXMLPath( std::string &XML );
void EnableAllTabsFromToolsList();
void ChangePageSelectionFromTabsToToolsOrderList();
void SetXMLFile( const wxString& whichXMLFile, bool refresh = true );
void TransferTraceOptionsToWindow( TraceOptions *traceOptions, std::vector< std::string > &whichToolIDsOrder );

void setOutputName( bool enable,
bool saveGeneratedName,
const std::string& sourceTrace = std::string("") );

wxTextCtrl* textCtrlInputTrace;
FileBrowserButton* fileBrowserButtonInputTrace;
wxStaticText* txtOutputTrace;
wxTextCtrl* textCtrlOutputTrace;
FileBrowserButton* fileBrowserButtonOutputTrace;
wxCheckBox* checkLoadResultingTrace;
wxCheckBox* checkRunAppWithResultingTrace;
wxTextCtrl* textCtrlXML;
FileBrowserButton* fileBrowserButtonXML;
wxBoxSizer* boxSizerExecutionChain;
wxStaticText* txtExecutionChain;
wxCheckListBox* checkListExecutionChain;
wxBitmapButton* buttonUp;
wxBitmapButton* buttonDown;
wxButton* buttonSaveXml;
wxNotebook* notebookTools;
wxRadioButton* radioCutterCutByTime;
wxRadioButton* radioCutterCutByTimePercent;
wxTextCtrl* textCutterBeginCut;
wxTextCtrl* textCutterEndCut;
wxTextCtrl* textCutterTasks;
wxButton* buttonCutterSelectRegion;
wxButton* buttonCutterAllWindow;
wxButton* buttonCutterAllTrace;
wxCheckBox* checkCutterUseOriginalTime;
wxCheckBox* checkCutterRemoveFirstState;
wxCheckBox* checkCutterDontBreakStates;
wxCheckBox* checkCutterRemoveLastState;
wxCheckBox* checkCutterKeepEvents;
wxCheckBox* checkCutterKeepEventsWithoutStates;
wxSpinCtrl* textCutterMaximumTraceSize;
wxCheckBox* checkFilterDiscardStateRecords;
wxCheckBox* checkFilterDiscardEventRecords;
wxCheckBox* checkFilterDiscardCommunicationRecords;
wxStaticBox* staticBoxSizerFilterStates;
wxCheckListBox* checkListFilterStates;
wxButton* buttonFilterSelectAll;
wxButton* buttonFilterUnselectAll;
wxStaticText* labelFilterMinBurstTime;
wxTextCtrl* textFilterMinBurstTime;
wxStaticBox* staticBoxSizerFilterEvents;
wxListBox* listboxFilterEvents;
wxButton* buttonFilterAdd;
wxButton* buttonFilterDelete;
wxCheckBox* checkFilterDiscardListedEvents;
wxStaticBox* staticBoxSizerFilterCommunications;
wxStaticText* staticTextFilterSize;
wxSpinCtrl* textFilterSize;
wxStaticText* staticTextFilterSizeUnit;
wxRadioButton* radioSCOnIntervals;
wxRadioButton* radioSCOnStates;
wxStaticText* staticTextSCSamplingInterval;
wxTextCtrl* textSCSamplingInterval;
wxStaticText* staticTextSCMinimumBurstTime;
wxTextCtrl* textSCMinimumBurstTime;
wxListBox* listSCSelectedEvents;
wxButton* buttonSCSelectedEventsAdd;
wxButton* buttonSCSelectedEventsDelete;
wxRadioButton* radioSCCountEvents;
wxRadioButton* radioSCAccumulateValues;
wxCheckBox* checkSCRemoveStates;
wxCheckBox* checkSCSummarizeUseful;
wxCheckBox* checkSCGlobalCounters;
wxCheckBox* checkSCOnlyInBurstsCounting;
wxListBox* listSCKeepEvents;
wxButton* buttonSCKeepEventsAdd;
wxButton* buttonSCKeepEventsDelete;
wxButton* buttonApply;
private:
bool changedXMLParameters;
std::vector< std::string > filterToolOrder;
std::string globalXMLsPath;
bool loadResultingTrace;
KernelConnection * localKernel;
std::string nameDestinyTrace;
std::string nameSourceTrace;
bool newXMLsPath;
bool runAppWithResultingTrace;
TraceOptions * traceOptions;
bool waitingGlobalTiming;

wxString xmlConfigurationFile;

std::vector< std::string > listToolOrder; 
std::map< std::string, int > TABINDEX;   
std::string outputPath;

wxString reAnySpaces;
wxString reSomeNumbers;
wxString reType;
wxString reNegativeSign;

wxString reSingleType;
wxString reRangeOfTypes;

wxString reIntegerValue;
wxString reSomeIntegersSepByComma;
wxString reValuesSepByComma;

wxString reValuesSepByCommaForType;

bool cutterByTimePreviouslyChecked;


bool isFileSelected( FileBrowserButton *fpc );
bool isFileSelected( const std::string& fpc );

bool isExecutionChainEmpty();
const std::vector< std::string > changeToolsNameToID( const std::vector< std::string >& listToolWithNames );
const std::vector< std::string > changeToolsIDsToNames( const std::vector< std::string >& listToolIDs );
bool globalEnable();
bool globalEnable( const std::string& auxInputTrace );

void swapTimeAndPercent();

void UpdateOutputTraceName();

void enableOutputTraceWidgets( bool enable );

void EnableSingleTab( int selected );

void ChangePageSelectionFromToolsOrderListToTabs( int selected );

void TransferToolOrderToCommonData();
void EnableToolTab( int i );

void UpdateGuiXMLSectionFromFile( TraceOptions *traceOptions, 
std::vector< std::string > &toolIDsOrder );
void UpdateGlobalXMLPath( const wxString& whichPath );

void TransferXMLFileToWindow(  const wxString& whichXMLFile );

Trace *getTrace();
};
