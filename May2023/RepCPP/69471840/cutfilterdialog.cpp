

#include "wx/wxprec.h"

#ifdef __BORLANDC__
#pragma hdrstop
#endif

#ifndef WX_PRECOMP
#include "wx/wx.h"
#endif

#include "wx/imaglist.h"

#include <cmath>
#include <string>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <wx/filedlg.h>
#include <wx/regex.h>
#include <wx/filename.h>

#include "cutfilterdialog.h"
#include "paraverconfig.h"
#include "kernelconnection.h"
#include "loadedwindows.h"
#include "runscript.h"
#include "histogram.h"
#include "filedialogext.h"
#include "gtimeline.h"
#include "labelconstructor.h"

#include "../icons/arrow_up.xpm"
#include "../icons/arrow_down.xpm"

using namespace std;



IMPLEMENT_DYNAMIC_CLASS( CutFilterDialog, wxDialog )




BEGIN_EVENT_TABLE( CutFilterDialog, wxDialog )

EVT_INIT_DIALOG( CutFilterDialog::OnInitDialog )
EVT_IDLE( CutFilterDialog::OnIdle )
EVT_KEY_DOWN( CutFilterDialog::OnKeyDown )
EVT_TEXT( ID_TEXTCTRL_CUT_FILTER_INPUT_TRACE, CutFilterDialog::OnTextctrlCutFilterInputTraceTextUpdated )
EVT_TEXT( ID_TEXTCTRL_CUT_FILTER_XML, CutFilterDialog::OnTextctrlCutFilterXmlTextUpdated )
EVT_LISTBOX_DCLICK( ID_CHECKLISTBOX_EXECUTION_CHAIN, CutFilterDialog::OnChecklistboxExecutionChainDoubleClicked )
EVT_LISTBOX( ID_CHECKLISTBOX_EXECUTION_CHAIN, CutFilterDialog::OnCheckListExecutionChainSelected )
EVT_CHECKLISTBOX( ID_CHECKLISTBOX_EXECUTION_CHAIN, CutFilterDialog::OnChecklistboxExecutionChainToggled )
EVT_BUTTON( ID_BITMAPBUTTON_PUSH_UP_FILTER, CutFilterDialog::OnBitmapbuttonPushUpFilterClick )
EVT_BUTTON( ID_BITMAPBUTTON_PUSH_DOWN_FILTER, CutFilterDialog::OnBitmapbuttonPushDownFilterClick )
EVT_BUTTON( ID_BUTTON_SAVE_XML, CutFilterDialog::OnButtonSaveXmlClick )
EVT_UPDATE_UI( ID_BUTTON_SAVE_XML, CutFilterDialog::OnButtonSaveXmlUpdate )
EVT_NOTEBOOK_PAGE_CHANGED( ID_NOTEBOOK_CUT_FILTER_OPTIONS, CutFilterDialog::OnNotebookCutFilterOptionsPageChanged )
EVT_RADIOBUTTON( ID_RADIOBUTTON_CUTTER_CUT_BY_TIME, CutFilterDialog::OnRadiobuttonCutterCutByTimeSelected )
EVT_RADIOBUTTON( ID_RADIOBUTTON_CUTTER_CUT_BY_PERCENT, CutFilterDialog::OnRadiobuttonCutterCutByPercentSelected )
EVT_BUTTON( ID_BUTTON_CUTTER_SELECT_REGION, CutFilterDialog::OnButtonCutterSelectRegionClick )
EVT_UPDATE_UI( ID_BUTTON_CUTTER_SELECT_REGION, CutFilterDialog::OnButtonCutterSelectRegionUpdate )
EVT_BUTTON( ID_BUTTON_CUTTER_ALL_WINDOW, CutFilterDialog::OnButtonCutterAllWindowClick )
EVT_UPDATE_UI( ID_BUTTON_CUTTER_ALL_WINDOW, CutFilterDialog::OnButtonCutterAllWindowUpdate )
EVT_BUTTON( ID_BUTTON_CUTTER_ALL_TRACE, CutFilterDialog::OnButtonCutterAllTraceClick )
EVT_UPDATE_UI( ID_CHECKBOX_CHECK_CUTTER_ORIGINAL_TIME, CutFilterDialog::OnCheckboxCheckCutterOriginalTimeUpdate )
EVT_UPDATE_UI( ID_CHECKBOX_CUTTER_KEEP_EVENTS, CutFilterDialog::OnCheckboxCutterKeepEventsUpdate )
EVT_UPDATE_UI( ID_CHECKBOX_FILTER_DISCARD_STATE, CutFilterDialog::OnCheckboxFilterDiscardStateUpdate )
EVT_UPDATE_UI( ID_CHECKBOX_FILTER_DISCARD_EVENT, CutFilterDialog::OnCheckboxFilterDiscardEventUpdate )
EVT_UPDATE_UI( ID_CHECKBOX_FILTER_DISCARD_COMMUNICATION, CutFilterDialog::OnCheckboxFilterDiscardCommunicationUpdate )
EVT_BUTTON( ID_BUTTON_FILTER_SELECT_ALL, CutFilterDialog::OnButtonFilterSelectAllClick )
EVT_BUTTON( ID_BUTTON_FILTER_UNSELECT_ALL, CutFilterDialog::OnButtonFilterUnselectAllClick )
EVT_BUTTON( ID_BUTTON_FILTER_ADD, CutFilterDialog::OnButtonFilterAddClick )
EVT_BUTTON( ID_BUTTON_FILTER_DELETE, CutFilterDialog::OnButtonFilterDeleteClick )
EVT_UPDATE_UI( ID_PANEL_SOFTWARE_COUNTERS, CutFilterDialog::OnPanelSoftwareCountersUpdate )
EVT_BUTTON( ID_BUTTON_SC_SELECTED_EVENTS_ADD, CutFilterDialog::OnButtonScSelectedEventsAddClick )
EVT_BUTTON( ID_BUTTON_SC_SELECTED_EVENTS_DELETE, CutFilterDialog::OnButtonScSelectedEventsDeleteClick )
EVT_BUTTON( ID_BUTTON_SC_KEEP_EVENTS_ADD, CutFilterDialog::OnButtonScKeepEventsAddClick )
EVT_BUTTON( ID_BUTTON_SC_KEEP_EVENTS_DELETE, CutFilterDialog::OnButtonScKeepEventsDeleteClick )
EVT_BUTTON( wxID_APPLY, CutFilterDialog::OnApplyClick )
EVT_UPDATE_UI( wxID_APPLY, CutFilterDialog::OnApplyUpdate )

END_EVENT_TABLE()




CutFilterDialog::CutFilterDialog()
{
Init();
}

CutFilterDialog::CutFilterDialog(
wxWindow* parent,
const wxString& whichXMLConfigurationFile, 
wxWindowID id,
const wxString& caption,
const wxPoint& pos,
const wxSize& size,
long style )
{
Init();
xmlConfigurationFile = whichXMLConfigurationFile;
Create(parent, id, caption, pos, size, style);
}




bool CutFilterDialog::Create( wxWindow* parent, wxWindowID id, const wxString& caption, const wxPoint& pos, const wxSize& size, long style )
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




CutFilterDialog::~CutFilterDialog()
{
delete traceOptions;
}




void CutFilterDialog::Init()
{
changedXMLParameters = false;
globalXMLsPath = "";
loadResultingTrace = true;
localKernel = nullptr;
nameSourceTrace = "";
newXMLsPath = false;
runAppWithResultingTrace = false;
waitingGlobalTiming = false;
textCtrlInputTrace = NULL;
fileBrowserButtonInputTrace = NULL;
txtOutputTrace = NULL;
textCtrlOutputTrace = NULL;
fileBrowserButtonOutputTrace = NULL;
checkLoadResultingTrace = NULL;
checkRunAppWithResultingTrace = NULL;
textCtrlXML = NULL;
fileBrowserButtonXML = NULL;
boxSizerExecutionChain = NULL;
txtExecutionChain = NULL;
checkListExecutionChain = NULL;
buttonUp = NULL;
buttonDown = NULL;
buttonSaveXml = NULL;
notebookTools = NULL;
radioCutterCutByTime = NULL;
radioCutterCutByTimePercent = NULL;
textCutterBeginCut = NULL;
textCutterEndCut = NULL;
textCutterTasks = NULL;
buttonCutterSelectRegion = NULL;
buttonCutterAllWindow = NULL;
buttonCutterAllTrace = NULL;
checkCutterUseOriginalTime = NULL;
checkCutterRemoveFirstState = NULL;
checkCutterDontBreakStates = NULL;
checkCutterRemoveLastState = NULL;
checkCutterKeepEvents = NULL;
checkCutterKeepEventsWithoutStates = NULL;
textCutterMaximumTraceSize = NULL;
checkFilterDiscardStateRecords = NULL;
checkFilterDiscardEventRecords = NULL;
checkFilterDiscardCommunicationRecords = NULL;
staticBoxSizerFilterStates = NULL;
checkListFilterStates = NULL;
buttonFilterSelectAll = NULL;
buttonFilterUnselectAll = NULL;
labelFilterMinBurstTime = NULL;
textFilterMinBurstTime = NULL;
staticBoxSizerFilterEvents = NULL;
listboxFilterEvents = NULL;
buttonFilterAdd = NULL;
buttonFilterDelete = NULL;
checkFilterDiscardListedEvents = NULL;
staticBoxSizerFilterCommunications = NULL;
staticTextFilterSize = NULL;
textFilterSize = NULL;
staticTextFilterSizeUnit = NULL;
radioSCOnIntervals = NULL;
radioSCOnStates = NULL;
staticTextSCSamplingInterval = NULL;
textSCSamplingInterval = NULL;
staticTextSCMinimumBurstTime = NULL;
textSCMinimumBurstTime = NULL;
listSCSelectedEvents = NULL;
buttonSCSelectedEventsAdd = NULL;
buttonSCSelectedEventsDelete = NULL;
radioSCCountEvents = NULL;
radioSCAccumulateValues = NULL;
checkSCRemoveStates = NULL;
checkSCSummarizeUseful = NULL;
checkSCGlobalCounters = NULL;
checkSCOnlyInBurstsCounting = NULL;
listSCKeepEvents = NULL;
buttonSCKeepEventsAdd = NULL;
buttonSCKeepEventsDelete = NULL;
buttonApply = NULL;
outputPath = "";
xmlConfigurationFile.Clear(); 

localKernel = paraverMain::myParaverMain->GetLocalKernel();
traceOptions = TraceOptions::create( GetLocalKernel() );

reAnySpaces =  wxString( wxT( "[[:space:]]*" ) );
reSomeNumbers =  wxString( wxT( "[[:digit:]]+" ) );
reType = reAnySpaces + reSomeNumbers + reAnySpaces;
reNegativeSign = wxString( wxT( "[-]?" ) );

reIntegerValue = reAnySpaces + reNegativeSign + reSomeNumbers + reAnySpaces;
reSomeIntegersSepByComma = wxString( wxT( "(" ) ) + reAnySpaces + wxString( wxT( "[,]" ) ) + reIntegerValue + wxString( wxT( ")*" ) ); 
reValuesSepByComma = reIntegerValue + reSomeIntegersSepByComma;

reSingleType = wxString( wxT( "^(" ) ) + reType + wxString( wxT( ")$" ) );
reRangeOfTypes = wxString( wxT( "^(" ) ) + reType + wxString( wxT( "[-]" ) ) + reType + wxString( wxT( ")$" ) );
reValuesSepByCommaForType = wxString( wxT( "^(" ) ) + reType + wxString( wxT( "[:]" ) ) + reValuesSepByComma + wxString( wxT( ")$" ) );

cutterByTimePreviouslyChecked = true;
}




void CutFilterDialog::CreateControls()
{    
CutFilterDialog* itemDialog1 = this;

wxBoxSizer* itemBoxSizer2 = new wxBoxSizer(wxVERTICAL);
itemDialog1->SetSizer(itemBoxSizer2);

wxStaticBox* itemStaticBoxSizer3Static = new wxStaticBox(itemDialog1, wxID_ANY, _(" Traces "));
wxStaticBoxSizer* itemStaticBoxSizer3 = new wxStaticBoxSizer(itemStaticBoxSizer3Static, wxVERTICAL);
itemBoxSizer2->Add(itemStaticBoxSizer3, 0, wxGROW|wxALL, 2);

wxBoxSizer* itemBoxSizer4 = new wxBoxSizer(wxVERTICAL);
itemStaticBoxSizer3->Add(itemBoxSizer4, 0, wxGROW|wxALL, 3);

wxBoxSizer* itemBoxSizer5 = new wxBoxSizer(wxHORIZONTAL);
itemBoxSizer4->Add(itemBoxSizer5, 0, wxGROW|wxALL, 2);

wxStaticText* itemStaticText6 = new wxStaticText( itemDialog1, wxID_STATIC, _("Input"), wxDefaultPosition, wxDefaultSize, 0 );
if (CutFilterDialog::ShowToolTips())
itemStaticText6->SetToolTip(_("Trace that will be used by the Cut/Filter toolkit."));
itemBoxSizer5->Add(itemStaticText6, 1, wxALIGN_CENTER_VERTICAL|wxALL, 2);

textCtrlInputTrace = new wxTextCtrl( itemDialog1, ID_TEXTCTRL_CUT_FILTER_INPUT_TRACE, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
if (CutFilterDialog::ShowToolTips())
textCtrlInputTrace->SetToolTip(_("Trace that will be used by the Cut/Filter toolkit."));
itemBoxSizer5->Add(textCtrlInputTrace, 3, wxALIGN_CENTER_VERTICAL|wxALL, 2);

fileBrowserButtonInputTrace = new FileBrowserButton( itemDialog1, ID_BUTTON_FILE_BROWSER_INPUT_TRACE, _("Browse"), wxDefaultPosition, wxDefaultSize, 0 );
if (CutFilterDialog::ShowToolTips())
fileBrowserButtonInputTrace->SetToolTip(_("Trace that will be used by the Cut/Filter toolkit."));
itemBoxSizer5->Add(fileBrowserButtonInputTrace, 1, wxALIGN_CENTER_VERTICAL|wxALL, 2);

wxBoxSizer* itemBoxSizer9 = new wxBoxSizer(wxVERTICAL);
itemBoxSizer4->Add(itemBoxSizer9, 0, wxGROW|wxALL, 0);

wxBoxSizer* itemBoxSizer10 = new wxBoxSizer(wxHORIZONTAL);
itemBoxSizer9->Add(itemBoxSizer10, 0, wxGROW|wxALL, 2);

txtOutputTrace = new wxStaticText( itemDialog1, wxID_STATIC, _("Output"), wxDefaultPosition, wxDefaultSize, 0 );
if (CutFilterDialog::ShowToolTips())
txtOutputTrace->SetToolTip(_("Trace that will be saved by the Cut/Filter toolkit."));
itemBoxSizer10->Add(txtOutputTrace, 1, wxALIGN_CENTER_VERTICAL|wxALL, 2);

textCtrlOutputTrace = new wxTextCtrl( itemDialog1, ID_TEXTCTRL_CUT_FILTER_OUTPUT_TRACE, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
if (CutFilterDialog::ShowToolTips())
textCtrlOutputTrace->SetToolTip(_("Trace generated by the Cut/Filter toolkit."));
itemBoxSizer10->Add(textCtrlOutputTrace, 3, wxALIGN_CENTER_VERTICAL|wxALL, 2);

fileBrowserButtonOutputTrace = new FileBrowserButton( itemDialog1, ID_BUTTON_FILE_BROWSER_OUTPUT_TRACE, _("Browse"), wxDefaultPosition, wxDefaultSize, 0 );
if (CutFilterDialog::ShowToolTips())
fileBrowserButtonOutputTrace->SetToolTip(_("Trace generated by the Cut/Filter toolkit."));
itemBoxSizer10->Add(fileBrowserButtonOutputTrace, 1, wxALIGN_CENTER_VERTICAL|wxALL, 2);

wxBoxSizer* itemBoxSizer14 = new wxBoxSizer(wxHORIZONTAL);
itemBoxSizer9->Add(itemBoxSizer14, 0, wxGROW, 2);

itemBoxSizer14->Add(115, 5, 1, wxALIGN_CENTER_VERTICAL, 2);

checkLoadResultingTrace = new wxCheckBox( itemDialog1, ID_CHECKBOX_LOAD_RESULTING_TRACE, _("Load the processed trace"), wxDefaultPosition, wxDefaultSize, 0 );
checkLoadResultingTrace->SetValue(true);
if (CutFilterDialog::ShowToolTips())
checkLoadResultingTrace->SetToolTip(_("After the selected tools are applied, the resulting trace is loaded."));
itemBoxSizer14->Add(checkLoadResultingTrace, 4, wxALIGN_CENTER_VERTICAL|wxLEFT|wxRIGHT, 2);

wxBoxSizer* itemBoxSizer17 = new wxBoxSizer(wxHORIZONTAL);
itemBoxSizer9->Add(itemBoxSizer17, 0, wxGROW, 2);

itemBoxSizer17->Add(100, 5, 1, wxALIGN_CENTER_VERTICAL, 2);

checkRunAppWithResultingTrace = new wxCheckBox( itemDialog1, ID_CHECKBOX_RUN_APP_WITH_RESULTING_TRACE, _("Run application with the processed trace"), wxDefaultPosition, wxDefaultSize, 0 );
checkRunAppWithResultingTrace->SetValue(false);
if (CutFilterDialog::ShowToolTips())
checkRunAppWithResultingTrace->SetToolTip(_("After the selected tools are applied, the resulting trace is passed to the external application selection window."));
itemBoxSizer17->Add(checkRunAppWithResultingTrace, 4, wxALIGN_CENTER_VERTICAL|wxLEFT|wxRIGHT, 2);

wxStaticBox* itemStaticBoxSizer20Static = new wxStaticBox(itemDialog1, wxID_ANY, _(" Cut/Filter Parameters "));
wxStaticBoxSizer* itemStaticBoxSizer20 = new wxStaticBoxSizer(itemStaticBoxSizer20Static, wxVERTICAL);
itemBoxSizer2->Add(itemStaticBoxSizer20, 1, wxGROW|wxALL, 2);

wxBoxSizer* itemBoxSizer21 = new wxBoxSizer(wxHORIZONTAL);
itemStaticBoxSizer20->Add(itemBoxSizer21, 0, wxGROW|wxALL, 2);

wxStaticText* itemStaticText22 = new wxStaticText( itemDialog1, wxID_STATIC, _("Configuration file"), wxDefaultPosition, wxDefaultSize, 0 );
if (CutFilterDialog::ShowToolTips())
itemStaticText22->SetToolTip(_("XML filter configuration that will be used by the Cut/Filter toolkit."));
itemBoxSizer21->Add(itemStaticText22, 1, wxALIGN_CENTER_VERTICAL|wxALL, 2);

textCtrlXML = new wxTextCtrl( itemDialog1, ID_TEXTCTRL_CUT_FILTER_XML, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
if (CutFilterDialog::ShowToolTips())
textCtrlXML->SetToolTip(_("Load new settings from an xml file. Not used tools won't be changed. "));
itemBoxSizer21->Add(textCtrlXML, 3, wxALIGN_CENTER_VERTICAL|wxALL, 2);

fileBrowserButtonXML = new FileBrowserButton( itemDialog1, ID_FILEBROWSERBUTTON_CUT_FILTER_XML, _("Browse"), wxDefaultPosition, wxDefaultSize, 0 );
if (CutFilterDialog::ShowToolTips())
fileBrowserButtonXML->SetToolTip(_("Load new settings from an xml file. Not used tools won't be changed. "));
itemBoxSizer21->Add(fileBrowserButtonXML, 1, wxALIGN_CENTER_VERTICAL|wxALL, 2);

boxSizerExecutionChain = new wxBoxSizer(wxHORIZONTAL);
itemStaticBoxSizer20->Add(boxSizerExecutionChain, 0, wxGROW|wxALL, 2);

boxSizerExecutionChain->Add(5, 5, 1, wxALIGN_CENTER_VERTICAL|wxALL, 5);

wxBoxSizer* itemBoxSizer27 = new wxBoxSizer(wxVERTICAL);
boxSizerExecutionChain->Add(itemBoxSizer27, 2, wxGROW|wxALL, 2);

txtExecutionChain = new wxStaticText( itemDialog1, wxID_STATIC, _("Execution chain"), wxDefaultPosition, wxDefaultSize, 0 );
itemBoxSizer27->Add(txtExecutionChain, 0, wxALIGN_LEFT|wxLEFT|wxRIGHT|wxBOTTOM, 2);

wxArrayString checkListExecutionChainStrings;
checkListExecutionChain = new wxCheckListBox( itemDialog1, ID_CHECKLISTBOX_EXECUTION_CHAIN, wxDefaultPosition, wxDefaultSize, checkListExecutionChainStrings, wxLB_SINGLE|wxLB_NEEDED_SB );
if (CutFilterDialog::ShowToolTips())
checkListExecutionChain->SetToolTip(_("Select the order of the Cut/Filter tools."));
itemBoxSizer27->Add(checkListExecutionChain, 1, wxGROW, 2);

wxBoxSizer* itemBoxSizer30 = new wxBoxSizer(wxVERTICAL);
boxSizerExecutionChain->Add(itemBoxSizer30, 0, wxALIGN_BOTTOM|wxALL, 2);

buttonUp = new wxBitmapButton( itemDialog1, ID_BITMAPBUTTON_PUSH_UP_FILTER, itemDialog1->GetBitmapResource(wxT("icons/arrow_up.xpm")), wxDefaultPosition, wxDefaultSize, wxBU_AUTODRAW );
if (CutFilterDialog::ShowToolTips())
buttonUp->SetToolTip(_("Select the order of the Cut/Filter tools."));
itemBoxSizer30->Add(buttonUp, 1, wxGROW|wxTOP, 2);

buttonDown = new wxBitmapButton( itemDialog1, ID_BITMAPBUTTON_PUSH_DOWN_FILTER, itemDialog1->GetBitmapResource(wxT("icons/arrow_down.xpm")), wxDefaultPosition, wxDefaultSize, wxBU_AUTODRAW );
if (CutFilterDialog::ShowToolTips())
buttonDown->SetToolTip(_("Select the order of the Cut/Filter tools."));
itemBoxSizer30->Add(buttonDown, 1, wxGROW|wxTOP, 2);

boxSizerExecutionChain->Add(5, 5, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5);

wxBoxSizer* itemBoxSizer34 = new wxBoxSizer(wxVERTICAL);
boxSizerExecutionChain->Add(itemBoxSizer34, 0, wxALIGN_CENTER_VERTICAL|wxALL, 2);

buttonSaveXml = new wxButton( itemDialog1, ID_BUTTON_SAVE_XML, _("Save..."), wxDefaultPosition, wxDefaultSize, 0 );
if (CutFilterDialog::ShowToolTips())
buttonSaveXml->SetToolTip(_("Save current settings to an XML file."));
itemBoxSizer34->Add(buttonSaveXml, 1, wxGROW|wxALL, 5);

boxSizerExecutionChain->Add(5, 5, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5);

notebookTools = new wxNotebook( itemDialog1, ID_NOTEBOOK_CUT_FILTER_OPTIONS, wxDefaultPosition, wxSize(-1, 500), wxBK_DEFAULT );

wxScrolledWindow* itemScrolledWindow38 = new wxScrolledWindow( notebookTools, ID_PANEL_CUTTER, wxDefaultPosition, wxDefaultSize, wxSUNKEN_BORDER|wxTAB_TRAVERSAL );
itemScrolledWindow38->SetScrollbars(1, 10, 0, 0);
wxBoxSizer* itemBoxSizer39 = new wxBoxSizer(wxVERTICAL);
itemScrolledWindow38->SetSizer(itemBoxSizer39);

wxStaticBox* itemStaticBoxSizer40Static = new wxStaticBox(itemScrolledWindow38, wxID_STATIC, _(" Trace Limits "));
wxStaticBoxSizer* itemStaticBoxSizer40 = new wxStaticBoxSizer(itemStaticBoxSizer40Static, wxVERTICAL);
itemBoxSizer39->Add(itemStaticBoxSizer40, 0, wxGROW|wxALL, 5);
wxBoxSizer* itemBoxSizer41 = new wxBoxSizer(wxHORIZONTAL);
itemStaticBoxSizer40->Add(itemBoxSizer41, 0, wxGROW|wxLEFT|wxTOP, 5);
wxBoxSizer* itemBoxSizer42 = new wxBoxSizer(wxVERTICAL);
itemBoxSizer41->Add(itemBoxSizer42, 1, wxGROW|wxLEFT|wxTOP, 2);
radioCutterCutByTime = new wxRadioButton( itemScrolledWindow38, ID_RADIOBUTTON_CUTTER_CUT_BY_TIME, _("Cut by time"), wxDefaultPosition, wxDefaultSize, wxRB_GROUP );
radioCutterCutByTime->SetValue(true);
if (CutFilterDialog::ShowToolTips())
radioCutterCutByTime->SetToolTip(_("Select this to cut [begin time, end time] region of the trace."));
itemBoxSizer42->Add(radioCutterCutByTime, 1, wxALIGN_LEFT|wxLEFT|wxTOP, 2);

radioCutterCutByTimePercent = new wxRadioButton( itemScrolledWindow38, ID_RADIOBUTTON_CUTTER_CUT_BY_PERCENT, _("Cut by time %"), wxDefaultPosition, wxDefaultSize, 0 );
radioCutterCutByTimePercent->SetValue(false);
if (CutFilterDialog::ShowToolTips())
radioCutterCutByTimePercent->SetToolTip(_("Select this to cut [begin % time, end % time] region of the trace."));
itemBoxSizer42->Add(radioCutterCutByTimePercent, 1, wxALIGN_LEFT|wxLEFT, 2);

wxBoxSizer* itemBoxSizer45 = new wxBoxSizer(wxVERTICAL);
itemBoxSizer41->Add(itemBoxSizer45, 2, wxGROW|wxRIGHT, 5);
wxBoxSizer* itemBoxSizer46 = new wxBoxSizer(wxHORIZONTAL);
itemBoxSizer45->Add(itemBoxSizer46, 0, wxGROW|wxLEFT|wxTOP, 2);
wxStaticText* itemStaticText47 = new wxStaticText( itemScrolledWindow38, wxID_STATIC, _("Begin"), wxDefaultPosition, wxDefaultSize, 0 );
if (CutFilterDialog::ShowToolTips())
itemStaticText47->SetToolTip(_("Initial timestamp or percent for the cut."));
itemBoxSizer46->Add(itemStaticText47, 1, wxALIGN_CENTER_VERTICAL|wxLEFT|wxTOP|wxBOTTOM, 2);

textCutterBeginCut = new wxTextCtrl( itemScrolledWindow38, ID_TEXTCTRL_CUTTER_BEGIN_CUT, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
if (CutFilterDialog::ShowToolTips())
textCutterBeginCut->SetToolTip(_("Initial timestamp or percent for the cut."));
itemBoxSizer46->Add(textCutterBeginCut, 3, wxGROW|wxLEFT|wxTOP|wxBOTTOM, 2);

wxBoxSizer* itemBoxSizer49 = new wxBoxSizer(wxHORIZONTAL);
itemBoxSizer45->Add(itemBoxSizer49, 0, wxGROW|wxLEFT|wxTOP, 5);
wxStaticText* itemStaticText50 = new wxStaticText( itemScrolledWindow38, wxID_STATIC, _("End"), wxDefaultPosition, wxDefaultSize, 0 );
if (CutFilterDialog::ShowToolTips())
itemStaticText50->SetToolTip(_("Final timestamp or percent for the cut."));
itemBoxSizer49->Add(itemStaticText50, 1, wxALIGN_CENTER_VERTICAL|wxLEFT|wxTOP|wxBOTTOM, 2);

textCutterEndCut = new wxTextCtrl( itemScrolledWindow38, ID_TEXTCTRL_CUTTER_END_CUT, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
if (CutFilterDialog::ShowToolTips())
textCutterEndCut->SetToolTip(_("Final timestamp or percent for the cut."));
itemBoxSizer49->Add(textCutterEndCut, 3, wxALIGN_CENTER_VERTICAL|wxLEFT|wxTOP|wxBOTTOM, 2);

wxBoxSizer* itemBoxSizer52 = new wxBoxSizer(wxHORIZONTAL);
itemStaticBoxSizer40->Add(itemBoxSizer52, 0, wxGROW|wxALL, 2);
wxStaticText* itemStaticText53 = new wxStaticText( itemScrolledWindow38, wxID_STATIC, _("Tasks"), wxDefaultPosition, wxDefaultSize, 0 );
itemBoxSizer52->Add(itemStaticText53, 0, wxALIGN_CENTER_VERTICAL|wxALL, 2);

textCutterTasks = new wxTextCtrl( itemScrolledWindow38, ID_TEXTCTRL_CUTTER_TASKS, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
if (CutFilterDialog::ShowToolTips())
textCutterTasks->SetToolTip(_("Keep only information of tasks specified, separated by commas. Ranges marked with \"-\" are allowed. I.e. \"1-32,33,64-128\". Leave it empty to select all the tasks."));
itemBoxSizer52->Add(textCutterTasks, 1, wxGROW|wxALL, 2);

wxStaticLine* itemStaticLine55 = new wxStaticLine( itemScrolledWindow38, wxID_STATIC, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
itemStaticBoxSizer40->Add(itemStaticLine55, 0, wxGROW|wxALL, 5);

wxBoxSizer* itemBoxSizer56 = new wxBoxSizer(wxHORIZONTAL);
itemStaticBoxSizer40->Add(itemBoxSizer56, 0, wxGROW|wxALL, 2);
buttonCutterSelectRegion = new wxButton( itemScrolledWindow38, ID_BUTTON_CUTTER_SELECT_REGION, _("Select Region..."), wxDefaultPosition, wxDefaultSize, 0 );
if (CutFilterDialog::ShowToolTips())
buttonCutterSelectRegion->SetToolTip(_("Fill times range directly clicking or dragging from timelines. You can click on different timelines."));
itemBoxSizer56->Add(buttonCutterSelectRegion, 1, wxALIGN_CENTER_VERTICAL|wxLEFT|wxTOP, 2);

buttonCutterAllWindow = new wxButton( itemScrolledWindow38, ID_BUTTON_CUTTER_ALL_WINDOW, _("Whole Window"), wxDefaultPosition, wxDefaultSize, 0 );
buttonCutterAllWindow->Enable(false);
itemBoxSizer56->Add(buttonCutterAllWindow, 1, wxALIGN_CENTER_VERTICAL|wxLEFT|wxTOP, 2);

buttonCutterAllTrace = new wxButton( itemScrolledWindow38, ID_BUTTON_CUTTER_ALL_TRACE, _("Whole Trace"), wxDefaultPosition, wxDefaultSize, 0 );
if (CutFilterDialog::ShowToolTips())
buttonCutterAllTrace->SetToolTip(_("Set range [0%, 100%]."));
itemBoxSizer56->Add(buttonCutterAllTrace, 1, wxALIGN_CENTER_VERTICAL|wxLEFT|wxTOP, 2);

wxStaticBox* itemStaticBoxSizer60Static = new wxStaticBox(itemScrolledWindow38, wxID_STATIC, _(" Trace Options "));
wxStaticBoxSizer* itemStaticBoxSizer60 = new wxStaticBoxSizer(itemStaticBoxSizer60Static, wxVERTICAL);
itemBoxSizer39->Add(itemStaticBoxSizer60, 0, wxGROW|wxALL, 5);
wxBoxSizer* itemBoxSizer61 = new wxBoxSizer(wxHORIZONTAL);
itemStaticBoxSizer60->Add(itemBoxSizer61, 0, wxGROW|wxLEFT|wxTOP, 2);
checkCutterUseOriginalTime = new wxCheckBox( itemScrolledWindow38, ID_CHECKBOX_CHECK_CUTTER_ORIGINAL_TIME, _("Use original time"), wxDefaultPosition, wxDefaultSize, 0 );
checkCutterUseOriginalTime->SetValue(false);
if (CutFilterDialog::ShowToolTips())
checkCutterUseOriginalTime->SetToolTip(_("If not set, after the cut the first timestamp will be set to 0, and the difference with the original time will be substracted to all the times. If set, original time is kept."));
itemBoxSizer61->Add(checkCutterUseOriginalTime, 1, wxALIGN_CENTER_VERTICAL|wxLEFT|wxTOP, 2);

checkCutterRemoveFirstState = new wxCheckBox( itemScrolledWindow38, ID_CHECKBOX_CUTTER_REMOVE_FIRST_STATE, _("Remove first state"), wxDefaultPosition, wxDefaultSize, 0 );
checkCutterRemoveFirstState->SetValue(false);
if (CutFilterDialog::ShowToolTips())
checkCutterRemoveFirstState->SetToolTip(_("If the begin limit is inside a burst, don't keep it."));
itemBoxSizer61->Add(checkCutterRemoveFirstState, 1, wxALIGN_CENTER_VERTICAL|wxLEFT|wxTOP, 2);

wxBoxSizer* itemBoxSizer64 = new wxBoxSizer(wxHORIZONTAL);
itemStaticBoxSizer60->Add(itemBoxSizer64, 0, wxGROW|wxLEFT|wxTOP, 2);
checkCutterDontBreakStates = new wxCheckBox( itemScrolledWindow38, ID_CHECKBOX_CUTTER_BREAK_STATES, _("Don't break states"), wxDefaultPosition, wxDefaultSize, 0 );
checkCutterDontBreakStates->SetValue(false);
if (CutFilterDialog::ShowToolTips())
checkCutterDontBreakStates->SetToolTip(_("If set, no matter the given limits, the states in the middle of the cut are maintained. If not set the limits will split them. This options is overriden if \"Use original time\" is set."));
itemBoxSizer64->Add(checkCutterDontBreakStates, 1, wxALIGN_CENTER_VERTICAL|wxLEFT|wxTOP, 2);

checkCutterRemoveLastState = new wxCheckBox( itemScrolledWindow38, ID_CHECKBOX_CUTTER_REMOVE_LAST_STATE, _("Remove last state"), wxDefaultPosition, wxDefaultSize, 0 );
checkCutterRemoveLastState->SetValue(false);
if (CutFilterDialog::ShowToolTips())
checkCutterRemoveLastState->SetToolTip(_("If the end limit is inside a burst, don't keep it."));
itemBoxSizer64->Add(checkCutterRemoveLastState, 1, wxALIGN_CENTER_VERTICAL|wxLEFT|wxTOP, 2);

wxBoxSizer* itemBoxSizer67 = new wxBoxSizer(wxHORIZONTAL);
itemStaticBoxSizer60->Add(itemBoxSizer67, 0, wxGROW|wxLEFT|wxTOP|wxBOTTOM, 2);
checkCutterKeepEvents = new wxCheckBox( itemScrolledWindow38, ID_CHECKBOX_CUTTER_KEEP_EVENTS, _("Keep boundary events"), wxDefaultPosition, wxDefaultSize, 0 );
checkCutterKeepEvents->SetValue(false);
itemBoxSizer67->Add(checkCutterKeepEvents, 1, wxALIGN_CENTER_VERTICAL|wxLEFT|wxTOP, 2);

checkCutterKeepEventsWithoutStates = new wxCheckBox( itemScrolledWindow38, ID_CHECKBOX_CUTTER_KEEP_EVENTS_WITHOUT_STATES, _("Keep events for threads without states"), wxDefaultPosition, wxDefaultSize, 0 );
checkCutterKeepEventsWithoutStates->SetValue(false);
itemBoxSizer67->Add(checkCutterKeepEventsWithoutStates, 1, wxALIGN_CENTER_VERTICAL|wxLEFT|wxTOP, 2);

wxStaticBox* itemStaticBoxSizer69Static = new wxStaticBox(itemScrolledWindow38, wxID_STATIC, _(" Output Trace "));
wxStaticBoxSizer* itemStaticBoxSizer69 = new wxStaticBoxSizer(itemStaticBoxSizer69Static, wxHORIZONTAL);
itemBoxSizer39->Add(itemStaticBoxSizer69, 0, wxGROW|wxALL, 5);
wxStaticText* itemStaticText70 = new wxStaticText( itemScrolledWindow38, wxID_STATIC, _("Maximum trace size"), wxDefaultPosition, wxDefaultSize, wxALIGN_CENTRE );
itemStaticBoxSizer69->Add(itemStaticText70, 1, wxGROW|wxALL, 2);

textCutterMaximumTraceSize = new wxSpinCtrl( itemScrolledWindow38, ID_SPINCTRL_CUTTER_MAXIMUM_SIZE, wxT("0"), wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 0, 1000000, 0 );
if (CutFilterDialog::ShowToolTips())
textCutterMaximumTraceSize->SetToolTip(_("Set upper limit for the size of the cutted trace  in MB. Once this limit is reached, no more records will be written to the resulting trace."));
itemStaticBoxSizer69->Add(textCutterMaximumTraceSize, 3, wxALIGN_CENTER_VERTICAL|wxALL, 2);

wxStaticText* itemStaticText72 = new wxStaticText( itemScrolledWindow38, wxID_STATIC, _("MB"), wxDefaultPosition, wxDefaultSize, 0 );
itemStaticBoxSizer69->Add(itemStaticText72, 0, wxALIGN_CENTER_VERTICAL|wxALL, 2);

itemScrolledWindow38->FitInside();
notebookTools->AddPage(itemScrolledWindow38, _("Cutter"));

wxScrolledWindow* itemScrolledWindow73 = new wxScrolledWindow( notebookTools, ID_PANEL_FILTER, wxDefaultPosition, wxDefaultSize, wxSUNKEN_BORDER|wxTAB_TRAVERSAL );
itemScrolledWindow73->SetScrollbars(1, 10, 0, 0);
wxBoxSizer* itemBoxSizer74 = new wxBoxSizer(wxVERTICAL);
itemScrolledWindow73->SetSizer(itemBoxSizer74);

wxStaticBox* itemStaticBoxSizer75Static = new wxStaticBox(itemScrolledWindow73, wxID_STATIC, _(" Discard Records "));
wxStaticBoxSizer* itemStaticBoxSizer75 = new wxStaticBoxSizer(itemStaticBoxSizer75Static, wxHORIZONTAL);
itemBoxSizer74->Add(itemStaticBoxSizer75, 0, wxGROW|wxALL, 3);
checkFilterDiscardStateRecords = new wxCheckBox( itemScrolledWindow73, ID_CHECKBOX_FILTER_DISCARD_STATE, _("State"), wxDefaultPosition, wxDefaultSize, 0 );
checkFilterDiscardStateRecords->SetValue(false);
if (CutFilterDialog::ShowToolTips())
checkFilterDiscardStateRecords->SetToolTip(_("Discard all the state records from the source trace."));
itemStaticBoxSizer75->Add(checkFilterDiscardStateRecords, 0, wxGROW|wxALL, 2);

checkFilterDiscardEventRecords = new wxCheckBox( itemScrolledWindow73, ID_CHECKBOX_FILTER_DISCARD_EVENT, _("Event"), wxDefaultPosition, wxDefaultSize, 0 );
checkFilterDiscardEventRecords->SetValue(false);
if (CutFilterDialog::ShowToolTips())
checkFilterDiscardEventRecords->SetToolTip(_("Discard all the event records from the source trace."));
itemStaticBoxSizer75->Add(checkFilterDiscardEventRecords, 0, wxGROW|wxALL, 2);

checkFilterDiscardCommunicationRecords = new wxCheckBox( itemScrolledWindow73, ID_CHECKBOX_FILTER_DISCARD_COMMUNICATION, _("Communication"), wxDefaultPosition, wxDefaultSize, 0 );
checkFilterDiscardCommunicationRecords->SetValue(false);
if (CutFilterDialog::ShowToolTips())
checkFilterDiscardCommunicationRecords->SetToolTip(_("Discard all the communication records from the source trace."));
itemStaticBoxSizer75->Add(checkFilterDiscardCommunicationRecords, 0, wxGROW|wxALL, 2);

staticBoxSizerFilterStates = new wxStaticBox(itemScrolledWindow73, wxID_STATIC, _("Keep states "));
wxStaticBoxSizer* itemStaticBoxSizer79 = new wxStaticBoxSizer(staticBoxSizerFilterStates, wxHORIZONTAL);
itemBoxSizer74->Add(itemStaticBoxSizer79, 1, wxGROW|wxALL, 3);
wxArrayString checkListFilterStatesStrings;
checkListFilterStatesStrings.Add(_("Idle"));
checkListFilterStatesStrings.Add(_("Running"));
checkListFilterStatesStrings.Add(_("Not created"));
checkListFilterStatesStrings.Add(_("Waiting a message"));
checkListFilterStatesStrings.Add(_("Blocking Send"));
checkListFilterStatesStrings.Add(_("Thd. Synchr."));
checkListFilterStatesStrings.Add(_("Test/Probe"));
checkListFilterStatesStrings.Add(_("Sched. and Fork/Join"));
checkListFilterStatesStrings.Add(_("Wait/WaitAll"));
checkListFilterStatesStrings.Add(_("Blocked"));
checkListFilterStatesStrings.Add(_("Inmediate Send"));
checkListFilterStatesStrings.Add(_("Inmediate Receive"));
checkListFilterStatesStrings.Add(_("I/O"));
checkListFilterStatesStrings.Add(_("Group Communication"));
checkListFilterStatesStrings.Add(_("Tracing Disabled"));
checkListFilterStatesStrings.Add(_("Others"));
checkListFilterStatesStrings.Add(_("Send Receive"));
checkListFilterStates = new wxCheckListBox( itemScrolledWindow73, ID_CHECKLISTBOX_FILTER_STATES, wxDefaultPosition, wxSize(-1, 100), checkListFilterStatesStrings, wxLB_SINGLE );
if (CutFilterDialog::ShowToolTips())
checkListFilterStates->SetToolTip(_("Check the states that you want to keep in the filtered trace."));
itemStaticBoxSizer79->Add(checkListFilterStates, 3, wxGROW|wxALL, 2);

wxBoxSizer* itemBoxSizer81 = new wxBoxSizer(wxVERTICAL);
itemStaticBoxSizer79->Add(itemBoxSizer81, 2, wxGROW|wxALL, 5);
buttonFilterSelectAll = new wxButton( itemScrolledWindow73, ID_BUTTON_FILTER_SELECT_ALL, _("Select all"), wxDefaultPosition, wxDefaultSize, 0 );
itemBoxSizer81->Add(buttonFilterSelectAll, 0, wxALIGN_CENTER_HORIZONTAL|wxTOP|wxBOTTOM, 5);

buttonFilterUnselectAll = new wxButton( itemScrolledWindow73, ID_BUTTON_FILTER_UNSELECT_ALL, _("Unselect all"), wxDefaultPosition, wxDefaultSize, 0 );
itemBoxSizer81->Add(buttonFilterUnselectAll, 0, wxALIGN_CENTER_HORIZONTAL|wxTOP|wxBOTTOM, 5);

wxBoxSizer* itemBoxSizer84 = new wxBoxSizer(wxHORIZONTAL);
itemBoxSizer81->Add(itemBoxSizer84, 1, wxALIGN_CENTER_HORIZONTAL|wxALL, 5);
labelFilterMinBurstTime = new wxStaticText( itemScrolledWindow73, wxID_STATIC, _("Min. burst time"), wxDefaultPosition, wxDefaultSize, 0 );
if (CutFilterDialog::ShowToolTips())
labelFilterMinBurstTime->SetToolTip(_("Specify the minimum burst time for the state records."));
itemBoxSizer84->Add(labelFilterMinBurstTime, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5);

textFilterMinBurstTime = new wxTextCtrl( itemScrolledWindow73, ID_TEXTCTRL_FILTER_MIN_BURST_TIME, _("0"), wxDefaultPosition, wxDefaultSize, 0 );
if (CutFilterDialog::ShowToolTips())
textFilterMinBurstTime->SetToolTip(_("Specify the minimum burst time for the state records."));
itemBoxSizer84->Add(textFilterMinBurstTime, 1, wxALIGN_CENTER_VERTICAL|wxALL, 2);

staticBoxSizerFilterEvents = new wxStaticBox(itemScrolledWindow73, wxID_STATIC, _(" Events "));
wxStaticBoxSizer* itemStaticBoxSizer87 = new wxStaticBoxSizer(staticBoxSizerFilterEvents, wxHORIZONTAL);
itemBoxSizer74->Add(itemStaticBoxSizer87, 1, wxGROW|wxALL, 3);
wxArrayString listboxFilterEventsStrings;
listboxFilterEvents = new wxListBox( itemScrolledWindow73, ID_LISTBOX_FILTER_EVENTS, wxDefaultPosition, wxDefaultSize, listboxFilterEventsStrings, wxLB_SINGLE );
if (CutFilterDialog::ShowToolTips())
listboxFilterEvents->SetToolTip(_("List of the allowed events."));
itemStaticBoxSizer87->Add(listboxFilterEvents, 3, wxGROW|wxALL, 2);

wxBoxSizer* itemBoxSizer89 = new wxBoxSizer(wxVERTICAL);
itemStaticBoxSizer87->Add(itemBoxSizer89, 2, wxGROW|wxALL, 5);
buttonFilterAdd = new wxButton( itemScrolledWindow73, ID_BUTTON_FILTER_ADD, _("Add"), wxDefaultPosition, wxDefaultSize, 0 );
itemBoxSizer89->Add(buttonFilterAdd, 0, wxALIGN_CENTER_HORIZONTAL|wxTOP|wxBOTTOM, 5);

buttonFilterDelete = new wxButton( itemScrolledWindow73, ID_BUTTON_FILTER_DELETE, _("Delete"), wxDefaultPosition, wxDefaultSize, 0 );
itemBoxSizer89->Add(buttonFilterDelete, 0, wxALIGN_CENTER_HORIZONTAL|wxTOP|wxBOTTOM, 5);

checkFilterDiscardListedEvents = new wxCheckBox( itemScrolledWindow73, ID_CHECKBOX_FILTER_DISCARD_LISTED_EVENTS, _("Discard"), wxDefaultPosition, wxDefaultSize, 0 );
checkFilterDiscardListedEvents->SetValue(false);
if (CutFilterDialog::ShowToolTips())
checkFilterDiscardListedEvents->SetToolTip(_("If set, all the listed events will be discarded instead of being kept."));
itemBoxSizer89->Add(checkFilterDiscardListedEvents, 1, wxALIGN_CENTER_HORIZONTAL|wxALL, 5);

staticBoxSizerFilterCommunications = new wxStaticBox(itemScrolledWindow73, wxID_STATIC, _("Keep communications "));
wxStaticBoxSizer* itemStaticBoxSizer93 = new wxStaticBoxSizer(staticBoxSizerFilterCommunications, wxHORIZONTAL);
itemBoxSizer74->Add(itemStaticBoxSizer93, 0, wxGROW|wxALL, 3);
staticTextFilterSize = new wxStaticText( itemScrolledWindow73, wxID_STATIC, _("Minimum size "), wxDefaultPosition, wxDefaultSize, 0 );
if (CutFilterDialog::ShowToolTips())
staticTextFilterSize->SetToolTip(_("Allow only communications with a minimum size."));
itemStaticBoxSizer93->Add(staticTextFilterSize, 1, wxALIGN_CENTER_VERTICAL|wxALL, 2);

textFilterSize = new wxSpinCtrl( itemScrolledWindow73, ID_SPINCTRL_FILTER_SIZE, wxT("0"), wxDefaultPosition, wxDefaultSize, wxSP_ARROW_KEYS, 0, 100000000, 0 );
if (CutFilterDialog::ShowToolTips())
textFilterSize->SetToolTip(_("Allow only communications with a minimum size."));
itemStaticBoxSizer93->Add(textFilterSize, 3, wxALIGN_CENTER_VERTICAL|wxALL, 2);

staticTextFilterSizeUnit = new wxStaticText( itemScrolledWindow73, wxID_STATIC, _("Bytes"), wxDefaultPosition, wxDefaultSize, 0 );
itemStaticBoxSizer93->Add(staticTextFilterSizeUnit, 0, wxALIGN_CENTER_VERTICAL|wxALL, 2);

itemScrolledWindow73->FitInside();
notebookTools->AddPage(itemScrolledWindow73, _("Filter"));

wxScrolledWindow* itemScrolledWindow97 = new wxScrolledWindow( notebookTools, ID_PANEL_SOFTWARE_COUNTERS, wxDefaultPosition, wxDefaultSize, wxSUNKEN_BORDER|wxTAB_TRAVERSAL );
itemScrolledWindow97->SetScrollbars(1, 10, 0, 0);
wxBoxSizer* itemBoxSizer98 = new wxBoxSizer(wxVERTICAL);
itemScrolledWindow97->SetSizer(itemBoxSizer98);

wxStaticBox* itemStaticBoxSizer99Static = new wxStaticBox(itemScrolledWindow97, wxID_STATIC, _(" Region "));
wxStaticBoxSizer* itemStaticBoxSizer99 = new wxStaticBoxSizer(itemStaticBoxSizer99Static, wxHORIZONTAL);
itemBoxSizer98->Add(itemStaticBoxSizer99, 0, wxGROW|wxALL, 3);
wxBoxSizer* itemBoxSizer100 = new wxBoxSizer(wxVERTICAL);
itemStaticBoxSizer99->Add(itemBoxSizer100, 0, wxGROW|wxALL, 0);
radioSCOnIntervals = new wxRadioButton( itemScrolledWindow97, ID_RADIOBUTTON_SC_ON_INTERVALS, _("On intervals"), wxDefaultPosition, wxDefaultSize, wxRB_GROUP );
radioSCOnIntervals->SetValue(true);
if (CutFilterDialog::ShowToolTips())
radioSCOnIntervals->SetToolTip(_("The software counters will be written periodically after every time interval"));
itemBoxSizer100->Add(radioSCOnIntervals, 1, wxGROW|wxALL, 2);

radioSCOnStates = new wxRadioButton( itemScrolledWindow97, ID_RADIOBUTTON_SC_ON_STATES, _("On states"), wxDefaultPosition, wxDefaultSize, 0 );
radioSCOnStates->SetValue(false);
if (CutFilterDialog::ShowToolTips())
radioSCOnStates->SetToolTip(_("The software counters will be written after every context switch of a running burst of at least the declared duration."));
itemBoxSizer100->Add(radioSCOnStates, 1, wxGROW|wxALL, 2);

wxStaticLine* itemStaticLine103 = new wxStaticLine( itemScrolledWindow97, wxID_STATIC, wxDefaultPosition, wxDefaultSize, wxLI_VERTICAL );
itemStaticBoxSizer99->Add(itemStaticLine103, 0, wxGROW|wxALL, 5);

wxBoxSizer* itemBoxSizer104 = new wxBoxSizer(wxVERTICAL);
itemStaticBoxSizer99->Add(itemBoxSizer104, 1, wxALIGN_CENTER_VERTICAL|wxALL, 0);
wxBoxSizer* itemBoxSizer105 = new wxBoxSizer(wxHORIZONTAL);
itemBoxSizer104->Add(itemBoxSizer105, 1, wxGROW|wxALL, 2);
staticTextSCSamplingInterval = new wxStaticText( itemScrolledWindow97, wxID_STATIC, _("Sampling Interval (ns)"), wxDefaultPosition, wxDefaultSize, 0 );
if (CutFilterDialog::ShowToolTips())
staticTextSCSamplingInterval->SetToolTip(_("The software counters will be written periodically after every time interval"));
itemBoxSizer105->Add(staticTextSCSamplingInterval, 1, wxALIGN_CENTER_VERTICAL|wxALL, 2);

textSCSamplingInterval = new wxTextCtrl( itemScrolledWindow97, ID_TEXTCTRL_SC_SAMPLING_INTERVAL, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
if (CutFilterDialog::ShowToolTips())
textSCSamplingInterval->SetToolTip(_("The software counters will be written periodically after every time interval."));
itemBoxSizer105->Add(textSCSamplingInterval, 2, wxALIGN_CENTER_VERTICAL|wxALL, 2);

wxBoxSizer* itemBoxSizer108 = new wxBoxSizer(wxHORIZONTAL);
itemBoxSizer104->Add(itemBoxSizer108, 1, wxGROW|wxALL, 2);
staticTextSCMinimumBurstTime = new wxStaticText( itemScrolledWindow97, wxID_STATIC, _("Min Burst Time (ns)"), wxDefaultPosition, wxDefaultSize, 0 );
if (CutFilterDialog::ShowToolTips())
staticTextSCMinimumBurstTime->SetToolTip(_("The software counters will be written after every context switch of a running burst of at least the declared duration."));
itemBoxSizer108->Add(staticTextSCMinimumBurstTime, 1, wxALIGN_CENTER_VERTICAL|wxALL, 2);

textSCMinimumBurstTime = new wxTextCtrl( itemScrolledWindow97, ID_TEXTCTRL_SC_MINIMUM_BURST_TIME, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
if (CutFilterDialog::ShowToolTips())
textSCMinimumBurstTime->SetToolTip(_("The software counters will be written after every context switch of a running burst of at least the declared duration."));
itemBoxSizer108->Add(textSCMinimumBurstTime, 2, wxALIGN_CENTER_VERTICAL|wxALL, 2);

wxStaticBox* itemStaticBoxSizer111Static = new wxStaticBox(itemScrolledWindow97, wxID_STATIC, _(" Selected events "));
wxStaticBoxSizer* itemStaticBoxSizer111 = new wxStaticBoxSizer(itemStaticBoxSizer111Static, wxHORIZONTAL);
itemBoxSizer98->Add(itemStaticBoxSizer111, 1, wxGROW|wxALL, 3);
wxBoxSizer* itemBoxSizer112 = new wxBoxSizer(wxHORIZONTAL);
itemStaticBoxSizer111->Add(itemBoxSizer112, 1, wxGROW|wxALL, 0);
wxArrayString listSCSelectedEventsStrings;
listSCSelectedEvents = new wxListBox( itemScrolledWindow97, ID_CHECKLISTBOX_SC_SELECTED_EVENTS, wxDefaultPosition, wxDefaultSize, listSCSelectedEventsStrings, wxLB_SINGLE );
if (CutFilterDialog::ShowToolTips())
listSCSelectedEvents->SetToolTip(_("The counters will express the number of calls for every type-value specified in this list."));
itemBoxSizer112->Add(listSCSelectedEvents, 3, wxGROW|wxALL, 2);

wxBoxSizer* itemBoxSizer114 = new wxBoxSizer(wxVERTICAL);
itemBoxSizer112->Add(itemBoxSizer114, 2, wxALIGN_CENTER_VERTICAL|wxALL, 5);
buttonSCSelectedEventsAdd = new wxButton( itemScrolledWindow97, ID_BUTTON_SC_SELECTED_EVENTS_ADD, _("Add"), wxDefaultPosition, wxDefaultSize, 0 );
itemBoxSizer114->Add(buttonSCSelectedEventsAdd, 0, wxALIGN_CENTER_HORIZONTAL|wxALL, 5);

buttonSCSelectedEventsDelete = new wxButton( itemScrolledWindow97, ID_BUTTON_SC_SELECTED_EVENTS_DELETE, _("Delete"), wxDefaultPosition, wxDefaultSize, 0 );
itemBoxSizer114->Add(buttonSCSelectedEventsDelete, 0, wxALIGN_CENTER_HORIZONTAL|wxALL, 5);

wxStaticBox* itemStaticBoxSizer117Static = new wxStaticBox(itemScrolledWindow97, wxID_STATIC, _(" Options "));
wxStaticBoxSizer* itemStaticBoxSizer117 = new wxStaticBoxSizer(itemStaticBoxSizer117Static, wxHORIZONTAL);
itemBoxSizer98->Add(itemStaticBoxSizer117, 0, wxGROW|wxALL, 3);
wxBoxSizer* itemBoxSizer118 = new wxBoxSizer(wxHORIZONTAL);
itemStaticBoxSizer117->Add(itemBoxSizer118, 1, wxGROW|wxALL, 0);
wxBoxSizer* itemBoxSizer119 = new wxBoxSizer(wxVERTICAL);
itemBoxSizer118->Add(itemBoxSizer119, 1, wxGROW|wxALL, 2);
radioSCCountEvents = new wxRadioButton( itemScrolledWindow97, ID_RADIOBUTTON_SC_COUNT_EVENTS, _("Count events"), wxDefaultPosition, wxDefaultSize, wxRB_GROUP );
radioSCCountEvents->SetValue(false);
if (CutFilterDialog::ShowToolTips())
radioSCCountEvents->SetToolTip(_("Count how many times the type-event pairs appear in the source trace."));
itemBoxSizer119->Add(radioSCCountEvents, 1, wxGROW|wxALL, 2);

radioSCAccumulateValues = new wxRadioButton( itemScrolledWindow97, ID_RADIOBUTTON8, _("Accumulate values"), wxDefaultPosition, wxDefaultSize, 0 );
radioSCAccumulateValues->SetValue(false);
if (CutFilterDialog::ShowToolTips())
radioSCAccumulateValues->SetToolTip(_("Add the values instead of counting how many times the type-event pairs appear in the source trace."));
itemBoxSizer119->Add(radioSCAccumulateValues, 1, wxGROW|wxALL, 2);

wxStaticLine* itemStaticLine122 = new wxStaticLine( itemScrolledWindow97, wxID_STATIC, wxDefaultPosition, wxDefaultSize, wxLI_VERTICAL );
itemBoxSizer118->Add(itemStaticLine122, 0, wxGROW|wxALL, 5);

wxGridSizer* itemGridSizer123 = new wxGridSizer(2, 2, 0, 0);
itemBoxSizer118->Add(itemGridSizer123, 2, wxGROW|wxALL, 0);
checkSCRemoveStates = new wxCheckBox( itemScrolledWindow97, ID_CHECKBOX_SC_REMOVE_STATES, _("Remove states"), wxDefaultPosition, wxDefaultSize, 0 );
checkSCRemoveStates->SetValue(false);
itemGridSizer123->Add(checkSCRemoveStates, 1, wxGROW|wxALIGN_CENTER_VERTICAL|wxALL, 2);

checkSCSummarizeUseful = new wxCheckBox( itemScrolledWindow97, ID_CHECKBOX_SC_SUMMARIZE_USEFUL, _("Summarize useful"), wxDefaultPosition, wxDefaultSize, 0 );
checkSCSummarizeUseful->SetValue(false);
itemGridSizer123->Add(checkSCSummarizeUseful, 1, wxGROW|wxALIGN_CENTER_VERTICAL|wxALL, 2);

checkSCGlobalCounters = new wxCheckBox( itemScrolledWindow97, ID_CHECKBOX_SC_GLOBAL_COUNTERS, _("Global counters"), wxDefaultPosition, wxDefaultSize, 0 );
checkSCGlobalCounters->SetValue(false);
itemGridSizer123->Add(checkSCGlobalCounters, 1, wxGROW|wxALIGN_CENTER_VERTICAL|wxALL, 2);

checkSCOnlyInBurstsCounting = new wxCheckBox( itemScrolledWindow97, ID_CHECKBOX_SC_ONLY_IN_BURSTS_COUNTING, _("Only in bursts counting"), wxDefaultPosition, wxDefaultSize, 0 );
checkSCOnlyInBurstsCounting->SetValue(false);
itemGridSizer123->Add(checkSCOnlyInBurstsCounting, 1, wxGROW|wxALIGN_CENTER_VERTICAL|wxALL, 2);

wxStaticBox* itemStaticBoxSizer128Static = new wxStaticBox(itemScrolledWindow97, wxID_STATIC, _(" Keep events "));
wxStaticBoxSizer* itemStaticBoxSizer128 = new wxStaticBoxSizer(itemStaticBoxSizer128Static, wxHORIZONTAL);
itemBoxSizer98->Add(itemStaticBoxSizer128, 1, wxGROW|wxALL, 3);
wxArrayString listSCKeepEventsStrings;
listSCKeepEvents = new wxListBox( itemScrolledWindow97, ID_LISTBOX_SC_KEEP_EVENTS, wxDefaultPosition, wxDefaultSize, listSCKeepEventsStrings, wxLB_SINGLE );
itemStaticBoxSizer128->Add(listSCKeepEvents, 2, wxGROW|wxALL, 2);

wxBoxSizer* itemBoxSizer130 = new wxBoxSizer(wxVERTICAL);
itemStaticBoxSizer128->Add(itemBoxSizer130, 1, wxALIGN_CENTER_VERTICAL|wxALL, 5);
buttonSCKeepEventsAdd = new wxButton( itemScrolledWindow97, ID_BUTTON_SC_KEEP_EVENTS_ADD, _("Add"), wxDefaultPosition, wxDefaultSize, 0 );
itemBoxSizer130->Add(buttonSCKeepEventsAdd, 0, wxALIGN_CENTER_HORIZONTAL|wxALL, 5);

buttonSCKeepEventsDelete = new wxButton( itemScrolledWindow97, ID_BUTTON_SC_KEEP_EVENTS_DELETE, _("Delete"), wxDefaultPosition, wxDefaultSize, 0 );
itemBoxSizer130->Add(buttonSCKeepEventsDelete, 0, wxALIGN_CENTER_HORIZONTAL|wxALL, 5);

itemScrolledWindow97->FitInside();
notebookTools->AddPage(itemScrolledWindow97, _("Software Counters"));

itemStaticBoxSizer20->Add(notebookTools, 1, wxGROW|wxALL, 2);

wxStdDialogButtonSizer* itemStdDialogButtonSizer133 = new wxStdDialogButtonSizer;

itemBoxSizer2->Add(itemStdDialogButtonSizer133, 0, wxALIGN_RIGHT|wxALL, 2);
wxButton* itemButton134 = new wxButton( itemDialog1, wxID_CANCEL, _("&Cancel"), wxDefaultPosition, wxDefaultSize, 0 );
itemStdDialogButtonSizer133->AddButton(itemButton134);

buttonApply = new wxButton( itemDialog1, wxID_APPLY, _("&Apply"), wxDefaultPosition, wxDefaultSize, 0 );
itemStdDialogButtonSizer133->AddButton(buttonApply);

itemStdDialogButtonSizer133->Realize();

buttonCutterSelectRegion->Connect(ID_BUTTON_CUTTER_SELECT_REGION, wxEVT_KEY_DOWN, wxKeyEventHandler(CutFilterDialog::OnKeyDown), NULL, this);



TABINDEX[ TraceCutter::getID() ] = 0;
TABINDEX[ TraceFilter::getID() ] = 1;
TABINDEX[ TraceSoftwareCounters::getID() ] = 2;

listToolOrder.push_back( TraceCutter::getName() );
listToolOrder.push_back( TraceFilter::getName() );
listToolOrder.push_back( TraceSoftwareCounters::getName() );

UpdateExecutionChain();
EnableAllTabsFromToolsList();


wxString tmpWildCard = wxT( "Paraver trace (*.prv;*.prv.gz;*.csv)|*.prv;*.prv.gz;*.csv|All files (*.*)|*.*" );
fileBrowserButtonInputTrace->SetDialogMessage( _( "Load Trace" ) );
fileBrowserButtonInputTrace->SetFileDialogWildcard( tmpWildCard );
fileBrowserButtonInputTrace->SetTextBox( textCtrlInputTrace );
fileBrowserButtonInputTrace->Enable();

fileBrowserButtonOutputTrace->SetDialogMessage( _( "Save Trace" ) );
fileBrowserButtonOutputTrace->SetFileDialogWildcard( tmpWildCard ); 
fileBrowserButtonOutputTrace->SetTextBox( textCtrlOutputTrace );
fileBrowserButtonOutputTrace->SetDialogStyle( wxFD_SAVE | wxFD_OVERWRITE_PROMPT | wxFD_CHANGE_DIR ); 
fileBrowserButtonOutputTrace->Enable();

tmpWildCard = wxT( "XML configuration files (*.xml)|*.xml|All files (*.*)|*" );
fileBrowserButtonXML->SetDialogMessage( wxT( "Load XML Cut/Filter configuration file" ) );
fileBrowserButtonXML->SetFileDialogWildcard( tmpWildCard ); 
fileBrowserButtonXML->SetTextBox( textCtrlXML );
fileBrowserButtonXML->Enable();

enableOutputTraceWidgets( false ); 

wxString directory;

if ( !xmlConfigurationFile.empty() )
{
directory = xmlConfigurationFile;
}
else if ( !globalXMLsPath.empty() )
{
wxFileName auxDirectory( wxString( globalXMLsPath.c_str(), wxConvUTF8 )  );
if( !auxDirectory.IsDir() )
auxDirectory = auxDirectory.GetPathWithSep();
directory = auxDirectory.GetFullPath();
fileBrowserButtonXML->SetPath( directory );
}


textCutterBeginCut->SetValidator( wxTextValidator( wxFILTER_NUMERIC ));
textCutterEndCut->SetValidator( wxTextValidator( wxFILTER_NUMERIC ));
textCutterTasks->SetValidator( wxTextValidator( wxFILTER_NUMERIC ));
textFilterMinBurstTime->SetValidator( wxTextValidator( wxFILTER_NUMERIC ));
textSCSamplingInterval->SetValidator( wxTextValidator( wxFILTER_NUMERIC ));
textSCMinimumBurstTime->SetValidator( wxTextValidator( wxFILTER_NUMERIC ));
}




bool CutFilterDialog::ShowToolTips()
{
return true;
}



wxBitmap CutFilterDialog::GetBitmapResource( const wxString& name )
{
wxUnusedVar(name);
if (name == wxT("icons/arrow_up.xpm"))
{
wxBitmap bitmap(arrow_up_xpm);
return bitmap;
}
else if (name == wxT("icons/arrow_down.xpm"))
{
wxBitmap bitmap(arrow_down_xpm);
return bitmap;
}
return wxNullBitmap;
}



wxIcon CutFilterDialog::GetIconResource( const wxString& name )
{
wxUnusedVar(name);
return wxNullIcon;
}


bool CutFilterDialog::isFileSelected( FileBrowserButton *fpc )
{
wxString path = fpc->GetPath();
wxFileName tmpName( path );

return !( path == _("") || tmpName.IsDir() );
}


bool CutFilterDialog::isFileSelected( const string& fpc )
{
wxString path = wxString( fpc.c_str(), wxConvUTF8 );
wxFileName tmpName( path );

return !( path == _("") || tmpName.IsDir() );
}




void CutFilterDialog::OnIdle( wxIdleEvent& event )
{
if( waitingGlobalTiming )
{
TTime auxBeginTime = wxGetApp().GetGlobalTimingBegin();
TTime auxEndTime = wxGetApp().GetGlobalTimingEnd();

if ( auxBeginTime > auxEndTime )
{
TTime tmpTime = auxBeginTime;
auxBeginTime = auxEndTime;
auxEndTime = tmpTime;
}

if ( radioCutterCutByTime->GetValue() )
{
textCutterBeginCut->SetValue( formatTime( auxBeginTime ) );
textCutterEndCut->SetValue( formatTime( auxEndTime ) );
}
else
{
textCutterBeginCut->SetValue( formatPercent( auxBeginTime ) );
textCutterEndCut->SetValue( formatPercent( auxEndTime ) );
}

if( !wxGetApp().GetGlobalTiming() )
{
waitingGlobalTiming = false;
}

}
}




void CutFilterDialog::OnInitDialog( wxInitDialogEvent& event )
{
fileBrowserButtonInputTrace->SetPath( wxString( nameSourceTrace.c_str(), wxConvUTF8 ) );
checkLoadResultingTrace->SetValue( loadResultingTrace );
checkRunAppWithResultingTrace->SetValue( runAppWithResultingTrace );
if ( globalXMLsPath.compare( "" ) == 0 )
globalXMLsPath = nameSourceTrace;
}



double CutFilterDialog::formatPercent(const wxString whichPercent )
{
TTime tmpTime;

bool done = LabelConstructor::getTimeValue( std::string( whichPercent ),
NS,                 
PERCENT_PRECISION,  
tmpTime );
if( !done )
whichPercent.ToDouble( &tmpTime );

return (double)tmpTime;
}

wxString CutFilterDialog::formatPercent( double value )
{
stringstream auxSStr;
wxString auxNumber;

LabelConstructor::presetBaseFormat( auxSStr );

auxSStr << value;
auxNumber << wxString::FromUTF8( auxSStr.str().c_str() );

return auxNumber;
}


TTime CutFilterDialog::formatTime( const wxString whichTime )
{
TTime tmpTime;
Trace *tmpTrace = getTrace();

bool done = LabelConstructor::getTimeValue( std::string( whichTime ),
tmpTrace != nullptr ? tmpTrace->getTimeUnit() : NS,
ParaverConfig::getInstance()->getTimelinePrecision(),
tmpTime );

if( !done )
whichTime.ToDouble( &tmpTime );

return tmpTime;
}


wxString CutFilterDialog::formatTime( TTime whichTime )
{
Trace *tmpTrace = getTrace();

return LabelConstructor::timeLabel( whichTime,
tmpTrace != nullptr ? tmpTrace->getTimeUnit() : NS,
ParaverConfig::getInstance()->getTimelinePrecision() );
}




void CutFilterDialog::OnButtonCutterSelectRegionClick( wxCommandEvent& event )
{
gTimeline * timeline = paraverMain::myParaverMain->GetSelectedTimeline();
if ( timeline != nullptr && 
timeline->GetMyWindow() == paraverMain::myParaverMain->GetCurrentTimeline() &&
!timeline->IsShown() )  
{
Timeline *tmpWin = timeline->GetMyWindow();
tmpWin->setShowWindow( !tmpWin->getShowWindow() );

if( tmpWin->getShowWindow() )
timeline->Raise();
} 

radioCutterCutByTime->SetValue( true );
wxGetApp().ActivateGlobalTiming( this );
waitingGlobalTiming = true;
cutterByTimePreviouslyChecked = true;
}





void CutFilterDialog::OnButtonCutterAllTraceClick( wxCommandEvent& event )
{
radioCutterCutByTimePercent->SetValue( true );
cutterByTimePreviouslyChecked = false;

textCutterBeginCut->SetValue( formatPercent( 0.0 ));
textCutterEndCut->SetValue( formatPercent( 100.0 ));
}



void CutFilterDialog::OnButtonCutterSelectRegionUpdate( wxUpdateUIEvent& event )
{
buttonCutterSelectRegion->Enable( !LoadedWindows::getInstance()->emptyWindows() );
}


bool CutFilterDialog::CheckStringTasks( wxString taskStr )
{
if( taskStr == _( "" ) )
return true;

stringstream sstr( string( taskStr.mb_str() ) );

while( !sstr.eof() )
{
string tmpStr;
long tmpLong;

std::getline( sstr, tmpStr, ',' );

stringstream tmpStream( tmpStr );
std::getline( tmpStream, tmpStr, '-' );
if( !( stringstream( tmpStr ) >> tmpLong ) )
return false;
else if( tmpLong == 0 )
return false;

if( !tmpStream.eof() )
{
std::getline( tmpStream, tmpStr );
if( !( stringstream( tmpStr ) >> tmpLong ) )
return false;
else if( tmpLong == 0 )
return false;
}
}

return true;
}


void CutFilterDialog::CheckCutterOptions( bool &previousWarning )
{
if ( !previousWarning &&
textCutterBeginCut->GetValue() == _("") &&
textCutterEndCut->GetValue() == _("") &&
textCutterTasks->GetValue() != _("") &&
CheckStringTasks( textCutterTasks->GetValue() ))
{
wxMessageDialog message( this,
_("Cutter:\nEmpty times.\n\nDo you want to cut these tasks"
" all along the 100% of the trace?"),
_("Warning"), wxYES_NO|wxYES_DEFAULT );
if ( message.ShowModal() == wxID_YES )
{
radioCutterCutByTimePercent->SetValue( true );
textCutterBeginCut->SetValue( formatPercent( 0.0 ));
textCutterEndCut->SetValue( formatPercent( 100.0 ));
}
else
{
textCutterBeginCut->SetFocus();
previousWarning = true;
}
}

if( !previousWarning && !CheckStringTasks( textCutterTasks->GetValue() ) )
{
wxMessageDialog message( this, _("Cutter:\nNot allowed format in tasks text.\n\nPlease set it properly."), _("Warning"), wxOK );
message.ShowModal();
textCutterTasks->SetFocus();
previousWarning = true;
}

if ( !previousWarning && textCutterBeginCut->GetValue() == _("") )
{
wxMessageDialog message( this, _("Cutter:\nPlease set the initial time."), _( "Warning" ), wxOK );
message.ShowModal();
textCutterBeginCut->SetFocus();
previousWarning = true;
}

if ( !previousWarning && textCutterEndCut->GetValue() == _("") )
{
wxMessageDialog message( this, _("Cutter:\nPlease set the final time."), _("Warning"), wxOK );
message.ShowModal();
textCutterEndCut->SetFocus();
previousWarning = true;
}

TTime cutterBeginTime, cutterEndTime;
readTimes( radioCutterCutByTime->GetValue(), cutterBeginTime, cutterEndTime );

if ( !previousWarning && cutterBeginTime < 0.0 )
{
wxMessageDialog message( this, _("Cutter:\nTimes must be positive numbers.\n\nPlease set begin time properly."), _("Warning"), wxOK );
message.ShowModal();
textCutterBeginCut->SetFocus();
previousWarning = true;

textCutterBeginCut->SetValue( formatTime( 0.0 ) );
}

if ( !previousWarning && cutterEndTime < 0.0 )
{
wxMessageDialog message( this, _("Cutter:\nTimes must be positive numbers.\n\nPlease set end time properly."), _("Warning"), wxOK );
message.ShowModal();
textCutterEndCut->SetFocus();
previousWarning = true;

textCutterEndCut->SetValue(  formatTime( 0.0 ) );
}

if ( !previousWarning && radioCutterCutByTimePercent->GetValue() && cutterBeginTime > 100.0 )
{
wxMessageDialog message( this, _("Cutter:\nBegin time percent greater than 100 %.\n\nPlease set time percent properly."), _("Warning"), wxOK );
message.ShowModal();
radioCutterCutByTimePercent->SetFocus();
previousWarning = true;

textCutterBeginCut->SetValue( formatPercent( 100.0 ) );
}

if ( !previousWarning && radioCutterCutByTimePercent->GetValue() && cutterEndTime > 100.0 )
{
wxMessageDialog message( this, _("Cutter:\nEnd time percent greater than 100 %.\n\nPlease set time percent properly."), _("Warning"), wxOK );
message.ShowModal();
radioCutterCutByTimePercent->SetFocus();
previousWarning = true;

textCutterEndCut->SetValue( formatPercent( 100.0 ) );
}

if ( !previousWarning && cutterBeginTime == cutterEndTime )
{
wxMessageDialog message( this, _("Cutter:\nSame time for both limits.\n\nPlease set time range properly."), _("Warning"), wxOK );
message.ShowModal();
textCutterBeginCut->SetFocus();
previousWarning = true;
}

if ( !previousWarning && cutterBeginTime > cutterEndTime ) 
{
wxMessageDialog message( this, _("Cutter:\nBegin time greater than end time.\n\nPlease set time range properly."), _("Warning"), wxOK );
message.ShowModal();
textCutterBeginCut->SetFocus();
previousWarning = true;
}
}


void CutFilterDialog::TransferCutterDataToWindow( TraceOptions *traceOptions )
{
stringstream aux;

aux.str("");
aux << traceOptions->get_max_trace_size();
textCutterMaximumTraceSize->SetValue( wxString::FromUTF8( aux.str().c_str() ) );

if ( traceOptions->get_by_time() )
radioCutterCutByTime->SetValue( true );
else
radioCutterCutByTimePercent->SetValue( true );

if ( radioCutterCutByTime->GetValue() )
{
textCutterBeginCut->SetValue( formatTime( (TTime)traceOptions->get_min_cutting_time() ) );
textCutterEndCut->SetValue(  formatTime( (TTime)traceOptions->get_max_cutting_time() ) );
}
else
{
textCutterBeginCut->SetValue( formatPercent( (TTime)traceOptions->get_minimum_time_percentage() ) );
textCutterEndCut->SetValue(  formatPercent( (TTime)traceOptions->get_maximum_time_percentage() ) );
}

checkCutterUseOriginalTime->SetValue( traceOptions->get_original_time() );
if ( traceOptions->get_original_time() )
{
checkCutterDontBreakStates->SetValue( false );
checkCutterDontBreakStates->Disable();
}
else
{
checkCutterDontBreakStates->SetValue( !traceOptions->get_break_states() );
checkCutterDontBreakStates->Enable();
}

checkCutterRemoveFirstState->SetValue( traceOptions->get_remFirstStates() );
checkCutterRemoveLastState->SetValue( traceOptions->get_remLastStates() );
checkCutterKeepEvents->SetValue( traceOptions->get_keep_boundary_events() );
checkCutterKeepEventsWithoutStates->SetValue( traceOptions->get_keep_all_events() );

TraceOptions::TTasksList auxList;
traceOptions->get_tasks_list( auxList );
textCutterTasks->SetValue( wxString::FromUTF8( auxList ).Trim( true ).Trim( false ) );
}


Trace *CutFilterDialog::getTrace()
{
Trace *tmpTrace = nullptr;

if( paraverMain::myParaverMain->GetCurrentTimeline() != nullptr )
tmpTrace = paraverMain::myParaverMain->GetCurrentTimeline()->getTrace();
else if( paraverMain::myParaverMain->GetCurrentHisto() != nullptr )
tmpTrace = paraverMain::myParaverMain->GetCurrentHisto()->getTrace();

return tmpTrace;
}


void CutFilterDialog::readTimes( bool byTime, TTime &whichBeginTime, TTime &whichEndTime )
{
if( byTime )
{
whichBeginTime = formatTime( textCutterBeginCut->GetValue() );
whichEndTime = formatTime( textCutterEndCut->GetValue() );
}
else
{
whichBeginTime = formatPercent( textCutterBeginCut->GetValue() );
whichEndTime = formatPercent( textCutterEndCut->GetValue() );
}
}


void CutFilterDialog::TransferWindowToCutterData( bool previousWarning )
{
if ( !previousWarning )
{
TTime tmpBeginTime, tmpEndTime;

unsigned long long auxBeginTime, auxEndTime;
unsigned long long auxBeginPercent, auxEndPercent;

traceOptions->set_max_trace_size( textCutterMaximumTraceSize->GetValue() );
traceOptions->set_by_time( radioCutterCutByTime->GetValue() );

bool byTime = radioCutterCutByTime->GetValue();
readTimes( byTime, tmpBeginTime, tmpEndTime );
if( byTime )
{
auxBeginTime = (unsigned long long) round( tmpBeginTime );
auxEndTime = (unsigned long long) round( tmpEndTime );
auxBeginPercent = 0;
auxEndPercent = 0;
}
else
{
auxBeginTime = 0;
auxEndTime = 0;
auxBeginPercent = (unsigned long long) round( tmpBeginTime );
auxEndPercent = (unsigned long long) round( tmpEndTime );
}
traceOptions->set_min_cutting_time( auxBeginTime );
traceOptions->set_max_cutting_time( auxEndTime );
traceOptions->set_minimum_time_percentage( auxBeginPercent );
traceOptions->set_maximum_time_percentage( auxEndPercent );

traceOptions->set_original_time( checkCutterUseOriginalTime->IsChecked() );
traceOptions->set_break_states( !checkCutterDontBreakStates->IsChecked() );
traceOptions->set_remFirstStates( checkCutterRemoveFirstState->IsChecked() );
traceOptions->set_remLastStates( checkCutterRemoveLastState->IsChecked() );
traceOptions->set_keep_boundary_events( checkCutterKeepEvents->IsChecked() );
traceOptions->set_keep_all_events( checkCutterKeepEventsWithoutStates->IsChecked() );

#ifdef UNICODE
traceOptions->set_tasks_list( (char *)textCutterTasks->GetValue().mb_str().data() );
#else
traceOptions->set_tasks_list( (char *)textCutterTasks->GetValue().mb_str() );
#endif
}
}




void CutFilterDialog::OnButtonFilterSelectAllClick( wxCommandEvent& event )
{
for( size_t i = 0; i < checkListFilterStates->GetCount(); ++i )
{
checkListFilterStates->Check( i );
}
}


void CutFilterDialog::OnButtonFilterUnselectAllClick( wxCommandEvent& event )
{
for( size_t i = 0; i < checkListFilterStates->GetCount(); ++i )
{
checkListFilterStates->Check( i, false );
}
}




void CutFilterDialog::OnButtonFilterDeleteClick( wxCommandEvent& event )
{
wxArrayInt selec;

if( listboxFilterEvents->GetSelections( selec ) == 0 )
return;

listboxFilterEvents->Delete( selec[ 0 ] );
}



void CutFilterDialog::CheckStatesList( size_t begin, bool value )
{
for( size_t i = begin; i < checkListFilterStates->GetCount(); ++i )
checkListFilterStates->Check( (int)i, value );
}

void CutFilterDialog::CheckStatesList( TraceOptions::TStateNames statesList )
{
size_t s = 0;
wxArrayString newStates;
size_t oldMaxStates = checkListFilterStates->GetCount();

while( s < 20 && statesList[ s ] != nullptr )
{ 
bool found = false;
wxString stateNameToCheck( statesList[ s ], wxConvUTF8 );
stateNameToCheck = stateNameToCheck.Trim( true ).Trim( false );
for( size_t i = 0; i < checkListFilterStates->GetCount(); ++i )
{
wxString stateName = checkListFilterStates->GetString( i ); 
if( stateNameToCheck == stateName )
{
checkListFilterStates->Check( i );
found = true;
break;
}
}

if( !found )
{
newStates.Add( stateNameToCheck );
}

++s;
}

if( s == 0 )
{
for( size_t i = 0; i < checkListFilterStates->GetCount(); ++i )
{
wxString stateName = checkListFilterStates->GetString( i );
if( wxString("Running", wxConvUTF8) == stateName )
checkListFilterStates->Check( i );
}
}

if ( newStates.GetCount() > 0 )
{
checkListFilterStates->InsertItems( newStates, checkListFilterStates->GetCount() );
CheckStatesList( oldMaxStates, !( checkListFilterStates->GetCount() == 0 ) );
}
}




void CutFilterDialog::OnButtonFilterAddClick( wxCommandEvent& event )
{
wxTextEntryDialog textEntry( this, 
wxString() << _("Allowed formats:\n")
<< _(" Single event type: \'Type\'\n")
<< _(" Range of event types: \'Begin type-End type\'\n")
<< _(" Values for a single type: \'Type:Value 1,...,Value n\'"),
_("Add events") );

if( textEntry.ShowModal() == wxID_OK )
{
wxString currentEntry( textEntry.GetValue() );
if( !currentEntry.IsEmpty() )
{
wxString allowedFormatsRE =
reSingleType + wxString( wxT( "|" ) ) + reRangeOfTypes + wxString( wxT( "|" ) ) + reValuesSepByCommaForType ;
if( !wxRegEx( allowedFormatsRE ).Matches( currentEntry ) )
{
wxMessageBox( _("Text inserted doesn't fit the allowed formats"), _("Not allowed format") );
}
else
{
currentEntry.Replace( _(" "), _("") );
listboxFilterEvents->Append( currentEntry );
}
}
}
}


bool CutFilterDialog::CheckEventsLine( wxString eventsStr )
{
if( eventsStr == _( "" ) )
return false;

stringstream sstr( string( eventsStr.mb_str() ) );

while( !sstr.eof() )
{
string tmpStr;
long tmpLong;

std::getline( sstr, tmpStr, ':' );

stringstream tmpStream( tmpStr );
std::getline( tmpStream, tmpStr, '-' );

if( !( stringstream( tmpStr ) >> tmpLong ) )
return false;
else if( tmpLong == 0 )
return false;

if( !tmpStream.eof() )
{
std::getline( tmpStream, tmpStr );
if( !( stringstream( tmpStr ) >> tmpLong ) )
return false;
else if( tmpLong == 0 )
return false;
}

while( !sstr.eof() )
{
std::getline( sstr, tmpStr, ',' );

if( !( stringstream( tmpStr ) >> tmpLong ) )
return false;
}
}

return true;
}


void CutFilterDialog::GetEventsFromLine( wxString eventsStr,
TraceOptions::TFilterTypes &eventTypes,
int &lastType )
{
if( eventsStr == _( "" ) )
return;

stringstream sstr( string( eventsStr.mb_str() ) );

while( !sstr.eof() )
{
string tmpStr;
long tmpLong;

std::getline( sstr, tmpStr, ':' );

stringstream tmpStream( tmpStr );
std::getline( tmpStream, tmpStr, '-' );

if( !( stringstream( tmpStr ) >> tmpLong ) )
return;
else if( tmpLong == 0 )
return;
else
{
eventTypes[ lastType ].type = tmpLong;
eventTypes[ lastType ].max_type = 0;
}

if( !tmpStream.eof() )
{
std::getline( tmpStream, tmpStr );
if( !( stringstream( tmpStr ) >> tmpLong ) )
return;
else if( tmpLong == 0 )
return;
else
{
eventTypes[ lastType ].max_type = tmpLong;
}
}

int index = 0;
while( !sstr.eof() )
{
std::getline( sstr, tmpStr, ',' );

if( !( stringstream( tmpStr ) >> tmpLong ) )
return;
else
{
eventTypes[ lastType ].value[index++] = tmpLong;
}
}
eventTypes[ lastType ].last_value = index;

++lastType;
}
}


void CutFilterDialog::GetEventsList( TraceOptions::TFilterTypes &eventTypes, int &lastType )
{
for( size_t i = 0; i < listboxFilterEvents->GetCount(); ++i )
{
if ( CheckEventsLine( listboxFilterEvents->GetString( i ) ) )
{
GetEventsFromLine( listboxFilterEvents->GetString( i ), eventTypes, lastType );
}
}
}


void CutFilterDialog::SetEventLine( TraceOptions::TFilterTypes eventTypes, int current )
{
stringstream auxLine;

auxLine << eventTypes[ current ].type;
if ( eventTypes[ current ].max_type != 0 )
{
auxLine << "-" << eventTypes[ current ].max_type;
}
else
{
if ( eventTypes[ current ].last_value != 0 )
{
auxLine << ":";
for( int j = 0; j < eventTypes[ current ].last_value; ++j )
{
auxLine << eventTypes[ current ].value[ j ];
if ( j < eventTypes[ current ].last_value - 1 )
{
auxLine << ",";
}
}
}
}

listboxFilterEvents->Append( wxString( auxLine.str().c_str(), wxConvUTF8 ) );
}


void CutFilterDialog::SetEventsList( TraceOptions::TFilterTypes eventTypes, int lastType )
{
listboxFilterEvents->Clear();

for( int i = 0; i < lastType; ++i )
{
SetEventLine( eventTypes, i );
}
}


void CutFilterDialog::CheckFilterOptions( bool &previousWarning )
{
if ( !previousWarning && !checkFilterDiscardStateRecords->IsChecked() )
{
bool statesSelected = false;
for (size_t i = 0; i < checkListFilterStates->GetCount(); ++i )
{
if ( checkListFilterStates->IsChecked( i ) )
{
statesSelected = true;
break;
}
}

if ( !statesSelected )
{
wxMessageDialog message( this, _("Filter:\nNo state selected.\n\nPlease select at least one or\ndiscard all the state records."), _( "Warning" ), wxOK );
message.ShowModal();
checkListFilterStates->SetFocus();
previousWarning = true;
}
}

double filterMinBurstTime;
textFilterMinBurstTime->GetValue().ToDouble( &filterMinBurstTime );
if ( !previousWarning && !checkFilterDiscardStateRecords->IsChecked() && filterMinBurstTime < 0.0 )
{
wxMessageDialog message( this, _("Filter:\nTimes must be positive numbers.\n\nPlease set minimum burst time properly."), _("Warning"), wxOK );
message.ShowModal();
textFilterMinBurstTime->SetFocus();
previousWarning = true;
}

if ( !previousWarning && !checkFilterDiscardEventRecords->IsChecked() && listboxFilterEvents->GetCount() == 0 )
{
wxMessageDialog message( this, _("Filter:\nThe list of events is empty.\n\nPlease add at least one event or\ndiscard all the event records."), _( "Warning" ), wxOK );
message.ShowModal();
buttonFilterAdd->SetFocus();
previousWarning = true;
}

if ( !previousWarning && !checkFilterDiscardEventRecords->IsChecked() && listboxFilterEvents->GetCount() > 20 )
{
wxMessageDialog message( this, _("Filter:\nToo much event lines.\n\nPlease delete at least one line."), _( "Warning" ), wxOK );
message.ShowModal();
listboxFilterEvents->SetFocus();
previousWarning = true;
}
}


void CutFilterDialog::TransferFilterDataToWindow( TraceOptions *traceOptions )
{
stringstream aux;

checkFilterDiscardStateRecords->SetValue( !traceOptions->get_filter_states() );
checkFilterDiscardEventRecords->SetValue( !traceOptions->get_filter_events() );
checkFilterDiscardCommunicationRecords->SetValue( !traceOptions->get_filter_comms() );

CheckStatesList( 0, traceOptions->get_all_states() ); 
if( !traceOptions->get_all_states() )
{
TraceOptions::TStateNames auxNames;
for( int i = 0; i < 20; ++i )
auxNames[ i ] = nullptr;

traceOptions->get_state_names( auxNames );
CheckStatesList( auxNames );
}

aux.str("");
aux << traceOptions->get_min_state_time();
textFilterMinBurstTime->SetValue( wxString::FromUTF8( aux.str().c_str() ));

checkFilterDiscardListedEvents->SetValue( traceOptions->get_discard_given_types() );

TraceOptions::TFilterTypes auxEvents;
traceOptions->get_filter_types( auxEvents );
SetEventsList( auxEvents, traceOptions->get_filter_last_type() );

aux.str("");
aux << traceOptions->get_min_comm_size();
textFilterSize->SetValue( wxString::FromUTF8( aux.str().c_str() ) );
}


void CutFilterDialog::TransferWindowToFilterData( bool previousWarning )
{
if ( !previousWarning )
{
traceOptions->set_filter_states( !checkFilterDiscardStateRecords->IsChecked() );
traceOptions->set_filter_events( !checkFilterDiscardEventRecords->IsChecked() );
traceOptions->set_filter_comms( !checkFilterDiscardCommunicationRecords->IsChecked() );

traceOptions->set_filter_by_call_time( false );

bool allStatesSelected = true;
for( size_t i = 0; i < checkListFilterStates->GetCount(); ++i )
{
if ( !checkListFilterStates->IsChecked( i ) )
{
allStatesSelected = false;
break;
}
}

TraceOptions::TStateNames auxNames;
for( int i = 0; i < 20; ++i )
auxNames[ i ] = nullptr;

traceOptions->set_all_states( allStatesSelected );
if ( allStatesSelected )
{
#ifdef _WIN32
auxNames[ 0 ] = _strdup( "All" );
#else
auxNames[ 0 ] = strdup( "All" );
#endif
}
else
{
int pos = 0;

for( size_t i = 0; i < checkListFilterStates->GetCount(); ++i )
{
if ( checkListFilterStates->IsChecked( i ) )
{
#ifdef _WIN32
auxNames[ pos++ ] = _strdup( (char *)(checkListFilterStates->GetString( i ).mb_str().data() ));
#elif defined UNICODE
auxNames[ pos++ ] = strdup( (char *)(checkListFilterStates->GetString( i ).mb_str().data() ));
#else
auxNames[ pos++ ] = strdup( (char *)(checkListFilterStates->GetString( i ).mb_str() ));
#endif
}
}

if( pos == 0 )
{
#ifdef _WIN32
auxNames[ 0 ] = _strdup( "Running" );
#else
auxNames[ 0 ] = strdup( "Running" );
#endif
}

traceOptions->set_state_names( auxNames );

unsigned long auxULong;
textFilterMinBurstTime->GetValue().ToULong( &auxULong );
traceOptions->set_min_state_time( (unsigned long long)auxULong );
}

traceOptions->set_discard_given_types( checkFilterDiscardListedEvents->IsChecked() );
TraceOptions::TFilterTypes auxEvents;
int lastType = 0;
GetEventsList( auxEvents, lastType );

traceOptions->set_filter_last_type( lastType );
traceOptions->set_filter_types( auxEvents );

traceOptions->set_min_comm_size( textFilterSize->GetValue() );
}
}





void CutFilterDialog::OnPanelSoftwareCountersUpdate( wxUpdateUIEvent& event )
{
staticTextSCSamplingInterval->Enable( radioSCOnIntervals->GetValue() );
textSCSamplingInterval->Enable( radioSCOnIntervals->GetValue() );

staticTextSCMinimumBurstTime->Enable( !radioSCOnIntervals->GetValue() );
textSCMinimumBurstTime->Enable( !radioSCOnIntervals->GetValue() );

checkSCRemoveStates->Enable( radioSCOnIntervals->GetValue() );
checkSCRemoveStates->SetValue( radioSCOnIntervals->GetValue() && checkSCRemoveStates->IsChecked() );

checkSCSummarizeUseful->Enable( radioSCOnIntervals->GetValue() );
checkSCSummarizeUseful->SetValue( radioSCOnIntervals->GetValue() && checkSCSummarizeUseful->IsChecked() );

checkSCGlobalCounters->Enable( !radioSCAccumulateValues->GetValue() );
checkSCGlobalCounters->SetValue( !radioSCAccumulateValues->GetValue() && checkSCGlobalCounters->IsChecked() );

checkSCOnlyInBurstsCounting->Enable( radioSCOnIntervals->GetValue() );
checkSCOnlyInBurstsCounting->SetValue( radioSCOnIntervals->GetValue() && checkSCOnlyInBurstsCounting->IsChecked() );
}




void CutFilterDialog::OnButtonScSelectedEventsAddClick( wxCommandEvent& event )
{
wxTextEntryDialog textEntry( this, 
wxString() << _("Allowed formats:\n")
<< _(" Single event type: \'Type\'\n")
<< _(" Values for a single type: \'Type:Value 1,...,Value n\'"),
_("Add events") );

if( textEntry.ShowModal() == wxID_OK )
{
wxString currentEntry( textEntry.GetValue() );
if( !currentEntry.IsEmpty() )
{
wxString allowedFormatsRE = reSingleType + wxString( wxT( "|" ) ) + reValuesSepByCommaForType;
if( !wxRegEx( allowedFormatsRE ).Matches( currentEntry ) )
{
wxMessageBox( _("Text inserted doesn't fit the allowed formats"), _("Not allowed format") );
}
else
{
currentEntry.Replace( _(" "), _("") );
listSCSelectedEvents->Append( currentEntry );
}
}
}
}




void CutFilterDialog::OnButtonScSelectedEventsDeleteClick( wxCommandEvent& event )
{
wxArrayInt selec;

if( listSCSelectedEvents->GetSelections( selec ) == 0 )
return;

listSCSelectedEvents->Delete( selec[ 0 ] );
}




void CutFilterDialog::OnButtonScKeepEventsAddClick( wxCommandEvent& event )
{
wxTextEntryDialog textEntry( this, 
wxString() << _("Allowed formats:\n")
<< _(" Single event type: \'Type\'\n")
<< _(" Range of event types: \'Begin type-End type\'\n"),
_("Add events") );

if( textEntry.ShowModal() == wxID_OK )
{
wxString currentEntry( textEntry.GetValue() );
if( !currentEntry.IsEmpty() )
{
wxString allowedFormatsRE = reSingleType + wxString( wxT( "|" ) ) + reRangeOfTypes;
if( !wxRegEx( allowedFormatsRE ).Matches( currentEntry ) )
{
wxMessageBox( _("Text inserted doesn't fit the allowed formats"), _("Not allowed format") );
}
else
{
currentEntry.Replace( _(" "), _("") );
listSCKeepEvents->Append( currentEntry );
}
}
}
}




void CutFilterDialog::OnButtonScKeepEventsDeleteClick( wxCommandEvent& event )
{
wxArrayInt selec;

if( listSCKeepEvents->GetSelections( selec ) == 0 )
return;

listSCKeepEvents->Delete( selec[ 0 ] );
}


bool CutFilterDialog::SetSoftwareCountersEventsListToString( string listEvents, wxListBox *selectedEvents )
{
selectedEvents->Clear();
stringstream auxList( listEvents );
while( !auxList.eof() )
{
string tmpStr;

std::getline( auxList, tmpStr, ';' );
selectedEvents->Append( wxString( tmpStr.c_str(), wxConvUTF8 ).Trim(true).Trim(false) );
}

return true;
}



char *CutFilterDialog::GetSoftwareCountersEventsListToString( wxListBox *selectedEvents )
{
string listStr;

for( size_t i = 0; i < selectedEvents->GetCount(); ++i )
{
#ifdef UNICODE
string auxLineStr = string( selectedEvents->GetString( i ).mb_str().data() );
#else
string auxLineStr = string( selectedEvents->GetString( i ).mb_str() );
#endif
listStr += auxLineStr;
if ( i != selectedEvents->GetCount() - 1 )
listStr += string(";");
}

#ifdef _WIN32
return _strdup( listStr.c_str() );
#else
return strdup( listStr.c_str() );
#endif
}


void CutFilterDialog::CheckSoftwareCountersOptions( bool &previousWarning )
{
if ( !previousWarning && radioSCOnIntervals->GetValue() && textSCSamplingInterval->GetValue() == _("") )
{
wxMessageDialog message( this, _("Software Counters:\nPlease set the sampling interval time."), _( "Warning" ), wxOK );
message.ShowModal();
textSCSamplingInterval->SetFocus();
previousWarning = true;
}

if ( !previousWarning && radioSCOnStates->GetValue() && textSCMinimumBurstTime->GetValue() == _("")  )
{
wxMessageDialog message( this, _("Software Counters:\nPlease set the minimum burst time."), _("Warning"), wxOK );
message.ShowModal();
textSCMinimumBurstTime->SetFocus();
previousWarning = true;
}

double regionTime;
if ( !previousWarning && radioSCOnIntervals->GetValue() )
textSCSamplingInterval->GetValue().ToDouble( &regionTime );
else
textSCMinimumBurstTime->GetValue().ToDouble( &regionTime );


if ( !previousWarning && regionTime < 0.0 )
{
if ( radioSCOnIntervals->GetValue() )
{
wxMessageDialog message( this, _("Software Counters:\nTimes must be positive numbers.\n\nPlease set sampling interval burst time properly."), _("Warning"), wxOK );
message.ShowModal();
textSCSamplingInterval->SetFocus();
}
else
{
wxMessageDialog message( this, _("Software Counters:\nTimes must be positive numbers.\n\nPlease set minimum burst time properly."), _("Warning"), wxOK );
message.ShowModal();
textSCMinimumBurstTime->SetFocus();
}
previousWarning = true;
}
if ( !previousWarning && listSCSelectedEvents->GetCount() == 0 )
{
wxMessageDialog message( this, _("Software Counters:\nThe list of event types is empty.\n\nPlease add at least one event type."), _("Warning"), wxOK );
message.ShowModal();
buttonSCSelectedEventsAdd->SetFocus();
previousWarning = true;
}
}


void CutFilterDialog::TransferWindowToSoftwareCountersData( bool previousWarning )
{
unsigned long auxULong;

if ( !previousWarning )
{
traceOptions->set_sc_onInterval( radioSCOnIntervals->GetValue() );

textSCSamplingInterval->GetValue().ToULong( &auxULong );
traceOptions->set_sc_sampling_interval( (unsigned long long)auxULong );
textSCMinimumBurstTime->GetValue().ToULong( &auxULong );
traceOptions->set_sc_minimum_burst_time( (unsigned long long)auxULong );

traceOptions->set_sc_types( GetSoftwareCountersEventsListToString( listSCSelectedEvents ) );

traceOptions->set_sc_acumm_counters( radioSCAccumulateValues->GetValue() );

traceOptions->set_sc_remove_states( checkSCRemoveStates->IsChecked() );
traceOptions->set_sc_summarize_states( checkSCSummarizeUseful->IsChecked() );
traceOptions->set_sc_global_counters( checkSCGlobalCounters->IsChecked() );
traceOptions->set_sc_only_in_bursts( checkSCOnlyInBurstsCounting->IsChecked() );

traceOptions->set_sc_types_kept( GetSoftwareCountersEventsListToString( listSCKeepEvents ));

}
}


void CutFilterDialog::TransferSoftwareCountersDataToWindow( TraceOptions *traceOptions )
{
stringstream aux;

if ( traceOptions->get_sc_onInterval() )
radioSCOnIntervals->SetValue( true );
else
radioSCOnStates->SetValue( true );

aux.str("");
aux << traceOptions->get_sc_sampling_interval();
textSCSamplingInterval->SetValue( wxString::FromUTF8( aux.str().c_str() ) );

aux.str("");
aux << traceOptions->get_sc_minimum_burst_time();
textSCMinimumBurstTime->SetValue( wxString::FromUTF8( aux.str().c_str() ) );

bool done = SetSoftwareCountersEventsListToString( string( traceOptions->get_sc_types() ),
listSCSelectedEvents );

radioSCAccumulateValues->SetValue( traceOptions->get_sc_acumm_counters() );
checkSCRemoveStates->SetValue( traceOptions->get_sc_remove_states() );
checkSCSummarizeUseful->SetValue( traceOptions->get_sc_summarize_states() );
checkSCGlobalCounters->SetValue( traceOptions->get_sc_global_counters() );
checkSCOnlyInBurstsCounting->SetValue( traceOptions->get_sc_only_in_bursts() );

done = SetSoftwareCountersEventsListToString( string( traceOptions->get_sc_types_kept() ),
listSCKeepEvents );

}


void CutFilterDialog::TransferCommonDataToWindow( vector< string > order )
{
if( order.size() > 0 )
{
vector< string > auxListToolOrder; 

for( size_t i = 0; i < order.size(); ++i )
{
auxListToolOrder.push_back( order[i] );
listToolOrder.erase( find( listToolOrder.begin(), listToolOrder.end(), order[i]));
}

for( size_t i = 0; i < listToolOrder.size(); ++i )
{
auxListToolOrder.push_back( listToolOrder[i] );
}

listToolOrder.swap( auxListToolOrder );
UpdateExecutionChain();

for( size_t i = 0; i < order.size(); ++i )
{
checkListExecutionChain->Check( i, true );
}

for( size_t i = order.size(); i < checkListExecutionChain->GetCount(); ++i )
{
checkListExecutionChain->Check( i, false );
}

checkListExecutionChain->SetSelection( 0 );
}
}




void CutFilterDialog::OnButtonSaveXmlClick( wxCommandEvent& event )
{
bool previousWarning = false;

TransferToolOrderToCommonData();

if ( !previousWarning )
{
for ( vector< string >::const_iterator it = filterToolOrder.begin(); it != filterToolOrder.end(); ++it )
{
if ( *it == TraceCutter::getID() )
{
CheckCutterOptions( previousWarning );
TransferWindowToCutterData( previousWarning );
}
else if ( *it == TraceFilter::getID() )
{
CheckFilterOptions( previousWarning );
TransferWindowToFilterData( previousWarning );
}
else if ( *it == TraceSoftwareCounters::getID() )
{
CheckSoftwareCountersOptions( previousWarning );
TransferWindowToSoftwareCountersData( previousWarning );
}
else
{
}
}
}

if ( !previousWarning )
{
wxFileName auxDirectory( wxString( globalXMLsPath.c_str(), wxConvUTF8 )); 

if( !auxDirectory.IsDir() )
auxDirectory = auxDirectory.GetPathWithSep();

wxString directory( auxDirectory.GetFullPath() );
wxString wildcard( _( "XML configuration file (*.xml)|*.xml|All files (*.*)|*.*" ) );
std::vector< wxString > extensions;
extensions.push_back( wxT( "xml" ) );
FileDialogExtension xmlSelectionDialog( this,
_( "Save XML Cut/Filter configuration file" ),
directory,
_( "" ),
wildcard,
wxFD_SAVE|wxFD_CHANGE_DIR,
wxDefaultPosition,
wxDefaultSize,
_( "filedlg" ),
extensions );

if( xmlSelectionDialog.ShowModal() == wxID_OK )
{
wxString path( xmlSelectionDialog.GetPath() );
traceOptions->saveXML( filterToolOrder, string( path.mb_str()) );

globalXMLsPath = string( xmlSelectionDialog.GetDirectory().mb_str() ) + PATH_SEP;
newXMLsPath = true;

fileBrowserButtonXML->SetPath( path );
}
}
}


bool CutFilterDialog::GetLoadedXMLPath( string &XMLPath )
{
if ( newXMLsPath )
{
XMLPath = globalXMLsPath;
}

return newXMLsPath;
}




void CutFilterDialog::OnBitmapbuttonPushDownFilterClick( wxCommandEvent& event )
{
int lastItemSelected = checkListExecutionChain->GetSelection();

if ( lastItemSelected != wxNOT_FOUND && lastItemSelected < 2  && lastItemSelected > -1 )
{
vector< bool > checked;
for( unsigned int i = 0; i < listToolOrder.size(); ++i )
checked.push_back( checkListExecutionChain->IsChecked( i ) );

bool auxFirst  = checkListExecutionChain->IsChecked( lastItemSelected );
bool auxSecond = checkListExecutionChain->IsChecked( lastItemSelected + 1 );

string auxNameFirst = listToolOrder[ lastItemSelected ];
listToolOrder[ lastItemSelected ] =  listToolOrder[ lastItemSelected + 1 ];
listToolOrder[ lastItemSelected + 1 ] = auxNameFirst;

UpdateExecutionChain();

for( unsigned int i = 0; i < listToolOrder.size(); ++i )
checkListExecutionChain->Check( i, checked[ i ] );

checkListExecutionChain->Check( lastItemSelected, auxSecond );
checkListExecutionChain->Check( lastItemSelected + 1, auxFirst );

checkListExecutionChain->SetSelection( ++lastItemSelected );

setOutputName( globalEnable(), false, std::string( fileBrowserButtonInputTrace->GetPath().mb_str() ) );
}
}


const vector< string > CutFilterDialog::changeToolsNameToID( const vector< string >& listToolWithNames )
{
vector< string > listToolWithIDs;
for ( vector< string >::const_iterator it = listToolWithNames.begin(); it != listToolWithNames.end(); ++it )
{
listToolWithIDs.push_back( GetLocalKernel()->getToolID( *it ) );
}

return listToolWithIDs;
}


const vector< string > CutFilterDialog::changeToolsIDsToNames( const vector< string >& listToolIDs )
{
vector< string > listToolWithNames;
for ( vector< string >::const_iterator it = listToolIDs.begin(); it != listToolIDs.end(); ++it )
{
listToolWithNames.push_back( GetLocalKernel()->getToolName( *it ) );
}

return listToolWithNames;
}


bool CutFilterDialog::isExecutionChainEmpty()
{
bool emptyChain = true;

for ( size_t i = 0; i < checkListExecutionChain->GetCount(); ++i )
{
if ( checkListExecutionChain->IsChecked( (int)i ) )
{
emptyChain = false;
break;
}
}

return emptyChain;
}


void CutFilterDialog::ChangePageSelectionFromTabsToToolsOrderList()
{
int pos = 0;

for( vector< string >::iterator it = listToolOrder.begin(); it != listToolOrder.end(); ++it )
{
if ( *it == string( notebookTools->GetPageText( notebookTools->GetSelection() ).mb_str()) )
{
checkListExecutionChain->SetSelection( pos );
break; 
}
pos++;
}
}




void CutFilterDialog::OnCheckListExecutionChainSelected( wxCommandEvent& event )
{
int iSel = event.GetSelection();

if ( iSel > -1 )
ChangePageSelectionFromToolsOrderListToTabs( iSel );
}




void CutFilterDialog::OnNotebookCutFilterOptionsPageChanged( wxNotebookEvent& event )
{
ChangePageSelectionFromTabsToToolsOrderList();
}


bool CutFilterDialog::globalEnable( const string& auxInputTrace )
{
return ( isFileSelected( auxInputTrace ) && !isExecutionChainEmpty() );
}


bool CutFilterDialog::globalEnable()
{
return ( isFileSelected( fileBrowserButtonInputTrace ) && !isExecutionChainEmpty() );
}


void CutFilterDialog::setOutputName( bool enable,
bool saveGeneratedName,
const string& sourceTrace )
{
if ( enable )
{
TransferToolOrderToCommonData();

string currentDstTrace =
GetLocalKernel()->getNewTraceName(
sourceTrace, outputPath, filterToolOrder, saveGeneratedName );
wxString outputName = wxString( currentDstTrace.c_str(), wxConvUTF8 );
fileBrowserButtonOutputTrace->SetPath( outputName );

outputPath = std::string( wxFileName( wxString( currentDstTrace.c_str(), wxConvUTF8 ) ).GetPathWithSep().mb_str() );
}
}



void CutFilterDialog::OnChecklistboxExecutionChainDoubleClicked( wxCommandEvent& event )
{
int iSel = event.GetSelection();
if ( iSel > -1 )
{
checkListExecutionChain->Check( iSel, !checkListExecutionChain->IsChecked( iSel ) );
enableOutputTraceWidgets( globalEnable() );
setOutputName( globalEnable(), false, std::string( fileBrowserButtonInputTrace->GetPath().mb_str() ) );

EnableSingleTab( iSel );
ChangePageSelectionFromToolsOrderListToTabs( iSel );
}
}


void CutFilterDialog::enableOutputTraceWidgets( bool enable )
{
txtOutputTrace->Enable( enable );
fileBrowserButtonOutputTrace->Enable( enable );
checkLoadResultingTrace->Enable( enable );
checkRunAppWithResultingTrace->Enable( enable );
}




void CutFilterDialog::OnChecklistboxExecutionChainToggled( wxCommandEvent& event )
{
int iSel = event.GetSelection();

if ( iSel > -1 )
{
UpdateOutputTraceName();
EnableToolTab( iSel );
}
}



void CutFilterDialog::OnBitmapbuttonPushUpFilterClick( wxCommandEvent& event )
{
int lastItemSelected = checkListExecutionChain->GetSelection();

if ( lastItemSelected != wxNOT_FOUND && lastItemSelected > 0 )
{
vector< bool > checked;
for( unsigned int i = 0; i < listToolOrder.size(); ++i )
checked.push_back( checkListExecutionChain->IsChecked( i ) );

bool auxFirst  = checkListExecutionChain->IsChecked( lastItemSelected - 1 );
bool auxSecond = checkListExecutionChain->IsChecked( lastItemSelected );

string auxNameFirst = listToolOrder[ lastItemSelected - 1 ];
listToolOrder[ lastItemSelected - 1 ] =  listToolOrder[ lastItemSelected ];
listToolOrder[ lastItemSelected ] = auxNameFirst;

UpdateExecutionChain();

for( unsigned int i = 0; i < listToolOrder.size(); ++i )
checkListExecutionChain->Check( i, checked[ i ] );

checkListExecutionChain->Check( lastItemSelected - 1, auxSecond );
checkListExecutionChain->Check( lastItemSelected, auxFirst );

checkListExecutionChain->SetSelection( --lastItemSelected );

setOutputName( globalEnable(), false, std::string( fileBrowserButtonInputTrace->GetPath().mb_str() ) );
}
}


void CutFilterDialog::UpdateExecutionChain()
{
wxArrayString items;
int order = 1;
for( vector< string >::iterator it = listToolOrder.begin(); it != listToolOrder.end(); ++it )
{
stringstream aux;
aux << order++;
items.Add(  wxString::FromUTF8( aux.str().c_str() ) + _( ".- " ) + wxString::FromUTF8( (*it).c_str() ) );
}

checkListExecutionChain->Clear();
checkListExecutionChain->InsertItems( items, 0 );
}


void CutFilterDialog::ChangePageSelectionFromToolsOrderListToTabs( int selected )
{
for( size_t i = 0; i < notebookTools->GetPageCount(); ++i )
{
if ( listToolOrder[ selected ] == string( notebookTools->GetPageText( i ).mb_str()) )
{
notebookTools->ChangeSelection( i );
}
}
}


void CutFilterDialog::EnableSingleTab( int selected )
{
string id = GetLocalKernel()->getToolID( listToolOrder[ selected ] );
int iTab = TABINDEX[ id ];
bool isChecked = checkListExecutionChain->IsChecked( selected );
(notebookTools->GetPage( iTab ))->Enable( isChecked );
}



void CutFilterDialog::EnableAllTabsFromToolsList()
{
for( size_t i = 0; i < listToolOrder.size(); ++i )
{
EnableSingleTab( (int)i );
}
}


void CutFilterDialog::CheckCommonOptions( bool &previousWarning, bool showWarning )
{
if ( !previousWarning && !isFileSelected( fileBrowserButtonInputTrace ) )
{
if ( showWarning )
{
wxMessageDialog message( this, _("Missing trace name.\nPlease choose the source trace."), _("Warning"), wxOK );
message.ShowModal();
}

fileBrowserButtonInputTrace->SetFocus();
previousWarning = true;
}

if ( !previousWarning && !isFileSelected( fileBrowserButtonOutputTrace ) )
{
if ( showWarning )
{
wxMessageDialog message( this, _("Missing trace name.\nPlease choose name for final trace."), _("Warning"), wxOK );
message.ShowModal();
}

fileBrowserButtonOutputTrace->SetFocus();
previousWarning = true;
}

if ( !previousWarning && isExecutionChainEmpty() )
{
if ( showWarning )
{
wxMessageDialog message( this, _("No utility selected.\nPlease choose the utilities to apply."), _( "Warning" ), wxOK );
message.ShowModal();
}

checkListExecutionChain->SetFocus();
previousWarning = true;
}
}


void CutFilterDialog::TransferToolOrderToCommonData()
{
filterToolOrder.clear();

for ( size_t i = 0; i < listToolOrder.size(); ++i )
{
if ( checkListExecutionChain->IsChecked( i ) )
{
filterToolOrder.push_back(
GetLocalKernel()->getToolID( listToolOrder[ i ] ));
}
}
}


void CutFilterDialog::TransferWindowToCommonData( bool previousWarning )
{
if ( !previousWarning )
{
nameSourceTrace = std::string( fileBrowserButtonInputTrace->GetPath().mb_str() );
nameDestinyTrace = std::string( fileBrowserButtonOutputTrace->GetPath().mb_str() );
loadResultingTrace = checkLoadResultingTrace->IsChecked();
runAppWithResultingTrace = checkRunAppWithResultingTrace->IsChecked();

TransferToolOrderToCommonData();
}
}




void CutFilterDialog::OnApplyUpdate( wxUpdateUIEvent& event )
{
buttonApply->Enable( globalEnable() );
}




void CutFilterDialog::OnApplyClick( wxCommandEvent& event )
{
bool previousWarning = false;

CheckCommonOptions( previousWarning, true );
TransferWindowToCommonData( previousWarning );

if ( !previousWarning )
{
for ( vector< string >::const_iterator it = filterToolOrder.begin(); it != filterToolOrder.end(); ++it )
{
if ( *it == TraceCutter::getID() )
{
CheckCutterOptions( previousWarning );
TransferWindowToCutterData( previousWarning );
}
else if ( *it == TraceFilter::getID() )
{
CheckFilterOptions( previousWarning );
TransferWindowToFilterData( previousWarning );
}
else if ( *it == TraceSoftwareCounters::getID() )
{
CheckSoftwareCountersOptions( previousWarning );
TransferWindowToSoftwareCountersData( previousWarning );
}
else
{
}
}

if( !previousWarning )
{
paraverMain::myParaverMain->OnOKCutFilterDialog( this );

delete this; 
}
}
}




void CutFilterDialog::OnButtonSaveXmlUpdate( wxUpdateUIEvent& event )
{
buttonSaveXml->Enable( !isExecutionChainEmpty() );
}




void CutFilterDialog::OnCheckboxFilterDiscardStateUpdate( wxUpdateUIEvent& event )
{
staticBoxSizerFilterStates->Enable( !checkFilterDiscardStateRecords->IsChecked() );
checkListFilterStates->Enable( !checkFilterDiscardStateRecords->IsChecked() );
buttonFilterSelectAll->Enable( !checkFilterDiscardStateRecords->IsChecked() );
buttonFilterUnselectAll->Enable( !checkFilterDiscardStateRecords->IsChecked() );
labelFilterMinBurstTime->Enable( !checkFilterDiscardStateRecords->IsChecked() );
textFilterMinBurstTime->Enable( !checkFilterDiscardStateRecords->IsChecked() );
}




void CutFilterDialog::OnCheckboxFilterDiscardEventUpdate( wxUpdateUIEvent& event )
{
staticBoxSizerFilterEvents->Enable( !checkFilterDiscardEventRecords->IsChecked() );
listboxFilterEvents->Enable( !checkFilterDiscardEventRecords->IsChecked() );
buttonFilterAdd->Enable( !checkFilterDiscardEventRecords->IsChecked() );
buttonFilterDelete->Enable( !checkFilterDiscardEventRecords->IsChecked() );
checkFilterDiscardListedEvents->Enable( !checkFilterDiscardEventRecords->IsChecked() );
}




void CutFilterDialog::OnCheckboxFilterDiscardCommunicationUpdate( wxUpdateUIEvent& event )
{
staticBoxSizerFilterCommunications->Enable( !checkFilterDiscardCommunicationRecords->IsChecked() );
staticTextFilterSize->Enable( !checkFilterDiscardCommunicationRecords->IsChecked() );
textFilterSize->Enable( !checkFilterDiscardCommunicationRecords->IsChecked() );
staticTextFilterSizeUnit->Enable( !checkFilterDiscardCommunicationRecords->IsChecked() );
}




void CutFilterDialog::OnCheckboxCheckCutterOriginalTimeUpdate( wxUpdateUIEvent& event )
{
if ( checkCutterUseOriginalTime->IsChecked() )
{
checkCutterDontBreakStates->SetValue( false );
checkCutterDontBreakStates->Disable();
}
else
{
checkCutterDontBreakStates->Enable();
}
}


void CutFilterDialog::SetXMLFile( const wxString& whichXMLFile, bool refresh )
{
wxString xmlSuffix = _(".xml");
wxString pathWithExtension;

if ( whichXMLFile.EndsWith( xmlSuffix )) 
{
wxString tmpFile = wxFileName( whichXMLFile ).GetFullPath();
if ( wxFileName::IsFileReadable( tmpFile ) )
{
fileBrowserButtonXML->SetPath( tmpFile );

if ( refresh )
TransferXMLFileToWindow( tmpFile );
}
}
}


void CutFilterDialog::TransferDataToWindow( vector< string > order, TraceOptions* traceOptions )
{
Freeze();

vector< string > toolsName = changeToolsIDsToNames( order );
TransferCommonDataToWindow( toolsName );

for( size_t i = 0; i < order.size(); ++i )
{
if ( order[ i ] == TraceCutter::getID() )
{
TransferCutterDataToWindow( traceOptions );
}
else if ( order[ i ] == TraceFilter::getID() )
{
TransferFilterDataToWindow( traceOptions );
}
else if ( order[ i ] == TraceSoftwareCounters::getID() )
{
TransferSoftwareCountersDataToWindow( traceOptions );
}
else
{
}
}

Thaw();  
}


void CutFilterDialog::UpdateGuiXMLSectionFromFile( TraceOptions *traceOptions,
vector< string > &toolIDsOrder )
{
TransferDataToWindow( toolIDsOrder, traceOptions );
EnableAllTabsFromToolsList();
ChangePageSelectionFromToolsOrderListToTabs( 0 );
}


void CutFilterDialog::UpdateGlobalXMLPath( const wxString& whichPath )
{
wxFileName auxDirectory( whichPath );
if( !auxDirectory.IsDir() )
auxDirectory = auxDirectory.GetPathWithSep();
wxString directory( auxDirectory.GetFullPath() );

globalXMLsPath = string( directory.mb_str() ) + PATH_SEP;
newXMLsPath = true;
}


void CutFilterDialog::EnableToolTab( int i )
{
EnableSingleTab( i );
ChangePageSelectionFromToolsOrderListToTabs( i );
}


void CutFilterDialog::UpdateOutputTraceName()
{
bool allowChangeOutputTrace = globalEnable();
if ( allowChangeOutputTrace )
{
enableOutputTraceWidgets( allowChangeOutputTrace );
setOutputName( allowChangeOutputTrace,
false,
std::string( fileBrowserButtonInputTrace->GetPath().mb_str() ) );
}
}


void CutFilterDialog::TransferTraceOptionsToWindow( TraceOptions *traceOptions, 
vector< string > &whichToolIDsOrder )
{
UpdateGuiXMLSectionFromFile( traceOptions, whichToolIDsOrder );
}


void CutFilterDialog::TransferXMLFileToWindow( const wxString& whichXMLFile )
{
if ( traceOptions != nullptr )
{
delete traceOptions;
traceOptions = TraceOptions::create( GetLocalKernel() );
}

#ifdef UNICODE
vector< string > toolIDsOrder = traceOptions->parseDoc( (char *)whichXMLFile.mb_str().data() );
#else
vector< string > toolIDsOrder = traceOptions->parseDoc( (char *)whichXMLFile.c_str() );
#endif

UpdateGuiXMLSectionFromFile( traceOptions, toolIDsOrder );
UpdateGlobalXMLPath( whichXMLFile );

UpdateOutputTraceName();
}



void CutFilterDialog::OnTextctrlCutFilterXmlTextUpdated( wxCommandEvent& event )
{
TransferXMLFileToWindow( fileBrowserButtonXML->GetPath() );
}




void CutFilterDialog::OnTextctrlCutFilterInputTraceTextUpdated( wxCommandEvent& event )
{
wxString tmpPath( fileBrowserButtonInputTrace->GetPath() );

wxFileName auxDirectory( tmpPath );
if( !auxDirectory.IsDir() )
auxDirectory = auxDirectory.GetPathWithSep();
wxString directory( auxDirectory.GetFullPath() );
outputPath = string( directory.mb_str() );

string auxSourceTrace = string( tmpPath.mb_str() ); 

bool localEnable = globalEnable( auxSourceTrace );
enableOutputTraceWidgets( localEnable );
setOutputName( localEnable, false, auxSourceTrace );
}




void CutFilterDialog::OnButtonCutterAllWindowClick( wxCommandEvent& event )
{
if( paraverMain::myParaverMain->GetCurrentTimeline() != nullptr )
{
textCutterBeginCut->SetValue(
LabelConstructor::timeLabel( paraverMain::myParaverMain->GetCurrentTimeline()->getWindowBeginTime(),
paraverMain::myParaverMain->GetCurrentTimeline()->getTrace()->getTimeUnit(),
ParaverConfig::getInstance()->getTimelinePrecision() ) );
textCutterEndCut->SetValue(
LabelConstructor::timeLabel( paraverMain::myParaverMain->GetCurrentTimeline()->getWindowEndTime(),
paraverMain::myParaverMain->GetCurrentTimeline()->getTrace()->getTimeUnit(),
ParaverConfig::getInstance()->getTimelinePrecision() ) );

radioCutterCutByTime->SetValue( true );
cutterByTimePreviouslyChecked = true;
}
else if( paraverMain::myParaverMain->GetCurrentHisto() != nullptr )
{
textCutterBeginCut->SetValue(
LabelConstructor::timeLabel( paraverMain::myParaverMain->GetCurrentHisto()->getBeginTime(),
paraverMain::myParaverMain->GetCurrentHisto()->getTrace()->getTimeUnit(),
ParaverConfig::getInstance()->getTimelinePrecision() ) );
textCutterEndCut->SetValue(
LabelConstructor::timeLabel( paraverMain::myParaverMain->GetCurrentHisto()->getEndTime(),
paraverMain::myParaverMain->GetCurrentHisto()->getTrace()->getTimeUnit(),
ParaverConfig::getInstance()->getTimelinePrecision() ) );

radioCutterCutByTime->SetValue( true );
cutterByTimePreviouslyChecked = true;
}
}




void CutFilterDialog::OnButtonCutterAllWindowUpdate( wxUpdateUIEvent& event )
{
buttonCutterAllWindow->Enable( paraverMain::myParaverMain->GetCurrentTimeline() != nullptr ||
paraverMain::myParaverMain->GetCurrentHisto() != nullptr );

}


void CutFilterDialog::swapTimeAndPercent()
{
Trace *tmpTrace = getTrace();

if ( tmpTrace != nullptr &&
!textCutterBeginCut->GetValue().IsEmpty() &&
!textCutterEndCut->GetValue().IsEmpty() )
{
bool byTime = radioCutterCutByTime->GetValue();

TTime auxBeginTime, auxEndTime;
readTimes( byTime, auxBeginTime, auxEndTime );

wxString beginValue;
wxString endValue;
TTime maxTraceTime = tmpTrace->getEndTime();
if ( byTime )
{
beginValue = formatTime( ( auxBeginTime / 100.0 ) * maxTraceTime );
endValue = formatTime( ( auxEndTime / 100.0 ) * maxTraceTime );
}
else
{
beginValue = formatPercent( 100.0 * ( auxBeginTime / maxTraceTime ) );
endValue = formatPercent( 100.0 * ( auxEndTime / maxTraceTime ) );
}

textCutterBeginCut->SetValue( beginValue );
textCutterEndCut->SetValue( endValue );
}
}





void CutFilterDialog::OnRadiobuttonCutterCutByTimeSelected( wxCommandEvent& event )
{
if ( !cutterByTimePreviouslyChecked )
swapTimeAndPercent(); 

cutterByTimePreviouslyChecked = true;
}




void CutFilterDialog::OnRadiobuttonCutterCutByPercentSelected( wxCommandEvent& event )
{
if ( cutterByTimePreviouslyChecked )
swapTimeAndPercent(); 

cutterByTimePreviouslyChecked = false;
}




void CutFilterDialog::OnKeyDown( wxKeyEvent& event )
{
if ( ( (wxKeyEvent&) event ).GetKeyCode() == WXK_ESCAPE )
{
if ( wxGetApp().GetGlobalTiming() )
wxGetApp().DeactivateGlobalTiming();
} 
}




void CutFilterDialog::OnCheckboxCutterKeepEventsUpdate( wxUpdateUIEvent& event )
{
event.Enable( checkCutterDontBreakStates->IsChecked() );
}

