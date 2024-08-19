


#pragma once




#include "wx/statline.h"
#include "cfg.h"
#include "window.h"
#include "histogram.h"
#include "cfgs4d.h"






#define ID_SAVECONFIGURATIONDIALOG 10012
#define ID_CHOICE_TRACE_SELECTOR 10191
#define ID_LISTTIMELINES 10013
#define ID_BUTTON_SET_ALL_TIMELINES 10188
#define ID_BUTTON_UNSET_ALL_TIMELINES 10189
#define ID_LISTHISTOGRAMS 10014
#define ID_BUTTON_SET_ALL_HISTOGRAMS 10000
#define ID_BUTTON_UNSET_ALL_HISTOGRAMS 10001
#define ID_CHECKBEGIN 10016
#define ID_CHECKEND 10017
#define ID_CHECKSEMANTIC 10018
#define ID_RADIOALLTRACE 10019
#define ID_RADIOALLWINDOW 10020
#define ID_CHECKGRADIENT 10022
#define ID_TEXTDESCRIPTION 10015
#define ID_CHECKBOX_SAVE_BASIC_MODE 10190
#define SYMBOL_SAVECONFIGURATIONDIALOG_STYLE wxCAPTION|wxRESIZE_BORDER|wxSYSTEM_MENU|wxSTAY_ON_TOP|wxCLOSE_BOX|wxTAB_TRAVERSAL
#define SYMBOL_SAVECONFIGURATIONDIALOG_TITLE _("Save configuration")
#define SYMBOL_SAVECONFIGURATIONDIALOG_IDNAME ID_SAVECONFIGURATIONDIALOG
#define SYMBOL_SAVECONFIGURATIONDIALOG_SIZE wxDefaultSize
#define SYMBOL_SAVECONFIGURATIONDIALOG_POSITION wxDefaultPosition




class SaveConfigurationDialog: public wxDialog
{    
DECLARE_DYNAMIC_CLASS( SaveConfigurationDialog )
DECLARE_EVENT_TABLE()

public:
SaveConfigurationDialog();
SaveConfigurationDialog( wxWindow* parent, wxWindowID id = SYMBOL_SAVECONFIGURATIONDIALOG_IDNAME, const wxString& caption = SYMBOL_SAVECONFIGURATIONDIALOG_TITLE, const wxPoint& pos = SYMBOL_SAVECONFIGURATIONDIALOG_POSITION, const wxSize& size = SYMBOL_SAVECONFIGURATIONDIALOG_SIZE, long style = SYMBOL_SAVECONFIGURATIONDIALOG_STYLE );

bool Create( wxWindow* parent, wxWindowID id = SYMBOL_SAVECONFIGURATIONDIALOG_IDNAME, const wxString& caption = SYMBOL_SAVECONFIGURATIONDIALOG_TITLE, const wxPoint& pos = SYMBOL_SAVECONFIGURATIONDIALOG_POSITION, const wxSize& size = SYMBOL_SAVECONFIGURATIONDIALOG_SIZE, long style = SYMBOL_SAVECONFIGURATIONDIALOG_STYLE );

~SaveConfigurationDialog();

void Init();

void CreateControls();


void OnChoiceTraceSelectorSelected( wxCommandEvent& event );

void OnButtonSetAllTimelinesClick( wxCommandEvent& event );

void OnButtonUnsetAllTimelinesClick( wxCommandEvent& event );

void OnButtonSetAllHistogramsClick( wxCommandEvent& event );

void OnButtonUnsetAllHistogramsClick( wxCommandEvent& event );

void OnSaveClick( wxCommandEvent& event );



std::vector<Histogram *> GetHistograms() const { return histograms ; }
void SetHistograms(std::vector<Histogram *> value) { histograms = value ; }

Trace * GetInitialTrace() const { return initialTrace ; }
void SetInitialTrace(Trace * value) { initialTrace = value ; }

SaveOptions GetOptions() const { return options ; }
void SetOptions(SaveOptions value) { options = value ; }

std::vector< Histogram * > GetSelectedHistograms() const { return selectedHistograms ; }
void SetSelectedHistograms(std::vector< Histogram * > value) { selectedHistograms = value ; }

std::vector< Timeline * > GetSelectedTimelines() const { return selectedTimelines ; }
void SetSelectedTimelines(std::vector< Timeline * > value) { selectedTimelines = value ; }

std::vector<Timeline *> GetTimelines() const { return timelines ; }
void SetTimelines(std::vector<Timeline *> value) { timelines = value ; }

wxBitmap GetBitmapResource( const wxString& name );

wxIcon GetIconResource( const wxString& name );

const CFGS4DLinkedPropertiesManager& getLinkedPropertiesManager() const;

static bool ShowToolTips();

bool TransferDataToWindow();
bool TransferDataFromWindow();

wxChoice* choiceTraceSelector;
wxCheckListBox* listTimelines;
wxButton* buttonSetAllTimelines;
wxButton* buttonUnsetAllTimelines;
wxCheckListBox* listHistograms;
wxButton* buttonSetAllHistograms;
wxButton* buttonUnsetAllHistograms;
wxCheckBox* optRelativeBegin;
wxCheckBox* optRelativeEnd;
wxCheckBox* optComputeSemantic;
wxRadioButton* radioAllTrace;
wxRadioButton* radioAllWindow;
wxCheckBox* optComputeGradient;
wxTextCtrl* textDescription;
wxCheckBox* checkboxSaveCFGBasicMode;
private:
std::vector<Histogram *> histograms;
Trace * initialTrace;
SaveOptions options;
std::vector< Histogram * > selectedHistograms;
std::vector< Timeline * > selectedTimelines;
std::vector<Timeline *> timelines;
std::vector< std::string > traces;
std::vector< Timeline * > displayedTimelines;
std::vector< Histogram * > displayedHistograms;
CFGS4DLinkedPropertiesManager linkedProperties;

};
