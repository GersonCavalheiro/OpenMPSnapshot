

#pragma once





#include "wx/tglbtn.h"
#include "wx/statline.h"

#include <vector>

#include "paraverkerneltypes.h"
#include "window.h"
#include "loadedwindows.h"




class wxToggleButton;



#define ID_HISTOGRAMDIALOG 10061
#define ID_HISTOGRAM_CONTROLTIMELINETEXT 10065
#define ID_HISTOGRAM_CONTROLTIMELINEBUTTON 10001
#define ID_HISTOGRAM_CONTROLTIMELINEAUTOFIT 10067
#define ID_HISTOGRAM_CONTROLTIMELINEMIN 10068
#define ID_HISTOGRAM_CONTROLTIMELINEMAX 10005
#define ID_HISTOGRAM_CONTROLTIMELINEDELTA 10006
#define ID_HISTOGRAM_DATATIMELINETEXT 10003
#define ID_HISTOGRAM_DATATIMELINEBUTTON 10002
#define ID_HISTOGRAM_3DTIMELINETEXT 10004
#define ID_HISTOGRAM_3DTIMELINEBUTTON 10007
#define ID_HISTOGRAM_3DTIMELINEAUTOFIT 10066
#define ID_HISTOGRAM_3DTIMELINEMIN 10008
#define ID_HISTOGRAM_3DTIMELINEMAX 10009
#define ID_HISTOGRAM_3DTIMELINEDELTA 10010
#define ID_HISTOGRAM_BEGINTIME 10011
#define ID_HISTOGRAM_ENDTIME 10012
#define ID_RADIOBUTTON_ALLWINDOW 10063
#define ID_RADIOBUTTON_ALLTRACE 10000
#define ID_RADIOBUTTON_MANUAL 10064
#define ID_HISTOGRAM_BUTTONSELECT 10062
#define SYMBOL_HISTOGRAMDIALOG_STYLE wxCAPTION|wxRESIZE_BORDER|wxSYSTEM_MENU|wxCLOSE_BOX|wxTAB_TRAVERSAL
#define SYMBOL_HISTOGRAMDIALOG_TITLE _("Create Histogram")
#define SYMBOL_HISTOGRAMDIALOG_IDNAME ID_HISTOGRAMDIALOG
#define SYMBOL_HISTOGRAMDIALOG_SIZE wxSize(400, 300)
#define SYMBOL_HISTOGRAMDIALOG_POSITION wxDefaultPosition


enum class TTimeRangeSource
{
WINDOW_RANGE = 0,
TRACE_RANGE,
SELECTION_RANGE
};



class HistogramDialog: public wxDialog
{    
DECLARE_DYNAMIC_CLASS( HistogramDialog )
DECLARE_EVENT_TABLE()

public:
HistogramDialog();
HistogramDialog( wxWindow* parent, wxWindowID id = SYMBOL_HISTOGRAMDIALOG_IDNAME, const wxString& caption = SYMBOL_HISTOGRAMDIALOG_TITLE, const wxPoint& pos = SYMBOL_HISTOGRAMDIALOG_POSITION, const wxSize& size = SYMBOL_HISTOGRAMDIALOG_SIZE, long style = SYMBOL_HISTOGRAMDIALOG_STYLE );

bool Create( wxWindow* parent, wxWindowID id = SYMBOL_HISTOGRAMDIALOG_IDNAME, const wxString& caption = SYMBOL_HISTOGRAMDIALOG_TITLE, const wxPoint& pos = SYMBOL_HISTOGRAMDIALOG_POSITION, const wxSize& size = SYMBOL_HISTOGRAMDIALOG_SIZE, long style = SYMBOL_HISTOGRAMDIALOG_STYLE );

~HistogramDialog();

void Init();

void CreateControls();


void OnIdle( wxIdleEvent& event );

void OnHistogramControltimelinebuttonClick( wxCommandEvent& event );

void OnHistogramControltimelineautofitClick( wxCommandEvent& event );

void OnHistogramControltimelineautofitUpdate( wxUpdateUIEvent& event );

void OnHistogramDatatimelinebuttonClick( wxCommandEvent& event );

void OnHistogram3dtimelinebuttonClick( wxCommandEvent& event );

void OnHistogram3dtimelineautofitClick( wxCommandEvent& event );

void OnHistogram3dtimelineautofitUpdate( wxUpdateUIEvent& event );

void OnRadiobuttonAllwindowSelected( wxCommandEvent& event );

void OnRadiobuttonAllwindowUpdate( wxUpdateUIEvent& event );

void OnRadiobuttonAlltraceSelected( wxCommandEvent& event );

void OnRadiobuttonAlltraceUpdate( wxUpdateUIEvent& event );

void OnRadiobuttonManualSelected( wxCommandEvent& event );

void OnRadiobuttonManualUpdate( wxUpdateUIEvent& event );

void OnHistogramButtonselectClick( wxCommandEvent& event );

void OnHistogramButtonselectUpdate( wxUpdateUIEvent& event );

void OnCancelClick( wxCommandEvent& event );

void OnOkClick( wxCommandEvent& event );



bool GetControlTimelineAutofit() const { return controlTimelineAutofit ; }
void SetControlTimelineAutofit(bool value) { controlTimelineAutofit = value ; }

double GetControlTimelineDelta() const { return controlTimelineDelta ; }
void SetControlTimelineDelta(double value) { controlTimelineDelta = value ; }

double GetControlTimelineMax() const { return controlTimelineMax ; }
void SetControlTimelineMax(double value) { controlTimelineMax = value ; }

double GetControlTimelineMin() const { return controlTimelineMin ; }
void SetControlTimelineMin(double value) { controlTimelineMin = value ; }

Timeline * GetControlTimelineSelected() const { return controlTimelineSelected ; }
void SetControlTimelineSelected(Timeline * value) { controlTimelineSelected = value ; }

std::vector<TWindowID> GetControlTimelines() const { return controlTimelines ; }
void SetControlTimelines(std::vector<TWindowID> value) { controlTimelines = value ; }

Timeline * GetCurrentWindow() const { return currentWindow ; }
void SetCurrentWindow(Timeline * value) { currentWindow = value ; }

Timeline * GetDataTimelineSelected() const { return dataTimelineSelected ; }
void SetDataTimelineSelected(Timeline * value) { dataTimelineSelected = value ; }

std::vector<TWindowID> GetDataTimelines() const { return dataTimelines ; }
void SetDataTimelines(std::vector<TWindowID> value) { dataTimelines = value ; }

bool GetExtraControlTimelineAutofit() const { return extraControlTimelineAutofit ; }
void SetExtraControlTimelineAutofit(bool value) { extraControlTimelineAutofit = value ; }

double GetExtraControlTimelineDelta() const { return extraControlTimelineDelta ; }
void SetExtraControlTimelineDelta(double value) { extraControlTimelineDelta = value ; }

double GetExtraControlTimelineMax() const { return extraControlTimelineMax ; }
void SetExtraControlTimelineMax(double value) { extraControlTimelineMax = value ; }

double GetExtraControlTimelineMin() const { return extraControlTimelineMin ; }
void SetExtraControlTimelineMin(double value) { extraControlTimelineMin = value ; }

Timeline * GetExtraControlTimelineSelected() const { return extraControlTimelineSelected ; }
void SetExtraControlTimelineSelected(Timeline * value) { extraControlTimelineSelected = value ; }

std::vector<TWindowID> GetExtraControlTimelines() const { return extraControlTimelines ; }
void SetExtraControlTimelines(std::vector<TWindowID> value) { extraControlTimelines = value ; }

std::vector<std::pair<TRecordTime,TRecordTime> > GetTimeRange() const { return timeRange ; }
void SetTimeRange(std::vector<std::pair<TRecordTime,TRecordTime> > value) { timeRange = value ; }

bool GetWaitingGlobalTiming() const { return waitingGlobalTiming ; }
void SetWaitingGlobalTiming(bool value) { waitingGlobalTiming = value ; }

wxBitmap GetBitmapResource( const wxString& name );

wxIcon GetIconResource( const wxString& name );

bool TransferDataToWindow( Timeline *current );
bool TransferDataFromWindow();

static bool ShowToolTips();

wxTextCtrl* txtControlTimelines;
wxBitmapButton* buttonControlTimelines;
wxToggleButton* buttonControlTimelineAutoFit;
wxStaticText* labelControlTimelineMin;
wxTextCtrl* txtControlTimelineMin;
wxStaticText* labelControlTimelineMax;
wxTextCtrl* txtControlTimelineMax;
wxStaticText* labelControlTimelineDelta;
wxTextCtrl* txtControlTimelineDelta;
wxTextCtrl* txtDataTimelines;
wxBitmapButton* buttonDataTimelines;
wxTextCtrl* txt3DTimelines;
wxBitmapButton* button3DTimelines;
wxToggleButton* button3DTimelineAutoFit;
wxStaticText* label3DTimelineMin;
wxTextCtrl* txt3DTimelineMin;
wxStaticText* label3DTimelineMax;
wxTextCtrl* txt3DTimelineMax;
wxStaticText* label3DTimelineDelta;
wxTextCtrl* txt3DTimelineDelta;
wxTextCtrl* txtBeginTime;
wxTextCtrl* txtEndTime;
wxRadioButton* radioAllWindow;
wxRadioButton* radioAllTrace;
wxRadioButton* radioManual;
wxButton* buttonSelect;
private:
bool controlTimelineAutofit;
double controlTimelineDelta;
double controlTimelineMax;
double controlTimelineMin;
Timeline * controlTimelineSelected;
std::vector<TWindowID> controlTimelines;
Timeline * currentWindow;
Timeline * dataTimelineSelected;
std::vector<TWindowID> dataTimelines;
bool extraControlTimelineAutofit;
double extraControlTimelineDelta;
double extraControlTimelineMax;
double extraControlTimelineMin;
Timeline * extraControlTimelineSelected;
std::vector<TWindowID> extraControlTimelines;
std::vector<std::pair<TRecordTime,TRecordTime> > timeRange;
bool waitingGlobalTiming;

wxString formatNumber( double value );

TSemanticValue computeDelta( TSemanticValue min, TSemanticValue max );
void computeColumns( Timeline *timeline, TSemanticValue &min, TSemanticValue &max, TSemanticValue &delta );
void updateControlTimelineAutofit();
void updateExtraControlTimelineAutofit();
PRV_UINT32 fillList( Timeline *current, std::vector< TWindowID > listTimelines, wxChoice *listWidget );
void enable3DFields( bool autofit );
};


