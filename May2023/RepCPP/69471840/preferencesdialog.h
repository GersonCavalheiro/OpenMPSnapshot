

#pragma once





#include <map>
#include <string>

#include "wx/propdlg.h"
#include "wx/spinctrl.h"
#include "filebrowserbutton.h"
#include "wx/clrpicker.h"
#include "wx/statline.h"

#include "wx/bmpbuttn.h"
#include "wx/filedlg.h"
#include "wx/radiobut.h"

#include "paraverconfig.h"
#include "workspace.h"



class wxSpinCtrl;
class DirBrowserButton;
class wxColourPickerCtrl;
class FileBrowserButton;
class wxCheckBox;
class wxListBox;


#define ID_PREFERENCESDIALOG 10069
#define ID_PREFERENCES_GLOBAL 10073
#define ID_PREFERENCES_GLOBAL_FILLGAPS 10085
#define ID_PREFERENCES_GLOBAL_FULLTRACEPATH 10010
#define ID_TEXTCTRL_MAXIMUM_LOADABLE_TRACE_SIZE 10156
#define ID_TEXTCTRL_DEFAULT_TRACE 10220
#define ID_BUTTON_DIR_BROWSER_TRACE 10238
#define ID_TEXTCTRL_DEFAULT_CFGS 10089
#define ID_DIRBROWSERBUTTON_DEFAULT_CFGS 10243
#define ID_TEXTCTRL_DEFAULT_XMLS 10239
#define ID_BUTTON_DEFAULT_XMLS 10242
#define ID_TEXTCTRL_DEFAULT_TUTORIALS 10240
#define ID_DIRBROWSERBUTTON_DEFAULT_TUTORIALS 10244
#define ID_TEXTCTRL_DEFAULT_TMP 10241
#define ID_DIRBROWSERBUTTON_DEFAULT_TMP 10245
#define ID_PREFERENCES_GLOBAL_SINGLE_INSTANCE 10158
#define ID_PREFERENCES_GLOBAL_TIME_SESSION 10168
#define ID_GLOBAL_ASK_FOR_PREV_SESSION 10039
#define ID_HELP_CONTENTS_IN_BROWSER 10159
#define ID_DISABLE_TIMELINE_ZOOM_MOUSE_WHEEL 10043
#define ID_PREFERENCES_TIMELINE 10072
#define ID_PREFERENCES_TIMELINE_CREATE_DIALOG_KEEP_SYNC_GROUP 10033
#define ID_PREFERENCES_TIMELINE_NAME_PREFIX 10098
#define ID_PREFERENCES_TIMELINE_NAME_FULL 10099
#define ID_PREFERENCES_TIMELINE_COMMUNICATION_LINES 10090
#define ID_PREFERENCES_TIMELINE_EVENT_LINES 10088
#define ID_PREFERENCES_SEMANTIC_SCALE_MIN_AT_ZERO 10044
#define ID_PREFERENCES_TIMELINE_COLOR 10086
#define ID_PREFERENCES_TIMELINE_GRADIENT 10015
#define ID_PREFERENCES_TIMELINE_DRAWMODE_TIME 10012
#define ID_PREFERENCES_TIMELINE_DRAWMODE_OBJECTS 10013
#define ID_PREFERENCES_TIMELINE_PIXEL_SIZE 10016
#define ID_PREFERENCES_TIMELINE_LABELS 10208
#define ID_PREFERENCES_TIMELINE_OBJECT_AXIS 10254
#define ID_CHECKBOX_TIMELINE_WW_SEMANTIC 10093
#define ID_CHECKBOX_TIMELINE_WW_EVENTS 10094
#define ID_CHECKBOX_TIMELINE_WW_COMMUNICATIONS 10095
#define ID_CHECKBOX_TIMELINE_WW_PREVIOUS_NEXT 10096
#define ID_CHECKBOX_TIMELINE_WW_TEXT 10097
#define ID_PREFERENCES_TIMELINE_WW_PRECISION 10000
#define ID_PREFERENCES_TIMELINE_WW_EVENT_PIXELS 10167
#define ID_PREFERENCES_TIMELINE_SAVE_AS_IMAGE 10014
#define ID_PREFERENCES_TIMELINE_SAVE_AS_TEXT 10017
#define ID_PREFERENCES_HISTOGRAM 10071
#define ID_PREFERENCES_HISTOGRAM_SKIP_CREATE_DIALOG 10033
#define ID_PREFERENCES_HISTOGRAM_CREATE_DIALOG_KEEP_SYNC_GROUP 10033
#define ID_PREFERENCES_HISTOGRAM_NAME_PREFIX 10018
#define ID_PREFERENCES_HISTOGRAM_NAME_FULL 10019
#define ID_PREFERENCES_HISTOGRAM_MATRIX_ZOOM 10092
#define ID_PREFERENCES_HISTOGRAM_MATRIX_HORIZONTAL 10023
#define ID_PREFERENCES_HISTOGRAM_MATRIX_HIDE_EMPTY 10024
#define ID_PREFERENCES_HISTOGRAM_MATRIX_GRADIENT 10022
#define ID_PREFERENCES_HISTOGRAM_MATRIX_LABELS_COLOR 10102
#define ID_PREFERENCES_HISTOGRAM_MATRIX_GRADIENT_FUNCTION 10020
#define ID_PREFERENCES_HISTOGRAM_MATRIX_DRAWMODE_SEMANTIC 10021
#define ID_PREFERENCES_HISTOGRAM_MATRIX_DRAWMODE_OBJECTS 10025
#define ID_PREFERENCES_HISTOGRAM_SCIENTIFIC_NOTATION 10026
#define ID_PREFERENCES_HISTOGRAM_THOUSANDS_SEPARATOR 10027
#define ID_PREFERENCES_HISTOGRAM_SHOW_UNITS 10028
#define ID_PREFERENCES_HISTOGRAM_PRECISION 10074
#define ID_PREFERENCES_HISTOGRAM_AUTOFIT_CONTROL 10030
#define ID_PREFERENCES_HISTOGRAM_AUTOFIT_CONTROL_ZERO 10041
#define ID_PREFERENCES_HISTOGRAM_AUTOFIT_3D 10030
#define ID_PREFERENCES_HISTOGRAM_AUTOFIT_DATA_GRADIENT 10029
#define ID_PREFERENCES_HISTOGRAM_NUMCOLUMNS 10075
#define ID_PREFERENCES_HISTOGRAM_SAVE_IMAGE_FORMAT 10031
#define ID_PREFERENCES_HISTOGRAM_SAVE_TXT_FORMAT 10032
#define ID_PREFERENCES_COLOR 10086
#define ID_COLOURPICKER_BACKGROUND 10002
#define ID_COLOURPICKER_AXIS 10001
#define ID_COLOURPICKER_ZERO 10104
#define ID_COLOURPICKER_PUNCTUAL 10034
#define ID_COLOURPICKER_LOGICAL 10007
#define ID_COLOURPICKER_PHYSICAL 10008
#define ID_BUTTON_DEFAULT_TIMELINE 10087
#define ID_COLOURPICKER_GRADBEGIN 10003
#define ID_COLOURPICKER_GRADEND 10004
#define ID_COLOURPICKER_NEGATIVE_GRADBEGIN 10036
#define ID_COLOURPICKER_NEGATIVE_GRADEND 10035
#define ID_COLOURPICKER_GRADLOW 10005
#define ID_COLOURPICKER_GRADTOP 10006
#define ID_BUTTON_DEFAULT_GRADIENT 10009
#define ID_PREFERENCES_WORKSPACES 10269
#define ID_CHECKBOX_DISCARDED_SUBMENU 10045
#define ID_LISTBOX_WORKSPACES 10270
#define ID_BUTTON_WORKSPACES_ADD 10271
#define ID_BUTTON_WORKSPACES_DELETE 10272
#define ID_BUTTON_WORKSPACES_UP 10273
#define ID_BUTTON_WORKSPACES_DOWN 10274
#define ID_BUTTON_WORKSPACES_IMPORT 10276
#define ID_BUTTON_WORKSPACES_EXPORT 10275
#define ID_TEXT_WORKSPACE_NAME 10275
#define ID_RADIOSTATES 10037
#define ID_RADIOEVENTYPES 10038
#define ID_TEXT_WORKSPACE_AUTOTYPES 10011
#define ID_LISTBOX_HINTS_WORKSPACE 10276
#define ID_BUTTON_HINT_ADD 10277
#define ID_BUTTON_HINT_DELETE 10278
#define ID_BITMAP_HINT_UP 10279
#define ID_BUTTON_HINT_DOWN 10280
#define ID_TEXTCTRL_WORKSPACE_HINT_PATH 10283
#define ID_FILE_BUTTON_WORKSPACE_HINT_PATH 10282
#define ID_TEXTCTRL_WORKSPACE_HINT_DESCRIPTION 10281
#define ID_PREFERENCES_EXTERNAL 10040
#define ID_LISTBOX_TEXT_EDITORS 10042
#define ID_BUTTON_TXT_ADD 10343
#define ID_BUTTON_TXT_DEL 10344
#define ID_BUTTON_TXT_UP 10345
#define ID_BUTTON_TXT_DOWN 10346
#define ID_LISTBOX_PDF_READERS 10048
#define ID_BUTTON_PDF_ADD 10349
#define ID_BUTTON_PDF_DEL 10350
#define ID_BUTTON_PDF_UP 10351
#define ID_BUTTON_PDF_DOWN 10352
#define ID_PREFERENCES_FILTERS 10070
#define SYMBOL_PREFERENCESDIALOG_STYLE wxCAPTION|wxRESIZE_BORDER|wxSYSTEM_MENU|wxCLOSE_BOX
#define SYMBOL_PREFERENCESDIALOG_TITLE _("Preferences")
#define SYMBOL_PREFERENCESDIALOG_IDNAME ID_PREFERENCESDIALOG
#define SYMBOL_PREFERENCESDIALOG_SIZE wxSize(400, 300)
#define SYMBOL_PREFERENCESDIALOG_POSITION wxDefaultPosition




class PreferencesDialog: public wxPropertySheetDialog
{    
DECLARE_DYNAMIC_CLASS( PreferencesDialog )
DECLARE_EVENT_TABLE()

public:
enum class ItemCheck { ITEM_SHOW_UNITS, ITEM_THOUSAND_SEPARATOR };

PreferencesDialog();
PreferencesDialog( wxWindow* parent, wxWindowID id = SYMBOL_PREFERENCESDIALOG_IDNAME, const wxString& caption = SYMBOL_PREFERENCESDIALOG_TITLE, const wxPoint& pos = SYMBOL_PREFERENCESDIALOG_POSITION, const wxSize& size = SYMBOL_PREFERENCESDIALOG_SIZE, long style = SYMBOL_PREFERENCESDIALOG_STYLE );

bool Create( wxWindow* parent, wxWindowID id = SYMBOL_PREFERENCESDIALOG_IDNAME, const wxString& caption = SYMBOL_PREFERENCESDIALOG_TITLE, const wxPoint& pos = SYMBOL_PREFERENCESDIALOG_POSITION, const wxSize& size = SYMBOL_PREFERENCESDIALOG_SIZE, long style = SYMBOL_PREFERENCESDIALOG_STYLE );

~PreferencesDialog();

void Init();

void CreateControls();


void OnPreferencesGlobalTimeSessionUpdated( wxSpinEvent& event );

void OnColourpickerBackgroundColourPickerChanged( wxColourPickerEvent& event );

void OnColourpickerZeroUpdate( wxUpdateUIEvent& event );

void OnButtonDefaultTimelineClick( wxCommandEvent& event );

void OnButtonDefaultGradientClick( wxCommandEvent& event );

void OnListboxWorkspacesSelected( wxCommandEvent& event );

void OnButtonWorkspacesAddClick( wxCommandEvent& event );

void OnButtonWorkspacesDeleteClick( wxCommandEvent& event );

void OnButtonWorkspacesDeleteUpdate( wxUpdateUIEvent& event );

void OnButtonWorkspacesUpClick( wxCommandEvent& event );

void OnButtonWorkspacesUpUpdate( wxUpdateUIEvent& event );

void OnButtonWorkspacesDownClick( wxCommandEvent& event );

void OnButtonWorkspacesDownUpdate( wxUpdateUIEvent& event );

void OnButtonWorkspacesImportClick( wxCommandEvent& event );

void OnButtonWorkspacesImportUpdate( wxUpdateUIEvent& event );

void OnButtonWorkspacesExportClick( wxCommandEvent& event );

void OnButtonWorkspacesExportUpdate( wxUpdateUIEvent& event );

void OnTextWorkspaceNameTextUpdated( wxCommandEvent& event );

void OnTextWorkspaceNameEnter( wxCommandEvent& event );

void OnTextWorkspaceNameUpdate( wxUpdateUIEvent& event );

void OnTextWorkspaceNameKillFocus( wxFocusEvent& event );

void OnRadiostatesSelected( wxCommandEvent& event );

void OnRadiostatesUpdate( wxUpdateUIEvent& event );

void OnRadioeventypesSelected( wxCommandEvent& event );

void OnRadioeventypesUpdate( wxUpdateUIEvent& event );

void OnTextWorkspaceAutotypesTextUpdated( wxCommandEvent& event );

void OnTextWorkspaceAutotypesUpdate( wxUpdateUIEvent& event );

void OnListboxHintsWorkspaceSelected( wxCommandEvent& event );

void OnListboxHintsWorkspaceUpdate( wxUpdateUIEvent& event );

void OnButtonHintAddClick( wxCommandEvent& event );

void OnButtonHintAddUpdate( wxUpdateUIEvent& event );

void OnButtonHintDeleteClick( wxCommandEvent& event );

void OnButtonHintDeleteUpdate( wxUpdateUIEvent& event );

void OnBitmapHintUpClick( wxCommandEvent& event );

void OnBitmapHintUpUpdate( wxUpdateUIEvent& event );

void OnButtonHintDownClick( wxCommandEvent& event );

void OnButtonHintDownUpdate( wxUpdateUIEvent& event );

void OnTextctrlWorkspaceHintPathTextUpdated( wxCommandEvent& event );

void OnTextctrlWorkspaceHintPathUpdate( wxUpdateUIEvent& event );

void OnFileButtonWorkspaceHintPathUpdate( wxUpdateUIEvent& event );

void OnTextctrlWorkspaceHintDescriptionTextUpdated( wxCommandEvent& event );

void OnTextctrlWorkspaceHintDescriptionUpdate( wxUpdateUIEvent& event );

void OnListboxTextEditorsSelected( wxCommandEvent& event );

void OnButtonTxtAddClick( wxCommandEvent& event );

void OnButtonTxtDelClick( wxCommandEvent& event );

void OnButtonTxtDelUpdate( wxUpdateUIEvent& event );

void OnButtonTxtUpClick( wxCommandEvent& event );

void OnButtonTxtUpUpdate( wxUpdateUIEvent& event );

void OnButtonTxtDownClick( wxCommandEvent& event );

void OnButtonTxtDownUpdate( wxUpdateUIEvent& event );

void OnListboxPdfReadersSelected( wxCommandEvent& event );

void OnButtonPdfAddClick( wxCommandEvent& event );

void OnButtonPdfDelClick( wxCommandEvent& event );

void OnButtonPdfDelUpdate( wxUpdateUIEvent& event );

void OnButtonPdfUpClick( wxCommandEvent& event );

void OnButtonPdfUpUpdate( wxUpdateUIEvent& event );

void OnButtonPdfDownClick( wxCommandEvent& event );

void OnButtonPdfDownUpdate( wxUpdateUIEvent& event );



bool GetAskForPrevSessionLoad() const { return askForPrevSessionLoad ; }
void SetAskForPrevSessionLoad(bool value) { askForPrevSessionLoad = value ; }

std::string GetCfgsPath() const { return cfgsPath ; }
void SetCfgsPath(std::string value) { cfgsPath = value ; }

bool GetColorUseZero() const { return colorUseZero ; }
void SetColorUseZero(bool value) { colorUseZero = value ; }

bool GetDisableTimelineZoomMouseWheel() const { return disableTimelineZoomMouseWheel ; }
void SetDisableTimelineZoomMouseWheel(bool value) { disableTimelineZoomMouseWheel = value ; }

std::string GetFiltersXMLPath() const { return filtersXMLPath ; }
void SetFiltersXMLPath(std::string value) { filtersXMLPath = value ; }

bool GetGlobalFillStateGaps() const { return globalFillStateGaps ; }
void SetGlobalFillStateGaps(bool value) { globalFillStateGaps = value ; }

bool GetGlobalFullTracePath() const { return globalFullTracePath ; }
void SetGlobalFullTracePath(bool value) { globalFullTracePath = value ; }

rgb GetGradientColourBegin() const { return gradientColourBegin ; }
void SetGradientColourBegin(rgb value) { gradientColourBegin = value ; }

rgb GetGradientColourEnd() const { return gradientColourEnd ; }
void SetGradientColourEnd(rgb value) { gradientColourEnd = value ; }

rgb GetGradientColourLow() const { return gradientColourLow ; }
void SetGradientColourLow(rgb value) { gradientColourLow = value ; }

rgb GetGradientColourNegativeBegin() const { return gradientColourNegativeBegin ; }
void SetGradientColourNegativeBegin(rgb value) { gradientColourNegativeBegin = value ; }

rgb GetGradientColourNegativeEnd() const { return gradientColourNegativeEnd ; }
void SetGradientColourNegativeEnd(rgb value) { gradientColourNegativeEnd = value ; }

rgb GetGradientColourTop() const { return gradientColourTop ; }
void SetGradientColourTop(rgb value) { gradientColourTop = value ; }

bool GetHelpContentsUsesBrowser() const { return helpContentsUsesBrowser ; }
void SetHelpContentsUsesBrowser(bool value) { helpContentsUsesBrowser = value ; }

bool GetHistogramAutofit3DScale() const { return histogramAutofit3DScale ; }
void SetHistogramAutofit3DScale(bool value) { histogramAutofit3DScale = value ; }

bool GetHistogramAutofitControlScale() const { return histogramAutofitControlScale ; }
void SetHistogramAutofitControlScale(bool value) { histogramAutofitControlScale = value ; }

bool GetHistogramAutofitControlScaleZero() const { return histogramAutofitControlScaleZero ; }
void SetHistogramAutofitControlScaleZero(bool value) { histogramAutofitControlScaleZero = value ; }

bool GetHistogramAutofitDataGradient() const { return histogramAutofitDataGradient ; }
void SetHistogramAutofitDataGradient(bool value) { histogramAutofitDataGradient = value ; }

PRV_UINT32 GetHistogramDrawmodeObjects() const { return histogramDrawmodeObjects ; }
void SetHistogramDrawmodeObjects(PRV_UINT32 value) { histogramDrawmodeObjects = value ; }

PRV_UINT32 GetHistogramDrawmodeSemantic() const { return histogramDrawmodeSemantic ; }
void SetHistogramDrawmodeSemantic(PRV_UINT32 value) { histogramDrawmodeSemantic = value ; }

PRV_UINT32 GetHistogramGradientFunction() const { return histogramGradientFunction ; }
void SetHistogramGradientFunction(PRV_UINT32 value) { histogramGradientFunction = value ; }

bool GetHistogramHideEmpty() const { return histogramHideEmpty ; }
void SetHistogramHideEmpty(bool value) { histogramHideEmpty = value ; }

bool GetHistogramHorizontal() const { return histogramHorizontal ; }
void SetHistogramHorizontal(bool value) { histogramHorizontal = value ; }

bool GetHistogramKeepSyncGroupClone() const { return histogramKeepSyncGroupClone ; }
void SetHistogramKeepSyncGroupClone(bool value) { histogramKeepSyncGroupClone = value ; }

bool GetHistogramLabelsColor() const { return histogramLabelsColor ; }
void SetHistogramLabelsColor(bool value) { histogramLabelsColor = value ; }

THistogramColumn GetHistogramMaxNumColumns() const { return histogramMaxNumColumns ; }
void SetHistogramMaxNumColumns(THistogramColumn value) { histogramMaxNumColumns = value ; }

PRV_UINT32 GetHistogramMaxPrecision() const { return histogramMaxPrecision ; }
void SetHistogramMaxPrecision(PRV_UINT32 value) { histogramMaxPrecision = value ; }

std::string GetHistogramNameFormatFull() const { return histogramNameFormatFull ; }
void SetHistogramNameFormatFull(std::string value) { histogramNameFormatFull = value ; }

std::string GetHistogramNameFormatPrefix() const { return histogramNameFormatPrefix ; }
void SetHistogramNameFormatPrefix(std::string value) { histogramNameFormatPrefix = value ; }

THistogramColumn GetHistogramNumColumns() const { return histogramNumColumns ; }
void SetHistogramNumColumns(THistogramColumn value) { histogramNumColumns = value ; }

PRV_UINT32 GetHistogramPrecision() const { return histogramPrecision ; }
void SetHistogramPrecision(PRV_UINT32 value) { histogramPrecision = value ; }

PRV_UINT32 GetHistogramSaveImageFormat() const { return histogramSaveImageFormat ; }
void SetHistogramSaveImageFormat(PRV_UINT32 value) { histogramSaveImageFormat = value ; }

PRV_UINT32 GetHistogramSaveTextFormat() const { return histogramSaveTextFormat ; }
void SetHistogramSaveTextFormat(PRV_UINT32 value) { histogramSaveTextFormat = value ; }

bool GetHistogramScientificNotation() const { return histogramScientificNotation ; }
void SetHistogramScientificNotation(bool value) { histogramScientificNotation = value ; }

bool GetHistogramShowGradient() const { return histogramShowGradient ; }
void SetHistogramShowGradient(bool value) { histogramShowGradient = value ; }

bool GetHistogramShowUnits() const { return histogramShowUnits ; }
void SetHistogramShowUnits(bool value) { histogramShowUnits = value ; }

bool GetHistogramSkipCreateDialog() const { return histogramSkipCreateDialog ; }
void SetHistogramSkipCreateDialog(bool value) { histogramSkipCreateDialog = value ; }

bool GetHistogramThousandSeparator() const { return histogramThousandSeparator ; }
void SetHistogramThousandSeparator(bool value) { histogramThousandSeparator = value ; }

bool GetHistogramZoom() const { return histogramZoom ; }
void SetHistogramZoom(bool value) { histogramZoom = value ; }

float GetMaximumTraceSize() const { return maximumTraceSize ; }
void SetMaximumTraceSize(float value) { maximumTraceSize = value ; }

wxArrayString GetGlobalExternalPDFReaders() const { return pdfReaderOptions ; }
void SetGlobalExternalPDFReaders(wxArrayString value) { pdfReaderOptions = value ; }

PRV_UINT16 GetSessionSaveTime() const { return sessionSaveTime ; }
void SetSessionSaveTime(PRV_UINT16 value) { sessionSaveTime = value ; }

bool GetSingleInstance() const { return singleInstance ; }
void SetSingleInstance(bool value) { singleInstance = value ; }

wxArrayString GetGlobalExternalTextEditors() const { return textEditorOptions ; }
void SetGlobalExternalTextEditors(wxArrayString value) { textEditorOptions = value ; }

PRV_UINT32 GetTimelineColor() const { return timelineColor ; }
void SetTimelineColor(PRV_UINT32 value) { timelineColor = value ; }

rgb GetTimelineColourAxis() const { return timelineColourAxis ; }
void SetTimelineColourAxis(rgb value) { timelineColourAxis = value ; }

rgb GetTimelineColourBackground() const { return timelineColourBackground ; }
void SetTimelineColourBackground(rgb value) { timelineColourBackground = value ; }

rgb GetTimelineColourLogical() const { return timelineColourLogical ; }
void SetTimelineColourLogical(rgb value) { timelineColourLogical = value ; }

rgb GetTimelineColourPhysical() const { return timelineColourPhysical ; }
void SetTimelineColourPhysical(rgb value) { timelineColourPhysical = value ; }

rgb GetTimelineColourPunctual() const { return timelineColourPunctual ; }
void SetTimelineColourPunctual(rgb value) { timelineColourPunctual = value ; }

rgb GetTimelineColourZero() const { return timelineColourZero ; }
void SetTimelineColourZero(rgb value) { timelineColourZero = value ; }

bool GetTimelineCommunicationLines() const { return timelineCommunicationLines ; }
void SetTimelineCommunicationLines(bool value) { timelineCommunicationLines = value ; }

PRV_UINT32 GetTimelineDrawmodeObjects() const { return timelineDrawmodeObjects ; }
void SetTimelineDrawmodeObjects(PRV_UINT32 value) { timelineDrawmodeObjects = value ; }

PRV_UINT32 GetTimelineDrawmodeTime() const { return timelineDrawmodeTime ; }
void SetTimelineDrawmodeTime(PRV_UINT32 value) { timelineDrawmodeTime = value ; }

bool GetTimelineEventLines() const { return timelineEventLines ; }
void SetTimelineEventLines(bool value) { timelineEventLines = value ; }

PRV_UINT32 GetTimelineGradientFunction() const { return timelineGradientFunction ; }
void SetTimelineGradientFunction(PRV_UINT32 value) { timelineGradientFunction = value ; }

bool GetTimelineKeepSyncGroupClone() const { return timelineKeepSyncGroupClone ; }
void SetTimelineKeepSyncGroupClone(bool value) { timelineKeepSyncGroupClone = value ; }

std::string GetTimelineNameFormatFull() const { return timelineNameFormatFull ; }
void SetTimelineNameFormatFull(std::string value) { timelineNameFormatFull = value ; }

std::string GetTimelineNameFormatPrefix() const { return timelineNameFormatPrefix ; }
void SetTimelineNameFormatPrefix(std::string value) { timelineNameFormatPrefix = value ; }

PRV_UINT32 GetTimelineObjectAxis() const { return timelineObjectAxis ; }
void SetTimelineObjectAxis(PRV_UINT32 value) { timelineObjectAxis = value ; }

PRV_UINT32 GetTimelineObjectLabels() const { return timelineObjectLabels ; }
void SetTimelineObjectLabels(PRV_UINT32 value) { timelineObjectLabels = value ; }

PRV_UINT32 GetTimelinePixelSize() const { return timelinePixelSize ; }
void SetTimelinePixelSize(PRV_UINT32 value) { timelinePixelSize = value ; }

PRV_UINT32 GetTimelineSaveImageFormat() const { return timelineSaveImageFormat ; }
void SetTimelineSaveImageFormat(PRV_UINT32 value) { timelineSaveImageFormat = value ; }

PRV_UINT32 GetTimelineSaveTextFormat() const { return timelineSaveTextFormat ; }
void SetTimelineSaveTextFormat(PRV_UINT32 value) { timelineSaveTextFormat = value ; }

bool GetTimelineSemanticScaleMinAtZero() const { return timelineSemanticScaleMinAtZero ; }
void SetTimelineSemanticScaleMinAtZero(bool value) { timelineSemanticScaleMinAtZero = value ; }

bool GetTimelineWWCommunications() const { return timelineWWCommunications ; }
void SetTimelineWWCommunications(bool value) { timelineWWCommunications = value ; }

PRV_INT16 GetTimelineWWEventPixels() const { return timelineWWEventPixels ; }
void SetTimelineWWEventPixels(PRV_INT16 value) { timelineWWEventPixels = value ; }

bool GetTimelineWWEvents() const { return timelineWWEvents ; }
void SetTimelineWWEvents(bool value) { timelineWWEvents = value ; }

PRV_UINT32 GetTimelineWWPrecision() const { return timelineWWPrecision ; }
void SetTimelineWWPrecision(PRV_UINT32 value) { timelineWWPrecision = value ; }

bool GetTimelineWWPreviousNext() const { return timelineWWPreviousNext ; }
void SetTimelineWWPreviousNext(bool value) { timelineWWPreviousNext = value ; }

bool GetTimelineWWSemantic() const { return timelineWWSemantic ; }
void SetTimelineWWSemantic(bool value) { timelineWWSemantic = value ; }

bool GetTimelineWWText() const { return timelineWWText ; }
void SetTimelineWWText(bool value) { timelineWWText = value ; }

std::string GetTmpPath() const { return tmpPath ; }
void SetTmpPath(std::string value) { tmpPath = value ; }

std::string GetTracesPath() const { return tracesPath ; }
void SetTracesPath(std::string value) { tracesPath = value ; }

std::string GetTutorialsPath() const { return tutorialsPath ; }
void SetTutorialsPath(std::string value) { tutorialsPath = value ; }

PRV_UINT32 GetWhatWhereMaxPrecision() const { return whatWhereMaxPrecision ; }
void SetWhatWhereMaxPrecision(PRV_UINT32 value) { whatWhereMaxPrecision = value ; }

std::map<wxString,Workspace> GetWorkspaceContainer() const { return workspaceContainer ; }
void SetWorkspaceContainer(std::map<wxString,Workspace> value) { workspaceContainer = value ; }

bool GetWorkspaceDiscardedSubmenu() const { return workspaceDiscardedSubmenu ; }
void SetWorkspaceDiscardedSubmenu(bool value) { workspaceDiscardedSubmenu = value ; }

wxBitmap GetBitmapResource( const wxString& name );

wxIcon GetIconResource( const wxString& name );

static bool ShowToolTips();

bool TransferDataToWindow();
bool TransferDataFromWindow();


bool SetPanel( wxWindowID whichPanelID );

wxPanel* panelGlobal;
wxCheckBox* checkGlobalFillStateGaps;
wxCheckBox* checkGlobalFullTracePath;
wxSpinCtrl* txtMaximumTraceSize;
wxTextCtrl* textCtrlTrace;
DirBrowserButton* dirBrowserButtonTrace;
wxTextCtrl* textCtrlCFG;
DirBrowserButton* dirBrowserButtonCFG;
wxTextCtrl* textCtrlXML;
DirBrowserButton* dirBrowserButtonXML;
wxTextCtrl* textCtrlTutorials;
DirBrowserButton* dirBrowserButtonTutorials;
wxTextCtrl* textCtrlTmp;
DirBrowserButton* dirBrowserButtonTmp;
wxCheckBox* checkGlobalSingleInstance;
wxSpinCtrl* spinSessionTime;
wxCheckBox* checkGlobalAskForPrevSessionLoad;
wxCheckBox* checkGlobalHelpOnBrowser;
wxCheckBox* checkDisableTimelineZoomMouseWheel;
wxPanel* panelTimeline;
wxCheckBox* checkTimelineKeepSyncGroupClone;
wxTextCtrl* txtTimelineNameFormatPrefix;
wxTextCtrl* txtTimelineNameFormatFull;
wxCheckBox* checkTimelineCommunicationLines;
wxCheckBox* checkTimelineEventLines;
wxCheckBox* checkSemanticScaleMinAtZero;
wxChoice* choiceTimelineColor;
wxChoice* choiceTimelineGradientFunction;
wxChoice* choiceTimelineDrawmodeTime;
wxChoice* choiceTimelineDrawmodeObjects;
wxChoice* choiceTimelinePixelSize;
wxChoice* choiceTimelineLabels;
wxChoice* choiceTimelineObjectAxis;
wxCheckBox* checkTimelineWWSemantic;
wxCheckBox* checkTimelineWWEvents;
wxCheckBox* checkTimelineWWCommunications;
wxCheckBox* checkTimelineWWPreviousNext;
wxCheckBox* checkTimelineWWText;
wxSpinCtrl* txtTimelineWWPrecision;
wxSpinCtrl* txtTimelineWWEventPixels;
wxChoice* choiceTimelineSaveImageFormat;
wxChoice* choiceTimelineSaveTextFormat;
wxPanel* panelHistogram;
wxCheckBox* checkHistogramSkipCreateDialog;
wxCheckBox* checkHistogramKeepSyncGroupClone;
wxTextCtrl* txtHistogramNameFormatPrefix;
wxTextCtrl* txtHistogramNameFormatFull;
wxCheckBox* checkHistogramZoom;
wxCheckBox* checkHistogramHorizontal;
wxCheckBox* checkHistogramHideEmpty;
wxCheckBox* checkHistogramShowGradient;
wxCheckBox* checkHistogramLabelsColor;
wxChoice* choiceHistogramGradientFunction;
wxChoice* choiceHistogramDrawmodeSemantic;
wxChoice* choiceHistogramDrawmodeObjects;
wxCheckBox* checkHistogramScientificNotation;
wxCheckBox* checkHistogramThousandsSeparator;
wxCheckBox* checkHistogramShowUnits;
wxSpinCtrl* txtHistogramPrecision;
wxCheckBox* checkHistogramAutofitControlScale;
wxCheckBox* checkHistogramAutofitControlScaleZero;
wxCheckBox* checkHistogramAutofit3DScale;
wxCheckBox* checkHistogramAutofitDataGradient;
wxSpinCtrl* txtHistogramNumColumns;
wxChoice* choiceHistogramSaveImageFormat;
wxChoice* choiceHistogramSaveTextFormat;
wxPanel* panelColor;
wxColourPickerCtrl* colourPickerBackground;
wxColourPickerCtrl* colourPickerAxis;
wxCheckBox* checkZero;
wxColourPickerCtrl* colourPickerZero;
wxColourPickerCtrl* colourPickerPunctual;
wxColourPickerCtrl* colourPickerLogical;
wxColourPickerCtrl* colourPickerPhysical;
wxColourPickerCtrl* colourPickerGradientBegin;
wxColourPickerCtrl* colourPickerGradientEnd;
wxColourPickerCtrl* colourPickerNegativeGradientBegin;
wxColourPickerCtrl* colourPickerNegativeGradientEnd;
wxColourPickerCtrl* colourPickerGradientLow;
wxColourPickerCtrl* colourPickerGradientTop;
wxPanel* panelWorkspaces;
wxCheckBox* checkDiscardedSubmenu;
wxListBox* listWorkspaces;
wxBitmapButton* buttonAddWorkspace;
wxBitmapButton* buttonDeleteWorkspace;
wxBitmapButton* buttonUpWorkspace;
wxBitmapButton* buttonDownWorkspace;
wxBitmapButton* buttonImportWorkspace;
wxBitmapButton* buttonExportWorkspace;
wxTextCtrl* txtWorkspaceName;
wxRadioButton* radioStates;
wxRadioButton* radioEventTypes;
wxTextCtrl* txtAutoTypes;
wxListBox* listHintsWorkspace;
wxBitmapButton* buttonAddHint;
wxBitmapButton* buttonDeleteHint;
wxBitmapButton* buttonUpHint;
wxBitmapButton* buttonDownHint;
wxTextCtrl* txtHintPath;
FileBrowserButton* fileBrowserHintPath;
wxTextCtrl* txtHintDescription;
wxPanel* panelExternal;
wxListBox* listTextEditors;
wxBitmapButton* buttonAddTextEditor;
wxBitmapButton* buttonDeleteTextEditor;
wxBitmapButton* buttonUpTextEditor;
wxBitmapButton* buttonDownTextEditor;
wxListBox* listPDFReaders;
wxBitmapButton* buttonAddPDFReader;
wxBitmapButton* buttonDeletePDFReader;
wxBitmapButton* buttonUpPDFReader;
wxBitmapButton* buttonDownPDFReader;
wxPanel* panelFilters;
private:
bool askForPrevSessionLoad;
std::string cfgsPath;
bool colorUseZero;
bool disableTimelineZoomMouseWheel;
std::string filtersXMLPath;
bool globalFillStateGaps;
bool globalFullTracePath;
rgb gradientColourBegin;
rgb gradientColourEnd;
rgb gradientColourLow;
rgb gradientColourNegativeBegin;
rgb gradientColourNegativeEnd;
rgb gradientColourTop;
bool helpContentsUsesBrowser;
bool histogramAutofit3DScale;
bool histogramAutofitControlScale;
bool histogramAutofitControlScaleZero;
bool histogramAutofitDataGradient;
PRV_UINT32 histogramDrawmodeObjects;
PRV_UINT32 histogramDrawmodeSemantic;
PRV_UINT32 histogramGradientFunction;
bool histogramHideEmpty;
bool histogramHorizontal;
bool histogramKeepSyncGroupClone;
bool histogramLabelsColor;
THistogramColumn histogramMaxNumColumns;
PRV_UINT32 histogramMaxPrecision;
std::string histogramNameFormatFull;
std::string histogramNameFormatPrefix;
THistogramColumn histogramNumColumns;
PRV_UINT32 histogramPrecision;
PRV_UINT32 histogramSaveImageFormat;
PRV_UINT32 histogramSaveTextFormat;
bool histogramScientificNotation;
bool histogramShowGradient;
bool histogramShowUnits;
bool histogramSkipCreateDialog;
bool histogramThousandSeparator;
bool histogramZoom;
float maximumTraceSize;
wxArrayString pdfReaderOptions;
PRV_UINT16 sessionSaveTime;
bool singleInstance;
wxArrayString textEditorOptions;
PRV_UINT32 timelineColor;
rgb timelineColourAxis;
rgb timelineColourBackground;
rgb timelineColourLogical;
rgb timelineColourPhysical;
rgb timelineColourPunctual;
rgb timelineColourZero;
bool timelineCommunicationLines;
PRV_UINT32 timelineDrawmodeObjects;
PRV_UINT32 timelineDrawmodeTime;
bool timelineEventLines;
PRV_UINT32 timelineGradientFunction;
bool timelineKeepSyncGroupClone;
std::string timelineNameFormatFull;
std::string timelineNameFormatPrefix;
PRV_UINT32 timelineObjectAxis;
PRV_UINT32 timelineObjectLabels;
PRV_UINT32 timelinePixelSize;
PRV_UINT32 timelineSaveImageFormat;
PRV_UINT32 timelineSaveTextFormat;
bool timelineSemanticScaleMinAtZero;
bool timelineWWCommunications;
PRV_INT16 timelineWWEventPixels;
bool timelineWWEvents;
PRV_UINT32 timelineWWPrecision;
bool timelineWWPreviousNext;
bool timelineWWSemantic;
bool timelineWWText;
std::string tmpPath;
std::string tracesPath;
std::string tutorialsPath;
PRV_UINT32 whatWhereMaxPrecision;
std::map<wxString,Workspace> workspaceContainer;
bool workspaceDiscardedSubmenu;

wxString originalWorkspaceName;

std::map< wxWindowID, size_t > panelID;

wxString formatNumber( long value );
void setLabelsChoiceBox( const std::vector< std::string > &list,
const PRV_UINT32 &selected,
wxChoice *choiceBox );
rgb wxColourToRGB( wxColour colour ) ;
wxColour RGBTowxColour( rgb colour );

void workSpaceNameKillFocus( const wxString& whichName );

};
