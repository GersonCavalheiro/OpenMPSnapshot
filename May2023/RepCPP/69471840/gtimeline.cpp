

#include "wx/wxprec.h"
#ifdef __BORLANDC__
#pragma hdrstopclo
#endif

#ifndef WX_PRECOMP
#include "wx/wx.h"
#endif

#include <wx/version.h>
#include <wx/dcgraph.h>

#include "wx/imaglist.h"
#include <wx/dcbuffer.h>
#include <wx/statline.h>
#include <algorithm>
#include <wx/filename.h>
#include <wx/display.h>
#include <wx/valnum.h>

#include "config_traits.h"
#include "wxparaverapp.h"
#include "gtimeline.h"
#include "finddialog.h"
#include "window.h"
#include "labelconstructor.h"
#include "drawmode.h"
#include "loadedwindows.h"
#include "output.h"
#include "filedialogext.h"
#include "progresscontroller.h"
#include "paravermain.h"
#include "saveimagedialog.h"
#include "sequencedriver.h"
#include "popupmenu.h"

#include "autoredraw_refresh.xpm"
#include "caution.xpm"
#include "caution_yellow.xpm"

#ifdef __WXMAC__
#include <wx/rawbmp.h>
#endif

#define wxTEST_GRAPHICS 1

#if wxTEST_GRAPHICS
#include "wx/graphics.h"
#if wxUSE_GRAPHICS_CONTEXT == 0
#undef wxTEST_GRAPHICS
#define wxTEST_GRAPHICS 0
#endif
#else
#undef wxUSE_GRAPHICS_CONTEXT
#define wxUSE_GRAPHICS_CONTEXT 0
#endif

using namespace std;

#ifdef __WXMAC__
constexpr int TIMER_SIZE_DURATION = 750;
#else
constexpr int TIMER_SIZE_DURATION = 250;
#endif

constexpr size_t MAX_LEGEND_COLORS = 512;





IMPLEMENT_CLASS( gTimeline, wxFrame )




BEGIN_EVENT_TABLE( gTimeline, wxFrame )

EVT_CLOSE( gTimeline::OnCloseWindow )
EVT_IDLE( gTimeline::OnIdle )
EVT_RIGHT_DOWN( gTimeline::OnRightDown )
EVT_SPLITTER_DCLICK( ID_SPLITTER_TIMELINE, gTimeline::OnSplitterTimelineSashDClick )
EVT_SPLITTER_UNSPLIT( ID_SPLITTER_TIMELINE, gTimeline::OnSplitterTimelineSashUnsplit )
EVT_UPDATE_UI( ID_SCROLLED_DRAW, gTimeline::OnScrolledWindowUpdate )
EVT_NOTEBOOK_PAGE_CHANGING( ID_NOTEBOOK_INFO, gTimeline::OnNotebookInfoPageChanging )
EVT_CHECKBOX( ID_CHECKBOX, gTimeline::OnCheckWhatWhere )
EVT_CHECKBOX( ID_CHECKBOX1, gTimeline::OnCheckWhatWhere )
EVT_CHECKBOX( ID_CHECKBOX2, gTimeline::OnCheckWhatWhere )
EVT_CHECKBOX( ID_CHECKBOX3, gTimeline::OnCheckWhatWhere )
EVT_CHECKBOX( ID_CHECKBOX4, gTimeline::OnCheckWhatWhereText )
EVT_CHECKBOX( ID_CHECKBOX5, gTimeline::OnCheckWhatWhereText )
EVT_UPDATE_UI( ID_CHECKBOX5, gTimeline::OnCheckWWShowDateUpdate )
EVT_CHECKBOX( ID_CHECKBOX6, gTimeline::OnCheckWhatWhereText )
EVT_UPDATE_UI( wxID_STATIC_SLOPE, gTimeline::OnStaticSlopeUpdate )
EVT_CHECKBOX( ID_CHECKBOX_CUSTOM_PALETTE, gTimeline::OnCheckboxCustomPaletteClick )
EVT_UPDATE_UI( ID_CHECKBOX_CUSTOM_PALETTE, gTimeline::OnCheckboxCustomPaletteUpdate )
EVT_BUTTON( ID_BUTTON_CUSTOM_PALETTE_APPLY, gTimeline::OnButtonCustomPaletteApplyClick )
EVT_UPDATE_UI( ID_BUTTON_CUSTOM_PALETTE_APPLY, gTimeline::OnButtonCustomPaletteApplyUpdate )
EVT_UPDATE_UI( ID_SCROLLED_COLORS, gTimeline::OnScrolledColorsUpdate )
EVT_UPDATE_UI( wxID_STATIC_RED, gTimeline::OnStaticSelectedColorUpdate )
EVT_SLIDER( ID_SLIDER_RED, gTimeline::OnSliderSelectedColorUpdated )
EVT_UPDATE_UI( ID_SLIDER_RED, gTimeline::OnSliderSelectedColorUpdateUI )
EVT_TEXT( ID_TEXT_RED, gTimeline::OnTextSelectedColorUpdated )
EVT_UPDATE_UI( ID_TEXT_RED, gTimeline::OnTextSelectedColorUpdate )
EVT_UPDATE_UI( wxID_STATIC_GREEN, gTimeline::OnStaticSelectedColorUpdate )
EVT_SLIDER( ID_SLIDER_GREEN, gTimeline::OnSliderSelectedColorUpdated )
EVT_UPDATE_UI( ID_SLIDER_GREEN, gTimeline::OnSliderSelectedColorUpdateUI )
EVT_TEXT( ID_TEXT_GREEN, gTimeline::OnTextSelectedColorUpdated )
EVT_UPDATE_UI( ID_TEXT_GREEN, gTimeline::OnTextSelectedColorUpdate )
EVT_UPDATE_UI( wxID_STATIC_BLUE, gTimeline::OnStaticSelectedColorUpdate )
EVT_SLIDER( ID_SLIDER_BLUE, gTimeline::OnSliderSelectedColorUpdated )
EVT_UPDATE_UI( ID_SLIDER_BLUE, gTimeline::OnSliderSelectedColorUpdateUI )
EVT_TEXT( ID_TEXT_BLUE, gTimeline::OnTextSelectedColorUpdated )
EVT_UPDATE_UI( ID_TEXT_BLUE, gTimeline::OnTextSelectedColorUpdate )

EVT_TIMER( ID_TIMER_SIZE, gTimeline::OnTimerSize )
EVT_TIMER( ID_TIMER_MOTION, gTimeline::OnTimerMotion )
EVT_TIMER( ID_TIMER_WHEEL, gTimeline::OnTimerWheel )

END_EVENT_TABLE()

wxProgressDialog *gTimeline::dialogProgress = nullptr;
int gTimeline::numberOfProgressDialogUsers = 0;


#include "paravermain.h"
gTimeline::gTimeline() :
gWindow()
{
Init();
parent = nullptr;
}

gTimeline::gTimeline( wxWindow* whichParent,
wxWindowID id,
const wxString& caption,
const wxPoint& pos,
const wxSize& size,
long style ) :
gWindow()
{
Init();
Create( whichParent, id, caption, pos, size, style );
parent = whichParent;
}




bool gTimeline::Create( wxWindow* parent, wxWindowID id, const wxString& caption, const wxPoint& pos, const wxSize& size, long style )
{
wxFrame::Create( parent, id, caption, pos, size, style );

CreateControls();
splitter->Unsplit();
redrawStopWatch->Pause();
return true;
}




gTimeline::~gTimeline()
{
gPasteWindowProperties::getInstance()->verifyRemove( this );

delete timerMotion;
delete timerSize;
delete timerWheel;
delete myWindow;
delete redrawStopWatch;
}




void gTimeline::Init()
{
canRedraw = false;
drawCaution = false;
drawCautionNegatives = false;
escapePressed = false;
findBeginTime = 0;
findEndTime = 0;
findFirstObject = 0;
findLastObject = 0;
firstUnsplit = false;
infoZoneLastSize = 200;
lastEventFoundTime = 0;
lastFoundObject = 0;
lastSemanticFoundTime = 0;
myWindow = nullptr;
objectHeight = 1;
redoColors = false;
redrawStopWatch = new wxStopWatch();
splitChanged = false;
timerMotion = new wxTimer( this, ID_TIMER_MOTION );
timerSize = new wxTimer( this, ID_TIMER_SIZE );
timerWheel = new wxTimer( this, ID_TIMER_WHEEL );
timing = false;
wheelZoomBeginObject = 0;
wheelZoomBeginTime = 0;
wheelZoomEndObject = 0;
wheelZoomEndTime = 0;
wheelZoomFactor = 1;
zooming = false;
splitter = NULL;
drawZone = NULL;
infoZone = NULL;
whatWherePanel = NULL;
checkWWSemantic = NULL;
checkWWEvents = NULL;
checkWWCommunications = NULL;
checkWWPreviousNext = NULL;
checkWWText = NULL;
checkWWShowDate = NULL;
checkWWHex = NULL;
whatWhereText = NULL;
timingZone = NULL;
initialTimeText = NULL;
initialSemanticLabel = NULL;
initialSemanticText = NULL;
finalTimeText = NULL;
finalSemanticLabel = NULL;
finalSemanticText = NULL;
durationText = NULL;
slopeLabel = NULL;
slopeText = NULL;
colorsPanelGlobal = NULL;
checkboxCustomPalette = NULL;
buttonCustomPaletteApply = NULL;
colorsPanel = NULL;
colorsSizer = NULL;
sizerSelectedColor = NULL;
panelSelectedColor = NULL;
labelSelectedColorRed = NULL;
sliderSelectedRed = NULL;
textSelectedRed = NULL;
labelSelectedColorGreen = NULL;
sliderSelectedGreen = NULL;
textSelectedGreen = NULL;
labelSelectedColorBlue = NULL;
sliderSelectedBlue = NULL;
textSelectedBlue = NULL;

zoomXY = false;
bufferImage.Create( 1, 1 );
#ifdef __WXMAC__
objectFont = wxFont( 8, wxFONTFAMILY_TELETYPE, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD );
timeFont = wxFont( 8, wxFONTFAMILY_TELETYPE, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL );
semanticFont = wxFont( 10, wxFONTFAMILY_TELETYPE, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD );
#else
objectFont = wxFont( 7, wxFONTFAMILY_MODERN, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL );
timeFont = wxFont( 6, wxFONTFAMILY_MODERN, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_NORMAL );
semanticFont = wxFont( 8, wxFONTFAMILY_MODERN, wxFONTSTYLE_NORMAL, wxFONTWEIGHT_BOLD );
#endif
whatWhereTime = 0.0;
whatWhereRow = 0;
whatWhereSemantic = 0.0;

forceRedoColors = false;
enableApplyButton = false;
lastType = NO_TYPE;
lastMin = 0;
lastMax = 15;
lastValuesSize = semanticValuesToColor.size();
codeColorSet = true;
gradientFunc = TGradientFunction::LINEAR;

#ifdef _WIN32
wheelZoomObjects = false;
#endif

selectedItemColor = nullptr;
selectedCustomValue = 0;

enabledAutoRedrawIcon = false;
}




void gTimeline::CreateControls()
{
gTimeline* itemFrame1 = this;

splitter = new wxSplitterWindow( itemFrame1, ID_SPLITTER_TIMELINE, wxDefaultPosition, wxDefaultSize, wxSP_BORDER|wxSP_3DSASH|wxSP_PERMIT_UNSPLIT );
splitter->SetMinimumPaneSize(wxDLG_UNIT(itemFrame1, wxSize(0, -1)).x);

drawZone = new wxScrolledWindow( splitter, ID_SCROLLED_DRAW, wxDefaultPosition, wxDefaultSize, wxNO_BORDER|wxWANTS_CHARS|wxFULL_REPAINT_ON_RESIZE );
drawZone->SetScrollbars(1, 1, 0, 0);
infoZone = new wxNotebook( splitter, ID_NOTEBOOK_INFO, wxDefaultPosition, wxDefaultSize, wxBK_DEFAULT );

whatWherePanel = new wxScrolledWindow( infoZone, ID_SCROLLED_WHATWHERE, wxDefaultPosition, wxDefaultSize, wxSUNKEN_BORDER|wxHSCROLL|wxVSCROLL );
whatWherePanel->SetScrollbars(1, 1, 0, 0);
wxBoxSizer* itemBoxSizer6 = new wxBoxSizer(wxVERTICAL);
whatWherePanel->SetSizer(itemBoxSizer6);

wxBoxSizer* itemBoxSizer7 = new wxBoxSizer(wxHORIZONTAL);
itemBoxSizer6->Add(itemBoxSizer7, 0, wxALIGN_LEFT|wxALL, wxDLG_UNIT(whatWherePanel, wxSize(5, -1)).x);
checkWWSemantic = new wxCheckBox( whatWherePanel, ID_CHECKBOX, _("Semantic"), wxDefaultPosition, wxDefaultSize, 0 );
checkWWSemantic->SetValue(true);
itemBoxSizer7->Add(checkWWSemantic, 0, wxALIGN_CENTER_VERTICAL, wxDLG_UNIT(whatWherePanel, wxSize(2, -1)).x);

checkWWEvents = new wxCheckBox( whatWherePanel, ID_CHECKBOX1, _("Events"), wxDefaultPosition, wxDefaultSize, 0 );
checkWWEvents->SetValue(true);
itemBoxSizer7->Add(checkWWEvents, 0, wxALIGN_CENTER_VERTICAL|wxLEFT, wxDLG_UNIT(whatWherePanel, wxSize(2, -1)).x);

checkWWCommunications = new wxCheckBox( whatWherePanel, ID_CHECKBOX2, _("Communications"), wxDefaultPosition, wxDefaultSize, 0 );
checkWWCommunications->SetValue(true);
itemBoxSizer7->Add(checkWWCommunications, 0, wxALIGN_CENTER_VERTICAL|wxLEFT, wxDLG_UNIT(whatWherePanel, wxSize(2, -1)).x);

checkWWPreviousNext = new wxCheckBox( whatWherePanel, ID_CHECKBOX3, _("Prev./Next"), wxDefaultPosition, wxDefaultSize, 0 );
checkWWPreviousNext->SetValue(true);
itemBoxSizer7->Add(checkWWPreviousNext, 0, wxALIGN_CENTER_VERTICAL|wxLEFT, wxDLG_UNIT(whatWherePanel, wxSize(2, -1)).x);

checkWWText = new wxCheckBox( whatWherePanel, ID_CHECKBOX4, _("Text"), wxDefaultPosition, wxDefaultSize, 0 );
checkWWText->SetValue(true);
itemBoxSizer7->Add(checkWWText, 0, wxALIGN_CENTER_VERTICAL|wxLEFT, wxDLG_UNIT(whatWherePanel, wxSize(2, -1)).x);

checkWWShowDate = new wxCheckBox( whatWherePanel, ID_CHECKBOX5, _("Show Date"), wxDefaultPosition, wxDefaultSize, 0 );
checkWWShowDate->SetValue(false);
itemBoxSizer7->Add(checkWWShowDate, 0, wxALIGN_CENTER_VERTICAL|wxLEFT, wxDLG_UNIT(whatWherePanel, wxSize(2, -1)).x);

checkWWHex = new wxCheckBox( whatWherePanel, ID_CHECKBOX6, _("Hex"), wxDefaultPosition, wxDefaultSize, 0 );
checkWWHex->SetValue(false);
itemBoxSizer7->Add(checkWWHex, 0, wxALIGN_CENTER_VERTICAL|wxLEFT, wxDLG_UNIT(whatWherePanel, wxSize(2, -1)).x);

whatWhereText = new wxRichTextCtrl( whatWherePanel, ID_RICHTEXTCTRL, wxEmptyString, wxDefaultPosition, wxDefaultSize, wxTE_READONLY|wxWANTS_CHARS );
itemBoxSizer6->Add(whatWhereText, 1, wxGROW|wxLEFT|wxRIGHT|wxBOTTOM, wxDLG_UNIT(whatWherePanel, wxSize(5, -1)).x);

whatWherePanel->FitInside();
infoZone->AddPage(whatWherePanel, _("What / Where"));

timingZone = new wxScrolledWindow( infoZone, ID_SCROLLED_TIMING, wxDefaultPosition, wxDefaultSize, wxSUNKEN_BORDER|wxTAB_TRAVERSAL );
timingZone->SetScrollbars(1, 5, 0, 0);
wxFlexGridSizer* itemFlexGridSizer15 = new wxFlexGridSizer(3, 4, 0, 0);
timingZone->SetSizer(itemFlexGridSizer15);

wxStaticText* itemStaticText16 = new wxStaticText( timingZone, wxID_STATIC, _("Initial time"), wxDefaultPosition, wxDefaultSize, 0 );
itemFlexGridSizer15->Add(itemStaticText16, 0, wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxLEFT|wxRIGHT, wxDLG_UNIT(timingZone, wxSize(5, -1)).x);

initialTimeText = new wxTextCtrl( timingZone, ID_TEXTCTRL_INITIALTIME, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
itemFlexGridSizer15->Add(initialTimeText, 1, wxGROW|wxALIGN_CENTER_VERTICAL|wxRIGHT|wxTOP|wxBOTTOM, wxDLG_UNIT(timingZone, wxSize(5, -1)).x);

initialSemanticLabel = new wxStaticText( timingZone, wxID_STATIC_INITIALSEMANTIC, _("Inital semantic"), wxDefaultPosition, wxDefaultSize, 0 );
itemFlexGridSizer15->Add(initialSemanticLabel, 0, wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxLEFT|wxRIGHT, wxDLG_UNIT(timingZone, wxSize(5, -1)).x);

initialSemanticText = new wxTextCtrl( timingZone, ID_TEXTCTRL_INITIALSEMANTIC, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
itemFlexGridSizer15->Add(initialSemanticText, 1, wxGROW|wxALIGN_CENTER_VERTICAL|wxRIGHT|wxTOP|wxBOTTOM, wxDLG_UNIT(timingZone, wxSize(5, -1)).x);

wxStaticText* itemStaticText20 = new wxStaticText( timingZone, wxID_STATIC, _("Final time"), wxDefaultPosition, wxDefaultSize, 0 );
itemFlexGridSizer15->Add(itemStaticText20, 0, wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxLEFT|wxRIGHT, wxDLG_UNIT(timingZone, wxSize(5, -1)).x);

finalTimeText = new wxTextCtrl( timingZone, ID_TEXTCTRL_FINALTIME, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
itemFlexGridSizer15->Add(finalTimeText, 1, wxGROW|wxALIGN_CENTER_VERTICAL|wxRIGHT|wxTOP|wxBOTTOM, wxDLG_UNIT(timingZone, wxSize(5, -1)).x);

finalSemanticLabel = new wxStaticText( timingZone, wxID_STATIC_FINALSEMANTIC, _("Final semantic"), wxDefaultPosition, wxDefaultSize, 0 );
itemFlexGridSizer15->Add(finalSemanticLabel, 0, wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxLEFT|wxRIGHT, wxDLG_UNIT(timingZone, wxSize(5, -1)).x);

finalSemanticText = new wxTextCtrl( timingZone, ID_TEXTCTRL_FINALSEMANTIC, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
itemFlexGridSizer15->Add(finalSemanticText, 1, wxGROW|wxALIGN_CENTER_VERTICAL|wxRIGHT|wxTOP|wxBOTTOM, wxDLG_UNIT(timingZone, wxSize(5, -1)).x);

wxStaticText* itemStaticText24 = new wxStaticText( timingZone, wxID_STATIC, _("Duration"), wxDefaultPosition, wxDefaultSize, 0 );
itemFlexGridSizer15->Add(itemStaticText24, 0, wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxLEFT|wxRIGHT, wxDLG_UNIT(timingZone, wxSize(5, -1)).x);

durationText = new wxTextCtrl( timingZone, ID_TEXTCTRL_DURATION, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
itemFlexGridSizer15->Add(durationText, 1, wxGROW|wxALIGN_CENTER_VERTICAL|wxRIGHT|wxTOP|wxBOTTOM, wxDLG_UNIT(timingZone, wxSize(5, -1)).x);

slopeLabel = new wxStaticText( timingZone, wxID_STATIC_SLOPE, _("Slope"), wxDefaultPosition, wxDefaultSize, 0 );
itemFlexGridSizer15->Add(slopeLabel, 0, wxALIGN_CENTER_HORIZONTAL|wxALIGN_CENTER_VERTICAL|wxLEFT|wxRIGHT, wxDLG_UNIT(timingZone, wxSize(5, -1)).x);

slopeText = new wxTextCtrl( timingZone, ID_TEXTCTRL_SLOPE, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
itemFlexGridSizer15->Add(slopeText, 1, wxGROW|wxALIGN_CENTER_VERTICAL|wxRIGHT|wxTOP|wxBOTTOM, wxDLG_UNIT(timingZone, wxSize(5, -1)).x);

itemFlexGridSizer15->AddGrowableCol(1);
itemFlexGridSizer15->AddGrowableCol(3);

timingZone->FitInside();
infoZone->AddPage(timingZone, _("Timing"));

colorsPanelGlobal = new wxScrolledWindow( infoZone, ID_SCROLLEDWINDOW, wxDefaultPosition, wxDefaultSize, wxSUNKEN_BORDER|wxHSCROLL|wxVSCROLL );
colorsPanelGlobal->SetScrollbars(1, 1, 0, 0);
wxBoxSizer* itemBoxSizer2 = new wxBoxSizer(wxVERTICAL);
colorsPanelGlobal->SetSizer(itemBoxSizer2);

wxBoxSizer* itemBoxSizer3 = new wxBoxSizer(wxHORIZONTAL);
itemBoxSizer2->Add(itemBoxSizer3, 0, wxGROW, wxDLG_UNIT(colorsPanelGlobal, wxSize(2, -1)).x);
checkboxCustomPalette = new wxCheckBox( colorsPanelGlobal, ID_CHECKBOX_CUSTOM_PALETTE, _("Custom palette"), wxDefaultPosition, wxDefaultSize, 0 );
checkboxCustomPalette->SetValue(false);
itemBoxSizer3->Add(checkboxCustomPalette, 0, wxALIGN_CENTER_VERTICAL|wxALL, wxDLG_UNIT(colorsPanelGlobal, wxSize(2, -1)).x);

buttonCustomPaletteApply = new wxButton( colorsPanelGlobal, ID_BUTTON_CUSTOM_PALETTE_APPLY, _("Apply"), wxDefaultPosition, wxDefaultSize, 0 );
itemBoxSizer3->Add(buttonCustomPaletteApply, 0, wxALIGN_CENTER_VERTICAL, wxDLG_UNIT(colorsPanelGlobal, wxSize(2, -1)).x);

wxBoxSizer* itemBoxSizer5 = new wxBoxSizer(wxHORIZONTAL);
itemBoxSizer2->Add(itemBoxSizer5, 1, wxGROW, wxDLG_UNIT(colorsPanelGlobal, wxSize(2, -1)).x);
colorsPanel = new wxScrolledWindow( colorsPanelGlobal, ID_SCROLLED_COLORS, wxDefaultPosition, wxDefaultSize, wxSUNKEN_BORDER|wxHSCROLL|wxVSCROLL|wxTAB_TRAVERSAL );
itemBoxSizer5->Add(colorsPanel, 1, wxGROW, wxDLG_UNIT(colorsPanelGlobal, wxSize(5, -1)).x);
colorsPanel->SetScrollbars(1, 5, 0, 0);
colorsSizer = new wxBoxSizer(wxVERTICAL);
colorsPanel->SetSizer(colorsSizer);

colorsPanel->FitInside();
sizerSelectedColor = new wxBoxSizer(wxHORIZONTAL);
itemBoxSizer5->Add(sizerSelectedColor, 2, wxALIGN_TOP|wxALL, wxDLG_UNIT(colorsPanelGlobal, wxSize(2, -1)).x);
wxBoxSizer* itemBoxSizer1 = new wxBoxSizer(wxVERTICAL);
sizerSelectedColor->Add(itemBoxSizer1, 1, wxGROW|wxALL, wxDLG_UNIT(colorsPanelGlobal, wxSize(2, -1)).x);
panelSelectedColor = new wxPanel( colorsPanelGlobal, ID_PANEL, wxDefaultPosition, wxDefaultSize, wxSUNKEN_BORDER|wxTAB_TRAVERSAL );
panelSelectedColor->SetExtraStyle(wxWS_EX_VALIDATE_RECURSIVELY);
itemBoxSizer1->Add(panelSelectedColor, 1, wxGROW|wxALL, wxDLG_UNIT(colorsPanelGlobal, wxSize(2, -1)).x);

wxBoxSizer* itemBoxSizer4 = new wxBoxSizer(wxVERTICAL);
sizerSelectedColor->Add(itemBoxSizer4, 3, wxALIGN_TOP|wxLEFT|wxRIGHT, wxDLG_UNIT(colorsPanelGlobal, wxSize(2, -1)).x);
wxBoxSizer* itemBoxSizer10 = new wxBoxSizer(wxHORIZONTAL);
itemBoxSizer4->Add(itemBoxSizer10, 0, wxGROW|wxLEFT|wxRIGHT, wxDLG_UNIT(colorsPanelGlobal, wxSize(2, -1)).x);
labelSelectedColorRed = new wxStaticText( colorsPanelGlobal, wxID_STATIC_RED, _("Red"), wxDefaultPosition, wxDLG_UNIT(colorsPanelGlobal, wxSize(30, -1)), 0 );
itemBoxSizer10->Add(labelSelectedColorRed, 0, wxALIGN_CENTER_VERTICAL|wxALL, wxDLG_UNIT(colorsPanelGlobal, wxSize(2, -1)).x);

sliderSelectedRed = new wxSlider( colorsPanelGlobal, ID_SLIDER_RED, 0, 0, 255, wxDefaultPosition, wxDefaultSize, wxSL_HORIZONTAL );
itemBoxSizer10->Add(sliderSelectedRed, 1, wxGROW|wxALL, wxDLG_UNIT(colorsPanelGlobal, wxSize(2, -1)).x);

textSelectedRed = new wxTextCtrl( colorsPanelGlobal, ID_TEXT_RED, wxEmptyString, wxDefaultPosition, wxDLG_UNIT(colorsPanelGlobal, wxSize(30, -1)), 0 );
textSelectedRed->SetMaxLength(3);
itemBoxSizer10->Add(textSelectedRed, 0, wxALIGN_CENTER_VERTICAL|wxALL, wxDLG_UNIT(colorsPanelGlobal, wxSize(2, -1)).x);

wxBoxSizer* itemBoxSizer13 = new wxBoxSizer(wxHORIZONTAL);
itemBoxSizer4->Add(itemBoxSizer13, 0, wxGROW|wxLEFT|wxRIGHT, wxDLG_UNIT(colorsPanelGlobal, wxSize(2, -1)).x);
labelSelectedColorGreen = new wxStaticText( colorsPanelGlobal, wxID_STATIC_GREEN, _("Green"), wxDefaultPosition, wxDLG_UNIT(colorsPanelGlobal, wxSize(30, -1)), 0 );
itemBoxSizer13->Add(labelSelectedColorGreen, 0, wxALIGN_CENTER_VERTICAL|wxALL, wxDLG_UNIT(colorsPanelGlobal, wxSize(2, -1)).x);

sliderSelectedGreen = new wxSlider( colorsPanelGlobal, ID_SLIDER_GREEN, 0, 0, 255, wxDefaultPosition, wxDefaultSize, wxSL_HORIZONTAL );
itemBoxSizer13->Add(sliderSelectedGreen, 1, wxGROW|wxALL, wxDLG_UNIT(colorsPanelGlobal, wxSize(2, -1)).x);

textSelectedGreen = new wxTextCtrl( colorsPanelGlobal, ID_TEXT_GREEN, wxEmptyString, wxDefaultPosition, wxDLG_UNIT(colorsPanelGlobal, wxSize(30, -1)), 0 );
textSelectedGreen->SetMaxLength(3);
itemBoxSizer13->Add(textSelectedGreen, 0, wxALIGN_CENTER_VERTICAL|wxALL, wxDLG_UNIT(colorsPanelGlobal, wxSize(2, -1)).x);

wxBoxSizer* itemBoxSizer16 = new wxBoxSizer(wxHORIZONTAL);
itemBoxSizer4->Add(itemBoxSizer16, 0, wxGROW|wxLEFT|wxRIGHT, wxDLG_UNIT(colorsPanelGlobal, wxSize(2, -1)).x);
labelSelectedColorBlue = new wxStaticText( colorsPanelGlobal, wxID_STATIC_BLUE, _("Blue"), wxDefaultPosition, wxDLG_UNIT(colorsPanelGlobal, wxSize(30, -1)), 0 );
itemBoxSizer16->Add(labelSelectedColorBlue, 0, wxALIGN_CENTER_VERTICAL|wxALL, wxDLG_UNIT(colorsPanelGlobal, wxSize(2, -1)).x);

sliderSelectedBlue = new wxSlider( colorsPanelGlobal, ID_SLIDER_BLUE, 0, 0, 255, wxDefaultPosition, wxDefaultSize, wxSL_HORIZONTAL );
sliderSelectedBlue->SetHelpText(_("blueSlider"));
if (gTimeline::ShowToolTips())
sliderSelectedBlue->SetToolTip(_("blueSlider"));
itemBoxSizer16->Add(sliderSelectedBlue, 1, wxGROW|wxALL, wxDLG_UNIT(colorsPanelGlobal, wxSize(2, -1)).x);

textSelectedBlue = new wxTextCtrl( colorsPanelGlobal, ID_TEXT_BLUE, wxEmptyString, wxDefaultPosition, wxDLG_UNIT(colorsPanelGlobal, wxSize(30, -1)), 0 );
textSelectedBlue->SetMaxLength(3);
itemBoxSizer16->Add(textSelectedBlue, 0, wxALIGN_CENTER_VERTICAL|wxALL, wxDLG_UNIT(colorsPanelGlobal, wxSize(2, -1)).x);

colorsPanelGlobal->FitInside();
infoZone->AddPage(colorsPanelGlobal, _("Colors"));

splitter->SplitHorizontally(drawZone, infoZone, wxDLG_UNIT(itemFrame1, wxSize(0, -1)).x);

drawZone->Connect(ID_SCROLLED_DRAW, wxEVT_SIZE, wxSizeEventHandler(gTimeline::OnScrolledWindowSize), NULL, this);
drawZone->Connect(ID_SCROLLED_DRAW, wxEVT_PAINT, wxPaintEventHandler(gTimeline::OnScrolledWindowPaint), NULL, this);
drawZone->Connect(ID_SCROLLED_DRAW, wxEVT_ERASE_BACKGROUND, wxEraseEventHandler(gTimeline::OnScrolledWindowEraseBackground), NULL, this);
drawZone->Connect(ID_SCROLLED_DRAW, wxEVT_LEFT_DOWN, wxMouseEventHandler(gTimeline::OnScrolledWindowLeftDown), NULL, this);
drawZone->Connect(ID_SCROLLED_DRAW, wxEVT_LEFT_UP, wxMouseEventHandler(gTimeline::OnScrolledWindowLeftUp), NULL, this);
drawZone->Connect(ID_SCROLLED_DRAW, wxEVT_LEFT_DCLICK, wxMouseEventHandler(gTimeline::OnScrolledWindowLeftDClick), NULL, this);
drawZone->Connect(ID_SCROLLED_DRAW, wxEVT_MIDDLE_UP, wxMouseEventHandler(gTimeline::OnScrolledWindowMiddleUp), NULL, this);
drawZone->Connect(ID_SCROLLED_DRAW, wxEVT_RIGHT_DOWN, wxMouseEventHandler(gTimeline::OnScrolledWindowRightDown), NULL, this);
drawZone->Connect(ID_SCROLLED_DRAW, wxEVT_MOTION, wxMouseEventHandler(gTimeline::OnScrolledWindowMotion), NULL, this);
drawZone->Connect(ID_SCROLLED_DRAW, wxEVT_MOUSEWHEEL, wxMouseEventHandler(gTimeline::OnScrolledWindowMouseWheel), NULL, this);
drawZone->Connect(ID_SCROLLED_DRAW, wxEVT_KEY_DOWN, wxKeyEventHandler(gTimeline::OnScrolledWindowKeyDown), NULL, this);

SetMinSize( wxSize( 100, 50 ) );

ParaverConfig *paraverConfig = ParaverConfig::getInstance();

checkWWSemantic->SetValue( paraverConfig->getTimelineWhatWhereSemantic() );
checkWWCommunications->SetValue( paraverConfig->getTimelineWhatWhereCommunications() );
checkWWEvents->SetValue( paraverConfig->getTimelineWhatWhereEvents() );
checkWWPreviousNext->SetValue( paraverConfig->getTimelineWhatWherePreviousNext() );
checkWWText->SetValue( paraverConfig->getTimelineWhatWhereText() );

wxIntegerValidator<unsigned short> tmpVal;
tmpVal.SetRange( 0, 255 );
textSelectedRed->SetValidator( tmpVal );
textSelectedGreen->SetValidator( tmpVal );
textSelectedBlue->SetValidator( tmpVal );
}




bool gTimeline::ShowToolTips()
{
return true;
}



wxBitmap gTimeline::GetBitmapResource( const wxString& name )
{
wxUnusedVar(name);
return wxNullBitmap;
}



wxIcon gTimeline::GetIconResource( const wxString& name )
{
wxUnusedVar(name);
return wxNullIcon;
}


std::vector< TObjectOrder > gTimeline::getCurrentZoomRange()
{
vector< TObjectOrder > zoomRange;

zoomRange.push_back( myWindow->getZoomSecondDimension().first );
zoomRange.push_back( myWindow->getZoomSecondDimension().second );

return zoomRange;
}


#ifdef __WXMAC__
void gTimeline::drawStackedImages( wxDC& dc )
{
if( bufferImage.IsOk() )
dc.DrawBitmap( bufferImage, 0, 0, false );

doDrawCaution( dc );

if( myWindow->getDrawFlags() && eventImage.IsOk() )
dc.DrawBitmap( eventImage, 0, 0, true );

if( myWindow->getDrawCommLines() && commImage.IsOk() )
dc.DrawBitmap( commImage, 0, 0, true );

if( zooming && zoomBMP.IsOk() && !motionEvent.ShiftDown() )
{
dc.DrawBitmap( zoomBMP, 0, 0, true );
}
}
#endif

void gTimeline::redraw()
{
#ifdef __WXGTK__
if( splitChanged )
{
resizeDrawZone( myWindow->getWidth(), myWindow->getHeight() );
splitChanged = false;
return;
}
#endif
myWindow->setReady( false );
redoColors = true;
enableApplyButton = false;
enabledAutoRedrawIcon = false;

semanticValuesToColor.clear();
semanticColorsToValue.clear();
semanticPixelsToValue.clear();

rgb rgbForegroundColour = ((paraverMain *)parent)->GetParaverConfig()->getColorsTimelineAxis();
foregroundColour = wxColour( rgbForegroundColour.red,
rgbForegroundColour.green,
rgbForegroundColour.blue );
rgb rgbBackgroundColour = ((paraverMain *)parent)->GetParaverConfig()->getColorsTimelineBackground();
backgroundColour = wxColour( rgbBackgroundColour.red,
rgbBackgroundColour.green,
rgbBackgroundColour.blue );

rgb rgbLogicalColour = ((paraverMain *)parent)->GetParaverConfig()->getColorsTimelineLogicalCommunications();
logicalColour = wxColour( rgbLogicalColour.red, rgbLogicalColour.green ,rgbLogicalColour.blue );

rgb rgbPhysicalColour = ((paraverMain *)parent)->GetParaverConfig()->getColorsTimelinePhysicalCommunications();
physicalColour = wxColour( rgbPhysicalColour.red, rgbPhysicalColour.green ,rgbPhysicalColour.blue );

logicalPen = wxPen( logicalColour );
physicalPen = wxPen( physicalColour );


wxString winTitle = GetTitle();
SetTitle( _("(Working...) ") + winTitle );

ProgressController *progress = nullptr;
if ( myWindow->getShowProgressBar() )
{
progress = ProgressController::create( myWindow->getKernel() );
progress->setHandler( progressFunctionTimeline, this );

#ifndef _WIN32
if( gTimeline::dialogProgress == nullptr )
{
gTimeline::dialogProgress = new wxProgressDialog( wxT("Drawing window..."),
wxT(""),
MAX_PROGRESS_BAR_VALUE,
this,
wxPD_CAN_ABORT|wxPD_AUTO_HIDE|\
wxPD_APP_MODAL|wxPD_ELAPSED_TIME|\
wxPD_ESTIMATED_TIME|wxPD_REMAINING_TIME );
}
++gTimeline::numberOfProgressDialogUsers;

gTimeline::dialogProgress->Pulse( winTitle + _( "\t" ) );
gTimeline::dialogProgress->Fit();
#endif 
}
vector<bool>         selected;
vector<TObjectOrder> selectedSet;
TObjectOrder beginRow = myWindow->getZoomSecondDimension().first;
TObjectOrder endRow =  myWindow->getZoomSecondDimension().second;
myWindow->getSelectedRows( myWindow->getLevel(), selected, true );
myWindow->getSelectedRows( myWindow->getLevel(), selectedSet, beginRow, endRow, true );

TObjectOrder maxObj;
if ( selectedSet.size() == 0 )
maxObj = endRow; 
else
maxObj = selectedSet[ selectedSet.size() - 1 ];

bufferImage.Create( drawZone->GetClientSize().GetWidth(), drawZone->GetClientSize().GetHeight() );
drawImage.Create( drawZone->GetClientSize().GetWidth(), drawZone->GetClientSize().GetHeight() );
commImage.Create( drawZone->GetClientSize().GetWidth(), drawZone->GetClientSize().GetHeight() );
eventImage.Create( drawZone->GetClientSize().GetWidth(), drawZone->GetClientSize().GetHeight() );

wxMemoryDC bufferDraw( bufferImage );
wxGraphicsContext *gc = wxGraphicsContext::Create( bufferDraw );
gc->SetAntialiasMode( wxANTIALIAS_NONE );

wxMemoryDC commdc( commImage );
wxMemoryDC eventdc( eventImage );
wxGraphicsContext *gcComm = wxGraphicsContext::Create( commdc );
gcComm->SetAntialiasMode( wxANTIALIAS_NONE );
wxGraphicsContext *gcEvent = wxGraphicsContext::Create( eventdc );
gcEvent->SetAntialiasMode( wxANTIALIAS_NONE );

commdc.SetBackgroundMode( wxTRANSPARENT );
commdc.SetBackground( *wxTRANSPARENT_BRUSH );
commdc.Clear();
eventdc.SetBackgroundMode( wxTRANSPARENT );
eventdc.SetBackground( *wxTRANSPARENT_BRUSH );
eventdc.Clear();
wxBitmap commMask;
commMask.Create( drawZone->GetClientSize().GetWidth(), drawZone->GetClientSize().GetHeight(), 1 );
wxMemoryDC commmaskdc( commMask );

wxGraphicsContext *gcCommMask = wxGraphicsContext::Create( commmaskdc );
gcCommMask->SetAntialiasMode( wxANTIALIAS_NONE );

commmaskdc.SetBackground( *wxBLACK_BRUSH );
commmaskdc.SetPen( wxPen( wxColour( 255, 255, 255 ), 1 ) );
commmaskdc.Clear();
wxBitmap eventMask;
eventMask.Create( drawZone->GetClientSize().GetWidth(), drawZone->GetClientSize().GetHeight(), 1 );
wxMemoryDC eventmaskdc( eventMask );

wxGraphicsContext *gcEventMask = wxGraphicsContext::Create( eventmaskdc );
gcEventMask->SetAntialiasMode( wxANTIALIAS_NONE );

eventmaskdc.SetBackground( *wxBLACK_BRUSH );
eventmaskdc.SetPen( wxPen( wxColour( 255, 255, 255 ), 1 ) );
eventmaskdc.Clear();

bufferDraw.SetBackground( wxBrush( backgroundColour ) );
bufferDraw.Clear();

wxClientDC dc( drawZone );
#ifdef __WXGTK__
dc.DrawBitmap( bufferImage, 0, 0, false );
drawZone->Update();
#endif
if( !drawAxis( bufferDraw, selectedSet ) )
{
SetTitle( winTitle );
myWindow->setReady( true );
return;
}
myWindow->init( myWindow->getWindowBeginTime(), CREATECOMMS + CREATEEVENTS );

drawCaution          = false;
drawCautionNegatives = false;

eventdc.SetTextForeground( *wxGREEN );
eventdc.SetTextBackground( backgroundColour );
eventdc.SetPen( *wxGREEN_PEN );
eventdc.SetBrush( *wxGREEN_BRUSH );
eventdc.SetBackgroundMode( wxTRANSPARENT );
eventmaskdc.SetPen( *wxWHITE_PEN );
eventmaskdc.SetBrush( *wxWHITE_BRUSH );

float magnify = float( myWindow->getPixelSize() );
TTime timeStep = (( myWindow->getWindowEndTime() - myWindow->getWindowBeginTime() )  * magnify) /
( drawZone->GetClientSize().GetWidth() - objectAxisPos - drawBorder - magnify );
PRV_INT32 timePos = objectAxisPos + 1;

vector< vector< TSemanticValue > > valuesToDraw;
vector< unordered_set< PRV_INT32 > > eventsToDraw;
#ifdef _MSC_VER
vector< unordered_set< commCoord > > commsToDraw;
#else
vector< unordered_set< commCoord, hashCommCoord > > commsToDraw;
#endif

vector< vector< vector< pair<TSemanticValue,TSemanticValue> > > > valuesToDrawPunctual;
if( myWindow->isPunctualColorSet() )
{
if( myWindow->getPunctualColorWindow() != nullptr )
myWindow->getPunctualColorWindow()->init( myWindow->getWindowBeginTime(), NOCREATE );

myWindow->computeSemanticPunctualParallel( selectedSet, selected,
timeStep, timePos, objectAxisPos,
objectPosList,
maxObj,
drawCaution,
valuesToDrawPunctual, eventsToDraw, commsToDraw,
progress );
}
else
{
myWindow->computeSemanticParallel( selectedSet, selected,
timeStep, timePos, objectAxisPos,
objectPosList,
maxObj,
drawCaution,
valuesToDraw, eventsToDraw, commsToDraw,
progress );
}

drawZeroAxis( bufferDraw, selectedSet );

PRV_UINT32 rowToDraw = 0;
if( myWindow->getWindowBeginTime() != myWindow->getWindowEndTime() )
{
for( vector< TObjectOrder >::iterator obj = selectedSet.begin(); obj != selectedSet.end(); ++obj )
{
TObjectOrder firstObj = *obj;
TObjectOrder lastObj = firstObj;
if( !myWindow->isFusedLinesColorSet() )
{
while( ( lastObj + 1 ) <= maxObj && objectPosList[ lastObj + 1 ] == objectPosList[ firstObj ] )
{
++obj;
lastObj = *obj;
}
}

if( myWindow->isPunctualColorSet() )
{
drawRow( bufferDraw, firstObj,
valuesToDrawPunctual[ rowToDraw ], eventsToDraw[ rowToDraw ], commsToDraw[ rowToDraw ],
eventdc, eventmaskdc, commdc, commmaskdc );
}
else
{
drawRow( bufferDraw, firstObj,
valuesToDraw[ rowToDraw ], eventsToDraw[ rowToDraw ], commsToDraw[ rowToDraw ],
eventdc, eventmaskdc, commdc, commmaskdc );
}
++rowToDraw;

if( myWindow->isPunctualColorSet() )
{
if( rowToDraw >= valuesToDrawPunctual.size() )
break;
}
else
{
if( rowToDraw >= valuesToDraw.size() )
break;
}
}
}

if( ( myWindow->getShowProgressBar() ) && ( gTimeline::dialogProgress != nullptr ) )
{ 
--gTimeline::numberOfProgressDialogUsers;
if ( gTimeline::numberOfProgressDialogUsers == 0 )
{
gTimeline::dialogProgress->Show( false );
delete gTimeline::dialogProgress;
gTimeline::dialogProgress = nullptr;

if ( progress != nullptr )
delete progress;
}
}


bufferDraw.SelectObject( wxNullBitmap );
bufferDraw.SelectObject( drawImage );
bufferDraw.DrawBitmap( bufferImage, 0, 0, false );

doDrawCaution( bufferDraw );

eventmaskdc.SetPen( *wxBLACK_PEN );
eventmaskdc.SetBrush( *wxBLACK_BRUSH );
eventmaskdc.DrawRectangle( 0, 0, objectAxisPos + 1, drawZone->GetSize().GetHeight() );
eventmaskdc.SelectObject(wxNullBitmap);
wxMask *mask = new wxMask( eventMask );
eventImage.SetMask( mask );

if( myWindow->getDrawFlags() )
bufferDraw.DrawBitmap( eventImage, 0, 0, true );

commmaskdc.SetPen( *wxBLACK_PEN );
commmaskdc.SetBrush( *wxBLACK_BRUSH );
commmaskdc.DrawRectangle( 0, 0, objectAxisPos + 1, drawZone->GetSize().GetHeight() );
commmaskdc.DrawRectangle( drawZone->GetSize().GetWidth() - drawBorder, 0, drawBorder, drawZone->GetSize().GetHeight() );
commmaskdc.DrawRectangle( 0, timeAxisPos, drawZone->GetSize().GetWidth(), drawZone->GetSize().GetHeight() - timeAxisPos );
commmaskdc.SelectObject(wxNullBitmap);
mask = new wxMask( commMask );
commImage.SetMask( mask );

if( myWindow->getDrawCommLines() )
bufferDraw.DrawBitmap( commImage, 0, 0, true );

delete gc;
delete gcEvent;
delete gcComm;
delete gcCommMask;
delete gcEventMask;

SetTitle( winTitle );

myWindow->setReady( true );

drawZone->Refresh();

SetFocus();
}


bool gTimeline::drawAxis( wxDC& dc, vector<TObjectOrder>& selected )
{
PRV_UINT32 precision = 0;

size_t numObjects = selected.size();
float magnify = float( myWindow->getPixelSize() );

rgb rgbAxisColour = ((paraverMain *)parent)->GetParaverConfig()->getColorsTimelineAxis();
wxColour axisColour = wxColour( rgbAxisColour.red, rgbAxisColour.green ,rgbAxisColour.blue );

dc.SetPen( wxPen( axisColour, 1 ) );
dc.SetTextForeground( axisColour );

dc.SetFont( timeFont );
wxSize timeExt = dc.GetTextExtent( wxString::FromUTF8( LabelConstructor::timeLabel( myWindow->traceUnitsToWindowUnits( myWindow->getWindowBeginTime() ),
myWindow->getTimeUnit(), precision ).c_str() ) );
timeAxisPos = dc.GetSize().GetHeight() - ( drawBorder + timeExt.GetHeight() + drawBorder );
if( timeAxisPos + drawBorder + 1 > dc.GetSize().GetHeight() )
{
return false;
}

dc.SetFont( objectFont );
wxSize objectExt, tmpExt;
TTraceLevel endLevel = TTraceLevel::THREAD;
string tmpLongestLabel;
wxString tmpCurrentLabel;
size_t tmpMaxLength = 0;
size_t tmpCurrentMaxLength = 0;

switch( myWindow->getObjectAxisSize() )
{
case TObjectAxisSize::CURRENT_LEVEL:
if( myWindow->isFusedLinesColorSet() )
{
objectExt = dc.GetTextExtent( wxString::FromUTF8( LabelConstructor::semanticLabel( myWindow,
myWindow->getMaximumY(),
false,
ParaverConfig::getInstance()->getTimelinePrecision(),
false ).c_str() ) ) + wxSize( 5, 0 );
break;
}

tmpCurrentMaxLength = LabelConstructor::objectLabel( myWindow->getTrace()->getLevelObjects( myWindow->getLevel() ) - 1,
myWindow->getLevel(),
myWindow->getTrace() ).length();

tmpMaxLength = myWindow->getTrace()->getMaxLengthRow( myWindow->getLevel() );

if ( tmpCurrentMaxLength > tmpMaxLength ) 
tmpMaxLength = tmpCurrentMaxLength;

tmpLongestLabel = string( tmpMaxLength, 'Z' );
objectExt = dc.GetTextExtent( wxString::FromUTF8( tmpLongestLabel.c_str() ) );
break;

case TObjectAxisSize::ALL_LEVELS:
if( myWindow->getTrace()->existResourceInfo() )
endLevel = TTraceLevel::CPU;

for( TTraceLevel iLevel = TTraceLevel::WORKLOAD; iLevel <= endLevel; ++iLevel )
{
tmpCurrentMaxLength = myWindow->getTrace()->getMaxLengthRow( iLevel );
tmpCurrentLabel = wxString::FromUTF8( LabelConstructor::objectLabel( myWindow->getTrace()->getLevelObjects( iLevel ) - 1,
iLevel,
myWindow->getTrace() ).c_str() );
if ( tmpCurrentMaxLength < tmpCurrentLabel.Len() )
tmpCurrentMaxLength = tmpCurrentLabel.Len();

if ( tmpMaxLength < tmpCurrentMaxLength )
tmpMaxLength = tmpCurrentMaxLength;
}

tmpLongestLabel = string( tmpMaxLength, 'Z' );
objectExt = dc.GetTextExtent( wxString::FromUTF8( tmpLongestLabel.c_str() ) );
break;

case TObjectAxisSize::ZERO_PERC:
objectExt = wxSize( 0 , 0 );
break;

case TObjectAxisSize::FIVE_PERC:
objectExt = wxSize( dc.GetSize().GetWidth() * 0.05 , 0 );
break;

case TObjectAxisSize::TEN_PERC:
objectExt = wxSize( dc.GetSize().GetWidth() * 0.10 , 0 );
break;

case TObjectAxisSize::TWENTYFIVE_PERC:
objectExt = wxSize( dc.GetSize().GetWidth() * 0.25 , 0 );
break;

default:
break;
}

objectAxisPos = drawBorder + objectExt.GetWidth() + drawBorder;
if( objectAxisPos + drawBorder + 1 > dc.GetSize().GetWidth() )
{
return false;
}

dc.DrawLine( objectAxisPos, drawBorder,
objectAxisPos, timeAxisPos );
dc.DrawLine( objectAxisPos, timeAxisPos,
dc.GetSize().GetWidth() - drawBorder, timeAxisPos );

if( myWindow->isFusedLinesColorSet() )
{
dc.DrawText( wxString::FromUTF8( LabelConstructor::semanticLabel( myWindow,
myWindow->getMaximumY(),
false,
ParaverConfig::getInstance()->getTimelinePrecision(),
false ).c_str() ),
drawBorder, drawBorder );

dc.DrawText( wxString::FromUTF8( LabelConstructor::semanticLabel( myWindow,
myWindow->getMinimumY(),
false,
ParaverConfig::getInstance()->getTimelinePrecision(),
false ).c_str() ),
drawBorder, timeAxisPos - objectExt.GetHeight() );

if( myWindow->getMinimumY() < 0.0 && myWindow->getMaximumY() > 0.0 )
{
TSemanticValue relativeZero = myWindow->getMaximumY() / ( myWindow->getMaximumY() - myWindow->getMinimumY() );
dc.DrawText( wxString::FromUTF8( LabelConstructor::semanticLabel( myWindow,
0.0,
false,
ParaverConfig::getInstance()->getTimelinePrecision(),
false ).c_str() ),
drawBorder, ( relativeZero * ( timeAxisPos - objectExt.GetHeight() ) ) + drawBorder + 1 );
}
else
{
dc.DrawText( wxString::FromUTF8( LabelConstructor::semanticLabel( myWindow,
( myWindow->getMaximumY() + myWindow->getMinimumY() ) / 2.0,
false,
ParaverConfig::getInstance()->getTimelinePrecision(),
false ).c_str() ),
drawBorder, ( timeAxisPos - objectExt.GetHeight() ) / 2 );
}
}
else
{
wxCoord y;
double inc = (double)( timeAxisPos - drawBorder ) / (double)( numObjects );
bool drawLabel = true;
TObjectOrder power2 = 1;

objectPosList.clear();
objectPosList.insert( objectPosList.begin(), myWindow->getWindowLevelObjects(), 0 );
objectHeight = 1;
vector< TObjectOrder >::iterator it = selected.begin();

wxCoord accumHeight = 0;
wxCoord stepHeight = objectExt.GetHeight();

int numpixels = timeAxisPos - 2 * drawBorder; 
int everyobj = 1; 
bool printlast = numpixels > objectFont.GetPointSize(); 
bool printall = numpixels > 0 && numpixels > int( numObjects * objectFont.GetPointSize() ); 


if (!printall && printlast)
{

int every = 1;
while( ( numObjects * objectFont.GetPointSize() ) / every > numpixels-objectFont.GetPointSize() )
every = every << 1;

every = every << 1;

if( int( numObjects ) > every )
everyobj = numObjects / ( numObjects / every );
else
everyobj = numObjects;
}

for( TObjectOrder obj = (TObjectOrder)0; obj < numObjects; obj++ )
{
y = ( (wxCoord) ( inc * ( obj ) ) ) + drawBorder;
if( ( inc * 0.25 ) >= 1.0 )
{
if( obj > (TObjectOrder)0 && objectHeight < ( y - objectPosList[ selected[ obj - 1 ] ] ) * 0.75 )
objectHeight = ( y - objectPosList[ selected[ obj - 1 ] ] ) * 0.75;
y += (wxCoord)( inc * 0.25 );
}
else
{
if( obj > (TObjectOrder)0 && objectHeight < ( y - objectPosList[ selected[ obj - 1 ] ] ) )
objectHeight = ( y - objectPosList[ selected[ obj - 1 ] ] );
}
objectPosList[ selected[ obj ] ] = y;

switch( myWindow->getObjectLabels() )
{
case TObjectLabels::ALL_LABELS:
drawLabel = true;
break;

case  TObjectLabels::SPACED_LABELS:
if( ( printlast && ( obj % everyobj == 0 || obj == numObjects - 1 ) ) ||
( !printlast && obj == 0 ) )
drawLabel = true;
else
drawLabel = false;
break;

case TObjectLabels::POWER2_LABELS:
if( obj == power2 - 1 )
{
drawLabel = true;
power2 = power2 << 1;
}
else
drawLabel = false;
break;
default:
break;
}

if( ( printall || drawLabel ) &&
!( myWindow->getObjectAxisSize() == TObjectAxisSize::ZERO_PERC ) )
{
if( myWindow->getLevel() == TTraceLevel::CPU || myWindow->getLevel() == TTraceLevel::NODE || myWindow->getLevel() == TTraceLevel::SYSTEM )
dc.DrawText( wxString::FromUTF8( LabelConstructor::objectLabel( *it + 1, myWindow->getLevel(), myWindow->getTrace() ).c_str() ),
drawBorder, y );
else
dc.DrawText( wxString::FromUTF8( LabelConstructor::objectLabel( *it, myWindow->getLevel(), myWindow->getTrace() ).c_str() ),
drawBorder, y );
accumHeight += stepHeight;
}

++it;
}

if( numObjects == 1 )
objectHeight = ( timeAxisPos - objectPosList[ selected[ 0 ] ] );
else
{
if( ( inc * 0.25 ) < 1.0 && magnify > objectHeight )
{
for( vector<wxCoord>::iterator it = objectPosList.begin(); it != objectPosList.end(); ++it )
*it = ( floor( ( *it - drawBorder ) / magnify ) * magnify ) + drawBorder;
objectHeight = magnify;
}
}
}

dc.SetFont( timeFont );
dc.DrawText( wxString::FromUTF8( LabelConstructor::timeLabel( myWindow->traceUnitsToWindowUnits( myWindow->getWindowBeginTime() ),
myWindow->getTimeUnit(), precision ).c_str() ),
objectAxisPos, timeAxisPos + drawBorder );
dc.DrawText( wxString::FromUTF8( LabelConstructor::timeLabel( myWindow->traceUnitsToWindowUnits( myWindow->getWindowEndTime() ),
myWindow->getTimeUnit(), precision ).c_str() ),
dc.GetSize().GetWidth() -
( dc.GetTextExtent( wxString::FromUTF8( LabelConstructor::timeLabel( myWindow->traceUnitsToWindowUnits( myWindow->getWindowEndTime() ),
myWindow->getTimeUnit(), precision ).c_str() ) )
.GetWidth() + drawBorder ),
timeAxisPos + drawBorder );

dc.SetPen( wxPen( backgroundColour ) );
dc.SetBrush( wxBrush( backgroundColour ) );
dc.DrawRectangle( objectAxisPos + 1, 0, drawZone->GetSize().GetWidth(), timeAxisPos );

return true;
}

void gTimeline::drawZeroAxis( wxDC& dc, vector<TObjectOrder>& selected )
{
if( myWindow->getMaximumY() > 0.0 && myWindow->getMinimumY() < 0.0 )
{
rgb rgbAxisColour = ((paraverMain *)parent)->GetParaverConfig()->getColorsTimelineZeroDashLine();
wxColour axisColour;
if( rgbAxisColour == ((paraverMain *)parent)->GetParaverConfig()->getColorsTimelineAxis() )
axisColour = wxColour( rgbAxisColour.red - 1, rgbAxisColour.green ,rgbAxisColour.blue );
else
axisColour = wxColour( rgbAxisColour.red, rgbAxisColour.green ,rgbAxisColour.blue );
dc.SetPen( wxPen( axisColour, 1, wxPENSTYLE_LONG_DASH ) );

TSemanticValue relativeZero = Normalizer::calculate( 0.0,
myWindow->getMinimumY(),
myWindow->getMaximumY(),
GetMyWindow()->getGradientColor().getGradientFunction(), 
true );

if( myWindow->isFusedLinesColorSet() )
{
wxCoord zeroAxisHeight = ( timeAxisPos - drawBorder ) - ( relativeZero * ( timeAxisPos - drawBorder ) ) + drawBorder + 1;
dc.DrawLine( objectAxisPos, zeroAxisHeight, dc.GetSize().GetWidth() - drawBorder, zeroAxisHeight );
}
else if( myWindow->isFunctionLineColorSet() || myWindow->isPunctualColorSet() )
{
wxCoord lastPos = 0;
wxCoord relativePos = ( 1.0 - relativeZero ) * objectHeight;
for( vector<TObjectOrder>::iterator it = selected.begin(); it != selected.end(); ++it )
{
wxCoord tmpObjPos = objectPosList[ *it ];
if( tmpObjPos != lastPos )
{
lastPos = tmpObjPos;
dc.DrawLine( objectAxisPos, relativePos + lastPos + 1,
dc.GetSize().GetWidth() - drawBorder, relativePos + lastPos + 1 );

}
}
}
}
}

#ifdef _MSC_VER
template<typename ValuesType>
void gTimeline::drawRow( wxDC& dc,
TObjectOrder firstRow,
vector< ValuesType >& valuesToDraw,
unordered_set< PRV_INT32 >& eventsToDraw,
unordered_set< commCoord >& commsToDraw,
wxMemoryDC& eventdc, wxMemoryDC& eventmaskdc,
wxMemoryDC& commdc, wxMemoryDC& commmaskdc )
#else
template<typename ValuesType>
void gTimeline::drawRow( wxDC& dc,
TObjectOrder firstRow,
vector< ValuesType >& valuesToDraw,
unordered_set< PRV_INT32 >& eventsToDraw,
unordered_set< commCoord, hashCommCoord >& commsToDraw,
wxMemoryDC& eventdc, wxMemoryDC& eventmaskdc,
wxMemoryDC& commdc, wxMemoryDC& commmaskdc )
#endif
{
float magnify  = float( myWindow->getPixelSize() );
if( myWindow->isPunctualColorSet() )
{
rgb rgbPunctualColour = ((paraverMain *)parent)->GetParaverConfig()->getColorsTimelinePunctual();
wxColour punctualColor( rgbPunctualColour.red,
rgbPunctualColour.green,
rgbPunctualColour.blue );
dc.SetPen( punctualColor );
dc.SetBrush( punctualColor );
}

wxCoord objectPos;
if( myWindow->isFusedLinesColorSet() )
objectPos = 0;
else
objectPos = objectPosList[ firstRow ];
wxCoord timePos   = objectAxisPos + 1;
int lineLastPos   = 0;

for( typename vector< ValuesType >::iterator it = valuesToDraw.begin(); it != valuesToDraw.end(); ++it )
{
if( myWindow->isPunctualColorSet() )
{
drawRowPunctual( dc, *it, objectPos, timePos, magnify );
}
else if( myWindow->isCodeColorSet() ||
myWindow->isGradientColorSet() ||
myWindow->isNotNullGradientColorSet() ||
myWindow->isAlternativeGradientColorSet() )
{
drawRowColor( dc, *it, objectPos, timePos, magnify );
}
else if( myWindow->isFunctionLineColorSet() )
{
drawRowFunction( dc, *it, lineLastPos, objectPos, timePos, magnify );
}
else if( myWindow->isFusedLinesColorSet() )
{
drawRowFusedLines( dc, *it, lineLastPos, firstRow, timePos, magnify );
}

timePos += (int) magnify ;
}

if( !myWindow->isFusedLinesColorSet() )
{
drawRowEvents( eventdc, eventmaskdc, objectPosList[ firstRow ], eventsToDraw );
drawRowComms( commdc, commmaskdc, objectPosList[ firstRow ], commsToDraw );
}
}


template<typename ValuesType>
void gTimeline::drawRowColor( wxDC& dc, ValuesType valueToDraw, wxCoord objectPos, wxCoord timePos, float magnify )
{
abort();
}

template<>
void gTimeline::drawRowColor( wxDC& dc, TSemanticValue valueToDraw, wxCoord objectPos, wxCoord timePos, float magnify )
{
rgb colorToDraw = myWindow->calcColor( valueToDraw, *myWindow );

if( myWindow->isCodeColorSet() && valueToDraw < 0.0 )
drawCautionNegatives = true;

semanticValuesToColor[ valueToDraw ] = colorToDraw;
semanticColorsToValue[ colorToDraw ].insert( valueToDraw );

dc.SetPen( wxPen( wxColour( colorToDraw.red, colorToDraw.green, colorToDraw.blue ) ) );

if ( magnify == 1.0 )
{
if( objectPos + objectHeight < timeAxisPos )
dc.DrawLine( timePos, objectPos, timePos, objectPos + objectHeight );
else
dc.DrawLine( timePos, objectPos, timePos, timeAxisPos - 1 );
}
else
{
dc.SetBrush( wxBrush( wxColour( colorToDraw.red, colorToDraw.green, colorToDraw.blue ) ) );
dc.DrawRectangle( timePos , objectPos, magnify , objectHeight );
}
}


template<typename ValuesType>
void gTimeline::drawRowFunction( wxDC& dc, ValuesType valueToDraw, int& lineLastPos, wxCoord objectPos, wxCoord timePos, float magnify )
{
abort();
}

template<>
void gTimeline::drawRowFunction( wxDC& dc, TSemanticValue valueToDraw, int& lineLastPos, wxCoord objectPos, wxCoord timePos, float magnify )
{
TSemanticValue realMin = myWindow->getMinimumY();

if( myWindow->getSemanticScaleMinAtZero() && myWindow->getExistSemanticZero() && myWindow->getMinimumY() > 0.0 )
realMin = 0.0;

if( valueToDraw < realMin )
valueToDraw = realMin;
else if( valueToDraw > myWindow->getMaximumY() )
valueToDraw = myWindow->getMaximumY();

double normalizedSemanticValue = Normalizer::calculate( valueToDraw, 
myWindow->getMinimumY(),
myWindow->getMaximumY(),
GetMyWindow()->getGradientColor().getGradientFunction(),
true );
int currentPos = objectHeight * normalizedSemanticValue;
semanticPixelsToValue[ currentPos ].insert( valueToDraw );

dc.SetPen( foregroundColour );
if( currentPos != lineLastPos )
{
int from = ( currentPos > lineLastPos ) ? currentPos : lineLastPos;
int to   = ( currentPos < lineLastPos ) ? currentPos : lineLastPos;
dc.DrawLine( timePos, objectPos + objectHeight - from,
timePos, objectPos + objectHeight - to + 1 );
if( magnify > 1.0 )
dc.DrawLine( timePos,           objectPos + objectHeight - currentPos,
timePos + magnify, objectPos + objectHeight - currentPos ); 
}
else
{
if ( magnify == 1.0 )
dc.DrawPoint( timePos, objectPos + objectHeight - currentPos );
else
dc.DrawLine( timePos,           objectPos + objectHeight - currentPos,
timePos + magnify, objectPos + objectHeight - currentPos ); 
}

lineLastPos = currentPos;
}


template<typename ValuesType>
void gTimeline::drawRowFusedLines( wxDC& dc, ValuesType valueToDraw, int& lineLastPos, TObjectOrder whichObject, wxCoord timePos, float magnify )
{
abort();
}

template<>
void gTimeline::drawRowFusedLines( wxDC& dc, TSemanticValue valueToDraw, int& lineLastPos, TObjectOrder whichObject, wxCoord timePos, float magnify )
{
TSemanticValue realMin = myWindow->getMinimumY();

if( myWindow->getSemanticScaleMinAtZero() && myWindow->getExistSemanticZero() && myWindow->getMinimumY() > 0.0 )
realMin = 0.0;

if( valueToDraw < realMin )
valueToDraw = realMin;
else if( valueToDraw > myWindow->getMaximumY() )
valueToDraw = myWindow->getMaximumY();

double tmpPos = ( valueToDraw - realMin ) 
/ ( myWindow->getMaximumY() - realMin );
int currentPos = ( timeAxisPos - drawBorder ) * tmpPos;

rgb colorToDraw = myWindow->getCodeColor().calcColor( whichObject + 1, 0, whichObject + 1, false );
semanticColorsToValue[ colorToDraw ].insert( whichObject );
dc.SetPen( wxPen( wxColour( colorToDraw.red, colorToDraw.green, colorToDraw.blue ) ) );
if( currentPos != lineLastPos )
{
int from = ( currentPos > lineLastPos ) ? currentPos : lineLastPos;
int to   = ( currentPos < lineLastPos ) ? currentPos : lineLastPos;
dc.DrawLine( timePos, timeAxisPos - from,
timePos, timeAxisPos - to + 1 );
if( magnify > 1.0 )
dc.DrawLine( timePos,           timeAxisPos - currentPos,
timePos + magnify, timeAxisPos - currentPos ); 
}
else
{
if ( magnify == 1.0 )
dc.DrawPoint( timePos, timeAxisPos - currentPos );
else
dc.DrawLine( timePos,           timeAxisPos - currentPos,
timePos + magnify, timeAxisPos - currentPos ); 
}

lineLastPos = currentPos;
}


template<typename ValuesType>
void gTimeline::drawRowPunctual( wxDC& dc, ValuesType& valuesToDrawList, wxCoord objectPos, wxCoord timePos, float magnify )
{
abort();
}

template<>
void gTimeline::drawRowPunctual( wxDC& dc, vector< pair<TSemanticValue,TSemanticValue> >& valuesToDrawList, wxCoord objectPos, wxCoord timePos, float magnify )
{
TSemanticValue realMin = myWindow->getMinimumY();

if( myWindow->getSemanticScaleMinAtZero() && myWindow->getExistSemanticZero() && myWindow->getMinimumY() > 0.0 )
realMin = 0.0;

TSemanticValue valueToDraw;
for( vector< pair<TSemanticValue,TSemanticValue> >::iterator itValues = valuesToDrawList.begin(); itValues != valuesToDrawList.end(); ++itValues )
{
valueToDraw = (*itValues).first;

if( valueToDraw < realMin || valueToDraw > myWindow->getMaximumY() )
continue;

double normalizedSemanticValue = Normalizer::calculate( valueToDraw,
myWindow->getMinimumY(),
myWindow->getMaximumY(),
myWindow->getGradientColor().getGradientFunction(),
true );
int currentPos = floor( ( (double)objectHeight / (double)magnify ) * normalizedSemanticValue ) * magnify;

if( myWindow->getPunctualColorWindow() != nullptr )
{
TSemanticValue valueToColor = (*itValues).second;
rgb colorToDraw = myWindow->getPunctualColorWindow()->calcColor( valueToColor, *( myWindow->getPunctualColorWindow() ) );

semanticValuesToColor[ valueToColor ] = colorToDraw;
semanticColorsToValue[ colorToDraw ].insert( valueToColor );

dc.SetPen( wxPen( wxColour( colorToDraw.red, colorToDraw.green, colorToDraw.blue ) ) );
dc.SetBrush( wxBrush( wxColour( colorToDraw.red, colorToDraw.green, colorToDraw.blue ) ) );
}
if( magnify == 1.0 )
dc.DrawPoint( timePos, objectPos + objectHeight - currentPos - 1 );
else
dc.DrawRectangle( timePos, objectPos + objectHeight - currentPos - magnify, magnify, magnify );
}
}


void gTimeline::drawRowEvents( wxDC& eventdc, wxDC& eventmaskdc, TObjectOrder rowPos, unordered_set< PRV_INT32 >& eventsToDraw )
{
for( unordered_set< PRV_INT32 >::iterator it = eventsToDraw.begin(); it != eventsToDraw.end(); ++it )
{
eventdc.DrawLine( *it, rowPos - 6, *it, rowPos );
eventdc.DrawLine( *it+1, rowPos - 6, *it+1, rowPos-3 );
eventdc.DrawLine( *it+2, rowPos - 6, *it+2, rowPos-3 );
eventdc.DrawLine( *it+3, rowPos - 6, *it+3, rowPos-3 );
eventdc.DrawLine( *it+4, rowPos - 6, *it+4, rowPos-3 );

eventmaskdc.DrawLine( *it, rowPos - 6, *it, rowPos );
eventmaskdc.DrawLine( *it+1, rowPos - 6, *it+1, rowPos-3 );
eventmaskdc.DrawLine( *it+2, rowPos - 6, *it+2, rowPos-3 );
eventmaskdc.DrawLine( *it+3, rowPos - 6, *it+3, rowPos-3 );
eventmaskdc.DrawLine( *it+4, rowPos - 6, *it+4, rowPos-3 );
}

}


#ifdef _MSC_VER
void gTimeline::drawRowComms( wxDC& commdc, wxDC& commmaskdc, TObjectOrder rowPos, unordered_set< commCoord >& commsToDraw )
#else
void gTimeline::drawRowComms( wxDC& commdc, wxDC& commmaskdc, TObjectOrder rowPos, unordered_set< commCoord, hashCommCoord >& commsToDraw )
#endif
{
#ifdef _MSC_VER
for( unordered_set<commCoord>::iterator it = commsToDraw.begin(); it != commsToDraw.end(); ++it )
#else
for( unordered_set<commCoord,hashCommCoord>::iterator it = commsToDraw.begin(); it != commsToDraw.end(); ++it )
#endif
{
if( it->recType & LOG )
commdc.SetPen( logicalPen );
else if( it->recType & PHY )
commdc.SetPen(  physicalPen );

if( it->toRow != rowPos )
{
commdc.DrawLine( it->toTime, it->toRow,
it->fromTime, rowPos );
commmaskdc.DrawLine( it->toTime, it->toRow,
it->fromTime, rowPos );
}
else
{
commdc.DrawLine( it->toTime, it->toRow,
it->fromTime, rowPos + ( objectHeight / 2 ) );
commmaskdc.DrawLine( it->toTime, it->toRow,
it->fromTime, rowPos + ( objectHeight / 2 ) );
}
}
}


void gTimeline::OnScrolledWindowEraseBackground( wxEraseEvent& event )
{
#ifdef __WXMAC__
event.Skip();
#endif
}



void gTimeline::OnScrolledWindowPaint( wxPaintEvent& event )
{
wxPaintDC dc( drawZone );

if( !myWindow->getReady() )
return;

if( drawImage.IsOk() )
dc.DrawBitmap( drawImage, 0, 0, false );
}



void gTimeline::OnScrolledWindowSize( wxSizeEvent& event )
{
if( !IsShown() )
return;
else if( canRedraw &&
( event.GetSize().GetWidth() != myWindow->getWidth() ||
event.GetSize().GetHeight() != myWindow->getHeight() ) )
{
#ifndef _WIN32
if( !splitChanged )
{
#endif
myWindow->setWidth( event.GetSize().GetWidth() );
myWindow->setHeight( event.GetSize().GetHeight() );
#ifndef _WIN32
}
#endif
timerSize->StartOnce( TIMER_SIZE_DURATION );
}

event.Skip();
}



void gTimeline::OnIdle( wxIdleEvent& event )
{

if( paraverMain::stopOnIdle )
return;

if( myWindow->getDestroy() )
return;

string composedName = myWindow->getName() + " @ " +
myWindow->getTrace()->getTraceNameNumbered();

this->SetTitle( wxString::FromUTF8( composedName.c_str() ) );

#ifdef _WIN32
if( !firstUnsplit )
{
firstUnsplit = true;
splitter->Unsplit();
drawZone->SetClientSize( myWindow->getWidth(), myWindow->getHeight() );
canRedraw = true;
}
#else
canRedraw = true;
#endif

if( !wxparaverApp::mainWindow->IsIconized() && myWindow->getShowWindow() )
{
this->Show();
if( !myWindow->getReady() && !wxparaverApp::mainWindow->GetSomeWinIsRedraw() )
{
myWindow->setReady( true );
wxparaverApp::mainWindow->SetSomeWinIsRedraw( true );
redraw();
wxparaverApp::mainWindow->SetSomeWinIsRedraw( false );
}
}
else
{
this->Show( false );
}

if( myWindow->getRaiseWindow() )
{
if( !myWindow->getShowWindow() )
myWindow->setShowWindow( true );
else
this->Raise();

myWindow->setRaiseWindow( false );
}


int currentDisplay = wxDisplay::GetFromWindow( this );
if ( currentDisplay != wxNOT_FOUND && currentDisplay >= 0 )
{
wxDisplay tmpDisplay( currentDisplay );
myWindow->setPosX( this->GetPosition().x - tmpDisplay.GetGeometry().x );
myWindow->setPosY( this->GetPosition().y - tmpDisplay.GetGeometry().y );
}

bool state = false;
if( myWindow->isFunctionLineColorSet() || myWindow->isPunctualColorSet() || myWindow->isFusedLinesColorSet() )
state = true;
initialSemanticLabel->Show( state );
initialSemanticText->Show( state );
finalSemanticLabel->Show( state );
finalSemanticText->Show( state );
slopeLabel->Show( state );
slopeText->Show( state );
timingZone->Layout();
}



void gTimeline::OnCloseWindow( wxCloseEvent& event )
{
myWindow->setShowWindow( false );
}



void gTimeline::OnScrolledWindowLeftDown( wxMouseEvent& event )
{
wxBitmap tmpImageCaution( caution_xpm );
wxBitmap tmpImageAutoRedraw( autoredraw_refresh_xpm );
if( event.GetX() < tmpImageCaution.GetWidth() + drawBorder &&
event.GetY() > drawZone->GetSize().GetHeight() - tmpImageCaution.GetHeight() - drawBorder )
{
if( drawCautionNegatives )
{
wxMessageDialog dialog( this,
wxT( "Negative values are drawn as zero in code color view." ),
wxT( "Code color negatives" ),
wxOK|wxICON_EXCLAMATION );
dialog.ShowModal();
}
else 
{
wxCommandEvent dummyEvent;
OnPopUpFitSemanticScale( dummyEvent );
}

return;
}
else if( enabledAutoRedrawIcon &&
event.GetX() < tmpImageAutoRedraw.GetWidth() + 2 * drawBorder &&
event.GetY() < tmpImageAutoRedraw.GetWidth() + 2 * drawBorder )
{
myWindow->setForceRedraw( true );
return;
}

drawZone->SetFocus();
zooming = true;
firstMotionEvent = event;
zoomBeginX = event.GetX();
zoomBeginY = event.GetY();
}



void gTimeline::OnScrolledWindowLeftUp( wxMouseEvent& event )
{
TObjectOrder beginRow = myWindow->getZoomSecondDimension().first;
TObjectOrder endRow = myWindow->getZoomSecondDimension().second;
wxMemoryDC dc( bufferImage );
bool outOfDraw = false;

if( escapePressed )
{
escapePressed = false;
return;
}

if( timing && !wxGetApp().GetGlobalTiming() )
{
zooming = false;
return;
}

zoomEndX = event.GetX();
zoomEndY = event.GetY();
zoomXY = event.ControlDown();

if( event.ShiftDown() )
{
zooming = false;
zoomXY = false;
MousePanLeftUp( event );
return;
}

if( zooming )
{
if( zoomEndX < zoomBeginX )
{
long tmp = zoomEndX; zoomEndX = zoomBeginX; zoomBeginX = tmp;
}
if( zoomBeginX < objectAxisPos )
zoomBeginX = 0;
else
zoomBeginX -= objectAxisPos;
}
if( zoomEndX < objectAxisPos )
{
zoomEndX = 0;
outOfDraw = true;
}
else if( zoomEndX > dc.GetSize().GetWidth() - drawBorder )
{
zoomEndX = dc.GetSize().GetWidth() - drawBorder - objectAxisPos;
outOfDraw = true;
}
else
zoomEndX -= objectAxisPos;

TTime timeStep = ( myWindow->getWindowEndTime() - myWindow->getWindowBeginTime() ) /
( dc.GetSize().GetWidth() - objectAxisPos - drawBorder );

TTime endTime = ( timeStep * zoomEndX ) + myWindow->getWindowBeginTime();
TTime beginTime = ( timeStep * zoomBeginX ) + myWindow->getWindowBeginTime();

if( wxGetApp().GetGlobalTiming() )
{
if( zooming )
{
zooming = false;
if( beginTime != endTime )
{
wxGetApp().SetGlobalTimingBegin( beginTime );
wxGetApp().SetGlobalTimingEnd( endTime );
wxGetApp().DeactivateGlobalTiming();
}
else
{
if( wxGetApp().GetGlobalTimingBeginIsSet() )
{
wxGetApp().SetGlobalTimingEnd( endTime );
wxGetApp().DeactivateGlobalTiming();
}
else
{
wxGetApp().SetGlobalTimingBegin( endTime );
wxGetApp().SetGlobalTimingBeginIsSet( true );
}
}
}
return;
}

if( zooming )
{
if( zoomEndY < zoomBeginY )
{
long tmp = zoomEndY; zoomEndY = zoomBeginY; zoomBeginY = tmp;
}
if( zoomBeginY > timeAxisPos )
zoomBeginY = timeAxisPos - 1;
if( zoomBeginY < drawBorder )
zoomBeginY = drawBorder + 1;
}

if( zoomEndY > timeAxisPos )
{
zoomEndY = timeAxisPos;
outOfDraw = true;
}
if( zoomEndY < drawBorder )
{
zoomEndY = drawBorder;
outOfDraw = true;
}

vector<TObjectOrder> selected;
myWindow->getSelectedRows( myWindow->getLevel(), selected, beginRow, endRow, true );
TObjectOrder numObjects = selected.size();
double heightPerRow = (double)( timeAxisPos - drawBorder - 1 ) / (double)numObjects;


beginRow = TObjectOrder( floor( (zoomBeginY - drawBorder - 1) / heightPerRow ) );
endRow = TObjectOrder( floor( (zoomEndY - drawBorder - 1) / heightPerRow ) );

if( endRow >= numObjects )
endRow = numObjects - 1;
if( beginRow >= numObjects )
beginRow = numObjects - 1;

beginRow = selected[ beginRow ];
endRow   = selected[ endRow ];

if( zooming && myWindow->getReady() &&
zoomEndX - zoomBeginX > 5.0 &&
( ( zoomXY && zoomEndY - zoomBeginY > 5.0 ) || !zoomXY ) )
{
if( !zoomXY )
{
beginRow = myWindow->getZoomSecondDimension().first;
endRow =  myWindow->getZoomSecondDimension().second;
}
if( ( endTime - beginTime ) < 10 )
endTime = beginTime + 10;

if( zoomXY && 
( selected.size() == 1 && ( myWindow->isFunctionLineColorSet() || myWindow->isPunctualColorSet() )
|| myWindow->isFusedLinesColorSet() )
)
{
if( zoomBeginY < drawBorder )
zoomBeginY = timeAxisPos - drawBorder;
else
zoomBeginY = timeAxisPos - zoomBeginY;

if( zoomEndY < drawBorder )
zoomEndY = timeAxisPos - drawBorder;
else
zoomEndY = timeAxisPos - zoomEndY;

TSemanticValue semanticStep = ( myWindow->getMaximumY() - myWindow->getMinimumY() ) /
( timeAxisPos - drawBorder );
TSemanticValue beginSemantic = ( semanticStep * zoomEndY ) + myWindow->getMinimumY();
TSemanticValue endSemantic = ( semanticStep * zoomBeginY ) + myWindow->getMinimumY();

myWindow->setMinimumY( beginSemantic );
myWindow->setMaximumY( endSemantic );
}
else
{
myWindow->addZoom( beginTime, endTime, beginRow, endRow );
}
myWindow->setWindowBeginTime( beginTime, true );
myWindow->setWindowEndTime( endTime, true );

myWindow->setRedraw( true );
myWindow->setChanged( true );
}

zooming = false;
zoomXY = false;
}



void gTimeline::OnScrolledWindowUpdate( wxUpdateUIEvent& event )
{
if( this->IsShown() && 
( myWindow->getForceRedraw() || 
( wxparaverApp::mainWindow->getAutoRedraw() && myWindow->getRedraw() && !wxparaverApp::mainWindow->GetSomeWinIsRedraw() ) 
)
)
{
wxparaverApp::mainWindow->SetSomeWinIsRedraw( true );
myWindow->setForceRedraw( false );
myWindow->setRedraw( false );
splitChanged = false;
redraw();
drawZone->Refresh();
wxparaverApp::mainWindow->SetSomeWinIsRedraw( false );
}
else if( myWindow->getRedraw() && !wxparaverApp::mainWindow->getAutoRedraw() && !enabledAutoRedrawIcon )
{
enabledAutoRedrawIcon = true;

wxMemoryDC bufferDraw;
bufferDraw.SelectObject(wxNullBitmap);
bufferDraw.SelectObject( drawImage );

wxBitmap refreshImage( autoredraw_refresh_xpm );
bufferDraw.SetPen( wxPen( backgroundColour ) );
bufferDraw.SetBrush( wxBrush( backgroundColour ) );
bufferDraw.DrawRectangle( 0, 0,
refreshImage.GetWidth() + 2 * drawBorder, refreshImage.GetHeight() + 2 * drawBorder );
bufferDraw.DrawBitmap( refreshImage,
drawBorder,
drawBorder,
true );
drawZone->Refresh();
}

}


void gTimeline::OnPopUpCopy( wxCommandEvent& event )
{
gPasteWindowProperties::getInstance()->copy( this );
}


gTimeline *gTimeline::clone( Timeline *clonedWindow,
wxWindow *parent,
wxTreeItemId idRoot1,
wxTreeItemId idRoot2,
bool mustRedraw )
{
if ( clonedWindow == nullptr )
clonedWindow = myWindow->clone(); 

wxSize titleBarSize = GetSize() - GetClientSize();
if ( titleBarSize.GetHeight() == 0 )
titleBarSize = paraverMain::defaultTitleBarSize;
wxPoint position =  wxPoint( GetPosition().x + titleBarSize.GetHeight(),
GetPosition().y + titleBarSize.GetHeight() );
wxSize size = wxSize( clonedWindow->getWidth(), clonedWindow->getHeight() );

string composedName = myWindow->getName() + " @ " +
myWindow->getTrace()->getTraceNameNumbered();

gTimeline *clonedTimeline = new gTimeline( parent, wxID_ANY, wxString::FromUTF8( composedName.c_str() ), position );
clonedTimeline->SetMyWindow( clonedWindow );
clonedTimeline->SetClientSize( size );
clonedWindow->setPixelSize( myWindow->getPixelSize() );

LoadedWindows::getInstance()->add( clonedWindow );

wxChoicebook *choiceWindowBrowser = paraverMain::myParaverMain->choiceWindowBrowser;
PRV_INT16 currentTrace = paraverMain::myParaverMain->getTracePosition( clonedWindow->getTrace() );
wxTreeCtrl *allTracesPage = (wxTreeCtrl *) choiceWindowBrowser->GetPage( 0 ); 
wxTreeCtrl *currentPage = (wxTreeCtrl *) choiceWindowBrowser->GetPage( currentTrace + 1 ); 

TreeBrowserItemData *currentData =  new TreeBrowserItemData( wxString::FromUTF8( clonedWindow->getName().c_str() ), clonedTimeline );

int iconNumber = getIconNumber( clonedWindow );

wxTreeItemId currentWindowId1 = allTracesPage->AppendItem( idRoot1, wxString::FromUTF8( clonedWindow->getName().c_str() ), iconNumber, -1, currentData );
wxTreeItemId currentWindowId2 = currentPage->AppendItem( idRoot2, wxString::FromUTF8( clonedWindow->getName().c_str() ), iconNumber, -1, new TreeBrowserItemData( *currentData ) );

if ( clonedWindow->isDerivedWindow() )
{

clone( clonedWindow->getParent( 0 ), parent, currentWindowId1, currentWindowId2, mustRedraw );
clone( clonedWindow->getParent( 1 ), parent, currentWindowId1, currentWindowId2, mustRedraw );

allTracesPage->CollapseAllChildren( currentWindowId1 );
currentPage->CollapseAllChildren( currentWindowId2 );
}


clonedTimeline->SetBufferImage( bufferImage );
clonedTimeline->SetCommImage( commImage );
clonedTimeline->SetEventImage( eventImage );
clonedTimeline->SetDrawImage( drawImage );
clonedTimeline->SetDrawCaution( drawCaution );
clonedTimeline->SetDrawCautionNegatives( drawCautionNegatives );
clonedTimeline->SetObjectAxisPos( objectAxisPos );
clonedTimeline->SetTimeAxisPos( timeAxisPos );
clonedTimeline->SetObjectPosList( objectPosList );
clonedTimeline->SetForegroundColour( foregroundColour );
clonedTimeline->SetBackgroundColour( backgroundColour );
clonedTimeline->SetSemanticValuesToColor( semanticValuesToColor );
clonedTimeline->SetSemanticColorsToValue( semanticColorsToValue );
clonedTimeline->SetSemanticPixelsToValue( semanticPixelsToValue );

clonedTimeline->myWindow->setReady( myWindow->getReady() );
if( mustRedraw )
{
if( myWindow->getShowWindow() )
clonedTimeline->Show();
else
clonedTimeline->Show( false );
}
else
{
clonedWindow->setShowWindow( false );
clonedTimeline->Show( false );
clonedTimeline->myWindow->setReady( false );
}

return clonedTimeline;
}


void gTimeline::OnPopUpClone( wxCommandEvent& event )
{
clone( nullptr, parent, getAllTracesTree()->GetRootItem(), getSelectedTraceTree( myWindow->getTrace() )->GetRootItem());
}

void gTimeline::OnPopUpRename( wxCommandEvent& event )
{
( (paraverMain *) parent )->renameTreeItem( );
}



void gTimeline::OnPopUpFitTimeScale( wxCommandEvent& event )
{
myWindow->setWindowBeginTime( 0, true );
myWindow->setWindowEndTime( myWindow->getTrace()->getEndTime(), true );
myWindow->addZoom( 0, myWindow->getTrace()->getEndTime() );
myWindow->setRedraw( true );
myWindow->setChanged( true );
}


void gTimeline::OnPopUpFitSemanticScaleMin( wxCommandEvent& event )
{
myWindow->computeYScaleMin();
myWindow->setRedraw( true );
myWindow->setChanged( true );
}

void gTimeline::OnPopUpFitSemanticScaleMax( wxCommandEvent& event )
{
myWindow->computeYScaleMax();
myWindow->setRedraw( true );
myWindow->setChanged( true );
}

void gTimeline::OnPopUpFitSemanticScale( wxCommandEvent& event )
{
myWindow->computeYScale();
myWindow->setRedraw( true );
myWindow->setChanged( true );
}

void gTimeline::OnPopUpFitObjects( wxCommandEvent& event )
{
myWindow->addZoom( myWindow->getZoomFirstDimension().first, myWindow->getZoomFirstDimension().second,
TObjectOrder( 0 ), myWindow->getWindowLevelObjects() - 1 );
myWindow->setRedraw( true );
myWindow->setChanged( true );
}

void gTimeline::OnPopUpViewCommunicationLines( wxCommandEvent& event )
{
drawCommunicationLines( event.IsChecked() );
}

void gTimeline::OnPopUpViewEventFlags( wxCommandEvent& event )
{
drawEventFlags( event.IsChecked() );
}


void gTimeline::OnPopUpFunctionLineColor( wxCommandEvent& event )
{
drawFunctionLineColor();
}


void gTimeline::OnPopUpFusedLinesColor( wxCommandEvent& event )
{
drawFusedLinesColor();
}


void gTimeline::OnPopUpPunctualColor( wxCommandEvent& event )
{
myWindow->setPunctualColorMode();
myWindow->setRedraw( true );
}

void gTimeline::OnPopUpPunctualColorWindow( wxCommandEvent& event )
{
vector<Timeline *> compatWindows;
int selIndex = 0;

setEnableDestroyButton( false );

LoadedWindows::getInstance()->getDerivedCompatible( myWindow->getTrace(), compatWindows );
compatWindows.erase( std::find( compatWindows.begin(), compatWindows.end(), myWindow ) );

wxArrayString choices;
choices.Add( _( "None" ) );
int tmpIndex = 1;
for( vector<Timeline *>::iterator it = compatWindows.begin(); it != compatWindows.end(); ++it )
{
choices.Add( wxString::FromUTF8( (*it)->getName().c_str() ) );
if( (*it) == myWindow->getPunctualColorWindow() )
selIndex = tmpIndex;
++tmpIndex;
}

wxSingleChoiceDialog *dialog = new wxSingleChoiceDialog( this, _("Select window to use for coloring points:"), _("Punctual Window"), choices );
dialog->SetSelection( selIndex );
if ( dialog->ShowModal() == wxID_OK )
{
if( dialog->GetSelection() == 0 )
myWindow->setPunctualColorWindow( nullptr );
else
myWindow->setPunctualColorWindow( compatWindows[ dialog->GetSelection() - 1 ] );
myWindow->setRedraw( true );
}

delete dialog;

setEnableDestroyButton( true );
}

void gTimeline::OnPopUpCodeColor( wxCommandEvent& event )
{
myWindow->setCodeColorMode();
myWindow->setRedraw( true );
}


void gTimeline::OnPopUpPasteDefaultSpecial( wxCommandEvent& event )
{
gPasteWindowProperties* pasteActions = gPasteWindowProperties::getInstance();

pasteActions->paste( this, "Time" );
myWindow->addZoom( pasteActions->getBeginTime(), pasteActions->getEndTime() );
pasteActions->paste( this, "Size" );

if ( pasteActions->isAllowed( this, STR_OBJECTS ) )
pasteActions->paste( this, "Objects" );

myWindow->setRedraw( true );
myWindow->setChanged( true );
}


void gTimeline::OnPopUpPasteTime( wxCommandEvent& event )
{
gPasteWindowProperties* pasteActions = gPasteWindowProperties::getInstance();
pasteActions->paste( this, "Time" );
myWindow->addZoom( pasteActions->getBeginTime(), pasteActions->getEndTime() );
myWindow->setRedraw( true );
myWindow->setChanged( true );
}


void gTimeline::OnPopUpPasteObjects( wxCommandEvent& event )
{
gPasteWindowProperties::getInstance()->paste( this, "Objects" );
myWindow->setRedraw( true );
}


void gTimeline::OnPopUpPasteSize( wxCommandEvent& event )
{
gPasteWindowProperties::getInstance()->paste( this, "Size" );
myWindow->setRedraw( true );
}

void gTimeline::OnPopUpPasteDuration( wxCommandEvent& event )
{
gPasteWindowProperties* pasteActions = gPasteWindowProperties::getInstance();
pasteActions->paste( this, "Duration" );
myWindow->addZoom( pasteActions->getBeginTime(), pasteActions->getEndTime() );
myWindow->setRedraw( true );
}

void gTimeline::OnPopUpPasteSemanticScale( wxCommandEvent& event )
{
gPasteWindowProperties::getInstance()->paste( this, "Semantic Scale" );
myWindow->setRedraw( true );
myWindow->setChanged( true );
}

void gTimeline::OnPopUpPasteCustomPalette( wxCommandEvent& event )
{
gPasteWindowProperties::getInstance()->paste( this, STR_CUSTOM_PALETTE );
myWindow->setRedraw( true );
}

void gTimeline::OnPopUpPasteFilterAll( wxCommandEvent& event )
{
gPasteWindowProperties::getInstance()->paste( this, "Events" );
gPasteWindowProperties::getInstance()->paste( this, "Communications" );
myWindow->setRedraw( true );
myWindow->setChanged( true );
}


void gTimeline::OnPopUpPasteFilterCommunications( wxCommandEvent& event )
{
gPasteWindowProperties::getInstance()->paste( this, "Communications" );
myWindow->setRedraw( true );
myWindow->setChanged( true );
}


void gTimeline::OnPopUpPasteFilterEvents( wxCommandEvent& event )
{
gPasteWindowProperties::getInstance()->paste( this, "Events" );
myWindow->setRedraw( true );
myWindow->setChanged( true );
}


void gTimeline::OnPopUpPasteSpecial( wxCommandEvent& event )
{
wxArrayString choices;

wxMultiChoiceDialog *dialog = gPopUpMenu<gTimeline>::createPasteSpecialDialog( choices, this );

if ( dialog->ShowModal() == wxID_OK )
{
wxArrayInt selections = dialog->GetSelections();
if ( selections.GetCount() > 0 )
{
for ( size_t i = 0; i < selections.GetCount(); i++ )
{
gPasteWindowProperties* pasteActions = gPasteWindowProperties::getInstance();
if ( pasteActions->isAllowed( this, gPopUpMenu<gTimeline>::getOption( choices, selections[i] ) ) )
pasteActions->paste( this, gPopUpMenu<gTimeline>::getOption( choices, selections[i] ) );
}

myWindow->setRedraw( true );
myWindow->setChanged( true );
}
}

delete dialog;
}


void gTimeline::OnPopUpRowSelection( wxCommandEvent& event )
{
setEnableDestroyButton( false );

RowsSelectionDialog *dialog = gPopUpMenu<gTimeline>::createRowSelectionDialog( this );

if ( dialog->ShowModal() == wxID_OK )
{
if ( dialog->ShouldChangeTimelineZoom() )
{
myWindow->addZoom( dialog->GetNewBeginZoom(), dialog->GetNewEndZoom() );
}

myWindow->setRedraw( true );
myWindow->setChanged( true );
}

delete dialog;

setEnableDestroyButton( true );
}


void gTimeline::OnPopUpGradientColor( wxCommandEvent& event )
{
myWindow->setGradientColorMode();
myWindow->setRedraw( true );
}

void gTimeline::OnPopUpNotNullGradientColor( wxCommandEvent& event )
{
myWindow->setNotNullGradientColorMode();
myWindow->setRedraw( true );
}

void gTimeline::OnPopUpAlternativeGradientColor( wxCommandEvent& event )
{
myWindow->setAlternativeGradientColorMode();
myWindow->setRedraw( true );
}

void gTimeline::OnPopUpGradientFunction( wxCommandEvent& event )
{
TGradientFunction gradFunc;

switch( event.GetId() )
{
case ID_MENU_GRADIENT_FUNCTION_LINEAR: gradFunc = TGradientFunction::LINEAR; break;
case ID_MENU_GRADIENT_FUNCTION_STEPS: gradFunc = TGradientFunction::STEPS; break;
case ID_MENU_GRADIENT_FUNCTION_LOGARITHMIC: gradFunc = TGradientFunction::LOGARITHMIC; break;
case ID_MENU_GRADIENT_FUNCTION_EXPONENTIAL: gradFunc = TGradientFunction::EXPONENTIAL; break;
default: gradFunc = TGradientFunction::LINEAR;
}

OnMenuGradientFunction( gradFunc );
}

void gTimeline::OnPopUpSemanticScaleMinAtZero( wxCommandEvent& event )
{
myWindow->setSemanticScaleMinAtZero( event.IsChecked() );
myWindow->setRedraw( true );
}

void gTimeline::OnPopUpDrawModeTimeLast( wxCommandEvent& event )
{
myWindow->setDrawModeTime( DrawModeMethod::DRAW_LAST );
myWindow->setRedraw( true );
}

void gTimeline::OnPopUpDrawModeTimeRandom( wxCommandEvent& event )
{
myWindow->setDrawModeTime( DrawModeMethod::DRAW_RANDOM );
myWindow->setRedraw( true );
}

void gTimeline::OnPopUpDrawModeTimeRandomNotZero( wxCommandEvent& event )
{
myWindow->setDrawModeTime( DrawModeMethod::DRAW_RANDNOTZERO );
myWindow->setRedraw( true );
}

void gTimeline::OnPopUpDrawModeTimeMaximum( wxCommandEvent& event )
{
myWindow->setDrawModeTime( DrawModeMethod::DRAW_MAXIMUM );
myWindow->setRedraw( true );
}

void gTimeline::OnPopUpDrawModeTimeMinimumNotZero( wxCommandEvent& event )
{
myWindow->setDrawModeTime( DrawModeMethod::DRAW_MINNOTZERO );
myWindow->setRedraw( true );
}

void gTimeline::OnPopUpDrawModeTimeAbsoluteMaximum( wxCommandEvent& event )
{
myWindow->setDrawModeTime( DrawModeMethod::DRAW_ABSOLUTE_MAXIMUM );
myWindow->setRedraw( true );
}

void gTimeline::OnPopUpDrawModeTimeAbsoluteMinimumNotZero( wxCommandEvent& event )
{
myWindow->setDrawModeTime( DrawModeMethod::DRAW_ABSOLUTE_MINNOTZERO );
myWindow->setRedraw( true );
}

void gTimeline::OnPopUpDrawModeTimeAverage( wxCommandEvent& event )
{
myWindow->setDrawModeTime( DrawModeMethod::DRAW_AVERAGE );
myWindow->setRedraw( true );
}

void gTimeline::OnPopUpDrawModeTimeAverageNotZero( wxCommandEvent& event )
{
myWindow->setDrawModeTime( DrawModeMethod::DRAW_AVERAGENOTZERO );
myWindow->setRedraw( true );
}

void gTimeline::OnPopUpDrawModeTimeMode( wxCommandEvent& event )
{
myWindow->setDrawModeTime( DrawModeMethod::DRAW_MODE );
myWindow->setRedraw( true );
}

void gTimeline::OnPopUpDrawModeObjectsLast( wxCommandEvent& event )
{
myWindow->setDrawModeObject( DrawModeMethod::DRAW_LAST );
myWindow->setRedraw( true );
}

void gTimeline::OnPopUpDrawModeObjectsRandom( wxCommandEvent& event )
{
myWindow->setDrawModeObject( DrawModeMethod::DRAW_RANDOM );
myWindow->setRedraw( true );
}

void gTimeline::OnPopUpDrawModeObjectsRandomNotZero( wxCommandEvent& event )
{
myWindow->setDrawModeObject( DrawModeMethod::DRAW_RANDNOTZERO );
myWindow->setRedraw( true );
}

void gTimeline::OnPopUpDrawModeObjectsMaximum( wxCommandEvent& event )
{
myWindow->setDrawModeObject( DrawModeMethod::DRAW_MAXIMUM );
myWindow->setRedraw( true );
}

void gTimeline::OnPopUpDrawModeObjectsMinimumNotZero( wxCommandEvent& event )
{
myWindow->setDrawModeObject( DrawModeMethod::DRAW_MINNOTZERO );
myWindow->setRedraw( true );
}

void gTimeline::OnPopUpDrawModeObjectsAbsoluteMaximum( wxCommandEvent& event )
{
myWindow->setDrawModeObject( DrawModeMethod::DRAW_ABSOLUTE_MAXIMUM );
myWindow->setRedraw( true );
}

void gTimeline::OnPopUpDrawModeObjectsAbsoluteMinimumNotZero( wxCommandEvent& event )
{
myWindow->setDrawModeObject( DrawModeMethod::DRAW_ABSOLUTE_MINNOTZERO );
myWindow->setRedraw( true );
}

void gTimeline::OnPopUpDrawModeObjectsAverage( wxCommandEvent& event )
{
myWindow->setDrawModeObject( DrawModeMethod::DRAW_AVERAGE );
myWindow->setRedraw( true );
}

void gTimeline::OnPopUpDrawModeObjectsAverageNotZero( wxCommandEvent& event )
{
myWindow->setDrawModeObject( DrawModeMethod::DRAW_AVERAGENOTZERO );
myWindow->setRedraw( true );
}

void gTimeline::OnPopUpDrawModeObjectsMode( wxCommandEvent& event )
{
myWindow->setDrawModeObject( DrawModeMethod::DRAW_MODE );
myWindow->setRedraw( true );
}

void gTimeline::OnPopUpDrawModeBothLast( wxCommandEvent& event )
{
myWindow->setDrawModeObject( DrawModeMethod::DRAW_LAST );
myWindow->setDrawModeTime( DrawModeMethod::DRAW_LAST );
myWindow->setRedraw( true );
}

void gTimeline::OnPopUpDrawModeBothRandom( wxCommandEvent& event )
{
myWindow->setDrawModeObject( DrawModeMethod::DRAW_RANDOM );
myWindow->setDrawModeTime( DrawModeMethod::DRAW_RANDOM );
myWindow->setRedraw( true );
}

void gTimeline::OnPopUpDrawModeBothRandomNotZero( wxCommandEvent& event )
{
myWindow->setDrawModeObject( DrawModeMethod::DRAW_RANDNOTZERO );
myWindow->setDrawModeTime( DrawModeMethod::DRAW_RANDNOTZERO );
myWindow->setRedraw( true );
}

void gTimeline::OnPopUpDrawModeBothMaximum( wxCommandEvent& event )
{
myWindow->setDrawModeObject( DrawModeMethod::DRAW_MAXIMUM );
myWindow->setDrawModeTime( DrawModeMethod::DRAW_MAXIMUM );
myWindow->setRedraw( true );
}

void gTimeline::OnPopUpDrawModeBothMinimumNotZero( wxCommandEvent& event )
{
myWindow->setDrawModeObject( DrawModeMethod::DRAW_MINNOTZERO );
myWindow->setDrawModeTime( DrawModeMethod::DRAW_MINNOTZERO );
myWindow->setRedraw( true );
}

void gTimeline::OnPopUpDrawModeBothAbsoluteMaximum( wxCommandEvent& event )
{
myWindow->setDrawModeObject( DrawModeMethod::DRAW_ABSOLUTE_MAXIMUM );
myWindow->setDrawModeTime( DrawModeMethod::DRAW_ABSOLUTE_MAXIMUM );
myWindow->setRedraw( true );
}

void gTimeline::OnPopUpDrawModeBothAbsoluteMinimumNotZero( wxCommandEvent& event )
{
myWindow->setDrawModeObject( DrawModeMethod::DRAW_ABSOLUTE_MINNOTZERO );
myWindow->setDrawModeTime( DrawModeMethod::DRAW_ABSOLUTE_MINNOTZERO );
myWindow->setRedraw( true );
}

void gTimeline::OnPopUpDrawModeBothAverage( wxCommandEvent& event )
{
myWindow->setDrawModeObject( DrawModeMethod::DRAW_AVERAGE );
myWindow->setDrawModeTime( DrawModeMethod::DRAW_AVERAGE );
myWindow->setRedraw( true );
}

void gTimeline::OnPopUpDrawModeBothAverageNotZero( wxCommandEvent& event )
{
myWindow->setDrawModeObject( DrawModeMethod::DRAW_AVERAGENOTZERO );
myWindow->setDrawModeTime( DrawModeMethod::DRAW_AVERAGENOTZERO );
myWindow->setRedraw( true );
}

void gTimeline::OnPopUpDrawModeBothMode( wxCommandEvent& event )
{
myWindow->setDrawModeObject( DrawModeMethod::DRAW_MODE );
myWindow->setDrawModeTime( DrawModeMethod::DRAW_MODE );
myWindow->setRedraw( true );
}


void gTimeline::OnPopUpPixelSize( wxCommandEvent& event )
{
PRV_UINT16 pixelSize; 

switch( event.GetId() )
{
case ID_MENU_PIXEL_SIZE_x1: pixelSize = 1; break;
case ID_MENU_PIXEL_SIZE_x2: pixelSize = 2; break;
case ID_MENU_PIXEL_SIZE_x4: pixelSize = 4; break;
case ID_MENU_PIXEL_SIZE_x8: pixelSize = 8; break;
default:                    pixelSize = 1; break;
}

myWindow->setPixelSize( pixelSize );
myWindow->setRedraw( true );
}


void gTimeline::OnPopUpLabels( wxCommandEvent& event )
{
switch( event.GetId() )
{
case ID_MENU_LABELS_SPACED:
myWindow->setObjectLabels( TObjectLabels::SPACED_LABELS );
break;
case ID_MENU_LABELS_POWER2:
myWindow->setObjectLabels( TObjectLabels::POWER2_LABELS );
break;
case ID_MENU_LABELS_ALL:
default:
myWindow->setObjectLabels( TObjectLabels::ALL_LABELS );
break;
}

myWindow->setRedraw( true );
}


void gTimeline::OnPopUpObjectAxis( wxCommandEvent& event )
{
switch( event.GetId() )
{
case ID_MENU_OBJECT_AXIS_ALL:
myWindow->setObjectAxisSize( TObjectAxisSize::ALL_LEVELS );
break;
case ID_MENU_OBJECT_AXIS_ZERO:
myWindow->setObjectAxisSize( TObjectAxisSize::ZERO_PERC );
break;
case ID_MENU_OBJECT_AXIS_FIVE:
myWindow->setObjectAxisSize( TObjectAxisSize::FIVE_PERC );
break;
case ID_MENU_OBJECT_AXIS_TEN:
myWindow->setObjectAxisSize( TObjectAxisSize::TEN_PERC );
break;
case ID_MENU_OBJECT_AXIS_TWENTYFIVE:
myWindow->setObjectAxisSize( TObjectAxisSize::TWENTYFIVE_PERC );
break;
case ID_MENU_OBJECT_AXIS_CURRENT:
default:
myWindow->setObjectAxisSize( TObjectAxisSize::CURRENT_LEVEL );
break;
}

myWindow->setRedraw( true );
}


void gTimeline::OnPopUpRunApp( wxCommandEvent& event )
{
switch( event.GetId() )
{
case ID_MENU_CUTTER:
SequenceDriver::sequenceCutter( this );
break;
case ID_MENU_DIMEMAS:
SequenceDriver::sequenceDimemas( this );
break;
case ID_MENU_CLUSTERING:
paraverMain::myParaverMain->SetClusteringWindow( myWindow );
SequenceDriver::sequenceClustering( this );
break;
case ID_MENU_FOLDING:
SequenceDriver::sequenceFolding( this );
break;
case ID_MENU_SPECTRAL:
SequenceDriver::sequenceSpectral( this );
break;
case ID_MENU_PROFET:
SequenceDriver::sequenceProfet( this );
break;
case ID_MENU_USER_COMMAND:
SequenceDriver::sequenceUserCommand( this );
break;
default:
break;
}
}


void gTimeline::OnPopUpSynchronize( wxCommandEvent& event )
{
if( event.GetId() == ID_MENU_NEWGROUP )
{
myWindow->addToSyncGroup( SyncWindows::getInstance()->newGroup() );
}
else
{
vector<TGroupId> tmpGroups;
SyncWindows::getInstance()->getGroups( tmpGroups );

TGroupId group = tmpGroups[ event.GetId() - ID_MENU_SYNC_GROUP_BASE ];
if( myWindow->isSync() && group == myWindow->getSyncGroup() )
myWindow->removeFromSync();
else
myWindow->addToSyncGroup( group );
}
}


void gTimeline::OnPopUpRemoveGroup( wxCommandEvent& event )
{
vector< TGroupId > tmpGroups;
SyncWindows::getInstance()->getGroups( tmpGroups );
SyncWindows::getInstance()->removeAllWindows( tmpGroups[ event.GetId() - ID_MENU_SYNC_REMOVE_GROUP_BASE ] );
}


void gTimeline::OnPopUpRemoveAllGroups( wxCommandEvent& event )
{
SyncWindows::getInstance()->removeAllGroups();
}



void gTimeline::OnPopUpUndoZoom( wxCommandEvent& event )
{
if ( !myWindow->emptyPrevZoom() )
{
TObjectOrder prevZoomBegin = myWindow->getPrevZoomSecondDimension().first;
TObjectOrder prevZoomEnd = myWindow->getPrevZoomSecondDimension().second;
vector<TObjectOrder> selectedSet;
myWindow->getSelectedRows( myWindow->getLevel(), selectedSet, prevZoomBegin, prevZoomEnd, true );

if ( selectedSet.size() > 0 )
{
myWindow->prevZoom();
myWindow->setWindowBeginTime( myWindow->getZoomFirstDimension().first, true );
myWindow->setWindowEndTime( myWindow->getZoomFirstDimension().second, true );
myWindow->setRedraw( true );
myWindow->setChanged( true );
}
else
{
wxString tmpMsg( wxT( "Unable to browse current objects in previous zoom!\n\nPlease select objects." ) );
wxMessageDialog tmpDialog( nullptr, tmpMsg, _( "Warning" ), wxOK | wxICON_EXCLAMATION );
if ( tmpDialog.ShowModal() == wxID_OK )
{
wxCommandEvent dummyEvent;
OnPopUpRowSelection( dummyEvent );
}
}
}
}


void gTimeline::OnPopUpRedoZoom( wxCommandEvent& event )
{
if ( !myWindow->emptyNextZoom() )
{
TObjectOrder nextZoomBegin = myWindow->getNextZoomSecondDimension().first;
TObjectOrder nextZoomEnd = myWindow->getNextZoomSecondDimension().second;
vector<TObjectOrder> selectedSet;
myWindow->getSelectedRows( myWindow->getLevel(), selectedSet, nextZoomBegin, nextZoomEnd, true );

if ( selectedSet.size() > 0 )
{
myWindow->nextZoom();
myWindow->setWindowBeginTime( myWindow->getZoomFirstDimension().first, true );
myWindow->setWindowEndTime( myWindow->getZoomFirstDimension().second, true );
myWindow->setRedraw( true );
myWindow->setChanged( true );
}
else
{
wxString tmpMsg( wxT( "Unable to browse current objects in next zoom!\n\nPlease select objects." ) );
wxMessageDialog tmpDialog( nullptr, tmpMsg, _( "Warning" ), wxOK | wxICON_EXCLAMATION );
if ( tmpDialog.ShowModal() == wxID_OK )
{
wxCommandEvent dummyEvent;
OnPopUpRowSelection( dummyEvent );
}
}
}
}


void gTimeline::rightDownManager()
{
gPopUpMenu<gTimeline> popUpMenu( this );

popUpMenu.enable( "Undo Zoom", !myWindow->emptyPrevZoom() );
popUpMenu.enable( "Redo Zoom", !myWindow->emptyNextZoom() );

popUpMenu.enableMenu( this );
PopupMenu( &popUpMenu );
}



void gTimeline::OnScrolledWindowRightDown( wxMouseEvent& event )
{
rightDownManager();
}



void gTimeline::OnScrolledWindowMotion( wxMouseEvent& event )
{
if( gTimeline::dialogProgress != nullptr )
return;

if( wheelZoomFactor != 1.0 )
return;

motionEvent = event;
if( !event.ShiftDown() )
timerMotion->Start( 20, true );

wxMemoryDC dc( bufferImage );
PRV_UINT32 precision = 0;

rgb rgbForegroundColour = ((paraverMain *)parent)->GetParaverConfig()->getColorsTimelineAxis();
wxColour foregroundColour = wxColour( rgbForegroundColour.red,
rgbForegroundColour.green,
rgbForegroundColour.blue );

rgb rgbBackgroundColour = ((paraverMain *)parent)->GetParaverConfig()->getColorsTimelineBackground();
wxColour backgroundColour = wxColour( rgbBackgroundColour.red,
rgbBackgroundColour.green,
rgbBackgroundColour.blue );

TTime timeStep = ( myWindow->getWindowEndTime() - myWindow->getWindowBeginTime() ) /
( dc.GetSize().GetWidth() - objectAxisPos - drawBorder );

if( zooming )
{
if( event.ShiftDown() )
{
MousePanMotion();
return;
}

if( !timing )
zoomXY = event.ControlDown();
else
zoomXY = false;

long beginX = zoomBeginX > event.GetX() ? event.GetX() : zoomBeginX;
long beginY = drawBorder;
long endX = zoomBeginX < event.GetX() ? event.GetX() : zoomBeginX;
long endY = timeAxisPos;
if( beginX < objectAxisPos )
beginX = objectAxisPos;
if( endX > drawImage.GetWidth() - drawBorder )
endX = drawImage.GetWidth() - drawBorder;
wxCoord width = endX - beginX;
wxCoord height = timeAxisPos - drawBorder + 1;

if ( zoomXY )
{
beginY = zoomBeginY > event.GetY() ? event.GetY() : zoomBeginY;
endY = zoomBeginY < event.GetY() ? event.GetY() : zoomBeginY;
if( beginY > timeAxisPos )
beginY = timeAxisPos;
if( endY > timeAxisPos )
endY = timeAxisPos;
if( beginY < drawBorder )
beginY = drawBorder;
if( endY < drawBorder )
endY = drawBorder;
height = endY - beginY;
}

if( endX - beginX > 3.0 )
{
wxMemoryDC memdc( drawImage );
memdc.SetBackgroundMode( wxTRANSPARENT );
memdc.SetBackground( *wxTRANSPARENT_BRUSH );
memdc.Clear();
#if wxTEST_GRAPHICS == 1
wxGCDC dc( memdc );
dc.SetBrush( wxBrush( wxColour( 255, 255, 255, 80 ) ) );
#else
wxDC& dc = memdc;
dc.SetBrush( *wxTRANSPARENT_BRUSH );
#endif
dc.SetPen( wxPen( foregroundColour ) );

dc.DrawBitmap( bufferImage, 0, 0, false );
if( myWindow->getDrawFlags() )
dc.DrawBitmap( eventImage, 0, 0, true );
if( myWindow->getDrawCommLines() )
dc.DrawBitmap( commImage, 0, 0, true );
dc.DrawRectangle( beginX, beginY, width, height );

drawZone->Refresh();
}

if( beginX < objectAxisPos )
beginX = 0;
else
beginX -= objectAxisPos;

if( endX > dc.GetSize().GetWidth() - drawBorder )
endX = dc.GetSize().GetWidth() - drawBorder - objectAxisPos;
else
endX -= objectAxisPos;

TTime endTime = ( timeStep * endX ) + myWindow->getWindowBeginTime();
TTime beginTime = ( timeStep * beginX ) + myWindow->getWindowBeginTime();

initialTimeText->SetValue( wxString::FromUTF8( LabelConstructor::timeLabel( myWindow->traceUnitsToWindowUnits( beginTime ),
myWindow->getTimeUnit(), precision ).c_str() ) );
finalTimeText->SetValue( wxString::FromUTF8( LabelConstructor::timeLabel( myWindow->traceUnitsToWindowUnits( endTime ),
myWindow->getTimeUnit(), precision ).c_str() ) );
durationText->SetValue( wxString::FromUTF8( LabelConstructor::timeLabel( myWindow->traceUnitsToWindowUnits( endTime - beginTime ),
myWindow->getTimeUnit(), precision ).c_str() ) );
vector<TObjectOrder> selectedSet;
TObjectOrder beginRow = myWindow->getZoomSecondDimension().first;
TObjectOrder endRow =  myWindow->getZoomSecondDimension().second;
myWindow->getSelectedRows( myWindow->getLevel(), selectedSet, beginRow, endRow, true );
if( zoomXY &&
( selectedSet.size() == 1 && ( myWindow->isFunctionLineColorSet() || myWindow->isPunctualColorSet() )
|| myWindow->isFusedLinesColorSet() ) )
{
if( beginY < drawBorder )
beginY = timeAxisPos - drawBorder;
else
beginY = timeAxisPos - beginY;

if( endY < drawBorder )
endY = timeAxisPos - drawBorder;
else
endY = timeAxisPos - endY;

TSemanticValue semanticStep = ( myWindow->getMaximumY() - myWindow->getMinimumY() ) /
( timeAxisPos - drawBorder );
TSemanticValue beginSemantic = ( semanticStep * endY ) + myWindow->getMinimumY();
TSemanticValue endSemantic = ( semanticStep * beginY ) + myWindow->getMinimumY();

if( myWindow->getMaximumY() - myWindow->getMinimumY() < 1 ) precision = 6;
initialSemanticText->SetValue( wxString::FromUTF8( LabelConstructor::semanticLabel( myWindow, beginSemantic, false, precision, false ).c_str() ) );
finalSemanticText->SetValue( wxString::FromUTF8( LabelConstructor::semanticLabel( myWindow, endSemantic, false, precision, false ).c_str() ) );
slopeText->SetValue( wxString::FromUTF8( LabelConstructor::semanticLabel( myWindow, 
( endSemantic - beginSemantic ) / 
myWindow->traceUnitsToWindowUnits( endTime - beginTime ),
false,
6,
false ).c_str() ) );
}
}
else if( timing || wxGetApp().GetGlobalTiming() )
{
long beginX = event.GetX();
if( beginX < objectAxisPos )
beginX = 0;
else if( beginX > dc.GetSize().GetWidth() - drawBorder )
beginX = dc.GetSize().GetWidth() - drawBorder - objectAxisPos;
else
beginX -= objectAxisPos;
TTime time = ( timeStep * beginX ) + myWindow->getWindowBeginTime();

initialTimeText->SetValue( wxString::FromUTF8( LabelConstructor::timeLabel( myWindow->traceUnitsToWindowUnits( time ), myWindow->getTimeUnit(), precision ).c_str() ) );
finalTimeText->SetValue( _( "" ) );
durationText->SetValue( _( "" ) );
}
}




void gTimeline::OnSplitterTimelineSashUnsplit( wxSplitterEvent& event )
{
Unsplit();
}




void gTimeline::OnSplitterTimelineSashDClick( wxSplitterEvent& event )
{
infoZoneLastSize = infoZone->GetSize().GetHeight();
canRedraw = false;
event.Skip();
}




void gTimeline::OnRightDown( wxMouseEvent& event )
{
}




void gTimeline::OnNotebookInfoPageChanging( wxNotebookEvent& event )
{
if( myWindow == nullptr )
return;
canRedraw = false;
infoZone->ChangeSelection( event.GetSelection() );
if( !splitter->IsSplit() )
{
Split();
}
canRedraw = true;
}


wxString gTimeline::formatTime( TRecordTime whichTime, bool showDate )
{
wxString formattedTime;

if ( !showDate )
{
formattedTime = wxString::FromUTF8( LabelConstructor::timeLabel( myWindow->traceUnitsToWindowUnits( whichTime ),
myWindow->getTimeUnit(), 0 ).c_str() );
}
else
{
ptime headerTime = myWindow->getTrace()->getTraceTime();
ptime clickTime = headerTime + boost::posix_time::nanoseconds( (long)myWindow->traceUnitsToCustomUnits( whichTime, NS ) );
formattedTime = wxString::FromUTF8( LabelConstructor::timeLabel( clickTime, 0 ).c_str() );
}

return formattedTime;
}


void gTimeline::computeWhatWhere( TRecordTime whichTime,
TObjectOrder whichRow,
TSemanticValue whichSemantic,
bool textMode,
bool showDate,
bool hexMode )
{
whatWhereTime = whichTime;
whatWhereRow = whichRow;
whatWhereSemantic = whichSemantic;

whatWhereLines.clear();
whatWhereSelectedTimeEventLines = 0;
whatWhereSelectedTimeCommunicationLines = 0;

whatWhereLines.push_back( make_pair( TWhatWhereLine::BEGIN_OBJECT_SECTION, _( "" )));

wxString txt;

if( myWindow->isFusedLinesColorSet() )
{
txt << _( "Semantic: " ) << wxString::FromUTF8( LabelConstructor::semanticLabel( myWindow,
whichSemantic,
false,
ParaverConfig::getInstance()->getTimelinePrecision(),
false ).c_str() );
}
else
{
if( myWindow->getLevel() == TTraceLevel::CPU )
txt << _( "Object: " ) << wxString::FromUTF8( LabelConstructor::objectLabel( whichRow + 1, myWindow->getLevel(), myWindow->getTrace() ).c_str() );
else
txt << _( "Object: " ) << wxString::FromUTF8( LabelConstructor::objectLabel( whichRow, myWindow->getLevel(), myWindow->getTrace() ).c_str() );
}
txt << _( "\t  Click time: " ) << formatTime( whichTime, showDate );
txt << _( "\n" );
whatWhereLines.push_back( make_pair( TWhatWhereLine::RAW_LINE, txt ) );

whatWhereLines.push_back( make_pair( TWhatWhereLine::END_OBJECT_SECTION, _( "" )));

if( myWindow->isFusedLinesColorSet() )
return;

myWindow->init( whichTime, CREATEEVENTS + CREATECOMMS, false );
myWindow->initRow( whichRow, whichTime, CREATEEVENTS + CREATECOMMS, false );

TRecordTime tmpBeginTime = myWindow->getBeginTime( whichRow );

if ( tmpBeginTime > 0.0 )
{
myWindow->getRecordList( whichRow )->erase( myWindow->getRecordList( whichRow )->begin(),
myWindow->getRecordList( whichRow )->end() );

--tmpBeginTime;
myWindow->init( tmpBeginTime, CREATEEVENTS + CREATECOMMS, false );
myWindow->initRow( whichRow, tmpBeginTime, CREATEEVENTS + CREATECOMMS, false );

if( myWindow->getEndTime( whichRow ) < myWindow->getTrace()->getEndTime() )
{
printWWSemantic( whichRow, false, textMode, hexMode );
printWWRecords( whichRow, false, textMode, showDate );
myWindow->calcNext( whichRow, false );
while( myWindow->getEndTime( whichRow ) < myWindow->getTrace()->getEndTime() &&
myWindow->getBeginTime( whichRow ) == myWindow->getEndTime( whichRow ) )
{
printWWSemantic( whichRow, false, textMode, hexMode );
printWWRecords( whichRow, false, textMode, showDate );
myWindow->calcNext( whichRow, false );
}
}
}

printWWSemantic( whichRow, true, textMode, hexMode );
printWWRecords( whichRow, true, textMode, showDate );

if( myWindow->getEndTime( whichRow ) < myWindow->getTrace()->getEndTime() )
{
myWindow->calcNext( whichRow, false );
while( myWindow->getEndTime( whichRow ) < myWindow->getTrace()->getEndTime() &&
myWindow->getBeginTime( whichRow ) == myWindow->getEndTime( whichRow ) )
{
printWWSemantic( whichRow, false, textMode, hexMode );
printWWRecords( whichRow, false, textMode, showDate );
myWindow->calcNext( whichRow, false );
}
printWWSemantic( whichRow, false, textMode, hexMode );
printWWRecords( whichRow, false, textMode, showDate );
}
}


void gTimeline::printWhatWhere( )
{
int fontSize = 10;
bool allowedLine, allowedSection = false, tooMuchMessage = true;
int recordsCount = 0;

bool boldOpen = false;
bool italicOpen = false;
bool fontSizeOpen = false;
bool textColourOpen = false;  

whatWhereText->Clear();

whatWhereText->BeginFontSize( fontSize );
fontSizeOpen = true;

for ( vector< pair< TWhatWhereLine, wxString > >::iterator it = whatWhereLines.begin(); 
it != whatWhereLines.end(); ++it )
{
switch( it->first )
{
case TWhatWhereLine::BEGIN_OBJECT_SECTION:
allowedSection = true;
break;

case TWhatWhereLine::BEGIN_PREVNEXT_SECTION:
allowedSection = checkWWPreviousNext->IsChecked();
whatWhereText->BeginTextColour( wxColour( 0xb0b0b0 ) ); 
textColourOpen = true;
break;

case TWhatWhereLine::END_PREVNEXT_SECTION:
whatWhereText->EndTextColour();
textColourOpen = false;
break;

case TWhatWhereLine::BEGIN_CURRENT_SECTION:
allowedSection = true;
whatWhereText->BeginTextColour( wxSystemSettings::GetColour( wxSYS_COLOUR_WINDOWTEXT ) );
textColourOpen = true;
break;

case TWhatWhereLine::END_CURRENT_SECTION:
whatWhereText->EndTextColour();
textColourOpen = false;

break;
case TWhatWhereLine::BEGIN_RECORDS_SECTION:
allowedSection = checkWWEvents->IsChecked() || checkWWCommunications->IsChecked();
whatWhereText->BeginFontSize( fontSize - 1 );
fontSizeOpen = true;
whatWhereText->BeginItalic();
italicOpen = true;
break;

case TWhatWhereLine::END_RECORDS_SECTION:
whatWhereText->EndItalic();
whatWhereText->EndFontSize();
fontSizeOpen = false;
italicOpen = false;
break;

case TWhatWhereLine::BEGIN_SEMANTIC_SECTION:
allowedSection = checkWWSemantic->IsChecked();
whatWhereText->BeginBold();
boldOpen = true;
break;

case TWhatWhereLine::END_SEMANTIC_SECTION:
whatWhereText->EndBold();
boldOpen = false;
break;

default:
if ( allowedSection )
{
switch( it->first )
{
case TWhatWhereLine::EVENT_LINE:
if( recordsCount >= 100 )
allowedLine = false;
else
{
allowedLine = checkWWEvents->IsChecked();
if( allowedLine ) ++recordsCount;
}
break;
case TWhatWhereLine::COMMUNICATION_LINE:
if( recordsCount >= 100 )
allowedLine = false;
else
{
allowedLine = checkWWCommunications->IsChecked();
if( allowedLine ) ++recordsCount;
}
break;

case TWhatWhereLine::SEMANTIC_LINE:
recordsCount = 0;
allowedLine = checkWWSemantic->IsChecked();
break;

case TWhatWhereLine::MARK_LINE:
allowedLine = (( checkWWEvents->IsChecked() && whatWhereSelectedTimeEventLines > 0 ) ||
( checkWWCommunications->IsChecked() && whatWhereSelectedTimeCommunicationLines > 0 ));
break;

default:
allowedLine = true;
break;
}

if ( allowedLine )
whatWhereText->WriteText( it->second );
else if( tooMuchMessage && recordsCount >= 100 )
{
whatWhereText->BeginBold();
whatWhereText->WriteText( _( "Too much records. Reduce time scale.\n" ) );
whatWhereText->EndBold();
tooMuchMessage = false;
}

break;
}
}
}

if ( fontSizeOpen )
whatWhereText->EndFontSize();

if ( boldOpen )
whatWhereText->EndBold();

if ( italicOpen )
whatWhereText->EndItalic();

if ( textColourOpen )
whatWhereText->EndTextColour();

infoZone->ChangeSelection( 0 );
}


void gTimeline::printWWSemantic( TObjectOrder whichRow, bool clickedValue, bool textMode, bool hexMode )
{
wxString onString;

whatWhereLines.push_back( make_pair( TWhatWhereLine::BEGIN_SEMANTIC_SECTION, _( "" )));

if( clickedValue )
whatWhereLines.push_back( make_pair( TWhatWhereLine::BEGIN_CURRENT_SECTION, _( "" )));
else
whatWhereLines.push_back( make_pair( TWhatWhereLine::BEGIN_PREVNEXT_SECTION, _( "" )));

if ( !textMode )
onString << _("Semantic value: ");
onString << wxString::FromUTF8( LabelConstructor::semanticLabel( myWindow, myWindow->getValue( whichRow ), textMode, 
ParaverConfig::getInstance()->getTimelinePrecision(), hexMode ).c_str() );
onString << wxT( "\t  Duration: " ) << wxString::FromUTF8( LabelConstructor::timeLabel(
myWindow->traceUnitsToWindowUnits( myWindow->getEndTime( whichRow )
- myWindow->getBeginTime( whichRow ) ),
myWindow->getTimeUnit(), 
ParaverConfig::getInstance()->getTimelinePrecision() ).c_str() );
onString << _( "\n" );
whatWhereLines.push_back( make_pair( TWhatWhereLine::SEMANTIC_LINE, onString ));

if( clickedValue )
whatWhereLines.push_back( make_pair( TWhatWhereLine::END_CURRENT_SECTION, _( "" )));
else
whatWhereLines.push_back( make_pair( TWhatWhereLine::END_PREVNEXT_SECTION, _( "" )));

whatWhereLines.push_back( make_pair( TWhatWhereLine::END_SEMANTIC_SECTION, _( "" )));
}


void gTimeline::printWWRecords( TObjectOrder whichRow, bool clickedValue, bool textMode, bool showDate )
{
wxString onString;

whatWhereLines.push_back( make_pair( TWhatWhereLine::BEGIN_RECORDS_SECTION, _( "" )));

RecordList *rl = myWindow->getRecordList( whichRow );
RecordList::iterator it = rl->begin();

while( it != rl->end() &&
( (*it).getTime() < myWindow->getWindowBeginTime() ||
(*it).getTime() < myWindow->getBeginTime( whichRow ) ) )
{
++it;
}

if( clickedValue )
whatWhereLines.push_back( make_pair( TWhatWhereLine::BEGIN_CURRENT_SECTION, _( "" )));
else
whatWhereLines.push_back( make_pair( TWhatWhereLine::BEGIN_PREVNEXT_SECTION, _( "" )));

TRecordTime timePerPixel = ( myWindow->getWindowEndTime() - myWindow->getWindowBeginTime() ) /
( bufferImage.GetWidth() - objectAxisPos - drawBorder );
TRecordTime stepTime = timePerPixel * ParaverConfig::getInstance()->getTimelineWhatWhereEventPixels();
TRecordTime fromTime = whatWhereTime - stepTime;
if( fromTime < 0.0 ) fromTime = 0.0;
TRecordTime toTime = whatWhereTime + stepTime;

while( it != rl->end() &&
(*it).getTime() < fromTime )
++it;

while( it != rl->end() && 
(*it).getTime() >= fromTime &&
(*it).getTime() <= toTime )
{
if( (*it).getRecordType() & EVENT )
{
onString << wxT( "User Event at " ) << formatTime( (*it).getTime(), showDate );
onString << wxT( "    " );
onString << wxString::FromUTF8( LabelConstructor::eventLabel( myWindow,
(*it).getEventType(),
(*it).getEventValue(),
textMode ).c_str() );
onString << wxT( "\n" );

whatWhereLines.push_back( make_pair( TWhatWhereLine::EVENT_LINE, onString ));
onString.clear();

if ( clickedValue )
whatWhereSelectedTimeEventLines++;
}
else if( (*it).getRecordType() & COMM )
{
if( (*it).getRecordType() & LOG )
onString << wxT( "Logical " );
else if( (*it).getRecordType() & PHY )
onString << wxT( "Physical " );

if( (*it).getRecordType() & SEND )
onString << wxT( "SEND " );
else if( (*it).getRecordType() & RECV )
onString << wxT( "RECEIVE " );

onString << wxT( "at " ) << formatTime( (*it).getTime(), showDate );

if( (*it).getRecordType() & SEND )
onString << wxT( " to " );
else if( (*it).getRecordType() & RECV )
onString << wxT( " from " );
if ( myWindow->getLevel() >= TTraceLevel::WORKLOAD && myWindow->getLevel() <= TTraceLevel::THREAD )
onString << wxString::FromUTF8( LabelConstructor::objectLabel( myWindow->threadObjectToWindowObject( (*it).getCommPartnerObject() ),
myWindow->getLevel(),
myWindow->getTrace() ).c_str() );
else
onString << wxString::FromUTF8( LabelConstructor::objectLabel( myWindow->cpuObjectToWindowObject( (*it).getCommPartnerObject() ),
myWindow->getLevel(),
myWindow->getTrace() ).c_str() );

onString << wxT( " at " ) << formatTime( (*it).getCommPartnerTime(), showDate );

if( (*it).getRecordType() & SEND )
onString << wxT( ", Duration: " ) << wxString::FromUTF8( LabelConstructor::timeLabel(
myWindow->traceUnitsToWindowUnits( (*it).getCommPartnerTime() 
- (*it).getTime() ),
myWindow->getTimeUnit(),
0 ).c_str() );
else if( (*it).getRecordType() & RECV )
onString << wxT( ", Duration: " ) << wxString::FromUTF8( LabelConstructor::timeLabel(
myWindow->traceUnitsToWindowUnits( (*it).getTime()
- (*it).getCommPartnerTime() ),
myWindow->getTimeUnit(), 
0 ).c_str() );

onString << wxT( " (size: " ) << (*it).getCommSize() << 
wxT( ", tag: " ) << (*it).getCommTag() << wxT( ")" );
onString << wxT( "\n" );

whatWhereLines.push_back( make_pair( TWhatWhereLine::COMMUNICATION_LINE, onString ));
onString.clear();

if ( clickedValue )
whatWhereSelectedTimeCommunicationLines++;
}
++it;
}

rl->erase( rl->begin(), it );

if( clickedValue )
whatWhereLines.push_back( make_pair( TWhatWhereLine::END_CURRENT_SECTION, _( "" )));
else
whatWhereLines.push_back( make_pair( TWhatWhereLine::END_PREVNEXT_SECTION, _( "" )));

whatWhereLines.push_back( make_pair( TWhatWhereLine::END_RECORDS_SECTION, _( "" )));
whatWhereLines.push_back( make_pair( TWhatWhereLine::RAW_LINE, _( "\n" )));
}


void gTimeline::resizeDrawZone( int width, int height )
{
canRedraw = false;

if( !splitter->IsSplit() )
this->SetClientSize( width, height );
else
{
#ifdef _WIN32
this->SetClientSize( width, height + infoZoneLastSize + 5 );
#endif
splitter->SetSashPosition( height );
#ifndef _WIN32
drawZone->SetClientSize( width, height );
this->SetClientSize( width, height + infoZoneLastSize + 5 );
#endif
}
myWindow->setWidth( width );
myWindow->setHeight( height );
canRedraw = true;
}

bool gTimeline::IsSplit() const
{
return splitter->IsSplit();
}

void gTimeline::OnPopUpInfoPanel( wxCommandEvent& event )
{
if( splitter->IsSplit() )
{
canRedraw = false;
infoZoneLastSize = infoZone->GetSize().GetHeight();
splitter->Unsplit();
Unsplit();
}
else
Split();
}

void gTimeline::OnPopUpSaveCFG( wxCommandEvent& event )
{
saveCFG();
}

void gTimeline::OnPopUpSaveImageDialog( wxCommandEvent& event )
{
saveImageDialog();
}

void gTimeline::OnPopUpSaveText( wxCommandEvent& event )
{
saveText();
}

void gTimeline::Unsplit()
{
canRedraw = false;
this->Freeze();


this->SetClientSize( this->GetClientSize().GetWidth(), myWindow->getHeight() );

drawZone->SetClientSize( myWindow->getWidth(), myWindow->getHeight() );
this->Thaw();
canRedraw = true;
splitChanged = true;
}


void gTimeline::Split()
{
canRedraw = false;
this->Freeze();
splitter->SplitHorizontally( drawZone, infoZone, myWindow->getHeight() );
resizeDrawZone( myWindow->getWidth(), myWindow->getHeight() );
infoZone->SetClientSize( myWindow->getWidth(), infoZoneLastSize );
this->Thaw();
canRedraw = true;
splitChanged = true;
}


void gTimeline::OnPopUpTiming( wxCommandEvent& event )
{
EnableTiming( event.IsChecked() );
}

void gTimeline::EnableTiming( bool state )
{
SetTiming( state );
if( timing )
drawZone->SetCursor( *wxCROSS_CURSOR );
else
drawZone->SetCursor( wxNullCursor  );
}

class CustomColorSemValue : public wxObject
{
public:
TSemanticValue myValue;
wxPanel *myPanel;
};

void gTimeline::OnItemColorLeftUp( wxMouseEvent& event )
{
selectedItemColor = ( (CustomColorSemValue *)event.m_callbackUserData )->myPanel;
panelSelectedColor->SetBackgroundColour( selectedItemColor->GetBackgroundColour() );
panelSelectedColor->Refresh();
sliderSelectedRed->SetValue( selectedItemColor->GetBackgroundColour().Red() );
sliderSelectedGreen->SetValue( selectedItemColor->GetBackgroundColour().Green() );
sliderSelectedBlue->SetValue( selectedItemColor->GetBackgroundColour().Blue() );
textSelectedRed->ChangeValue( wxString::Format( wxT( "%i" ), selectedItemColor->GetBackgroundColour().Red() ) );
textSelectedGreen->ChangeValue( wxString::Format( wxT( "%i" ), selectedItemColor->GetBackgroundColour().Green() ) );
textSelectedBlue->ChangeValue( wxString::Format( wxT( "%i" ), selectedItemColor->GetBackgroundColour().Blue() ) );

selectedCustomValue = ( (CustomColorSemValue *)event.m_callbackUserData )->myValue;
}

void gTimeline::OnTextColorLeftUp( wxMouseEvent& event )
{
selectedItemColor = ( (CustomColorSemValue *)event.m_callbackUserData )->myPanel;
panelSelectedColor->SetBackgroundColour( selectedItemColor->GetBackgroundColour() );
panelSelectedColor->Refresh();
sliderSelectedRed->SetValue( selectedItemColor->GetBackgroundColour().Red() );
sliderSelectedGreen->SetValue( selectedItemColor->GetBackgroundColour().Green() );
sliderSelectedBlue->SetValue( selectedItemColor->GetBackgroundColour().Blue() );
textSelectedRed->ChangeValue( wxString::Format( wxT( "%i" ), selectedItemColor->GetBackgroundColour().Red() ) );
textSelectedGreen->ChangeValue( wxString::Format( wxT( "%i" ), selectedItemColor->GetBackgroundColour().Green() ) );
textSelectedBlue->ChangeValue( wxString::Format( wxT( "%i" ), selectedItemColor->GetBackgroundColour().Blue() ) );

selectedCustomValue = ( (CustomColorSemValue *)event.m_callbackUserData )->myValue;
}



void gTimeline::OnScrolledColorsUpdate( wxUpdateUIEvent& event )
{
if( !myWindow->getReady() )
return;

PRV_UINT32 precision = ParaverConfig::getInstance()->getTimelinePrecision();

if( forceRedoColors || 
( redoColors &&
( myWindow->getSemanticInfoType() != lastType ||
myWindow->getMinimumY() != lastMin ||
myWindow->getMaximumY() != lastMax ||
semanticValuesToColor.size() != lastValuesSize ||
myWindow->isCodeColorSet() != codeColorSet ||
myWindow->getGradientColor().getGradientFunction() != gradientFunc )
) 
)
{
lastType = myWindow->getSemanticInfoType();
lastMin = myWindow->getMinimumY();
lastMax = myWindow->getMaximumY();
lastValuesSize = semanticValuesToColor.size();
codeColorSet = myWindow->isCodeColorSet();
gradientFunc = myWindow->getGradientColor().getGradientFunction();

colorsSizer->Clear( true );
wxBoxSizer *itemSizer;
wxStaticText *itemText;
wxPanel *itemColor;

if( myWindow->isFusedLinesColorSet() )
{
TObjectOrder beginRow = myWindow->getZoomSecondDimension().first;
TObjectOrder endRow = myWindow->getZoomSecondDimension().second;
vector<TObjectOrder> selected;
myWindow->getSelectedRows( myWindow->getLevel(), selected, beginRow, endRow, true );

size_t i = 0;
for( vector<TObjectOrder>::iterator it = selected.begin(); it != selected.end(); ++it )
{
string tmpstr = LabelConstructor::objectLabel( *it, myWindow->getLevel(), myWindow->getTrace() );

itemSizer = new wxBoxSizer(wxHORIZONTAL);

itemText = new wxStaticText( colorsPanel, wxID_ANY, _T("") );
wxString tmpStr = wxString::FromUTF8( tmpstr.c_str() );
itemText->SetLabel( tmpStr );

wxSize tmpSize( 20, itemText->GetSize().GetHeight() );
itemColor = new wxPanel( colorsPanel, wxID_ANY, wxDefaultPosition, tmpSize );
rgb tmprgb = myWindow->getCodeColor().calcColor( (*it) + 1, 0, myWindow->getTrace()->getLevelObjects( myWindow->getLevel() ), false );
wxColour tmpColor( tmprgb.red, tmprgb.green, tmprgb.blue );
itemColor->SetBackgroundColour( tmpColor );

itemSizer->Add( itemColor );
itemSizer->AddSpacer( 5 );
itemSizer->Add( itemText, 1, wxGROW );
colorsSizer->Add( itemSizer, 0, wxGROW|wxALL, 2 );

if( i < selected.size() - 1 )
colorsSizer->Add( new wxStaticLine( colorsPanel, wxID_ANY ), 0, wxGROW|wxALL, 2 );
++i;
}
}
else if( myWindow->isCodeColorSet() )
{
int endLimit = 0;

for( auto it = semanticValuesToColor.cbegin(); it != semanticValuesToColor.cend(); ++it )
{
int i = std::ceil( it->first );

if( lastType == EVENTTYPE_TYPE && !myWindow->getTrace()->eventLoaded( i ) )
continue;

string tmpstr;
if( lastType == STATE_TYPE &&
!myWindow->getTrace()->getStateLabels().getStateLabel( i, tmpstr ) )
continue;
if( lastType == EVENTVALUE_TYPE &&
!myWindow->getTrace()->getEventLabels().getEventValueLabel( i, tmpstr ) )
continue;

if( endLimit > MAX_LEGEND_COLORS )
break;
else
++endLimit;

itemSizer = new wxBoxSizer(wxHORIZONTAL);

itemText = new wxStaticText( colorsPanel, wxID_ANY, _T("") );
wxString tmpStr = wxString::FromUTF8( LabelConstructor::semanticLabel( myWindow, i, true, precision,false ).c_str() );
itemText->SetLabel( tmpStr );

wxSize tmpSize( 20, itemText->GetSize().GetHeight() );
itemColor = new wxPanel( colorsPanel, wxID_ANY, wxDefaultPosition, tmpSize );
rgb tmprgb = myWindow->getCodeColor().calcColor( i, myWindow->getMinimumY(), myWindow->getMaximumY(), myWindow->getUseCustomPalette() );
wxColour tmpColor( tmprgb.red, tmprgb.green, tmprgb.blue );
itemColor->SetBackgroundColour( tmpColor );

itemSizer->Add( itemColor );
itemSizer->AddSpacer( 5 );
itemSizer->Add( itemText, 1, wxGROW );
colorsSizer->Add( itemSizer, 0, wxGROW|wxALL, 2 );

CustomColorSemValue *tmpItemSemValue = new CustomColorSemValue();
tmpItemSemValue->myValue = i;
tmpItemSemValue->myPanel = itemColor;
CustomColorSemValue *tmpTextSemValue = new CustomColorSemValue();
*tmpTextSemValue = *tmpItemSemValue;
if ( checkboxCustomPalette->IsChecked() )
{
itemColor->Connect( itemColor->GetId(), wxEVT_LEFT_UP, wxMouseEventHandler(gTimeline::OnItemColorLeftUp), tmpItemSemValue, this);
itemText->Connect( itemText->GetId(), wxEVT_LEFT_UP, wxMouseEventHandler(gTimeline::OnTextColorLeftUp), tmpTextSemValue, this);
}

if( i < ceil( myWindow->getMaximumY() ) )
colorsSizer->Add( new wxStaticLine( colorsPanel, wxID_ANY ), 0, wxGROW|wxALL, 2 );
}
}
else
{
itemSizer = new wxBoxSizer(wxHORIZONTAL);

itemText = new wxStaticText( colorsPanel, wxID_ANY, _T("") );
wxString tmpStr;
tmpStr << wxT("< ") << wxString::FromUTF8( LabelConstructor::semanticLabel( myWindow, lastMin, false, precision, false ).c_str() );
itemText->SetLabel( tmpStr );

wxSize tmpSize( 20, itemText->GetSize().GetHeight() );
itemColor = new wxPanel( colorsPanel, wxID_ANY, wxDefaultPosition, tmpSize );
rgb tmprgb = myWindow->getGradientColor().calcColor( lastMin - 1, lastMin, lastMax );
wxColour tmpColor( tmprgb.red, tmprgb.green, tmprgb.blue );
itemColor->SetBackgroundColour( tmpColor );

itemSizer->Add( itemColor );
itemSizer->AddSpacer( 5 );
itemSizer->Add( itemText, 1, wxGROW );
colorsSizer->Add( itemSizer, 0, wxGROW|wxALL, 2 );

colorsSizer->Add( new wxStaticLine( colorsPanel, wxID_ANY ), 0, wxGROW|wxALL, 2 );

TSemanticValue step = ( lastMax - lastMin ) / 20.0;
TSemanticValue lastValueToUse = 0.0;
for( int i = 0; i <= 20; ++i )
{
TSemanticValue valueToUse;
if( i > 0 && lastValueToUse != 0.0 && ( ( i - 1 ) * step ) + lastMin < 0.0 && ( i * step ) + lastMin > 0.0 )
{
lastValueToUse = valueToUse = 0.0;
--i;
}
else
lastValueToUse = valueToUse = ( i * step ) + lastMin;

itemSizer = new wxBoxSizer(wxHORIZONTAL);

itemText = new wxStaticText( colorsPanel, wxID_ANY, _T("") );
tmpStr.Clear();
tmpStr << wxString::FromUTF8( LabelConstructor::semanticLabel( myWindow, valueToUse, false, precision, false ).c_str() );
itemText->SetLabel( tmpStr );

tmpSize = wxSize( 20, itemText->GetSize().GetHeight() );
itemColor = new wxPanel( colorsPanel, wxID_ANY, wxDefaultPosition, tmpSize );
tmprgb = myWindow->getGradientColor().calcColor( valueToUse, lastMin, lastMax );
tmpColor = wxColour( tmprgb.red, tmprgb.green, tmprgb.blue );
itemColor->SetBackgroundColour( tmpColor );

itemSizer->Add( itemColor );
itemSizer->AddSpacer( 5 );
itemSizer->Add( itemText );
colorsSizer->Add( itemSizer, 0, wxGROW|wxALL, 2 );

colorsSizer->Add( new wxStaticLine( colorsPanel, wxID_ANY ), 0, wxGROW|wxALL, 2 );
}
itemSizer = new wxBoxSizer(wxHORIZONTAL);

itemText = new wxStaticText( colorsPanel, wxID_ANY, _T("") );
tmpStr.Clear();
tmpStr << wxT("> ") << wxString::FromUTF8( LabelConstructor::semanticLabel( myWindow, lastMax, false, precision, false ).c_str() );
itemText->SetLabel( tmpStr );

tmpSize = wxSize( 20, itemText->GetSize().GetHeight() );
itemColor = new wxPanel( colorsPanel, wxID_ANY, wxDefaultPosition, tmpSize );
tmprgb = myWindow->getGradientColor().calcColor( lastMax + 1, lastMin, lastMax );
tmpColor = wxColour( tmprgb.red, tmprgb.green, tmprgb.blue );
itemColor->SetBackgroundColour( tmpColor );

itemSizer->Add( itemColor );
itemSizer->AddSpacer( 5 );
itemSizer->Add( itemText );
colorsSizer->Add( itemSizer, 0, wxGROW|wxALL, 2 );
}
colorsPanel->Layout();
colorsPanel->FitInside();

selectedItemColor = nullptr;
panelSelectedColor->Enable( false );
}
forceRedoColors = false;
redoColors = false;
}


void gTimeline::drawCommunicationLines( bool draw )
{
wxMemoryDC bufferDraw;

myWindow->setDrawCommLines( draw );

if( !myWindow->getReady() )
return;

bufferDraw.SelectObject(wxNullBitmap);
bufferDraw.SelectObject( drawImage );
bufferDraw.DrawBitmap( bufferImage, 0, 0, false );

doDrawCaution( bufferDraw );

if( myWindow->getDrawFlags() )
bufferDraw.DrawBitmap( eventImage, 0, 0, true );

if( myWindow->getDrawCommLines() )
bufferDraw.DrawBitmap( commImage, 0, 0, true );

drawZone->Refresh();
}


void gTimeline::drawEventFlags( bool draw )
{
wxMemoryDC bufferDraw;

myWindow->setDrawFlags( draw );

if( !myWindow->getReady() )
return;

bufferDraw.SelectObject(wxNullBitmap);
bufferDraw.SelectObject( drawImage );
bufferDraw.DrawBitmap( bufferImage, 0, 0, false );

doDrawCaution( bufferDraw );

if( myWindow->getDrawFlags() )
bufferDraw.DrawBitmap( eventImage, 0, 0, true );

if( myWindow->getDrawCommLines() )
bufferDraw.DrawBitmap( commImage, 0, 0, true );

drawZone->Refresh();
}


void gTimeline::drawFunctionLineColor()
{
myWindow->setFunctionLineColorMode();
myWindow->setRedraw( true );
}

void gTimeline::drawFusedLinesColor()
{
myWindow->setFusedLinesColorMode();
myWindow->setRedraw( true );
}


wxString gTimeline::buildFormattedFileName() const
{
return ( paraverMain::buildFormattedFileName( myWindow->getName(), myWindow->getTrace()->getTraceNameNumbered() ) );
}


void gTimeline::saveImageDialog( wxString whichFileName )
{
wxString imageName;
wxString tmpSuffix; 

setEnableDestroyButton( false );
imageName = buildFormattedFileName();


wxFileName startingDir( wxString::FromUTF8( myWindow->getTrace()->getFileName().c_str() ) );
wxString defaultDir = startingDir.GetPath();

TImageFormat filterIndex = ParaverConfig::getInstance()->getTimelineSaveImageFormat();

wxString legendSuffix = _( ".code_legend" );
if ( myWindow->isGradientColorSet() )
legendSuffix= _( ".gradient_legend" );
else if ( myWindow->isNotNullGradientColorSet() )
legendSuffix= _( ".nn_gradient_legend" );
else if ( myWindow->isAlternativeGradientColorSet() )
legendSuffix= _( ".alt_gradient_legend" );

SaveImageDialog saveDialog( this, defaultDir, imageName, false, legendSuffix );
if ( saveDialog.ShowModal() != wxID_OK )
{
setEnableDestroyButton( true );
return;
}

filterIndex = TImageFormat( saveDialog.GetFilterIndex() ); 


if ( saveDialog.DialogSavesImage() )
{
imageName = saveDialog.GetImageFilePath(); 
saveImage( imageName, filterIndex );
}

if ( saveDialog.DialogSavesLegend() )
{
imageName = saveDialog.GetLegendFilePath(); 
bool appendLegendSuffix = false;
saveImageLegend( imageName, filterIndex, appendLegendSuffix );
}
}

void gTimeline::saveImage( wxString whichFileName, TImageFormat filterIndex )
{
wxString imagePath;


setEnableDestroyButton( false );

if( !whichFileName.IsEmpty() )
{
imagePath = whichFileName;
}
else
{
wxString imageName;
wxString tmpSuffix;
wxString defaultDir;

imageName = buildFormattedFileName();

#ifdef _WIN32
defaultDir = _(".\\");
#else
defaultDir = _("./");
#endif

filterIndex = ParaverConfig::getInstance()->getTimelineSaveImageFormat();
tmpSuffix = _(".") +
wxString::FromUTF8( LabelConstructor::getImageFileSuffix( filterIndex ).c_str() );
imagePath = imageName + tmpSuffix;
}

wxString longTitle = wxString::FromUTF8(
( myWindow->getName() + " @ " +
myWindow->getTrace()->getTraceNameNumbered() ).c_str());
wxString shortTitle = wxString::FromUTF8( ( myWindow->getName() ).c_str() );
wxString writtenTitle = longTitle;

wxColour foregroundColour = GetForegroundColour();
wxColour backgroundColour = GetBackgroundColour();

wxFont titleFont = semanticFont;

wxImage img = drawImage.ConvertToImage();
int timelineWidth = img.GetWidth();
int timelineHeight = img.GetHeight();

int titleMargin = 5; 
int titleHeigth = titleFont.GetPointSize() + ( 2 * titleMargin ); 
int titleWidth = timelineWidth;
int titleWritableWidth = titleWidth - ( 2 * titleMargin );

int imageHeigth = titleHeigth + timelineHeight;
int imageWidth = timelineWidth;

wxBitmap titleBitmap( titleWidth, titleHeigth );
wxMemoryDC titleDC( titleBitmap );

titleDC.SetFont( titleFont );
wxSize titleSize = titleDC.GetTextExtent( writtenTitle );

if ( titleSize.GetWidth() > titleWritableWidth )
{
titleSize = titleDC.GetTextExtent( shortTitle );
writtenTitle = shortTitle;
}

titleDC.SetBackground( wxBrush( backgroundColour ) );
titleDC.Clear();

titleDC.SetPen( wxPen( backgroundColour, 1 ) );
titleDC.SetTextBackground( backgroundColour );
titleDC.SetTextForeground( foregroundColour );

titleDC.DrawText( writtenTitle, titleMargin, titleMargin );

wxBitmap imageBitmap( imageWidth, imageHeigth );
wxMemoryDC imageDC( imageBitmap );
wxCoord xsrc = 0;
wxCoord ysrc = 0;
wxCoord xdst = 0;
wxCoord ydst = 0;
imageDC.Blit( xdst, ydst, titleWidth, titleHeigth, &titleDC, xsrc, ysrc );

#ifdef __WXMAC__
wxBitmap tmpDrawImage( drawImage.GetWidth(), drawImage.GetHeight() );
wxMemoryDC timelineDC( tmpDrawImage );
drawStackedImages( timelineDC );
#else
wxMemoryDC timelineDC( drawImage );
#endif
xsrc = 0;
ysrc = 0;
xdst = 0;
ydst = titleHeigth;
imageDC.Blit( xdst, ydst, timelineWidth, timelineHeight, &timelineDC, xsrc, ysrc );

wxBitmapType imageType;

switch( filterIndex )
{
case  TImageFormat::BMP:
imageType = wxBITMAP_TYPE_BMP;
break;
case  TImageFormat::JPG:
imageType = wxBITMAP_TYPE_JPEG;
break;
case  TImageFormat::PNG:
imageType = wxBITMAP_TYPE_PNG;
break;
case TImageFormat::XPM:
imageType = wxBITMAP_TYPE_XPM;
break;
default:
imageType = wxBITMAP_TYPE_PNG;
break;
}

imageDC.SelectObject( wxNullBitmap );
wxImage baseLayer = imageBitmap.ConvertToImage();

if ( myWindow->isGradientColorSet() ||
myWindow->isNotNullGradientColorSet() ||
myWindow->isAlternativeGradientColorSet() )
{
ScaleImageVertical *tmpImage;

wxMemoryDC tmpTimelineDC( imageBitmap ); 

int wantedWidth = baseLayer.GetWidth();
int backgroundMode = wxSOLID;
tmpImage = new ScaleImageHorizontalGradientColor(
myWindow,
semanticValuesToColor,
backgroundColour, foregroundColour, backgroundMode,
titleFont,
imagePath, wxString( _( "horiz.labels" ) ),
imageType,
wantedWidth );
tmpImage->process();

wxBitmap *tmpScale = tmpImage->getBitmap();
wxMemoryDC tmpScaleDC( *tmpScale );

int totalHeigth = baseLayer.GetHeight() + tmpScale->GetHeight();
int totalWidth  = baseLayer.GetWidth() > tmpScale->GetWidth() ? baseLayer.GetWidth(): tmpScale->GetWidth();
wxBitmap tmpScaledTimelineBitmap( totalWidth, totalHeigth );
wxMemoryDC tmpScaledTimelineDC( tmpScaledTimelineBitmap );
tmpScaledTimelineDC.SetBackground( backgroundColour );
tmpScaledTimelineDC.Clear();

int xsrc = 0;
int ysrc = 0;
int xdst = 0;
int ydst = 0;

tmpScaledTimelineDC.Blit( xdst, ydst,
imageBitmap.GetWidth(), imageBitmap.GetHeight(),
&tmpTimelineDC,
xsrc, ysrc );

xdst = totalWidth - tmpScale->GetWidth();
if ( xdst < 0 )
{
xdst = 0;
}
ydst = imageBitmap.GetHeight();
tmpScaledTimelineDC.Blit( xdst, ydst,
tmpScale->GetWidth(), tmpScale->GetHeight(),
&tmpScaleDC,
xsrc, ysrc );


wxImage tmpScaledTimeline( tmpScaledTimelineBitmap.ConvertToImage() );
wxString currentFormat =
wxString::FromUTF8( LabelConstructor::getImageFileSuffix(
TImageFormat( filterIndex ) ).c_str() );
wxString tmpScaledTimelinePath = wxFileName( imagePath ).GetPathWithSep() +
wxFileName( imagePath ).GetName() +
wxString( _(".w_legend.") ) +
currentFormat;
tmpScaledTimeline.SaveFile( tmpScaledTimelinePath, imageType );
}

baseLayer.SaveFile( imagePath, imageType );

setEnableDestroyButton( true );
}


void gTimeline::saveImageLegend( wxString whichFileName, TImageFormat filterIndex, bool appendLegendSuffix )
{
wxString imageName;
wxString tmpSuffix;
wxString defaultDir;

setEnableDestroyButton( false );

if( !whichFileName.IsEmpty() )
imageName = whichFileName;
else
imageName = buildFormattedFileName();

#ifdef _WIN32
defaultDir = _(".\\");
#else
defaultDir = _("./");
#endif

tmpSuffix = _("");
if ( appendLegendSuffix )
{
tmpSuffix += _(".");
if ( myWindow->isGradientColorSet() )
tmpSuffix +=
wxString( _( "gradient_legend" ) ) +
_(".") +
wxString::FromUTF8( LabelConstructor::getImageFileSuffix( filterIndex ).c_str() );
else if ( myWindow->isNotNullGradientColorSet() )
tmpSuffix +=
wxString( _( "nn_gradient_legend" ) ) +
_(".") +
wxString::FromUTF8( LabelConstructor::getImageFileSuffix( filterIndex ).c_str() );
else if ( myWindow->isAlternativeGradientColorSet() )
tmpSuffix +=
wxString( _( "alt_gradient_legend" ) ) +
_(".") +
wxString::FromUTF8( LabelConstructor::getImageFileSuffix( filterIndex ).c_str() );
else
tmpSuffix +=
wxString( _( "code_legend" ) ) +
_(".") +
wxString::FromUTF8( LabelConstructor::getImageFileSuffix( filterIndex ).c_str() );
}

wxString imagePath = imageName + tmpSuffix;

wxColour foregroundColour = GetForegroundColour();
wxColour backgroundColour = GetBackgroundColour();

wxFont titleFont = semanticFont;

wxBitmapType imageType;

int backgroundMode = wxTRANSPARENT; 
switch( filterIndex )
{
case  TImageFormat::BMP:
imageType = wxBITMAP_TYPE_BMP;
backgroundMode = wxSOLID;
break;
case  TImageFormat::JPG:
imageType = wxBITMAP_TYPE_JPEG;
backgroundMode = wxSOLID;
break;
case  TImageFormat::PNG:
imageType = wxBITMAP_TYPE_PNG;
break;
case TImageFormat::XPM:
imageType = wxBITMAP_TYPE_XPM;
break;
default:
imageType = wxBITMAP_TYPE_PNG;
break;
}

ScaleImageVertical *tmpImage;
if ( myWindow->isGradientColorSet() ||
myWindow->isNotNullGradientColorSet() ||
myWindow->isAlternativeGradientColorSet() )
{
tmpImage = new ScaleImageHorizontalGradientColor( myWindow, semanticValuesToColor,
*wxWHITE, *wxBLACK, backgroundMode,
titleFont,
imagePath, wxString( _( "" ) ),
imageType );
tmpImage->process();
tmpImage->save();
delete tmpImage;
}
else if ( myWindow->isCodeColorSet() )
{
tmpImage = new ScaleImageVerticalCodeColor( myWindow, semanticValuesToColor,
*wxWHITE, *wxBLACK, backgroundMode,
titleFont,
imagePath, wxString( _( "" ) ),
imageType );
tmpImage->process();
tmpImage->save();
delete tmpImage;
}
else if ( myWindow->isFusedLinesColorSet() )
{
std::map< TSemanticValue, rgb > tmpObjects;

TObjectOrder beginRow = myWindow->getZoomSecondDimension().first;
TObjectOrder endRow = myWindow->getZoomSecondDimension().second;
vector<TObjectOrder> selected;
myWindow->getSelectedRows( myWindow->getLevel(), selected, beginRow, endRow, true );

for( vector<TObjectOrder>::iterator it = selected.begin(); it != selected.end(); ++it )
{
rgb tmprgb = myWindow->getCodeColor().calcColor( (*it) + 1, 0, myWindow->getTrace()->getLevelObjects( myWindow->getLevel() ) - 1, false );
tmpObjects[ (TSemanticValue)(*it) ] = tmprgb;
}
tmpImage = new ScaleImageVerticalFusedLines( myWindow, tmpObjects,
*wxWHITE, *wxBLACK, backgroundMode,
titleFont,
imagePath, wxString( _( "" ) ),
imageType );
tmpImage->process();
tmpImage->save();
delete tmpImage;
}

setEnableDestroyButton( true );
}


gTimeline::ScaleImageVertical::ScaleImageVertical(
Timeline* whichMyWindow,
const std::map< TSemanticValue, rgb >& whichSemanticValues,
wxColour whichBackground,
wxColour whichForeground,
int whichBackgroundMode,
wxFont whichTextFont,
wxString& whichImagePath,
const wxString& whichImageInfix,
wxBitmapType& whichImageType 
) : myWindow( whichMyWindow ),
semValues( whichSemanticValues ),
background( whichBackground ),
foreground( whichForeground ),
backgroundMode( whichBackgroundMode ),
textFont( whichTextFont ),
imagePath( whichImagePath ),
imageInfix( whichImageInfix ),
imageType( whichImageType )
{
}


gTimeline::ScaleImageVertical::~ScaleImageVertical()
{
destroyDC();
}


wxString gTimeline::ScaleImageVertical::buildScaleImagePath()
{
wxString scaleImagePath;

if ( imageInfix.IsEmpty() )
scaleImagePath = imagePath;
else
scaleImagePath =
wxFileName( imagePath ).GetPathWithSep() +
wxFileName( imagePath ).GetName() +
_(".") + imageInfix + _(".") + tmpSuffix;

return scaleImagePath;
}


void gTimeline::ScaleImageVertical::bitmapToImage()
{
scaleDC->SelectObject( wxNullBitmap );

#ifndef __WXMAC__
if ( backgroundMode == wxTRANSPARENT )
{
scaleMaskDC->SelectObject( wxNullBitmap );
wxMask *mask  = new wxMask( *scaleMaskBitmap, *wxWHITE );
scaleBitmap->SetMask( mask );
}
#endif

scaleImage = new wxImage( scaleBitmap->ConvertToImage() );
}


void gTimeline::ScaleImageVertical::process()
{
init();
sortSemanticValues();
computeMaxLabelSize();
computeImageSize();
createDC();
draw();
bitmapToImage();
}


void gTimeline::ScaleImageVertical::save()
{
wxString scaleImagePath = buildScaleImagePath();
scaleImage->SaveFile( scaleImagePath, imageType );

#if 0
::wxInitAllImageHandlers();
wxBitmap bmp( 64, 64, 32 );
wxBitmap maskbmp( 64, 64, 32 );

wxMemoryDC dc( bmp );
dc.SetBackground( *wxWHITE_BRUSH );
dc.SetBrush( *wxWHITE_BRUSH );
dc.Clear();

wxMemoryDC maskDC( maskbmp );


dc.SetFont( textFont );
maskDC.SetBackground( *wxBLACK_BRUSH );
maskDC.Clear();
maskDC.SetBrush( *wxWHITE_BRUSH );
maskDC.SetPen( *wxWHITE_PEN );
maskDC.SetTextForeground( *wxWHITE );
maskDC.SetFont( textFont );

dc.SetBrush( *wxRED_BRUSH );
dc.SetPen( *wxBLACK_PEN );
dc.DrawRectangle( 10, 30, 10, 10 );
maskDC.DrawRectangle( 10, 30, 10, 10 );
dc.SetTextForeground( *wxBLACK );
dc.DrawText( wxT( "HELLO" ), 10, 10 );
maskDC.DrawText( wxT( "HELLO" ), 10, 10 );
dc.SelectObject( wxNullBitmap );
maskDC.SelectObject( wxNullBitmap );
maskbmp.SaveFile( wxT( "./mask.png" ), wxBITMAP_TYPE_PNG );
wxMask *tmpMask = new wxMask( maskbmp, *wxBLACK );
bmp.SetMask( tmpMask );
bmp.SaveFile( wxT( "./test.png" ), wxBITMAP_TYPE_PNG );
#endif
}


void gTimeline::ScaleImageVertical::init()
{
filterIndex = ParaverConfig::getInstance()->getTimelineSaveImageFormat();
tmpSuffix = _("labels.");
tmpSuffix += wxString::FromUTF8( LabelConstructor::getImageFileSuffix( filterIndex ).c_str() );

currentMin = myWindow->getMinimumY();
currentMax = myWindow->getMaximumY();

precision = ParaverConfig::getInstance()->getTimelinePrecision();
symbolicDesc = true;

extraPrefixOutlier = wxT("");

titleMargin = 5; 

widthRect = 10;
heightRect = textFont.GetPointSize() + 3;

imageStepY = textFont.GetPointSize() + 2 * titleMargin;
imageStepXRectangle = widthRect;

xdst = titleMargin;
ydst = titleMargin;
}


void gTimeline::ScaleImageVertical::computeMaxLabelSize()
{
wxString curLabel;
wxString maxLabel;
size_t maxLengthLabel = 0;
size_t curLengthLabel;
for ( auto it = semValues.begin(); it != semValues.end(); ++it )
{
curLabel = wxString::FromUTF8( LabelConstructor::semanticLabel( myWindow, it->first, symbolicDesc, precision, false ).c_str() );
semanticValueLabel[ it->first ] = curLabel;

curLengthLabel = curLabel.Len();
if ( maxLengthLabel < curLengthLabel )
{
maxLengthLabel = curLengthLabel;
semanticValueWithLongestLabel = it->first;
}
}

wxBitmap maxLabelBitmap( 1920, 64 );
wxMemoryDC maxLabelDC( maxLabelBitmap );
maxLabelDC.SetFont( textFont );
maxLabelSize = maxLabelDC.GetTextExtent( extraPrefixOutlier +
semanticValueLabel[ semanticValueWithLongestLabel ] );
}


void gTimeline::ScaleImageVertical::sortSemanticValues()
{}


void gTimeline::ScaleImageVertical::computeImageSize()
{
imageHeight = semanticValueLabel.size() * imageStepY + ( 2 * titleMargin );

if ( drawOutliers )
imageHeight += ( imageStepY + titleMargin ) * 2;

imageWidth = maxLabelSize.GetWidth() + ( 2 * titleMargin ) + ( widthRect + titleMargin );
}


void gTimeline::ScaleImageVertical::createDC()
{
scaleBitmap = new wxBitmap( imageWidth, imageHeight );
scaleDC = new wxMemoryDC( *scaleBitmap );

scaleDC->SetFont( textFont );
if ( backgroundMode == wxSOLID )
{
scaleDC->SetBackground( background );
scaleDC->SetTextBackground( background );
scaleDC->SetTextForeground( foreground );
scaleDC->SetPen( foreground );
}
else
{
scaleDC->SetBackground( *wxWHITE_BRUSH );
scaleDC->SetTextForeground( *wxBLACK );
}

scaleDC->Clear();

scaleMaskBitmap = new wxBitmap( imageWidth, imageHeight );
scaleMaskDC = new wxMemoryDC( *scaleMaskBitmap );
scaleMaskDC->SetBackground( *wxWHITE_BRUSH );
scaleMaskDC->SetBrush( *wxBLACK_BRUSH );
scaleMaskDC->SetPen( *wxBLACK_PEN );
scaleMaskDC->SetTextForeground( *wxBLACK );
scaleMaskDC->SetFont( textFont );
scaleMaskDC->Clear();
}


void gTimeline::ScaleImageVertical::draw()
{
for ( auto it = semValues.begin(); it != semValues.end(); ++it )
{
drawLabeledRectangle( it->second, semanticValueLabel[ it->first ] );
ydst += imageStepY;
}
}


void gTimeline::ScaleImageVertical::drawLabeledRectangle( rgb semanticColour,
wxString semanticValueLabel,
bool drawIt )
{
scaleDC->SetBrush( wxColour( semanticColour.red,
semanticColour.green,
semanticColour.blue ) );
scaleDC->DrawRectangle( xdst, ydst, widthRect, heightRect );
scaleMaskDC->DrawRectangle( xdst, ydst, widthRect, heightRect );

if ( drawIt )
{
scaleDC->DrawText( semanticValueLabel, xdst + widthRect + titleMargin, ydst );
scaleMaskDC->DrawText( semanticValueLabel, xdst + widthRect + titleMargin, ydst );
}
}


void gTimeline::ScaleImageVertical::destroyDC()
{
delete scaleDC;
delete scaleMaskDC;
delete scaleBitmap;
delete scaleMaskBitmap;
delete scaleImage;
}


gTimeline::ScaleImageVerticalCodeColor::ScaleImageVerticalCodeColor(
Timeline* whichMyWindow, 
const std::map< TSemanticValue, rgb >& whichSemanticValues,
wxColour whichBackground,
wxColour whichForeground,
int whichBackgroundMode,
wxFont whichTextFont,
wxString& whichImagePath,
const wxString& whichImageInfix,
wxBitmapType& whichImageType 
) : ScaleImageVertical( whichMyWindow,
whichSemanticValues,
whichBackground,
whichForeground,
whichBackgroundMode,
whichTextFont,
whichImagePath,
whichImageInfix,
whichImageType )
{
}


void gTimeline::ScaleImageVerticalCodeColor::init()
{
gTimeline::ScaleImageVertical::init();

drawOutliers = false;
symbolicDesc = true;
}


gTimeline::ScaleImageVerticalGradientColor::ScaleImageVerticalGradientColor(
Timeline* whichMyWindow, 
const std::map< TSemanticValue, rgb >& whichSemanticValues,
wxColour whichBackground,
wxColour whichForeground,
int whichBackgroundMode,
wxFont whichTextFont,
wxString& whichImagePath,
const wxString& whichImageInfix,
wxBitmapType& whichImageType 
) : ScaleImageVertical( whichMyWindow,
whichSemanticValues,
whichBackground,
whichForeground,
whichBackgroundMode,
whichTextFont,
whichImagePath,
whichImageInfix,
whichImageType )
{
}


void gTimeline::ScaleImageVerticalGradientColor::init()
{
gTimeline::ScaleImageVertical::init();

drawOutliers = true;
symbolicDesc = false;

numSquaresWithoutOutliers = 20;
if ( myWindow->isNotNullGradientColorSet() || myWindow->isAlternativeGradientColorSet() )
++numSquaresWithoutOutliers; 
totalSquares = numSquaresWithoutOutliers + 2; 

extraPrefixOutlier = wxT( "< " );
}


void gTimeline::ScaleImageVerticalGradientColor::sortSemanticValues()
{
semValues.clear();
TSemanticValue step = ( currentMax - currentMin ) / double( numSquaresWithoutOutliers - 1 ); 
for( int i = 0; i < numSquaresWithoutOutliers; ++i )
{
TSemanticValue current = ( i * step ) + currentMin;
if ( current > currentMax )
current = currentMax;

semValues[ current ] = myWindow->getGradientColor().calcColor( current, currentMin, currentMax );
}
}


void gTimeline::ScaleImageVerticalGradientColor::draw()
{
rgb tmprgb = myWindow->getGradientColor().getBelowOutlierColor();
wxString tmpSemanticValueLabel =
wxT( "< " ) +
wxString::FromUTF8( LabelConstructor::semanticLabel( myWindow, semValues.begin()->first, symbolicDesc, precision, false ).c_str() );
drawLabeledRectangle( tmprgb, tmpSemanticValueLabel );
ydst += imageStepY;

for ( auto it = semValues.begin(); it != semValues.end(); ++it )
{
drawLabeledRectangle( it->second, semanticValueLabel[ it->first ] );   
ydst += imageStepY;
}

tmprgb = myWindow->getGradientColor().getAboveOutlierColor();
tmpSemanticValueLabel =
wxT( "> " ) +
wxString::FromUTF8( LabelConstructor::semanticLabel( myWindow, (--semValues.end())->first, symbolicDesc, precision, false ).c_str() );
drawLabeledRectangle( tmprgb, tmpSemanticValueLabel );
ydst += imageStepY;
}


gTimeline::ScaleImageVerticalFusedLines::ScaleImageVerticalFusedLines(
Timeline* whichMyWindow, 
const std::map< TSemanticValue, rgb >& whichSemanticValues,
wxColour whichBackground,
wxColour whichForeground,
int whichBackgroundMode,
wxFont whichTextFont,
wxString& whichImagePath,
const wxString& whichImageInfix,
wxBitmapType& whichImageType 
) : ScaleImageVertical( whichMyWindow,
whichSemanticValues,
whichBackground,
whichForeground,
whichBackgroundMode,
whichTextFont,
whichImagePath,
whichImageInfix,
whichImageType )
{
}


void gTimeline::ScaleImageVerticalFusedLines::init()
{
gTimeline::ScaleImageVertical::init();

currentMin = 0;
currentMax = myWindow->getTrace()->getLevelObjects( myWindow->getLevel() ) - 1;
}

void gTimeline::ScaleImageVerticalFusedLines::computeMaxLabelSize()
{
wxString curLabel;
wxString maxLabel;
size_t maxLengthLabel = 0;
size_t curLengthLabel;
for ( auto it = semValues.begin(); it != semValues.end(); ++it )
{
curLabel = wxString::FromUTF8( LabelConstructor::objectLabel( it->first, myWindow->getLevel(), myWindow->getTrace() ).c_str() );
semanticValueLabel[ it->first ] = curLabel;

curLengthLabel = curLabel.Len();
if ( maxLengthLabel < curLengthLabel )
{
maxLengthLabel = curLengthLabel;
semanticValueWithLongestLabel = it->first;
}
}

wxBitmap maxLabelBitmap( 1920, 64 );
wxMemoryDC maxLabelDC( maxLabelBitmap );
maxLabelDC.SetFont( textFont );
maxLabelSize = maxLabelDC.GetTextExtent( extraPrefixOutlier +
semanticValueLabel[ semanticValueWithLongestLabel ] );
}

gTimeline::ScaleImageHorizontalGradientColor::ScaleImageHorizontalGradientColor(
Timeline* whichMyWindow, 
const std::map< TSemanticValue, rgb >& whichSemanticValues,
wxColour whichBackground,
wxColour whichForeground,
int whichBackgroundMode,
wxFont whichTextFont,
wxString& whichImagePath,
const wxString& whichImageInfix,
wxBitmapType& whichImageType,
int whichWantedWidth
) : ScaleImageVerticalGradientColor( whichMyWindow,
whichSemanticValues,
whichBackground,
whichForeground,
whichBackgroundMode,
whichTextFont,
whichImagePath,
whichImageInfix,
whichImageType )
{
wantedWidth = whichWantedWidth;
}


void gTimeline::ScaleImageHorizontalGradientColor::init()
{
gTimeline::ScaleImageVerticalGradientColor::init();

widthRect = 20;
imageStepXRectangle = widthRect;
outlierMargin = 2 * titleMargin;
SIZE_OF_TINY_MARK = 4;
}


void gTimeline::ScaleImageHorizontalGradientColor::computeImageSize()
{
imageHeight = heightRect + maxLabelSize.GetHeight() + 3 * titleMargin;

if ( wantedWidth > 0 )
{
imageWidth = totalSquares * widthRect + 2 * outlierMargin + 2 * titleMargin;
if ( imageWidth > wantedWidth )
{
float MINIMUM_WIDTH_RECT = 10.0;
float variableWidthRect = (float)(( wantedWidth - 2 * outlierMargin - 2 * titleMargin ) / totalSquares );
while (( totalSquares > 0 ) &&
( variableWidthRect < MINIMUM_WIDTH_RECT ) &&
( variableWidthRect > 0.0 ))
{
--totalSquares;
variableWidthRect = (float)(( wantedWidth - 2 * outlierMargin - 2 * titleMargin ) / totalSquares );
}

widthRect = rint( variableWidthRect );
imageStepXRectangle = widthRect;
}
else if ( imageWidth < wantedWidth )
{
int extraSquares = rint( floorf( float( ( wantedWidth - imageWidth ) / widthRect )));
totalSquares += extraSquares;

if ( myWindow->isNotNullGradientColorSet() )
numSquaresWithoutOutliers = totalSquares - 3;
else
numSquaresWithoutOutliers = totalSquares - 2;
}

sortSemanticValues();
computeMaxLabelSize();
}

imageWidth = totalSquares * widthRect + 2 * outlierMargin + 2 * titleMargin;
}


void gTimeline::ScaleImageHorizontalGradientColor::draw()
{

int initialXdst = xdst;

rgb tmprgb = myWindow->getGradientColor().getBelowOutlierColor();
drawRectangle( tmprgb );
xdst += imageStepXRectangle + outlierMargin;

int elem = 0;
int MIDDLE_POS = rint( numSquaresWithoutOutliers / 2 ) + 1; 
for ( auto it = semValues.begin(); it != semValues.end(); ++it )
{
++elem;
if ( it == semValues.begin() )
drawRectangle( it->second, FIRST );
else if ( it == --semValues.end() )
drawRectangle( it->second, LAST  );
else if ( elem == MIDDLE_POS )
drawRectangle( it->second, MIDDLE );
else
drawRectangle( it->second, ANY );

xdst += imageStepXRectangle;
}

xdst += outlierMargin;
tmprgb = myWindow->getGradientColor().getAboveOutlierColor();
drawRectangle( tmprgb );


xdst = initialXdst;
ydst += imageStepY - titleMargin + 2;

wxString tmpSemanticValueLabel = wxT( "<" );
drawLabel( tmpSemanticValueLabel );
xdst += imageStepXRectangle + outlierMargin;

bool drawIt;
elem = 0;
for ( auto it = semValues.begin(); it != semValues.end(); ++it )
{
++elem;
drawIt = it == semValues.begin() || elem == MIDDLE_POS || it == --semValues.end();

if ( it == semValues.begin() )
drawLabel( semanticValueLabel[ it->first ], drawIt, LEFT );
else if ( it == --semValues.end() )
{
drawLabel( semanticValueLabel[ it->first ], drawIt, RIGHT );
}
else if ( elem == MIDDLE_POS )
drawLabel( semanticValueLabel[ it->first ], drawIt, CENTER );
else
drawLabel( semanticValueLabel[ it->first ], drawIt, LEFT );

xdst += imageStepXRectangle;
}

xdst += outlierMargin;
tmpSemanticValueLabel = wxT( ">" );
drawIt = true;
drawLabel( tmpSemanticValueLabel, drawIt, RIGHT );
}


void gTimeline::ScaleImageHorizontalGradientColor::drawRectangle( rgb semanticColour, TPosition position )
{
scaleDC->SetBrush( wxColour( semanticColour.red,
semanticColour.green,
semanticColour.blue ) );
scaleDC->DrawRectangle( xdst, ydst, widthRect, heightRect );

scaleMaskDC->DrawRectangle( xdst, ydst, widthRect, heightRect );

switch( position )
{
case FIRST:
scaleDC->DrawLine( xdst, ydst + heightRect,
xdst, ydst + heightRect + SIZE_OF_TINY_MARK );
scaleMaskDC->DrawLine( xdst, ydst + heightRect,
xdst, ydst + heightRect + SIZE_OF_TINY_MARK );
break;
case MIDDLE:
scaleDC->DrawLine( xdst + ceil( (double)widthRect / 2 ), ydst + heightRect,
xdst + ceil( (double)widthRect / 2 ), ydst + heightRect + SIZE_OF_TINY_MARK );
scaleMaskDC->DrawLine( xdst + ceil( (double)widthRect / 2 ), ydst + heightRect,
xdst + ceil( (double)widthRect / 2 ), ydst + heightRect + SIZE_OF_TINY_MARK );
break;
case LAST:
scaleDC->DrawLine( xdst + widthRect - 1, ydst + heightRect,
xdst + widthRect - 1, ydst + heightRect + SIZE_OF_TINY_MARK );
scaleMaskDC->DrawLine( xdst + widthRect - 1, ydst + heightRect,
xdst + widthRect - 1, ydst + heightRect + SIZE_OF_TINY_MARK );
break;
default:
break;
}  
}


void gTimeline::ScaleImageHorizontalGradientColor::drawLabel( wxString semanticValueLabel, bool writeIt, TAlign align )
{
if ( writeIt )
{
wxSize labelSize = scaleDC->GetTextExtent( semanticValueLabel );

int shiftLeft;
switch( align )
{
case LEFT:
shiftLeft = 0;
break;
case CENTER:
shiftLeft = ceil( (double)labelSize.GetWidth() / 2 - (double)widthRect / 2 );
break;
case RIGHT:
shiftLeft = labelSize.GetWidth() - widthRect;
break;
default:
break;
}

int tmpXdst = xdst;
if (( xdst - shiftLeft ) >= 0 )
{
tmpXdst = xdst - shiftLeft;
}

scaleDC->DrawText( semanticValueLabel, tmpXdst, ydst);
scaleMaskDC->DrawText( semanticValueLabel, tmpXdst, ydst );
}
}



void gTimeline::saveText()
{
wxString fileName;
wxString tmpSuffix;
wxString defaultDir;

setEnableDestroyButton( false );

fileName = buildFormattedFileName();

#ifdef _WIN32
defaultDir = _(".\\");
#else
defaultDir = _("./");
#endif

tmpSuffix = _(".") +
wxString::FromUTF8( LabelConstructor::getDataFileSuffix(
ParaverConfig::getInstance()->getTimelineSaveTextFormat() ).c_str() );

wxString tmpWildcard;
vector< wxString > extensions;
for ( PRV_UINT16 i = 0; i < PRV_UINT16( TTextFormat::PLAIN ); ++i )
{
wxString currentFormat =
wxString::FromUTF8( LabelConstructor::getDataFileSuffix(
TTextFormat( i ) ).c_str() );
tmpWildcard +=
currentFormat.Upper() + _(" (*.") + currentFormat + _(")|*.") + currentFormat + _("|");

extensions.push_back( currentFormat );
}
tmpWildcard = tmpWildcard.BeforeLast( '|' );

FileDialogExtension saveDialog( this,
_("Save as..."),
defaultDir,
fileName + tmpSuffix,
tmpWildcard,
wxFD_SAVE | wxFD_CHANGE_DIR,
wxDefaultPosition,
wxDefaultSize,
_( "filedlg" ),
extensions );

saveDialog.SetFilterIndex( static_cast< int >( ParaverConfig::getInstance()->getTimelineSaveTextFormat() ) );

if ( saveDialog.ShowModal() == wxID_OK )
{
ProgressController *progress = ProgressController::create( paraverMain::myParaverMain->GetLocalKernel() );
progress->setHandler( progressFunction, this );

if( paraverMain::dialogProgress == nullptr )
paraverMain::dialogProgress = new wxProgressDialog( wxT("Save Timeline Text"),
wxT(""),
MAX_PROGRESS_BAR_VALUE,
this,
wxPD_CAN_ABORT|wxPD_AUTO_HIDE|\
wxPD_APP_MODAL|wxPD_ELAPSED_TIME|\
wxPD_ESTIMATED_TIME|wxPD_REMAINING_TIME );

string fileName = string( saveDialog.GetPath().mb_str() );
string reducePath;

if( fileName.length() > 36 && fileName.find_last_of( PATH_SEP ) != string::npos )
{
string file = fileName.substr( fileName.find_last_of( PATH_SEP ) );
string tmp = fileName.substr( 0, fileName.find_last_of( PATH_SEP ) );
if ( tmp.find_last_of( PATH_SEP ) != string::npos )
{
reducePath = "/..." + fileName.substr( tmp.find_last_of( PATH_SEP ),
tmp.length() - tmp.find_last_of( PATH_SEP ) )
+ file;
}
else
{
reducePath = "/..." + file;
}
}
else
reducePath = fileName;
reducePath += "\t";

paraverMain::dialogProgress->Pulse( wxString::FromUTF8( reducePath.c_str() ) );
paraverMain::dialogProgress->Fit();
paraverMain::dialogProgress->Show();

Output *output = Output::createOutput( (TOutput)saveDialog.GetFilterIndex() );
output->setMultipleFiles( false );

output->setObjectHierarchy( true );
output->setWindowTimeUnits( false );


output->dumpWindow( myWindow, fileName, progress );
delete output;

paraverMain::dialogProgress->Show( false );
delete paraverMain::dialogProgress;
paraverMain::dialogProgress = nullptr;
delete progress;
}

setEnableDestroyButton( true );
}


void gTimeline::saveCFG()
{
vector< Timeline * > timelines;
timelines.push_back( GetMyWindow() );

setEnableDestroyButton( false );

CFGS4DLinkedPropertiesManager dummyManager;
vector<CFGS4DLinkedPropertiesManager> dummyList;
dummyList.push_back( dummyManager );
paraverMain::myParaverMain->SaveConfigurationFile( (wxWindow *)this,
SaveOptions(),
timelines,
vector< Histogram * >(),
dummyList );

setEnableDestroyButton( true );
}


void gTimeline::setEnableDestroyParents( bool value )
{
gWindow::setEnableDestroyButton( value );

if ( myWindow->getParent( 0 ) != nullptr )
{
for( int i = 0; i < 2; ++i )
{
bool dummyFound;
gTimeline *tmpTimeline = getGTimelineFromWindow( getAllTracesTree()->GetRootItem(), myWindow->getParent( i ), dummyFound );
tmpTimeline->setEnableDestroyParents( value );
}
}
}


void gTimeline::setEnableDestroyButton( bool value )
{
if ( myWindow->getChild() != nullptr )
{
bool dummyFound;
gTimeline *tmpTimeline = getGTimelineFromWindow( getAllTracesTree()->GetRootItem(), myWindow->getChild(), dummyFound );
tmpTimeline->setEnableDestroyButton( value );
}
else
{
setEnableDestroyParents( value );
}
}


void gTimeline::OnTimerSize( wxTimerEvent& event )
{
if( myWindow->getReady() )
{
if ( !wxparaverApp::mainWindow->GetSomeWinIsRedraw() )
{
timerSize->Stop();
#ifdef _WIN32
wxparaverApp::mainWindow->SetSomeWinIsRedraw( true );
redraw();
wxparaverApp::mainWindow->SetSomeWinIsRedraw( false );
#else
if( splitChanged )
{
wxparaverApp::mainWindow->SetSomeWinIsRedraw( true );
redraw();
wxparaverApp::mainWindow->SetSomeWinIsRedraw( false );
}
else
myWindow->setRedraw( true );

Refresh();
}
else
{
timerSize->StartOnce( TIMER_SIZE_DURATION );
#endif
}
}
}


void gTimeline::OnTimerMotion( wxTimerEvent& event )
{
wxMemoryDC dc( bufferImage );
wxColour tmpColor;
wxString label;
wxBitmap refreshImage( autoredraw_refresh_xpm );
bool autoRedrawText = false;

if( motionEvent.GetX() < refreshImage.GetWidth() + 2 * drawBorder &&
motionEvent.GetY() < refreshImage.GetHeight() + 2 * drawBorder &&
enabledAutoRedrawIcon )
{
autoRedrawText = true;
label = "Autoredraw disabled";
}
else if( motionEvent.GetX() < objectAxisPos + 1 || motionEvent.GetX() > bufferImage.GetWidth() - drawBorder ||
motionEvent.GetY() < drawBorder || motionEvent.GetY() > timeAxisPos - 1 )
{
return;
}
else if( zooming || timing || wxGetApp().GetGlobalTiming() )
{
long beginX;
PRV_UINT32 precision = 0;
TTime timeStep = ( myWindow->getWindowEndTime() - myWindow->getWindowBeginTime() ) /
( dc.GetSize().GetWidth() - objectAxisPos - drawBorder );
TTime time;

if( zooming )
{
beginX = firstMotionEvent.GetX();
if( beginX < objectAxisPos )
beginX = 0;
else
beginX -= objectAxisPos;
time = ( timeStep * beginX ) + myWindow->getWindowBeginTime();
label = wxString::FromUTF8( LabelConstructor::timeLabel( myWindow->traceUnitsToWindowUnits( time ), myWindow->getTimeUnit(), precision ).c_str() );
label += wxT( " - " );
}
beginX = motionEvent.GetX();
beginX -= objectAxisPos;
time = ( timeStep * beginX ) + myWindow->getWindowBeginTime();
label += wxString::FromUTF8( LabelConstructor::timeLabel( myWindow->traceUnitsToWindowUnits( time ), myWindow->getTimeUnit(), precision ).c_str() );
}
else
{
TSemanticValue firstValue, secondValue;
Timeline *winToUse = myWindow;
if( myWindow->isPunctualColorSet() && myWindow->getPunctualColorWindow() != nullptr )
winToUse = myWindow->getPunctualColorWindow();

if( !myWindow->isFunctionLineColorSet() )
{
#ifdef __WXMAC__
wxImage tmpImage = bufferImage.ConvertToImage();
tmpColor = wxColour( tmpImage.GetRed( motionEvent.GetX(), motionEvent.GetY() ),
tmpImage.GetGreen( motionEvent.GetX(), motionEvent.GetY() ),
tmpImage.GetBlue( motionEvent.GetX(), motionEvent.GetY() ) );

#else  
dc.GetPixel( motionEvent.GetX(), motionEvent.GetY(), &tmpColor );
#endif

if( tmpColor == backgroundColour )
return;

rgb color = { (ParaverColor)tmpColor.Red(), (ParaverColor)tmpColor.Green(), (ParaverColor)tmpColor.Blue() };

if( winToUse->isCodeColorSet() )
{
string tmpString;
firstValue = *( semanticColorsToValue[ color ].begin() );
tmpString = LabelConstructor::semanticLabel( winToUse, firstValue, true, ParaverConfig::getInstance()->getTimelinePrecision(), false );
if( winToUse->getSemanticInfoType() == EVENTVALUE_TYPE )
LabelConstructor::transformToShort( tmpString );
label = wxString::FromUTF8( tmpString.c_str() );
}
else if( winToUse->isColorOutlier( color ) )
{
if( color == winToUse->getGradientColor().getAboveOutlierColor() )
label = wxT( "> " ) + wxString::FromUTF8( LabelConstructor::semanticLabel( winToUse, winToUse->getMaximumY(), false,
ParaverConfig::getInstance()->getTimelinePrecision(), false ).c_str() );
else if( color == winToUse->getGradientColor().getBelowOutlierColor() )
label = wxT( "< " ) + wxString::FromUTF8( LabelConstructor::semanticLabel( winToUse, winToUse->getMinimumY(), false,
ParaverConfig::getInstance()->getTimelinePrecision(), false ).c_str() );
else
return;
}
else
{
if( winToUse->isFusedLinesColorSet() )
{
string tmpString;
firstValue = *( semanticColorsToValue[ color ].begin() );
tmpString = LabelConstructor::objectLabel( (TObjectOrder)firstValue, winToUse->getLevel(), winToUse->getTrace() );
label = wxString::FromUTF8( tmpString.c_str() );
}
else
{
firstValue = *( semanticColorsToValue[ color ].begin() );
label = wxString::FromUTF8( LabelConstructor::semanticLabel( winToUse, firstValue, false,
ParaverConfig::getInstance()->getTimelinePrecision(), false ).c_str() );
if( semanticColorsToValue[ color ].size() > 1 )
{
secondValue = *( --( semanticColorsToValue[ color ].end() ) );
label += wxT( " - " ) + wxString::FromUTF8( LabelConstructor::semanticLabel( winToUse, secondValue, false,
ParaverConfig::getInstance()->getTimelinePrecision(), false ).c_str() );
}
}
}
}
else 
{
tmpColor = GetBackgroundColour();
TObjectOrder object;
TTime time;
if( pixelToTimeObject( motionEvent.GetX(), motionEvent.GetY(), time, object ) )
{
int pixelPos;
if( objectHeight > 10 && getPixelFromFunctionLine( motionEvent.GetX(), motionEvent.GetY(), object, pixelPos ) )
{
firstValue = *( semanticPixelsToValue[ pixelPos ].begin() );
label = wxString::FromUTF8( LabelConstructor::semanticLabel( winToUse, firstValue, false,
ParaverConfig::getInstance()->getTimelinePrecision(),
false ).c_str() );
if( semanticPixelsToValue[ pixelPos ].size() > 1 )
{
secondValue = *( --( semanticPixelsToValue[ pixelPos ].end() ) );
label += wxT( " - " ) + wxString::FromUTF8( LabelConstructor::semanticLabel( winToUse, secondValue, false,
ParaverConfig::getInstance()->getTimelinePrecision(),
false ).c_str() );
}
}
else
label = wxString( wxT( "" ) );
}
else
{
label = wxT( "ERROR: Zoom high!" );
}      
}
}

wxClientDC paintDC( drawZone );
paintDC.DrawBitmap( drawImage, 0, 0 );

paintDC.SetFont( semanticFont );
wxSize objectExt = paintDC.GetTextExtent( label );

paintDC.SetPen( backgroundColour );
paintDC.SetBrush( backgroundColour );

if( !( autoRedrawText || zooming || timing || wxGetApp().GetGlobalTiming() ) )
{
paintDC.SetBrush( tmpColor );
paintDC.DrawRectangle( ( bufferImage.GetWidth() - objectAxisPos ) / 2, timeAxisPos + 2, 10, bufferImage.GetHeight() - timeAxisPos - 3 );
}
paintDC.SetTextForeground( foregroundColour );
if( zooming )
paintDC.DrawText( label, ( bufferImage.GetWidth() - objectAxisPos ) / 2 + objectAxisPos - ( objectExt.GetWidth() / 2 ), timeAxisPos + 3 );
else
paintDC.DrawText( label, ( bufferImage.GetWidth() - objectAxisPos ) / 2 + 12, timeAxisPos + 3 );
}

void gTimeline::OnTimerWheel( wxTimerEvent& event )
{
#ifdef _WIN32
wheelZoomObjects = false;
#endif
wheelZoomFactor = 1.0;
myWindow->addZoom( wheelZoomBeginTime, wheelZoomEndTime, wheelZoomBeginObject, wheelZoomEndObject );
myWindow->setWindowBeginTime( wheelZoomBeginTime, true );
myWindow->setWindowEndTime( wheelZoomEndTime, true );
myWindow->setRedraw( true );
myWindow->setChanged( true );
}



void gTimeline::OnCheckWhatWhereText( wxCommandEvent& event )
{
checkWWSemantic->Enable( false );
checkWWEvents->Enable( false );
checkWWCommunications->Enable( false );
checkWWPreviousNext->Enable( false );
checkWWText->Enable( false );
checkWWShowDate->Enable( false );
checkWWHex->Enable( false );

computeWhatWhere( whatWhereTime, whatWhereRow, whatWhereSemantic,
checkWWText->IsChecked(),
checkWWShowDate->IsChecked(),
checkWWHex->IsChecked() );
printWhatWhere();

checkWWSemantic->Enable( true );
checkWWEvents->Enable( true );
checkWWCommunications->Enable( true );
checkWWPreviousNext->Enable( true );
checkWWText->Enable( true );
checkWWShowDate->Enable( true );
checkWWHex->Enable( true );
}

void gTimeline::OnCheckWhatWhere( wxCommandEvent& event )
{
checkWWSemantic->Enable( false );
checkWWEvents->Enable( false );
checkWWCommunications->Enable( false );
checkWWPreviousNext->Enable( false );
checkWWText->Enable( false );
checkWWShowDate->Enable( false );
checkWWHex->Enable( false );

printWhatWhere();

checkWWSemantic->Enable( true );
checkWWEvents->Enable( true );
checkWWCommunications->Enable( true );
checkWWPreviousNext->Enable( true );
checkWWText->Enable( true );
checkWWShowDate->Enable( true );
checkWWHex->Enable( true );
}

void gTimeline::OnMenuGradientFunction( TGradientFunction function )
{
myWindow->getGradientColor().setGradientFunction( function );
myWindow->setRedraw( true );
}




void gTimeline::OnScrolledWindowMiddleUp( wxMouseEvent& event )
{
if( myWindow->getLevel() != TTraceLevel::THREAD )
return;

wxMemoryDC dc( bufferImage );
long X = event.GetX();

if( X < objectAxisPos || X > dc.GetSize().GetWidth() - drawBorder
|| event.GetY() < drawBorder || event.GetY() > timeAxisPos )
return;
else
X -= objectAxisPos;

TTime timeStep = ( myWindow->getWindowEndTime() - myWindow->getWindowBeginTime() ) /
( dc.GetSize().GetWidth() - objectAxisPos - drawBorder );

TTime posTime = ( timeStep * X ) + myWindow->getWindowBeginTime();

TObjectOrder beginRow = myWindow->getZoomSecondDimension().first;
TObjectOrder endRow = myWindow->getZoomSecondDimension().second;
vector<TObjectOrder> selected;
myWindow->getSelectedRows( myWindow->getLevel(), selected, beginRow, endRow, true );
TObjectOrder numObjects = selected.size();
double heightPerRow = (double)( timeAxisPos - drawBorder - 1 ) / (double)numObjects;
TObjectOrder posRow = TObjectOrder( floor( (event.GetY() - drawBorder - 1) / heightPerRow ) );

if( posRow >= numObjects )
posRow = numObjects - 1;

posRow = selected[ posRow ];

TEventType type;
TEventValue value;
vector<TEventType> events;
wxString command;

wxString alienApp;
wxString alienType;
if ( wxGetEnv( wxString( wxT( "PARAVER_ALIEN_APP" ) ), &alienApp ) &&
wxGetEnv( wxString( wxT( "PARAVER_ALIEN_TYPE" ) ), &alienType ) )
{
long tmpType;
if (!alienType.ToLong(&tmpType))
{
wxMessageBox( _( "Unable to interpret $PARAVER_ALIEN_TYPE as an event type\n\nValue: " ) + alienType,
_( "Application invocation" ) );
return;
}
else
{
events.push_back( TEventType( tmpType ) );
if( myWindow->getTrace()->findLastEventValue( posRow, posTime, events, type, value ) )
{
string valueStr = LabelConstructor::eventValueLabel( myWindow, type, value );
if ( valueStr.empty() )
{
command << alienApp <<  _(" ") << value;
}
else
{
command << alienApp <<  _(" ") << wxString::FromUTF8( valueStr.c_str() );
}


std::string absolutePathCurrentTrace = GetMyWindow()->getTrace()->getFileName();
#ifdef _WIN32
_putenv_s( "PARAVER_ALIEN_TRACE_FULL_PATH", (const char *)absolutePathCurrentTrace.c_str() );
wxExecute( command );
#else
setenv( "PARAVER_ALIEN_TRACE_FULL_PATH", (const char *)absolutePathCurrentTrace.c_str(), 1 );
if( wxExecute( command ) == 0 )
{
}
#endif
}
else
{
}
}
}
else
{
events.push_back( wxGetApp().GetEventTypeForCode() );
if( myWindow->getTrace()->findLastEventValue( posRow, posTime, events, type, value ) )
{
string valueStr = LabelConstructor::eventValueLabel( myWindow, type, value );
string lineStr = valueStr.substr( 0, valueStr.find_first_of( ' ', 0 ) );
string fileStr = valueStr.substr( valueStr.find_first_of( '(', 0 ) + 1,
valueStr.length() - valueStr.find_first_of( '(', 0 ) - 2 );
wxDirDialog dirDialog( nullptr, _( "Choose the directory to find to source files" ) );
if( dirDialog.ShowModal() == wxID_OK )
{
wxString path = dirDialog.GetPath();
wxString command;

wxArrayString textEditor = paraverMain::FromVectorStringToWxArray( ParaverConfig::getInstance()->getGlobalExternalTextEditors(), "txt" );
bool cmdExecuted = false;

for (int idx = 0 ; !cmdExecuted && idx < textEditor.size(); ++idx)
{
#ifdef _WIN32
command << textEditor[ idx ] << path << _( "\\" ) << wxString::FromUTF8( fileStr.c_str() );
cmdExecuted = ( wxExecute( command + wxT( " --version" ), wxEXEC_SYNC ) == 0 );
#else
command << textEditor[ idx ] << _( " + " ) << wxString::FromUTF8( lineStr.c_str() ) << _( " " ) << path << _( "/" ) << wxString::FromUTF8( fileStr.c_str() );
cmdExecuted = ( wxExecute( command + wxT( " --version" ), wxEXEC_SYNC ) == 0 );
#endif
if ( !cmdExecuted )
command.Clear();
else 
cmdExecuted = ( wxExecute( command ) != 0 );
}
if ( !cmdExecuted )
wxMessageBox( _( "Unable to find an external app. Please check the external application's text editors list at Preferences." ), _( "No external app found" ) );
}
}
else
{
wxMessageBox( _( "Event " ) + (wxString() << wxGetApp().GetEventTypeForCode()) + _( " not found." ), _( "Show source code" ) );
}
}
}




void gTimeline::OnScrolledWindowKeyDown( wxKeyEvent& event )
{
wxCommandEvent dummyEvent;

if( event.ControlDown() && event.GetKeyCode() == (long) 'C' )
{
OnPopUpCopy( dummyEvent );
return;
}
if( event.ControlDown() && event.GetKeyCode() == (long) 'V' )
{
if( gPasteWindowProperties::getInstance()->isAllowed( this, STR_PASTE_SPECIAL ) )
OnPopUpPasteDefaultSpecial( dummyEvent );
return;
}
if( event.ControlDown() && event.GetKeyCode() == (long) 'U' )
{
OnPopUpUndoZoom( dummyEvent );
return;
}
if( event.ControlDown() && event.GetKeyCode() == (long) 'R' )
{
OnPopUpRedoZoom( dummyEvent );
return;
}
if( event.ControlDown() && event.GetKeyCode() == (long) 'T' )
{
EnableTiming( !timing );
return;
}

if( zooming && event.GetKeyCode() == WXK_ESCAPE )
{
escapePressed = true;
zooming = false;

wxMemoryDC memdc( drawImage );
memdc.SetBackgroundMode( wxTRANSPARENT );
memdc.SetBackground( *wxTRANSPARENT_BRUSH );
memdc.Clear();
#if wxTEST_GRAPHICS == 1
wxGCDC dc( memdc );
dc.SetBrush( wxBrush( wxColour( 255, 255, 255, 80 ) ) );
#else
wxDC& dc = memdc;
dc.SetBrush( *wxTRANSPARENT_BRUSH );
#endif
dc.DrawBitmap( bufferImage, 0, 0, false );
if( myWindow->getDrawFlags() )
dc.DrawBitmap( eventImage, 0, 0, true );
if( myWindow->getDrawCommLines() )
dc.DrawBitmap( commImage, 0, 0, true );
drawZone->Refresh();
}

event.Skip();
}

bool gTimeline::pixelToTimeObject( long x, long y, TTime& onTime, TObjectOrder& onObject )
{
if( x < objectAxisPos ||
x > bufferImage.GetWidth() - drawBorder )
return false;
else
x -= objectAxisPos;

if( y > timeAxisPos ||
y < drawBorder )
return false;

TTime timeStep = ( myWindow->getWindowEndTime() - myWindow->getWindowBeginTime() ) /
( bufferImage.GetWidth() - objectAxisPos - drawBorder );
onTime = ( timeStep * x ) + myWindow->getWindowBeginTime();

TObjectOrder beginRow = myWindow->getZoomSecondDimension().first;
TObjectOrder endRow = myWindow->getZoomSecondDimension().second;
vector<TObjectOrder> selected;
myWindow->getSelectedRows( myWindow->getLevel(), selected, beginRow, endRow, true );
TObjectOrder numObjects = selected.size();
double heightPerRow = (double)( timeAxisPos - drawBorder - 1 ) / (double)numObjects;
endRow = TObjectOrder( floor( ( y - drawBorder - 1 ) / heightPerRow ) );

if( endRow >= numObjects )
endRow = numObjects - 1;
onObject = selected[ endRow ];

return true;
}


bool gTimeline::getPixelFromFunctionLine( int whichX, int whichY, TObjectOrder whichObject, int& whichPixelPos )
{
int minPos = objectPosList[ whichObject ];
int maxPos = minPos + objectHeight;

wxColour pixelColor;
wxMemoryDC dc( bufferImage );

std::vector< wxColour > nearbyPixels;
int tmpY;
for ( int i = -1; i <= 1; ++i )
{
tmpY = whichY + i;
dc.GetPixel( whichX, tmpY, &pixelColor ); 
if ( tmpY > drawBorder && tmpY < timeAxisPos && pixelColor == GetForegroundColour() ) 
nearbyPixels.push_back( pixelColor );
else
nearbyPixels.push_back( GetBackgroundColour() );
}

if ( nearbyPixels[ 0 ] == nearbyPixels[ 1 ]  && 
nearbyPixels[ 1 ] == nearbyPixels[ 2 ] )
{
return false;
}

if ( nearbyPixels[ 0 ] == GetBackgroundColour() && 
nearbyPixels[ 1 ] == GetBackgroundColour() && 
nearbyPixels[ 2 ] == GetForegroundColour() ) 
{
whichPixelPos = objectHeight - ( whichY + 1 - minPos );
return true;
}

if ( nearbyPixels[ 0 ] == GetForegroundColour() && 
nearbyPixels[ 1 ] == GetBackgroundColour() && 
nearbyPixels[ 2 ] == GetBackgroundColour() ) 
{
whichPixelPos = objectHeight - ( whichY - 1 - minPos );
return true;
}

if ( nearbyPixels[ 1 ] == GetForegroundColour() ) 
{
whichPixelPos = objectHeight - ( whichY - minPos );
return true;
}

return false;
}


TSemanticValue gTimeline::getSemanticValueFromFusedLines( int whichY )
{
TSemanticValue tmpSemantic = 0.0;

if( myWindow->isFusedLinesColorSet() )
{
TSemanticValue semanticStep = ( myWindow->getMaximumY() - myWindow->getMinimumY() ) /
( timeAxisPos - drawBorder );
tmpSemantic = myWindow->getMaximumY() - ( semanticStep * ( whichY - drawBorder - 1 ) );
}

return tmpSemantic;
}




void gTimeline::OnScrolledWindowLeftDClick( wxMouseEvent& event )
{
TObjectOrder object;
TTime time;

if( !pixelToTimeObject( event.GetX(), event.GetY(), time, object ) )
return;

TSemanticValue tmpSemantic = getSemanticValueFromFusedLines( event.GetY() );

if( !splitter->IsSplit() )
{
Split();
}
whatWhereText->Clear();
whatWhereText->AppendText( _( "Working..." ) );
Update();
computeWhatWhere( time, object, tmpSemantic, checkWWText->IsChecked(), checkWWShowDate->IsChecked(), checkWWHex->IsChecked() );
printWhatWhere();
}


void gTimeline::drawTimeMarks( std::vector< TRecordTime > times,
vector< TObjectOrder > &selectedObjects,
bool drawXCross,
bool allObjects,
TObjectOrder lastFoundObject )
{
wxMemoryDC bufferDraw;

if( !myWindow->getReady() )
return;

if ( allObjects )
{
myWindow->getSelectedRows( myWindow->getLevel(), selectedObjects, true );
lastFoundObject = selectedObjects.size() -1;
}

bufferDraw.SelectObject( wxNullBitmap );
bufferDraw.SelectObject( drawImage );
bufferDraw.DrawBitmap( bufferImage, 0, 0, false );

doDrawCaution( bufferDraw );

if( myWindow->getDrawFlags() )
bufferDraw.DrawBitmap( eventImage, 0, 0, true );

if( myWindow->getDrawCommLines() )
bufferDraw.DrawBitmap( commImage, 0, 0, true );

for( std::vector< TRecordTime >::iterator it = times.begin(); it != times.end(); ++it )
{
if ( *it >= myWindow->getWindowBeginTime() && *it <= myWindow->getWindowEndTime() )
{

wxCoord xTime = ( ( ( *it - myWindow->getWindowBeginTime() ) * ( drawZone->GetSize().GetWidth() - objectAxisPos - drawBorder ) )
/ ( myWindow->getWindowEndTime() - myWindow->getWindowBeginTime() ) )
+ objectAxisPos;

if ( drawXCross )
{
bufferDraw.SetPen( wxPen( *wxRED, 2 ) );
bufferDraw.DrawLine( xTime - 5, objectPosList[ lastFoundObject ] - 5, xTime + 5, objectPosList[ lastFoundObject ] + 5 );
bufferDraw.DrawLine( xTime - 5, objectPosList[ lastFoundObject ] + 5, xTime + 5, objectPosList[ lastFoundObject ] - 5 );
}

bufferDraw.SetPen( wxPen( *wxRED, 2, wxPENSTYLE_SHORT_DASH ) );
bufferDraw.DrawLine( xTime + 1, 0, xTime + 1, drawZone->GetSize().GetHeight() );
}
}

drawZone->Refresh();
}


void gTimeline::OnFindDialog()
{
FindDialog dialog( this );
dialog.SetMyWindow( myWindow );
dialog.InitControlsBeforeShow();

if( dialog.ShowModal() == wxID_OK )
{
TRecordTime beginTime, endTime;
unsigned int objectSelection = dialog.choiceObjects->GetSelection();
vector<TObjectOrder> selectedObjects;
myWindow->getSelectedRows( myWindow->getLevel(), selectedObjects, true );

if( dialog.radioObjects->GetValue() )
{
beginTime = myWindow->getWindowBeginTime();
endTime = myWindow->getWindowEndTime();
}
else if( dialog.radioEvents->GetValue() )
{
bool found = false;
TRecordTime newFoundTime, timeToSearch;
set<TEventType> events = myWindow->getTrace()->getLoadedEvents();
set<TEventType>::iterator itEvt = events.begin();
int i = 0;
while( i < dialog.choiceEventType->GetSelection() )
{
++i; ++itEvt;
}
TEventType eventType = *itEvt;
if( lastEventFoundTime >= myWindow->getWindowBeginTime() &&
lastEventFoundTime <= myWindow->getWindowEndTime() )
timeToSearch = lastEventFoundTime;
else
timeToSearch = myWindow->getWindowBeginTime();

if( dialog.checkNextObject->GetValue() )
{
while( !found && objectSelection < selectedObjects.size() )
{
if( ( found = myWindow->getTrace()->findNextEvent( selectedObjects[ objectSelection ],
timeToSearch,
eventType,
newFoundTime ) ) )
break;
++objectSelection;
timeToSearch = 0.0;
}
}
else
{
found = myWindow->getTrace()->findNextEvent( selectedObjects[ objectSelection ],
timeToSearch,
eventType,
newFoundTime );
}

if( !found )
{
wxMessageBox( wxT( "Event '" ) + dialog.choiceEventType->GetStringSelection() + wxT( "' not found." ),
wxT( "Not found" ) );
return;
}

beginTime = newFoundTime - myWindow->getTrace()->getEndTime() * 0.001;
if( beginTime < 0.0 ) beginTime = 0.0;
endTime = newFoundTime + myWindow->getTrace()->getEndTime() * 0.001;

lastEventFoundTime = newFoundTime;
}
else if( dialog.radioSemantic->GetValue() )
{
bool found = false;
TRecordTime timeToSearch, duration;
TSemanticValue valueToSearch;

if( lastSemanticFoundTime >= myWindow->getWindowBeginTime() &&
lastSemanticFoundTime <= myWindow->getWindowEndTime() )
timeToSearch = lastSemanticFoundTime;
else
timeToSearch = myWindow->getWindowBeginTime();

if( dialog.comboSemanticValue->GetSelection() == wxNOT_FOUND )
{
wxString tmpStr = dialog.comboSemanticValue->GetValue();
tmpStr.ToDouble( &valueToSearch );
}
else
valueToSearch = dialog.comboSemanticValue->GetSelection() + myWindow->getMinimumY();

duration = dialog.spinSemanticDuration->GetValue();

myWindow->init( timeToSearch, NOCREATE, false );
myWindow->initRow( selectedObjects[ objectSelection ], timeToSearch, NOCREATE, false );
while( lastSemanticFoundTime == myWindow->getBeginTime( selectedObjects[ objectSelection ] ) )
myWindow->calcNext( selectedObjects[ objectSelection ] );

if( dialog.checkNextObject->GetValue() )
{
while( !found && objectSelection < selectedObjects.size() )
{
while( !found && myWindow->getBeginTime( selectedObjects[ objectSelection ] ) < myWindow->getWindowEndTime() )
{
if( myWindow->getValue( selectedObjects[ objectSelection ] ) == valueToSearch )
{
if( dialog.choiceDurationFunction->GetSelection() == 0 &&
myWindow->getEndTime( selectedObjects[ objectSelection ] ) - myWindow->getBeginTime( selectedObjects[ objectSelection ] )
> duration )
{
found = true;
break;
}
else if( dialog.choiceDurationFunction->GetSelection() == 1 &&
myWindow->getEndTime( selectedObjects[ objectSelection ] ) - myWindow->getBeginTime( selectedObjects[ objectSelection ] )
< duration )
{
found = true;
break;
}
}
myWindow->calcNext( selectedObjects[ objectSelection ] );
}
++objectSelection;
}
}
else
{
while( !found && myWindow->getBeginTime( selectedObjects[ objectSelection ] ) < myWindow->getWindowEndTime() )
{
if( myWindow->getValue( selectedObjects[ objectSelection ] ) == valueToSearch )
{
if( dialog.choiceDurationFunction->GetSelection() == 0 &&
myWindow->getEndTime( selectedObjects[ objectSelection ] ) - myWindow->getBeginTime( selectedObjects[ objectSelection ] )
> duration )
{
found = true;
break;
}
else if( dialog.choiceDurationFunction->GetSelection() == 1 &&
myWindow->getEndTime( selectedObjects[ objectSelection ] ) - myWindow->getBeginTime( selectedObjects[ objectSelection ] )
< duration )
{
found = true;
break;
}
}
myWindow->calcNext( selectedObjects[ objectSelection ] );
}
}

if( !found )
{
wxMessageBox( wxT( "Semantic '" ) + dialog.comboSemanticValue->GetValue() + wxT( "' not found." ),
wxT( "Not found" ) );
return;
}

TRecordTime addTime = ( myWindow->getEndTime( selectedObjects[ objectSelection ] ) 
- myWindow->getBeginTime( selectedObjects[ objectSelection ] ) )
* 0.1;
if( myWindow->getBeginTime( selectedObjects[ objectSelection ] ) - myWindow->getEndTime( selectedObjects[ objectSelection ] )
<
( myWindow->getWindowEndTime() - myWindow->getWindowBeginTime() ) * 0.25 )
{
beginTime = myWindow->getBeginTime( selectedObjects[ objectSelection ] ) - addTime;
if( beginTime < 0.0 ) beginTime = 0.0;
endTime = myWindow->getEndTime( selectedObjects[ objectSelection ] ) + addTime;
}
else
{
beginTime = myWindow->getBeginTime( selectedObjects[ objectSelection ] ) - addTime;
if( beginTime < 0.0 ) beginTime = 0.0;
endTime = beginTime + ( ( myWindow->getWindowEndTime() - myWindow->getWindowBeginTime() ) * 0.25 );
}

lastSemanticFoundTime = myWindow->getBeginTime( selectedObjects[ objectSelection ] );
}

TObjectOrder first, last;
unsigned int objectsToShow = floor( (double)( timeAxisPos - drawBorder ) / 20 );

if( objectsToShow >= selectedObjects.size() )
{
first = selectedObjects[ 0 ];
last = selectedObjects[ selectedObjects.size() - 1 ];
}
else if( dialog.choicePosition->GetSelection() == 0 ) 
{
first = selectedObjects[ objectSelection ];
if( objectSelection + objectsToShow >= selectedObjects.size() )
last = selectedObjects[ selectedObjects.size() - 1 ];
else
last = selectedObjects[ objectSelection + objectsToShow ];
}
else if( dialog.choicePosition->GetSelection() == 2 ) 
{
last = selectedObjects[ objectSelection ];
if( objectsToShow > objectSelection )
first = selectedObjects[ 0 ];
else
first = selectedObjects[ objectsToShow - objectSelection ];
}
else 
{ 
objectsToShow /= 2;
if( objectsToShow > objectSelection )
{
first = selectedObjects[ 0 ];
last = selectedObjects[ objectSelection + objectsToShow ];
}
else if( objectSelection + objectsToShow > selectedObjects.size() )
{
first = selectedObjects[ objectSelection - objectsToShow ];
last = selectedObjects[ selectedObjects.size() ];
}
else
{
first = selectedObjects[ objectSelection - objectsToShow ];
last = selectedObjects[ objectSelection + objectsToShow ];
}
}


findBeginTime   = beginTime;
findEndTime     = endTime;
findFirstObject = first;
findLastObject  = last;
lastFoundObject = objectSelection;

std::vector< TRecordTime > tmpTimes;
tmpTimes.push_back( lastSemanticFoundTime );
bool drawXCross = true;
bool allObjects = false;
drawTimeMarks( tmpTimes, selectedObjects, drawXCross, allObjects, objectSelection );
}
}


void progressFunctionTimeline( ProgressController *progress, void *callerWindow )
{
if( gTimeline::dialogProgress != nullptr )
{
gTimeline::dialogProgress->Refresh();
gTimeline::dialogProgress->Update();
}

int p;
if ( progress->getCurrentProgress() > progress->getEndLimit() )
p = MAX_PROGRESS_BAR_VALUE;
else 
p = (int)floor( ( progress->getCurrentProgress() * MAX_PROGRESS_BAR_VALUE ) / progress->getEndLimit() );

wxString newMessage;
if( progress->getMessageChanged() )
{
newMessage = wxString::FromUTF8( progress->getMessage().c_str() );
progress->clearMessageChanged();
}



if( gTimeline::dialogProgress != nullptr && !gTimeline::dialogProgress->Update( p, newMessage ) )
progress->setStop( true );
}




void gTimeline::OnStaticSlopeUpdate( wxUpdateUIEvent& event )
{
event.SetText( wxString( _( "Slope (by ")  ) + wxString( LABEL_TIMEUNIT[ myWindow->getTimeUnit() ].c_str(), wxConvUTF8 ) + wxString( _( ")" ) ) );
}




void gTimeline::OnScrolledWindowMouseWheel( wxMouseEvent& event )
{
double newWheelFactor = wheelZoomFactor;
TRecordTime newWheelZoomBeginTime;
TRecordTime newWheelZoomEndTime;
TObjectOrder newWheelZoomBeginObject;
TObjectOrder newWheelZoomEndObject;
bool zoomOut = event.GetWheelRotation() < 0;
bool zoomIn = !zoomOut;

if( !myWindow->getReady() || ParaverConfig::getInstance()->getDisableTimelineZoomMouseWheel() )
return;

if( zoomOut )
{

newWheelFactor -= 0.1;
if( newWheelFactor <= std::numeric_limits<double>::epsilon() )
return;
}
else
{
newWheelFactor += 0.1;
}

TRecordTime timeWidth = myWindow->getWindowEndTime() - myWindow->getWindowBeginTime();
if( zoomIn && timeWidth <= 10 )
return;
if( zoomOut && myWindow->getWindowBeginTime() == 0 && myWindow->getWindowEndTime() == myWindow->getTrace()->getEndTime() )
return;

#ifdef _WIN32
if( event.ControlDown() )
wheelZoomObjects = true;
#endif

double wheelZoomFactorX = newWheelFactor;
double wheelZoomFactorY = 1;
#ifdef _WIN32
if( wheelZoomObjects )
#else
if( event.ControlDown() )
#endif
wheelZoomFactorY = newWheelFactor;

wxCoord posX = event.GetX();
if( posX <= objectAxisPos )
posX = objectAxisPos + 1;
else if( posX >= drawZone->GetClientSize().GetWidth() - drawBorder )
posX = drawZone->GetClientSize().GetWidth() - drawBorder - 1;
wxCoord pixelsWidth = drawZone->GetClientSize().GetWidth() - objectAxisPos - 1 - drawBorder;
double ratioLeft = ( (double)( posX - objectAxisPos - 1 ) / (double)pixelsWidth );
double ratioRight = 1.0 - ratioLeft;
ratioLeft = ratioLeft * ( 1 - 1 / wheelZoomFactorX );
ratioRight = ratioRight * ( 1 - 1 / wheelZoomFactorX );

wxCoord pixelsHeight = timeAxisPos;
double ratioUp = (double)event.GetY() / (double)pixelsHeight;
double ratioDown = 1.0 - ratioUp;
ratioUp = ratioUp * ( 1 - 1 / wheelZoomFactorY );
ratioDown = ratioDown * ( 1 - 1 / wheelZoomFactorY );

#ifdef _WIN32
if( wheelZoomObjects )
#else
if( event.ControlDown() )
#endif
{
vector<TObjectOrder> selectedObjects;
myWindow->getSelectedRows( myWindow->getLevel(),
selectedObjects,
myWindow->getZoomSecondDimension().first,
myWindow->getZoomSecondDimension().second,
true );

TObjectOrder objectHeight = selectedObjects.size();
PRV_INT64 dummyAppliedAmount;
newWheelZoomBeginObject = myWindow->shiftFirst( myWindow->getZoomSecondDimension().first, (double)objectHeight * ratioUp, dummyAppliedAmount, myWindow->getLevel() );
newWheelZoomEndObject = myWindow->shiftLast( myWindow->getZoomSecondDimension().second, - (double)objectHeight * ratioDown, dummyAppliedAmount, myWindow->getLevel() );
}
else
{
newWheelZoomBeginObject = myWindow->getZoomSecondDimension().first;
newWheelZoomEndObject = myWindow->getZoomSecondDimension().second;
}

newWheelZoomBeginTime = myWindow->getWindowBeginTime() + timeWidth * ratioLeft;
newWheelZoomEndTime = myWindow->getWindowEndTime() -  timeWidth * ratioRight;

if( newWheelZoomEndTime - newWheelZoomBeginTime < 10 || 
newWheelZoomEndTime < newWheelZoomBeginTime )
return;

if( newWheelZoomBeginTime < 0 )
{
newWheelZoomEndTime -= newWheelZoomBeginTime;
newWheelZoomBeginTime = 0;
}
if( newWheelZoomEndTime > myWindow->getTrace()->getEndTime() )
{
newWheelZoomBeginTime -= newWheelZoomEndTime - myWindow->getTrace()->getEndTime();
newWheelZoomEndTime = myWindow->getTrace()->getEndTime();

if( newWheelZoomBeginTime < 0 )
newWheelZoomBeginTime = 0;
}

wheelZoomFactor = newWheelFactor;
wheelZoomBeginTime = newWheelZoomBeginTime;
wheelZoomEndTime = newWheelZoomEndTime;
wheelZoomBeginObject = newWheelZoomBeginObject;
wheelZoomEndObject = newWheelZoomEndObject;

wxBitmap tmpBMP;
tmpBMP.Create( pixelsWidth, pixelsHeight );
wxMemoryDC tmpDC( tmpBMP );
tmpDC.SetBrush( wxBrush( backgroundColour ) );
tmpDC.Clear();
#if defined __WXMAC__ || defined _WIN32
tmpDC.DrawRectangle( 0, 0, pixelsWidth, pixelsHeight );
#endif
tmpDC.SetUserScale( wheelZoomFactorX, wheelZoomFactorY );

wxCoord pixelBeginX = (double)pixelsWidth * ratioLeft;
wxCoord pixelBeginY = (double)pixelsHeight * ratioUp;

#ifdef __WXMAC__
wxBitmap tmpDrawImage( drawImage.GetWidth(), drawImage.GetHeight() );
wxMemoryDC srcDC( tmpDrawImage );
srcDC.SetBrush( wxBrush( backgroundColour ) );
srcDC.Clear();
drawStackedImages( srcDC );
#else
wxMemoryDC srcDC( drawImage );
#endif

tmpDC.Blit( 0,
0,
tmpDC.DeviceToLogicalX( pixelsWidth ),
tmpDC.DeviceToLogicalY( pixelsHeight ),
&srcDC,
objectAxisPos + 1 + pixelBeginX,
pixelBeginY );

if( newWheelFactor < 1.0 )
{
tmpDC.SetPen( wxPen( backgroundColour ) );
tmpDC.SetBrush( wxBrush( backgroundColour ) );
tmpDC.DrawRectangle( 0, 0, -pixelBeginX, tmpDC.DeviceToLogicalY( timeAxisPos - drawBorder + 1 ) );
#ifdef _WIN32
if( wheelZoomObjects )
#else
if( event.ControlDown() )
#endif
{
tmpDC.DrawRectangle( 0, -pixelBeginY + timeAxisPos, tmpDC.DeviceToLogicalX( tmpBMP.GetWidth() ), tmpBMP.GetHeight() );
}
}

tmpDC.SelectObject( wxNullBitmap );

wxClientDC dstDC( drawZone );
dstDC.SetPen( wxPen( backgroundColour ) );
dstDC.SetBrush( wxBrush( backgroundColour ) );
dstDC.DrawRectangle( objectAxisPos + 1, 0, drawZone->GetClientSize().GetWidth() - objectAxisPos - 1, timeAxisPos );
dstDC.DrawBitmap( tmpBMP, objectAxisPos + 1, 0 );

timerWheel->Start( 750, true );
}


void gTimeline::MousePanMotion()
{
wxCoord pixelsWidth = drawZone->GetClientSize().GetWidth() - objectAxisPos - drawBorder;
wxCoord pixelsHeight = timeAxisPos;

wxBitmap tmpBMP;
tmpBMP.Create( pixelsWidth, pixelsHeight );
wxMemoryDC tmpDC( tmpBMP );
tmpDC.SetBrush( wxBrush( backgroundColour ) );
tmpDC.Clear();
tmpDC.DrawRectangle( 0, 0, pixelsWidth, pixelsHeight );

#ifdef __WXMAC__
wxBitmap tmpDrawImage( drawImage.GetWidth(), drawImage.GetHeight() );
wxMemoryDC srcDC( tmpDrawImage );
srcDC.SetBrush( wxBrush( backgroundColour ) );
srcDC.Clear();
drawStackedImages( srcDC );
#else
wxMemoryDC srcDC( drawImage );
#endif

wxCoord dstX = zoomBeginX < motionEvent.GetX() ? motionEvent.GetX() - zoomBeginX : 0;
wxCoord srcX = zoomBeginX < motionEvent.GetX() ? 0 : zoomBeginX - motionEvent.GetX();
wxCoord dstY;
wxCoord srcY;
if( motionEvent.ControlDown() )
{
dstY = zoomBeginY < motionEvent.GetY() ? motionEvent.GetY() - zoomBeginY : 0;
srcY = zoomBeginY < motionEvent.GetY() ? 0 : zoomBeginY - motionEvent.GetY();
}
else
{
dstY = 0;
srcY = 0;
}

wxCoord tmpBlitHeight;
if( motionEvent.ControlDown() )
tmpBlitHeight = pixelsHeight - ( zoomBeginY < motionEvent.GetY() ? motionEvent.GetY() - zoomBeginY : zoomBeginY - motionEvent.GetY() );
else
tmpBlitHeight = pixelsHeight;

TRecordTime timeWidth = myWindow->getWindowEndTime() - myWindow->getWindowBeginTime();
TRecordTime panBeginTime = myWindow->getWindowBeginTime() + ( zoomBeginX * timeWidth ) / pixelsWidth;
TRecordTime panEndTime   = myWindow->getWindowBeginTime() + ( motionEvent.GetX() * timeWidth ) / pixelsWidth;
TRecordTime deltaTime = panEndTime - panBeginTime;
panBeginTime = myWindow->getWindowBeginTime() - deltaTime;
panEndTime   = myWindow->getWindowEndTime()   - deltaTime;
if( panBeginTime < 0 || panEndTime > myWindow->getTrace()->getEndTime() )
return;



tmpDC.Blit( dstX,
dstY,
pixelsWidth - ( zoomBeginX < motionEvent.GetX() ? motionEvent.GetX() - zoomBeginX : zoomBeginX - motionEvent.GetX() ),
tmpBlitHeight,
&srcDC,
srcX + objectAxisPos + 1,
srcY );

tmpDC.SelectObject( wxNullBitmap );
srcDC.SelectObject( wxNullBitmap );

wxClientDC dstDC( drawZone );
dstDC.DrawBitmap( tmpBMP, objectAxisPos + 1, 0 );
}


void gTimeline::MousePanLeftUp( wxMouseEvent& event )
{
TRecordTime  panBeginTime;
TRecordTime  panEndTime;
TObjectOrder panBeginObject;
TObjectOrder panEndObject;
wxCoord      pixelsWidth = drawZone->GetClientSize().GetWidth() - objectAxisPos - drawBorder;
wxCoord      pixelsHeight = timeAxisPos;

TRecordTime timeWidth = myWindow->getWindowEndTime() - myWindow->getWindowBeginTime();
panBeginTime = myWindow->getWindowBeginTime() + ( zoomBeginX * timeWidth ) / pixelsWidth;
panEndTime   = myWindow->getWindowBeginTime() + ( zoomEndX * timeWidth ) / pixelsWidth;
TRecordTime deltaTime = panEndTime - panBeginTime;

panBeginTime = myWindow->getWindowBeginTime() - deltaTime;
panEndTime   = myWindow->getWindowEndTime()   - deltaTime;

if( panBeginTime < 0 )
{
panBeginTime = 0;
panEndTime   = timeWidth;
}
if( panEndTime > myWindow->getTrace()->getEndTime() )
{
panEndTime   = myWindow->getTrace()->getEndTime();
panBeginTime = panEndTime - timeWidth;
}

if( event.ControlDown() )
{
vector<TObjectOrder> selectedObjects;
myWindow->getSelectedRows( myWindow->getLevel(),
selectedObjects,
myWindow->getZoomSecondDimension().first,
myWindow->getZoomSecondDimension().second,
true );

TObjectOrder objectHeight = selectedObjects.size();
PRV_INT64 tmpPanBeginObject = (PRV_INT64)myWindow->getZoomSecondDimension().first + (double)( zoomBeginY * objectHeight ) / pixelsHeight;
PRV_INT64 tmpPanEndObject   = (PRV_INT64)myWindow->getZoomSecondDimension().first + (double)( zoomEndY * objectHeight ) / pixelsHeight;
PRV_INT64 deltaObject = tmpPanEndObject - tmpPanBeginObject;

PRV_INT64 appliedDeltaObject;
if( deltaObject < 0 )
{
panEndObject   = myWindow->shiftLast( myWindow->getZoomSecondDimension().second, -deltaObject, appliedDeltaObject, myWindow->getLevel() );
panBeginObject = myWindow->shiftFirst( myWindow->getZoomSecondDimension().first, appliedDeltaObject, appliedDeltaObject, myWindow->getLevel() );
}
else
{
panBeginObject = myWindow->shiftFirst( myWindow->getZoomSecondDimension().first, -deltaObject, appliedDeltaObject, myWindow->getLevel() );
panEndObject   = myWindow->shiftLast( myWindow->getZoomSecondDimension().second, appliedDeltaObject, appliedDeltaObject, myWindow->getLevel() );
}
}
else
{
panBeginObject = myWindow->getZoomSecondDimension().first;
panEndObject = myWindow->getZoomSecondDimension().second;
}

if( panBeginTime   != myWindow->getWindowBeginTime() ||
panBeginObject != myWindow->getZoomSecondDimension().first )
{
myWindow->addZoom( panBeginTime, panEndTime, panBeginObject, panEndObject );
myWindow->setWindowBeginTime( panBeginTime, true );
myWindow->setWindowEndTime( panEndTime, true );
myWindow->setRedraw( true );
myWindow->setChanged( true );
}
else
{
wxClientDC tmpDC( drawZone );
#ifdef __WXMAC__
drawStackedImages( tmpDC );
#else
tmpDC.DrawBitmap( drawImage, 0, 0 );
#endif
}
}

void gTimeline::doDrawCaution( wxDC& whichDC )
{
if( drawCautionNegatives )
{
wxBitmap cautionImage( caution_yellow_xpm );
whichDC.SetPen( wxPen( backgroundColour ) );
whichDC.SetBrush( wxBrush( backgroundColour ) );
whichDC.DrawRectangle( 0, drawZone->GetClientSize().GetHeight() - cautionImage.GetHeight() - drawBorder - 2,
drawBorder + cautionImage.GetWidth() + 2, drawZone->GetClientSize().GetHeight() );
whichDC.DrawBitmap( cautionImage,
drawBorder,
drawZone->GetClientSize().GetHeight() - cautionImage.GetHeight() - drawBorder,
true );
}
else if( drawCaution )
{
wxBitmap cautionImage( caution_xpm );
whichDC.SetPen( wxPen( backgroundColour ) );
whichDC.SetBrush( wxBrush( backgroundColour ) );
whichDC.DrawRectangle( 0, drawZone->GetClientSize().GetHeight() - cautionImage.GetHeight() - drawBorder - 2,
drawBorder + cautionImage.GetWidth() + 2, drawZone->GetClientSize().GetHeight() );
whichDC.DrawBitmap( cautionImage,
drawBorder,
drawZone->GetClientSize().GetHeight() - cautionImage.GetHeight() - drawBorder,
true );
}
}




void gTimeline::OnSliderSelectedColorUpdated( wxCommandEvent& event )
{
int redColor = sliderSelectedRed->GetValue();
int greenColor = sliderSelectedGreen->GetValue();
int blueColor = sliderSelectedBlue->GetValue();

wxColor tmpColor( redColor, greenColor, blueColor );
panelSelectedColor->SetBackgroundColour( tmpColor );
panelSelectedColor->Refresh();
selectedItemColor->SetBackgroundColour( tmpColor );
selectedItemColor->Refresh();

textSelectedRed->ChangeValue( wxString::Format( wxT( "%i" ), redColor ) );
textSelectedGreen->ChangeValue( wxString::Format( wxT( "%i" ), greenColor ) );
textSelectedBlue->ChangeValue( wxString::Format( wxT( "%i" ), blueColor ) );

rgb tmpRGBColor;
tmpRGBColor.red = redColor;
tmpRGBColor.green = greenColor;
tmpRGBColor.blue = blueColor;
myWindow->getCodeColor().setCustomColor( selectedCustomValue, tmpRGBColor );

enableApplyButton = true;
}




void gTimeline::OnCheckboxCustomPaletteUpdate( wxUpdateUIEvent& event )
{
event.Enable( myWindow->isCodeColorSet() );
event.Check( myWindow->getUseCustomPalette() );
}




void gTimeline::OnStaticSelectedColorUpdate( wxUpdateUIEvent& event )
{
event.Enable( myWindow->isCodeColorSet() && checkboxCustomPalette->IsChecked() && selectedItemColor != nullptr );
}




void gTimeline::OnSliderSelectedColorUpdateUI( wxUpdateUIEvent& event )
{
event.Enable( myWindow->isCodeColorSet() && checkboxCustomPalette->IsChecked() && selectedItemColor != nullptr );
}




void gTimeline::OnCheckboxCustomPaletteClick( wxCommandEvent& event )
{
myWindow->setUseCustomPalette( event.IsChecked() );
if ( !event.IsChecked() )
{
enableApplyButton = false;
selectedItemColor = nullptr;
sliderSelectedRed->SetValue( 0 );
sliderSelectedGreen->SetValue( 0 );
sliderSelectedBlue->SetValue( 0 );
panelSelectedColor->SetBackgroundColour( wxSystemSettings::GetColour( wxSYS_COLOUR_WINDOW ) );
}

if( myWindow->getCodeColor().existCustomColors() )
myWindow->setForceRedraw( true );

forceRedoColors = true;
}




void gTimeline::OnButtonCustomPaletteApplyClick( wxCommandEvent& event )
{
enableApplyButton = false;

myWindow->setForceRedraw( true );
}




void gTimeline::OnButtonCustomPaletteApplyUpdate( wxUpdateUIEvent& event )
{
event.Enable( enableApplyButton );
}




void gTimeline::OnTextSelectedColorUpdate( wxUpdateUIEvent& event )
{
event.Enable( myWindow->isCodeColorSet() && checkboxCustomPalette->IsChecked() && selectedItemColor != nullptr );
}




void gTimeline::OnTextSelectedColorUpdated( wxCommandEvent& event )
{
unsigned long redColor;
unsigned long greenColor;
unsigned long blueColor;

textSelectedRed->GetValue().ToULong( &redColor );
textSelectedGreen->GetValue().ToULong( &greenColor );
textSelectedBlue->GetValue().ToULong( &blueColor );

wxColor tmpColor( redColor, greenColor, blueColor );
panelSelectedColor->SetBackgroundColour( tmpColor );
panelSelectedColor->Refresh();
selectedItemColor->SetBackgroundColour( tmpColor );
selectedItemColor->Refresh();

sliderSelectedRed->SetValue( redColor );
sliderSelectedGreen->SetValue( greenColor );
sliderSelectedBlue->SetValue( blueColor );

rgb tmpRGBColor;
tmpRGBColor.red = redColor;
tmpRGBColor.green = greenColor;
tmpRGBColor.blue = blueColor;
myWindow->getCodeColor().setCustomColor( selectedCustomValue, tmpRGBColor );

enableApplyButton = true;
}




void gTimeline::OnCheckWWShowDateUpdate( wxUpdateUIEvent& event )
{
event.Enable( !myWindow->getTrace()->getTraceTime().is_not_a_date_time() );
}

