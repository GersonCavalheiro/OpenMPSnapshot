


#pragma once





#include "wx/spinctrl.h"
#include "wx/statline.h"

#include "window.h"



class wxSpinCtrl;
class wxBoxSizer;



#define ID_FINDDIALOG 10171
#define ID_RADIOOBJECTS 10178
#define ID_CHOICEOBJECT 10172
#define ID_CHOICEPOSITION 10173
#define ID_CHECKNEXTOBJECT 10181
#define ID_RADIOEVENTS 10174
#define ID_STATICTYPE 10179
#define ID_CHOICEEVENTS 10175
#define ID_RADIOSEMANTIC 10176
#define ID_STATICSEMANTICVALUE 10180
#define ID_COMBOSEMANTICVALUE 10177
#define ID_STATICSEMANTICDURATION 10184
#define ID_CHOICEDURATIONFUNCTION 10182
#define ID_TEXTSEMANTICDURATION 10183
#define SYMBOL_FINDDIALOG_STYLE wxCAPTION|wxRESIZE_BORDER|wxSYSTEM_MENU|wxCLOSE_BOX|wxTAB_TRAVERSAL
#define SYMBOL_FINDDIALOG_TITLE _("Find")
#define SYMBOL_FINDDIALOG_IDNAME ID_FINDDIALOG
#define SYMBOL_FINDDIALOG_SIZE wxSize(400, 300)
#define SYMBOL_FINDDIALOG_POSITION wxDefaultPosition




class FindDialog: public wxDialog
{    
DECLARE_DYNAMIC_CLASS( FindDialog )
DECLARE_EVENT_TABLE()

public:
FindDialog();
FindDialog( wxWindow* parent, wxWindowID id = SYMBOL_FINDDIALOG_IDNAME, const wxString& caption = SYMBOL_FINDDIALOG_TITLE, const wxPoint& pos = SYMBOL_FINDDIALOG_POSITION, const wxSize& size = SYMBOL_FINDDIALOG_SIZE, long style = SYMBOL_FINDDIALOG_STYLE );

bool Create( wxWindow* parent, wxWindowID id = SYMBOL_FINDDIALOG_IDNAME, const wxString& caption = SYMBOL_FINDDIALOG_TITLE, const wxPoint& pos = SYMBOL_FINDDIALOG_POSITION, const wxSize& size = SYMBOL_FINDDIALOG_SIZE, long style = SYMBOL_FINDDIALOG_STYLE );

~FindDialog();

void Init();

void CreateControls();


void OnChecknextobjectUpdate( wxUpdateUIEvent& event );

void OnStatictypeUpdate( wxUpdateUIEvent& event );

void OnChoiceeventsUpdate( wxUpdateUIEvent& event );

void OnStaticsemanticvalueUpdate( wxUpdateUIEvent& event );

void OnCombosemanticvalueUpdate( wxUpdateUIEvent& event );

void OnStaticsemanticdurationUpdate( wxUpdateUIEvent& event );

void OnChoicedurationfunctionUpdate( wxUpdateUIEvent& event );

void OnTextsemanticdurationUpdate( wxUpdateUIEvent& event );



Timeline * GetMyWindow() const { return myWindow ; }
void SetMyWindow(Timeline * value) { myWindow = value ; }

wxBitmap GetBitmapResource( const wxString& name );

wxIcon GetIconResource( const wxString& name );

static bool ShowToolTips();

void InitControlsBeforeShow();

wxRadioButton* radioObjects;
wxListBox* choiceObjects;
wxChoice* choicePosition;
wxCheckBox* checkNextObject;
wxRadioButton* radioEvents;
wxChoice* choiceEventType;
wxRadioButton* radioSemantic;
wxComboBox* comboSemanticValue;
wxChoice* choiceDurationFunction;
wxSpinCtrl* spinSemanticDuration;
wxBoxSizer* boxSizerOperatorsChoice;
private:
Timeline * myWindow;
};
