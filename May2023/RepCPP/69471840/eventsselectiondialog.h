


#pragma once





#include <vector>

#include "paraverkerneltypes.h"
#include "gtimeline.h"

#include "wx/statline.h"
#include "wx/tglbtn.h"

#include <wx/regex.h>
#include <wx/checkbox.h>
#include <wx/valtext.h> 
#include <wx/stattext.h>




class wxBoxSizer;
class wxToggleButton;

class gTimeline;
class Filter;



#define ID_EVENTSSELECTIONDIALOG 10053
#define ID_STATIC_TEXT_FUNCTION_TYPES 10170
#define ID_CHOICE_OPERATOR_FUNCTION_TYPES 10054
#define ID_CHECKBOX_SET_ALL_TYPES 10007
#define ID_TEXTCTRL_TYPES_REGEX_SEARCH 10006
#define ID_CHECKLISTBOX_TYPES 10161
#define ID_BUTTON_SET_ALL_TYPES 10163
#define ID_BUTTON_UNSET_ALL_TYPES 10164
#define ID_CHOICE_OPERATOR_TYPE_VALUE 10055
#define ID_CHOICE_OPERATOR_FUNCTION_VALUES 10056
#define ID_CHECKBOX_SET_ALL_VALUES 10008
#define ID_TEXTCTRL_VALUES_REGEX_SEARCH 10009
#define ID_CHECKLISTBOX_VALUES 10162
#define ID_TEXTCTRL_ADD_VALUES 10168
#define ID_BUTTON_ADD_VALUES 10169
#define ID_TOGGLEBUTTON_SHORT_LABELS 10000
#define ID_BUTTON_SET_ALL_VALUES 10165
#define ID_BUTTON_UNSET_ALL_VALUES 10166
#define SYMBOL_EVENTSSELECTIONDIALOG_STYLE wxCAPTION|wxRESIZE_BORDER|wxSYSTEM_MENU|wxCLOSE_BOX|wxTAB_TRAVERSAL
#define SYMBOL_EVENTSSELECTIONDIALOG_TITLE _("Events Selection")
#define SYMBOL_EVENTSSELECTIONDIALOG_IDNAME ID_EVENTSSELECTIONDIALOG
#define SYMBOL_EVENTSSELECTIONDIALOG_SIZE wxSize(400, 300)
#define SYMBOL_EVENTSSELECTIONDIALOG_POSITION wxDefaultPosition


class EventInfoManager
{
public:
EventInfoManager( Timeline *whichWindow, Filter *whichFilter );
virtual ~EventInfoManager();

virtual void transferFrom( wxCheckListBox *whichList ) = 0;

virtual bool isEmpty() const = 0;

virtual void setAllVisible() = 0;
int getFirstPosSelectedVisible() { return firstPosSelectedVisible; };

virtual void getSelectedFromVisible( wxArrayString& whichVisible,
wxArrayInt &whichPosVisible,
wxArrayInt &whichGlobalSelection,
wxArrayInt &whichGUISelection,
int &whichFirstPosSelectedVisible,
bool updateFirstPosSelectedVisible = true ) = 0;

bool getChangedSelected() { return changedSelection ; };

bool add( wxString whichRegEx );
void clearAllRegEx();

protected:
Timeline *currentWindow;
Filter *currentFilter;

int firstPosSelectedVisible;
bool changedSelection;

std::vector< wxRegEx * > filterRegEx;

virtual void setChangedSelection() = 0;
bool matchesAllRegex( std::string whichName, std::string whichValue );
};


class EventTypesInfoManager : public EventInfoManager
{
public:
EventTypesInfoManager( Timeline *whichWindow, Filter *whichFilter );
virtual ~EventTypesInfoManager() {};

void init();
virtual void transferFrom( wxCheckListBox *whichList ) {}

bool isEmpty() const { return fullList.empty(); }
TEventType getCurrent() { return currentType ; }
void setCurrent( TEventType whichType ) { currentType = whichType ; }

virtual void setAllVisible();
void setVisible( wxArrayInt whichVisible ) { visible = whichVisible ; }

TEventType getVisible( int pos ) { return fullList[ visible[ pos ] ] ; }
unsigned int countVisible() { return visible.GetCount() ; }
TEventType getFirstTypeVisible() { return fullList[ visible[ 0 ] ] ; }
void updateVisible();

void setSelected( int pos, bool isChecked );
void setSelected( TEventType whichSelected, bool isChecked );
void setAllSelected();
void setAllUnselected();

wxArrayInt getSelected();
virtual void getSelectedFromVisible( wxArrayString& whichVisible,
wxArrayInt &whichPosVisible,
wxArrayInt &whichGlobalSelection,
wxArrayInt &whichGUISelection,
int &whichFirstPosSelectedVisible,
bool updateFirstPosSelectedVisible = true );
protected:
virtual void setChangedSelection();

private:
std::vector< TEventType > fullList; 
wxArrayString             labels;   
wxArrayInt                selected; 
wxArrayInt                visible;  

TEventType currentType;
wxArrayInt initialSelected;
};


class EventValuesInfoManager : public EventInfoManager
{
public:
EventValuesInfoManager( Timeline *whichWindow, Filter *whichFilter, TEventType whichType );
virtual ~EventValuesInfoManager() {};

void init( TEventType whichType, bool shortVersion, bool keepSelected = false );
virtual void transferFrom( wxCheckListBox *whichList );
bool isEmpty() const { return fullList.IsEmpty(); }
bool insert( double whichValue, wxString whichLabel ); 

virtual void setAllVisible();
unsigned int countVisible() { return visible.GetCount() ; }
void updateVisible();
void setVisible( wxArrayInt whichVisible );

void setAllSelected();
void setAllUnselected();

wxArrayDouble getSelected();
virtual void getSelectedFromVisible( wxArrayString& whichVisible,
wxArrayInt &whichPosVisible,
wxArrayInt &whichGlobalSelection,
wxArrayInt &whichGUISelection,
int &whichFirstPosSelectedVisible,
bool updateFirstPosSelectedVisible = true );
protected:
virtual void setChangedSelection();

private:
TEventType currentType;

wxArrayDouble fullList;    
wxArrayString labels; 
wxArrayDouble selected; 
wxArrayDouble visible; 

wxArrayDouble addedFullList;
wxArrayDouble initialSelected;
};




class EventsSelectionDialog: public wxDialog
{    
DECLARE_DYNAMIC_CLASS( EventsSelectionDialog )
DECLARE_EVENT_TABLE()

public:
EventsSelectionDialog();

EventsSelectionDialog( wxWindow* parent,
Timeline *whichWindow,
bool hideOperatorsList = false,
const wxString& caption = SYMBOL_EVENTSSELECTIONDIALOG_TITLE,
wxWindowID id = SYMBOL_EVENTSSELECTIONDIALOG_IDNAME,
const wxPoint& pos = SYMBOL_EVENTSSELECTIONDIALOG_POSITION,
const wxSize& size = SYMBOL_EVENTSSELECTIONDIALOG_SIZE,
long style = SYMBOL_EVENTSSELECTIONDIALOG_STYLE );

EventsSelectionDialog( wxWindow* parent,
std::vector<TEventType> types,
std::vector<TEventValue> values,
bool hideOperatorsList = false,
wxWindowID id = SYMBOL_EVENTSSELECTIONDIALOG_IDNAME,
const wxString& caption = SYMBOL_EVENTSSELECTIONDIALOG_TITLE,
const wxPoint& pos = SYMBOL_EVENTSSELECTIONDIALOG_POSITION,
const wxSize& size = SYMBOL_EVENTSSELECTIONDIALOG_SIZE,
long style = SYMBOL_EVENTSSELECTIONDIALOG_STYLE );
EventsSelectionDialog( wxWindow* parent,
wxArrayString &whichTypes,
wxArrayInt &whichSelectedEventTypes,
bool whichHideOperatorsList,
const wxString& caption,
wxWindowID id,
const wxPoint& pos,
const wxSize& size,
long style);

bool Create( wxWindow* parent, wxWindowID id = SYMBOL_EVENTSSELECTIONDIALOG_IDNAME, const wxString& caption = SYMBOL_EVENTSSELECTIONDIALOG_TITLE, const wxPoint& pos = SYMBOL_EVENTSSELECTIONDIALOG_POSITION, const wxSize& size = SYMBOL_EVENTSSELECTIONDIALOG_SIZE, long style = SYMBOL_EVENTSSELECTIONDIALOG_STYLE );

~EventsSelectionDialog();

void Init();

void CreateControls();


void OnIdle( wxIdleEvent& event );

void OnChoiceOperatorFunctionTypesSelected( wxCommandEvent& event );

void OnCheckboxSetAllTypesClick( wxCommandEvent& event );

void OnCheckboxSetAllTypesUpdate( wxUpdateUIEvent& event );

void OnTextctrlTypesRegexSearchTextUpdated( wxCommandEvent& event );

void OnChecklistboxTypesDoubleClicked( wxCommandEvent& event );

void OnChecklistboxTypesSelected( wxCommandEvent& event );

void OnChecklistboxTypesToggled( wxCommandEvent& event );

void OnButtonSetAllTypesClick( wxCommandEvent& event );

void OnButtonUnsetAllTypesClick( wxCommandEvent& event );

void OnChoiceOperatorTypeValueSelected( wxCommandEvent& event );

void OnChoiceOperatorFunctionValuesSelected( wxCommandEvent& event );

void OnCheckboxSetAllValuesClick( wxCommandEvent& event );

void OnCheckboxSetAllValuesUpdate( wxUpdateUIEvent& event );

void OnTextctrlValuesRegexSearchTextUpdated( wxCommandEvent& event );

void OnChecklistboxValuesDoubleClicked( wxCommandEvent& event );

void OnChecklistboxValuesToggled( wxCommandEvent& event );

void OnTextCtrlKeyDown( wxKeyEvent& event );

void OnButtonAddValuesClick( wxCommandEvent& event );

void OnTogglebuttonShortLabelsClick( wxCommandEvent& event );

void OnButtonSetAllValuesClick( wxCommandEvent& event );

void OnButtonUnsetAllValuesClick( wxCommandEvent& event );

void OnApplyClick( wxCommandEvent& event );

void OnApplyUpdate( wxUpdateUIEvent& event );



wxBitmap GetBitmapResource( const wxString& name );

wxIcon GetIconResource( const wxString& name );

static bool ShowToolTips();

int GetIndexEventTypesFunction() const;
std::string GetNameEventTypesFunction() const;
wxArrayInt GetEventTypesSelection() const;

int GetIndexOperatorTypeValue() const;
std::string GetNameOperatorTypeValue() const;

int GetIndexEventValuesFunction() const;
std::string GetNameEventValuesFunction() const;
wxArrayDouble GetEventValues() const;

bool ChangedEventTypesFunction() const;
bool ChangedEventTypesSelection() const;
bool ChangedOperatorTypeValue() const ;
bool ChangedEventValuesFunction() const;
bool ChangedEventValuesSelection() const;

private:

wxBoxSizer* boxSizerFunctionTypes;
wxStaticText* staticTextFunctionTypes;
wxChoice* choiceOperatorFunctionTypes;
wxCheckBox* checkboxSetAllTypes;
wxTextCtrl* typesRegexSearch;
wxCheckListBox* checkListSelectTypes;
wxButton* buttonSetAllTypes;
wxButton* buttonUnsetAllTypes;
wxChoice* choiceOperatorTypeValue;
wxBoxSizer* boxSizerFunctionValues;
wxStaticText* staticTextFunctionValues;
wxChoice* choiceOperatorFunctionValues;
wxCheckBox* checkboxSetAllValues;
wxTextCtrl* valuesRegexSearch;
wxCheckListBox* checkListSelectValues;
wxTextCtrl* textCtrlAddValues;
wxButton* buttonAddValues;
wxToggleButton* buttonShortLabels;
wxButton* buttonSetAllValues;
wxButton* buttonUnsetAllValues;
wxButton* applyButton;

bool hideOperatorsList; 

Timeline               *currentWindow;
Filter               *currentFilter;     

EventValuesInfoManager *valuesHandler;
EventTypesInfoManager  *typesHandler;

int                  previousEventTypesFunction;
bool                 changedEventTypesFunction;

TEventType           currentType;
bool                 changedEventTypesSelection;

int                  previousOperatorTypeValue;
bool                 changedOperatorTypeValue;

int                  previousEventValuesFunction;
bool                 changedEventValuesFunction;

bool                 changedEventValues;

void UpdateWidgetChecklistboxTypes();

void checkAll( wxCheckListBox *boxlist, bool value );
void GetEventValueLabels( wxArrayString & whichEventValues );
void UpdateWidgetChecklistboxValues();
void UpdateChecklistboxValues( TEventType type, bool keepSelected = true );

bool HasChanged( wxChoice *choice, int selectedFunction ) const;
bool HasChanged( wxCheckListBox *checkList, wxArrayInt &index ) const;

bool HasChanged( wxCheckListBox *checkList, EventTypesInfoManager *manager ) const;
bool HasChanged( wxCheckListBox *checkList, EventValuesInfoManager *manager ) const;

bool HasChanged( wxCheckListBox *checkList, wxArrayDouble &index ) const;
bool HasChanged( wxArrayInt &arr1, wxArrayInt &arr2 ) const;
bool HasChanged( wxArrayDouble &arr1, wxArrayDouble &arr2 ) const;

bool CopyChanges( wxChoice *choice, int &selectedFunction );
bool CopyChanges( wxCheckListBox *checkList,
wxArrayInt &index,
wxArrayString &selected,
bool copyStrings = false );
void InsertValueFromTextCtrl();

void TransferDataToWindowPreCreateControls( Timeline *whichWindow,
bool whichHideOperatorsList);
void EnableApplyButton();
void TransferDataToWindowPostCreateControls();

void TransferWindowToData();
unsigned int GetSelections( wxCheckListBox *checkList, wxArrayInt &index ) const;
};
