

#include "wx/wxprec.h"

#ifdef __BORLANDC__
#pragma hdrstop
#endif

#ifndef WX_PRECOMP
#include "wx/wx.h"
#endif

#include <iostream>


#include "filter.h"

#include "eventsselectiondialog.h"
#include "labelconstructor.h"

using namespace std;





IMPLEMENT_DYNAMIC_CLASS( EventsSelectionDialog, wxDialog )




BEGIN_EVENT_TABLE( EventsSelectionDialog, wxDialog )

EVT_IDLE( EventsSelectionDialog::OnIdle )
EVT_CHOICE( ID_CHOICE_OPERATOR_FUNCTION_TYPES, EventsSelectionDialog::OnChoiceOperatorFunctionTypesSelected )
EVT_CHECKBOX( ID_CHECKBOX_SET_ALL_TYPES, EventsSelectionDialog::OnCheckboxSetAllTypesClick )
EVT_UPDATE_UI( ID_CHECKBOX_SET_ALL_TYPES, EventsSelectionDialog::OnCheckboxSetAllTypesUpdate )
EVT_TEXT( ID_TEXTCTRL_TYPES_REGEX_SEARCH, EventsSelectionDialog::OnTextctrlTypesRegexSearchTextUpdated )
EVT_LISTBOX_DCLICK( ID_CHECKLISTBOX_TYPES, EventsSelectionDialog::OnChecklistboxTypesDoubleClicked )
EVT_LISTBOX( ID_CHECKLISTBOX_TYPES, EventsSelectionDialog::OnChecklistboxTypesSelected )
EVT_CHECKLISTBOX( ID_CHECKLISTBOX_TYPES, EventsSelectionDialog::OnChecklistboxTypesToggled )
EVT_BUTTON( ID_BUTTON_SET_ALL_TYPES, EventsSelectionDialog::OnButtonSetAllTypesClick )
EVT_BUTTON( ID_BUTTON_UNSET_ALL_TYPES, EventsSelectionDialog::OnButtonUnsetAllTypesClick )
EVT_CHOICE( ID_CHOICE_OPERATOR_TYPE_VALUE, EventsSelectionDialog::OnChoiceOperatorTypeValueSelected )
EVT_CHOICE( ID_CHOICE_OPERATOR_FUNCTION_VALUES, EventsSelectionDialog::OnChoiceOperatorFunctionValuesSelected )
EVT_CHECKBOX( ID_CHECKBOX_SET_ALL_VALUES, EventsSelectionDialog::OnCheckboxSetAllValuesClick )
EVT_UPDATE_UI( ID_CHECKBOX_SET_ALL_VALUES, EventsSelectionDialog::OnCheckboxSetAllValuesUpdate )
EVT_TEXT( ID_TEXTCTRL_VALUES_REGEX_SEARCH, EventsSelectionDialog::OnTextctrlValuesRegexSearchTextUpdated )
EVT_LISTBOX_DCLICK( ID_CHECKLISTBOX_VALUES, EventsSelectionDialog::OnChecklistboxValuesDoubleClicked )
EVT_CHECKLISTBOX( ID_CHECKLISTBOX_VALUES, EventsSelectionDialog::OnChecklistboxValuesToggled )
EVT_BUTTON( ID_BUTTON_ADD_VALUES, EventsSelectionDialog::OnButtonAddValuesClick )
EVT_TOGGLEBUTTON( ID_TOGGLEBUTTON_SHORT_LABELS, EventsSelectionDialog::OnTogglebuttonShortLabelsClick )
EVT_BUTTON( ID_BUTTON_SET_ALL_VALUES, EventsSelectionDialog::OnButtonSetAllValuesClick )
EVT_BUTTON( ID_BUTTON_UNSET_ALL_VALUES, EventsSelectionDialog::OnButtonUnsetAllValuesClick )
EVT_BUTTON( wxID_APPLY, EventsSelectionDialog::OnApplyClick )
EVT_UPDATE_UI( wxID_APPLY, EventsSelectionDialog::OnApplyUpdate )

END_EVENT_TABLE()


static int wxCMPFUNC_CONV compare_int(int *a, int *b)
{
if ( *a > *b )
return 1;
else if ( *a < *b )
return -1;
else
return 0;
}


static int wxCMPFUNC_CONV compare_double(double *a, double *b)
{
if ( *a > *b )
return 1;
else if ( *a < *b )
return -1;
else
return 0;
}



EventsSelectionDialog::EventsSelectionDialog()
{
Init();
}



void EventsSelectionDialog::Init()
{
boxSizerFunctionTypes = nullptr;
staticTextFunctionTypes = nullptr;
choiceOperatorFunctionTypes = nullptr;
checkboxSetAllTypes = nullptr;
typesRegexSearch = nullptr;
checkListSelectTypes = nullptr;
buttonSetAllTypes = nullptr;
buttonUnsetAllTypes = nullptr;
choiceOperatorTypeValue = nullptr;
boxSizerFunctionValues = nullptr;
staticTextFunctionValues = nullptr;
choiceOperatorFunctionValues = nullptr;
checkboxSetAllValues = nullptr;
valuesRegexSearch = nullptr;
checkListSelectValues = nullptr;
textCtrlAddValues = nullptr;
buttonAddValues = nullptr;
buttonShortLabels = nullptr;
buttonSetAllValues = nullptr;
buttonUnsetAllValues = nullptr;
applyButton = nullptr;

hideOperatorsList = false;

currentWindow = nullptr;
currentFilter = nullptr;

changedEventTypesFunction = false;
previousEventTypesFunction = 0;

changedEventTypesSelection = false;
currentType = TEventType( 0 );

changedOperatorTypeValue = false;
previousOperatorTypeValue = 0;

changedEventValuesFunction = false;
previousEventValuesFunction = 0;

changedEventValues = false;
}


void EventsSelectionDialog::TransferDataToWindowPreCreateControls( Timeline *whichWindow,
bool whichHideOperatorsList )
{
currentWindow = whichWindow;
currentFilter = currentWindow->getFilter();
hideOperatorsList = whichHideOperatorsList;

typesHandler = new EventTypesInfoManager( currentWindow, currentFilter );
if( !typesHandler->isEmpty() )
{
currentType = typesHandler->getFirstTypeVisible();
typesHandler->setCurrent( currentType );
}
valuesHandler = new EventValuesInfoManager( currentWindow, currentFilter, currentType );
}


void EventsSelectionDialog::TransferDataToWindowPostCreateControls()
{
if ( hideOperatorsList )
{
bool recursive = true;
staticTextFunctionTypes->Hide();
choiceOperatorFunctionTypes->Hide();
boxSizerFunctionTypes->Hide( recursive );

staticTextFunctionValues->Hide();
choiceOperatorFunctionValues->Hide();
boxSizerFunctionValues->Hide( recursive );
}

vector<string> filterFunctions;
currentFilter->getAllFilterFunctions( filterFunctions );

for( vector<string>::iterator it = filterFunctions.begin(); it != filterFunctions.end(); ++it )
{
int i = choiceOperatorFunctionTypes->Append( wxString::FromUTF8( (*it).c_str() ) );
if( (*it) == currentFilter->getEventTypeFunction() )
{
choiceOperatorFunctionTypes->SetSelection( i );
previousEventTypesFunction = i;
}
}

choiceOperatorTypeValue->Append( _( "And" ) );
choiceOperatorTypeValue->Append( _( "Or" ) );
if( currentFilter->getOpTypeValue() == Filter::AND )
{
choiceOperatorTypeValue->SetSelection( 0 );
previousOperatorTypeValue = 0;
}
else
{
choiceOperatorTypeValue->SetSelection( 1 );
previousOperatorTypeValue = 1;
}

for( vector<string>::iterator it = filterFunctions.begin(); it != filterFunctions.end(); ++it )
{
int i = choiceOperatorFunctionValues->Append( wxString::FromUTF8( (*it).c_str() ) );
if( (*it) == currentFilter->getEventValueFunction() )
{
choiceOperatorFunctionValues->SetSelection( i );
previousEventValuesFunction = i;
}
}

UpdateWidgetChecklistboxTypes();
if( typesHandler->countVisible() > 0 )
{
currentType = typesHandler->getCurrent();
bool keepSelected = false; 
UpdateChecklistboxValues( currentType, keepSelected );
}

EnableApplyButton();
}


EventsSelectionDialog::EventsSelectionDialog(  wxWindow* parent,
Timeline *whichWindow,
bool whichHideOperatorsList,
const wxString& caption,
wxWindowID id,
const wxPoint& pos,
const wxSize& size,
long style)
{
Init();
TransferDataToWindowPreCreateControls( whichWindow, whichHideOperatorsList );
Create(parent, id, caption, pos, size, style );
}



bool EventsSelectionDialog::Create( wxWindow* parent,
wxWindowID id,
const wxString& caption,
const wxPoint& pos,
const wxSize& size,
long style )
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



EventsSelectionDialog::~EventsSelectionDialog()
{
delete typesHandler;
delete valuesHandler;
}



void EventsSelectionDialog::CreateControls()
{
EventsSelectionDialog* itemDialog1 = this;

wxBoxSizer* itemBoxSizer2 = new wxBoxSizer(wxVERTICAL);
itemDialog1->SetSizer(itemBoxSizer2);

wxBoxSizer* itemBoxSizer3 = new wxBoxSizer(wxVERTICAL);
itemBoxSizer2->Add(itemBoxSizer3, 1, wxGROW|wxLEFT|wxRIGHT|wxTOP, 5);

wxBoxSizer* itemBoxSizer4 = new wxBoxSizer(wxHORIZONTAL);
itemBoxSizer3->Add(itemBoxSizer4, 1, wxGROW|wxALL, 5);

wxStaticBox* itemStaticBoxSizer5Static = new wxStaticBox(itemDialog1, wxID_ANY, _(" Types "));
wxStaticBoxSizer* itemStaticBoxSizer5 = new wxStaticBoxSizer(itemStaticBoxSizer5Static, wxVERTICAL);
itemBoxSizer4->Add(itemStaticBoxSizer5, 1, wxGROW|wxLEFT|wxRIGHT, 5);

boxSizerFunctionTypes = new wxBoxSizer(wxHORIZONTAL);
itemStaticBoxSizer5->Add(boxSizerFunctionTypes, 0, wxALIGN_RIGHT|wxLEFT|wxRIGHT, 5);

staticTextFunctionTypes = new wxStaticText( itemDialog1, ID_STATIC_TEXT_FUNCTION_TYPES, _("Function"), wxDefaultPosition, wxDefaultSize, 0 );
boxSizerFunctionTypes->Add(staticTextFunctionTypes, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5);

wxArrayString choiceOperatorFunctionTypesStrings;
choiceOperatorFunctionTypes = new wxChoice( itemDialog1, ID_CHOICE_OPERATOR_FUNCTION_TYPES, wxDefaultPosition, wxDefaultSize, choiceOperatorFunctionTypesStrings, 0 );
boxSizerFunctionTypes->Add(choiceOperatorFunctionTypes, 2, wxALIGN_CENTER_VERTICAL|wxLEFT|wxTOP|wxBOTTOM, 5);

wxBoxSizer* itemBoxSizer5 = new wxBoxSizer(wxHORIZONTAL);
itemStaticBoxSizer5->Add(itemBoxSizer5, 0, wxGROW|wxLEFT|wxRIGHT, 5);

checkboxSetAllTypes = new wxCheckBox( itemDialog1, ID_CHECKBOX_SET_ALL_TYPES, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
checkboxSetAllTypes->SetValue(false);
checkboxSetAllTypes->SetHelpText(_("Check/uncheck visible all types."));
if (EventsSelectionDialog::ShowToolTips())
checkboxSetAllTypes->SetToolTip(_("Check/uncheck visible all types."));
itemBoxSizer5->Add(checkboxSetAllTypes, 0, wxALIGN_CENTER_VERTICAL, 5);

wxStaticText* itemStaticText2 = new wxStaticText( itemDialog1, wxID_STATIC, _("Search:"), wxDefaultPosition, wxDefaultSize, 0 );
if (EventsSelectionDialog::ShowToolTips())
itemStaticText2->SetToolTip(_("Search by type or label."));
itemBoxSizer5->Add(itemStaticText2, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5);

typesRegexSearch = new wxTextCtrl( itemDialog1, ID_TEXTCTRL_TYPES_REGEX_SEARCH, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
if (EventsSelectionDialog::ShowToolTips())
typesRegexSearch->SetToolTip(_("Search by type or label."));
itemBoxSizer5->Add(typesRegexSearch, 1, wxGROW|wxTOP|wxBOTTOM, 5);

wxArrayString checkListSelectTypesStrings;
checkListSelectTypes = new wxCheckListBox( itemDialog1, ID_CHECKLISTBOX_TYPES, wxDefaultPosition, wxSize(500, -1), checkListSelectTypesStrings, wxLB_EXTENDED|wxLB_HSCROLL );
itemStaticBoxSizer5->Add(checkListSelectTypes, 1, wxGROW|wxALL, 5);

wxBoxSizer* itemBoxSizer10 = new wxBoxSizer(wxHORIZONTAL);
itemStaticBoxSizer5->Add(itemBoxSizer10, 0, wxALIGN_RIGHT|wxLEFT, 5);

buttonSetAllTypes = new wxButton( itemDialog1, ID_BUTTON_SET_ALL_TYPES, _("Set all"), wxDefaultPosition, wxDefaultSize, 0 );
buttonSetAllTypes->Show(false);
itemBoxSizer10->Add(buttonSetAllTypes, 0, wxALIGN_CENTER_VERTICAL|wxRIGHT|wxTOP|wxBOTTOM, 5);

buttonUnsetAllTypes = new wxButton( itemDialog1, ID_BUTTON_UNSET_ALL_TYPES, _("Unset all"), wxDefaultPosition, wxDefaultSize, 0 );
buttonUnsetAllTypes->Show(false);
itemBoxSizer10->Add(buttonUnsetAllTypes, 0, wxALIGN_CENTER_VERTICAL|wxRIGHT|wxTOP|wxBOTTOM, 5);

wxBoxSizer* itemBoxSizer13 = new wxBoxSizer(wxVERTICAL);
itemBoxSizer4->Add(itemBoxSizer13, 0, wxGROW|wxTOP, 5);

wxStaticLine* itemStaticLine14 = new wxStaticLine( itemDialog1, wxID_STATIC, wxDefaultPosition, wxDefaultSize, wxLI_VERTICAL );
itemBoxSizer13->Add(itemStaticLine14, 1, wxALIGN_CENTER_HORIZONTAL|wxLEFT|wxRIGHT, 5);

wxBoxSizer* itemBoxSizer15 = new wxBoxSizer(wxHORIZONTAL);
itemBoxSizer13->Add(itemBoxSizer15, 0, wxALIGN_CENTER_HORIZONTAL|wxLEFT|wxRIGHT|wxBOTTOM, 5);

wxArrayString choiceOperatorTypeValueStrings;
choiceOperatorTypeValue = new wxChoice( itemDialog1, ID_CHOICE_OPERATOR_TYPE_VALUE, wxDefaultPosition, wxDefaultSize, choiceOperatorTypeValueStrings, 0 );
itemBoxSizer15->Add(choiceOperatorTypeValue, 1, wxALIGN_CENTER_VERTICAL|wxTOP|wxBOTTOM, 5);

wxStaticLine* itemStaticLine17 = new wxStaticLine( itemDialog1, wxID_STATIC, wxDefaultPosition, wxDefaultSize, wxLI_VERTICAL );
itemBoxSizer13->Add(itemStaticLine17, 1, wxALIGN_CENTER_HORIZONTAL|wxLEFT|wxRIGHT, 5);

wxStaticBox* itemStaticBoxSizer18Static = new wxStaticBox(itemDialog1, wxID_ANY, _(" Values "));
wxStaticBoxSizer* itemStaticBoxSizer18 = new wxStaticBoxSizer(itemStaticBoxSizer18Static, wxVERTICAL);
itemBoxSizer4->Add(itemStaticBoxSizer18, 1, wxGROW|wxLEFT|wxRIGHT, 5);

boxSizerFunctionValues = new wxBoxSizer(wxHORIZONTAL);
itemStaticBoxSizer18->Add(boxSizerFunctionValues, 0, wxALIGN_RIGHT|wxLEFT|wxRIGHT, 5);

staticTextFunctionValues = new wxStaticText( itemDialog1, wxID_STATIC, _("Function"), wxDefaultPosition, wxDefaultSize, 0 );
boxSizerFunctionValues->Add(staticTextFunctionValues, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5);

wxArrayString choiceOperatorFunctionValuesStrings;
choiceOperatorFunctionValues = new wxChoice( itemDialog1, ID_CHOICE_OPERATOR_FUNCTION_VALUES, wxDefaultPosition, wxDefaultSize, choiceOperatorFunctionValuesStrings, 0 );
boxSizerFunctionValues->Add(choiceOperatorFunctionValues, 0, wxALIGN_CENTER_VERTICAL|wxLEFT|wxTOP|wxBOTTOM, 5);

wxBoxSizer* itemBoxSizer7 = new wxBoxSizer(wxHORIZONTAL);
itemStaticBoxSizer18->Add(itemBoxSizer7, 0, wxGROW|wxLEFT|wxRIGHT, 5);

checkboxSetAllValues = new wxCheckBox( itemDialog1, ID_CHECKBOX_SET_ALL_VALUES, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
checkboxSetAllValues->SetValue(false);
if (EventsSelectionDialog::ShowToolTips())
checkboxSetAllValues->SetToolTip(_("Check/uncheck all visible values."));
itemBoxSizer7->Add(checkboxSetAllValues, 0, wxALIGN_CENTER_VERTICAL, 5);

wxStaticText* itemStaticText9 = new wxStaticText( itemDialog1, wxID_STATIC, _("Search:"), wxDefaultPosition, wxDefaultSize, 0 );
if (EventsSelectionDialog::ShowToolTips())
itemStaticText9->SetToolTip(_("Search by value or label."));
itemBoxSizer7->Add(itemStaticText9, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5);

valuesRegexSearch = new wxTextCtrl( itemDialog1, ID_TEXTCTRL_VALUES_REGEX_SEARCH, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
if (EventsSelectionDialog::ShowToolTips())
valuesRegexSearch->SetToolTip(_("Search by value or label."));
itemBoxSizer7->Add(valuesRegexSearch, 1, wxGROW|wxTOP|wxBOTTOM, 5);

wxArrayString checkListSelectValuesStrings;
checkListSelectValues = new wxCheckListBox( itemDialog1, ID_CHECKLISTBOX_VALUES, wxDefaultPosition, wxDefaultSize, checkListSelectValuesStrings, wxLB_EXTENDED|wxLB_HSCROLL );
itemStaticBoxSizer18->Add(checkListSelectValues, 1, wxGROW|wxALL, 5);

wxBoxSizer* itemBoxSizer23 = new wxBoxSizer(wxHORIZONTAL);
itemStaticBoxSizer18->Add(itemBoxSizer23, 0, wxGROW|wxALL, 5);

textCtrlAddValues = new wxTextCtrl( itemDialog1, ID_TEXTCTRL_ADD_VALUES, wxEmptyString, wxDefaultPosition, wxDefaultSize, 0 );
if (EventsSelectionDialog::ShowToolTips())
textCtrlAddValues->SetToolTip(_("Insert new value (allowed  0-9 only)."));
itemBoxSizer23->Add(textCtrlAddValues, 1, wxALIGN_CENTER_VERTICAL|wxRIGHT|wxTOP|wxBOTTOM, 5);

buttonAddValues = new wxButton( itemDialog1, ID_BUTTON_ADD_VALUES, _("Add"), wxDefaultPosition, wxDefaultSize, 0 );
itemBoxSizer23->Add(buttonAddValues, 0, wxALIGN_CENTER_VERTICAL|wxTOP|wxBOTTOM, 5);

wxBoxSizer* itemBoxSizer26 = new wxBoxSizer(wxHORIZONTAL);
itemStaticBoxSizer18->Add(itemBoxSizer26, 0, wxGROW|wxRIGHT, 5);

buttonShortLabels = new wxToggleButton( itemDialog1, ID_TOGGLEBUTTON_SHORT_LABELS, _("Short Labels"), wxDefaultPosition, wxDefaultSize, 0 );
buttonShortLabels->SetValue(true);
buttonShortLabels->SetHelpText(_("Show short labels instead of complete"));
if (EventsSelectionDialog::ShowToolTips())
buttonShortLabels->SetToolTip(_("Show short labels instead of complete"));
itemBoxSizer26->Add(buttonShortLabels, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5);

itemBoxSizer26->Add(5, 5, 1, wxALIGN_CENTER_VERTICAL|wxALL, 5);

buttonSetAllValues = new wxButton( itemDialog1, ID_BUTTON_SET_ALL_VALUES, _("Set all"), wxDefaultPosition, wxDefaultSize, 0 );
buttonSetAllValues->Show(false);
itemBoxSizer26->Add(buttonSetAllValues, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5);

buttonUnsetAllValues = new wxButton( itemDialog1, ID_BUTTON_UNSET_ALL_VALUES, _("Unset all"), wxDefaultPosition, wxDefaultSize, 0 );
buttonUnsetAllValues->Show(false);
itemBoxSizer26->Add(buttonUnsetAllValues, 0, wxALIGN_CENTER_VERTICAL|wxTOP|wxBOTTOM, 5);

wxStaticLine* itemStaticLine31 = new wxStaticLine( itemDialog1, wxID_STATIC, wxDefaultPosition, wxDefaultSize, wxLI_HORIZONTAL );
itemBoxSizer2->Add(itemStaticLine31, 0, wxGROW|wxALL, 5);

wxStdDialogButtonSizer* itemStdDialogButtonSizer32 = new wxStdDialogButtonSizer;

itemBoxSizer2->Add(itemStdDialogButtonSizer32, 0, wxALIGN_RIGHT|wxALL, 5);
wxButton* itemButton33 = new wxButton( itemDialog1, wxID_CANCEL, _("&Cancel"), wxDefaultPosition, wxDefaultSize, 0 );
itemStdDialogButtonSizer32->AddButton(itemButton33);

applyButton = new wxButton( itemDialog1, wxID_APPLY, _("&Apply"), wxDefaultPosition, wxDefaultSize, 0 );
itemStdDialogButtonSizer32->AddButton(applyButton);

itemStdDialogButtonSizer32->Realize();

textCtrlAddValues->Connect(ID_TEXTCTRL_ADD_VALUES, wxEVT_KEY_DOWN, wxKeyEventHandler(EventsSelectionDialog::OnTextCtrlKeyDown), nullptr, this);

textCtrlAddValues->SetValidator( wxTextValidator( wxFILTER_NUMERIC ));

TransferDataToWindowPostCreateControls();
}



bool EventsSelectionDialog::ShowToolTips()
{
return true;
}



wxBitmap EventsSelectionDialog::GetBitmapResource( const wxString& name )
{
wxUnusedVar(name);
return wxNullBitmap;
}



wxIcon EventsSelectionDialog::GetIconResource( const wxString& name )
{
wxUnusedVar(name);
return wxNullIcon;
}


int EventsSelectionDialog::GetIndexEventTypesFunction() const
{
return choiceOperatorFunctionTypes->GetSelection();
}


std::string EventsSelectionDialog::GetNameEventTypesFunction() const
{
return std::string( choiceOperatorFunctionTypes->GetString( choiceOperatorFunctionTypes->GetSelection() ).mb_str( wxConvUTF8 ) );
}


wxArrayInt EventsSelectionDialog::GetEventTypesSelection() const
{
return typesHandler->getSelected();
}


int EventsSelectionDialog::GetIndexOperatorTypeValue() const
{
return choiceOperatorTypeValue->GetSelection();
}


std::string EventsSelectionDialog::GetNameOperatorTypeValue() const
{
return std::string( choiceOperatorTypeValue->GetString( choiceOperatorTypeValue->GetSelection() ).mb_str( wxConvUTF8 ) );
}


int EventsSelectionDialog::GetIndexEventValuesFunction() const
{
return choiceOperatorFunctionValues->GetSelection();
}


std::string EventsSelectionDialog::GetNameEventValuesFunction() const
{
return std::string( choiceOperatorFunctionValues->GetString( choiceOperatorFunctionValues->GetSelection() ).mb_str( wxConvUTF8 ) );
}


wxArrayDouble EventsSelectionDialog::GetEventValues() const
{
return valuesHandler->getSelected();
}


bool EventsSelectionDialog::ChangedEventTypesFunction() const
{
return changedEventTypesFunction;
}


bool EventsSelectionDialog::ChangedEventTypesSelection() const
{
return typesHandler->getChangedSelected();
}


bool EventsSelectionDialog::ChangedOperatorTypeValue() const
{
return changedOperatorTypeValue;
}


bool EventsSelectionDialog::ChangedEventValuesFunction() const
{
return changedEventValuesFunction;
}


bool EventsSelectionDialog::ChangedEventValuesSelection() const
{
return changedEventValues;
}



void EventsSelectionDialog::OnIdle( wxIdleEvent& event )
{
buttonSetAllTypes->Enable( true );
buttonUnsetAllTypes->Enable( checkListSelectTypes->GetCount() > 0 );
buttonSetAllValues->Enable( true );
buttonUnsetAllValues->Enable( checkListSelectValues->GetCount() > 0 );
buttonAddValues->Enable( !textCtrlAddValues->IsEmpty() );
}


void EventsSelectionDialog::checkAll( wxCheckListBox *boxlist, bool value )
{
for ( unsigned int i = 0; i < boxlist->GetCount(); ++i )
{
boxlist->Check( i, value );
}
}



void EventsSelectionDialog::OnButtonSetAllTypesClick( wxCommandEvent& event )
{
checkAll( checkListSelectTypes, true );
typesHandler->setAllSelected();
}



void EventsSelectionDialog::OnButtonUnsetAllTypesClick( wxCommandEvent& event )
{
checkAll( checkListSelectTypes, false );
typesHandler->setAllUnselected();
}



void EventsSelectionDialog::OnButtonSetAllValuesClick( wxCommandEvent& event )
{
checkAll( checkListSelectValues, true );
changedEventValues = HasChanged( checkListSelectValues, valuesHandler );
valuesHandler->setAllSelected();
}



void EventsSelectionDialog::OnButtonUnsetAllValuesClick( wxCommandEvent& event )
{
checkAll( checkListSelectValues, false );
changedEventValues = HasChanged( checkListSelectValues, valuesHandler );
valuesHandler->setAllUnselected();
}



void EventsSelectionDialog::OnApplyClick( wxCommandEvent& event )
{
TransferWindowToData();
EndModal( wxID_OK );
}


bool EventsSelectionDialog::HasChanged( wxChoice *choice, int selectedFunction ) const
{
return ( choice->GetCurrentSelection() != selectedFunction );
}


unsigned int EventsSelectionDialog::GetSelections( wxCheckListBox *checkList, wxArrayInt &index ) const
{
unsigned int numSelections = 0;
for( unsigned int i = 0; i < checkList->GetCount(); ++i )
{
if ( checkList->IsChecked( i ) )
{
numSelections++;
index.Add( i );
}
}

return numSelections;
}


bool EventsSelectionDialog::HasChanged( wxCheckListBox *checkList, wxArrayInt &index ) const
{
bool changed = false;

wxArrayInt tmpIndex;
unsigned int numSelections = GetSelections( checkList, tmpIndex ); 

if ( index.Count() == numSelections )
{
for( unsigned int i = 0; i < numSelections; ++i )
{
if ( tmpIndex[ i ] != index[ i ] )
{
changed = true;
break;
}
}
}
else
changed = true;

return changed;
}


bool EventsSelectionDialog::HasChanged( wxCheckListBox *checkList, EventTypesInfoManager *manager ) const
{
bool changed = false;

wxArrayString dummyVisible;
wxArrayInt dummyPosVisible;
wxArrayInt dummyGlobalSelected;
wxArrayInt tmpGUISelected;
int dummyFirstPosSelectedVisible;

typesHandler->getSelectedFromVisible( dummyVisible, dummyPosVisible, dummyGlobalSelected, tmpGUISelected, dummyFirstPosSelectedVisible );

wxArrayInt tmpIndex;
unsigned int numSelections = GetSelections( checkList, tmpIndex ); 

if ( tmpGUISelected.Count() == numSelections )
{
for( unsigned int i = 0; i < numSelections; ++i )
{
if ( tmpIndex[ i ] != tmpGUISelected[ i ] )
{
changed = true;
break;
}
}
}
else
changed = true;

return changed;
}


bool EventsSelectionDialog::HasChanged( wxCheckListBox *checkList, EventValuesInfoManager *manager ) const
{
bool changed = false;

wxArrayString dummyVisible;
wxArrayInt dummyPosVisible;
wxArrayInt dummyGlobalSelected;
wxArrayInt tmpGUISelected;
int dummyFirstPosSelectedVisible;

valuesHandler->getSelectedFromVisible( dummyVisible, dummyPosVisible, dummyGlobalSelected, tmpGUISelected, dummyFirstPosSelectedVisible );

wxArrayInt tmpIndex;
unsigned int numSelections = GetSelections( checkList, tmpIndex ); 

if ( tmpGUISelected.Count() == numSelections )
{
for( unsigned int i = 0; i < numSelections; ++i )
{
if ( tmpIndex[ i ] != tmpGUISelected[ i ] )
{
changed = true;
break;
}
}
}
else
changed = true;

return changed;
}


bool EventsSelectionDialog::HasChanged( wxCheckListBox *checkList, wxArrayDouble &index ) const
{
bool changed = false;

wxArrayInt tmpIndex;
unsigned int numSelections = GetSelections( checkList, tmpIndex ); 

if ( index.Count() == numSelections )
{
for( unsigned int i = 0; i < numSelections; ++i )
{
if ( tmpIndex[ i ] != index[ i ] )
{
changed = true;
break;
}
}
}
else
changed = true;

return changed;
}


bool EventsSelectionDialog::HasChanged( wxArrayInt &arr1, wxArrayInt &arr2 ) const
{
bool changed = false;

if ( arr1.Count() == arr2.Count() )
{
for( unsigned int i = 0; i < arr1.Count(); ++i )
{
if ( arr2.Index( arr1[ i ] ) == wxNOT_FOUND )
{
changed = true;
break;
}
}
}
else
changed = true;

return changed;
}


bool EventsSelectionDialog::HasChanged( wxArrayDouble &arr1, wxArrayDouble &arr2 ) const
{
bool changed = false;

if ( arr1.Count() == arr2.Count() )
{
for( unsigned int i = 0; i < arr1.Count(); ++i )
{
if ( arr2.Index( arr1[ i ] ) == wxNOT_FOUND )
{
changed = true;
break;
}
}
}
else
changed = true;

return changed;
}


bool EventsSelectionDialog::CopyChanges( wxChoice *choice, int &selectedFunction )
{
bool changed = HasChanged( choice, selectedFunction );
selectedFunction = choice->GetCurrentSelection();

return changed;
}


bool EventsSelectionDialog::CopyChanges( wxCheckListBox *checkList,
wxArrayInt &index,
wxArrayString &selected,
bool copyStrings )
{
bool changed = HasChanged( checkList, index );

index.Clear();

unsigned int numSelections = GetSelections( checkList, index ); 

if ( copyStrings )
{
selected.Clear();

for( unsigned int i = 0; i < numSelections; ++i )
{
selected.Add( checkList->GetString( i ) );
}
}

return changed;
}


void EventsSelectionDialog::TransferWindowToData()
{
changedEventTypesFunction = CopyChanges( choiceOperatorFunctionTypes, previousEventTypesFunction );
changedEventTypesSelection = typesHandler->getChangedSelected();
changedOperatorTypeValue = CopyChanges( choiceOperatorTypeValue, previousOperatorTypeValue );
changedEventValuesFunction = CopyChanges( choiceOperatorFunctionValues, previousEventValuesFunction );
changedEventValues = valuesHandler->getChangedSelected();
}



void EventsSelectionDialog::OnChecklistboxTypesToggled( wxCommandEvent& event )
{
int pos = event.GetInt();

typesHandler->setSelected( pos, checkListSelectTypes->IsChecked( pos ) );
TEventType tmpNewCurrentType = typesHandler->getVisible( pos );
if ( currentType != tmpNewCurrentType )
{
currentType = tmpNewCurrentType;
typesHandler->setCurrent( currentType );
UpdateChecklistboxValues( currentType );
}
}



void EventsSelectionDialog::OnChecklistboxValuesToggled( wxCommandEvent& event )
{
valuesHandler->transferFrom( checkListSelectValues );
changedEventValues = valuesHandler->getChangedSelected();
}



void EventsSelectionDialog::OnChoiceOperatorFunctionTypesSelected( wxCommandEvent& event )
{
changedEventTypesFunction = HasChanged( choiceOperatorFunctionTypes, previousEventTypesFunction );
}



void EventsSelectionDialog::OnChoiceOperatorFunctionValuesSelected( wxCommandEvent& event )
{
changedEventValuesFunction = HasChanged( choiceOperatorFunctionValues, previousEventValuesFunction );
}



void EventsSelectionDialog::OnChoiceOperatorTypeValueSelected( wxCommandEvent& event )
{
changedOperatorTypeValue = HasChanged( choiceOperatorTypeValue, previousOperatorTypeValue );
}



void EventsSelectionDialog::OnChecklistboxTypesDoubleClicked( wxCommandEvent& event )
{
int pos = event.GetInt();
if ( pos >= 0 )
{
checkListSelectTypes->Check( pos, !checkListSelectTypes->IsChecked( pos ) );
TEventType tmpNewCurrentType = typesHandler->getVisible( pos );
typesHandler->setSelected( tmpNewCurrentType, checkListSelectTypes->IsChecked( pos ) );

if ( currentType != tmpNewCurrentType )
{
currentType = tmpNewCurrentType;
typesHandler->setCurrent( currentType );
typesHandler->setSelected( currentType, checkListSelectTypes->IsChecked( pos ) );
UpdateChecklistboxValues( currentType );
}
}
}


void EventsSelectionDialog::UpdateWidgetChecklistboxValues()
{
wxArrayString tmpVisible;
wxArrayInt dummyPosVisible;
wxArrayInt dummyGlobalSelected;
wxArrayInt tmpGUISelected;
int firstPos;

valuesHandler->getSelectedFromVisible( tmpVisible, dummyPosVisible,  dummyGlobalSelected, tmpGUISelected, firstPos );

checkListSelectValues->Clear();
if( !tmpVisible.IsEmpty() )
{
checkListSelectValues->InsertItems( tmpVisible, 0 );

for ( unsigned int i = 0; i < tmpGUISelected.GetCount(); ++i )
{
checkListSelectValues->Check( tmpGUISelected[ i ] );
}

checkListSelectValues->SetSelection( firstPos );
}
}


void EventsSelectionDialog::UpdateWidgetChecklistboxTypes()
{
wxArrayString tmpVisible;
wxArrayInt dummyPosVisible;
wxArrayInt dummyGlobalSelected;
wxArrayInt tmpGUISelected;
int firstPos = 0;
bool tmpUpdateFirstPos = true;

typesHandler->getSelectedFromVisible( tmpVisible, dummyPosVisible, dummyGlobalSelected, tmpGUISelected, firstPos, tmpUpdateFirstPos );

TEventType tmpCurrent = typesHandler->getCurrent();
currentType = tmpCurrent;
checkListSelectTypes->Clear(); 

typesHandler->setCurrent( tmpCurrent );

if( !tmpVisible.IsEmpty() )
{
checkListSelectTypes->InsertItems( tmpVisible, 0 );

for ( unsigned int i = 0; i < tmpGUISelected.GetCount(); ++i )
{
checkListSelectTypes->Check( tmpGUISelected[ i ] );
}

checkListSelectTypes->SetSelection( firstPos );
}
}


void EventsSelectionDialog::UpdateChecklistboxValues( TEventType whichType, bool keepSelected )
{
valuesHandler->init( whichType, buttonShortLabels->GetValue(), keepSelected );
UpdateWidgetChecklistboxValues();
}



void EventsSelectionDialog::OnChecklistboxTypesSelected( wxCommandEvent& event )
{
int pos = event.GetInt();
if ( pos >= 0 )
{
TEventType tmpNewCurrentType = typesHandler->getVisible( pos );
if ( currentType != tmpNewCurrentType )
{
currentType = tmpNewCurrentType;
typesHandler->setCurrent( currentType );
typesHandler->setSelected( currentType, checkListSelectTypes->IsChecked( pos ) );
UpdateChecklistboxValues( currentType );
}
}
}


void EventsSelectionDialog::GetEventValueLabels( wxArrayString & whichEventValues )
{
PRV_UINT32 precision = 0;
SemanticInfoType lastType = currentWindow->getSemanticInfoType();
TSemanticValue lastMin = currentWindow->getMinimumY();
TSemanticValue lastMax = currentWindow->getMaximumY();

if( currentWindow->isCodeColorSet() )
{
int endLimit = ceil( lastMax );

if( lastType != EVENTTYPE_TYPE )
{
if( lastType == APPL_TYPE )
endLimit = currentWindow->getTrace()->totalApplications() - 1;
else if( lastType == TASK_TYPE )
endLimit = currentWindow->getTrace()->totalTasks() - 1;
else if( lastType == THREAD_TYPE )
endLimit = currentWindow->getTrace()->totalThreads() - 1;
else if( lastType == NODE_TYPE )
endLimit = currentWindow->getTrace()->totalNodes() - 1;
else if( lastType == CPU_TYPE )
endLimit = currentWindow->getTrace()->totalCPUs() - 1;
else if( lastMax - lastMin > 200 )
endLimit = 200 + floor( lastMin );
}
int typeEndLimit = 0;

for( int i = floor( lastMin ); i <= endLimit; ++i )
{
if( lastType == EVENTTYPE_TYPE && !currentWindow->getTrace()->eventLoaded( i ) )
continue;

string tmpstr;
if( lastType == STATE_TYPE &&
!currentWindow->getTrace()->getStateLabels().getStateLabel( i, tmpstr ) )
continue;
if( lastType == EVENTVALUE_TYPE &&
!currentWindow->getTrace()->getEventLabels().getEventValueLabel( i, tmpstr ) )
continue;

if( ( lastType == EVENTTYPE_TYPE ||
lastType == EVENTVALUE_TYPE ||
lastType == STATE_TYPE )
&& typeEndLimit > 200 )
break;
else
++typeEndLimit;

wxString tmpStr = wxString::FromUTF8( LabelConstructor::semanticLabel( currentWindow, i, true, precision, false ).c_str() );

whichEventValues.Add( tmpStr );
}
}
else
{
wxString tmpStr;
tmpStr << wxT("< ") << wxString::FromUTF8( LabelConstructor::semanticLabel( currentWindow, lastMin, false, precision, false ).c_str() );
whichEventValues.Add( tmpStr );

TSemanticValue step = ( lastMax - lastMin ) / 20.0;
for( int i = 0; i <= 20; ++i )
{
tmpStr.Clear();
tmpStr << wxString::FromUTF8( LabelConstructor::semanticLabel( currentWindow, ( i * step ) + lastMin, false, precision, false ).c_str() );
whichEventValues.Add( tmpStr );
}

tmpStr.Clear();
tmpStr << wxT("> ") << wxString::FromUTF8( LabelConstructor::semanticLabel( currentWindow, lastMax, false, precision, false ).c_str() );
whichEventValues.Add( tmpStr );
}
}



void EventsSelectionDialog::OnChecklistboxValuesDoubleClicked( wxCommandEvent& event )
{
int pos = event.GetInt();
checkListSelectValues->Check( pos, !checkListSelectValues->IsChecked( pos ) );
valuesHandler->transferFrom( checkListSelectValues );
changedEventValues = valuesHandler->getChangedSelected();
}


void EventsSelectionDialog::InsertValueFromTextCtrl()
{
double tmpDouble;
textCtrlAddValues->GetValue().ToDouble( &tmpDouble );

changedEventValues = valuesHandler->insert( tmpDouble, _("") );

if ( changedEventValues )
{
UpdateChecklistboxValues( currentType );
}

textCtrlAddValues->Clear();
}



void EventsSelectionDialog::OnButtonAddValuesClick( wxCommandEvent& event )
{
InsertValueFromTextCtrl();
}



void EventsSelectionDialog::OnTextCtrlKeyDown( wxKeyEvent& event )
{
if ( event.GetKeyCode() == WXK_RETURN )
InsertValueFromTextCtrl();
else
event.Skip();
}


void EventsSelectionDialog::EnableApplyButton()
{
string tmpCurrentTypeFunc  = currentFilter->getEventTypeFunction();
string tmpCurrentValueFunc = currentFilter->getEventValueFunction();

currentFilter->setEventTypeFunction( GetNameEventTypesFunction() );
currentFilter->setEventValueFunction( GetNameEventValuesFunction() );

wxArrayInt dummyIndex;
unsigned int numTypesSelected  = typesHandler->getSelected().Count();
unsigned int numValuesSelected = valuesHandler->getSelected().Count();
bool enabledByFunctionTypes  = currentFilter->allowedEventTypeFunctionNumParams( numTypesSelected );
bool enabledByFunctionValues = currentFilter->allowedEventValueFunctionNumParams( numValuesSelected );

currentFilter->setEventTypeFunction( tmpCurrentTypeFunc );
currentFilter->setEventValueFunction( tmpCurrentValueFunc );

bool someChange = typesHandler->getChangedSelected() || valuesHandler->getChangedSelected();

bool on;
if ( choiceOperatorTypeValue->GetSelection() == 0 ) 
on = ( enabledByFunctionTypes && enabledByFunctionValues ) && someChange;
else
on = ( enabledByFunctionTypes || enabledByFunctionValues ) && someChange;

applyButton->Enable( on );
}



void EventsSelectionDialog::OnApplyUpdate( wxUpdateUIEvent& event )
{
EnableApplyButton();
}



void EventsSelectionDialog::OnTogglebuttonShortLabelsClick( wxCommandEvent& event )
{
UpdateChecklistboxValues( currentType );
}


EventInfoManager::EventInfoManager( Timeline *whichWindow, Filter *whichFilter )
{
currentWindow = whichWindow;
currentFilter = whichFilter;
firstPosSelectedVisible = 0;
filterRegEx = vector< wxRegEx * >();
changedSelection = false;
}


EventInfoManager::~EventInfoManager()
{
clearAllRegEx();
}


bool EventInfoManager::add( wxString whichRegEx )
{
bool regExAdded = false;

wxRegEx *labelRegEx = new wxRegEx();

if ( labelRegEx->Compile( whichRegEx ) )
{
filterRegEx.push_back( labelRegEx );
regExAdded = true;
}

return regExAdded;
}


void EventInfoManager::clearAllRegEx()
{
for( vector< wxRegEx * >::iterator it = filterRegEx.begin(); it != filterRegEx.end(); ++it )
{
delete *it;
}

filterRegEx.clear();
}


bool EventInfoManager::matchesAllRegex( string whichName, string whichValue )
{
bool matchesAll = true;

for( vector< wxRegEx * >::iterator it = filterRegEx.begin(); it != filterRegEx.end(); ++it )
{
if ( !(*it)->Matches( wxString::FromUTF8( whichName.c_str() ) ) &&
!(*it)->Matches( wxString::FromUTF8( whichValue.c_str() ) ) )
{
matchesAll = false;
break;
}
}

return matchesAll;
}


EventTypesInfoManager::EventTypesInfoManager( Timeline *whichWindow, Filter *whichFilter )
: EventInfoManager( whichWindow, whichFilter )
{
init();
}


void EventTypesInfoManager::setAllVisible()
{
visible.Clear();

for( size_t i = 0; i < fullList.size(); ++i )
{
visible.Add( i );
}
}


void EventTypesInfoManager::setSelected( int vPos, bool isChecked )
{
firstPosSelectedVisible = visible[ vPos ];

int elemPos = selected.Index( visible[ vPos ] );
if ( isChecked )
{
if ( elemPos == wxNOT_FOUND )
selected.Add( visible[ vPos ] );
}
else
{
if ( elemPos != wxNOT_FOUND )
selected.RemoveAt( elemPos );
}

selected.Sort( compare_int );

setChangedSelection();
}


void EventTypesInfoManager::setSelected( TEventType whichSelected, bool isChecked )
{
int elemPos = wxNOT_FOUND;

vector< TEventType >::iterator itType = find( fullList.begin(), fullList.end(), whichSelected );
if ( itType != fullList.end() )
{
elemPos = int( itType - fullList.begin() );
}

int posInSelected = selected.Index( elemPos );
if ( isChecked )
{
if ( posInSelected == wxNOT_FOUND )
selected.Add( elemPos );
}
else
{
if ( posInSelected != wxNOT_FOUND )
selected.RemoveAt( posInSelected );
}

selected.Sort( compare_int );

setChangedSelection();
}


void EventTypesInfoManager::setAllSelected()
{
for( size_t i = 0; i < visible.GetCount(); ++i )
{
if ( selected.Index( visible[ i ] ) == wxNOT_FOUND )
selected.Add( visible[ i ] );
}

selected.Sort( compare_int );

setChangedSelection();
}


void EventTypesInfoManager::setAllUnselected()
{
for( size_t i = 0; i < visible.GetCount(); ++i )
{
int tmpPos = selected.Index( visible[ i ] );
if ( tmpPos != wxNOT_FOUND )
selected.RemoveAt( tmpPos );
}

selected.Sort( compare_int );

setChangedSelection();
}


void EventTypesInfoManager::init()
{
fullList.clear();
labels.Clear();
visible.Clear();
selected.Clear();
initialSelected.Clear();

set< TEventType > tmpEventTypes = currentWindow->getTrace()->getLoadedEvents();

vector< TEventType > labeledTypes;
currentWindow->getTrace()->getEventLabels().getTypes( labeledTypes );

tmpEventTypes.insert( labeledTypes.begin(), labeledTypes.end() );

fullList.clear();
fullList.assign( tmpEventTypes.begin(), tmpEventTypes.end() ); 

vector< TEventType > tmpSelectedTypes;
currentFilter->getEventType( tmpSelectedTypes );

for( vector< TEventType >::iterator it = fullList.begin(); it != fullList.end(); ++it )
{
string tmpstr;
currentWindow->getTrace()->getEventLabels().getEventTypeLabel( (*it), tmpstr );

labels.Add( wxString() << ( *it ) << _( " " ) << wxString::FromUTF8( tmpstr.c_str() ) );

vector< TEventType >::iterator itType = find( tmpSelectedTypes.begin(), tmpSelectedTypes.end(), ( *it ) );
if ( itType != tmpSelectedTypes.end() )
{
int tmpPos = int( it - fullList.begin() );
selected.Add( tmpPos );
initialSelected.Add( tmpPos );
}

visible.Add( int( it - fullList.begin() ) );
}

if ( selected.GetCount() > 0 )
{
firstPosSelectedVisible = selected[ 0 ];
currentType = firstPosSelectedVisible;
}

changedSelection = false;
}





wxArrayInt EventTypesInfoManager::getSelected()
{
wxArrayInt tmpTypesSelected;

for( size_t i = 0; i < selected.GetCount(); ++i )
{
tmpTypesSelected.Add( fullList[ selected[ i ]] );
}


return tmpTypesSelected;
}


void EventTypesInfoManager::getSelectedFromVisible( wxArrayString& whichVisible,
wxArrayInt &whichPosVisible,
wxArrayInt &whichGlobalSelection,
wxArrayInt &whichGUISelection,
int &whichFirstPosSelectedVisible,
bool updateFirstPosSelectedVisible )
{
wxArrayString tmpVisible;
wxArrayInt tmpPosVisible;
wxArrayInt tmpGlobalSelection;
wxArrayInt tmpGUISelection;
int tmpFirstPosSelectedVisible = 0;

bool foundFirst = false;

unsigned int selectedPositionInGUI = 0;

for ( unsigned int i = 0; i < visible.GetCount(); ++i )
{
stringstream tmpValue;
tmpValue << fullList[ visible[ i ] ];

string tmpLabel( labels[ visible[ i ] ].mb_str() );

if ( matchesAllRegex( tmpLabel, tmpValue.str() ) )
{
tmpVisible.Add( labels[ visible[ i ] ] );
tmpPosVisible.Add( visible[ i ] );
if ( selected.Index( visible[ i ] ) != wxNOT_FOUND )
{
tmpGUISelection.Add( selectedPositionInGUI );
tmpGlobalSelection.Add( visible[ i ] );
}


if (( !foundFirst ) && ( fullList[ visible[ i ] ] == currentType ))
{
tmpFirstPosSelectedVisible = selectedPositionInGUI;
foundFirst = true;
}

++selectedPositionInGUI;
}
}

whichVisible = tmpVisible;
whichPosVisible = tmpPosVisible;
whichGlobalSelection = tmpGlobalSelection;
whichGUISelection = tmpGUISelection;
whichFirstPosSelectedVisible = tmpFirstPosSelectedVisible;

if ( updateFirstPosSelectedVisible )
{
firstPosSelectedVisible = tmpFirstPosSelectedVisible;
}
}


void EventTypesInfoManager::updateVisible()
{
wxArrayString dummyVisible;
wxArrayInt posVisible;
wxArrayInt dummyGlobalSelected;
wxArrayInt dummyGUISelected;
int dummyFirstPosSelectedVisible;

getSelectedFromVisible( dummyVisible, posVisible, dummyGlobalSelected, dummyGUISelected, dummyFirstPosSelectedVisible );

setVisible( posVisible );
}


void EventTypesInfoManager::setChangedSelection()
{
bool changed = false;

if ( selected.GetCount() != initialSelected.GetCount() )
{
changed = true;
}
else
{
for( size_t i = 0; i < selected.GetCount(); ++i )
{
if ( selected[ i ] != initialSelected[ i ] )
{
changed = true;
break;
}
}
}

changedSelection = changed;
}


EventValuesInfoManager::EventValuesInfoManager( Timeline *whichWindow, Filter *whichFilter, TEventType whichType )
: EventInfoManager( whichWindow, whichFilter )
{
currentType = whichType;
}


void EventValuesInfoManager::setAllVisible()
{
visible.Clear();
for( size_t i = 0; i < fullList.size(); ++i )
{
visible.Add( fullList[i] );
}
}


void EventValuesInfoManager::init( TEventType whichType, bool shortVersion, bool keepSelected )
{
fullList.Clear();
labels.Clear();
visible.Clear();
if ( ! keepSelected )
selected.Clear();

map< TEventValue, string > auxValues;
currentWindow->getTrace()->getEventLabels().getValues( whichType, auxValues );
for( map< TEventValue, string >::iterator it = auxValues.begin(); it != auxValues.end(); ++it )
{
fullList.Add( (*it).first );
}

if ( ! keepSelected )
{
vector< TSemanticValue > tmpValues;
currentFilter->getEventValue( tmpValues );
for( vector<TSemanticValue>::iterator it = tmpValues.begin(); it != tmpValues.end(); ++it )
{
selected.Add( (*it) );
}
}

for( unsigned int i = 0; i < selected.GetCount(); ++i )
{
if ( fullList.Index( selected[ i ] ) == wxNOT_FOUND )
{
fullList.Add( selected[ i ] );
}
}

for( unsigned int i = 0; i < addedFullList.GetCount(); ++i )
{
if ( fullList.Index( addedFullList[ i ] ) == wxNOT_FOUND )
{
fullList.Add( addedFullList[ i ] );
}
}

fullList.Sort( compare_double );

wxArrayString tmpEventValues;
for( unsigned int i = 0; i < fullList.GetCount(); ++i )
{
stringstream tmpValue;
tmpValue << fullList[ i ];

string tmpLabel = LabelConstructor::eventValueLabel( currentWindow, whichType, fullList[ i ], true );
if ( tmpLabel == "" )
{
tmpLabel = tmpValue.str();
}
else
{
if( shortVersion )
LabelConstructor::transformToShort( tmpLabel );
}

tmpEventValues.Add( wxString::FromUTF8( tmpLabel.c_str() ) );

if ( matchesAllRegex( tmpLabel, tmpValue.str() ) )
{
visible.Add( fullList[ i ] );
}
}

labels = tmpEventValues;
}


wxArrayDouble EventValuesInfoManager::getSelected()
{
return selected;
}


void EventValuesInfoManager::setAllSelected()
{
for( size_t i = 0; i < visible.GetCount(); ++i )
{
if ( selected.Index( visible[ i ] ) == wxNOT_FOUND )
selected.Add( visible[ i ] );
}

selected.Sort( compare_double );

setChangedSelection();
}


void EventValuesInfoManager::setAllUnselected()
{
for( size_t i = 0; i < visible.GetCount(); ++i )
{
int tmpPos = selected.Index( visible[ i ] );
if ( tmpPos != wxNOT_FOUND )
selected.RemoveAt( tmpPos );
}

selected.Sort( compare_double );

setChangedSelection();
}


void EventValuesInfoManager::getSelectedFromVisible( wxArrayString& whichVisible,
wxArrayInt &whichPosVisible,
wxArrayInt &whichGlobalSelection,
wxArrayInt &whichGUISelection,
int &whichFirstPosSelectedVisible,
bool updateFirstPosSelectedVisible )
{
wxArrayString tmpVisible;
wxArrayInt tmpPosVisible;
wxArrayInt tmpGlobalSelection;
wxArrayInt tmpGUISelection;

firstPosSelectedVisible = 0;
int tmpFirstPosSelectedVisible = 0;

bool foundFirst = false;

unsigned int selectedPositionInGUI = 0;

for ( unsigned int i = 0; i < visible.GetCount(); ++i )
{
int fullListPos = fullList.Index( visible[ i ] );
string tmpLabel( labels[ fullListPos ].mb_str() );

stringstream tmpValue;
tmpValue << fullList[ fullListPos ];
if ( matchesAllRegex( tmpLabel, tmpValue.str() ) )
{
tmpVisible.Add( labels[ fullListPos ] );
tmpPosVisible.Add( fullListPos );

if ( selected.Index( visible[ i ] ) != wxNOT_FOUND )
{
tmpGUISelection.Add( selectedPositionInGUI );
tmpGlobalSelection.Add( visible[ i ] );

if ( !foundFirst )
{
tmpFirstPosSelectedVisible = selectedPositionInGUI;
foundFirst = true;
}
}

++selectedPositionInGUI;
}
}

whichVisible = tmpVisible;
whichPosVisible = tmpPosVisible;
whichGlobalSelection = tmpGlobalSelection;
whichGUISelection = tmpGUISelection;
whichFirstPosSelectedVisible = tmpFirstPosSelectedVisible;

if ( updateFirstPosSelectedVisible )
{
firstPosSelectedVisible = tmpFirstPosSelectedVisible;
}
}


bool EventValuesInfoManager::insert( double whichValue, wxString whichLabel )
{
bool changed = false;

if( fullList.Index( whichValue ) == wxNOT_FOUND )
{
fullList.Add( whichValue );
fullList.Sort( compare_double );

int insertPos = fullList.Index( whichValue );
int maxPos = fullList.GetCount() - 1;
wxArrayString tmpLabeledEventValues;
for( int i = 0; i < insertPos; ++i )
{
tmpLabeledEventValues.Add( labels[ i ] );
}

tmpLabeledEventValues.Add( whichLabel );

for( int i = insertPos; i < maxPos; ++i )
{
tmpLabeledEventValues.Add( labels[ i ] );
}

labels = tmpLabeledEventValues;

changed = true;
}

if( selected.Index( whichValue ) == wxNOT_FOUND )
{
selected.Add( whichValue );
selected.Sort( compare_double );
changed = true;
}

if ( visible.Index( whichValue ) == wxNOT_FOUND )
{
stringstream tmpValue;
tmpValue << whichValue;
if ( matchesAllRegex( string( whichLabel.mb_str() ), tmpValue.str() ) )
{
visible.Add( whichValue );
visible.Sort( compare_double );
}
}

addedFullList.Add( whichValue );

setChangedSelection();

return changed;
}


void EventValuesInfoManager::transferFrom( wxCheckListBox *whichList )
{
for( unsigned int i = 0; i < visible.GetCount(); ++i )
{
if ( whichList->IsChecked( i ) )
{
if( selected.Index( visible[i] ) == wxNOT_FOUND )
{
selected.Add( visible[i] );
}
}
else
{
int pos = selected.Index( visible[ i ] );
if( pos != wxNOT_FOUND )
{
selected.RemoveAt( pos );
}
}
}

selected.Sort( compare_double );

setChangedSelection();
}


void EventValuesInfoManager::setChangedSelection()
{
bool changed = false;

if ( selected.GetCount() != initialSelected.GetCount() )
{
changed = true;
}
else
{
for( size_t i = 0; i < selected.GetCount(); ++i )
{
if ( selected[ i ] != initialSelected[ i ] )
{
changed = true;
break;
}
}
}

changedSelection = changed;
}


void EventValuesInfoManager::setVisible( wxArrayInt whichVisible )
{
visible.Clear();
for ( size_t i = 0; i < whichVisible.Count(); ++i )
{
visible.Add( fullList[ whichVisible[ i ] ] );
}
}


void EventValuesInfoManager::updateVisible()
{
wxArrayString dummyVisible;
wxArrayInt posVisible;
wxArrayInt dummyGlobalSelected;
wxArrayInt dummyGUISelected;
int dummyFirstPosSelectedVisible;

getSelectedFromVisible( dummyVisible, posVisible, dummyGlobalSelected, dummyGUISelected, dummyFirstPosSelectedVisible );

setVisible( posVisible );
}



void EventsSelectionDialog::OnTextctrlTypesRegexSearchTextUpdated( wxCommandEvent& event )
{
typesHandler->clearAllRegEx();
typesHandler->setAllVisible();
checkboxSetAllTypes->SetValue( false );

if ( typesRegexSearch->GetValue().Len() > 0 )
{
if ( typesHandler->add( typesRegexSearch->GetValue() ) )
{
typesHandler->updateVisible();
UpdateWidgetChecklistboxTypes();
checkListSelectValues->Clear();
}
}
else
{
typesHandler->updateVisible();
UpdateWidgetChecklistboxTypes();
checkListSelectValues->Clear();
}
}



void EventsSelectionDialog::OnCheckboxSetAllTypesClick( wxCommandEvent& event )
{
bool checked = checkboxSetAllTypes->GetValue();

checkAll( checkListSelectTypes, checked );
if ( checked )
{
typesHandler->setAllSelected();
}
else
{
typesHandler->setAllUnselected();
}
}



void EventsSelectionDialog::OnCheckboxSetAllValuesClick( wxCommandEvent& event )
{
bool checked = checkboxSetAllValues->GetValue();

checkAll( checkListSelectValues, checked );
changedEventValues = HasChanged( checkListSelectValues, valuesHandler );
if ( checked )
{
valuesHandler->setAllSelected();
}
else
{
valuesHandler->setAllUnselected();
}
}



void EventsSelectionDialog::OnTextctrlValuesRegexSearchTextUpdated( wxCommandEvent& event )
{
valuesHandler->clearAllRegEx();
valuesHandler->setAllVisible();
checkboxSetAllValues->SetValue( false );

if ( valuesRegexSearch->GetValue().Len() > 0 )
{
if ( valuesHandler->add( valuesRegexSearch->GetValue() ) )
{
valuesHandler->updateVisible();
UpdateChecklistboxValues( currentType );
}
}
else
{
valuesHandler->updateVisible();
UpdateChecklistboxValues( currentType );
}
}



void EventsSelectionDialog::OnCheckboxSetAllTypesUpdate( wxUpdateUIEvent& event )
{
checkboxSetAllTypes->Enable( checkListSelectTypes->GetCount() > 0 );
}



void EventsSelectionDialog::OnCheckboxSetAllValuesUpdate( wxUpdateUIEvent& event )
{
checkboxSetAllValues->Enable( checkListSelectValues->GetCount() > 0 );
}

