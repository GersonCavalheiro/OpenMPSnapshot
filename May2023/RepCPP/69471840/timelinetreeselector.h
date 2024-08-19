


#pragma once


#include <wx/dialog.h>
#include <wx/treectrl.h>
#include "loadedwindows.h"

class TimelineTreeSelector : public wxDialog
{
DECLARE_DYNAMIC_CLASS( TimelineTreeSelector )
DECLARE_EVENT_TABLE()

public:
TimelineTreeSelector()
{}

TimelineTreeSelector( wxWindow* parent,
wxWindowID id,
const wxString& title,
const std::vector<TWindowID>& windows,
const Timeline *currentWindow,
const Trace *currentTrace,
bool needNoneElement = false,
const wxPoint& pos = wxDefaultPosition,
const wxSize& size = wxDefaultSize,
long style = wxDEFAULT_DIALOG_STYLE,
const wxString& name = wxT( "dialogBox" ) );

~TimelineTreeSelector()
{}

Timeline *getSelection() const;

private:
void CreateControls();
void fillTree( const std::vector<TWindowID>& windows,
const Timeline *currentWindow,
const Trace *currentTrace,
bool needNoneElement );
void addTreeItem( Timeline *whichWindow, const Timeline *currentWindow, wxTreeItemId whichParent );
void OnTreeItemActivated( wxTreeEvent& event );

wxTreeCtrl *timelineTree;
};


