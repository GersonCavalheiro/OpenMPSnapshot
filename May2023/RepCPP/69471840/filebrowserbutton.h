


#pragma once


#include <wx/button.h>
#include <wx/filedlg.h>
#include <wx/filename.h>
#include <wx/textctrl.h>

class BrowserButton : public wxButton
{
DECLARE_DYNAMIC_CLASS( BrowserButton )
DECLARE_EVENT_TABLE()

public:
static const wxString DEFAULT_WILDCARD;

BrowserButton() {}
BrowserButton( wxWindow* parent,
wxWindowID id,
const wxString& label = wxEmptyString,
const wxPoint& pos = wxDefaultPosition,
const wxSize& size = wxDefaultSize,
long style = 0,

wxTextCtrl *whichTextCtrl = nullptr, 
const wxString& whichDialogMessage = wxT( "Choose a file" ),
const wxString& whichDialogDefaultDir = wxT( "" ),
long whichDialogStyle = wxFD_DEFAULT_STYLE, 

const wxValidator& validator = wxDefaultValidator,
const wxString& name = wxT( "button" ) );
~BrowserButton() {}

void SetTextBox( wxTextCtrl *whichTextCtrl, bool readOnly = true );
void SetDialogMessage( const wxString& whichDialogMessage ) { dialogMessage = whichDialogMessage; }
void SetDialogDefaultDir( const wxString& defaultDir ) { dialogDefaultDir = defaultDir; }
void SetDialogStyle( long whichDialogStyle ) { dialogStyle = whichDialogStyle; }
void SetPath( const wxString& whichFullPath ) { fullPath = wxFileName( whichFullPath ); }

wxString GetPath() const
{ 
return ( fullPath.GetPath( wxPATH_GET_VOLUME | wxPATH_GET_SEPARATOR ) + fullPath.GetFullName() ); 
}

bool Enable( bool enable = true );

void OnButton( wxMouseEvent& event ) {} 

protected:  
wxTextCtrl *associatedTextCtrl; 

wxString dialogMessage;
wxString dialogDefaultDir;
long dialogStyle;

void Init();

private:
wxFileName fullPath;
};


class FileBrowserButton : public BrowserButton
{
DECLARE_DYNAMIC_CLASS( FileBrowserButton )
DECLARE_EVENT_TABLE()

public:
FileBrowserButton() { Init(); }
FileBrowserButton( wxWindow* parent,
wxWindowID id,
const wxString& label = wxEmptyString,
const wxPoint& pos = wxDefaultPosition,
const wxSize& size = wxDefaultSize,
long style = 0,

wxTextCtrl *whichTextCtrl = nullptr, 
const wxString& whichDialogMessage = wxT( "Choose a file" ),
const wxString& whichDialogDefaultDir = wxT( "" ),
const wxString& whichFileDialogDefaultFile = wxT( "" ),
const wxString& whichFileDialogWildcard = DEFAULT_WILDCARD,
long whichDialogStyle = wxFD_DEFAULT_STYLE, 

const wxValidator& validator = wxDefaultValidator,
const wxString& name = wxT( "button" ) );
~FileBrowserButton() {}    

void SetFileDialogDefaultFile( const wxString& defaultFile )
{ fileDialogDefaultFile = defaultFile; }

void SetFileDialogWildcard( const wxString& whichFileDialogWildcard )
{ fileDialogWildcard = whichFileDialogWildcard; }

void SetPath( const wxString& whichPath );
void ChangePath( const wxString& whichPath );

void OnButton( wxMouseEvent& event );


private:
wxString fileDialogDefaultFile;
wxString fileDialogWildcard;

void Init();
};


class DirBrowserButton : public BrowserButton
{
DECLARE_DYNAMIC_CLASS( FileBrowserButton )
DECLARE_EVENT_TABLE()

public:
DirBrowserButton() { Init(); }
DirBrowserButton( wxWindow* parent,
wxWindowID id,
const wxString& label = wxEmptyString,
const wxPoint& pos = wxDefaultPosition,
const wxSize& size = wxDefaultSize,
long style = 0,

wxTextCtrl *whichTextCtrl = nullptr, 
const wxString& whichDialogMessage = wxT( "Choose a file" ),
const wxString& whichDialogDefaultDir = wxT( "" ),
long whichDialogStyle = wxFD_DEFAULT_STYLE, 

const wxValidator& validator = wxDefaultValidator,
const wxString& name = wxT( "button" ) );
~DirBrowserButton() {}    

void SetPath( const wxString& whichPath );

void OnButton( wxMouseEvent& event );
};


