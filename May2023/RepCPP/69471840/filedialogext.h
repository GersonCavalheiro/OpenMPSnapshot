


#pragma once


#include <wx/window.h>
#include <wx/filedlg.h>

#include <vector>



class FileDialogExtension : public wxFileDialog
{
public:
FileDialogExtension( wxWindow* parent,
const wxString& message = wxT("Choose a file"),
const wxString& defaultDir = wxT(""),
const wxString& defaultFile = wxT(""),
const wxString& wildcard = wxT( "*.*"),
long style = wxFD_DEFAULT_STYLE,
const wxPoint& pos = wxDefaultPosition,
const wxSize& sz = wxDefaultSize,
const wxString& name = wxT("filedlg"),
const std::vector< wxString >& whichExtensions = std::vector< wxString >() );

~FileDialogExtension() {};

int ShowModal();
wxString GetPath() const { return path; }

private:
wxString path;
std::vector< wxString > extensions;

bool canWriteDir( wxString whichFile );
bool canWriteFile( wxString whichFile );
};


