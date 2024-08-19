

#pragma once





#include <wx/filectrl.h>
#include <wx/filename.h>
#include <wx/textfile.h>
#include <wx/dir.h>




class wxBoxSizer;
class wxFileCtrl;



#define ID_SAVEIMAGEDIALOG 10000
#define ID_TEXTPATH 10501
#define ID_FILENAVIGATOR 10002
#define ID_SAVEIMAGECHECKBOX 10504
#define ID_SAVEIMAGETEXTCTRL 10505
#define ID_SAVELEGENDCHECKBOX 10506
#define ID_SAVELEGENDTEXTCTRL 10507
#define SYMBOL_SAVEIMAGEDIALOG_STYLE wxCAPTION|wxRESIZE_BORDER|wxSYSTEM_MENU|wxCLOSE_BOX|wxTAB_TRAVERSAL
#define SYMBOL_SAVEIMAGEDIALOG_TITLE _("Save Image Dialog")
#define SYMBOL_SAVEIMAGEDIALOG_IDNAME ID_SAVEIMAGEDIALOG
#define SYMBOL_SAVEIMAGEDIALOG_SIZE wxDefaultSize
#define SYMBOL_SAVEIMAGEDIALOG_POSITION wxDefaultPosition




class SaveImageDialog: public wxDialog
{    
DECLARE_DYNAMIC_CLASS( SaveImageDialog )
DECLARE_EVENT_TABLE()

public:
SaveImageDialog();
SaveImageDialog( wxWindow* parent, wxString& directoryStartingPath, wxString defaultFileName, bool isHistogram = false, wxString legendSuffix = _( "_legend" ), wxWindowID id = SYMBOL_SAVEIMAGEDIALOG_IDNAME, const wxString& caption = SYMBOL_SAVEIMAGEDIALOG_TITLE, const wxPoint& pos = SYMBOL_SAVEIMAGEDIALOG_POSITION, const wxSize& size = SYMBOL_SAVEIMAGEDIALOG_SIZE, long style = SYMBOL_SAVEIMAGEDIALOG_STYLE );

bool Create( wxWindow* parent, wxWindowID id = SYMBOL_SAVEIMAGEDIALOG_IDNAME, const wxString& caption = SYMBOL_SAVEIMAGEDIALOG_TITLE, const wxPoint& pos = SYMBOL_SAVEIMAGEDIALOG_POSITION, const wxSize& size = SYMBOL_SAVEIMAGEDIALOG_SIZE, long style = SYMBOL_SAVEIMAGEDIALOG_STYLE );

~SaveImageDialog();

void Init();

void CreateControls();


wxString GetImageFilePath();
wxString GetLegendFilePath();

bool DialogSavesImage();
bool DialogSavesLegend();
int GetFilterIndex(); 


void OnTextpathEnter( wxCommandEvent& event );

void OnFilenavigatorUpdate( wxUpdateUIEvent& event );

void OnSaveimagecheckboxClick( wxCommandEvent& event );

void OnSavelegendcheckboxClick( wxCommandEvent& event );

void OnOkClick( wxCommandEvent& event );

void OnOkUpdate( wxUpdateUIEvent& event );

void OnCancelClick( wxCommandEvent& event );


void OnFileNavigatorChanged( wxFileCtrlEvent& event );

void updateFileNamesAndPaths();
void setImageFileName();


wxBitmap GetBitmapResource( const wxString& name );

wxIcon GetIconResource( const wxString& name );

static bool ShowToolTips();

wxBoxSizer* sizerMain;
wxBoxSizer* sizerPath;
wxTextCtrl* textPath;
wxFileCtrl* fileNavigator;
wxStaticBoxSizer* imageToSaveSizer;
wxBoxSizer* imageSizer;
wxCheckBox* imageCheckbox;
wxTextCtrl* imageFileName;
wxBoxSizer* legendSizer;
wxCheckBox* legendCheckbox;
wxTextCtrl* legendFileName;
wxButton* buttonSave;
wxButton* buttonCancel;
static wxString directoryStartingPath;
wxString defaultFileName;
wxString selectedImageFilePath;
wxString selectedLegendFilePath;

wxString fileTypeText;
bool isHistogram;
wxString legendSuffix;
};
