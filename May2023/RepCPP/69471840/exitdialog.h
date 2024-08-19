

#pragma once











#define ID_EXITDIALOG 10000
#define ID_BUTTON_SAVE_EXIT 10253
#define ID_BUTTON_CANCEL 10252
#define ID_BUTTON_CLOSE_NO_SAVE 10251
#define SYMBOL_EXITDIALOG_STYLE wxCAPTION|wxRESIZE_BORDER|wxSYSTEM_MENU|wxCLOSE_BOX|wxTAB_TRAVERSAL
#define SYMBOL_EXITDIALOG_TITLE _("Exit Dialog")
#define SYMBOL_EXITDIALOG_IDNAME ID_EXITDIALOG
#define SYMBOL_EXITDIALOG_SIZE wxDefaultSize
#define SYMBOL_EXITDIALOG_POSITION wxDefaultPosition




class ExitDialog: public wxDialog
{    
DECLARE_DYNAMIC_CLASS( ExitDialog )
DECLARE_EVENT_TABLE()

public:
ExitDialog();
ExitDialog( wxWindow* parent, wxWindowID id = SYMBOL_EXITDIALOG_IDNAME, const wxString& caption = SYMBOL_EXITDIALOG_TITLE, const wxPoint& pos = SYMBOL_EXITDIALOG_POSITION, const wxSize& size = SYMBOL_EXITDIALOG_SIZE, long style = SYMBOL_EXITDIALOG_STYLE );

bool Create( wxWindow* parent, wxWindowID id = SYMBOL_EXITDIALOG_IDNAME, const wxString& caption = SYMBOL_EXITDIALOG_TITLE, const wxPoint& pos = SYMBOL_EXITDIALOG_POSITION, const wxSize& size = SYMBOL_EXITDIALOG_SIZE, long style = SYMBOL_EXITDIALOG_STYLE );

~ExitDialog();

void Init();

void CreateControls();


void OnButtonSaveExitClick( wxCommandEvent& event );

void OnButtonCancelClick( wxCommandEvent& event );

void OnButtonCloseNoSaveClick( wxCommandEvent& event );



wxBitmap GetBitmapResource( const wxString& name );

wxIcon GetIconResource( const wxString& name );

static bool ShowToolTips();

wxButton* saveExitButton;
wxButton* cancelButton;
wxButton* noSaveExitButton;
};
