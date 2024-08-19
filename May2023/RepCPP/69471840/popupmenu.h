


#pragma once


#include <wx/menu.h>
#include <wx/choicdlg.h>
#include <wx/propdlg.h>
#include <wx/generic/propdlg.h>
#include "rowsselectiondialog.h"
#include "copypaste.h"

#include <map>

#define ID_MENU_COPY                                       30000
#define ID_MENU_PASTE_TIME                                 30001
#define ID_MENU_PASTE_OBJECTS                              30002
#define ID_MENU_PASTE_SIZE                                 30003
#define ID_MENU_PASTE_DURATION                             30004
#define ID_MENU_PASTE_SEMANTIC_SCALE                       30005
#define ID_MENU_PASTE_FILTER_ALL                           30006
#define ID_MENU_PASTE_FILTER_COMMS                         30007
#define ID_MENU_PASTE_FILTER_EVENTS                        30008
#define ID_MENU_PASTE_SPECIAL                              30009
#define ID_MENU_CLONE                                      30010
#define ID_MENU_UNDO_ZOOM                                  30011
#define ID_MENU_REDO_ZOOM                                  30012
#define ID_MENU_FIT_TIME                                   30013
#define ID_MENU_FIT_SEMANTIC_MIN                           30014
#define ID_MENU_FIT_SEMANTIC_MAX                           30015
#define ID_MENU_FIT_SEMANTIC_BOTH                          30016
#define ID_MENU_FIT_OBJECTS                                30017
#define ID_MENU_VIEW_COMMUNICATION_LINES                   30018
#define ID_MENU_VIEW_EVENT_FLAGS                           30019
#define ID_MENU_CODE_COLOR                                 30020
#define ID_MENU_GRADIENT_COLOR                             30021
#define ID_MENU_NOT_NULL_GRADIENT_COLOR                    30022
#define ID_MENU_DRAWMODE_TIME_LAST                         30023
#define ID_MENU_DRAWMODE_TIME_MAXIMUM                      30024
#define ID_MENU_DRAWMODE_TIME_MINIMUM_NOT_ZERO             30025
#define ID_MENU_DRAWMODE_TIME_RANDOM                       30026
#define ID_MENU_DRAWMODE_TIME_RANDOM_NOT_ZERO              30027
#define ID_MENU_DRAWMODE_TIME_AVERAGE                      30028
#define ID_MENU_DRAWMODE_OBJECTS_LAST                      30029
#define ID_MENU_DRAWMODE_OBJECTS_MAXIMUM                   30030
#define ID_MENU_DRAWMODE_OBJECTS_MINIMUM_NOT_ZERO          30031
#define ID_MENU_DRAWMODE_OBJECTS_RANDOM                    30032
#define ID_MENU_DRAWMODE_OBJECTS_RANDOM_NOT_ZERO           30033
#define ID_MENU_DRAWMODE_OBJECTS_AVERAGE                   30034
#define ID_MENU_DRAWMODE_BOTH_LAST                         30035
#define ID_MENU_DRAWMODE_BOTH_MAXIMUM                      30036
#define ID_MENU_DRAWMODE_BOTH_MINIMUM_NOT_ZERO             30037
#define ID_MENU_DRAWMODE_BOTH_RANDOM                       30038
#define ID_MENU_DRAWMODE_BOTH_RANDOM_NOT_ZERO              30039
#define ID_MENU_DRAWMODE_BOTH_AVERAGE                      30040
#define ID_MENU_PIXEL_SIZE_x1                              30041
#define ID_MENU_PIXEL_SIZE_x2                              30042
#define ID_MENU_PIXEL_SIZE_x4                              30043
#define ID_MENU_PIXEL_SIZE_x8                              30044
#define ID_MENU_ROW_SELECTION                              30045
#define ID_MENU_SAVE_IMAGE                                 30046
#define ID_MENU_INFO_PANEL                                 30047
#define ID_MENU_AUTO_CONTROL_SCALE                         30048
#define ID_MENU_AUTO_3D_SCALE                              30049
#define ID_MENU_AUTO_DATA_GRADIENT                         30050
#define ID_MENU_GRADIENT_FUNCTION_LINEAR                   30051
#define ID_MENU_GRADIENT_FUNCTION_STEPS                    30052
#define ID_MENU_GRADIENT_FUNCTION_LOGARITHMIC              30053
#define ID_MENU_GRADIENT_FUNCTION_EXPONENTIAL              30054
#define ID_MENU_PASTE_CONTROL_SCALE                        30055
#define ID_MENU_PASTE_3D_SCALE                             30056
#define ID_MENU_SAVE_TIMELINE_AS_TEXT                      30057
#define ID_MENU_SAVE_CURRENT_PLANE_AS_TEXT                 30058
#define ID_MENU_SAVE_ALL_PLANES_AS_TEXT                    30059
#define ID_MENU_NEWGROUP                                   30060
#define ID_MENU_REMOVE_GROUP                               30061
#define ID_MENU_VIEW_FUNCTION_LINE                         30062
#define ID_MENU_PASTE_DEFAULT_SPECIAL                      30063
#define ID_MENU_CODE_COLOR_2D                              30064
#define ID_MENU_GRADIENT_COLOR_2D                          30065
#define ID_MENU_LABELS_ALL                                 30066
#define ID_MENU_LABELS_SPACED                              30067
#define ID_MENU_LABELS_POWER2                              30068
#define ID_MENU_DRAWMODE_TIME_AVERAGE_NOT_ZERO             30069
#define ID_MENU_DRAWMODE_OBJECTS_AVERAGE_NOT_ZERO          30070
#define ID_MENU_DRAWMODE_BOTH_AVERAGE_NOT_ZERO             30071
#define ID_MENU_CLUSTERING                                 30072
#define ID_MENU_CUTTER                                     30073
#define ID_MENU_DIMEMAS                                    30074
#define ID_MENU_FOLDING                                    30075
#define ID_MENU_OBJECT_AXIS_CURRENT                        30076
#define ID_MENU_OBJECT_AXIS_ALL                            30077
#define ID_MENU_OBJECT_AXIS_ZERO                           30078
#define ID_MENU_OBJECT_AXIS_FIVE                           30079
#define ID_MENU_OBJECT_AXIS_TEN                            30080
#define ID_MENU_OBJECT_AXIS_TWENTYFIVE                     30081
#define ID_MENU_SAVE_TIMELINE_AS_CFG                       30082
#define ID_MENU_SAVE_HISTOGRAM_AS_CFG                      30083
#define ID_MENU_SAVE_IMAGE_LEGEND                          30084
#define ID_MENU_DRAWMODE_TIME_MODE                         30085
#define ID_MENU_DRAWMODE_OBJECTS_MODE                      30086
#define ID_MENU_DRAWMODE_BOTH_MODE                         30087
#define ID_MENU_SPECTRAL                                   30088
#define ID_MENU_PASTE_CONTROL_DIMENSIONS                   30089
#define ID_MENU_PUNCTUAL                                   30090
#define ID_MENU_PUNCTUAL_WINDOW                            30091
#define ID_MENU_TIMING                                     30092
#define ID_MENU_VIEW_FUSED_LINES                           30093
#define ID_MENU_DRAWMODE_TIME_ABSOLUTE_MAXIMUM             30094
#define ID_MENU_DRAWMODE_TIME_ABSOLUTE_MINIMUM_NOT_ZERO    30095
#define ID_MENU_DRAWMODE_OBJECTS_ABSOLUTE_MAXIMUM          30096
#define ID_MENU_DRAWMODE_OBJECTS_ABSOLUTE_MINIMUM_NOT_ZERO 30097
#define ID_MENU_DRAWMODE_BOTH_ABSOLUTE_MAXIMUM             30098
#define ID_MENU_DRAWMODE_BOTH_ABSOLUTE_MINIMUM_NOT_ZERO    30099
#define ID_MENU_NOT_NULL_GRADIENT_COLOR_2D                 30100
#define ID_MENU_RENAME                                     30101
#define ID_MENU_AUTO_CONTROL_SCALE_ZERO                    30102
#define ID_MENU_SEMANTIC_SCALE_MIN_AT_ZERO                 30103
#define ID_MENU_SYNC_REMOVE_ALL_GROUPS                     30104
#define ID_MENU_USER_COMMAND                               30105
#define ID_MENU_PASTE_CUSTOM_PALETTE                       30106
#define ID_MENU_PASTE_SEMANTIC_SORT                        30107
#define ID_MENU_PROFET                                     30108
#define ID_MENU_ALTERNATIVE_GRADIENT_COLOR                 30109
#define ID_MENU_ALTERNATIVE_GRADIENT_COLOR_2D              30110

#define ID_MENU_SYNC_GROUP_BASE                            31000
#define ID_MENU_SYNC_REMOVE_GROUP_BASE                     32000

template< class T >
class gPopUpMenu : public wxMenu
{

public:
gPopUpMenu() = delete;

gPopUpMenu( T *whichWindow );
virtual ~gPopUpMenu() = default;

void enablePaste( const std::string tag, bool checkPaste );
void enable( const std::string tag, bool enable );
void enable( const std::string tag );
void disable( const std::string tag );

void enableMenu( T *whichWindow );

static wxMultiChoiceDialog *createPasteSpecialDialog( wxArrayString& choices, T *whichWindow );
static RowsSelectionDialog *createRowSelectionDialog( T *whichWindow );
static std::string getOption( wxArrayString& choices, int position );

private:
T *window;

wxMenu *popUpMenuView;
wxMenu *popUpMenuColor;
wxMenu *popUpMenuPaste;
wxMenu *popUpMenuPasteFilter;
wxMenu *popUpMenuFitSemantic;
wxMenu *popUpMenuDrawMode;
wxMenu *popUpMenuDrawModeTime;
wxMenu *popUpMenuDrawModeObjects;
wxMenu *popUpMenuDrawModeBoth;
wxMenu *popUpMenuPixelSize;
wxMenu *popUpMenuGradientFunction;
wxMenu *popUpMenuSaveAsText;
wxMenu *popUpMenuColor2D;
wxMenu *popUpMenuLabels;
wxMenu *popUpMenuObjectAxis;
wxMenu *popUpMenuSave;
wxMenu *popUpMenuRun;
wxMenu *popUpMenuSync;
wxMenu *popUpMenuSyncRemove;

template< typename F >
wxMenuItem *buildItem( wxMenu *popUp,
const wxString &title,
wxItemKind itemType,
F function,
wxWindowID id,
bool checked = false );
};


template< class T >
template< typename F >
wxMenuItem *gPopUpMenu<T>::buildItem( wxMenu *popUp,
const wxString &title,
wxItemKind itemType,
F function,
wxWindowID id,
bool checked )
{
wxMenuItem *tmp;

tmp = new wxMenuItem( popUp, id, title, _( "" ), itemType );

popUp->Append( tmp );
if ( tmp->IsCheckable() )
tmp->Check( checked );

#ifdef _WIN32
Bind( wxEVT_COMMAND_MENU_SELECTED, function, window, id );
#else
popUp->Bind( wxEVT_COMMAND_MENU_SELECTED, function, window, id );
#endif

return tmp;
}


template< class T >
void gPopUpMenu<T>::enablePaste( const std::string tag, bool checkPaste )
{
if ( checkPaste )
{
Enable( FindItem( _( STR_PASTE ) ),
gPasteWindowProperties::getInstance()->isAllowed( window, STR_PASTE ));
Enable( FindItem( _( STR_PASTE_DEFAULT_SPECIAL ) ),
gPasteWindowProperties::getInstance()->isAllowed( window, STR_PASTE_DEFAULT_SPECIAL ));
Enable( FindItem( _( STR_PASTE_SPECIAL ) ),
gPasteWindowProperties::getInstance()->isAllowed( window, STR_PASTE_SPECIAL ));

}
Enable( FindItem( wxString::FromUTF8( tag.c_str() ) ), gPasteWindowProperties::getInstance()->isAllowed( window, tag ));
}


template< class T >
void gPopUpMenu<T>::enable( const std::string tag, bool enable )
{
Enable( FindItem( wxString::FromUTF8( tag.c_str() ) ), enable );
}


template< class T >
void gPopUpMenu<T>::enable( const std::string tag )
{
Enable( FindItem( wxString::FromUTF8( tag.c_str() ) ), true );
}


template< class T >
void gPopUpMenu<T>::disable( const std::string tag )
{
Enable( FindItem( wxString::FromUTF8( tag.c_str() ) ), false );
}


template< class T >
std::string gPopUpMenu<T>::getOption( wxArrayString& choices, int position )
{
if ( choices[ position ].Cmp( _( STR_FILTER_COMMS_XT ) ) == 0 )
return std::string( STR_FILTER_COMMS );
else if ( choices[ position ].Cmp( _( STR_FILTER_EVENTS_XT ) ) == 0 )
return std::string( STR_FILTER_EVENTS );
else  
return std::string( choices[ position ].mb_str() );
}
