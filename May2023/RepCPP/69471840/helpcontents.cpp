

#include "wx/wxprec.h"

#ifdef __BORLANDC__
#pragma hdrstop
#endif

#ifndef WX_PRECOMP
#include "wx/wx.h"
#endif

#include <wx/file.h>
#include <wx/choicdlg.h>
#include "wx/filesys.h"
#include <wx/mimetype.h>

#include <string>

#include "helpcontents.h"
#include "paravermain.h"
#include "tutorialsdownload.h"
#include "wxparaverapp.h"

#include "../icons/index.xpm"
#include "../icons/arrow_left.xpm"
#include "../icons/arrow_right.xpm"
#include "../icons/download.xpm"



IMPLEMENT_DYNAMIC_CLASS( HelpContents, wxDialog )
IMPLEMENT_DYNAMIC_CLASS( TutorialsBrowser, HelpContents )




BEGIN_EVENT_TABLE( HelpContents, wxDialog )

EVT_HTML_LINK_CLICKED( ID_HTMLWINDOW, HelpContents::OnHtmlwindowLinkClicked )
EVT_BUTTON( ID_BUTTON_INDEX, HelpContents::OnButtonIndexClick )
EVT_BUTTON( ID_BITMAPBUTTON_BACK, HelpContents::OnBitmapbuttonBackClick )
EVT_UPDATE_UI( ID_BITMAPBUTTON_BACK, HelpContents::OnBitmapbuttonBackUpdate )
EVT_BUTTON( ID_BITMAPBUTTON_FORWARD, HelpContents::OnBitmapbuttonForwardClick )
EVT_UPDATE_UI( ID_BITMAPBUTTON_FORWARD, HelpContents::OnBitmapbuttonForwardUpdate )
EVT_BUTTON( ID_BUTTON_CLOSE, HelpContents::OnButtonCloseClick )

END_EVENT_TABLE()


BEGIN_EVENT_TABLE( TutorialsBrowser, wxDialog )

EVT_HTML_LINK_CLICKED( ID_HTMLWINDOW, TutorialsBrowser::OnHtmlwindowLinkClicked )
EVT_BUTTON( ID_BUTTON_INDEX, TutorialsBrowser::OnButtonIndexClick )
EVT_BUTTON( ID_BITMAPBUTTON_BACK, TutorialsBrowser::OnBitmapbuttonBackClick )
EVT_UPDATE_UI( ID_BITMAPBUTTON_BACK, TutorialsBrowser::OnBitmapbuttonBackUpdate )
EVT_BUTTON( ID_BITMAPBUTTON_FORWARD, TutorialsBrowser::OnBitmapbuttonForwardClick )
EVT_UPDATE_UI( ID_BITMAPBUTTON_FORWARD, TutorialsBrowser::OnBitmapbuttonForwardUpdate )
EVT_BUTTON( ID_BUTTON_CLOSE, TutorialsBrowser::OnButtonCloseClick )
EVT_BUTTON( ID_BITMAPBUTTON_DOWNLOAD, TutorialsBrowser::OnButtonDownloadClick )

END_EVENT_TABLE()



HelpContents::HelpContents()
{
Init();
}


HelpContents* HelpContents::createObject( TContents whichObject,
wxWindow* parent,
const wxString& whichHelpContentsRoot,
const bool whichLookForContents,
wxWindowID id,
const wxString& caption,
const wxPoint& pos,
const wxSize& size,
long style)
{
HelpContents* item = nullptr;
switch( whichObject )
{
case TContents::HELP:
item = new HelpContents( parent,
whichHelpContentsRoot,
whichLookForContents,
id, SYMBOL_HELPCONTENTS_TITLE, 
pos, size, style );
break;
case TContents::TUTORIAL: 
item = new TutorialsBrowser( parent,
whichHelpContentsRoot,
id, SYMBOL_TUTORIALSBROWSER_TITLE, 
pos, 
size, 
style );
break;
default:
break;
}
if ( item != nullptr && whichLookForContents )
item->buildIndex();
return item;
}


HelpContents::HelpContents( wxWindow* parent,
const wxString& whichHelpContentsRoot,
const bool whichLookForContents,
wxWindowID id,
const wxString& caption,
const wxPoint& pos,
const wxSize& size,
long style ) :
helpContentsRoot( whichHelpContentsRoot ),
lookForContents( whichLookForContents ),
dialogCaption( caption )
{
Init();
Create(parent, id, caption, pos, size, style);

buttonDownloadTutorial->Hide();
staticLineDownloadSeparator->Hide();
}



bool HelpContents::Create( wxWindow* parent,
wxWindowID id,
const wxString& caption,
const wxPoint& pos,
const wxSize& size,
long style )
{
SetExtraStyle(wxWS_EX_BLOCK_EVENTS);
wxDialog::Create( parent, id, caption, pos, size, style );

CreateControls();
Centre();

return true;
}


HelpContents::~HelpContents()
{
}


void HelpContents::Init()
{
currentHelpContentsDir = wxT("");
indexFileName = wxT( "" );
subindexLink = wxT( "" );
}


const wxString HelpContents::appendIndexHtmlToURL( const wxString& path )
{
wxString htmlIndex;
wxFileName index( path + wxFileName::GetPathSeparator() + wxT("index.html") );

if ( index.FileExists() )
{
htmlIndex = index.GetFullPath();
}

return htmlIndex;
}


const wxString HelpContents::getTitle( int numTutorial, const wxString& path )
{
wxString helpContentsTitle;

wxHtmlWindow auxHtml( this );
auxHtml.LoadPage( path  + wxFileName::GetPathSeparator() + wxT("index.html") );
helpContentsTitle = auxHtml.GetOpenedPageTitle();
if ( helpContentsTitle.empty() || helpContentsTitle == wxT("index.html") )
{
helpContentsTitle = wxT("Section");
helpContentsTitle << numTutorial;
}

return helpContentsTitle;
}


void HelpContents::appendHelpContents( const wxString& title,
const wxString& path,
wxString& htmlDoc )
{
htmlDoc += wxT("<LI><P><A HREF=\"") + path + wxT("\">") + title + wxT("</A></P></LI>");
}


void HelpContents::helpMessage( wxString& htmlDoc )
{
htmlDoc += wxT("<P><H3>No Help Contents found!?</H3></P>");
htmlDoc += wxT("<P>Please check that a <B>root directory</B> to Help Contents exists.</P>");
htmlDoc += wxT("<P>The current one is ");
htmlDoc += GetHelpContentsRoot();
htmlDoc += wxT("</P>");
htmlDoc += wxT("<P>If missing, try to download a newer wxparaver version, or please contact us at paraver@bsc.es.</P>");
}


void HelpContents::linkToWebPage( wxString& htmlDoc )
{
}


bool HelpContents::helpContentsFound( wxArrayString & contentsList )
{
wxFileName helpContentsGlobalPath( GetHelpContentsRoot() );

wxString currentDir = wxFindFirstFile(
helpContentsGlobalPath.GetLongPath() + wxFileName::GetPathSeparator() + wxT("*"),
wxDIR );
while( !currentDir.empty() )
{
if ( appendIndexHtmlToURL( currentDir ) != wxT("")  )
{
contentsList.Add( currentDir );
}

currentDir = wxFindNextFile();
}

return ( contentsList.GetCount() > (size_t)0 );
}


void HelpContents::buildIndexTemplate( wxString title, wxString filePrefix )
{
wxString contentsHtmlIndex;
wxString contentsList = wxT("<UL>");

contentsHtmlIndex += wxT("<!DOCTYPE HTML PUBLIC \"-
contentsHtmlIndex += wxT("<HTML>");
contentsHtmlIndex += wxT("<HEAD>");
contentsHtmlIndex += wxT("<meta http-equiv=\"Content-Type\" content=\"text/html; charset=us-ascii\" />");
contentsHtmlIndex += wxT("<TITLE>" ) + title + wxT( "</TITLE>");
contentsHtmlIndex += wxT("</HEAD>");
contentsHtmlIndex += wxT("<BODY>");

contentsHtmlIndex += wxT("<P ALIGN=LEFT><A HREF=\"https:

wxArrayString contents;
if ( helpContentsFound( contents ) )
{
contents.Sort();
int numSections = int( contents.GetCount() );

for( int i = 0; i < numSections; ++i )
{
appendHelpContents( getTitle( numSections, contents[ i ] ),
appendIndexHtmlToURL( contents[ i ] ),
contentsList );
}

contentsList += wxT("</UL>");

contentsHtmlIndex += wxT("<P><H3><B>Index</B></H3></P>");
contentsHtmlIndex += contentsList;
}
else
{
helpMessage( contentsHtmlIndex );
linkToWebPage( contentsHtmlIndex );
}

contentsHtmlIndex += wxT("</BODY></HTML>");

indexFileName =
wxString::FromUTF8( paraverMain::myParaverMain->GetParaverConfig()->getParaverConfigDir().c_str() ) +
wxString( wxFileName::GetPathSeparator() ) +
filePrefix +
wxT( "_index.html" );

wxFile indexFile( indexFileName, wxFile::write );
if ( indexFile.IsOpened() )
{
indexFile.Write( contentsHtmlIndex );
indexFile.Close();
htmlWindow->LoadPage( indexFileName );
}
else
{
htmlWindow->SetPage( contentsHtmlIndex );
}
}


void HelpContents::buildIndex()
{
buildIndexTemplate( wxString( wxT( "Help Contents" ) ), wxString( wxT( "help_contents" ) ) );
}



void HelpContents::CreateControls()
{
HelpContents* itemDialog1 = this;

wxBoxSizer* itemBoxSizer2 = new wxBoxSizer(wxVERTICAL);
itemDialog1->SetSizer(itemBoxSizer2);

htmlWindow = new wxHtmlWindow( itemDialog1, ID_HTMLWINDOW, wxDefaultPosition, wxDefaultSize, wxHW_SCROLLBAR_AUTO|wxSUNKEN_BORDER|wxHSCROLL|wxVSCROLL );
itemBoxSizer2->Add(htmlWindow, 1, wxGROW|wxALL, 5);

wxBoxSizer* itemBoxSizer4 = new wxBoxSizer(wxHORIZONTAL);
itemBoxSizer2->Add(itemBoxSizer4, 0, wxGROW|wxALL, 5);

buttonIndex = new wxBitmapButton( itemDialog1, ID_BUTTON_INDEX, itemDialog1->GetBitmapResource(wxT("icons/index.xpm")), wxDefaultPosition, wxDefaultSize, wxBU_AUTODRAW );
if (HelpContents::ShowToolTips())
buttonIndex->SetToolTip(_("Main index page"));
itemBoxSizer4->Add(buttonIndex, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5);

buttonHistoryBack = new wxBitmapButton( itemDialog1, ID_BITMAPBUTTON_BACK, itemDialog1->GetBitmapResource(wxT("icons/arrow_left.xpm")), wxDefaultPosition, wxDefaultSize, wxBU_AUTODRAW );
if (HelpContents::ShowToolTips())
buttonHistoryBack->SetToolTip(_("Previous page"));
itemBoxSizer4->Add(buttonHistoryBack, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5);

buttonHistoryForward = new wxBitmapButton( itemDialog1, ID_BITMAPBUTTON_FORWARD, itemDialog1->GetBitmapResource(wxT("icons/arrow_right.xpm")), wxDefaultPosition, wxDefaultSize, wxBU_AUTODRAW );
if (HelpContents::ShowToolTips())
buttonHistoryForward->SetToolTip(_("Next page"));
itemBoxSizer4->Add(buttonHistoryForward, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5);

staticLineDownloadSeparator = new wxStaticLine( itemDialog1, wxID_STATIC, wxDefaultPosition, wxDefaultSize, wxLI_VERTICAL );
itemBoxSizer4->Add(staticLineDownloadSeparator, 0, wxGROW|wxALL, 5);

buttonDownloadTutorial = new wxBitmapButton( itemDialog1, ID_BITMAPBUTTON_DOWNLOAD, itemDialog1->GetBitmapResource(wxT("icons/download.xpm")), wxDefaultPosition, wxDefaultSize, wxBU_AUTODRAW );
if (HelpContents::ShowToolTips())
buttonDownloadTutorial->SetToolTip(_("Download and install tutorials"));
itemBoxSizer4->Add(buttonDownloadTutorial, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5);

itemBoxSizer4->Add(5, 5, 1, wxALIGN_CENTER_VERTICAL|wxALL, 5);

wxButton* itemButton9 = new wxButton( itemDialog1, ID_BUTTON_CLOSE, _("Close"), wxDefaultPosition, wxDefaultSize, 0 );
itemBoxSizer4->Add(itemButton9, 0, wxALIGN_CENTER_VERTICAL|wxALL, 5);

}



bool HelpContents::ShowToolTips()
{
return true;
}



wxBitmap HelpContents::GetBitmapResource( const wxString& name )
{
wxUnusedVar(name);
if (name == wxT("icons/index.xpm"))
{
wxBitmap bitmap(text_list_bullets_xpm);
return bitmap;
}
else if (name == wxT("icons/arrow_left.xpm"))
{
wxBitmap bitmap(arrow_left_xpm);
return bitmap;
}
else if (name == wxT("icons/arrow_right.xpm"))
{
wxBitmap bitmap(arrow_right_xpm);
return bitmap;
}
else if (name == wxT("icons/download.xpm"))
{
wxBitmap bitmap(download_xpm);
return bitmap;
}
return wxNullBitmap;
}



wxIcon HelpContents::GetIconResource( const wxString& name )
{
wxUnusedVar(name);
return wxNullIcon;
}


std::string HelpContents::getCurrentHelpContentsFullPath()
{
std::string fullPath = GetHelpContentsRootStr();
fullPath += wxString( wxFileName::GetPathSeparator() ).mb_str();
fullPath += currentHelpContentsDir.mb_str();
fullPath += wxString( wxFileName::GetPathSeparator() ).mb_str();

return fullPath;
}


std::string HelpContents::getHrefFullPath( wxHtmlLinkEvent &event )
{
std::string hrefFullPath = getCurrentHelpContentsFullPath();
hrefFullPath += std::string( event.GetLinkInfo().GetHref().mb_str() );

return hrefFullPath;
}


bool HelpContents::matchHrefExtension( wxHtmlLinkEvent &event, const wxString extension )
{
return ( event.GetLinkInfo().GetHref().Right( extension.Len() ).Cmp( extension ) == 0 );
}


bool HelpContents::isHtmlDoc( const wxString& whichPath )
{
bool isHtml = false;

wxFileName tmpPath( whichPath );

if ( ( tmpPath.GetExt().Cmp( wxT("html") ) == 0 ) ||
( tmpPath.GetExt().Cmp( wxT("htm") ) == 0  ) ||
( tmpPath.GetExt().Cmp( wxT("HTML") ) == 0 ) ||
( tmpPath.GetExt().Cmp( wxT("HTM") ) == 0 ) )
{

isHtml = ( tmpPath.GetFullName().Find( wxT("#") ) == wxNOT_FOUND ) &&
tmpPath.FileExists();
}
return isHtml;
}


bool HelpContents::isHtmlReferenceInDoc( const wxString& whichPath )
{
bool isHtmlReference = false;

if ( ( !isHtmlDoc( whichPath ) ) &&
( ( whichPath.Find( wxT(".html#") ) !=  wxNOT_FOUND ) ||
( whichPath.Find( wxT(".htm#") )  !=  wxNOT_FOUND ) ||
( whichPath.Find( wxT(".HTML#") ) !=  wxNOT_FOUND ) ||
( whichPath.Find( wxT(".HTM#") )  !=  wxNOT_FOUND ) ) )
{
bool fromEnd = true;
size_t untilHash = whichPath.Find( wxChar('#'), fromEnd );
size_t firstPos = 0;
wxString tmpCandidate = whichPath.Mid( firstPos, untilHash );
isHtmlReference = wxFileName( tmpCandidate ).FileExists(); 
}

return isHtmlReference;
}


void HelpContents::LoadHtml( const wxString& htmlFile )
{
if ( paraverMain::myParaverMain->GetParaverConfig()->getGlobalHelpContentsUsesBrowser() && 
dialogCaption == SYMBOL_HELPCONTENTS_TITLE )
{
if( !launchBrowser( htmlFile ) )
htmlWindow->LoadPage( htmlFile );
else if ( IsModal() )
EndModal( wxID_OK );
else
Close();
}
else
htmlWindow->LoadPage( htmlFile );
}


bool HelpContents::SetHelpContents( const wxString& whichPath )
{
bool htmlFound = false;
wxFileName candidate( whichPath );
if ( isHtmlDoc( whichPath ) )
{
SetHelpContentsRoot( candidate.GetPathWithSep() );
LoadHtml( indexFileName );
htmlFound = true;
}
else if ( isHtmlReferenceInDoc( whichPath ) )
{
SetHelpContentsRoot( candidate.GetPathWithSep() );
LoadHtml( indexFileName );
htmlFound = true;
}
else if ( candidate.IsDirReadable() )
{
wxString tmpTutorial = appendIndexHtmlToURL( candidate.GetPathWithSep() );
if ( !tmpTutorial.IsEmpty() )
{
LoadHtml( indexFileName );
htmlFound = true;
}
}

return htmlFound;
}


bool HelpContents::SetHelpContentsRoot( const wxString& whichRoot )
{
bool changedRoot = false;

if ( wxFileName::IsDirReadable( whichRoot ) )
{
helpContentsRoot = whichRoot;
changedRoot = true;
}

return changedRoot;
}


bool HelpContents::SetHelpContentsRoot( const std::string& whichRoot )
{
return SetHelpContentsRoot( wxString::FromUTF8( whichRoot.c_str() ) );
}


const wxString HelpContents::GetHelpContentsRoot()
{
return helpContentsRoot;
}


const std::string HelpContents::GetHelpContentsRootStr()
{
return std::string( GetHelpContentsRoot().mb_str() );
}


bool HelpContents::DetectHelpContentsIndexInPath( const wxString& whichPath )
{
bool indexFound = false;
subindexLink = wxT( "" );
if ( whichPath[0] == '#' )
subindexLink = whichPath;
wxString anyHelpContentsPath = GetHelpContentsRoot();
if ( anyHelpContentsPath[ anyHelpContentsPath.Len() - 1 ] != wxString( wxFileName::GetPathSeparator() ))
{
anyHelpContentsPath += wxString( wxFileName::GetPathSeparator() );
}

wxFileName anyHelpContentsDir( anyHelpContentsPath );

size_t dirsDepthHelpContents = anyHelpContentsDir.GetDirCount();

wxFileName currentLink( whichPath );
size_t dirsDepthCurrentLink = currentLink.GetDirCount();

if (( dirsDepthCurrentLink == dirsDepthHelpContents + 1 ) && 
( ( currentLink.GetFullName().Cmp( wxT("index.html") ) == 0 ) ||
isHtmlReferenceInDoc( whichPath ) ) )
{
wxArrayString dirs = currentLink.GetDirs();
currentHelpContentsDir = dirs[ dirsDepthCurrentLink - 1 ];
indexFound = true;
}

return indexFound;
}



void HelpContents::OnHtmlwindowLinkClicked( wxHtmlLinkEvent& event )
{
if ( event.GetLinkInfo().GetHref().SubString( 0, 4 ) ==  wxT( "https" ) ||
event.GetLinkInfo().GetHref().SubString( 0, 5 ) ==  wxT( "mailto" ) )
{
if( !launchBrowser( event.GetLinkInfo().GetHref() ) )
{
wxMessageDialog message( this, wxT( "Unable to find/open default browser." ), wxT( "Warning" ), wxOK );
message.ShowModal();
}
}
else
{
DetectHelpContentsIndexInPath( event.GetLinkInfo().GetHref() );

event.Skip();
}
}




void HelpContents::OnButtonCloseClick( wxCommandEvent& event )
{
if ( IsModal() )
{
EndModal( wxID_OK );
}
else
{
Close();
}
}




void HelpContents::OnButtonIndexClick( wxCommandEvent& event )
{
buildIndex();
currentHelpContentsDir = wxT( "" ); 
}




void HelpContents::OnBitmapbuttonBackClick( wxCommandEvent& event )
{
htmlWindow->HistoryBack();
}




void HelpContents::OnBitmapbuttonBackUpdate( wxUpdateUIEvent& event )
{
buttonHistoryBack->Enable( htmlWindow->HistoryCanBack() );
}




void HelpContents::OnBitmapbuttonForwardClick( wxCommandEvent& event )
{
htmlWindow->HistoryForward();
}




void HelpContents::OnBitmapbuttonForwardUpdate( wxUpdateUIEvent& event )
{
buttonHistoryForward->Enable( htmlWindow->HistoryCanForward() );
}


TutorialsBrowser::TutorialsBrowser( wxWindow* parent,
const wxString& whichHelpContentsRoot,
wxWindowID id,
const wxString& caption,
const wxPoint& pos,
const wxSize& size,
long style) :
HelpContents( parent, whichHelpContentsRoot, true, id, caption, pos, size, style )
{

buttonDownloadTutorial->Show();
staticLineDownloadSeparator->Show();
}


TutorialsBrowser::~TutorialsBrowser()
{
}

const wxString TutorialsBrowser::getTitle( int numTutorial, const wxString& path )
{
wxString helpContentsTitle;

wxHtmlWindow auxHtml( this );
auxHtml.LoadPage( path  + wxFileName::GetPathSeparator() + wxT("index.html") );
helpContentsTitle = auxHtml.GetOpenedPageTitle();
if ( helpContentsTitle.empty() || helpContentsTitle == wxT("index.html") )
{
std::string auxStrTitleFileName(
( path + wxFileName::GetPathSeparator() + wxT("tutorial_title") ).mb_str() );
std::string auxLine;

std::ifstream titleFile;
titleFile.open( auxStrTitleFileName.c_str() );
if ( titleFile.good() )
{
std::getline( titleFile, auxLine );

if ( auxLine.size() > 0 )
{
helpContentsTitle = wxString::FromUTF8( auxLine.c_str() );
}
else
{
helpContentsTitle = wxT("Tutorial ");
helpContentsTitle << numTutorial;
}
}
else
{
helpContentsTitle = wxT("Tutorial ");
helpContentsTitle << numTutorial;
}

titleFile.close();
}

return helpContentsTitle;
}


void TutorialsBrowser::linkToWebPage( wxString& htmlDoc )
{
htmlDoc += wxT( "<P><H3>Latest tutorials</H3></P>" );

htmlDoc += wxT( "<P>Find them available at <A HREF=\"https:
htmlDoc += wxT( "<UL>" );
htmlDoc += wxT( "<LI>As single <A HREF=\"https:
htmlDoc += wxT( "<LI>As single <A HREF=\"https:
htmlDoc += wxT( "</UL>" );
}


void TutorialsBrowser::helpMessage( wxString& htmlDoc )
{
htmlDoc += wxT( "<P><H2>No tutorials found!?</H2></P>" );
htmlDoc += wxT( "<P><H3>Install using the download dialog</H3></P>" );
htmlDoc += wxT( "<P>You can automatically download and install any of the available tutorials by clicking the <B>\"Download and Install\"</B> button.</P>" );
htmlDoc += wxT( "<P>Just check in the desired tutorials and press the <B>\"OK\"</B> button.</P>" );

htmlDoc += wxT( "<P><H3>Manual installation</H3></P>" );
htmlDoc += wxT( "<P>Please check that a <B>root directory</B> to tutorials is properly defined:</P>" );

htmlDoc += wxT( "<OL type=\"1\">" );
htmlDoc += wxT( "<LI>Open the <A HREF=\"init_preferences\"><I>Preferences Window</I></A>.</LI>" );
htmlDoc += wxT( "<LI>Select <I>Global</I> tab.</LI>" );
htmlDoc += wxT( "<LI>In the <I>Default directories</I> box, change the <I>Tutorials root</I> directory." );
htmlDoc += wxT( "<LI>Save your new settings by clicking the <I>Ok</I> button in the <I>Preferences Window</I>.</LI>" );
htmlDoc += wxT( "<LI>After that, we will automatically refresh the tutorials list.</LI>" );
htmlDoc += wxT( "<LI>If nothing happens, come back here and press the <I>Index</I> button (the first one at the bottom-left) " );
htmlDoc += wxT( "to rebuild the tutorials list.</OL>" );

htmlDoc += wxT( "<P>If the button <I>Index</I> doesn't seem to work (you're still reading this help!), please verify that:</P>" );

htmlDoc += wxT( "<UL>" );
htmlDoc += wxT( "<LI>Every tutorial is <B>uncompressed</B>.</LI>" );
htmlDoc += wxT( "<LI>Every tutorial is inside its own <B>subdirectory</B>.</LI>" );
htmlDoc += wxT( "<LI>These subdirectories (or tutorials) are copied/linked into the root directory that " );
htmlDoc += wxT( "you have selected before (i.e: /home/myuser/mytutorials/tut1/, /home/myuser/mytutorials/tut2/, etc).</LI>" );
htmlDoc += wxT( "<LI>Every tutorial has a main <B>index.html</B> (i.e: /home/myuser/mytutorials/tut1/index.html ).</LI>" );
htmlDoc += wxT( "</UL>" );

htmlDoc += wxT( "<P>If you still get this help after checking these steps again, please contact us at " );
htmlDoc += wxT( "<A HREF=\"mailto:paraver@bsc.es\">paraver@bsc.es</A>.</P>" );
}




void TutorialsBrowser::OnHtmlwindowLinkClicked( wxHtmlLinkEvent& event )
{
wxString auxCommand;

if ( event.GetLinkInfo().GetHref().StartsWith( wxT("init_command:"), &auxCommand ) )
{

}
else if ( event.GetLinkInfo().GetHref().Cmp( wxT("init_preferences") ) == 0 )
{
std::string oldTutorialsPath = GetHelpContentsRootStr();

paraverMain::myParaverMain->ShowPreferences();

std::string newTutorialsPath =
paraverMain::myParaverMain->GetParaverConfig()->getGlobalTutorialsPath();

SetHelpContentsRoot( newTutorialsPath );

if ( newTutorialsPath.compare( oldTutorialsPath ) != 0 )
{
buildIndex();
}
}
else if ( event.GetLinkInfo().GetHref().SubString( 0, 4 ) ==  wxT( "https" ) ||
event.GetLinkInfo().GetHref().SubString( 0, 5 ) ==  wxT( "mailto" ) )
{
if( !launchBrowser( event.GetLinkInfo().GetHref() ) )
{
wxMessageDialog message( this, wxT( "Unable to find/open default browser." ), wxT( "Warning" ), wxOK );
message.ShowModal();
}
}

else if ( matchHrefExtension( event, wxT(".prv") ) ||
matchHrefExtension( event, wxT(".prv.gz")))
{
paraverMain::myParaverMain->DoLoadTrace( getHrefFullPath( event ) );
}
else if ( matchHrefExtension( event, wxT(".cfg")))
{
if ( paraverMain::myParaverMain->GetLoadedTraces().size() > 0 )
{
paraverMain::myParaverMain->DoLoadCFG( getHrefFullPath( event )  );
}
else
{
wxMessageDialog message( this, wxT("No trace loaded."), wxT( "Warning" ), wxOK );
message.ShowModal();
}
}
else if ( matchHrefExtension( event, wxT(".xml")))
{
std::string traceName;
if ( paraverMain::myParaverMain->GetLoadedTraces().size() > 0 )
{
traceName = paraverMain::myParaverMain->GetLoadedTraces().front()->getFileName();
}
else
{
traceName = getCurrentHelpContentsFullPath();
}

bool loadTrace = true;
std::string strXmlFile = getHrefFullPath( event );

paraverMain::myParaverMain->ShowCutTraceWindow( traceName, loadTrace, strXmlFile );
}
else
{
if ( DetectHelpContentsIndexInPath( event.GetLinkInfo().GetHref() ) )
{
}

event.Skip();
}
}


void TutorialsBrowser::buildIndex()
{
buildIndexTemplate( wxString( wxT( "Tutorials" ) ), wxString( wxT( "tutorials" ) ) );
}


void TutorialsBrowser::OnButtonDownloadClick(  wxCommandEvent& event )
{
vector<TutorialData> tutorialsData;

tutorialsData = TutorialsDownload::getInstance()->getTutorialsList();
if ( tutorialsData.empty() )
return;

wxArrayString tutorialChoices;

for( auto it : tutorialsData )
tutorialChoices.Add( wxString::FromUTF8( it.getName().c_str() ) );

wxMultiChoiceDialog selDialog( this, wxT( "Select tutorials to download and install:" ), wxT( "Tutorials download" ), tutorialChoices );
if( selDialog.ShowModal() == wxID_OK )
{
wxArrayInt selection = selDialog.GetSelections();
vector<PRV_UINT16> tutorialsIndex;

for( size_t i = 0; i < selection.GetCount(); ++i )
tutorialsIndex.push_back( tutorialsData[ selection.Item( i ) ].getId() );

if( tutorialsIndex.size() > 0 )
TutorialsDownload::getInstance()->downloadInstall( tutorialsIndex );

buildIndex();
}
}
