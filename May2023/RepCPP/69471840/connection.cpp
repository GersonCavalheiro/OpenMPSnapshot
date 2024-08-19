

#include "wx/wxprec.h"

#ifdef __BORLANDC__
#pragma hdrstop
#endif

#ifndef WX_PRECOMP
#include "wx/wx.h"
#endif

#include "connection.h"
#include "wxparaverapp.h"

using namespace std;

wxConnectionBase *stServer::OnAcceptConnection( const wxString& topic )
{
if( topic.Lower() == wxT( "wxparaver" ) )
return new stConnection();
else
return nullptr;
}

bool stConnection::OnExecute( const wxString& WXUNUSED( topic ),
const void *data,
size_t WXUNUSED( size ),
wxIPCFormat WXUNUSED( format ) )
{
wxString dataStr( wxString::FromUTF8( (char *)data ));
static wxString tmpCommand;

if( dataStr.IsEmpty() )
{
if( wxparaverApp::mainWindow )
wxparaverApp::mainWindow->Raise();
}
else if( dataStr == wxT( "BEGIN" ) )
{
wxparaverApp::mainWindow->SetCanServeSignal( false );
tmpCommand.Clear();
}
else if( dataStr == wxT( "END" ) )
{
wxparaverApp::mainWindow->SetCanServeSignal( true );
wxCmdLineParser tmpLine( wxparaverApp::argumentsParseSyntax );
tmpLine.SetCmdLine( tmpCommand );
tmpLine.Parse();

wxGetApp().ParseCommandLine( tmpLine );
}
else
{
tmpCommand += dataStr + wxT( " " );
}

return true;
}
