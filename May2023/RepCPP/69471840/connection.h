


#pragma once


#include "wx/ipc.h"

class stServer: public wxServer
{
public:
wxConnectionBase *OnAcceptConnection( const wxString& topic );
};

class stConnection: public wxConnection
{
public:
stConnection() {}
~stConnection() {}

bool OnExecute( const wxString& topic, const void *data, size_t size, wxIPCFormat format );
};

class stClient: public wxClient
{
public:
stClient() {}
wxConnectionBase *OnMakeConnection() { return new stConnection; }
};

