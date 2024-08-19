

#pragma once


#include <string>
#include <vector>
#include <map>
#include <set>
#include "kernelconnection.h"
#include "paraverkerneltypes.h"
#include "workspace.h"

enum class TWorkspaceSet { ALL = 0, DISTRIBUTED, USER_DEFINED };

class WorkspaceManager
{
public:

static WorkspaceManager *getInstance( KernelConnection *whichKernel );

~WorkspaceManager();

void clear();

bool existWorkspace( std::string name, TWorkspaceSet whichSet ) const;
std::vector<std::string> getWorkspaces( TWorkspaceSet whichSet ) const;
void getMergedWorkspaces( const std::set<TState>& loadedStates,
const std::set<TEventType>& loadedTypes,
std::vector<std::string>& onWorkspaceVector,
size_t& userDefined );
Workspace& getWorkspace( std::string whichName, TWorkspaceSet whichSet  );
void addWorkspace( std::string whichName );
void addWorkspace( Workspace& whichWorkspace );
void loadXML();
void saveXML();

template< class Archive >
void serialize( Archive & ar, const unsigned int version )
{
ar & boost::serialization::make_nvp( "workspaces", *serializeBufferWorkspaces  );
ar & boost::serialization::make_nvp( "workspacesOrder", *serializeBufferWorkspacesOrder );
}


protected:

WorkspaceManager( KernelConnection *whichKernel );

private:
static WorkspaceManager *instance;

KernelConnection *myKernel;

std::map<std::string, Workspace> distWorkspaces;
std::vector<std::string> distWorkspacesOrder;
std::map<std::string, Workspace> userWorkspaces;
std::vector<std::string> userWorkspacesOrder;

std::map<std::string, Workspace> *serializeBufferWorkspaces;
std::vector<std::string> *serializeBufferWorkspacesOrder;
};

BOOST_CLASS_VERSION( WorkspaceManager, 0)


