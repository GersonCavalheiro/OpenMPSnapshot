


#pragma once


#include <vector>
#include <string>
#include <map>
#include <sstream>
#include "tracetypes.h"
#include "processmodelappl.h"

template< typename ApplOrderT = TApplOrder,
typename TaskOrderT = TTaskOrder,
typename ThreadOrderT = TThreadOrder,
typename NodeOrderT = TNodeOrder >
class ProcessModel
{

public:
struct ThreadLocation
{
ApplOrderT appl;
TaskOrderT task;
ThreadOrderT thread;

bool operator==( const ThreadLocation& other ) const
{
return appl   == other.appl &&
task   == other.task &&
thread == other.thread;
}
};

ProcessModel()
{
ready = false;
}

ProcessModel( std::istringstream& headerInfo, bool existResourceInfo );

~ProcessModel()
{}

bool operator<( const ProcessModel< ApplOrderT, TaskOrderT, ThreadOrderT, NodeOrderT >& other ) const;
bool operator==( const ProcessModel< ApplOrderT, TaskOrderT, ThreadOrderT, NodeOrderT >& other ) const;

bool isReady() const
{
return ready;
}

void setReady( bool newValue )
{
ready = newValue;
}

size_t size() const { return applications.size(); }
typename std::vector< ProcessModelAppl< ApplOrderT, TaskOrderT, ThreadOrderT, NodeOrderT > >::const_iterator cbegin() const { return applications.cbegin(); }
typename std::vector< ProcessModelAppl< ApplOrderT, TaskOrderT, ThreadOrderT, NodeOrderT > >::const_iterator cend() const { return applications.cend(); }

ApplOrderT totalApplications() const;

TaskOrderT totalTasks() const;
TaskOrderT getGlobalTask( const ApplOrderT& inAppl,
const TaskOrderT& inTask ) const;
void getTaskLocation( TaskOrderT globalTask,
ApplOrderT& inAppl,
TaskOrderT& inTask ) const;
TaskOrderT getFirstTask( ApplOrderT inAppl ) const;
TaskOrderT getLastTask( ApplOrderT inAppl ) const;

ThreadOrderT totalThreads() const;
ThreadOrderT totalThreads( ApplOrderT whichAppl ) const;
ThreadOrderT getGlobalThread( const ApplOrderT& inAppl,
const TaskOrderT& inTask,
const ThreadOrderT& inThread ) const;
void getThreadLocation( ThreadOrderT globalThread,
ApplOrderT& inAppl,
TaskOrderT& inTask,
ThreadOrderT& inThread ) const;
ThreadOrderT getFirstThread( ApplOrderT inAppl, TaskOrderT inTask ) const;
ThreadOrderT getLastThread( ApplOrderT inAppl, TaskOrderT inTask )const;

void getThreadsPerNode( NodeOrderT inNode, std::vector<ThreadOrderT>& onVector ) const;

bool isValidThread( ThreadOrderT whichThread ) const;
bool isValidThread( ApplOrderT whichAppl,
TaskOrderT whichTask,
ThreadOrderT whichThread ) const;
bool isValidThread( ApplOrderT whichAppl,
TaskOrderT whichTask,
ThreadOrderT whichThread,
NodeOrderT whichNode ) const;
bool isValidTask( TaskOrderT whichTask ) const;
bool isValidTask( ApplOrderT whichAppl,
TaskOrderT whichTask ) const;
bool isValidAppl( ApplOrderT whichAppl ) const;

ApplOrderT   addApplication();
TaskOrderT   addTask( ApplOrderT whichAppl );
ThreadOrderT addThread( ApplOrderT whichAppl, TaskOrderT whichTask, NodeOrderT execNode );
ThreadOrderT addApplTaskThread( const ThreadLocation& whichLocation, NodeOrderT execNode = 0 );

TNodeOrder getNode( TThreadOrder whichThread ) const
{ 
return applications[ threads[ whichThread ].appl ].tasks[ threads[ whichThread ].task ].threads[ threads[ whichThread ].thread ].nodeExecution; 
}

protected:

struct TaskLocation
{
ApplOrderT appl;
TaskOrderT task;

bool operator==( const TaskLocation& other ) const
{
return appl == other.appl &&
task == other.task;
}
};

std::vector< ThreadLocation > threads;
std::vector< TaskLocation > tasks;
typename std::vector< ProcessModelAppl< ApplOrderT, TaskOrderT, ThreadOrderT, NodeOrderT > > applications;
std::map< NodeOrderT, std::vector< ThreadOrderT > > threadsPerNode;

bool ready;

private:

};

template< typename ProcessModelT >
void dumpProcessModelToFile( ProcessModelT processModel, std::fstream& file, bool existResourceInfo );

#include "processmodel.cpp"
