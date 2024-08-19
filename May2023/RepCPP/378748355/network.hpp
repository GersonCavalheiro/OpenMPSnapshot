#pragma once

#include "component.hpp"
#include "events.hpp"
#include "gtaquat.hpp"
#include "types.hpp"
#include "values.hpp"
#include <array>
#include <cassert>
#include <string>
#include <vector>

#if OMP_BUILD_PLATFORM == OMP_WINDOWS
#include <Winsock2.h>
#include <unknwn.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#elif OMP_BUILD_PLATFORM == OMP_UNIX
#include <arpa/inet.h>
#endif

constexpr int INVALID_PACKET_ID = -1;

struct IPlayer;
struct PeerNetworkData;

enum PeerDisconnectReason
{
PeerDisconnectReason_Timeout,
PeerDisconnectReason_Quit,
PeerDisconnectReason_Kicked,
_PeerDisconnectReason_Custom,
PeerDisconnectReason_ModeEnd
};

enum OrderingChannel
{
OrderingChannel_Internal,
OrderingChannel_SyncPacket,
OrderingChannel_SyncRPC,
OrderingChannel_Unordered,
OrderingChannel_Reliable,
OrderingChannel_DownloadRequest
};

enum ENetworkType
{
ENetworkType_RakNetLegacy,
ENetworkType_ENet,

ENetworkType_End
};

enum ENetworkBitStreamReset
{
BSResetRead = (1 << 0), 
BSResetWrite = (1 << 1), 
BSReset = BSResetRead | BSResetWrite 
};



enum NewConnectionResult
{
NewConnectionResult_Ignore, 
NewConnectionResult_VersionMismatch,
NewConnectionResult_BadName,
NewConnectionResult_BadMod,
NewConnectionResult_NoPlayerSlot,
NewConnectionResult_Success
};

enum class ClientVersion : uint8_t
{
ClientVersion_SAMP_037,
ClientVersion_SAMP_03DL,
ClientVersion_openmp
};

struct PeerRequestParams
{
ClientVersion version;
StringView versionName;
bool bot;
StringView name;
StringView serial;
bool isUsingOfficialClient;
};

struct NetworkStats
{
unsigned connectionStartTime;
unsigned messageSendBuffer;
unsigned messagesSent;
unsigned totalBytesSent;
unsigned acknowlegementsSent;
unsigned acknowlegementsPending;
unsigned messagesOnResendQueue;
unsigned messageResends;
unsigned messagesTotalBytesResent;
float packetloss;
unsigned messagesReceived;
unsigned messagesReceivedPerSecond;
unsigned bytesReceived;
unsigned acknowlegementsReceived;
unsigned duplicateAcknowlegementsReceived;
double bitsPerSecond;
double bpsSent;
double bpsReceived;
bool isActive; 
int connectMode; 
unsigned connectionElapsedTime;
};

struct NetworkEventHandler
{
virtual void onPeerConnect(IPlayer& peer) { }
virtual void onPeerDisconnect(IPlayer& peer, PeerDisconnectReason reason) { }
};

class NetworkBitStream;

struct NetworkInEventHandler
{
virtual bool onReceivePacket(IPlayer& peer, int id, NetworkBitStream& bs) { return true; }
virtual bool onReceiveRPC(IPlayer& peer, int id, NetworkBitStream& bs) { return true; }
};

struct SingleNetworkInEventHandler
{
virtual bool onReceive(IPlayer& peer, NetworkBitStream& bs) { return true; }
};

struct NetworkOutEventHandler
{
virtual bool onSendPacket(IPlayer* peer, int id, NetworkBitStream& bs) { return true; }
virtual bool onSendRPC(IPlayer* peer, int id, NetworkBitStream& bs) { return true; }
};

struct SingleNetworkOutEventHandler
{
virtual bool onSend(IPlayer* peer, NetworkBitStream& bs) { return true; }
};

struct PeerAddress
{
using AddressString = HybridString<46>;

bool ipv6; 
union
{
uint32_t v4; 
union
{
uint16_t segments[8]; 
uint8_t bytes[16]; 
} v6;
};

bool operator<(const PeerAddress& other) const
{
return ipv6 < other.ipv6 && v4 < other.v4 && v6.segments[2] < other.v6.segments[2] && v6.segments[3] < other.v6.segments[3] && v6.segments[4] < other.v6.segments[4] && v6.segments[5] < other.v6.segments[5] && v6.segments[6] < other.v6.segments[6] && v6.segments[7] < other.v6.segments[7];
}

bool operator==(const PeerAddress& other) const
{
return ipv6 == other.ipv6 && v4 == other.v4 && v6.segments[2] == other.v6.segments[2] && v6.segments[3] == other.v6.segments[3] && v6.segments[4] == other.v6.segments[4] && v6.segments[5] == other.v6.segments[5] && v6.segments[6] == other.v6.segments[6] && v6.segments[7] == other.v6.segments[7];
}

static bool FromString(PeerAddress& out, StringView string)
{
if (out.ipv6)
{
in6_addr output;
if (inet_pton(AF_INET6, string.data(), &output))
{
for (int i = 0; i < 16; ++i)
{
out.v6.bytes[i] = output.s6_addr[i];
}
return true;
}
}
else
{
in_addr output;
if (inet_pton(AF_INET, string.data(), &output))
{
out.v4 = output.s_addr;
return true;
}
}

return false;
}

static bool ToString(const PeerAddress& in, AddressString& address)
{
if (in.ipv6)
{
in6_addr addr;
for (int i = 0; i < 16; ++i)
{
addr.s6_addr[i] = in.v6.bytes[i];
}
char output[INET6_ADDRSTRLEN] {};
bool res = inet_ntop(AF_INET6, &addr, output, INET6_ADDRSTRLEN) != nullptr;
if (res)
{
address = AddressString(output);
}
return res;
}
else
{
in_addr addr;
addr.s_addr = in.v4;
char output[INET_ADDRSTRLEN] {};
bool res = inet_ntop(AF_INET, &addr, output, INET_ADDRSTRLEN) != nullptr;
if (res)
{
address = AddressString(output);
}
return res;
}
}
};

struct BanEntry
{
public:
PeerAddress::AddressString address; 
WorldTimePoint time; 
HybridString<MAX_PLAYER_NAME + 1> name; 
HybridString<32> reason; 

BanEntry(StringView address, WorldTimePoint time = WorldTime::now())
: address(address)
, time(time)
{
}

BanEntry(StringView address, StringView name, StringView reason, WorldTimePoint time = WorldTime::now())
: address(address)
, time(time)
, name(name)
, reason(reason)
{
}

bool operator<(const BanEntry& other) const
{
return address.cmp(other.address) < 0;
}

bool operator==(const BanEntry& other) const
{
return address == other.address;
}
};

struct INetwork : public IExtensible
{
virtual ENetworkType getNetworkType() const = 0;

virtual IEventDispatcher<NetworkEventHandler>& getEventDispatcher() = 0;

virtual IEventDispatcher<NetworkInEventHandler>& getInEventDispatcher() = 0;

virtual IIndexedEventDispatcher<SingleNetworkInEventHandler>& getPerRPCInEventDispatcher() = 0;

virtual IIndexedEventDispatcher<SingleNetworkInEventHandler>& getPerPacketInEventDispatcher() = 0;

virtual IEventDispatcher<NetworkOutEventHandler>& getOutEventDispatcher() = 0;

virtual IIndexedEventDispatcher<SingleNetworkOutEventHandler>& getPerRPCOutEventDispatcher() = 0;

virtual IIndexedEventDispatcher<SingleNetworkOutEventHandler>& getPerPacketOutEventDispatcher() = 0;

virtual bool sendPacket(IPlayer& peer, Span<uint8_t> data, int channel, bool dispatchEvents = true) = 0;

virtual bool broadcastPacket(Span<uint8_t> data, int channel, const IPlayer* exceptPeer = nullptr, bool dispatchEvents = true) = 0;

virtual bool sendRPC(IPlayer& peer, int id, Span<uint8_t> data, int channel, bool dispatchEvents = true) = 0;

virtual bool broadcastRPC(int id, Span<uint8_t> data, int channel, const IPlayer* exceptPeer = nullptr, bool dispatchEvents = true) = 0;

virtual NetworkStats getStatistics(IPlayer* player = nullptr) = 0;

virtual unsigned getPing(const IPlayer& peer) = 0;

virtual void disconnect(const IPlayer& peer) = 0;

virtual void ban(const BanEntry& entry, Milliseconds expire = Milliseconds(0)) = 0;

virtual void unban(const BanEntry& entry) = 0;

virtual void update() = 0;
};

struct INetworkComponent : public IComponent
{
ComponentType componentType() const override { return ComponentType::Network; }

virtual INetwork* getNetwork() = 0;
};

static const UID NetworkQueryExtension_UID = UID(0xfd46e147ea474971);
struct INetworkQueryExtension : public IExtension
{
PROVIDE_EXT_UID(NetworkQueryExtension_UID);

virtual bool addRule(StringView rule, StringView value) = 0;

virtual bool removeRule(StringView rule) = 0;

virtual bool isValidRule(StringView rule) = 0;
};

struct PeerNetworkData
{
struct NetworkID
{
PeerAddress address; 
unsigned short port; 
};

INetwork* network; 
NetworkID networkID; 
};
