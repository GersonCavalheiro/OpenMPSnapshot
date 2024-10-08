

#pragma once

#include <network.hpp>
#include <player.hpp>
#include <types.hpp>

namespace NetCode
{
namespace RPC
{
struct SetCheckpoint : NetworkPacketBase<107, NetworkPacketType::RPC, OrderingChannel_SyncRPC>
{
Vector3 position;
float size;

bool read(NetworkBitStream& bs)
{
return false;
}

void write(NetworkBitStream& bs) const
{
bs.writeVEC3(position);
bs.writeFLOAT(size);
}
};

struct DisableCheckpoint : NetworkPacketBase<37, NetworkPacketType::RPC, OrderingChannel_SyncRPC>
{
bool read(NetworkBitStream& bs)
{
return false;
}

void write(NetworkBitStream& bs) const { }
};

struct SetRaceCheckpoint : NetworkPacketBase<38, NetworkPacketType::RPC, OrderingChannel_SyncRPC>
{
uint8_t type;
Vector3 position;
Vector3 nextPosition;
float size;

bool read(NetworkBitStream& bs)
{
return false;
}

void write(NetworkBitStream& bs) const
{
bs.writeUINT8(type);
bs.writeVEC3(position);
bs.writeVEC3(nextPosition);
bs.writeFLOAT(size);
}
};

struct DisableRaceCheckpoint : NetworkPacketBase<39, NetworkPacketType::RPC, OrderingChannel_SyncRPC>
{
bool read(NetworkBitStream& bs)
{
return false;
}

void write(NetworkBitStream& bs) const { }
};
}
}
