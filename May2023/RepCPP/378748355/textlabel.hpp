

#pragma once

#include <network.hpp>
#include <player.hpp>
#include <types.hpp>

namespace NetCode
{
namespace RPC
{
struct PlayerShowTextLabel : NetworkPacketBase<36, NetworkPacketType::RPC, OrderingChannel_SyncRPC>
{
bool PlayerTextLabel;
int TextLabelID;
Colour Col;
Vector3 Position;
float DrawDistance;
bool LOS;
int PlayerAttachID;
int VehicleAttachID;
HybridString<64> Text;

bool read(NetworkBitStream& bs)
{
return false;
}

void write(NetworkBitStream& bs) const
{
bs.writeUINT16(PlayerTextLabel ? TEXT_LABEL_POOL_SIZE + TextLabelID : TextLabelID);
bs.writeUINT32(Col.RGBA());
bs.writeVEC3(Position);
bs.writeFLOAT(DrawDistance);
bs.writeUINT8(LOS);
bs.writeUINT16(PlayerAttachID);
bs.writeUINT16(VehicleAttachID);
bs.WriteCompressedStr(Text);
}
};

struct PlayerHideTextLabel : NetworkPacketBase<58, NetworkPacketType::RPC, OrderingChannel_SyncRPC>
{
bool PlayerTextLabel;
int TextLabelID;

bool read(NetworkBitStream& bs)
{
return false;
}

void write(NetworkBitStream& bs) const
{
bs.writeUINT16(PlayerTextLabel ? TEXT_LABEL_POOL_SIZE + TextLabelID : TextLabelID);
}
};
}
}
