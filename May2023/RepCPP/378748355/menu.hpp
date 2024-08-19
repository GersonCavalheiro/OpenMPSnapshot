

#pragma once

#include <network.hpp>
#include <player.hpp>
#include <types.hpp>

namespace NetCode
{
namespace RPC
{
struct PlayerInitMenu : NetworkPacketBase<76, NetworkPacketType::RPC, OrderingChannel_SyncRPC>
{
uint8_t MenuID;
bool HasTwoColumns;
StaticString<MAX_MENU_TEXT_LENGTH> Title;
Vector2 Position;
float Col1Width;
float Col2Width;
bool MenuEnabled;
StaticArray<bool, MAX_MENU_ITEMS> RowEnabled;
StaticArray<StaticString<MAX_MENU_TEXT_LENGTH>, 2> ColumnHeaders;
StaticArray<uint8_t, 2> ColumnItemCount;
StaticArray<StaticArray<StaticString<MAX_MENU_TEXT_LENGTH>, MAX_MENU_ITEMS>, 2> MenuItems;

bool read(NetworkBitStream& bs)
{
return false;
}

void write(NetworkBitStream& bs) const
{
bs.writeUINT8(MenuID);
bs.writeUINT32(HasTwoColumns);

bs.writeArray(Title.data());

bs.writeVEC2(Position);
bs.writeFLOAT(Col1Width);

if (HasTwoColumns)
{
bs.writeFLOAT(Col2Width);
}

bs.writeUINT32(MenuEnabled);
for (bool isRowEnabled : RowEnabled)
{
bs.writeUINT32(isRowEnabled);
}

uint8_t firstColumnItemCount = ColumnItemCount.at(0);
auto& firstColumnHeader = ColumnHeaders.at(0);
auto& firstColumnItems = MenuItems.at(0);

bs.writeArray(firstColumnHeader.data());

bs.writeUINT8(firstColumnItemCount);
for (uint8_t i = 0; i < firstColumnItemCount; i++)
{
bs.writeArray(firstColumnItems.at(i).data());
}

if (HasTwoColumns)
{
uint8_t secondColumnItemCount = ColumnItemCount.at(1);
auto& secondColumnHeader = ColumnHeaders.at(1);
auto& secondColumnItems = MenuItems.at(1);

bs.writeArray(secondColumnHeader.data());

bs.writeUINT8(secondColumnItemCount);
for (uint8_t i = 0; i < secondColumnItemCount; i++)
{
bs.writeArray(secondColumnItems.at(i).data());
}
}
}
};

struct PlayerShowMenu : NetworkPacketBase<77, NetworkPacketType::RPC, OrderingChannel_SyncRPC>
{
uint8_t MenuID;

bool read(NetworkBitStream& bs)
{
return false;
}

void write(NetworkBitStream& bs) const
{
bs.writeUINT8(MenuID);
}
};

struct PlayerHideMenu : NetworkPacketBase<78, NetworkPacketType::RPC, OrderingChannel_SyncRPC>
{
uint8_t MenuID;

bool read(NetworkBitStream& bs)
{
return false;
}

void write(NetworkBitStream& bs) const
{
bs.writeUINT8(MenuID);
}
};

struct OnPlayerSelectedMenuRow : NetworkPacketBase<132, NetworkPacketType::RPC, OrderingChannel_SyncRPC>
{
uint8_t MenuRow;

bool read(NetworkBitStream& bs)
{
return bs.readUINT8(MenuRow);
}

void write(NetworkBitStream& bs) const
{
}
};

struct OnPlayerExitedMenu : NetworkPacketBase<140, NetworkPacketType::RPC, OrderingChannel_SyncRPC>
{
bool read(NetworkBitStream& bs)
{
return true;
}

void write(NetworkBitStream& bs) const
{
}
};
}
}
