#pragma once

#include "bitstream.hpp"
#include <core.hpp>
#include <network.hpp>
#include <player.hpp>

enum class NetworkPacketType {
Packet,
RPC,
};

template <int PktID, NetworkPacketType PktType, int PktChannel>
struct NetworkPacketBase {
static constexpr const int PacketID = PktID;
static constexpr const NetworkPacketType PacketType = PktType;
static constexpr const int PacketChannel = PktChannel;

constexpr static void addEventHandler(ICore& core, SingleNetworkInEventHandler* handler, event_order_t priority = EventPriority_Default)
{
if (PacketType == NetworkPacketType::RPC) {
core.addPerRPCInEventHandler<PacketID>(handler, priority);
} else if (PacketType == NetworkPacketType::Packet) {
core.addPerPacketInEventHandler<PacketID>(handler, priority);
}
}

constexpr static void removeEventHandler(ICore& core, SingleNetworkInEventHandler* handler, event_order_t priority = EventPriority_Default)
{
if (PacketType == NetworkPacketType::RPC) {
core.removePerRPCInEventHandler<PacketID>(handler);
} else if (PacketType == NetworkPacketType::Packet) {
core.removePerPacketInEventHandler<PacketID>(handler);
}
}
};

std::false_type is_network_packet_impl(...);
template <int PktID, NetworkPacketType PktType, int PktChannel>
std::true_type is_network_packet_impl(NetworkPacketBase<PktID, PktType, PktChannel> const volatile&);
template <typename T>
using is_network_packet = decltype(is_network_packet_impl(std::declval<T&>()));

struct PacketHelper {
template <typename Packet, typename E = std::enable_if_t<is_network_packet<Packet>::value>>
static bool send(const Packet& packet, IPlayer& peer)
{
NetworkBitStream bs;
packet.write(bs);
if constexpr (Packet::PacketType == NetworkPacketType::RPC) {
return peer.sendRPC(Packet::PacketID, Span<uint8_t>(bs.GetData(), bs.GetNumberOfBitsUsed()), Packet::PacketChannel);
} else if constexpr (Packet::PacketType == NetworkPacketType::Packet) {
return peer.sendPacket(Span<uint8_t>(bs.GetData(), bs.GetNumberOfBitsUsed()), Packet::PacketChannel);
}
}

template <typename Packet, typename E = std::enable_if_t<is_network_packet<Packet>::value>>
static void broadcastToSome(const Packet& packet, const FlatPtrHashSet<IPlayer>& players, const IPlayer* skipFrom = nullptr)
{
NetworkBitStream bs;
packet.write(bs);
for (IPlayer* peer : players) {
if (peer != skipFrom) {
if constexpr (Packet::PacketType == NetworkPacketType::RPC) {
peer->sendRPC(Packet::PacketID, Span<uint8_t>(bs.GetData(), bs.GetNumberOfBitsUsed()), Packet::PacketChannel);
} else if constexpr (Packet::PacketType == NetworkPacketType::Packet) {
peer->sendPacket(Span<uint8_t>(bs.GetData(), bs.GetNumberOfBitsUsed()), Packet::PacketChannel);
}
}
}
}

template <typename Packet, typename E = std::enable_if_t<is_network_packet<Packet>::value>>
static void broadcastToStreamed(const Packet& packet, IPlayer& player, bool skipFrom = false)
{
NetworkBitStream bs;
packet.write(bs);
if constexpr (Packet::PacketType == NetworkPacketType::RPC) {
return player.broadcastRPCToStreamed(Packet::PacketID, Span<uint8_t>(bs.GetData(), bs.GetNumberOfBitsUsed()), Packet::PacketChannel, skipFrom);
} else if constexpr (Packet::PacketType == NetworkPacketType::Packet) {
return player.broadcastPacketToStreamed(Span<uint8_t>(bs.GetData(), bs.GetNumberOfBitsUsed()), Packet::PacketChannel, skipFrom);
}
}

template <typename Packet, typename E = std::enable_if_t<is_network_packet<Packet>::value>>
static void broadcastSyncPacket(const Packet& packet, IPlayer& player)
{
static_assert(Packet::PacketType == NetworkPacketType::Packet, "broadcastSyncPacket can only be used with NetworkPacketType::Packet");
NetworkBitStream bs;
packet.write(bs);
return player.broadcastSyncPacket(Span<uint8_t>(bs.GetData(), bs.GetNumberOfBitsUsed()), Packet::PacketChannel);
}

template <typename Packet, typename E = std::enable_if_t<is_network_packet<Packet>::value, Packet>>
static void broadcast(const Packet& packet, IPlayerPool& players, const IPlayer* skipFrom = nullptr)
{
NetworkBitStream bs;
packet.write(bs);
if constexpr (Packet::PacketType == NetworkPacketType::RPC) {
players.broadcastRPC(Packet::PacketID, Span<uint8_t>(bs.GetData(), bs.GetNumberOfBitsUsed()), Packet::PacketChannel, skipFrom);
}
}
};
