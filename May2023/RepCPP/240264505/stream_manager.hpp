

#ifndef LBT_STREAM_MANAGER
#define LBT_STREAM_MANAGER
#pragma once

#include <ostream>


namespace lbt {


class StreamManager {
public:

explicit StreamManager(std::ostream& src) noexcept 
: src_stream{src}, src_buf{src.rdbuf()} {
return;
}
StreamManager() = delete;
StreamManager(StreamManager const&) = delete;
StreamManager(StreamManager&&) = delete;
StreamManager& operator=(StreamManager const&) = delete;
StreamManager& operator=(StreamManager&&) = delete;


void redirect(std::ostream& dst) noexcept {
src_stream.rdbuf(dst.rdbuf());
return;
}


bool restore() noexcept {
bool is_success {false};
if (src_buf != nullptr) {
src_stream.rdbuf(src_buf);
}
return is_success;
}


void turnOn() noexcept {
src_stream.clear();
return;
}


void turnOff() noexcept {
src_stream.setstate(std::ios_base::failbit);
return;
}


~StreamManager() noexcept {
restore();
turnOn();
return;
}

protected:
std::ostream& src_stream; 
std::streambuf* src_buf;  
};

}

#endif 
