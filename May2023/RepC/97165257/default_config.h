#pragma once
#define DEFAULT_CONFIG \
"First 64 characters (including spaces) in a line are ignored."                                                   CRLF\
"Empty lines are NOT ignored. Lines shorter than 64 characters"                                                   CRLF\
"are ignored. Order of settings is important! Use CLRF (Windows"                                                  CRLF\
"default) line endings."                                                                                          CRLF\
""                                                                                                                CRLF\
"Path to vlc.exe                                                 C:\\Program Files (x86)\\VideoLAN\\VLC\\vlc.exe" CRLF\
"Quality: bitrate (LAN speed)                             (kB/s) 8000"                                            CRLF\
"Resolution: height (???p)                                       720"                                             CRLF\
"Frames per second                                               25"                                              CRLF\
""                                                                                                                CRLF\
"Maximum ping (latency)                                     (ms) 200"                                             CRLF\
""                                                                                                                CRLF\
"Remote control port (any open on remote)              [1-65536] 8852"                                            CRLF\
"Remote password                           (max 1024 characters) Hey! Wake up!"                                   CRLF\
"Local control port (any free)                         [1-65536] 8853"                                            CRLF\
"Video output port (any open on PC)                    [1-65536] 8854"                                            CRLF\
""                                                                                                                CRLF\
"not implemented!"                                                                                                CRLF\
"Use static list of subnets                             [0 or 1] 0"                                               CRLF\
"Static list of subnets         [like in command line arguments] 192.168.0.0 255"                                 CRLF
