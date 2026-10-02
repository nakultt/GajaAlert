#include <uwebsockets/App.h>
#include <ixwebsocket/IXNetSystem.h>
#include <ixwebsocket/IXWebSocket.h>
#include <iostream>
#include <vector>
#include <cstdint>
#include <string>

// Global WebSocket client to forward data to ARM PC
ix::WebSocket ix_ws;

int main() {
    ix::initNetSystem();

    // Set the ARM PC WebSocket Server IP (testing locally on port 9000)
    ix_ws.setUrl("ws://localhost:9000"); 

    std::cout << "Arduino Relay: Connecting to ARM PC at ws://localhost:9000..." << std::endl;
    ix_ws.setOnMessageCallback([](const ix::WebSocketMessagePtr& msg) {
        if (msg->type == ix::WebSocketMessageType::Open) {
            std::cout << "Arduino Relay: Connected successfully to ARM PC!" << std::endl;
        } else if (msg->type == ix::WebSocketMessageType::Error) {
            std::cerr << "Arduino Relay: Error connecting to ARM PC: " << msg->errorInfo.reason << std::endl;
        }
    });

    ix_ws.start();

    // Create the uWS App to receive data from the Mobile Phone
    uWS::App().ws<int>("/*", {
        .open = [](auto *ws) {
            std::cout << "Mobile Phone connected to Arduino!" << std::endl;
        },
        .message = [](auto *ws, std::string_view message, uWS::OpCode opCode) {
            if (opCode == uWS::OpCode::BINARY && !message.empty()) {
                uint8_t header = static_cast<uint8_t>(message[0]);

                if (header == 0x01) {
                    std::cout << "[Relay] Forwarding video frame (" << message.size() << " bytes) to ARM PC" << std::endl;
                } else if (header == 0x02) {
                    // Audio frame received locally (not forwarded)
                }

                // Push exact payload to ARM PC
                if (ix_ws.getReadyState() == ix::ReadyState::Open) {
                    ix_ws.sendBinary(std::string(message));
                }
            }
        },
        .close = [](auto *ws, int code, std::string_view message) {
            std::cout << "Mobile Phone disconnected." << std::endl;
        }
    }).listen("0.0.0.0", 8000, [](auto *listen_socket) {
        if (listen_socket) {
            std::cout << "Arduino Relay Server listening on 0.0.0.0:8000" << std::endl;
        } else {
            std::cerr << "Failed to bind to port 8000" << std::endl;
        }
    }).run();

    ix_ws.stop();
    ix::uninitNetSystem();
    return 0;
}
