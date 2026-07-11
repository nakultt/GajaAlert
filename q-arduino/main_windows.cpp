#include <uwebsockets/App.h>
#include <iostream>
#include <vector>
#include <cstdint>

// Stub function for vision inference
void run_vision_inference(size_t frame_bytes) {
    std::cout << "[Vision] Received frame payload of size: " 
              << frame_bytes << " bytes" << std::endl;
}

// Stub function for audio inference
void run_audio_inference(std::vector<int16_t> pcm_chunk) {
    std::cout << "[Audio] Running inference on chunk: " 
              << pcm_chunk.size() << " samples" << std::endl;
}

int main() {
    // Create the uWS App. By default it supports WebSocket connections.
    uWS::App().ws<int>("/*", {
        .open = [](auto *ws) {
            std::cout << "Client connected." << std::endl;
        },
        .message = [](auto *ws, std::string_view message, uWS::OpCode opCode) {
            // Process only binary messages that are not empty
            if (opCode == uWS::OpCode::BINARY && !message.empty()) {
                uint8_t header = static_cast<uint8_t>(message[0]);
                std::string_view payload = message.substr(1);

                if (header == 0x01) {
                    // --- VIDEO PAYLOAD ---
                    // Temporarily removed OpenCV for native Windows test
                    run_vision_inference(payload.size());
                } 
                else if (header == 0x02) {
                    // --- AUDIO PAYLOAD ---
                    size_t num_samples = payload.size() / sizeof(int16_t);
                    const int16_t* pcm_data = reinterpret_cast<const int16_t*>(payload.data());
                    std::vector<int16_t> pcm_chunk(pcm_data, pcm_data + num_samples);
                    
                    run_audio_inference(pcm_chunk);
                } 
                else {
                    std::cerr << "Unknown header byte: 0x" << std::hex 
                              << static_cast<int>(header) << std::dec << std::endl;
                }
            }
        },
        .close = [](auto *ws, int code, std::string_view message) {
            std::cout << "Client disconnected." << std::endl;
        }
    }).listen("0.0.0.0", 8000, [](auto *listen_socket) {
        if (listen_socket) {
            std::cout << "Edge Server listening on 0.0.0.0:8000" << std::endl;
        } else {
            std::cerr << "Failed to bind to port 8000" << std::endl;
        }
    }).run();

    return 0;
}
