import asyncio
import os
import json
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def run_sarvam_workflow(input_text: str):
    """
    Connects to the Sarvam MCP server and executes a sequence of tools.
    1. sarvam_llm_complete: Generates a summary
    2. sarvam_translate: Translates to Tamil, Telugu, Malayalam, Kannada
    3. sarvam_tts_stream: Streams audio (e.g. for notifications)
    4. sarvam_tts_speak: Saves the audio as a file
    """
    
    # Configure the Sarvam MCP server parameters
    # Uses `uvx` to execute the sarvam-mcp server as recommended
    server_params = StdioServerParameters(
        command="uvx",
        args=["sarvam-mcp"],
        env={
            **os.environ,
            # Hardcoded SARVAM API key
            "SARVAM_API_KEY": "sk_jb27svzp_3KP7k5AjVWsC5OkqAMFwyWLo"
        }
    )

    print("Connecting to Sarvam MCP server...")
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("Connected and initialized to Sarvam MCP.\n")

            # -------------------------------------------------------------
            # 1. Generate Summary
            # -------------------------------------------------------------
            print("--- 1. Generating Summary (sarvam_llm_complete) ---")
            summary_args = {
                "prompt": f"Please summarize this alert notification concisely: {input_text}"
            }
            # Make the tool call to the MCP server
            try:
                summary_result = await session.call_tool("sarvam_llm_complete", summary_args)
                summary_text = summary_result.content[0].text
                print(f"Summary generated: {summary_text}\n")
            except Exception as e:
                print(f"Error calling sarvam_llm_complete: {e}")
                summary_text = input_text # Fallback to input text

            # -------------------------------------------------------------
            # 2. Translate to multiple languages
            # -------------------------------------------------------------
            print("--- 2. Translating (sarvam_translate) ---")
            target_languages = ["tamil", "telugu", "malayalam", "kannada"]
            translations = {}
            for lang in target_languages:
                try:
                    trans_args = {
                        "text": summary_text,
                        "target_language": lang,
                        "source_language": "english"
                    }
                    trans_result = await session.call_tool("sarvam_translate", trans_args)
                    translations[lang] = trans_result.content[0].text
                    print(f"{lang.capitalize()}: {translations[lang]}")
                except Exception as e:
                    print(f"Error calling sarvam_translate for {lang}: {e}")
            print()

            # -------------------------------------------------------------
            # 3. TTS Stream (Mobile Notification / Alert setup)
            # -------------------------------------------------------------
            print("--- 3. TTS Stream (sarvam_tts_stream) ---")
            for lang, text in translations.items():
                try:
                    stream_args = {
                        "text": text,
                        "language": lang
                    }
                    # This tool stream data (usually for WebSockets/WebRTC in mobile)
                    # For now we'll just initiate the tool call
                    await session.call_tool("sarvam_tts_stream", stream_args)
                    print(f"Initiated TTS stream for {lang.capitalize()} mobile alert...")
                except Exception as e:
                    print(f"Error calling sarvam_tts_stream for {lang}: {e}")
            print()

            # -------------------------------------------------------------
            # 4. TTS Speak (Save to Audio File)
            # -------------------------------------------------------------
            print("--- 4. TTS Speak (sarvam_tts_speak) ---")
            for lang, text in translations.items():
                try:
                    # Provide an output path to save the generated audio
                    output_filename = f"alert_{lang}.wav"
                    speak_args = {
                        "text": text,
                        "language": lang,
                        "output_file": output_filename 
                    }
                    speak_result = await session.call_tool("sarvam_tts_speak", speak_args)
                    print(f"Saved TTS audio to {output_filename}")
                except Exception as e:
                    print(f"Error calling sarvam_tts_speak for {lang}: {e}")

if __name__ == "__main__":
    # Test alert input
    test_alert = "A wild elephant has been spotted near the northern farm border. Please stay indoors and alert local authorities immediately."
    
    # Run the async workflow
    asyncio.run(run_sarvam_workflow(test_alert))
