package com.example.mobile

import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Bundle
import android.util.Log
import android.util.Size
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.core.content.ContextCompat
import com.example.mobile.ui.theme.MobileTheme
import okhttp3.*
import okio.ByteString.Companion.toByteString
import java.util.concurrent.Executors
import kotlin.concurrent.thread

class MainActivity : ComponentActivity() {

    private val TAG = "EdgeStreaming"
    private val WS_URL = "ws://10.42.0.1:8000/stream"

    private var webSocket: WebSocket? = null
    private val client = OkHttpClient()

    private val cameraExecutor = Executors.newSingleThreadExecutor()
    private var audioRecord: AudioRecord? = null
    private var isStreamingAudio = false

    private var streamingStatus by mutableStateOf("Initializing...")

    private val requestPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { permissions ->
            val cameraGranted = permissions[Manifest.permission.CAMERA] ?: false
            val audioGranted = permissions[Manifest.permission.RECORD_AUDIO] ?: false

            if (cameraGranted && audioGranted) {
                streamingStatus = "Permissions granted. Connecting to WebSocket..."
                connectWebSocket()
            } else {
                streamingStatus = "Permissions denied. Cannot stream."
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            MobileTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    Box(modifier = Modifier.padding(innerPadding).fillMaxSize(), contentAlignment = Alignment.Center) {
                        Text(text = streamingStatus)
                    }
                }
            }
        }

        checkPermissions()
    }

    private fun checkPermissions() {
        val cameraPermission = ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
        val audioPermission = ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)

        if (cameraPermission == PackageManager.PERMISSION_GRANTED &&
            audioPermission == PackageManager.PERMISSION_GRANTED) {
            streamingStatus = "Connecting to WebSocket..."
            connectWebSocket()
        } else {
            requestPermissionLauncher.launch(
                arrayOf(Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO)
            )
        }
    }

    private fun connectWebSocket() {
        val request = Request.Builder().url(WS_URL).build()
        webSocket = client.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                Log.d(TAG, "WebSocket Opened")
                streamingStatus = "Streaming connected to $WS_URL"
                runOnUiThread {
                    startCamera()
                    startAudio()
                }
            }

            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                Log.d(TAG, "WebSocket Closed")
                streamingStatus = "WebSocket Closed"
                stopStreaming()
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                Log.e(TAG, "WebSocket Failure", t)
                streamingStatus = "WebSocket Error: ${t.message}"
                stopStreaming()
            }
        })
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            val imageAnalysis = ImageAnalysis.Builder()
                .setTargetResolution(Size(640, 480))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()

            imageAnalysis.setAnalyzer(cameraExecutor) { imageProxy ->
                try {
                    val jpegBytes = YuvToJpegConverter.imageProxyToJpeg(imageProxy, 80)
                    // Prepend 0x01 for Video
                    val payload = ByteArray(jpegBytes.size + 1)
                    payload[0] = 0x01
                    System.arraycopy(jpegBytes, 0, payload, 1, jpegBytes.size)

                    webSocket?.send(payload.toByteString())
                } catch (e: Exception) {
                    Log.e(TAG, "Error processing frame", e)
                } finally {
                    imageProxy.close()
                }
            }

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, cameraSelector, imageAnalysis)
            } catch (e: Exception) {
                Log.e(TAG, "Use case binding failed", e)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    @SuppressLint("MissingPermission")
    private fun startAudio() {
        if (isStreamingAudio) return

        val sampleRate = 16000
        val channelConfig = AudioFormat.CHANNEL_IN_MONO
        val audioFormat = AudioFormat.ENCODING_PCM_16BIT
        val bufferSize = AudioRecord.getMinBufferSize(sampleRate, channelConfig, audioFormat) * 2

        audioRecord = AudioRecord(
            MediaRecorder.AudioSource.MIC,
            sampleRate,
            channelConfig,
            audioFormat,
            bufferSize
        )

        audioRecord?.startRecording()
        isStreamingAudio = true

        thread(start = true) {
            val audioBuffer = ByteArray(bufferSize)
            while (isStreamingAudio) {
                val readResult = audioRecord?.read(audioBuffer, 0, audioBuffer.size) ?: 0
                if (readResult > 0) {
                    // Prepend 0x02 for Audio
                    val payload = ByteArray(readResult + 1)
                    payload[0] = 0x02
                    System.arraycopy(audioBuffer, 0, payload, 1, readResult)
                    webSocket?.send(payload.toByteString())
                }
            }
        }
    }

    private fun stopStreaming() {
        isStreamingAudio = false
        audioRecord?.stop()
        audioRecord?.release()
        audioRecord = null
        
        try {
            val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
            cameraProviderFuture.get().unbindAll()
        } catch (e: Exception) {
            Log.e(TAG, "Error unbinding camera", e)
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        stopStreaming()
        cameraExecutor.shutdown()
        webSocket?.close(1000, "Activity Destroyed")
        client.dispatcher.executorService.shutdown()
    }
}