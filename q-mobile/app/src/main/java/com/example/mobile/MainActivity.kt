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
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.*
import androidx.compose.material3.Button
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.example.mobile.ui.theme.MobileTheme
import okhttp3.*
import okio.ByteString.Companion.toByteString
import java.util.concurrent.Executors
import kotlin.concurrent.thread

class MainActivity : ComponentActivity() {

    private val TAG = "EdgeStreaming"

    private var webSocket: WebSocket? = null
    private val client = OkHttpClient()

    private val cameraExecutor = Executors.newSingleThreadExecutor()
    private var audioRecord: AudioRecord? = null
    private var isStreamingAudio = false
    private var isConnected by mutableStateOf(false)

    private var streamingStatus by mutableStateOf("Ready to connect")
    private var pendingIpAddress = ""
    
    // We store the surface provider to bind it later when connected
    private var surfaceProvider: Preview.SurfaceProvider? = null

    private val requestPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { permissions ->
            val cameraGranted = permissions[Manifest.permission.CAMERA] ?: false
            val audioGranted = permissions[Manifest.permission.RECORD_AUDIO] ?: false

            if (cameraGranted && audioGranted) {
                streamingStatus = "Permissions granted. Connecting..."
                connectWebSocket(pendingIpAddress)
            } else {
                streamingStatus = "Permissions denied. Cannot stream."
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            MobileTheme {
                var ipAddress by remember { mutableStateOf("192.168.") }

                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    Column(
                        modifier = Modifier
                            .padding(innerPadding)
                            .fillMaxSize()
                            .padding(16.dp),
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.Top
                    ) {
                        OutlinedTextField(
                            value = ipAddress,
                            onValueChange = { ipAddress = it },
                            label = { Text("Arduino IP Address") },
                            modifier = Modifier.fillMaxWidth(),
                            enabled = !isConnected
                        )
                        Spacer(modifier = Modifier.height(16.dp))
                        Button(
                            onClick = {
                                if (isConnected) {
                                    disconnectWebSocket()
                                } else {
                                    checkPermissionsAndConnect(ipAddress)
                                }
                            },
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Text(if (isConnected) "Disconnect" else "Connect")
                        }
                        Spacer(modifier = Modifier.height(16.dp))
                        Text(text = streamingStatus)
                        Spacer(modifier = Modifier.height(16.dp))
                        
                        // Critical: We MUST have a PreviewView to force the Camera HAL to pump frames
                        AndroidView(
                            factory = { ctx ->
                                PreviewView(ctx).apply {
                                    // Extract the SurfaceProvider to use in our startCamera function
                                    this@MainActivity.surfaceProvider = this.surfaceProvider
                                }
                            },
                            modifier = Modifier
                                .fillMaxWidth()
                                .aspectRatio(3f/4f) // Typical camera aspect ratio
                        )
                    }
                }
            }
        }
    }

    private fun checkPermissionsAndConnect(ipAddress: String) {
        pendingIpAddress = ipAddress
        val cameraPermission = ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
        val audioPermission = ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO)

        if (cameraPermission == PackageManager.PERMISSION_GRANTED &&
            audioPermission == PackageManager.PERMISSION_GRANTED) {
            streamingStatus = "Connecting..."
            connectWebSocket(ipAddress)
        } else {
            requestPermissionLauncher.launch(
                arrayOf(Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO)
            )
        }
    }

    private fun connectWebSocket(ip: String) {
        val url = "ws://$ip:8000/"
        val request = Request.Builder().url(url).build()
        
        webSocket = client.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                Log.d(TAG, "WebSocket Opened")
                streamingStatus = "Streaming connected to $url"
                isConnected = true
                runOnUiThread {
                    startCamera()
                    startAudio()
                }
            }

            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                Log.d(TAG, "WebSocket Closed")
                streamingStatus = "WebSocket Closed"
                isConnected = false
                stopStreaming()
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                Log.e(TAG, "WebSocket Failure", t)
                streamingStatus = "WebSocket Error: ${t.message}"
                isConnected = false
                stopStreaming()
            }
        })
    }

    private fun disconnectWebSocket() {
        webSocket?.close(1000, "User disconnected")
        webSocket = null
        isConnected = false
        streamingStatus = "Disconnected"
        stopStreaming()
    }

    private fun startCamera() {
        if (surfaceProvider == null) {
            Log.e(TAG, "SurfaceProvider is null! Camera cannot start.")
            return
        }
        
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()

            // 1. Preview Usecase (Required by many OEMs to output frames)
            val preview = Preview.Builder()
                .build()
            preview.setSurfaceProvider(surfaceProvider)

            // 2. Image Analysis Usecase
            val imageAnalysis = ImageAnalysis.Builder()
                .setTargetResolution(Size(640, 480))
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()

            var lastFrameTimeMs = 0L

            imageAnalysis.setAnalyzer(cameraExecutor) { imageProxy ->
                try {
                    val currentTime = System.currentTimeMillis()
                    // Throttle to roughly 15 FPS (1000ms / 15 = 66ms)
                    if (currentTime - lastFrameTimeMs < 66) {
                        return@setAnalyzer
                    }
                    lastFrameTimeMs = currentTime

                    val bitmap = imageProxy.toBitmap()
                    val stream = java.io.ByteArrayOutputStream()
                    // Heavy compression (40) to prevent TCP buffer bloat while maintaining 640x480 size
                    bitmap.compress(android.graphics.Bitmap.CompressFormat.JPEG, 40, stream)
                    val jpegBytes = stream.toByteArray()
                    // Prepend 0x01 for Video
                    val payload = ByteArray(jpegBytes.size + 1)
                    payload[0] = 0x01
                    System.arraycopy(jpegBytes, 0, payload, 1, jpegBytes.size)

                    webSocket?.send(payload.toByteString())
                } catch (e: Exception) {
                    Log.e(TAG, "Error processing frame", e)
                } finally {
                    imageProxy.close() // ALWAYS close to get the next frame
                }
            }

            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA

            try {
                cameraProvider.unbindAll()
                cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageAnalysis)
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