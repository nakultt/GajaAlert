package com.example.mobile

import android.Manifest
import android.annotation.SuppressLint
import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Build
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
import androidx.compose.material3.Card
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat
import androidx.core.content.ContextCompat
import com.example.mobile.ui.theme.MobileTheme
import okhttp3.*
import okio.ByteString
import okio.ByteString.Companion.toByteString
import org.json.JSONObject
import java.util.concurrent.Executors
import kotlin.concurrent.thread

/**
 * Two roles, one app:
 *  - SENSOR: this phone stands in for the field sensors — streams camera (0x01 JPEG)
 *    and mic audio (0x02 PCM16) to the laptop's ws://<ip>:9000 ingest.
 *  - RECEIVER: a nearby phone that just listens on ws://<ip>:9001 for 0x03 + JSON
 *    elephant alerts broadcast by the laptop pipeline, and surfaces them as a
 *    notification + on-screen card in the user's preferred language.
 */
class MainActivity : ComponentActivity() {

    private val TAG = "EdgeStreaming"

    private enum class Mode { SENSOR, RECEIVER }

    companion object {
        private const val SENSOR_PORT = 9000
        private const val RECEIVER_PORT = 9001
        private const val ALERT_CHANNEL_ID = "elephant_alerts"
    }

    private data class AlertPayload(
        val id: String,
        val timestamp: String,
        val confidence: Double,
        val report: String,
        val alerts: Map<String, String>,
        val location: String,
    )

    private var webSocket: WebSocket? = null
    private val client = OkHttpClient()

    private val cameraExecutor = Executors.newSingleThreadExecutor()
    private var audioRecord: AudioRecord? = null
    private var isStreamingAudio = false
    private var isConnected by mutableStateOf(false)

    private var mode by mutableStateOf(Mode.SENSOR)
    private var pendingMode = Mode.SENSOR
    private var languagePref by mutableStateOf("en")
    private var latestAlert by mutableStateOf<AlertPayload?>(null)

    private var streamingStatus by mutableStateOf("Ready to connect")
    private var pendingIpAddress = ""

    // We store the surface provider to bind it later when connected
    private var surfaceProvider: Preview.SurfaceProvider? = null

    private fun requiredPermissions(forMode: Mode): Array<String> = when (forMode) {
        Mode.SENSOR -> arrayOf(Manifest.permission.CAMERA, Manifest.permission.RECORD_AUDIO)
        Mode.RECEIVER -> arrayOf(Manifest.permission.POST_NOTIFICATIONS)
    }

    private val requestPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { permissions ->
            val granted = requiredPermissions(pendingMode).all { permissions[it] == true }
            if (granted) {
                streamingStatus = "Permissions granted. Connecting..."
                connect(pendingMode, pendingIpAddress)
            } else {
                streamingStatus = "Permissions denied. Cannot ${if (pendingMode == Mode.SENSOR) "stream" else "receive alerts"}."
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        createNotificationChannel()
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
                        Row(
                            modifier = Modifier.fillMaxWidth(),
                            horizontalArrangement = Arrangement.spacedBy(8.dp)
                        ) {
                            ModeButton(
                                label = "Sensor",
                                selected = mode == Mode.SENSOR,
                                enabled = !isConnected,
                                modifier = Modifier.weight(1f)
                            ) { mode = Mode.SENSOR }
                            ModeButton(
                                label = "Alert Receiver",
                                selected = mode == Mode.RECEIVER,
                                enabled = !isConnected,
                                modifier = Modifier.weight(1f)
                            ) { mode = Mode.RECEIVER }
                        }
                        Spacer(modifier = Modifier.height(16.dp))
                        OutlinedTextField(
                            value = ipAddress,
                            onValueChange = { ipAddress = it },
                            label = { Text("Laptop Server IP Address") },
                            modifier = Modifier.fillMaxWidth(),
                            enabled = !isConnected
                        )
                        Spacer(modifier = Modifier.height(16.dp))

                        if (mode == Mode.RECEIVER) {
                            Row(
                                modifier = Modifier.fillMaxWidth(),
                                horizontalArrangement = Arrangement.spacedBy(8.dp)
                            ) {
                                listOf("en" to "English", "hi" to "Hindi", "ta" to "Tamil").forEach { (code, label) ->
                                    ModeButton(
                                        label = label,
                                        selected = languagePref == code,
                                        enabled = true,
                                        modifier = Modifier.weight(1f)
                                    ) { languagePref = code }
                                }
                            }
                            Spacer(modifier = Modifier.height(16.dp))
                        }

                        Button(
                            onClick = {
                                if (isConnected) {
                                    disconnectWebSocket()
                                } else {
                                    checkPermissionsAndConnect(mode, ipAddress)
                                }
                            },
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Text(
                                if (isConnected) "Disconnect"
                                else if (mode == Mode.SENSOR) "Connect as Sensor"
                                else "Listen for Alerts"
                            )
                        }
                        Spacer(modifier = Modifier.height(16.dp))
                        Text(text = streamingStatus)
                        Spacer(modifier = Modifier.height(16.dp))

                        if (mode == Mode.SENSOR) {
                            // Critical: We MUST have a PreviewView to force the Camera HAL to pump frames
                            AndroidView(
                                factory = { ctx ->
                                    PreviewView(ctx).apply {
                                        this@MainActivity.surfaceProvider = this.surfaceProvider
                                    }
                                },
                                modifier = Modifier
                                    .fillMaxWidth()
                                    .aspectRatio(3f / 4f)
                            )
                        } else {
                            latestAlert?.let { alert ->
                                Card(modifier = Modifier.fillMaxWidth().padding(top = 8.dp)) {
                                    Column(modifier = Modifier.padding(16.dp)) {
                                        Text(text = "Elephant Alert — ${alert.location}")
                                        Spacer(modifier = Modifier.height(8.dp))
                                        Text(text = alert.alerts[languagePref] ?: alert.alerts["en"] ?: alert.report)
                                        Spacer(modifier = Modifier.height(8.dp))
                                        Text(text = "Confidence: ${alert.confidence} · ${alert.timestamp}")
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    @Composable
    private fun ModeButton(
        label: String,
        selected: Boolean,
        enabled: Boolean,
        modifier: Modifier = Modifier,
        onClick: () -> Unit
    ) {
        if (selected) {
            Button(onClick = onClick, enabled = enabled, modifier = modifier) { Text(label) }
        } else {
            OutlinedButton(onClick = onClick, enabled = enabled, modifier = modifier) { Text(label) }
        }
    }

    private fun createNotificationChannel() {
        val channel = NotificationChannel(
            ALERT_CHANNEL_ID, "Elephant Alerts", NotificationManager.IMPORTANCE_HIGH
        ).apply { description = "Elephant early-warning alerts from the Gaja edge pipeline" }
        getSystemService(NotificationManager::class.java).createNotificationChannel(channel)
    }

    private fun checkPermissionsAndConnect(forMode: Mode, ipAddress: String) {
        pendingMode = forMode
        pendingIpAddress = ipAddress
        val missing = requiredPermissions(forMode).filter {
            ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
        }
        if (missing.isEmpty()) {
            streamingStatus = "Connecting..."
            connect(forMode, ipAddress)
        } else {
            requestPermissionLauncher.launch(missing.toTypedArray())
        }
    }

    private fun connect(forMode: Mode, ip: String) {
        if (forMode == Mode.SENSOR) connectSensorWebSocket(ip) else connectReceiverWebSocket(ip)
    }

    private fun connectSensorWebSocket(ip: String) {
        val url = "ws://$ip:$SENSOR_PORT/"
        val request = Request.Builder().url(url).build()

        webSocket = client.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                Log.d(TAG, "Sensor WebSocket Opened")
                streamingStatus = "Streaming connected to $url"
                isConnected = true
                runOnUiThread {
                    startCamera()
                    startAudio()
                }
            }

            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                Log.d(TAG, "Sensor WebSocket Closed")
                streamingStatus = "WebSocket Closed"
                isConnected = false
                stopStreaming()
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                Log.e(TAG, "Sensor WebSocket Failure", t)
                streamingStatus = "WebSocket Error: ${t.message}"
                isConnected = false
                stopStreaming()
            }
        })
    }

    private fun connectReceiverWebSocket(ip: String) {
        val url = "ws://$ip:$RECEIVER_PORT/"
        val request = Request.Builder().url(url).build()

        webSocket = client.newWebSocket(request, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                Log.d(TAG, "Receiver WebSocket Opened")
                streamingStatus = "Listening for alerts on $url"
                isConnected = true
            }

            override fun onMessage(webSocket: WebSocket, bytes: ByteString) {
                val data = bytes.toByteArray()
                if (data.isEmpty() || data[0] != 0x03.toByte()) return
                val json = String(data, 1, data.size - 1, Charsets.UTF_8)
                val alert = parseAlert(json) ?: return
                runOnUiThread {
                    latestAlert = alert
                    showAlertNotification(alert)
                }
            }

            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                Log.d(TAG, "Receiver WebSocket Closed")
                streamingStatus = "WebSocket Closed"
                isConnected = false
            }

            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                Log.e(TAG, "Receiver WebSocket Failure", t)
                streamingStatus = "WebSocket Error: ${t.message}"
                isConnected = false
            }
        })
    }

    private fun parseAlert(json: String): AlertPayload? {
        return try {
            val obj = JSONObject(json)
            val alertsObj = obj.optJSONObject("alerts")
            val alertsMap = mutableMapOf<String, String>()
            alertsObj?.keys()?.forEach { key -> alertsMap[key] = alertsObj.getString(key) }
            AlertPayload(
                id = obj.optString("id", ""),
                timestamp = obj.optString("timestamp", ""),
                confidence = obj.optDouble("confidence", 0.0),
                report = obj.optString("report", ""),
                alerts = alertsMap,
                location = obj.optString("location", ""),
            )
        } catch (e: Exception) {
            Log.e(TAG, "Failed to parse alert JSON", e)
            null
        }
    }

    @SuppressLint("MissingPermission")
    private fun showAlertNotification(alert: AlertPayload) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU &&
            ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS)
            != PackageManager.PERMISSION_GRANTED
        ) {
            return
        }
        val text = alert.alerts[languagePref] ?: alert.alerts["en"] ?: alert.report
        val notification = NotificationCompat.Builder(this, ALERT_CHANNEL_ID)
            .setSmallIcon(android.R.drawable.ic_dialog_alert)
            .setContentTitle("Elephant Alert — ${alert.location}")
            .setContentText(text)
            .setStyle(NotificationCompat.BigTextStyle().bigText(text))
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .setAutoCancel(true)
            .build()
        NotificationManagerCompat.from(this).notify(alert.id.hashCode(), notification)
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
                val currentTime = System.currentTimeMillis()
                // Throttle to roughly 15 FPS (1000ms / 15 = 66ms)
                if (currentTime - lastFrameTimeMs < 66) {
                    imageProxy.close()
                    return@setAnalyzer
                }

                // CRITICAL: Prevent OutOfMemoryError!
                // If the network is slow, OkHttp queues frames in RAM. If it exceeds 2MB, drop the frame!
                if ((webSocket?.queueSize() ?: 0L) > 2_000_000L) {
                    Log.w(TAG, "Network queue is too large! Dropping video frame.")
                    imageProxy.close()
                    return@setAnalyzer
                }

                lastFrameTimeMs = currentTime

                try {
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
                } catch (e: Error) {
                    Log.e(TAG, "Fatal Error processing frame (OOM)", e)
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
