package com.gajaalert.notify

import android.Manifest
import android.annotation.SuppressLint
import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.pm.PackageManager
import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.media.AudioAttributes
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.Settings
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.core.app.NotificationCompat
import androidx.core.app.NotificationManagerCompat
import androidx.core.content.ContextCompat
import com.gajaalert.notify.ui.theme.MobileTheme
import org.json.JSONObject

/**
 * Gaja-alert receiver: listens on ws://<laptop-ip>:9001 for 0x03 + JSON
 * elephant alerts broadcast by arm_server.py, surfaces a high-priority
 * notification (with sound) and shows the full incident report plus every
 * language the alert was translated into (Sarvam MCP picks the language set
 * per incident, so this list is whatever the payload actually contains, not
 * a fixed en/hi/ta trio).
 */
class MainActivity : ComponentActivity() {

    private val TAG = "GajaNotify"

    companion object {
        private const val RECEIVER_PORT = 9001
        // Notification-channel sound settings cannot be changed after Android creates
        // the channel. Use a new ID so existing installs migrate away from the old,
        // possibly silent channel.
        private const val ALERT_CHANNEL_ID = "elephant_alerts_v2"
        private const val LEGACY_ALERT_CHANNEL_ID = "elephant_alerts"
    }

    private data class AlertPayload(
        val id: String,
        val timestamp: String,
        val confidence: Double,
        val triggerSource: String,
        val report: String,
        val alerts: Map<String, String>,
        val location: String,
        val detectedSoundType: String?,
        val elephantVisibility: Boolean,
        val verificationStatus: String,
        val eventSeverity: String,
        val audioConfidence: Double?,
        val yoloConfidence: Double?,
    )

    private var isConnected by mutableStateOf(false)
    private var languagePref by mutableStateOf("en")
    private var latestAlert by mutableStateOf<AlertPayload?>(null)
    private var streamingStatus by mutableStateOf("Ready to connect")
    private var pendingIpAddress = ""

    private val serviceEvents = object : BroadcastReceiver() {
        override fun onReceive(context: Context?, intent: Intent?) {
            when (intent?.action) {
                AlertConnectionService.ACTION_STATUS -> {
                    streamingStatus = intent.getStringExtra(AlertConnectionService.EXTRA_STATUS)
                        ?: streamingStatus
                    isConnected = streamingStatus.startsWith("Listening") ||
                        streamingStatus.startsWith("Connecting") || streamingStatus.contains("retrying")
                }
                AlertConnectionService.ACTION_ALERT -> {
                    val alert = intent.getStringExtra(AlertConnectionService.EXTRA_JSON)
                        ?.let(::parseAlert) ?: return
                    latestAlert = alert
                    if (!alert.alerts.containsKey(languagePref)) {
                        languagePref = alert.alerts.keys.firstOrNull { it == "en" }
                            ?: alert.alerts.keys.firstOrNull() ?: languagePref
                    }
                    showAlertNotification(alert)
                }
            }
        }
    }

    private val requestPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
            if (granted) {
                streamingStatus = "Permission granted. Connecting..."
                connectReceiverWebSocket(pendingIpAddress)
            } else {
                streamingStatus = "Notification permission denied — alerts will still " +
                    "show on-screen but won't post a system notification."
                connectReceiverWebSocket(pendingIpAddress)
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        createNotificationChannel()
        registerReceiver(
            serviceEvents,
            IntentFilter().apply {
                addAction(AlertConnectionService.ACTION_STATUS)
                addAction(AlertConnectionService.ACTION_ALERT)
            },
            RECEIVER_NOT_EXPORTED,
        )
        setContent {
            MobileTheme {
                var ipAddress by remember { mutableStateOf("192.168.") }

                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    Column(
                        modifier = Modifier
                            .padding(innerPadding)
                            .fillMaxSize()
                            .verticalScroll(rememberScrollState())
                            .padding(16.dp),
                        horizontalAlignment = Alignment.CenterHorizontally,
                        verticalArrangement = Arrangement.Top
                    ) {
                        Text(text = "Gaja Alert — Receiver", fontWeight = FontWeight.Bold)
                        Spacer(modifier = Modifier.height(16.dp))
                        OutlinedTextField(
                            value = ipAddress,
                            onValueChange = { ipAddress = it },
                            label = { Text("Laptop Server IP Address") },
                            modifier = Modifier.fillMaxWidth(),
                            enabled = !isConnected
                        )
                        Spacer(modifier = Modifier.height(16.dp))
                        Button(
                            onClick = {
                                if (isConnected) disconnectWebSocket()
                                else checkPermissionAndConnect(ipAddress)
                            },
                            modifier = Modifier.fillMaxWidth()
                        ) {
                            Text(if (isConnected) "Disconnect" else "Listen for Alerts")
                        }
                        Spacer(modifier = Modifier.height(12.dp))
                        Text(text = streamingStatus)
                        Spacer(modifier = Modifier.height(16.dp))

                        latestAlert?.let { alert ->
                            val langs = alert.alerts.keys.toList()
                            if (langs.isNotEmpty()) {
                                LazyRow(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                                    items(langs) { lang ->
                                        ModeButton(
                                            label = lang.uppercase(),
                                            selected = languagePref == lang,
                                        ) { languagePref = lang }
                                    }
                                }
                                Spacer(modifier = Modifier.height(12.dp))
                            }
                            Card(modifier = Modifier.fillMaxWidth()) {
                                Column(modifier = Modifier.padding(16.dp)) {
                                    Row(verticalAlignment = Alignment.CenterVertically) {
                                        Text(
                                            text = "Elephant Alert — ${alert.location}",
                                            fontWeight = FontWeight.Bold,
                                            modifier = Modifier.weight(1f),
                                        )
                                        SeverityBadge(alert.eventSeverity)
                                    }
                                    Spacer(modifier = Modifier.height(4.dp))
                                    Text(text = "Confidence: ${"%.0f".format(alert.confidence * 100)}% " +
                                        "· source: ${alert.triggerSource} · ${alert.timestamp}")
                                    Spacer(modifier = Modifier.height(8.dp))
                                    Text(
                                        text = "Verification: ${alert.verificationStatus} " +
                                            "· visible: ${if (alert.elephantVisibility) "yes" else "no"}" +
                                            (alert.detectedSoundType?.let { " · sound: $it" } ?: ""),
                                    )
                                    Text(
                                        text = "Audio confidence: ${alert.audioConfidence
                                            ?.let { "%.0f%%".format(it * 100) } ?: "N/A"} " +
                                            "· YOLO confidence: ${alert.yoloConfidence
                                                ?.let { "%.0f%%".format(it * 100) } ?: "N/A"}",
                                    )
                                    Spacer(modifier = Modifier.height(12.dp))
                                    Text(
                                        text = alert.alerts[languagePref]
                                            ?: alert.alerts["en"]
                                            ?: alert.alerts.values.firstOrNull()
                                            ?: alert.report,
                                        fontWeight = FontWeight.Medium,
                                    )
                                    Spacer(modifier = Modifier.height(12.dp))
                                    Text(text = "Incident report", fontWeight = FontWeight.Bold)
                                    Spacer(modifier = Modifier.height(4.dp))
                                    Text(text = alert.report)
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    @Composable
    private fun ModeButton(label: String, selected: Boolean, onClick: () -> Unit) {
        if (selected) {
            Button(onClick = onClick) { Text(label) }
        } else {
            OutlinedButton(onClick = onClick) { Text(label) }
        }
    }

    @Composable
    private fun SeverityBadge(severity: String) {
        // Same high/medium/low -> red/amber/green convention as the
        // dashboard's status badges (gaja/dashboard.py).
        val color = when (severity.lowercase()) {
            "high" -> androidx.compose.ui.graphics.Color(0xFFEF4444)
            "medium" -> androidx.compose.ui.graphics.Color(0xFFF59E0B)
            else -> androidx.compose.ui.graphics.Color(0xFF10B981)
        }
        Text(
            text = severity.uppercase(),
            color = color,
            fontWeight = FontWeight.Bold,
        )
    }

    private fun createNotificationChannel() {
        val soundUri: Uri = Settings.System.DEFAULT_ALARM_ALERT_URI
            ?: Settings.System.DEFAULT_NOTIFICATION_URI
        val audioAttributes = AudioAttributes.Builder()
            .setUsage(AudioAttributes.USAGE_ALARM)
            .setContentType(AudioAttributes.CONTENT_TYPE_SONIFICATION)
            .build()
        val channel = NotificationChannel(
            ALERT_CHANNEL_ID, "Elephant Alerts", NotificationManager.IMPORTANCE_HIGH
        ).apply {
            description = "Elephant early-warning alerts from the Gaja edge pipeline"
            enableVibration(true)
            vibrationPattern = longArrayOf(0, 500, 250, 500)
            setSound(soundUri, audioAttributes)
        }
        getSystemService(NotificationManager::class.java).apply {
            createNotificationChannel(channel)
            deleteNotificationChannel(LEGACY_ALERT_CHANNEL_ID)
        }
    }

    private fun checkPermissionAndConnect(ipAddress: String) {
        pendingIpAddress = ipAddress
        streamingStatus = "Connecting..."
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU &&
            ContextCompat.checkSelfPermission(this, Manifest.permission.POST_NOTIFICATIONS)
            != PackageManager.PERMISSION_GRANTED
        ) {
            requestPermissionLauncher.launch(Manifest.permission.POST_NOTIFICATIONS)
        } else {
            connectReceiverWebSocket(ipAddress)
        }
    }

    private fun connectReceiverWebSocket(ip: String) {
        ContextCompat.startForegroundService(
            this,
            Intent(this, AlertConnectionService::class.java)
                .setAction(AlertConnectionService.ACTION_START)
                .putExtra(AlertConnectionService.EXTRA_IP, ip),
        )
        isConnected = true
    }

    private fun parseAlert(json: String): AlertPayload? {
        return try {
            val obj = JSONObject(json)
            val alertsObj = obj.optJSONObject("alerts") ?: obj.optJSONObject("notification_message")
            val alertsMap = mutableMapOf<String, String>()
            alertsObj?.keys()?.forEach { key -> alertsMap[key] = alertsObj.getString(key) }
            AlertPayload(
                id = obj.optString("id", ""),
                timestamp = obj.optString("timestamp", ""),
                confidence = obj.optDouble("confidence", 0.0),
                triggerSource = obj.optString("trigger_source", "unknown"),
                report = obj.optString("report", ""),
                alerts = alertsMap,
                location = obj.optString("location", ""),
                detectedSoundType = if (obj.isNull("detected_sound_type")) null
                    else obj.optString("detected_sound_type", null),
                elephantVisibility = obj.optBoolean("elephant_visibility", true),
                verificationStatus = obj.optString("verification_status", "confirmed"),
                eventSeverity = obj.optString("event_severity", "medium"),
                audioConfidence = if (obj.isNull("audio_confidence")) null
                    else obj.optDouble("audio_confidence").takeIf { !it.isNaN() },
                yoloConfidence = if (obj.isNull("yolo_confidence")) null
                    else obj.optDouble("yolo_confidence").takeIf { !it.isNaN() },
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
            .setStyle(NotificationCompat.BigTextStyle().bigText("$text\n\n${alert.report}"))
            .setPriority(NotificationCompat.PRIORITY_HIGH)
            .setCategory(NotificationCompat.CATEGORY_ALARM)
            .setDefaults(NotificationCompat.DEFAULT_SOUND or NotificationCompat.DEFAULT_VIBRATE)
            .setAutoCancel(true)
            .build()
        NotificationManagerCompat.from(this).notify(alert.id.hashCode(), notification)
    }

    private fun disconnectWebSocket() {
        startService(
            Intent(this, AlertConnectionService::class.java)
                .setAction(AlertConnectionService.ACTION_STOP)
        )
        isConnected = false
        streamingStatus = "Disconnected"
    }

    override fun onDestroy() {
        unregisterReceiver(serviceEvents)
        super.onDestroy()
    }
}
