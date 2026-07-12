package com.gajaalert.notify

import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.os.IBinder
import androidx.core.app.NotificationCompat
import android.media.AudioAttributes
import android.provider.Settings
import org.json.JSONObject
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.Response
import okhttp3.WebSocket
import okhttp3.WebSocketListener
import okio.ByteString
import java.util.concurrent.TimeUnit

class AlertConnectionService : Service() {
    companion object {
        const val ACTION_START = "com.gajaalert.notify.START"
        const val ACTION_STOP = "com.gajaalert.notify.STOP"
        const val ACTION_ALERT = "com.gajaalert.notify.ALERT"
        const val ACTION_STATUS = "com.gajaalert.notify.STATUS"
        const val EXTRA_IP = "ip"
        const val EXTRA_JSON = "json"
        const val EXTRA_STATUS = "status"
        private const val SERVICE_CHANNEL = "alert_connection"
        private const val SERVICE_NOTIFICATION = 41
        private const val ALERT_CHANNEL = "elephant_alerts_v2"
        private const val PORT = 9001
    }

    private val client = OkHttpClient.Builder().pingInterval(25, TimeUnit.SECONDS).build()
    private var socket: WebSocket? = null
    private var ip = ""
    private var stoppedByUser = false
    private var retryMs = 2_000L

    override fun onCreate() {
        super.onCreate()
        getSystemService(NotificationManager::class.java).createNotificationChannel(
            NotificationChannel(SERVICE_CHANNEL, "Alert connection", NotificationManager.IMPORTANCE_LOW)
        )
        val alarmSound = Settings.System.DEFAULT_ALARM_ALERT_URI
            ?: Settings.System.DEFAULT_NOTIFICATION_URI
        getSystemService(NotificationManager::class.java).createNotificationChannel(
            NotificationChannel(ALERT_CHANNEL, "Elephant Alerts", NotificationManager.IMPORTANCE_HIGH).apply {
                enableVibration(true)
                vibrationPattern = longArrayOf(0, 500, 250, 500)
                setSound(alarmSound, AudioAttributes.Builder().setUsage(AudioAttributes.USAGE_ALARM).build())
            }
        )
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_STOP) {
            stoppedByUser = true
            socket?.close(1000, "User disconnected")
            stopForeground(STOP_FOREGROUND_REMOVE)
            stopSelf()
            return START_NOT_STICKY
        }
        intent?.getStringExtra(EXTRA_IP)?.trim()?.takeIf { it.isNotEmpty() }?.let { ip = it }
        if (ip.isEmpty()) return START_NOT_STICKY
        stoppedByUser = false
        startForeground(SERVICE_NOTIFICATION, connectionNotification("Connecting to $ip…"))
        connect()
        return START_REDELIVER_INTENT
    }

    private fun connect() {
        socket?.cancel()
        val url = "ws://$ip:$PORT/"
        status("Connecting to $url")
        socket = client.newWebSocket(Request.Builder().url(url).build(), object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                retryMs = 2_000L
                status("Listening for alerts on $url")
            }

            override fun onMessage(webSocket: WebSocket, bytes: ByteString) {
                val data = bytes.toByteArray()
                if (data.isEmpty() || data[0] != 0x03.toByte()) return
                val json = String(data, 1, data.size - 1, Charsets.UTF_8)
                showAlert(json)
                sendBroadcast(Intent(ACTION_ALERT).setPackage(packageName).putExtra(EXTRA_JSON, json))
            }

            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) = reconnect()
            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) = reconnect()
        })
    }

    private fun reconnect() {
        if (stoppedByUser) return
        status("Connection lost — retrying…")
        val delay = retryMs
        retryMs = (retryMs * 2).coerceAtMost(60_000L)
        mainExecutor.execute { android.os.Handler(mainLooper).postDelayed({ connect() }, delay) }
    }

    private fun status(value: String) {
        getSystemService(NotificationManager::class.java)
            .notify(SERVICE_NOTIFICATION, connectionNotification(value))
        sendBroadcast(Intent(ACTION_STATUS).setPackage(packageName).putExtra(EXTRA_STATUS, value))
    }

    private fun connectionNotification(text: String): android.app.Notification {
        val openApp = PendingIntent.getActivity(
            this, 0, Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        return NotificationCompat.Builder(this, SERVICE_CHANNEL)
            .setSmallIcon(android.R.drawable.ic_popup_sync)
            .setContentTitle("Gaja Alert is active")
            .setContentText(text)
            .setContentIntent(openApp)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .build()
    }

    private fun showAlert(json: String) {
        try {
            val obj = JSONObject(json)
            val messages = obj.optJSONObject("alerts") ?: obj.optJSONObject("notification_message")
            val text = messages?.optString("en")?.takeIf { it.isNotBlank() }
                ?: messages?.keys()?.asSequence()?.firstOrNull()?.let { messages.optString(it) }
                ?: obj.optString("report", "Elephant detected nearby")
            val report = obj.optString("report", "")
            val id = obj.optString("id", json.hashCode().toString()).hashCode()
            val notification = NotificationCompat.Builder(this, ALERT_CHANNEL)
                .setSmallIcon(android.R.drawable.ic_dialog_alert)
                .setContentTitle("Elephant Alert — ${obj.optString("location", "nearby")}")
                .setContentText(text)
                .setStyle(NotificationCompat.BigTextStyle().bigText("$text\n\n$report"))
                .setPriority(NotificationCompat.PRIORITY_HIGH)
                .setCategory(NotificationCompat.CATEGORY_ALARM)
                .setDefaults(NotificationCompat.DEFAULT_SOUND or NotificationCompat.DEFAULT_VIBRATE)
                .setAutoCancel(true)
                .build()
            getSystemService(NotificationManager::class.java).notify(id, notification)
        } catch (_: Exception) {
            // Ignore malformed payloads; the connection remains active for the next alert.
        }
    }

    override fun onDestroy() {
        socket?.cancel()
        client.dispatcher.executorService.shutdown()
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null
}
