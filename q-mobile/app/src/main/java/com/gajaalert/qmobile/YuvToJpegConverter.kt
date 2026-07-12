package com.gajaalert.qmobile

import android.graphics.ImageFormat
import android.graphics.Rect
import android.graphics.YuvImage
import android.media.Image
import androidx.camera.core.ImageProxy
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer

object YuvToJpegConverter {
    fun imageProxyToJpeg(imageProxy: ImageProxy, quality: Int = 80): ByteArray {
        val yuvBytes = yuv420888ToNv21(imageProxy)
        val yuvImage = YuvImage(
            yuvBytes,
            ImageFormat.NV21,
            imageProxy.width,
            imageProxy.height,
            null
        )
        val out = ByteArrayOutputStream()
        yuvImage.compressToJpeg(Rect(0, 0, imageProxy.width, imageProxy.height), quality, out)
        return out.toByteArray()
    }

    private fun yuv420888ToNv21(image: ImageProxy): ByteArray {
        val nv21: ByteArray
        val yBuffer = image.planes[0].buffer
        val uBuffer = image.planes[1].buffer
        val vBuffer = image.planes[2].buffer

        val ySize = yBuffer.remaining()
        val uSize = uBuffer.remaining()
        val vSize = vBuffer.remaining()

        nv21 = ByteArray(ySize + image.width * image.height / 2)

        // U and V are swapped
        yBuffer.get(nv21, 0, ySize)
        vBuffer.get(nv21, ySize, vSize)
        uBuffer.get(nv21, ySize + vSize, uSize)

        // Note: The above simple buffer copy assumes the planes are contiguous NV21/NV12.
        // A more robust implementation handles row strides and pixel strides.
        // For standard CameraX YUV_420_888 to NV21 conversion:
        val width = image.width
        val height = image.height
        val yRowStride = image.planes[0].rowStride
        val uvRowStride = image.planes[1].rowStride
        val uvPixelStride = image.planes[1].pixelStride

        var pos = 0
        
        if (yRowStride == width) {
            yBuffer.position(0)
            yBuffer.get(nv21, 0, ySize)
            pos += ySize
        } else {
            var yBufferPos = 0
            for (row in 0 until height) {
                yBuffer.position(yBufferPos)
                yBuffer.get(nv21, pos, width)
                yBufferPos += yRowStride
                pos += width
            }
        }

        if (uvRowStride == width && uvPixelStride == 2) {
            vBuffer.position(0)
            vBuffer.get(nv21, pos, vSize) // V and U are interleaved
        } else {
            var vBufferPos = 0
            var uBufferPos = 0
            for (row in 0 until height / 2) {
                for (col in 0 until width / 2) {
                    vBuffer.position(vBufferPos + col * uvPixelStride)
                    nv21[pos++] = vBuffer.get()
                    uBuffer.position(uBufferPos + col * uvPixelStride)
                    nv21[pos++] = uBuffer.get()
                }
                vBufferPos += uvRowStride
                uBufferPos += uvRowStride
            }
        }
        return nv21
    }
}
