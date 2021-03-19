package com.mtcnn_tflite.utils;

import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.media.Image;
import java.nio.ByteBuffer;

public class BitmapUtil {
    static final int kMaxChannelValue = 262143;
    static int[] cacheRgbBytes = new int[1440*2560];

    public static int[] convertImageToBitmap(Image image, byte[][] cachedYuvBytes) {
        if (cachedYuvBytes == null || cachedYuvBytes.length != 3) {
            cachedYuvBytes = new byte[3][];
        }
        Image.Plane[] planes = image.getPlanes();
        fillBytes(planes, cachedYuvBytes);

        final int yRowStride = planes[0].getRowStride();
        final int uvRowStride = planes[1].getRowStride();
        final int uvPixelStride = planes[1].getPixelStride();

        convertYUV420ToARGB8888(cachedYuvBytes[0], cachedYuvBytes[1], cachedYuvBytes[2],
                image.getWidth(), image.getHeight(), yRowStride, uvRowStride, uvPixelStride, cacheRgbBytes);
        return cacheRgbBytes;
    }

    private static void convertYUV420ToARGB8888(byte[] yData, byte[] uData, byte[] vData, int width, int height,
                                                int yRowStride, int uvRowStride, int uvPixelStride, int[] out) {
        int i = 0;
        for (int y = 0; y < height; y++) {
            int pY = yRowStride * y;
            int uv_row_start = uvRowStride * (y >> 1);
            int pU = uv_row_start;
            int pV = uv_row_start;

            for (int x = 0; x < width; x++) {
                int uv_offset = (x >> 1) * uvPixelStride;
                out[i++] = YUV2RGB(
                        convertByteToInt(yData, pY + x),
                        convertByteToInt(uData, pU + uv_offset),
                        convertByteToInt(vData, pV + uv_offset));
            }
        }
    }

    private static int convertByteToInt(byte[] arr, int pos) {
        return arr[pos] & 0xFF;
    }

    private static int YUV2RGB(int nY, int nU, int nV) {
        nY -= 16;
        nU -= 128;
        nV -= 128;
        if (nY < 0) nY = 0;

        int nR = (int) (1192 * nY + 1634 * nV);
        int nG = (int) (1192 * nY - 833 * nV - 400 * nU);
        int nB = (int) (1192 * nY + 2066 * nU);

        nR = Math.min(kMaxChannelValue, Math.max(0, nR));
        nG = Math.min(kMaxChannelValue, Math.max(0, nG));
        nB = Math.min(kMaxChannelValue, Math.max(0, nB));

        nR = (nR >> 10) & 0xff;
        nG = (nG >> 10) & 0xff;
        nB = (nB >> 10) & 0xff;

        return 0xff000000 | (nR << 16) | (nG << 8) | nB;
    }

    private static void fillBytes(final Image.Plane[] planes, final byte[][] yuvBytes) {
        for (int i = 0; i < planes.length; ++i) {
            final ByteBuffer buffer = planes[i].getBuffer();
            if (yuvBytes[i] == null || yuvBytes[i].length != buffer.capacity()) {
                yuvBytes[i] = new byte[buffer.capacity()];
            }
            buffer.get(yuvBytes[i]);
        }
    }

    public static Bitmap adjustBitmap(Bitmap bitmap, int angle, boolean isMirror){
        Matrix matrix = new Matrix();
        matrix.postScale(0.2f, 0.2f);
        matrix.postRotate(angle);

        if(isMirror) {
            matrix.postScale(-1, 1);
        }

        Bitmap resizedBitmap = Bitmap.createBitmap(bitmap, 0, 0,
                bitmap.getWidth(), bitmap.getHeight(), matrix, true);
        if (resizedBitmap != bitmap && bitmap != null && !bitmap.isRecycled())
        {
            bitmap.recycle();
            bitmap = null;
        }
        return resizedBitmap;
    }

    public static Bitmap resizeBitmap(Bitmap bitmap, float widthScale, float heightScale){
        Matrix matrix = new Matrix();
        matrix.postScale(widthScale, heightScale);

        Bitmap resizedBitmap = Bitmap.createBitmap(bitmap, 0, 0,
                bitmap.getWidth(), bitmap.getHeight(), matrix, true);
        if (resizedBitmap != bitmap && bitmap != null && !bitmap.isRecycled())
        {
            bitmap.recycle();
            bitmap = null;
        }
        return resizedBitmap;
    }

}