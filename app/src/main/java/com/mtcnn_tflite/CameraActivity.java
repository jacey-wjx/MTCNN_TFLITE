package com.mtcnn_tflite;

import androidx.annotation.NonNull;
import androidx.annotation.RequiresApi;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import android.Manifest;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.ImageFormat;
import android.graphics.Paint;
import android.graphics.PixelFormat;
import android.graphics.PorterDuff;
import android.graphics.Rect;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.media.Image;
import android.media.ImageReader;
import android.os.Build;
import android.os.Bundle;
import android.os.Handler;
import android.os.HandlerThread;
import android.util.Log;
import android.util.Size;
import android.view.Surface;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.TextureView;
import android.widget.RadioGroup;
import android.widget.TextView;
import android.widget.Toast;

import com.mtcnn_tflite.model.Box;
import com.mtcnn_tflite.model.MTCNN;
import com.mtcnn_tflite.utils.BitmapUtil;
import com.mtcnn_tflite.utils.Permission;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Vector;
import java.util.concurrent.Semaphore;

public class CameraActivity extends AppCompatActivity implements RadioGroup.OnCheckedChangeListener {
    private static final String TAG="CameraActivity";

    private TextureView mPreview;
    private TextView mTimeCost;
    private RadioGroup mRadioGroup;
    private AssetManager manager;

    // Detector
    private MTCNN mtcnn;
    private SurfaceHolder mSurfaceHolder;
    private Paint paintRect;
    private Canvas canvas;

    // Camera2
    private Size mPreviewSize;
    private String mCameraId;
    private CameraDevice mCameraDevice;
    private CaptureRequest.Builder mCaptureRequestBuilder;
    private CaptureRequest mCaptureRequest;
    private CameraCaptureSession mPreviewSession;
    private ImageReader mImageReader;

    private HandlerThread mCameraThread;
    private Handler mCameraHandler;


    private TextureView.SurfaceTextureListener mTextureListener = new TextureView.SurfaceTextureListener() {
        @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
        @Override
        public void onSurfaceTextureAvailable(@NonNull SurfaceTexture surfaceTexture, int i, int i1) {
            setupCamera(i, i1);
            openCamera();
        }

        @Override
        public void onSurfaceTextureSizeChanged(@NonNull SurfaceTexture surfaceTexture, int i, int i1) {

        }

        @Override
        public boolean onSurfaceTextureDestroyed(@NonNull SurfaceTexture surfaceTexture) {
            return false;
        }

        @Override
        public void onSurfaceTextureUpdated(@NonNull SurfaceTexture surfaceTexture) {

        }
    };

    @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
    private final CameraDevice.StateCallback mStateCallback = new CameraDevice.StateCallback() {
        @Override
        public void onOpened(@NonNull CameraDevice cameraDevice) {
            mCameraDevice = cameraDevice;
            startPreview();
        }

        @Override
        public void onDisconnected(@NonNull CameraDevice cameraDevice) {

        }

        @Override
        public void onError(@NonNull CameraDevice cameraDevice, int i) {

        }
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_camera);

        mPreview = findViewById(R.id.preview);
        SurfaceView surfaceView = findViewById(R.id.detector_surfaceview);
        mTimeCost = findViewById(R.id.fps);

        mRadioGroup = findViewById(R.id.choices);
        mRadioGroup.setOnCheckedChangeListener(this);

        manager = this.getAssets();
        mtcnn = new MTCNN(this, manager);

        surfaceView.setZOrderOnTop(true);
        surfaceView.getHolder().setFormat(PixelFormat.TRANSPARENT);
        mSurfaceHolder = surfaceView.getHolder();
    }


    @Override
    protected void onStart() {
        super.onStart();
//        Permission.checkPermission(this);
        paintRect = new Paint();
        paintRect.setColor(Color.GREEN);
        paintRect.setStyle(Paint.Style.STROKE);
        paintRect.setStrokeWidth(5);
        canvas = new Canvas();
        startCameraThread();
    }

    @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
    @Override
    protected void onResume() {
        super.onResume();
        if (!mPreview.isAvailable()) {
            mPreview.setSurfaceTextureListener(mTextureListener);
        } else {
            openCamera();
        }
    }

    @Override
    protected void onStop() {
        super.onStop();
        if (mPreviewSession != null) {
            mPreviewSession.close();
            mPreviewSession = null;
        }

        if (mCameraDevice != null) {
            mCameraDevice.close();
            mCameraDevice = null;
        }

        if (mImageReader != null) {
            mImageReader.close();
            mImageReader = null;
        }

        mCameraThread.quitSafely();
        try {
            mCameraThread.join();
            mCameraThread = null;
            mCameraHandler = null;
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
    }


    private void startCameraThread() {
        mCameraThread = new HandlerThread("CameraThread");
        mCameraThread.start();
        mCameraHandler = new Handler(mCameraThread.getLooper());
    }

    private final ImageReader.OnImageAvailableListener mOnImageAvailableListener = new ImageReader.OnImageAvailableListener() {
        @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
        @Override
        public void onImageAvailable(ImageReader imageReader) {
            Image image = imageReader.acquireLatestImage();
            if(image==null) return ;
            if(mtcnn.loadSuccessful) {
                mCameraHandler.post(new Runnable() {
                    @Override
                    public void run() {
                        int width = image.getWidth(), height = image.getHeight();
                        byte[][] cachedYuvBytes = new byte[3][];
                        int[] cacheRgbBytes = BitmapUtil.convertImageToBitmap(image, cachedYuvBytes);
                        image.close();
                        Bitmap bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
                        bitmap.setPixels(cacheRgbBytes, 0, width, 0, 0, width, height);
                        bitmap = BitmapUtil.adjustBitmap(bitmap, 270, true);
                        long start = System.currentTimeMillis();
                        Vector<Box> boxes = mtcnn.detectFaces(bitmap);
                        long end = System.currentTimeMillis();
                        drawBoxes(boxes, bitmap.getWidth(), bitmap.getHeight(), end-start);
                    }
                });
            }else {
                image.close();
            }
        }
    };

    @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
    private void setupCamera(int width, int height) {
        CameraManager cameraManager = (CameraManager) getSystemService(CAMERA_SERVICE);
        try {
            for (String cameraId : cameraManager.getCameraIdList()) {
                CameraCharacteristics characteristics = cameraManager.getCameraCharacteristics(cameraId);

                if (characteristics.get(CameraCharacteristics.LENS_FACING) == CameraCharacteristics.LENS_FACING_BACK)
                    continue;

                StreamConfigurationMap map = characteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);

                mPreviewSize = getOptimalSize(map.getOutputSizes(SurfaceTexture.class), width, height);
                mSurfaceHolder.setFixedSize(mPreview.getWidth(), mPreview.getHeight());

                mImageReader = ImageReader.newInstance(mPreviewSize.getWidth(), mPreviewSize.getHeight(), ImageFormat.YUV_420_888, 2);
                mImageReader.setOnImageAvailableListener(mOnImageAvailableListener, mCameraHandler);
                mCameraId = cameraId;
                break;
            }
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }

    @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
    private void openCamera() {
        CameraManager cameraManager = (CameraManager) getSystemService(CAMERA_SERVICE);
        if (ActivityCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            Permission.checkPermission(this);
            return;
        }
        try {
            cameraManager.openCamera(mCameraId, mStateCallback, null);
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }

    @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
    private void startPreview(){
        SurfaceTexture mSurfaceTexture = mPreview.getSurfaceTexture();
        mSurfaceTexture.setDefaultBufferSize(mPreviewSize.getWidth(), mPreviewSize.getHeight());
        Surface previewSurface = new Surface(mSurfaceTexture);
        try {
            mCaptureRequestBuilder = mCameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
            mCaptureRequestBuilder.addTarget(previewSurface);
            // get real-time frame data
            mCaptureRequestBuilder.addTarget(mImageReader.getSurface());
            mCameraDevice.createCaptureSession(Arrays.asList(previewSurface, mImageReader.getSurface()), new CameraCaptureSession.StateCallback() {
                @Override
                public void onConfigured(CameraCaptureSession session) {
                    try {
                        mCaptureRequest = mCaptureRequestBuilder.build();
                        mPreviewSession = session;
                        mPreviewSession.setRepeatingRequest(mCaptureRequest, null, mCameraHandler);
                    } catch (CameraAccessException e) {
                        e.printStackTrace();
                    }
                }

                @Override
                public void onConfigureFailed(CameraCaptureSession session) {

                }
            }, mCameraHandler);
        } catch (CameraAccessException e) {
            e.printStackTrace();
        }
    }

    @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
    private Size getOptimalSize(Size[] sizeMap, int width, int height) {
        List<Size> sizeList = new ArrayList<>();
        for (Size option : sizeMap) {
            if (width > height) {
                if (option.getWidth() > width && option.getHeight() > height) {
                    sizeList.add(option);
                }
            } else {
                if (option.getWidth() > height && option.getHeight() > width) {
                    sizeList.add(option);
                }
            }
        }
        if (sizeList.size() > 0) {
            return Collections.min(sizeList, new Comparator<Size>() {
                @Override
                public int compare(Size lhs, Size rhs) {
                    return Long.signum(lhs.getWidth() * lhs.getHeight() - rhs.getWidth() * rhs.getHeight());
                }
            });
        }
        return sizeMap[0];
    }

    @Override
    public void onCheckedChanged(RadioGroup radioGroup, int i) {
        String type = "cpu";
        switch (i){
            case R.id.cpu:
                Log.i(TAG, "cpu");
                break;
            case R.id.gpu:
                Log.i(TAG, "gpu");
                type="gpu";
                break;
            case R.id.dsp:
                Log.i(TAG, "dsp");
                type="dsp";
                break;
            default:
                break;
        }

        try {
            mtcnn.loadModel(type);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void drawBoxes(Vector<Box> boxes, int width, int height, long timeCost){
        runOnUiThread(new Runnable() {
            @RequiresApi(api = Build.VERSION_CODES.LOLLIPOP)
            @Override
            public void run() {
                mTimeCost.setText("Time cost per frame: "+timeCost);
                ClearDraw();
                if(boxes.size()!=0){
                    Box box = boxes.get(0);
                    float widthScale = (float)mPreview.getWidth()/(float)width;
                    float heightScale = (float)mPreview.getHeight()/(float)height;
                    box.resizeBox(widthScale, heightScale);
                    box.toSquareShape();
                    Rect rect = box.transform2Rect();

                    canvas = mSurfaceHolder.lockCanvas();
                    canvas.drawRect(rect, paintRect);
                    mSurfaceHolder.unlockCanvasAndPost(canvas);
                    Log.i(TAG, "finish drawing");
                }
            }
        });
    }

    private void ClearDraw(){
        try{
            canvas = mSurfaceHolder.lockCanvas(null);
            canvas.drawColor(Color.WHITE);
            canvas.drawColor(Color.TRANSPARENT, PorterDuff.Mode.SRC);
        }catch(Exception e){
            e.printStackTrace();
        }finally{
            if(canvas != null){
                mSurfaceHolder.unlockCanvasAndPost(canvas);
            }
        }
    }
}