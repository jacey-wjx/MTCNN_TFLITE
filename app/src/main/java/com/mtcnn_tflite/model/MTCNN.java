package com.mtcnn_tflite.model;

import android.content.Context;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Point;
import android.util.Log;

import com.mtcnn_tflite.utils.MyUtil;

import org.tensorflow.lite.HexagonDelegate;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;
import java.util.Vector;

public class MTCNN {
    private final String TAG="MTCNN";
    // parameter
    private float factor = 0.709f;
    private float pNetThreshold = 0.6f;
    private float rNetThreshold = 0.7f;
    private float oNetThreshold = 0.7f;
    private int minFaceSize = 20;

    // model path
    private static final String MODEL_FILE_PNET = "p_net.tflite";
    private static final String MODEL_FILE_RNET = "r_net.tflite";
    private static final String MODEL_FILE_ONET = "o_net.tflite";

    // model interpreters
    private Interpreter pInterpreter;
    private Interpreter rInterpreter;
    private Interpreter oInterpreter;

    // delegates
    private GpuDelegate gpuDelegate;
    private HexagonDelegate dspDelegate;

    private AssetManager mAssetManager;
    private Context mContext;
    public boolean loadSuccessful;

    public MTCNN(Context context, AssetManager assetManager){
        mAssetManager = assetManager;
        mContext = context;

        gpuDelegate=null;
        dspDelegate=null;

        pInterpreter=null;
        rInterpreter=null;
        oInterpreter=null;
        loadSuccessful = false;
    }

    public synchronized void loadModel(String type) throws IOException {
        if(pInterpreter!=null){
            pInterpreter.close();
            pInterpreter=null;
        }
        if(rInterpreter!=null){
            rInterpreter.close();
            rInterpreter=null;
        }
        if(oInterpreter!=null){
            oInterpreter.close();
            oInterpreter=null;
        }

        Interpreter.Options options = new Interpreter.Options();
        switch (type){
            case "cpu":
                if(gpuDelegate!=null){
                    gpuDelegate.close();
                    gpuDelegate=null;
                }
                if(dspDelegate!=null){
                    dspDelegate.close();
                    dspDelegate=null;
                }
                break;
            case "gpu":
                if(dspDelegate!=null){
                    dspDelegate.close();
                    dspDelegate=null;
                }
                gpuDelegate = new GpuDelegate();
                options.addDelegate(gpuDelegate);
                break;
            case "dsp":
                if(gpuDelegate!=null){
                    gpuDelegate.close();
                    gpuDelegate=null;
                }
                dspDelegate = new HexagonDelegate(mContext);
                options.addDelegate(dspDelegate);
                break;
            default:
                break;
        }

        // load tflite model
        pInterpreter = new Interpreter(MyUtil.loadModelFile(mAssetManager, MODEL_FILE_PNET), options);
        rInterpreter = new Interpreter(MyUtil.loadModelFile(mAssetManager, MODEL_FILE_RNET), options);
        oInterpreter = new Interpreter(MyUtil.loadModelFile(mAssetManager, MODEL_FILE_ONET), options);

        Log.i(TAG, "Load Model Successfully");
        loadSuccessful = true;
    }


    /**
     * Detect Faces
     * @param bitmap input image
     * @return box list
     */
    public synchronized Vector<Box> detectFaces(Bitmap bitmap) {
        if(bitmap==null){
            Log.e(TAG, "Image not valid!");
            return null;
        }

        Vector<Box> boxes;
        try {
            // PNet
            boxes = pNet(bitmap);
            square_limit(boxes, bitmap.getWidth(), bitmap.getHeight());

            // RNet
            boxes = rNet(bitmap, boxes);
            square_limit(boxes, bitmap.getWidth(), bitmap.getHeight());

            // ONet
            boxes = oNet(bitmap, boxes);
        } catch (IllegalArgumentException e) {
            e.printStackTrace();
            boxes = new Vector<>();
        }
        return boxes;
    }

    private void square_limit(Vector<Box> boxes, int w, int h) {
        // square
        for (int i = 0; i < boxes.size(); i++) {
            boxes.get(i).toSquareShape();
            boxes.get(i).limitSquare(w, h);
        }
    }


    private Vector<Box> pNet(Bitmap bitmap) {
        int whMin = Math.min(bitmap.getWidth(), bitmap.getHeight());
        float currentFaceSize = minFaceSize; // currentFaceSize=minSize/(factor^k) k=0,1,2... until excced whMin
        Vector<Box> totalBoxes = new Vector<>();
        // Compute image pyramid and feed into PNet
        while (currentFaceSize <= whMin) {
            float scale = 12.0f / currentFaceSize;

            // Image Resize
            Bitmap bm = MyUtil.bitmapResize(bitmap, scale);
            int w = bm.getWidth();
            int h = bm.getHeight();

            // RUN PNet Model
            int outW = (int) (Math.ceil(w * 0.5 - 5) + 0.5);
            int outH = (int) (Math.ceil(h * 0.5 - 5) + 0.5);
            float[][][][] prob1 = new float[1][outW][outH][2];
            float[][][][] conv4_2_BiasAdd = new float[1][outW][outH][4];
            pNetForward(bm, prob1, conv4_2_BiasAdd);
            prob1 = MyUtil.transposeBatch(prob1);
            conv4_2_BiasAdd = MyUtil.transposeBatch(conv4_2_BiasAdd);

            // (3)数据解析
            Vector<Box> curBoxes = new Vector<>();
            generateBoxes(prob1, conv4_2_BiasAdd, scale, curBoxes);

            // (4)nms 0.5
            nms(curBoxes, 0.5f, "Union");

            // (5)add to totalBoxes
            for (int i = 0; i < curBoxes.size(); i++)
                if (!curBoxes.get(i).deleted)
                    totalBoxes.addElement(curBoxes.get(i));

            // Face Size等比递增
            currentFaceSize /= factor;
        }

        // NMS 0.7
        nms(totalBoxes, 0.7f, "Union");

        // BBR
        BoundingBoxReggression(totalBoxes);

        return updateBoxes(totalBoxes);
    }

    private void pNetForward(Bitmap bitmap, float[][][][] prob1, float[][][][] conv4_2_BiasAdd) {
        float[][][] img = MyUtil.normalizeImage(bitmap);
        float[][][][] pNetIn = new float[1][][][];
        pNetIn[0] = img;
        pNetIn = MyUtil.transposeBatch(pNetIn);

        Map<Integer, Object> outputs = new HashMap<>();

        outputs.put(pInterpreter.getOutputIndex("StatefulPartitionedCall:1"), prob1);
        outputs.put(pInterpreter.getOutputIndex("StatefulPartitionedCall:0"), conv4_2_BiasAdd);

        Object[] inputs = new Object[]{pNetIn};
        pInterpreter.runForMultipleInputsOutputs(inputs, outputs);
    }

    private int generateBoxes(float[][][][] prob1, float[][][][] conv4_2_BiasAdd, float scale, Vector<Box> boxes) {
        int h = prob1[0].length;
        int w = prob1[0][0].length;

        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                float score = prob1[0][y][x][1];
                // only accept prob >threadshold(0.6 here)
                if (score > pNetThreshold) {
                    Box box = new Box();
                    // core
                    box.score = score;
                    // box
                    box.box[0] = Math.round(x * 2 / scale);
                    box.box[1] = Math.round(y * 2 / scale);
                    box.box[2] = Math.round((x * 2 + 11) / scale);
                    box.box[3] = Math.round((y * 2 + 11) / scale);
                    // bbr
                    for (int i = 0; i < 4; i++) {
                        box.bbr[i] = conv4_2_BiasAdd[0][y][x][i];
                    }
                    // add
                    boxes.addElement(box);
                }
            }
        }
        return 0;
    }

    private void nms(Vector<Box> boxes, float threshold, String method) {
        // NMS.两两比对
        // int delete_cnt = 0;
        for (int i = 0; i < boxes.size(); i++) {
            Box box = boxes.get(i);
            if (!box.deleted) {
                // score<0表示当前矩形框被删除
                for (int j = i + 1; j < boxes.size(); j++) {
                    Box box2 = boxes.get(j);
                    if (!box2.deleted) {
                        int x1 = Math.max(box.box[0], box2.box[0]);
                        int y1 = Math.max(box.box[1], box2.box[1]);
                        int x2 = Math.min(box.box[2], box2.box[2]);
                        int y2 = Math.min(box.box[3], box2.box[3]);
                        if (x2 < x1 || y2 < y1) continue;
                        int areaIoU = (x2 - x1 + 1) * (y2 - y1 + 1);
                        float iou = 0f;
                        if (method.equals("Union"))
                            iou = 1.0f * areaIoU / (box.area() + box2.area() - areaIoU);
                        else if (method.equals("Min"))
                            iou = 1.0f * areaIoU / (Math.min(box.area(), box2.area()));
                        if (iou >= threshold) {
                            if (box.score > box2.score)
                                box2.deleted = true;
                            else
                                box.deleted = true;
                        }
                    }
                }
            }
        }
    }

    private void BoundingBoxReggression(Vector<Box> boxes) {
        for (int i = 0; i < boxes.size(); i++)
            boxes.get(i).calibrate();
    }

    private Vector<Box> rNet(Bitmap bitmap, Vector<Box> boxes) {
        if(boxes.size()==0) return boxes;

        // RNet Input Init
        int num = boxes.size();
        float[][][][] rNetIn = new float[num][24][24][3];
        for (int i = 0; i < num; i++) {
            float[][][] curCrop = MyUtil.cropAndResize(bitmap, boxes.get(i), 24);
            curCrop = MyUtil.transposeImage(curCrop);
            rNetIn[i] = curCrop;
        }

        // Run RNet
        rNetForward(rNetIn, boxes);

        // RNetThreshold
        for (int i = 0; i < num; i++) {
            if (boxes.get(i).score < rNetThreshold) {
                boxes.get(i).deleted = true;
            }
        }

        // Nms
        nms(boxes, 0.7f, "Union");
        BoundingBoxReggression(boxes);
        return updateBoxes(boxes);
    }

    private void rNetForward(float[][][][] rNetIn, Vector<Box> boxes) {
        int num = rNetIn.length;
        float[][] prob1 = new float[num][2];
        float[][] conv5_2_conv5_2 = new float[num][4];

        Map<Integer, Object> outputs = new HashMap<>();

        outputs.put(rInterpreter.getOutputIndex("StatefulPartitionedCall:1"), prob1);
        outputs.put(rInterpreter.getOutputIndex("StatefulPartitionedCall:0"), conv5_2_conv5_2);

        rInterpreter.runForMultipleInputsOutputs(new Object[]{rNetIn}, outputs);

        // 转换
        for (int i = 0; i < num; i++) {
            boxes.get(i).score = prob1[i][1];
            for (int j = 0; j < 4; j++) {
                boxes.get(i).bbr[j] = conv5_2_conv5_2[i][j];
            }
        }
    }

    private Vector<Box> oNet(Bitmap bitmap, Vector<Box> boxes) {
        if(boxes.size()==0) return boxes;

        // ONet Input Init
        int num = boxes.size();
        float[][][][] oNetIn = new float[num][48][48][3];
        for (int i = 0; i < num; i++) {
            float[][][] curCrop = MyUtil.cropAndResize(bitmap, boxes.get(i), 48);
            curCrop = MyUtil.transposeImage(curCrop);
            oNetIn[i] = curCrop;
        }

        // Run ONet
        oNetForward(oNetIn, boxes);
        // ONetThreshold
        for (int i = 0; i < num; i++) {
            if (boxes.get(i).score < oNetThreshold) {
                boxes.get(i).deleted = true;
            }
        }
        BoundingBoxReggression(boxes);
        // Nms
        nms(boxes, 0.7f, "Min");
        return updateBoxes(boxes);
    }

    private void oNetForward(float[][][][] oNetIn, Vector<Box> boxes) {
        int num = oNetIn.length;
        float[][] prob1 = new float[num][2];
        float[][] conv6_2_conv6_2 = new float[num][4];
        float[][] conv6_3_conv6_3 = new float[num][10];

        Map<Integer, Object> outputs = new HashMap<>();
        outputs.put(oInterpreter.getOutputIndex("StatefulPartitionedCall:2"), prob1);
        outputs.put(oInterpreter.getOutputIndex("StatefulPartitionedCall:0"), conv6_2_conv6_2);
        outputs.put(oInterpreter.getOutputIndex("StatefulPartitionedCall:1"), conv6_3_conv6_3);
        oInterpreter.runForMultipleInputsOutputs(new Object[]{oNetIn}, outputs);

        // 转换
        for (int i = 0; i < num; i++) {
            // prob
            boxes.get(i).score = prob1[i][1];
            // bias
            for (int j = 0; j < 4; j++) {
                boxes.get(i).bbr[j] = conv6_2_conv6_2[i][j];
            }
            // landmark
            for (int j = 0; j < 5; j++) {
                int x = Math.round(boxes.get(i).left() + (conv6_3_conv6_3[i][j] * boxes.get(i).width()));
                int y = Math.round(boxes.get(i).top() + (conv6_3_conv6_3[i][j + 5] * boxes.get(i).height()));
                boxes.get(i).landmark[j] = new Point(x, y);
            }
        }
    }

    public static Vector<Box> updateBoxes(Vector<Box> boxes) {
        Vector<Box> b = new Vector<>();
        for (int i = 0; i < boxes.size(); i++) {
            if (!boxes.get(i).deleted) {
                b.addElement(boxes.get(i));
            }
        }
        return b;
    }

}