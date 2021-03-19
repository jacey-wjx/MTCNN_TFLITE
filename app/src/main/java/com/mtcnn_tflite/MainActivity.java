package com.mtcnn_tflite;

import androidx.appcompat.app.AppCompatActivity;

import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.graphics.PixelFormat;
import android.graphics.Point;
import android.graphics.Rect;
import android.graphics.drawable.Drawable;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.RadioGroup;
import android.widget.TextView;
import android.widget.Toast;

import com.mtcnn_tflite.model.Align;
import com.mtcnn_tflite.model.Box;
import com.mtcnn_tflite.model.MTCNN;
import com.mtcnn_tflite.utils.MyUtil;

import java.io.IOException;
import java.io.InputStream;
import java.util.Vector;

public class MainActivity extends AppCompatActivity implements RadioGroup.OnCheckedChangeListener {
    private final String TAG="MainActivity";

    private ImageView oriImage;
    private ImageView faceImage;
    private RadioGroup radioGroup;
    private AssetManager manager;
    private TextView timeTV;

    private MTCNN mtcnn;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        oriImage = findViewById(R.id.imageView);
        faceImage = findViewById(R.id.imageView2);
        timeTV = findViewById(R.id.textView);

        radioGroup = findViewById(R.id.image_choices);
        radioGroup.setOnCheckedChangeListener(this);

        manager = this.getAssets();
        mtcnn = new MTCNN(this, manager);

        try{
            InputStream is = manager.open("no-faces.jpg");
            Bitmap bm = BitmapFactory.decodeStream(is);
            is.close();
            oriImage.setImageBitmap(bm);
        }catch (IOException e){
            e.printStackTrace();
        }
    }

    @Override
    public void onCheckedChanged(RadioGroup radioGroup, int i) {
        String type = "cpu";
        switch (i){
            case R.id.img_cpu:
                Log.i(TAG, "cpu");
                break;
            case R.id.img_gpu:
                Log.i(TAG, "gpu");
                type="gpu";
                break;
            case R.id.img_dsp:
                Log.i(TAG, "dsp");
                type="dsp";
                break;
            default:
                break;
        }
        try {
            mtcnn.loadModel(type);
            detectFace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void detectFace(){
        Bitmap ori = getBitmap(oriImage.getDrawable());
        Bitmap input = ori.copy(ori.getConfig(), false);

        long start = System.currentTimeMillis();
        Vector<Box> boxes = mtcnn.detectFaces(input);
        long end = System.currentTimeMillis();
        timeTV.setText("the time cost is :"+ (end-start));
        if(boxes.size()==0){
            Toast.makeText(MainActivity.this, "no faces detected", Toast.LENGTH_LONG).show();
            return ;
        }
        Box box = boxes.get(0);

        input = Align.face_align(input, box.landmark);
        boxes = mtcnn.detectFaces(input);
        box = boxes.get(0);
        box.toSquareShape();
        box.limitSquare(input.getWidth(), input.getHeight());
        Rect rect = box.transform2Rect();

        Bitmap faceCrop = MyUtil.crop(input, rect);
        faceImage.setImageBitmap(faceCrop);
    }

    private Bitmap getBitmap(Drawable drawable) {
        Bitmap bitmap = Bitmap.createBitmap(
                drawable.getIntrinsicWidth(),
                drawable.getIntrinsicHeight(),
                drawable.getOpacity() != PixelFormat.OPAQUE ? Bitmap.Config.ARGB_8888
                        : Bitmap.Config.RGB_565);
        Canvas canvas = new Canvas(bitmap);
        //canvas.setBitmap(bitmap);
        drawable.setBounds(0, 0, drawable.getIntrinsicWidth(), drawable.getIntrinsicHeight());
        drawable.draw(canvas);
        return bitmap;
    }

}