package com.example.opencltest;

import android.os.Bundle;
import android.util.Log;
import android.app.Activity;
import android.view.View;
import android.widget.TextView;
import android.widget.Button;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;

public class MainActivity extends Activity {

    private int mNumOfBottomUpInput = 1;
    private int mDepth = 1; // 30
    private int mHeight = 1; // 100
    private int mWidth = 100; // 100
    private int mRFSize = 45; //45
    private long runtimeCPU = 0;
    private long runtimeGPU = 0;
    private int mFourthDimension = mNumOfBottomUpInput * mRFSize * mRFSize;
    private float[][][][]  mCPUNormWeight;
    private float[][][][]  mGPUNormWeight;
	
    static boolean sfoundLibrary = true;  
    static {
    	android.os.Debug.waitForDebugger();
        try { 
        	System.load("/system/vendor/lib/libOpenCL.so");
        	 Log.i("Debug", "OpenCL lib Loaded");
        	 System.loadLibrary("OpenCLTest"); 
        	 Log.i("Debug","My Lib Loaded!");
        }
        catch (UnsatisfiedLinkError e) {
          sfoundLibrary = false;
        }
      }
    
	/*
	 * loads the kernel into the app_execdir 
	 */
	private void copyFile(final String f) {
		InputStream in;
		try {
			in = getAssets().open(f);
			final File of = new File(getDir("execdir",MODE_PRIVATE), f);
			
			final OutputStream out = new FileOutputStream(of);

			final byte b[] = new byte[65535];
			int sz = 0;
			while ((sz = in.read(b)) > 0) {
				out.write(b, 0, sz);
			}
			in.close();
			out.close();
		} catch (IOException e) {       
			e.printStackTrace();
		}
	}
	@Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        final Button button_initialize = (Button)findViewById(R.id.btn_initialize);
        button_initialize.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                mCPUNormWeight = new float[mDepth][mHeight][mWidth][mFourthDimension];
                Helper.initialize_4D_weight(mCPUNormWeight, mDepth, mHeight, mWidth, mFourthDimension, true);  // set flag = false to initialize weight to zeros
                
                mGPUNormWeight = new float[mDepth][mHeight][mWidth][mFourthDimension];
                for(int i = 0; i < mDepth; i++){
                	for(int j = 0; j < mHeight; j++){
                		for(int k = 0; k < mWidth; k++){
                            mGPUNormWeight[i][j][k] = mCPUNormWeight[i][j][k].clone();
               			}
               		}
                }
                String output = "Initialization done";
                ((TextView)findViewById(R.id.result)).setText(output);
            }
        });

        final Button button_cpu = (Button)findViewById(R.id.btn_run_cpu);
        button_cpu.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
            	long startTime = System.currentTimeMillis();
                int begin_index = 0;
                for (int depth_index = 0; depth_index < mDepth; depth_index++) {
                    for (int height_index = 0; height_index < mHeight; height_index++) {
                        for (int width_index = 0; width_index < mWidth; width_index++) {
                            for (int input_index = 0; input_index < mNumOfBottomUpInput; input_index++) {
                                begin_index = input_index * mRFSize * mRFSize;
                                float[] current_weight = new float[mRFSize * mRFSize];
                                System.arraycopy(mCPUNormWeight[depth_index][height_index][width_index], begin_index, current_weight, 0, mRFSize * mRFSize);
                                Helper.normalize(current_weight, mRFSize * mRFSize, 1);
                                System.arraycopy(current_weight, 0, mCPUNormWeight[depth_index][height_index][width_index], begin_index, mRFSize * mRFSize);
                            }
                        }
                    }
                }
                long endTime = System.currentTimeMillis();
                runtimeCPU = (endTime - startTime);
                String output = "CPU Mode done";
                ((TextView)findViewById(R.id.result)).setText(output);
            }
        });

        final Button button_gpu = (Button)findViewById(R.id.btn_run_gpu);
        button_gpu.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
            	long startTime = System.currentTimeMillis();
            	copyFile("normalize.cl");
                initOpenCL ("normalize");
                for(int depth_index = 0; depth_index < mDepth; depth_index++){
                    for(int height_index = 0; height_index < mHeight; height_index++){
                            NormalizeGPU(mGPUNormWeight[depth_index][height_index], mWidth, mFourthDimension, mRFSize, 1);
                    }
                }
                shutdownOpenCL();
                long endTime = System.currentTimeMillis();
                runtimeGPU = (endTime - startTime);
                String output = "GPU Mode done";
                ((TextView)findViewById(R.id.result)).setText(output);
            }
        });

        final Button button_check = (Button)findViewById(R.id.btn_check);
        button_check.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                boolean result = true;
                Log.d("CPU First element: ", Float.toString(mCPUNormWeight[0][0][0][0]));
                Log.d("GPU First element: ", Float.toString(mGPUNormWeight[0][0][0][0]));
                for(int depth_index = 0; depth_index < mDepth; depth_index++){
                    for(int height_index = 0; height_index < mHeight; height_index++){
                        for(int width_index = 0; width_index < mWidth; width_index++){
                        	for (int fourth_dimension = 0; fourth_dimension < (mFourthDimension); fourth_dimension++)
                            if(Math.abs(mCPUNormWeight[depth_index][height_index][width_index][fourth_dimension] - 
                            		mGPUNormWeight[depth_index][height_index][width_index][fourth_dimension]) > 0.001){
                            	Log.d("Depth: ", Integer.toString(depth_index));
                            	Log.d("Height: ", Integer.toString(height_index));
                    			Log.d("Width: ", Integer.toString(width_index));
                    			Log.d("Fourth Dimension: ", Integer.toString(fourth_dimension));
                    			Log.d("CPU: ", Float.toString(mCPUNormWeight[depth_index][height_index][width_index][fourth_dimension]));
                                Log.d("GPU: ", Float.toString(mGPUNormWeight[depth_index][height_index][width_index][fourth_dimension]));
                                result = false;
                                break;
                            }
                        	if (result==false) break;
                        }
                        if (result==false) break;
                    }
                    if (result==false) break;
                }

                String output = "Checked result: " + Boolean.toString(result);
                ((TextView)findViewById(R.id.result)).setText(output);
                ((TextView)findViewById(R.id.result)).append("\nCPU Runtime: " + 
                Long.toString(runtimeCPU) + "ms");
                ((TextView)findViewById(R.id.result)).append("\nGPU Runtime: " + 
                        Long.toString(runtimeGPU) + "ms");
            }
        });
    }

    private native void initOpenCL (String kernelName);
    private native void shutdownOpenCL ();
    public native void nativemultiply(float[][] first, float[][] second, float[][] result, int NumRowsA, int NumColsARowsB, int NumColsB);
    public native void NormalizeGPU(float[][] weight, int num_rows, int num_cols, int rf_size, int flag);


}