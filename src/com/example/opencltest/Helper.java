package com.example.opencltest;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Arrays;
import java.util.Random;

/**
 * Created by zhengzej on 12/9/2015.
 */
public class Helper {
    private static float t1 = 20;
    private static float t2 = 200;
    private static float c = 2;
    private static float gamma = 10000;

    public static final float firing_threshold = (float) 0.5;
    public static final float epsilon = (float) 0.0011;
    public static final float learning_threhold_bottom_up = (float) 0.8;
    public static float learning_threhold_top_down = (float) 0.8; // 1.2 with attention excitation

    public void setT1(float t1) {
        Helper.t1 = t1;
    }

    public void setT2(float t2) {
        Helper.t2 = t2;
    }

    public void setC(float c) {
        Helper.c = c;
    }

    public void setGamma(float gamma) {
        Helper.gamma = gamma;
    }

    public static void initialize_weight(float[] weight, int size, boolean randomFlag) {
        // if randomFlag == true, generate weight randomly
        // otherwise, initialize all weight to 0.0
        Random randomGenerator = new Random();
        for (int i = 0; i < size; i++){
            if (randomFlag == true){
                weight[i] = randomGenerator.nextFloat();
            }
            else {
                weight[i]=(float) 0.0;
            }
        }
    }

    public static float[] normalize(float[] weight, int size, int flag) {
        if (flag ==1){
            float min = weight[0];
            float max = weight[0];
            for (int i = 0; i < size; i++){
                if(weight[i] < min){min = weight[i];}
                if(weight[i] > max){max = weight[i];}
            }
            float diff = max-min + Helper.epsilon;
            for(int i = 0; i < size; i++){
                weight[i] = (weight[i]-min)/diff;
            }
            float mean = 0;
            for (int i = 0; i < size; i++){
                mean += weight[i];
            }
            mean = mean/size;
            for (int i = 0; i < size; i++){
                weight[i] = weight[i]-mean + Helper.epsilon;
            }
            float norm = 0;
            for (int i = 0; i < size; i++){
                norm += weight[i]*weight[i];
            }
            norm = (float) Math.sqrt(norm);
            if (norm > 0){
                for (int i = 0; i < size; i++){
                    weight[i] = weight[i]/norm;
                }
            }
        }

        if(flag==2){
            float norm = 0;
            for (int i = 0; i < size; i++){
                norm += weight[i]* weight[i];
            }
            norm = (float) Math.sqrt(norm);
            if (norm > 0){
                for (int i = 0; i < size; i++){
                    weight[i] = weight[i]/norm;
                }
            }
        }

        if (flag == 3){
            float norm = 0;
            for (int i = 0; i < size; i++){
                norm += weight[i];
            }
            norm = norm+Helper.epsilon;
            if (norm > 0){
                for (int i = 0; i < size; i++){
                    weight[i] = weight[i]/norm;
                }
            }
        }
        return null;
    }

    public static void setLearning_threhold_top_down(float threhold_top_down){
        learning_threhold_top_down = threhold_top_down;
    }

    public static float compute_response(float[] weight, float[] input, int size) {
        float response = 0;
        for (int i = 0; i < size; i++){
            response += weight[i]*input[i];
        }
        return response;
    }

    public static void top_k_competition(float[] response_array, int size,
                                         int k) {
        float[] copy_of_array = new float[size];
        System.arraycopy( response_array, 0, copy_of_array, 0, size);
        Arrays.sort(copy_of_array);

        for (int i = 0; i < size; i++){
            if(response_array[i]<copy_of_array[size-k]){
                response_array[i]=0;
            }
            else{
                //response_array[i]= (response_array[i]- copy_of_array[size-k-1])/
                //		(copy_of_array[size-1]-copy_of_array[size-k-1]);
                response_array[i] = 1;
            }
        }
    }

    public static void top_k_competition(float[][][] response_array, int k) {
        int depth = response_array.length;
        int height = response_array[0].length;
        int width = response_array[0][0].length;
        int find_min_flag = 0;

        float[] copy_of_array = new float[depth* height* width];
        for (int depth_index = 0; depth_index < depth; depth_index++){
            for (int height_index = 0; height_index < height; height_index++){
                System.arraycopy(response_array[depth_index][height_index], 0,
                        copy_of_array, (depth_index * height + height_index) * width, width);
            }
        }

        Arrays.sort(copy_of_array);

        for (int depth_index = 0; depth_index < depth; depth_index++){
            for (int height_index = 0; height_index < height; height_index++){
                for (int width_index = 0; width_index < width; width_index++){
                    if(response_array[depth_index][height_index][width_index] <
                            copy_of_array[depth * height * width - k]){
                        response_array[depth_index][height_index][width_index] = 0;
                    } else if((response_array[depth_index][height_index][width_index] ==
                            copy_of_array[depth * height * width - k])&& find_min_flag == 1){
                        response_array[depth_index][height_index][width_index] = 0;
                    } else if((response_array[depth_index][height_index][width_index] ==
                            copy_of_array[depth * height * width - k])&& find_min_flag == 0){
                        response_array[depth_index][height_index][width_index] = 1;
                        find_min_flag = 1;
                    } else{
                        response_array[depth_index][height_index][width_index] = 1;
                    }
                }
            }
        }
    }

    public static float get_learning_rate(int age) {
        float mu;
        if(age < t1){
            mu = 0;
        }
        else if((age < t2) & (age >= t1)){
            mu = c * ((float)age-t1)/(t2-t1);
        }
        else{
            mu = c + ((float)age-t2)/gamma;
        }
        float learning_rate = (float) ((1+ mu)/((float)age + 1.0));
        return learning_rate;
    }

    public static void flatten_2D_vector(float[][] src_vec, float[] target_vec){
        int height = src_vec.length;
        int width = src_vec[0].length;
        for (int i = 0; i < height; i++){
            System.arraycopy(src_vec[i], 0, target_vec, i*width, width);
        }
    }

    public static void flatten_3D_vector(float[][][] src_vec, float[] target_vec) {
        int depth = src_vec.length;
        int height = src_vec[0].length;
        int width = src_vec[0][0].length;
        for (int i = 0; i < depth; i++){
            for (int j = 0; j < height; j++){
                System.arraycopy(src_vec[i][j], 0, target_vec, (i* height + j)*width, width);
            }
        }
    }

    public static void update_weight(float[] weight, float[] input,
                                     float learning_rate) {
        assert weight.length == input.length;
        for(int i = 0; i < input.length; i++){
            weight[i] = (float) ((1.0 - learning_rate) * weight[i] + learning_rate * input[i]);
        }
    }

    public static void initialize_4D_weight(float[][][][] weight, int depth, int height,
                                            int width, int length, boolean randomFlag) {
        for(int loop1 = 0; loop1 < depth; loop1++){
            for(int loop2 = 0; loop2 < height; loop2++){
                for(int loop3 = 0; loop3 < width; loop3++){
                    Helper.initialize_weight(weight[loop1][loop2][loop3],
                            length,randomFlag);
                }
            }
        }
    }

    public static void initialize_4D_factor(float[][][][] weight, int depth, int height,
                                            int width, int input,int rfsize_rfsize, boolean synapse_factor_flag) {
        // if synapse_factor_flag == true initialize to ones, else initialize to 1/rfsize_rfsize;
        int length = input*rfsize_rfsize;
        for(int loop1 = 0; loop1 < depth; loop1++){
            for(int loop2 = 0; loop2 < height; loop2++){
                for(int loop3 = 0; loop3 < width; loop3++){
                    for(int loop4 = 0; loop4 < length; loop4++){
                        if (synapse_factor_flag)
                            weight[loop1][loop2][loop3][loop4]=(float) (1.0/3.4641);
                        else
                            weight[loop1][loop2][loop3][loop4]=(float) (1.0/(float)rfsize_rfsize);
                    }
                }
            }
        }
    }

    public static void set_zero(float[] input, int begin_index, int end_index){
        for (int i = begin_index; i <= end_index; i++){
            input[i] = 0;
        }
    }

    public static void shuffle_array(int[] array)
    {
        int index, temp;
        Random random = new Random();
        for (int i = array.length - 1; i > 0; i--)
        {
            index = random.nextInt(i + 1);
            temp = array[index];
            array[index] = array[i];
            array[i] = temp;
        }
    }

    public static int[] fromIntString(String string) {
        String[] strings = string.replace("[", "").replace("]", "").split(", ");
        int result[] = new int[strings.length];
        for (int i = 0; i < result.length; i++) {
            result[i] = Integer.parseInt(strings[i]);
        }
        return result;
    }

    public static int[] fromSpacedIntString(String string) {
        String[] strings = string.split("\\t");
        int result[] = new int[strings.length];
        for (int i = 0; i < result.length; i++) {
            result[i] = Integer.parseInt(strings[i]);
        }
        return result;
    }


    public static float[] fromFloatString(String string){
        String[] strings = string.replace("[", "").replace("]", "").split(", ");
        float result[] = new float[strings.length];
        for (int i = 0; i < result.length; i++) {
            result[i] = Float.parseFloat(strings[i]);
        }
        return result;
    }

    public static boolean[] fromBoolString(String string){
        String[] strings = string.replace("[", "").replace("]", "").split(", ");
        boolean result[] = new boolean[strings.length];
        for (int i = 0; i < result.length; i++) {
            result[i] = Boolean.parseBoolean(strings[i]);
        }
        return result;
    }

    public static int[] get_debug_training_order(String path) throws IOException {
        File fin = new File(path);
        BufferedReader br;
        br = new BufferedReader(new FileReader(fin));
        int[] order = fromSpacedIntString(br.readLine());
        br.close();
        return order;
    }
}