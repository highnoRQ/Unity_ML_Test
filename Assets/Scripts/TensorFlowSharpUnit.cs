using UnityEngine;
using System;
using System.Collections;
using System.IO;
using System.Linq;

using TensorFlow;

public class TensorFlowSharpUnit
{
    private TFGraph model;
    public void LoadModel(string model_name)
    {
        string model_path = Application.dataPath + @"/MLModel/" + model_name + ".pb";

        var model_input = File.ReadAllBytes(model_path);
        model = new TFGraph();
        model.Import(model_input);
        
        loaded = true;
    }
    private bool loaded = false;
    public bool GetLoaded() { return loaded; }

    public int Inference(Texture2D input)
    {
        var session = new TFSession(model);

        // Texture2Dをモデルの入力に合った形に整形、正規化する。        
        var float_values = GetFloatFromTex2DWithFlip(input);

        // TensorFlowSharpでの入力形式であるTFTensorを作成する
        var shape = new TFShape(1, input.width, input.height, 3);
        var input_tensor = TFTensor.FromBuffer(shape, float_values, 0, float_values.Length);

        //  データの入力・推論
        //  input_2:0およびoutput_node0:0はpbファイル作成時につけた入力ノードと出力ノードの名前。
        var runner = session.GetRunner();
        runner.AddInput(model["input"][0], input_tensor);
        runner.Fetch(model["content_vgg/prob"][0]);

        // 推論を実行
        var output = runner.Run();
        var scores = ((float[][])output[0].GetValue(true))[0];

        return GetBestScorePos(scores);
    }

    private float[] GetFloatFromTex2DWithFlip(Texture2D input)
    {
        Color32[] pix = input.GetPixels32();
        float[] output = new float[pix.Length * 3];

        float[] mean = { 0.485f, 0.456f, 0.406f };
        float[] std = { 0.229f, 0.224f, 0.225f };

        // keras.ApplicationのVGG19をpb形式に変換して利用したため、
        // 画像はBGR形式で、-127~127に正規化した。
        // 面倒だったので軸が適当になってます、結果が違ったらそのせい。
        for (int y = 0; y < input.height; y++)
        {
            for (int x = 0; x < input.width; x++)
            {
                int c; float set;

                c = 0;
                set = pix[(input.width-x-1) + y * input.width].r / 255f;
                output[y * 3 + x * input.width * 3 + c] = (set - mean[c]) / std[c] * 127f;

                c = 1;
                set = pix[(input.width - x - 1) + y * input.width].g / 255f;
                output[y * 3 + x * input.width * 3 + c] = (set - mean[c]) / std[c] * 127f;

                c = 2;
                set = pix[(input.width - x - 1) + y * input.width].b / 255f;
                output[y * 3 + x * input.width * 3 + c] = (set - mean[c]) / std[c] * 127f;

            }
        }
        return output;
    }
    private int GetBestScorePos(float[] score)
    {
        int key = 0;
        float max = 0.0f;
        for (int i = 0; i < score.Length; i++)
        {
            if (score[i] > max)
            {
                max = score[i];
                key = i;
            }
        }
        return key;
    }
}
