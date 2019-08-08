using UnityEngine;
using System;
using System.Collections;
using System.IO;
using System.Linq;

using OpenCvSharp;
using OpenCvSharp.Dnn;

public class OpenCVSharpUnit
{
    private Net model;
    public void LoadModel(string model_name)
    {
        string model_path = Application.dataPath + @"/MLModel/" + model_name + ".onnx";
        model = Net.ReadNetFromONNX(model_path);
        loaded = true;
    }
    private bool loaded = false;
    public bool GetLoaded() { return loaded; }

    public int Inference(Texture2D input)
    {
        // Texture2Dをモデルの入力に合った形に整形する
        Mat img = FixTexture2Input(input);

        // OpenCVSharpでの入力形式であるblobを作成、同時に正規化を行う。
        Scalar mean = new Scalar(0.485f, 0.456f, 0.406f);
        var blob = CvDnn.BlobFromImage(img, 1.0 / 255.0 /0.225f, new Size(input.width, input.height), mean, swapRB: true, crop: false);
        model.SetInput(blob);

        // 推論を実行
        Mat scores = model.Forward();

        return GetBestScorePos(scores);
    }

    private Mat FixTexture2Input(Texture2D input)
    {
        Color32[] c = input.GetPixels32();
        Mat mat = new Mat(input.width, input.height, MatType.CV_8UC3);
        var SourceImageData = new Vec3b[input.width * input.height];
        for (int i = 0; i < input.width; i++)
        {
            for (int j = 0; j < input.height; j++)
            {
                var col = c[i + j * 224];
                var vec3 = new Vec3b
                {
                    Item0 = col.b,
                    Item1 = col.g,
                    Item2 = col.r
                };
                SourceImageData[i + j * input.height] = vec3;
            }
        }
        mat.SetArray(0, 0, SourceImageData);

        // Unityでのy軸とtensorflow入力のy軸が逆向きのため反転して出力した。
        return mat.Flip(FlipMode.X);
    }
    private int GetBestScorePos(Mat score)
    {
        int key = 0;
        float max = 0.0f;
        for (int i = 0; i < score.Width; i++)
        {
            float value = score.Get<float>(0, i);
            if (value > max)
            {
                key = i;
                max = value;
            }
        }
        return key;
    }
}