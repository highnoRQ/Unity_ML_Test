using UnityEngine;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

using System.Numerics.Tensors;
using Microsoft.ML.OnnxRuntime;

class OnnxRuntimeUnit {
    private InferenceSession session;
    public void LoadModel (string model_name) {
        string model_path = Application.dataPath + @"/MLModel/" + model_name + ".onnx";

        var opitions = new SessionOptions ();
        session = new InferenceSession (model_path);
        loaded = true;
    }
    private bool loaded = false;
    public bool GetLoaded() { return loaded; }

    public int Inference (Texture2D input) {
        var input_nodes_name = session.InputMetadata.First ().Key;
        var input_nodes_dim = session.InputMetadata.First ().Value.Dimensions;

        // Texture2Dをモデルの入力に合った形に整形、正規化する。
        var input_floats = GetFloatFromTex2DWithFlip (input);
        var input_tensor = new DenseTensor<float> (input_floats, input_nodes_dim);

        // OnnxRuntimeでの入力形式であるNamedOnnxValueを作成する
        var input_onnx_values = new List<NamedOnnxValue> {
            NamedOnnxValue.CreateFromTensor (input_nodes_name, input_tensor)
        };

        // 推論を実行
        var results = session.Run (input_onnx_values);
        var scores = results.First ().AsTensor<float> ().ToArray ();

        return GetBestScorePos (scores);
    }

    private float[] GetFloatFromTex2DWithFlip (Texture2D input) {
        Color32[] pix = input.GetPixels32 ();
        float[] output = new float[pix.Length * 3];

        float[] mean = { 0.485f, 0.456f, 0.406f };
        float[] std = { 0.229f, 0.224f, 0.225f };

        // 上述の平均値および標準偏差で正規化した。
        // また、Unityでのy軸とtensorflow入力のy軸が逆向きのため反転を同時に行なった。
        for (int y = 0; y < input.height; y++) {
            for (int x = 0; x < input.width; x++) {
                int c;             
                float set;

                c = 0;
                set = pix[x + (input.height - y - 1) * input.width].r / 255f;
                output[x + y * input.width + c * pix.Length] = (set - mean[c]) / std[c];

                c = 1;
                set = pix[x + (input.height - y - 1) * input.width].g / 255f;
                output[x + y * input.width + c * pix.Length] = (set - mean[c]) / std[c];

                c = 2;
                set = pix[x + (input.height - y - 1) * input.width].b / 255f;
                output[x + y * input.width + c * pix.Length] = (set - mean[c]) / std[c];
            }
        }

        return output;
    }
    private int GetBestScorePos (float[] score) {
        int key = 0;
        float max = 0.0f;
        for (int i = 0; i < score.Length; i++) {
            if (score[i] > max) {
                max = score[i];
                key = i;
            }
        }
        return key;
    }
}