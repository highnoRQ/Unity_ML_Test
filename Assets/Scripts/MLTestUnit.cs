using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Rendering;

public class MLTestUnit : MonoBehaviour
{
    [SerializeField] private string model_name;

    private List<string[]> classficatation_data;

    private delegate int run_unit(Texture2D input);
    private run_unit RunMLUnit;
        
    private OnnxRuntimeUnit onnx_unit;
    private OpenCVSharpUnit opencv_unit;
    private TensorFlowSharpUnit tensor_unit;

    [SerializeField] private RenderTexture cam_tex;
    private Texture2D tex2d;

    [SerializeField] private Text Result;

    private void Start()
    {
        InitializeMLUnit();
        InitializeClassficator();

        tex2d = new Texture2D(cam_tex.width, cam_tex.height, TextureFormat.RGB24, false);
    }

    private void InitializeMLUnit()
    {   
        onnx_unit = new OnnxRuntimeUnit();
        opencv_unit = new OpenCVSharpUnit();
        tensor_unit = new TensorFlowSharpUnit();
    }
    //ボタンを押した時、ロード済みでなければモデルをロードする
    public void LoadOnnxRuntimeUnit()
    {
        float before = Time.realtimeSinceStartup;
        bool load = onnx_unit.GetLoaded();

        if (!load) onnx_unit.LoadModel(model_name);
        RunMLUnit = onnx_unit.Inference;

        ShowLoadTime(load, Time.realtimeSinceStartup - before);
    }
    public void LoaOpenCVUnit()
    {
        float before = Time.realtimeSinceStartup;
        bool load = opencv_unit.GetLoaded();

        if (!load) opencv_unit.LoadModel(model_name);
        RunMLUnit = opencv_unit.Inference;

        ShowLoadTime(load, Time.realtimeSinceStartup - before);
    }
    public void LoadTensorFlowSharpUnit()
    {
        float before = Time.realtimeSinceStartup;
        bool load = tensor_unit.GetLoaded();

        if (!load) tensor_unit.LoadModel(model_name);
        RunMLUnit = tensor_unit.Inference;

        ShowLoadTime(load, Time.realtimeSinceStartup - before);
    }

    // csv形式で保存したImageNetの分類項目をロードする。
    private void InitializeClassficator()
    {
        classficatation_data = new List<string[]>();
        TextAsset csv_file = Resources.Load("classfication") as TextAsset;
        StringReader reader = new StringReader(csv_file.text);

        while (reader.Peek() > -1)
        {
            string line = reader.ReadLine();
            classficatation_data.Add(line.Split(','));
        }
    }

    public void Inference()
    {
        UpdateTex2D();

        //推論を実行し分類番号を取得する。
        if (RunMLUnit != null)
        {
            float before = Time.realtimeSinceStartup;
            int key = RunMLUnit(tex2d);
            ShowInferTime(Time.realtimeSinceStartup - before);

            Result.text = classficatation_data[key][0];
        }
    }
    // RenderTextureに写っている画像をTexture2Dに取り込む。
    private void UpdateTex2D()
    {
        // ReadPixelsはactiveになっているRenderTextureの画像をTexture2D形式で読み込む。
        RenderTexture.active = cam_tex;
        tex2d.ReadPixels(new Rect(0, 0, tex2d.width, tex2d.height), 0, 0);
        tex2d.Apply();
    }

    [SerializeField] private SpriteRenderer target_sprite;
    public void ChangeSample()
    {
        Sprite[] all = Resources.LoadAll<Sprite>("Sample");
        int s = Random.Range(0, all.Length);
        target_sprite.sprite = all[s];

        Result.text = "Result";
    }
    [SerializeField] private Text time_text;
    private void ShowLoadTime(bool loaded,float time)
    {
        if (!loaded)
        {
            time_text.text = string.Concat("success!!\n", time, "[sec]");
        } else
        {
            time_text.text = "loaded\nSet inference unit";
        }
    }
    private void ShowInferTime(float time)
    {
        time_text.text = string.Concat("Inference\n", time, "[sec]");
    }


}
