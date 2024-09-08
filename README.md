# Enhancing Trust in Clinically Significant Prostate Cancer Prediction with Multiple Magnetic Resonance Imaging Modalities

In the United States, prostate cancer is the second leading cause of deaths in males with a predicted 35,250 deaths in 2024. However, most diagnosis of prostate cancer are not lethal and are deemed clinically insignificant which means that the patient will likely not be impacted by the cancer over their lifetime. Subsequently, numerous research studies have explored the accuracy of predicting clinical significance of prostate cancer based on magnetic resonance imaging (MRI) modalities and deep neural networks. Despite their high performance, these models are not trusted by most clinical scientists as most models are trained solely on a single modality but clinical scientists often use multiple magnetic resonance imaging modalities during their diagnosis. In this paper, we investigate combining multiple MRI modalities to train a deep learning model to enhance trust in the models for clinically significant prostate cancer prediction. The promising performance and proposed training pipeline showcase the benefits of incorporating multiple MRI modalities for enhanced trust and accuracy.  

This repository contains the code and notebooks to replicate our findings:


## Preprocess Data into 3 Modalities

```bash
python src/preprocess_data_3_modalities.py --raw_data_loc "data/pca_processed_data" --modality1 'DWIb3' --modality2 'T2w' --modality3 'ADC' --output_loc 'data/pca_processed_data/3_modalities_combined'
```

## Train Model

```bash
python src/run_model_training.py --model-config 'config/monai_resnet_34_23_clinic_sig.yml' --modality '3_modalities_combined' --gpu-id 0
```

## Explainable AI

The notebook located at `src/xai/visualize_xai.ipynb` generates the attention maps for a given model. 

The scripts and notebooks located in `src/visualizing` can then be used to visualize the generated attention maps. 

