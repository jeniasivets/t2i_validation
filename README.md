## Image Text Evaluation

### Overview

Text-image alignment metrics measure the degree to which a textual prompt is aligned with a generated image. There is a range of text-image metrics that evaluate alignment by comparing text-image embeddings, along with metrics that assess the content of both the image and the corresponding prompt. [4]

To estimate the similarity between images and text, I will use three baseline metrics:

1. **CLIP Score**: A widely used metric that measures the alignment between an image and a text prompt. However, it fails to produce reliable scores for complex prompts involving compositions of objects, attributes, and relations. CLIP model jointly trains an image encoder and a text encoder to maximize the cosine similarity between the correct image and text embedding pairs. CLIP Score measures the cosine similarity of the embedded image and text prompt (paper [1]).

2. **BLIP Score**: BLIP/BLIPv2 is a framework that enables a wider range of downstream tasks, including image captioning, image-text retrieval, and visual question answering (VQA). The framework jointly optimizes three training objectives: part of this framework is trained in the same manner as CLIP, where text and image embedding vectors are learned in an image-text contrastive manner. Another part allows image-grounded text generation. However, for the BLIP Score, we utilize a part of the framework that is trained to learn a binary classification task, where the model predicts whether an image-text pair is positive (matched) or negative (unmatched). This part produces the image-text matching (ITM) embeddings that are utilized to compute the alignment score between text and image (paper [2]).

3. **VQA Score**: One of the recent papers on the VQA approach, shows promising results in evaluating image-text data, demonstrating significantly stronger agreement with human judgments. To train the model, authors use the input to the model as an image \(I\) and a question \(Q\) in the format: "Does this figure show {text}? Please answer yes or no," where {text} is the prompt used to generate image \(I\). They fine-tune a VQA model (encoder-decoder language model combined with pre-trained CLIP vision encoder) to predict answer likelihoods (paper [3]).


While the BLIP and VQA Scores seem to provide more comprehensive similarity signals for image-text pairs, they have limitations in terms of computational expenses and interpretability, as they output probabilities without reasoning accompanying the predicted scores. For example, evaluation results show that BLIP overconfidently predicts that an image and text are matching.

To compare the metrics with each other, I measure the Spearman rank correlation between the metric values. The correlation between these metrics is not very strong, however, CLIP and BLIP correlate more strongly than with VQA.

### Installation Packages

```bash
pip install torch torchvision torchaudio
pip install git+https://github.com/openai/CLIP.git
pip install t2v-metrics
pip install git+https://github.com/huggingface/transformers
```

### Repository Structure

```
repo/
│
├── results/                # contains csv files with computed scores / performance metrics
├── data/                   # contains original images and prompts
├── tests/                  # validation tests
├── validation/             #
├── notebooks/              # metrics analysis & visualization
│
├── README.md               
```

### References

[1] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, Ilya Sutskever:
Learning Transferable Visual Models From Natural Language Supervision. ICML 2021: 8748-8763

[2] Junnan Li, Dongxu Li, Silvio Savarese, Steven C. H. Hoi:
BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models. ICML 2023: 19730-19742

[3] Zhiqiu Lin, Deepak Pathak, Baiqi Li, Jiayao Li, Xide Xia, Graham Neubig, Pengchuan Zhang, Deva Ramanan:
Evaluating Text-to-Visual Generation with Image-to-Text Generation. ECCV (9) 2024: 366-384

[4] Hartwig, Sebastian, et al. "Evaluating Text to Image Synthesis: Survey and Taxonomy of Image Quality Metrics." arXiv preprint arXiv:2403.11821 (2024).
