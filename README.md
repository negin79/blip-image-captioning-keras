# Image Captioning and Summarization Using BLIP

### Computer Vision | Deep Learning | Vision-Language Models

## Project Overview

This project explores **image captioning and text generation using a pretrained Vision-Language Model (VLM)**. It demonstrates how the BLIP (Bootstrapping Language-Image Pre-training) model can be integrated into a TensorFlow/Keras workflow to automatically generate natural-language descriptions of images.

The implementation uses a pretrained BLIP model from Hugging Face and wraps it inside a custom Keras layer, allowing images to be processed using different text-generation prompts.

The project was completed as a hands-on learning exercise associated with the [Introduction to Deep Learning with Keras](https://www.coursera.org/learn/introduction-to-deep-learning-with-keras) course on Coursera.

## Objectives

- Understand the fundamentals of Vision-Language Models.
- Explore pretrained deep learning models for image-to-text generation.
- Implement a custom TensorFlow/Keras layer for image captioning.
- Integrate Hugging Face Transformers with TensorFlow.
- Compare generated image descriptions using different prompts.
- Gain practical experience with computer vision inference pipelines.

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| TensorFlow / Keras | Custom layer implementation and integration |
| PyTorch | Backend for BLIP model inference |
| Hugging Face Transformers | Loading the pretrained BLIP model and processor |
| Pillow (PIL) | Image loading and preprocessing |
| Matplotlib | Image visualization |
| NumPy | Numerical operations |
| Jupyter Notebook | Development and experimentation |

## Model Architecture

The project uses the pretrained model:

**`Salesforce/blip-image-captioning-base`**

BLIP combines visual understanding and natural-language generation to produce textual descriptions of images.

Two main components are used:

**1. BlipProcessor**

Prepares images and text prompts for the model, converting them into the required input format.

**2. BlipForConditionalGeneration**

Generates textual descriptions based on the processed image and the supplied prompt.

### Processing Pipeline

```text
Input Image
    |
    v
Image Loading & RGB Conversion
    |
    v
Task Selection (Caption / Summary)
    |
    v
Prompt Construction
    |
    v
BLIP Processor
    |
    v
Pretrained BLIP Model
    |
    v
Text Generation
    |
    v
Generated Image Description
```

## Implementation

### 1. Loading the Pretrained Model

The BLIP model and processor are loaded from Hugging Face.

```python
from transformers import BlipProcessor, BlipForConditionalGeneration

processor = BlipProcessor.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)

model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
)
```

### 2. Custom Keras Layer

A custom `BlipCaptionSummaryLayer` is implemented by subclassing `tf.keras.layers.Layer`.

The layer performs the following operations:

- Loads an image from the specified file path.
- Converts the image to RGB format.
- Selects a text prompt based on the requested task.
- Preprocesses the image and prompt with `BlipProcessor`.
- Generates text using the pretrained BLIP model.
- Decodes and returns the generated description.

The implementation uses `tf.py_function` to connect the Python-based BLIP inference process with the TensorFlow/Keras layer interface.

### 3. Task-Based Text Generation

Two generation modes are implemented:

**Caption mode**

```python
prompt = "This is a picture of"
```

Generates a description of the image.

**Summary mode**

```python
prompt = "This is a detailed photo showing"
```

Uses a different prompt to encourage a more descriptive output.

Both modes use the same pretrained captioning model. The summary mode is therefore a prompt-based variation, rather than a separately trained image summarization system.

## Example Results

The notebook demonstrates the model using images from an aircraft damage dataset.

For one sample image, the generated outputs were:

**Caption:**
> this is a picture of a plane

**Summary:**
> this is a detailed photo showing the engine of a boeing 747

These examples illustrate how changing the input prompt can influence the generated description.

The outputs are model predictions and have not been independently verified against ground-truth image annotations.

## How to Run

**1. Clone the repository**

```bash
git clone <repository-url>
cd <repository-name>
```

**2. Install the required dependencies**

The notebook uses the following dependencies:

```text
pandas==2.2.3
tensorflow-cpu==2.17.1
pillow==11.1.0
matplotlib==3.9.2
transformers==4.38.2
torch==2.2.0
torchvision==0.17.0
torchaudio==2.2.0
numpy<2
```

The notebook installs CPU-only PyTorch packages from the official PyTorch wheel index.

**3. Open the Jupyter Notebook**

```bash
jupyter notebook
```

Open:

`Image Captioning and Summarization using BLIP Pretrained Model.ipynb`

**4. Set an image path**

Update the example image path to point to an existing local image.

```python
image_path = tf.constant("path/to/your/image.jpg")
```

**5. Generate a description**

```python
caption = generate_text(
    image_path,
    tf.constant("caption")
)

print(caption.numpy().decode("utf-8"))
```

To use the alternative prompt, change `"caption"` to `"summary"`.

**Note:** The pretrained model is downloaded from Hugging Face when first loaded. The aircraft damage dataset referenced in the notebook must also be available locally, or the image paths must be replaced.

## Limitations

- The BLIP model is used for inference only; no additional training or fine-tuning is performed.
- Generated descriptions may contain incorrect or unsupported details.
- The summary mode is based on prompt modification rather than a specialized summarization architecture.
- No quantitative evaluation metrics are included in the notebook.
- The custom layer uses `tf.py_function` and PyTorch inference, making it primarily suitable for experimentation rather than direct production deployment.

## Key Learning Outcomes

Through this project, I gained practical experience with:

- Applying pretrained Vision-Language Models to computer vision tasks.
- Working with Hugging Face Transformers.
- Connecting PyTorch-based models to TensorFlow/Keras workflows.
- Building custom Keras layers.
- Implementing image preprocessing and text-generation pipelines.
- Exploring how different prompts affect generated image descriptions.

## Future Improvements

Potential extensions include:

- Evaluating captions against reference descriptions using BLEU, METEOR, or CIDEr.
- Comparing different pretrained image-captioning models.
- Implementing batch inference for larger image datasets.
- Fine-tuning a model on a domain-specific dataset.
- Developing an interactive interface for image uploads and caption generation.
- Exploring image captioning for medical imaging applications.

## Acknowledgments

This is an educational practice project based on the concepts and exercises from the Coursera course:

[Introduction to Deep Learning with Keras](https://www.coursera.org/learn/introduction-to-deep-learning-with-keras)

Pretrained model: [Salesforce BLIP Image Captioning Base](https://huggingface.co/Salesforce/blip-image-captioning-base)

The project demonstrates the application and integration of existing pretrained models for learning purposes.
