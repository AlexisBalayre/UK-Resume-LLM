# Resume Insight: Fine-Tuning LLM for Resume Analysis

Resume Insight is a sophisticated project designed to harness the power of large language models (LLMs) for deep analysis and understanding of resume data. It seamlessly blends technologies like OCR, natural language processing (NLP), and machine learning (ML) to extract, structure, and generate insightful question-answer pairs from resumes. This data then serves as a foundation for fine-tuning an LLM, aiming to produce a model adept at generating nuanced, context-aware responses based on resume content.

## Key Features

- **Comprehensive Data Extraction**: Utilizes OCR to transform PDF resumes into analyzable text.
- **Intelligent Data Structuring**: Applies NLP techniques to categorize and structure resume text into meaningful data points.
- **Dynamic Dataset Creation**: Generates a rich dataset of question-answer pairs tailored for LLM training, focusing on resume insights.
- **Advanced Model Fine-Tuning**: Employs cutting-edge fine-tuning strategies, including memory optimization and parameter-efficient fine-tuning (PEFT), to enhance an LLM's capability in generating contextually relevant text.

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch compatible with your CUDA version
- Hugging Face Transformers, Datasets
- pdf2image, pytesseract for OCR
- Spacy for NLP tasks
- bitsandbytes for memory optimization
- Ollama to generate the training dataset

### Installation

First, clone this repository to your local machine:

```bash
git clone https://your-repository-link.git
cd your-repository-directory
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Ensure you have Tesseract-OCR and the necessary language data installed for `pytesseract`.

### Preparing Your Dataset

1. **Extract Text from PDFs**: Convert your PDF resumes into text for analysis.

   ```bash
   python pdf_data_extraction.py --pdf_path /path/to/your/pdf
   ```

2. **Structure Resume Data**: Analyze and categorize the extracted text into structured data.

3. **Generate Q&A Pairs**: Create a dataset of question-answer pairs based on the structured data. You have to install [ollama](https://ollama.com/) to generate the training dataset.

   ```bash
   ollama run mistral
   python generate_training_dataset.py --pdf_path /path/to/your/resume.pdf --output /path/to/output.jsonl
   ```

4. **Prepare Training and Validation Sets**: Shuffle and split the generated dataset into training and validation sets.

   ```bash
   python split_training_dataset.py
   ```

### Fine-Tuning the Language Model

To start the fine-tuning process:

```bash
python train_llm.py
```

This script will train the model on your custom dataset, leveraging the previously generated question-answer pairs.

## Converting and Quantizing the Model

1. Clone the [llama.cpp](https://github.com/ggerganov/llama.cpp) repository and build the `llama` binary.

2. Convert the fine-tuned model to the `gguf` format:

   ```bash
   cd /path/to/llama.cpp
   python3 convert.py --vocab-type hfft /path/to/your/fine-tuned-model-directory
   ```

3. Quantize the model to a lower precision:

   ```bash
   ./quantize /path/to/your/fine-tuned-model-directory/ggml-model.gguf /path/to/your/fine-tuned-model-directory/ggml-model-Q4_K_M.gguf Q4_K_M
   ```

4. Finally, you can test the quantized model using [ollama](https://ollama.com/), [LM Studio](https://lmstudio.ai/) or any other compatible tool. For exemple with `ollama`, modify the path to the quantized model in the `Modelfile` file and run the following command:

   ```bash
   ollama create your-model-name -f Modelfile
   ollaam run your-model-name
   ```

## Customization

Feel free to adjust the scripts to better fit your needs. For example, you might want to:

- Modify regular expressions in `uk_resume_data_extraction.py` for improved data extraction.
- Tweak the training parameters in `train_llm.py` to optimize model performance.

## Contributing

We welcome contributions! If you have suggestions for improvements or new features, please feel free to fork the repository, make your changes, and submit a pull request.

## License

This project is open-source and available under the [MIT License](LICENSE.md).

## Acknowledgments

This project makes extensive use of the following libraries and frameworks:

- [Hugging Face Transformers](https://github.com/huggingface/transformers) for NLP model support.
- [pdf2image](https://github.com/Belval/pdf2image) and [pytesseract](https://github.com/madmaze/pytesseract) for PDF text extraction.
- [Spacy](https://spacy.io/) for advanced NLP processing.
- [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) for model memory optimization.

Your feedback and contributions are highly appreciated as we aim to continuously improve and expand the capabilities of Resume Insight.

./quantize /mnt/beegfs/home/s425500/llm-finetunning/merged_model/ggml-model-f16.gguf /mnt/beegfs/home/s425500/llm-finetunning/merged_model/ggml-model-Q4_K_M.gguf Q4_K_M