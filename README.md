# 🔄 Self-Healing Text Classifier using LangGraph & Fine-Tuned Transformer

This project implements a **self-healing classification DAG** using [LangGraph](https://github.com/langchain-ai/langgraph) with a fine-tuned transformer model (e.g., DistilBERT). It performs **text classification** with a **fallback strategy** triggered on low-confidence predictions, enabling human-in-the-loop recovery.

<br><br>

## Project Overview

* **Goal**: Build a robust NLP system that can classify text but intelligently fall back for clarification when it’s unsure.
* **Key Features**:

  * Fine-tuned transformer (LoRA/full training)
  * LangGraph-based pipeline with dynamic routing
  * CLI interface for real-time interaction
  * Logging of predictions, fallback triggers, and user corrections

<br><br>

## Components

### Nodes in the DAG

| Node                    | Function                                             |
| ----------------------- | ---------------------------------------------------- |
| `inference_node`        | Classifies the input using the fine-tuned model      |
| `confidence_check_node` | Checks if prediction confidence is above threshold   |
| `fallback_node`         | Requests user clarification if confidence is too low |

<br><br>

## Model Training

### Dataset

Used IMDB moview review dataset from kaggle
<a href="https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews" target="_blank"> IMDB Dataset of 50K Movie Reviews <a/>
<br><br>
### Fine-Tuning 'distilbert-base-uncased' (LoRA)

```bash
lora_config=LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_lin", "v_lin"],
    task_type=TaskType.SEQ_CLS,
    lora_dropout=0.1,
    bias="none"
)
model=get_peft_model(base_model, lora_config)
model.print_trainable_parameters()
```
`Selected parameters:` trainable params: 739,586 || all params: 67,694,596 || trainable%: 1.0925

<br><br>

## DAG Setup

LangGraph pipeline is defined as follows:

```python
graph=StateGraph(State)

graph.add_node("inference", RunnableLambda(inference_node))
graph.add_node("confidence_check", RunnableLambda(confidence_check_node))
graph.add_node("fallback", RunnableLambda(fallback_node))

graph.set_entry_point("inference")
graph.add_edge("inference", "confidence_check")

graph.add_conditional_edges(
    "confidence_check",
    should_fallback,
    {
        "fallback": "fallback",
        END: END
    }
)

graph.add_edge("fallback", END)
```

<br><br>

## CLI Usage

`Live Cloud notebook execution:` <a href="https://colab.research.google.com/drive/1FD5NqfWS5KR_QUrztflAvXYV90N3PPKK?usp=sharing" target="_blank"> Colab <a/>

### Launch the app:

```bash
python app.py
```

### Example Interaction:

```
------------------------------------------------------------------------------------------
Enter a review: The movie was boring and time consuming
[InferenceNode] Predicted label: Negative | Confidence: 56%
[ConfidenceCheckNode] Confidence too low. Triggering fallback...
[FallbackNode] Could you clarify your intent? Was this a negative review?

User: yes it's negative
Final Label: Negative (Corrected via user clarification)
------------------------------------------------------------------------------------------
```
<br><br>

## Setup Instructions

1. **Clone the repo**

   ```bash
   git clone https://github.com/nishantksingh0/LangGraph-based-self-healing-classifier.git
   cd self-healing-text-classifier
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**

   ```bash
   python app.py
   ```

<br><br>

## Technologies Used

* Hugging Face Transformers
* LangGraph
* KaggleDataset
* Python 3.10+
* CLI + Structured Logging

