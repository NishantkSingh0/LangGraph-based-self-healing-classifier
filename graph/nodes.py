from typing import TypedDict

class State(TypedDict):
    text: str
    prediction: str
    confidence: float

def inference_node(state: State) -> State:
    # Assume classifier and tokenizer are loaded globally or passed in
    inputs=tokenizer(state["text"], return_tensors="pt", padding="max_length", truncation=True, max_length=256)
    outputs=model(**inputs)
    probs=torch.softmax(outputs.logits, dim=1)
    confidence, prediction=torch.max(probs, dim=1)
    label="Positive" if prediction.item() == 1 else "Negative"
    print(f"[InferenceNode] Predicted label: {label} | Confidence: {round(confidence.item()*100)}%")
    return {"text": state["text"], "prediction": label, "confidence": confidence.item()}

def confidence_check_node(state: State) -> State:
    if state["confidence"] < 0.7:
        print("[ConfidenceCheckNode] Confidence too low. Triggering fallback...")
    else:
        print("[ConfidenceCheckNode] Confidence acceptable.")
    return state

def should_fallback(state: State) -> str:
    return "fallback" if state["confidence"] < 0.7 else "end"

def fallback_node(state: State) -> State:
    print(f"[FallbackNode] Could you clarify your intent? Was this a {'negative' if state['prediction']=='negative' else 'positive'} review?")
    correction=input("User: ")
    corrected_label="Negative" if "not" in correction.lower() or "negative" in correction.lower() else "Positive"
    return {"text": state["text"], "prediction": corrected_label, "confidence": state["confidence"]}
