from graph.dag import build_graph

if __name__ == "__main__":
    print("---" * 30)
    inp = input("Enter a review: ")
    app = build_graph()
    result = app.invoke({"text": inp, "prediction": "", "confidence": 0.0})
    print("---" * 30)
