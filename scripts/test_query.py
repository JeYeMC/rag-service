# scripts/test_query.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app.rag.pipeline import answer_question

def main():
    print("🔥 Test rápido de consulta RAG\n")

    while True:
        q = input("Pregunta: ")

        if q.lower() in ["exit", "quit"]:
            break

        res = answer_question(q, top_k=5)

        print("\n=== RESPUESTA ===")
        print(res["answer"])
        print("\n=== FUENTES ===")
        for src in res["sources"]:
            print(f"- {src}")
        print("\n")

if __name__ == "__main__":
    main()
