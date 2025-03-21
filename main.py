from src.rag_system import RAG


def main():
    rag_system = RAG()
    response = rag_system.get_query(
        "Gas fee olacak mı? Olmayacaksa işlemler nasıl gerçekleşmektedir?")
    print(response)


if __name__ == "__main__":
    main()
