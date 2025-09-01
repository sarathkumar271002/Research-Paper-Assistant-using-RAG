from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_text_splitter(chunk_size=800, chunk_overlap=120):
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "]
    )
