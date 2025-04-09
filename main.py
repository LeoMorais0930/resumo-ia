from transformers import pipeline

def gerar_resumo(texto):
    # Modelo pré-treinado de resumo
    summarizer = pipeline("summarization")
    
    # Gerar o resumo
    resumo = summarizer(texto, max_length=100, min_length=30, do_sample=False)
    
    return resumo[0]['summary_text']

if __name__ == "__main__":
    # Entrada do usuário
    texto = input("Digite o texto para resumir: ")
    
    # Gerar e exibir o resumo
    resumo = gerar_resumo(texto)
    print("\nResumo Gerado:\n")
    print(resumo)