from flask import Flask, render_template, request
from transformers import pipeline

# Inicializa o Flask
app = Flask(__name__)

# Inicializa o pipeline de resumo utilizando o modelo do Facebook BART.
summarizer = pipeline(
    "summarization",
    model="facebook/bart-large-cnn",
    device=0  # Usa GPU, se disponível.
)

@app.route("/")
def home():
    # Renderiza a página inicial sem texto pré-preenchido
    return render_template("index.html", texto="", resumo="", erro="")

@app.route("/resumo", methods=["POST"])
def gerar_resumo():
    # Recebe o texto do formulário
    texto = request.form.get("texto", "").strip()

    if not texto:
        erro = "Por favor, insira um texto para resumir."
        return render_template("index.html", texto=texto, resumo="", erro=erro)

    try:
        # Parâmetros para o resumo detalhado (mais contexto, mas ainda resumido)
        min_length, max_length = 150, 300
        do_sample = False  # Garante consistência e precisão.

        # Se o texto for muito grande, utiliza o processo de chunking.
        words = texto.split()
        threshold_word_count = 700  # Se o texto tiver mais que 700 palavras, divide-o em partes.
        if len(words) > threshold_word_count:
            # Divide o texto em chunks baseando-se em sentenças.
            sentences = texto.split(". ")
            chunks = []
            current_chunk = ""
            current_chunk_words = 0
            max_chunk_words = 600  # Evita ultrapassar o limite de palavras do chunk.

            for sentence in sentences:
                sentence_with_period = sentence if sentence.endswith(".") else sentence + "."
                sentence_words = sentence_with_period.split()
                if current_chunk_words + len(sentence_words) <= max_chunk_words:
                    current_chunk += sentence_with_period + " "
                    current_chunk_words += len(sentence_words)
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    current_chunk = sentence_with_period + " "
                    current_chunk_words = len(sentence_words)
            if current_chunk:
                chunks.append(current_chunk.strip())

            # Processamento de cada chunk
            summarized_chunks = []
            for chunk in chunks:
                summary_chunk = summarizer(
                    chunk,
                    max_length=max_length,
                    min_length=min_length,
                    do_sample=do_sample
                )
                if summary_chunk and len(summary_chunk) > 0 and "summary_text" in summary_chunk[0]:
                    summarized_chunks.append(summary_chunk[0]["summary_text"])
            combined_summary = " ".join(summarized_chunks)

            # Segundo resumo para agregar os detalhes (opcional, mas agrega o contexto)
            final_summary_result = summarizer(
                combined_summary,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample
            )
            if final_summary_result and len(final_summary_result) > 0 and "summary_text" in final_summary_result[0]:
                final_summary = final_summary_result[0]["summary_text"]
            else:
                final_summary = combined_summary
        else:
            # Se o texto não for tão grande, processa-o diretamente.
            summary = summarizer(
                texto,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample
            )
            if summary and len(summary) > 0 and "summary_text" in summary[0]:
                final_summary = summary[0]["summary_text"]
            else:
                final_summary = "Não foi possível gerar um resumo para o texto fornecido."

        # Pós-processamento leve para melhorar a formatação
        resumo_limpo = final_summary.replace(" . ", ". ").replace(" , ", ", ").strip()

        return render_template("index.html", texto=texto, resumo=resumo_limpo, erro="")
    except Exception as e:
        erro = f"Ocorreu um erro ao processar o texto: {str(e)}"
        return render_template("index.html", texto=texto, resumo="", erro=erro)

if __name__ == "__main__":
    # Executa o servidor Flask
    app.run(host="0.0.0.0", port=5000, debug=True)
