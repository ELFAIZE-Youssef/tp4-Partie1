package ma.emsi.elfaize.test2;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import ma.emsi.elfaize.test1.Assistant;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.List;
import java.util.Scanner;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

public class ragnaif {
    private static void configureLogger() {
        // Configure le logger sous-jacent (java.util.logging)
        Logger packageLogger = Logger.getLogger("dev.langchain4j");
        packageLogger.setLevel(Level.FINE); // Ajuster niveau
        // Ajouter un handler pour la console pour faire afficher les logs
        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);
        packageLogger.addHandler(handler);
    }

    public static void main(String[] args) throws Exception {
        configureLogger();
        System.out.println("=== Phase 1 : Initialisation et enregistrement des embeddings ===");

        //Instanciation du parser PDF basé sur Apache Tika
        DocumentParser documentParser = new ApacheTikaDocumentParser();

        //Chargement du support de cours RAG en PDF
        Path path = Paths.get("src/main/resources/rag.pdf");
        Document document = FileSystemDocumentLoader.loadDocument(path, documentParser);
        System.out.println(" Fichier chargé : " + path.getFileName());

        //Découpage du document en fragments de texte
        DocumentSplitter splitter = DocumentSplitters.recursive(300, 30);
        List<TextSegment> segments = splitter.split(document);
        System.out.println("🔹 Nombre de fragments extraits : " + segments.size());

        //Création du modèle d’embeddings (AllMiniLmL6V2)
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        //Génération des vecteurs d’embedding pour chaque fragment
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();
        System.out.println("Nombre total d’embeddings créés : " + embeddings.size());

        //Mise en place d’un magasin d’embeddings en mémoire (InMemoryEmbeddingStore)
        EmbeddingStore<TextSegment> embeddingStore = new InMemoryEmbeddingStore<>();

        //Association des embeddings et des segments correspondants
        embeddingStore.addAll(embeddings, segments)
        System.out.println("Les embeddings ont été enregistrés avec succès!");

        System.out.println("Phase 1 terminée : préparation du RAG réussie !");

        // === Phase 2 : Assistant RAG ===
        System.out.println("\n===  Phase 2 : Assistant RAG (Gemini + Embeddings) ===");


        var retriever = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(embeddingStore)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();

        String GEMINI_API_KEY = System.getenv("GEMINI_KEY");
        if (GEMINI_API_KEY == null) {
            throw new IllegalStateException("Variable d'environnement manquante !");
        }

        ChatModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(GEMINI_API_KEY)
                .temperature(0.3)
                .logRequestsAndResponses(true)
                .modelName("gemini-2.5-flash")
                .build();

        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .contentRetriever(retriever)
                .build();

        Scanner scanner = new Scanner(System.in);
        System.out.println("\nPosez vos questions sur le contenu du PDF (tapez 'exit' pour quitter)");
        System.out.print("👉 ");

        while (true) {
            String question = scanner.nextLine();
            if (question.equalsIgnoreCase("exit")) break;
            String reponse = assistant.chat(question);
            System.out.println("\n Réponse : " + reponse);
            System.out.print("\n > ");
        }

        System.out.println("\nFin du programme. Assistant RAG exécuté avec succès !");
    }
}
