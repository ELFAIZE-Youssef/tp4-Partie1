package ma.emsi.elfaize.test3;

import dev.langchain4j.data.document.Document;
import dev.langchain4j.data.document.DocumentParser;
import dev.langchain4j.data.document.DocumentSplitter;
import dev.langchain4j.data.document.loader.FileSystemDocumentLoader;
import dev.langchain4j.data.document.parser.apache.tika.ApacheTikaDocumentParser;
import dev.langchain4j.data.document.splitter.DocumentSplitters;
import dev.langchain4j.data.embedding.Embedding;
import dev.langchain4j.data.segment.TextSegment;
import dev.langchain4j.memory.chat.MessageWindowChatMemory;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.embedding.onnx.allminilml6v2.AllMiniLmL6V2EmbeddingModel;
import dev.langchain4j.model.googleai.GoogleAiGeminiChatModel;
import dev.langchain4j.rag.query.router.LanguageModelQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import ma.emsi.elfaize.test1.Assistant;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

public class TestRoutage {

    private static void configureLogger() {
        Logger packageLogger = Logger.getLogger("dev.langchain4j");
        packageLogger.setLevel(Level.FINE);
        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);
        packageLogger.addHandler(handler);
    }

    public static void main(String[] args) throws Exception {
        configureLogger();
        System.out.println("=== Test 3 - Routage ===");

        // 1. Création du modèle d'embeddings et du parser PDF
        DocumentParser parser = new ApacheTikaDocumentParser();
        DocumentSplitter splitter = DocumentSplitters.recursive(300, 30);
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        // 2. Chargement des 2 documents et création des stores
        EmbeddingStore<TextSegment> storeRag = chargerDocument("src/main/resources/rag.pdf", parser, splitter, embeddingModel);
        EmbeddingStore<TextSegment> storeEmsi = chargerDocument("src/main/resources/emsi.pdf", parser, splitter, embeddingModel);

        // 3. Création des 2 ContentRetrievers
        ContentRetriever retrieverRag = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(storeRag)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();

        ContentRetriever retrieverEmsi = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(storeEmsi)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();

        // 4. Chargement de la clé Gemini depuis les variables d'environnement
        String GEMINI_API_KEY = System.getenv("GEMINI_KEY");
        if (GEMINI_API_KEY == null) {
            throw new IllegalStateException("Variable d'environnement GEMINI_KEY non définie !");
        }

        // 5. Modèle Gemini
        ChatModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(GEMINI_API_KEY)
                .modelName("gemini-2.5-flash")
                .temperature(0.3)
                .logRequestsAndResponses(true)
                .build();

        // 6. Configuration du routage
        Map<ContentRetriever, String> sources = new HashMap<>();
        sources.put(retrieverRag, "Documents sur le RAG, le fine-tuning et l'intelligence artificielle");
        sources.put(retrieverEmsi, "Documents généraux sur le management, l'organisation et les entreprises EMSI");

        QueryRouter router = new LanguageModelQueryRouter(model, sources);

        // 7. Création du RetrievalAugmentor
        RetrievalAugmentor augmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(router)
                .build();

        // 8. Création de l'assistant avec le RetrievalAugmentor
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .retrievalAugmentor(augmentor)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .build();

        // 9. Interface utilisateur simple
        Scanner scanner = new Scanner(System.in);
        System.out.println("\nPosez vos questions (tapez 'exit' pour quitter)");
        System.out.print("-> ");

        while (true) {
            String question = scanner.nextLine();
            if (question.equalsIgnoreCase("exit")) break;
            String reponse = assistant.chat(question);
            System.out.println("\nRéponse : " + reponse);
            System.out.print("\n> ");
        }

        System.out.println("Fin du test 3 - Routage terminé !");
    }

    private static EmbeddingStore<TextSegment> chargerDocument(String chemin, DocumentParser parser, DocumentSplitter splitter, EmbeddingModel model) throws Exception {
        Path path = Paths.get(chemin);
        Document doc = FileSystemDocumentLoader.loadDocument(path, parser);
        List<TextSegment> segments = splitter.split(doc);
        List<Embedding> embeddings = model.embedAll(segments).content();
        EmbeddingStore<TextSegment> store = new InMemoryEmbeddingStore<>();
        store.addAll(embeddings, segments);
        System.out.println("Document chargé : " + path.getFileName() + " -> " + segments.size() + " segments");
        return store;
    }
}
