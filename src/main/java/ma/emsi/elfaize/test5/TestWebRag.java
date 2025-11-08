package ma.emsi.elfaize.test5;

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
import dev.langchain4j.rag.DefaultRetrievalAugmentor;
import dev.langchain4j.rag.RetrievalAugmentor;
import dev.langchain4j.rag.content.retriever.ContentRetriever;
import dev.langchain4j.rag.content.retriever.EmbeddingStoreContentRetriever;
import dev.langchain4j.rag.content.retriever.WebSearchContentRetriever;
import dev.langchain4j.rag.query.router.DefaultQueryRouter;
import dev.langchain4j.rag.query.router.QueryRouter;
import dev.langchain4j.service.AiServices;
import dev.langchain4j.store.embedding.EmbeddingStore;
import dev.langchain4j.store.embedding.inmemory.InMemoryEmbeddingStore;
import dev.langchain4j.web.search.tavily.TavilyWebSearchEngine;
import ma.emsi.elfaize.test1.Assistant;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;
import java.util.logging.ConsoleHandler;
import java.util.logging.Level;
import java.util.logging.Logger;

public class TestWebRag {

    private static void configureLogger() {
        Logger packageLogger = Logger.getLogger("dev.langchain4j");
        packageLogger.setLevel(Level.FINE);
        ConsoleHandler handler = new ConsoleHandler();
        handler.setLevel(Level.FINE);
        packageLogger.addHandler(handler);
    }

    public static void main(String[] args) throws Exception {
        configureLogger();
        System.out.println("=== Test 5 - RAG avec récupération Web (Tavily) ===");

        // 1. Parser, splitter et modèle d'embedding
        DocumentParser parser = new ApacheTikaDocumentParser();
        DocumentSplitter splitter = DocumentSplitters.recursive(300, 30);
        EmbeddingModel embeddingModel = new AllMiniLmL6V2EmbeddingModel();

        // 2. Charger ton document PDF (rag.pdf)
        Path path = Paths.get("src/main/resources/rag.pdf");
        Document document = FileSystemDocumentLoader.loadDocument(path, parser);
        List<TextSegment> segments = splitter.split(document);
        System.out.println("Document chargé : " + path.getFileName() + " (" + segments.size() + " segments)");

        // 3. Générer les embeddings et les stocker
        List<Embedding> embeddings = embeddingModel.embedAll(segments).content();
        EmbeddingStore<TextSegment> store = new InMemoryEmbeddingStore<>();
        store.addAll(embeddings, segments);

        // 4. ContentRetriever local (RAG)
        ContentRetriever retrieverLocal = EmbeddingStoreContentRetriever.builder()
                .embeddingStore(store)
                .embeddingModel(embeddingModel)
                .maxResults(2)
                .minScore(0.5)
                .build();

        // 5. Moteur Tavily
        String TAVILY_KEY = System.getenv("TAVILY_API_KEY");
        if (TAVILY_KEY == null) {
            throw new IllegalStateException("Variable d'environnement TAVILY_API_KEY non définie !");
        }

        var tavilySearchEngine = TavilyWebSearchEngine.builder()
                .apiKey(TAVILY_KEY)
                .build();

        // 6. ContentRetriever Web
        ContentRetriever retrieverWeb = WebSearchContentRetriever.builder()
                .webSearchEngine(tavilySearchEngine)
                .maxResults(3)
                .build();

        // 7. QueryRouter combiné (PDF + Web)
        QueryRouter router = new DefaultQueryRouter(retrieverLocal, retrieverWeb);

        // 8. Clé Gemini
        String GEMINI_KEY = System.getenv("GEMINI_KEY");
        if (GEMINI_KEY == null) {
            throw new IllegalStateException("Variable d'environnement GEMINI_KEY non définie !");
        }

        ChatModel model = GoogleAiGeminiChatModel.builder()
                .apiKey(GEMINI_KEY)
                .modelName("gemini-2.5-flash")
                .temperature(0.3)
                .logRequestsAndResponses(true)
                .build();

        // 9. RetrievalAugmentor avec QueryRouter
        RetrievalAugmentor augmentor = DefaultRetrievalAugmentor.builder()
                .queryRouter(router)
                .build();

        // 10. Assistant
        Assistant assistant = AiServices.builder(Assistant.class)
                .chatModel(model)
                .retrievalAugmentor(augmentor)
                .chatMemory(MessageWindowChatMemory.withMaxMessages(10))
                .build();

        // 11. Interaction utilisateur
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

        System.out.println("Fin du test 5 - Web RAG terminé !");
    }
}
